
"""
CTS Predictor - Predicts skew, power, and wirelength for unseen VLSI designs.

Architecture:
  - Feature extraction from clustered graph files (topology + spatial + activity)
  - CSV placement/knob features
  - LightGBM ensemble with per-target tuning
  - Zero-shot generalization to unseen designs via scale-invariant graph features
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from scipy.stats import entropy as scipy_entropy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

BENCH_ROOT = Path(__file__).parent / "CTS-Bench"
GRAPH_ROOT  = BENCH_ROOT / "dataset_root" / "graphs" / "clustered_graphs"

# Clustered graph node feature indices
# [centroid_x, centroid_y, spread_x, spread_y, log_area, is_ff,
#  n_cells, drive_strength, cap_stat, activity_stat]
_FX, _FY   = 0, 1   # cluster centroid (normalized 0-1)
_SX, _SY   = 2, 3   # spatial spread
_LAREA     = 4      # log(cluster area)
_FF        = 5      # FF indicator / count
_NCELLS    = 6      # cells per cluster
_DRIVE     = 7      # drive strength
_CAP       = 8      # capacitance stat
_ACT       = 9      # activity stat

TARGETS = ["skew_setup", "power_total", "wirelength"]

# Physics-normalization: predict per-FF rates so model can extrapolate
# to designs with different numbers of FFs
# log(power)  ← log(power/n_ff)  + log(n_ff)   [n_ff_log is feature index 21]
# log(wl)     ← log(wl/n_ff)    + log(n_ff)
N_FF_FEAT_IDX = 9 + 12  # 12 csv feats + 9th graph feat = n_ff_log (feat group 2, pos 0)

# ─────────────────────────────────────────────────────────────────────────────
#  Graph feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _gini(arr: np.ndarray) -> float:
    """Gini coefficient of an array (0=equal, 1=unequal)."""
    arr = arr.flatten()
    arr = arr[arr > 0]
    if len(arr) < 2:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def extract_graph_features(graph_path: str | Path) -> np.ndarray:
    """
    Extract a 52-dimensional feature vector from a clustered graph .pt file.
    All features are scale-aware (encode circuit size) or normalized (transfer
    across designs).

    Feature groups:
      [0-2]   Scale: log(n_nodes), log(n_edges), log(n_edges/n_nodes)
      [3-8]   Spatial: mean_x, std_x, mean_y, std_y, hpwl_proxy, spread_ratio
      [9-15]  FF / Activity: n_ff_log, ff_frac, act_mean, act_std, act_sum_log,
              act_max, act_gini
      [16-22] Cluster size: size_mean_log, size_std, size_max_log, size_gini,
              cap_mean, cap_std, drive_mean
      [23-30] Spatial distribution: 8x8 grid entropy (x), 8x8 grid entropy (y),
              x_gini, y_gini, centroid_x, centroid_y, spread_x_mean, spread_y_mean
      [31-36] Degree: mean_deg, std_deg, max_deg_log, deg_gini, edge_attr_mean,
              edge_attr_std
      [37-43] Pairwise distance proxy: hpwl_ff, ff_centroid_x, ff_centroid_y,
              ff_spread_x, ff_spread_y, ff_nn_dist_mean, ff_nn_dist_std
      [44-51] Derived ratios: ff_frac*act_mean, size*ff (cap proxy),
              spread_ratio*hpwl, act_sum/n_ff, cap*drive_mean,
              edge_density, clustering_proxy, size_entropy
    """
    path = Path(graph_path)
    if not path.is_absolute():
        path = GRAPH_ROOT / path.name

    data = torch.load(str(path), weights_only=False)
    x = data.x.numpy().astype(np.float32)        # [N, 10]
    ei = data.edge_index.numpy()                  # [2, E]
    ea = data.edge_attr.numpy().flatten()         # [E]

    n, feat_dim = x.shape
    e = ea.shape[0]
    assert feat_dim >= 10, f"Expected ≥10 features, got {feat_dim}"

    cx, cy = x[:, _FX], x[:, _FY]
    sx, sy = x[:, _SX], x[:, _SY]
    larea  = x[:, _LAREA]
    ff     = x[:, _FF]
    ncells = x[:, _NCELLS]
    drive  = x[:, _DRIVE]
    cap    = x[:, _CAP]
    act    = x[:, _ACT]

    # ── Scale features ─────────────────────────────────────────────────────
    f_scale = np.array([
        np.log1p(n),
        np.log1p(e),
        np.log1p(e / max(n, 1)),
    ])

    # ── Spatial features ───────────────────────────────────────────────────
    hpwl_proxy    = (cx.max() - cx.min()) + (cy.max() - cy.min())
    spread_ratio  = ((cx.max() - cx.min()) /
                     max(cy.max() - cy.min(), 1e-6))
    f_spatial = np.array([
        cx.mean(), cx.std(),
        cy.mean(), cy.std(),
        hpwl_proxy, np.clip(np.log(spread_ratio + 1e-6), -3, 3),
    ])

    # ── FF / Activity ──────────────────────────────────────────────────────
    total_ff  = ff.sum()
    ff_frac   = total_ff / max(ncells.sum(), 1)
    act_pos   = act[act > 0]
    f_ff = np.array([
        np.log1p(total_ff),
        ff_frac,
        act.mean(),
        act.std(),
        np.log1p(act.sum()),
        act.max(),
        _gini(act),
    ])

    # ── Cluster size ───────────────────────────────────────────────────────
    size_vals = np.exp(larea)
    f_clust = np.array([
        np.log1p(larea.mean()),
        larea.std(),
        larea.max(),
        _gini(size_vals),
        cap.mean(),
        cap.std(),
        drive.mean(),
    ])

    # ── Spatial distribution (grid entropy) ───────────────────────────────
    bins = np.linspace(0, 1, 9)
    x_hist = np.histogram(cx, bins=bins)[0].astype(float) + 1e-9
    y_hist = np.histogram(cy, bins=bins)[0].astype(float) + 1e-9
    x_ent  = float(scipy_entropy(x_hist / x_hist.sum()))
    y_ent  = float(scipy_entropy(y_hist / y_hist.sum()))
    f_distrib = np.array([
        x_ent, y_ent,
        _gini(cx), _gini(cy),
        cx.mean(), cy.mean(),
        sx.mean(), sy.mean(),
    ])

    # ── Degree statistics ──────────────────────────────────────────────────
    deg = np.bincount(ei[0], minlength=n).astype(float)
    f_degree = np.array([
        deg.mean(),
        deg.std(),
        np.log1p(deg.max()),
        _gini(deg),
        ea.mean(),
        ea.std(),
    ])

    # ── FF spatial proxy ───────────────────────────────────────────────────
    ff_mask = ff > 0.5
    if ff_mask.sum() >= 2:
        fcx, fcy = cx[ff_mask], cy[ff_mask]
        # Approximate nearest-neighbor distance via sorted pairwise
        fcx_s = np.sort(fcx)
        nn_x  = np.diff(fcx_s).mean() if len(fcx_s) > 1 else 0.0
        fcy_s = np.sort(fcy)
        nn_y  = np.diff(fcy_s).mean() if len(fcy_s) > 1 else 0.0
        ff_hpwl = (fcx.max() - fcx.min()) + (fcy.max() - fcy.min())
    else:
        fcx, fcy = cx, cy
        nn_x, nn_y, ff_hpwl = 0.0, 0.0, 0.0

    f_ff_spatial = np.array([
        ff_hpwl,
        fcx.mean(), fcy.mean(),
        fcx.std(),  fcy.std(),
        nn_x,       nn_y,
    ])

    # ── Derived interaction features ───────────────────────────────────────
    n_ff_eff = max(total_ff, 1)
    edge_dens = e / max(n * (n - 1), 1)
    size_ent  = float(scipy_entropy(x_hist / x_hist.sum() * y_hist.sum()))  # reuse
    f_derived = np.array([
        ff_frac * act.mean(),                     # activity-weighted FF density
        np.log1p(cap.mean() * ncells.sum()),       # total cap proxy
        spread_ratio * hpwl_proxy,                # layout aspect effect
        act.sum() / n_ff_eff,                     # mean activity per FF
        cap.mean() * drive.mean(),                # cap-drive product
        edge_dens,                                # graph edge density
        deg.mean() / max(np.log1p(n), 1),         # normalized connectivity
        float(scipy_entropy(x_hist / x_hist.sum())),  # x entropy (same as x_ent)
    ])

    feats = np.concatenate([
        f_scale, f_spatial, f_ff, f_clust, f_distrib,
        f_degree, f_ff_spatial, f_derived,
    ])
    return feats.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  CSV feature extraction
# ─────────────────────────────────────────────────────────────────────────────

_SYNTH_STRATEGIES = {
    "AREA 0": 0, "AREA 1": 1, "AREA 2": 2, "AREA 3": 3,
    "DELAY 0": 4, "DELAY 1": 5, "DELAY 2": 6, "DELAY 3": 7,
}


def extract_csv_features(row: pd.Series) -> np.ndarray:
    """
    Extract 11 features from a manifest row (placement params + CTS knobs).
    """
    synth = _SYNTH_STRATEGIES.get(str(row.get("synth_strategy", "AREA 0")), 0)
    area_mode  = int("AREA"  in str(row.get("synth_strategy", "")))
    delay_mode = int("DELAY" in str(row.get("synth_strategy", "")))

    return np.array([
        float(row.get("aspect_ratio",    1.0)),
        float(row.get("core_util",       50.0)),
        float(row.get("density",         0.7)),
        float(area_mode),
        float(delay_mode),
        float(row.get("io_mode",         1)),
        float(row.get("time_driven",     1)),
        float(row.get("routability_driven", 0)),
        float(row.get("cts_max_wire",    200)),
        float(row.get("cts_buf_dist",    100)),
        float(row.get("cts_cluster_size", 20)),
        float(row.get("cts_cluster_dia",  50)),
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    manifest_path: str | Path,
    bench_root: str | Path = BENCH_ROOT,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build feature matrix X and target matrix Y from a manifest CSV.

    Returns:
        X  – [N, 64] float32 feature matrix
        Y  – [N, 3] float32 targets: [skew_setup, log(power), log(wl)]
        df – original dataframe (for metadata)
    """
    df = pd.read_csv(manifest_path)
    bench_root = Path(bench_root)

    X_list, Y_list, valid_idx = [], [], []

    for i, row in df.iterrows():
        graph_path = bench_root / str(row["cluster_graph_path"])
        if not graph_path.exists():
            if verbose:
                print(f"  [skip] missing graph for row {i}: {graph_path.name}")
            continue

        try:
            g_feats  = extract_graph_features(graph_path)
            csv_feats = extract_csv_features(row)
            x = np.concatenate([csv_feats, g_feats])
            X_list.append(x)
        except Exception as exc:
            if verbose:
                print(f"  [skip] row {i} error: {exc}")
            continue

        # Physics-normalized targets: predict per-FF rates so cross-design
        # extrapolation works.  n_ff_log = log1p(total_ff) from graph features.
        n_ff_log    = float(g_feats[9])    # log1p(total_ff)
        act_sum_log = float(g_feats[13])   # log1p(total_activity)
        y = np.array([
            float(row["skew_setup"]),
            np.log(float(row["power_total"]) + 1e-9) - act_sum_log,  # log(power/act_sum)
            np.log(float(row["wirelength"]) + 1)     - n_ff_log,     # log(wl/n_ff)
        ], dtype=np.float32)
        Y_list.append(y)
        valid_idx.append(i)

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    df = df.iloc[valid_idx].reset_index(drop=True)

    if verbose:
        print(f"Dataset: {len(X)} samples, {X.shape[1]} features, 3 targets")
        for j, name in enumerate(TARGETS):
            print(f"  {name}: mean={Y[:,j].mean():.4f}  std={Y[:,j].std():.4f}")

    return X, Y, df


# ─────────────────────────────────────────────────────────────────────────────
#  Model
# ─────────────────────────────────────────────────────────────────────────────

_LGB_DEFAULTS = dict(
    n_estimators=1200,
    learning_rate=0.03,
    num_leaves=63,
    max_depth=8,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    n_jobs=-1,
    verbose=-1,
)

# Task-specific overrides (skew is harder – needs more regularization)
_TASK_PARAMS: dict[str, dict] = {
    "skew_setup": dict(num_leaves=31, max_depth=6, n_estimators=1500,
                       learning_rate=0.02, min_child_samples=30),
    "power_total": dict(num_leaves=63, max_depth=8),
    "wirelength":  dict(num_leaves=63, max_depth=8),
}


class CTSPredictor:
    """
    Gradient-boosted ensemble that predicts CTS outcomes for unseen designs.

    Usage:
        pred = CTSPredictor()
        pred.fit(X_train, Y_train)
        Y_pred = pred.predict(X_test)         # shape [N, 3]  (skew, log_power, log_wl)
        Y_natural = pred.inverse_transform(Y_pred)  # back to original units
    """

    def __init__(self, task_params: Optional[dict] = None):
        self.task_params = task_params or _TASK_PARAMS
        self.models_: list[lgb.LGBMRegressor] = []
        self.scaler_      = StandardScaler()
        self.skew_debias_ = LinearRegression()  # skew ~ n_ff_log + act_sum_log
        self._fitted      = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        eval_set: Optional[tuple[np.ndarray, np.ndarray]] = None,
        verbose: bool = True,
    ) -> "CTSPredictor":
        """
        Train one LightGBM model per target.

        Args:
            X       – [N, D] features
            Y       – [N, 3] targets (skew, log_power, log_wl)
            eval_set – optional (X_val, Y_val) for early stopping
        """
        # ── Physics debiasing for skew ──────────────────────────────────────
        # Skew scales with log(n_ff) and log(act_sum) at the design level.
        # Fit a linear prior so LightGBM only learns the CTS knob residual.
        # act_sum_log is N_FF_FEAT_IDX + 4 = feat 25
        _ACT_SUM_IDX = N_FF_FEAT_IDX + 4
        _HPWL_IDX    = N_FF_FEAT_IDX - 2   # spatial[4] = hpwl_proxy = idx 19
        skew_phy_feats = np.column_stack([
            X[:, N_FF_FEAT_IDX],
            X[:, _ACT_SUM_IDX],
            X[:, _HPWL_IDX],
        ])
        self.skew_debias_.fit(skew_phy_feats, Y[:, 0])
        skew_baseline = self.skew_debias_.predict(skew_phy_feats)
        Y_skew_residual = Y[:, 0] - skew_baseline  # zero-mean residual

        # Build Y_to_fit with residual skew
        Y_fit = Y.copy()
        Y_fit[:, 0] = Y_skew_residual

        X_sc = self.scaler_.fit_transform(X)
        self.models_ = []

        for j, tname in enumerate(TARGETS):
            params = {**_LGB_DEFAULTS, **self.task_params.get(tname, {})}
            model  = lgb.LGBMRegressor(**params)

            fit_kwargs: dict = {}
            if eval_set is not None:
                X_val_sc = self.scaler_.transform(eval_set[0])
                Y_val_fit = eval_set[1].copy()
                if j == 0:
                    vphy = np.column_stack([
                        eval_set[0][:, N_FF_FEAT_IDX],
                        eval_set[0][:, N_FF_FEAT_IDX + 4],
                        eval_set[0][:, N_FF_FEAT_IDX - 2],
                    ])
                    Y_val_fit[:, 0] = eval_set[1][:, 0] - \
                        self.skew_debias_.predict(vphy)
                fit_kwargs = dict(
                    eval_set=[(X_val_sc, Y_val_fit[:, j])],
                    callbacks=[lgb.early_stopping(50, verbose=False),
                               lgb.log_evaluation(-1)],
                )

            model.fit(X_sc, Y_fit[:, j], **fit_kwargs)
            self.models_.append(model)

            if verbose:
                train_preds = model.predict(X_sc)
                mae = mean_absolute_error(Y[:, j], train_preds)
                r2  = r2_score(Y[:, j], train_preds)
                print(f"  [{tname}] train MAE={mae:.4f}  R²={r2:.4f}  "
                      f"trees={model.n_estimators_}")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return physics-normalized predictions:
        [skew (ns), log(power/n_ff), log(wl/n_ff)].

        Skew = skew_physics_baseline + lgb_residual.
        """
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X_sc = self.scaler_.transform(X)
        preds = np.stack([m.predict(X_sc) for m in self.models_], axis=1)

        # Add back skew physics baseline
        phy_feats = np.column_stack([
            X[:, N_FF_FEAT_IDX],
            X[:, N_FF_FEAT_IDX + 4],
            X[:, N_FF_FEAT_IDX - 2],
        ])
        preds[:, 0] = preds[:, 0] + self.skew_debias_.predict(phy_feats)
        return preds

    # ------------------------------------------------------------------
    def predict_natural(self, X: np.ndarray) -> pd.DataFrame:
        """
        Return predictions in natural units as a DataFrame with columns:
        skew_pred, power_pred, wirelength_pred.

        Adds back n_ff_log offset (physics normalization inverse).
        """
        Y_norm      = self.predict(X)
        n_ff_log    = X[:, N_FF_FEAT_IDX]         # log1p(total_ff)
        act_sum_log = X[:, N_FF_FEAT_IDX + 4]     # log1p(total_activity)
        return pd.DataFrame({
            "skew_pred":       Y_norm[:, 0],
            "power_pred":      np.exp(Y_norm[:, 1] + act_sum_log),
            "wirelength_pred": np.exp(Y_norm[:, 2] + n_ff_log) - 1,
        })

    # ------------------------------------------------------------------
    def evaluate(
        self,
        X: np.ndarray,
        Y_norm: np.ndarray,
        label: str = "eval",
        df_meta: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Evaluate on a dataset.  Prints MAE / R² per target.
        Returns per-sample result DataFrame.

        Y_norm should be [N, 3]: (skew, log(power/n_ff), log(wl/n_ff)).
        Natural-unit MAE is computed by adding back n_ff_log from X.
        """
        Y_pred_norm = self.predict(X)
        n_ff_log    = X[:, N_FF_FEAT_IDX]
        act_sum_log = X[:, N_FF_FEAT_IDX + 4]

        rows = []
        print(f"\n── {label} ({'N=' + str(len(X))}) ──")
        for j, tname in enumerate(TARGETS):
            yt, yp = Y_norm[:, j], Y_pred_norm[:, j]
            mae = mean_absolute_error(yt, yp)
            r2  = r2_score(yt, yp)
            if tname == "skew_setup":
                print(f"  skew_setup   MAE={mae:.4f} ns   R²={r2:.4f}")
            elif tname == "power_total":
                nat_true = np.exp(yt + act_sum_log)
                nat_pred = np.exp(yp + act_sum_log)
                nat_mae  = mean_absolute_error(nat_true, nat_pred)
                print(f"  power_total  MAE={nat_mae:.6f} W    R²={r2:.4f}  "
                      f"(norm MAE={mae:.4f})")
            else:
                nat_true = np.exp(yt + n_ff_log) - 1
                nat_pred = np.exp(yp + n_ff_log) - 1
                nat_mae  = mean_absolute_error(nat_true, nat_pred)
                print(f"  wirelength   MAE={nat_mae:.0f} µm   R²={r2:.4f}  "
                      f"(norm MAE={mae:.4f})")
            rows.append({"target": tname, "mae_log": mae, "r2": r2})

        results = pd.DataFrame(rows)

        # Per-design breakdown if metadata available
        if df_meta is not None and "design_name" in df_meta.columns:
            print("\n  Per-design MAE (natural units):")
            for dn, grp in df_meta.groupby("design_name"):
                idx   = grp.index.tolist()
                nff   = n_ff_log[idx]
                asl   = act_sum_log[idx]
                for j, tname in enumerate(TARGETS):
                    yt, yp = Y_norm[idx, j], Y_pred_norm[idx, j]
                    if tname == "skew_setup":
                        nat_mae = mean_absolute_error(yt, yp)
                        unit = "ns"
                    elif tname == "power_total":
                        nat_mae = mean_absolute_error(np.exp(yt + asl),
                                                       np.exp(yp + asl))
                        unit = "W"
                    else:
                        nat_mae = mean_absolute_error(np.exp(yt + nff) - 1,
                                                       np.exp(yp + nff) - 1)
                        unit = "µm"
                    print(f"    {dn:20s} | {tname:14s} | MAE={nat_mae:.5g} {unit}")

        return results

    # ------------------------------------------------------------------
    def feature_importance(self, top_k: int = 20) -> pd.DataFrame:
        """Return top-k features by importance across all targets."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        dfs = []
        for j, m in enumerate(self.models_):
            imp = m.feature_importances_
            dfs.append(pd.DataFrame({
                "feature": [f"feat_{i}" for i in range(len(imp))],
                "importance": imp,
                "target": TARGETS[j],
            }))
        df = pd.concat(dfs)
        agg = (df.groupby("feature")["importance"]
               .sum().sort_values(ascending=False)
               .head(top_k).reset_index())
        return agg

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        """Save predictor to disk."""
        with open(path, "wb") as f:
            pickle.dump({"models":      self.models_,
                         "scaler":      self.scaler_,
                         "skew_debias": self.skew_debias_,
                         "task_params": self.task_params}, f)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "CTSPredictor":
        """Load a saved predictor."""
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(task_params=d["task_params"])
        obj.models_       = d["models"]
        obj.scaler_       = d["scaler"]
        obj.skew_debias_  = d.get("skew_debias", LinearRegression())
        obj._fitted       = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-validation helper
# ─────────────────────────────────────────────────────────────────────────────

def leave_one_design_out_cv(
    X: np.ndarray,
    Y: np.ndarray,
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Leave-one-design-out cross-validation.
    For each design, train on the remaining designs and test on it.
    Returns a DataFrame with per-fold MAE for each target.
    """
    designs = df["design_name"].unique()
    results = []

    for test_design in designs:
        is_test  = df["design_name"] == test_design
        X_tr, Y_tr = X[~is_test], Y[~is_test]
        X_te, Y_te = X[is_test],  Y[is_test]

        pred = CTSPredictor()
        pred.fit(X_tr, Y_tr, verbose=False)
        Y_pred = pred.predict(X_te)

        n_ff_log    = X_te[:, N_FF_FEAT_IDX]
        act_sum_log = X_te[:, N_FF_FEAT_IDX + 4]
        row = {"test_design": test_design, "n_test": int(is_test.sum())}
        for j, tname in enumerate(TARGETS):
            yt, yp = Y_te[:, j], Y_pred[:, j]
            row[f"{tname}_mae_log"] = mean_absolute_error(yt, yp)
            row[f"{tname}_r2"]      = r2_score(yt, yp)
            if tname == "skew_setup":
                row[f"{tname}_mae_nat"] = mean_absolute_error(yt, yp)
            elif tname == "power_total":
                row[f"{tname}_mae_nat"] = mean_absolute_error(
                    np.exp(yt + act_sum_log), np.exp(yp + act_sum_log))
            else:
                row[f"{tname}_mae_nat"] = mean_absolute_error(
                    np.exp(yt + n_ff_log)-1, np.exp(yp + n_ff_log)-1)
        results.append(row)

        if verbose:
            print(f"\n  Fold [{test_design}] (n={int(is_test.sum())}):")
            print(f"    skew  MAE={row['skew_setup_mae_nat']:.4f} ns")
            print(f"    power MAE={row['power_total_mae_nat']:.6f} W")
            print(f"    WL    MAE={row['wirelength_mae_nat']:.0f} µm")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
#  Main training script
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Train CTS predictor")
    parser.add_argument("--bench",   default=str(BENCH_ROOT))
    parser.add_argument("--cv",      action="store_true", help="Run LODO CV")
    parser.add_argument("--save",    default="cts_lgb_predictor.pkl")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    bench = Path(args.bench)
    train_csv = bench / "dataset_root" / "clocknet_unified_manifest.csv"
    test_csv  = bench / "dataset_root" / "clocknet_unified_manifest_test.csv"

    print("=" * 60)
    print("Building training dataset …")
    X_tr, Y_tr, df_tr = build_dataset(train_csv, bench, verbose=True)

    print("\nBuilding test dataset (zero-shot: zipdiv) …")
    X_te, Y_te, df_te = build_dataset(test_csv, bench, verbose=True)

    if args.cv:
        print("\n" + "=" * 60)
        print("Leave-One-Design-Out Cross-Validation")
        cv_results = leave_one_design_out_cv(X_tr, Y_tr, df_tr, verbose=True)
        print("\n── CV Summary ──")
        print(cv_results[["test_design",
                           "skew_setup_mae_nat",
                           "power_total_mae_nat",
                           "wirelength_mae_nat"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("Training on all training designs …")
    predictor = CTSPredictor()
    predictor.fit(X_tr, Y_tr, verbose=True)

    print("\n" + "=" * 60)
    predictor.evaluate(X_tr, Y_tr, label="TRAIN", df_meta=df_tr)

    print()
    predictor.evaluate(X_te, Y_te, label="ZERO-SHOT (zipdiv)", df_meta=df_te)

    predictor.save(args.save)

    print("\n── Top feature importances ──")
    print(predictor.feature_importance(top_k=15).to_string(index=False))


if __name__ == "__main__":
    main()
