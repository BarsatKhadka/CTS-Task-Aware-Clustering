"""
Advanced CTS Predictor — targets MAE < 0.10 on z-scores for unseen designs.

Architecture (informed by GAN-CTS, PreRoutGNN, PowPrediCT, iCTS literature):
  1. Rich feature extraction: timing paths + DEF spatial + SAIF activity + knobs
  2. LightGBM per-target with physics-informed features
  3. XGBoost ensemble for diversity
  4. Stacked final predictor
  5. Cross-design generalization via z-score targets + design-agnostic features

Run:
    python advanced_predictor.py [--cv] [--save]
"""

from __future__ import annotations

import argparse
import os
import pickle
import warnings
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")

from rich_features import (
    BASE,
    DATASET_DIR,
    FEATURE_DIM,
    TARGETS,
    build_dataset,
    extract_knob_features,
    extract_row_features,
)

TRAIN_CSV = DATASET_DIR / "unified_manifest_normalized.csv"
TEST_CSV  = DATASET_DIR / "unified_manifest_normalized_test.csv"
TRAIN_CACHE = BASE / "cache_train_features.pkl"
TEST_CACHE  = BASE / "cache_test_features.pkl"


# ─────────────────────────────────────────────────────────────────────────────
#  LightGBM hyper-params per target
# ─────────────────────────────────────────────────────────────────────────────

_LGB_BASE = dict(
    n_estimators   = 2000,
    learning_rate  = 0.02,
    num_leaves     = 63,
    max_depth      = 7,
    min_child_samples = 20,
    subsample      = 0.8,
    colsample_bytree = 0.8,
    reg_alpha      = 0.05,
    reg_lambda     = 0.1,
    n_jobs         = -1,
    verbose        = -1,
)

_LGB_PARAMS = {
    "z_skew_setup": {
        **_LGB_BASE,
        "num_leaves":       31,
        "max_depth":        6,
        "n_estimators":     3000,
        "learning_rate":    0.015,
        "min_child_samples": 30,
        "reg_alpha":        0.1,
        "reg_lambda":       0.2,
    },
    "z_power_total": {**_LGB_BASE},
    "z_wirelength":  {**_LGB_BASE},
}

_XGB_BASE = dict(
    n_estimators    = 1500,
    learning_rate   = 0.025,
    max_depth       = 6,
    subsample       = 0.8,
    colsample_bytree = 0.8,
    reg_alpha       = 0.1,
    reg_lambda      = 1.0,
    tree_method     = "hist",
    n_jobs          = -1,
    verbosity       = 0,
)


# ─────────────────────────────────────────────────────────────────────────────
#  Feature engineering: add interaction features
# ─────────────────────────────────────────────────────────────────────────────

TIMING_DIM = 17
DEF_DIM    = 22
SAIF_DIM   = 10
KNOB_DIM   = 11

def augment_features(X: np.ndarray) -> np.ndarray:
    """
    Add 20 physics-inspired interaction features to the base 60.

    Groups (indices in X):
      timing: 0-16
      def:    17-38
      saif:   39-48
      knobs:  49-59
    """
    # Index aliases
    # timing
    t_range  = X[:, 8]    # slack range
    t_std    = X[:, 7]    # slack std
    t_vio    = X[:, 14]   # violation fraction
    t_paths  = X[:, 0]    # log(n_paths)
    t_nff    = X[:, 3]    # log(n_unique_ff)
    # def
    d_lognff = X[:, 17]   # log(n_ff)
    d_hpwl   = X[:, 21]   # HPWL (normalized)
    d_sx     = X[:, 24]   # FF spread x
    d_sy     = X[:, 25]   # FF spread y
    d_ent    = X[:, 27]   # grid entropy
    d_logW   = X[:, 35]   # log(die_w)
    d_logH   = X[:, 36]   # log(die_h)
    d_hpwl_x_nff = X[:, 37]  # hpwl × n_ff
    # saif
    a_logtc  = X[:, 39]   # log(total TC)
    a_mean   = X[:, 40]   # mean TC
    a_gini   = X[:, 46]   # TC gini
    # knobs (z-scored)
    k_zmw    = X[:, 53]   # z_max_wire
    k_zbd    = X[:, 54]   # z_buf_dist
    k_zcs    = X[:, 55]   # z_cluster_size
    k_zcd    = X[:, 56]   # z_cluster_dia

    def col(arr):
        return arr.reshape(-1, 1)

    extra = np.hstack([
        # Skew predictors (timing × knobs)
        col(t_range  * k_zbd),     # timing window × buf_dist  (PreRoutGNN insight)
        col(t_std    * k_zmw),     # slack std × max_wire
        col(t_range  * d_hpwl),   # timing window × layout spread
        col(t_vio    * d_lognff),  # violation frac × circuit size
        col(t_nff    * k_zcs),     # unique FFs × cluster size

        # Power predictors (activity × knobs)
        col(a_logtc  * k_zbd),     # activity × buf_dist
        col(a_logtc  * d_lognff),  # activity × n_ff  (PowPrediCT key feature)
        col(a_mean   * k_zcs),     # mean activity × cluster size
        col(a_gini   * d_ent),     # activity imbalance × spatial entropy

        # WL predictors (spatial × knobs)
        col(d_hpwl   * k_zmw),     # layout spread × max_wire  (iCTS key)
        col(d_hpwl_x_nff * k_zbd), # hpwl×n_ff × buf_dist
        col(d_sx     * k_zcs),     # x-spread × cluster size
        col(d_sy     * k_zcd),     # y-spread × cluster dia
        col(d_logW   * k_zmw),     # log(die_w) × max_wire
        col(d_logH   * k_zmw),     # log(die_h) × max_wire

        # Cross-target interaction
        col(t_range  * a_logtc),   # timing × activity (skew-power correlation)
        col(d_hpwl   * a_logtc),   # spatial × activity (WL-power coupling)
        col(t_paths  * k_zcd),     # path count × cluster dia
        col(d_ent    * k_zcs),     # spatial entropy × cluster size
        col(t_range  * d_sx * d_sy),  # 3-way: timing × spatial spread
    ])

    return np.hstack([X, extra]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  AdvancedCTSPredictor
# ─────────────────────────────────────────────────────────────────────────────

class AdvancedCTSPredictor:
    """
    Stacked ensemble: LightGBM + XGBoost → Ridge meta-learner.

    Designed for zero-shot cross-design generalization via:
    - Design-agnostic features (normalized by die size, z-scored knobs)
    - Physics-grounded interaction features
    - Per-target tuned LightGBM models
    - Ensemble diversity via XGBoost
    """

    def __init__(self):
        self.lgb_models_: list[lgb.LGBMRegressor]  = []
        self.xgb_models_: list[xgb.XGBRegressor]   = []
        self.meta_: list[Ridge]                     = []
        self.scaler_  = StandardScaler()
        self._fitted  = False

    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        verbose: bool = True,
    ) -> "AdvancedCTSPredictor":
        """Train LightGBM + XGBoost + stacked meta-learner per target."""
        X_aug = augment_features(X)
        X_sc  = self.scaler_.fit_transform(X_aug)

        self.lgb_models_ = []
        self.xgb_models_ = []
        self.meta_       = []

        for j, tname in enumerate(TARGETS):
            y = Y[:, j]

            # LightGBM
            lgb_p = _LGB_PARAMS[tname]
            lgb_m = lgb.LGBMRegressor(**lgb_p)
            lgb_m.fit(X_sc, y)
            lgb_pred = lgb_m.predict(X_sc)

            # XGBoost
            xgb_m = xgb.XGBRegressor(**_XGB_BASE)
            xgb_m.fit(X_sc, y)
            xgb_pred = xgb_m.predict(X_sc)

            # Ridge meta-learner on base predictions
            meta_X = np.column_stack([lgb_pred, xgb_pred])
            meta   = Ridge(alpha=1.0)
            meta.fit(meta_X, y)

            self.lgb_models_.append(lgb_m)
            self.xgb_models_.append(xgb_m)
            self.meta_.append(meta)

            if verbose:
                train_pred = meta.predict(meta_X)
                mae = mean_absolute_error(y, train_pred)
                r2  = r2_score(y, train_pred)
                print(f"  [{tname}]  MAE={mae:.4f}  R²={r2:.4f}  "
                      f"lgb_trees={lgb_m.n_estimators_}  "
                      f"meta_w={meta.coef_}")

        self._fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return [N, 3] z-score predictions."""
        if not self._fitted:
            raise RuntimeError("Call fit() first.")
        X_aug = augment_features(X)
        X_sc  = self.scaler_.transform(X_aug)
        preds = []
        for j in range(len(TARGETS)):
            lp = self.lgb_models_[j].predict(X_sc)
            xp = self.xgb_models_[j].predict(X_sc)
            p  = self.meta_[j].predict(np.column_stack([lp, xp]))
            preds.append(p)
        return np.stack(preds, axis=1)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        label: str = "eval",
        df_meta: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        Y_pred = self.predict(X)

        rows = []
        print(f"\n{'='*55}")
        print(f"  {label}  (N={len(X)})")
        print(f"{'='*55}")
        for j, tname in enumerate(TARGETS):
            yt, yp = Y[:, j], Y_pred[:, j]
            mae  = mean_absolute_error(yt, yp)
            r2   = r2_score(yt, yp)
            rho, _ = spearmanr(yt, yp)
            ok = "✓" if mae < 0.10 else "✗"
            print(f"  {ok} {tname:20s} MAE={mae:.4f}  R²={r2:.4f}  ρ={rho:.3f}")
            rows.append({"target": tname, "mae": mae, "r2": r2, "spearman": rho})

        if df_meta is not None and "design_name" in df_meta.columns:
            print()
            for dn, grp in df_meta.groupby("design_name"):
                idx = grp.index.tolist()
                s_mae = mean_absolute_error(Y[idx,0], Y_pred[idx,0])
                p_mae = mean_absolute_error(Y[idx,1], Y_pred[idx,1])
                w_mae = mean_absolute_error(Y[idx,2], Y_pred[idx,2])
                print(f"  {dn:12s}  "
                      f"skew={s_mae:.4f}  power={p_mae:.4f}  wl={w_mae:.4f}")

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    def feature_importance(self, top_k: int = 20) -> pd.DataFrame:
        n_base = FEATURE_DIM
        n_aug  = n_base + 20
        feat_names = [f"feat_{i}" for i in range(n_aug)]
        # Named groups
        group = (["timing"] * TIMING_DIM + ["def"] * DEF_DIM +
                 ["saif"] * SAIF_DIM + ["knob"] * KNOB_DIM +
                 ["interact"] * 20)
        records = []
        for j, (tname, m) in enumerate(zip(TARGETS, self.lgb_models_)):
            imp = m.feature_importances_
            top = np.argsort(imp)[::-1][:top_k]
            for i in top:
                records.append({"target": tname, "idx": i,
                                 "group": group[i] if i < len(group) else "?",
                                 "importance": imp[i]})
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump({
                "lgb": self.lgb_models_,
                "xgb": self.xgb_models_,
                "meta": self.meta_,
                "scaler": self.scaler_,
            }, f)
        print(f"Saved → {path}")

    @classmethod
    def load(cls, path: str | Path) -> "AdvancedCTSPredictor":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls()
        obj.lgb_models_ = d["lgb"]
        obj.xgb_models_ = d["xgb"]
        obj.meta_       = d["meta"]
        obj.scaler_     = d["scaler"]
        obj._fitted     = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  Cross-validation
# ─────────────────────────────────────────────────────────────────────────────

def leave_one_design_out_cv(
    X: np.ndarray,
    Y: np.ndarray,
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """Leave-one-design-out CV using AdvancedCTSPredictor."""
    designs = df["design_name"].unique()
    results = []

    for test_design in designs:
        is_test = (df["design_name"] == test_design).values
        pred = AdvancedCTSPredictor()
        pred.fit(X[~is_test], Y[~is_test], verbose=False)
        Y_pred = pred.predict(X[is_test])

        row = {"test_design": test_design, "n_test": int(is_test.sum())}
        for j, tname in enumerate(TARGETS):
            yt, yp = Y[is_test, j], Y_pred[:, j]
            row[f"{tname}_mae"] = mean_absolute_error(yt, yp)
            row[f"{tname}_r2"]  = r2_score(yt, yp)
            row[f"{tname}_rho"] = spearmanr(yt, yp)[0]
        results.append(row)

        if verbose:
            r = results[-1]
            print(f"\n  [{test_design}] n={r['n_test']}")
            for tname in TARGETS:
                ok = "✓" if r[f"{tname}_mae"] < 0.10 else "·"
                print(f"    {ok} {tname}: MAE={r[f'{tname}_mae']:.4f}  "
                      f"R²={r[f'{tname}_r2']:.4f}  ρ={r[f'{tname}_rho']:.3f}")

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv",      action="store_true")
    parser.add_argument("--save",    default="advanced_cts_predictor.pkl")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    tr_cache = None if args.no_cache else TRAIN_CACHE
    te_cache = None if args.no_cache else TEST_CACHE

    print("=" * 55)
    print("Building training dataset …")
    X_tr, Y_tr, df_tr = build_dataset(TRAIN_CSV, BASE, verbose=True,
                                       cache_path=tr_cache)

    print("\nBuilding test dataset (zipdiv, zero-shot) …")
    X_te, Y_te, df_te = build_dataset(TEST_CSV, BASE, verbose=True,
                                       cache_path=te_cache)

    if args.cv:
        print("\n" + "=" * 55)
        print("Leave-One-Design-Out CV")
        cv = leave_one_design_out_cv(X_tr, Y_tr, df_tr, verbose=True)
        print("\n── CV Summary (mean MAE per target) ──")
        for tname in TARGETS:
            m = cv[f"{tname}_mae"].mean()
            ok = "✓" if m < 0.10 else "·"
            print(f"  {ok} {tname}: {m:.4f}")

    print("\n" + "=" * 55)
    print("Training final model …")
    pred = AdvancedCTSPredictor()
    pred.fit(X_tr, Y_tr, verbose=True)

    pred.evaluate(X_tr, Y_tr, label="TRAIN SET", df_meta=df_tr)
    pred.evaluate(X_te, Y_te, label="ZERO-SHOT  (zipdiv)", df_meta=df_te)

    pred.save(args.save)

    # Feature importance
    print("\n── Top 10 features per target ──")
    imp = pred.feature_importance(top_k=5)
    for tname, grp in imp.groupby("target"):
        print(f"\n  {tname}:")
        for _, row in grp.head(5).iterrows():
            print(f"    feat_{row['idx']:3d} ({row['group']:8s})  "
                  f"importance={row['importance']:.0f}")


if __name__ == "__main__":
    main()
