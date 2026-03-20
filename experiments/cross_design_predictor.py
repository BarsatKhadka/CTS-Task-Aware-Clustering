"""
Cross-design generalizable CTS predictor.

Key insight: Train with PER-DESIGN z-scores so the model learns
within-design relative effects of CTS knobs and placement features.
The per-design z-score is scale-invariant across designs.

Features added vs previous version:
  - core_util, density, aspect_ratio (placement geometry)
  - io_mode, time_driven, routability_driven (synthesis flags)
  - synth_strategy encoded
  - Per-design normalized SAIF features (relative activity within design)

Approach:
  - Leave-one-design-out (LODO) cross-validation for generalization check
  - LightGBM with per-target tuning
  - StandardScaler on features
"""
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

TARGETS_RAW = ["skew_setup", "power_total", "wirelength"]
TARGETS_Z   = ["z_skew_setup", "z_power_total", "z_wirelength"]

# ──────────────────────────────────────────────────────────────────────────────
# 1. Feature augmentation: add metadata + per-design normalized SAIF
# ──────────────────────────────────────────────────────────────────────────────

def _encode_synth(s: str) -> tuple[int, int]:
    """Encode 'DELAY 3' → (strategy_type, priority) as integers."""
    if isinstance(s, str):
        parts = s.strip().split()
        typ  = {"DELAY": 1, "AREA": 0}.get(parts[0], 0)
        pri  = int(parts[1]) if len(parts) > 1 else 0
    else:
        typ, pri = 0, 0
    return typ, pri


def add_meta_features(X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Append placement metadata features to X.
    Returns new X with shape (n, original_dim + n_meta).
    Meta features added (6):
        core_util / 100          (placement density proxy)
        density                  (actual post-placement density)
        aspect_ratio             (die geometry)
        io_mode                  (0/1 flag)
        time_driven              (0/1 flag)
        routability_driven       (0/1 flag)
        synth_type               (0=AREA, 1=DELAY)
        synth_priority           (integer 0-4)
    """
    meta = np.zeros((len(df), 8), dtype=np.float32)
    meta[:, 0] = df["core_util"].values / 100.0
    meta[:, 1] = df["density"].values
    meta[:, 2] = df["aspect_ratio"].values
    meta[:, 3] = df["io_mode"].values.astype(float)
    meta[:, 4] = df.get("time_driven", pd.Series(0, index=df.index)).values.astype(float)
    meta[:, 5] = df.get("routability_driven", pd.Series(0, index=df.index)).values.astype(float)
    if "synth_strategy" in df.columns:
        enc = [_encode_synth(s) for s in df["synth_strategy"]]
        meta[:, 6] = [e[0] for e in enc]
        meta[:, 7] = [e[1] for e in enc]
    return np.hstack([X, meta])


def add_relative_saif_features(X: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Normalize SAIF features (indices 39-48) by per-design statistics.
    Returns X with appended per-design-normalized SAIF (10 extra features).
    """
    saif_slice = X[:, 39:49].copy()   # 10 SAIF features
    designs = df["design_name"].values
    out = np.zeros_like(saif_slice)
    for des in np.unique(designs):
        mask = designs == des
        mu  = saif_slice[mask].mean(axis=0)
        sig = saif_slice[mask].std(axis=0)
        sig = np.where(sig < 1e-8, 1.0, sig)
        out[mask] = (saif_slice[mask] - mu) / sig
    return np.hstack([X, out])


def compute_per_design_z(df: pd.DataFrame) -> np.ndarray:
    """
    For each sample, compute z-score of (skew_setup, power_total, wirelength)
    within its design. Returns (n, 3) array matching TARGETS_Z order.
    """
    Y = np.zeros((len(df), 3), dtype=np.float32)
    for j, col in enumerate(TARGETS_RAW):
        vals = df[col].values.astype(np.float64)
        z    = np.zeros_like(vals)
        for des in df["design_name"].unique():
            mask = (df["design_name"] == des).values
            mu   = vals[mask].mean()
            sig  = vals[mask].std()
            if sig < 1e-9:
                sig = 1.0
            z[mask] = (vals[mask] - mu) / sig
        Y[:, j] = z.astype(np.float32)
    return Y


# ──────────────────────────────────────────────────────────────────────────────
# 2. Model training / evaluation
# ──────────────────────────────────────────────────────────────────────────────

LGB_PARAMS = {
    "z_skew_setup":    dict(n_estimators=800, learning_rate=0.04, num_leaves=63,
                            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=0.5, n_jobs=4, verbose=-1),
    "z_power_total":   dict(n_estimators=600, learning_rate=0.04, num_leaves=31,
                            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.5, reg_lambda=1.0, n_jobs=4, verbose=-1),
    "z_wirelength":    dict(n_estimators=600, learning_rate=0.04, num_leaves=31,
                            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=0.5, n_jobs=4, verbose=-1),
}


def train_and_eval(X_tr, Y_tr, X_te, Y_te, label="TEST", df_te=None):
    sc  = StandardScaler()
    Xsc = sc.fit_transform(X_tr)
    Xt  = sc.transform(X_te)
    models = {}
    print(f"\n{'─'*60}")
    print(f"Evaluation: {label}  (test n={len(Y_te)})")
    for j, tname in enumerate(TARGETS_Z):
        m = lgb.LGBMRegressor(**LGB_PARAMS[tname])
        m.fit(Xsc, Y_tr[:, j])
        yp  = m.predict(Xt)
        mae = mean_absolute_error(Y_te[:, j], yp)
        r2  = r2_score(Y_te[:, j], yp)
        rho,_ = spearmanr(Y_te[:, j], yp)
        ok  = "PASS✓" if mae < 0.10 else f"FAIL "
        print(f"  {ok} {tname}: MAE={mae:.4f} R²={r2:.4f} rho={rho:.3f}")
        models[tname] = (m, sc)
    return models


def lodo_cv(X, Y, df):
    """Leave-one-design-out cross-validation."""
    designs = df["design_name"].unique()
    all_results = {t: {"mae": [], "r2": [], "rho": []} for t in TARGETS_Z}
    print("\n" + "="*60)
    print("LODO Cross-Validation")
    print("="*60)
    for held in designs:
        tr_mask = (df["design_name"] != held).values
        te_mask = ~tr_mask
        X_tr, Y_tr = X[tr_mask], Y[tr_mask]
        X_te, Y_te = X[te_mask], Y[te_mask]
        sc  = StandardScaler()
        Xsc = sc.fit_transform(X_tr)
        Xt  = sc.transform(X_te)
        print(f"\nHeld-out: {held}  (n_test={te_mask.sum()})")
        for j, tname in enumerate(TARGETS_Z):
            m = lgb.LGBMRegressor(**LGB_PARAMS[tname])
            m.fit(Xsc, Y_tr[:, j])
            yp  = m.predict(Xt)
            mae = mean_absolute_error(Y_te[:, j], yp)
            r2  = r2_score(Y_te[:, j], yp)
            rho,_ = spearmanr(Y_te[:, j], yp)
            ok  = "PASS✓" if mae < 0.10 else f"FAIL "
            print(f"  {ok} {tname}: MAE={mae:.4f} R²={r2:.4f} rho={rho:.3f}")
            all_results[tname]["mae"].append(mae)
            all_results[tname]["r2"].append(r2)
            all_results[tname]["rho"].append(rho)
    print("\n" + "="*60)
    print("LODO Summary (mean ± std)")
    for tname in TARGETS_Z:
        maes = all_results[tname]["mae"]
        rhos = all_results[tname]["rho"]
        print(f"  {tname}: MAE={np.mean(maes):.4f}±{np.std(maes):.4f}  rho={np.mean(rhos):.3f}±{np.std(rhos):.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# 3. Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("Loading cached features...")
    with open("cache_train_features.pkl", "rb") as f:
        d = pickle.load(f)
    X_tr_raw, df_tr = d["X"], d["df"]

    with open("cache_test_features.pkl", "rb") as f:
        d = pickle.load(f)
    X_te_raw, df_te = d["X"], d["df"]

    print(f"Train: {X_tr_raw.shape}  Test: {X_te_raw.shape}")
    print(f"Train designs: {df_tr['design_name'].value_counts().to_dict()}")

    # ── Re-compute PER-DESIGN z-scores for training ──────────────────────────
    print("\nComputing per-design z-scores for training targets...")
    # Check if raw columns are available
    if TARGETS_RAW[0] in df_tr.columns:
        Y_tr = compute_per_design_z(df_tr)
        print("Per-design z-scores computed from raw columns.")
    else:
        print("Raw target columns not found; using cached global z-scores (suboptimal)")
        Y_tr = d["Y"]  # fallback

    # Test targets are already per-design z-scores in the test manifest
    Y_te = d["Y"]  # loaded above from cache_test_features
    # Reload Y_te properly
    with open("cache_test_features.pkl", "rb") as f:
        d2 = pickle.load(f)
    Y_te = d2["Y"]

    print(f"Y_tr shape: {Y_tr.shape}, Y_te shape: {Y_te.shape}")
    print(f"Y_tr stats: mean={Y_tr.mean(0).round(3)}, std={Y_tr.std(0).round(3)}")
    print(f"Y_te stats: mean={Y_te.mean(0).round(3)}, std={Y_te.std(0).round(3)}")

    # ── Add metadata features ─────────────────────────────────────────────────
    print("\nAdding metadata + relative SAIF features...")
    X_tr = add_meta_features(X_tr_raw, df_tr)
    X_te = add_meta_features(X_te_raw, df_te)

    # Add per-design normalized SAIF for training (can't do test cross-design here,
    # but we can z-score test by test-set stats since zipdiv is one design)
    X_tr = add_relative_saif_features(X_tr, df_tr)
    X_te = add_relative_saif_features(X_te, df_te)

    print(f"Final feature dims: train={X_tr.shape[1]}, test={X_te.shape[1]}")

    # ── LODO cross-validation ─────────────────────────────────────────────────
    print("\nRunning LODO cross-validation to validate generalization...")
    lodo_cv(X_tr, Y_tr, df_tr)

    # ── Full training → zipdiv evaluation ────────────────────────────────────
    print("\n" + "="*60)
    print("FULL MODEL: Train on all training designs → Test on zipdiv")
    print("="*60)
    train_and_eval(X_tr, Y_tr, X_te, Y_te, label="ZIPDIV (zero-shot)")

    # ── Feature importance ────────────────────────────────────────────────────
    sc  = StandardScaler()
    Xsc = sc.fit_transform(X_tr)
    print("\nTop-10 feature importances for skew:")
    m = lgb.LGBMRegressor(**LGB_PARAMS["z_skew_setup"])
    m.fit(Xsc, Y_tr[:, 0])
    imp = m.feature_importances_
    top = np.argsort(imp)[::-1][:10]
    n_orig = 60; n_meta = 8; n_rel = 10
    feat_names = ([f"base_{i}" for i in range(n_orig)] +
                  ["core_util","density","aspect_ratio","io_mode","td","rd","synth_t","synth_p"] +
                  [f"relSAIF_{i}" for i in range(n_rel)])
    for i in top:
        nm = feat_names[i] if i < len(feat_names) else f"feat_{i}"
        print(f"  [{i:3d}] {nm}: {imp[i]}")


if __name__ == "__main__":
    main()
