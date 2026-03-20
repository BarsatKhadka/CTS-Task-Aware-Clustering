"""
Absolute CTS Power + WL Predictor — No prior runs required.

Problem with z-score model: predicting z = (power - μ) / σ requires knowing μ, σ
from that placement's CTS runs. In deployment you have NO runs yet.

Solution: Predict absolute power (Watts) and WL (µm) directly using:
  - Design-scale features: n_ff, total_activity, gate_count (from DEF/SAIF)
  - CTS knobs: cluster_dia, cluster_size, buf_dist, max_wire (raw values)
  - Physics interactions: n_ff/cluster_size, HPWL/cluster_dia, etc.

Physical basis:
  P_total = P_buffer + P_wire
           ≈ k1 × (n_ff / cluster_size) × f × toggle
           + k2 × WL × C_wire × V² × f
  WL ≈ 1.2 × HPWL × f(cluster_dia)

k1, k2 are technology constants — same for all designs on same process node.
These are LEARNED from seen designs and GENERALIZE to unseen designs.

Evaluation: MAPE (%) — scale-invariant, doesn't require knowing μ, σ.
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------

with open('cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)

X_cache = cache['X']
Y_cache = cache['Y']   # Per-placement z-scores (not used for absolute model)
df = cache['df']

with open('tight_path_feats_cache.pkl', 'rb') as f:
    tight_path_cache = pickle.load(f)

pids = df['placement_id'].values
designs = df['design_name'].values
design_list = sorted(np.unique(designs))

# Raw absolute targets (log-transformed for prediction)
power_raw = df['power_total'].values.astype(np.float64)    # Watts
wl_raw    = df['wirelength'].values.astype(np.float64)     # µm

log_power = np.log(power_raw)
log_wl    = np.log(wl_raw)

print("Dataset stats:")
print(f"  power_total: mean={power_raw.mean():.5f}W, range=[{power_raw.min():.5f}, {power_raw.max():.5f}]")
print(f"  wirelength:  mean={wl_raw.mean():.1f}µm, range=[{wl_raw.min():.1f}, {wl_raw.max():.1f}]")
print()

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING FOR ABSOLUTE PREDICTION
#
# X_cache[:, 0:72] = design-level constants extracted from PT files / SAIF
#   col0 = log(n_total_nodes)    — total gate count
#   col2 = log(n_ff)             — flip-flop count
#   col4 = log(mean_toggle_ish)  — toggle activity proxy
#   col5 = log(sum_toggle_ish)   — total switching activity
#
# These were EXCLUDED from z-score model (design identity leakage).
# For ABSOLUTE prediction, they ARE the correct signal — they encode
# the physical scale of the design that determines baseline power.
# ---------------------------------------------------------------------------

# Design-scale features from cache (constant per design, available from DEF/SAIF)
log_n_total   = X_cache[:, 0]    # log(total gate count)
log_n_ff      = X_cache[:, 2]    # log(n_ff) — key scale driver
toggle_proxy  = X_cache[:, 4]    # log(toggle activity fraction)
activity_sum  = X_cache[:, 5]    # log(total switching activity)

# Some placement-level design features (vary within design due to synthesis)
# These capture placement-specific characteristics beyond design identity
design_feat_extra = X_cache[:, [6, 7, 8, 9, 18, 19, 21, 24, 25]]  # 9 placement stats

# Raw CTS knob values
cd  = df['cts_cluster_dia'].values.astype(np.float64)    # µm
cs  = df['cts_cluster_size'].values.astype(np.float64)   # count
bd  = df['cts_buf_dist'].values.astype(np.float64)       # µm
mw  = df['cts_max_wire'].values.astype(np.float64)       # µm

# Placement geometry (scale-invariant)
util        = df['core_util'].values.astype(np.float64) / 100.0
density     = df['density'].values.astype(np.float64)
aspect_ratio = df['aspect_ratio'].values.astype(np.float64)

def build_X_abs(include_tight=True):
    """
    Build absolute-scale feature matrix.

    Key physics:
      - n_ff / cluster_size  →  buffer count  →  P_buffer
      - HPWL / cluster_dia   →  cluster count →  WL
      - toggle_activity × cluster_count → total switched capacitance
    """
    n = len(df)

    # 1. Log-transformed design scale (the fundamental drivers)
    scale_feats = np.column_stack([
        log_n_total,         # log(n_gates) — total design size
        log_n_ff,            # log(n_ff) — FF count (key!)
        toggle_proxy,        # log(toggle fraction) — activity
        activity_sum,        # log(total toggle) — total switching
    ])

    # 2. Log-transformed raw knobs (preserve physical units)
    knob_log = np.column_stack([
        np.log(cd),          # log(cluster_dia)
        np.log(cs),          # log(cluster_size)
        np.log(bd),          # log(buf_dist)
        np.log(mw),          # log(max_wire)
    ])

    # 3. Physics interaction features
    n_ff_val = np.exp(log_n_ff)
    physics = np.column_stack([
        # Buffer count proxy: n_ff / cluster_size → dominant power driver
        np.log(n_ff_val / cs),          # log(n_ff/cs) = log_n_ff - log_cs

        # WL proxy: n_ff / (cluster_dia^2 / ff_spacing) — cluster count × dia
        log_n_ff - 2*np.log(cd),        # log(n_ff/cd^2) = WL scaling

        # Activity × buffer count: total dynamic power proxy
        activity_sum + np.log(n_ff_val / cs),  # log(activity × n_buf)

        # Cluster coverage: how many clusters span the design
        log_n_ff - np.log(cs) - np.log(cd),    # log(n_clusters / cd)

        # Wire budget vs cluster span
        np.log(mw / cd),                # max_wire / cluster_dia
        np.log(bd / cd),                # buf_dist / cluster_dia — buffer staging

        # Toggle × cluster_dia: activity × routing cost per cluster
        toggle_proxy + np.log(cd),      # log(toggle × cd)
        toggle_proxy - np.log(cd),      # log(toggle / cd)

        # n_ff × cluster_dia²: coverage area → routing WL
        log_n_ff + 2*np.log(cd),       # log(n_ff × cd²)
    ])

    # 4. Placement geometry
    geom = np.column_stack([
        util,
        density,
        aspect_ratio,
        util * np.log(cd),
        density / np.log(cd),
    ])

    # 5. Tight path features (pre-CTS timing, 20-dim)
    if include_tight:
        X_tight = np.zeros((n, 20), np.float32)
        for i, pid in enumerate(pids):
            if pid in tight_path_cache:
                X_tight[i] = tight_path_cache[pid]
        tight = X_tight.astype(np.float64)
    else:
        tight = np.zeros((n, 0))

    # 6. Extra placement-level features
    extra = design_feat_extra.astype(np.float64)

    X = np.hstack([scale_feats, knob_log, physics, geom, extra])
    if include_tight:
        X = np.hstack([X, tight])

    # Replace any NaN/inf
    X = np.nan_to_num(X, nan=0.0, posinf=10.0, neginf=-10.0)
    return X.astype(np.float32)


print("Building absolute features...")
X_abs = build_X_abs(include_tight=True)
X_abs_notight = build_X_abs(include_tight=False)
print(f"  X_abs shape: {X_abs.shape}")
print(f"  Feature range: [{X_abs.min():.3f}, {X_abs.max():.3f}]")
print()

# ---------------------------------------------------------------------------
# EVALUATION METRICS
# ---------------------------------------------------------------------------

def mape(y_true, y_pred):
    """Mean Absolute Percentage Error (%)."""
    return 100.0 * np.mean(np.abs((y_pred - y_true) / np.abs(y_true)))

def eval_absolute(y_true_log, y_pred_log, label=""):
    """Evaluate in log space (predict) but report % error on original scale."""
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)
    mape_val = mape(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred / y_true - 1) * 100  # % bias
    p90 = np.percentile(np.abs(y_pred - y_true) / y_true * 100, 90)
    return mape_val, mae_val, bias, p90

# ---------------------------------------------------------------------------
# LODO EVALUATION — ABSOLUTE POWER
# ---------------------------------------------------------------------------

print("=" * 65)
print("LODO EVALUATION: ABSOLUTE POWER PREDICTION")
print("=" * 65)
print("Target: predict power_total in Watts without prior CTS runs")
print()

# Model configs
LGB_CFG = dict(n_estimators=500, learning_rate=0.02, num_leaves=31,
               min_child_samples=10, n_jobs=4, verbose=-1)
XGB_CFG = dict(n_estimators=1000, learning_rate=0.01, max_depth=6,
               min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
               n_jobs=4, verbosity=0)

pw_results = {}  # label -> {folds: [...], preds: [...]}
wl_results = {}

for label, X in [('X_abs (scale+physics+tight)', X_abs),
                  ('X_abs_notight (scale+physics only)', X_abs_notight)]:
    pw_folds, wl_folds = [], []
    pw_oofs = np.zeros(len(pids))
    wl_oofs = np.zeros(len(pids))

    for held in design_list:
        tr = designs != held
        te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])

        # Power
        m_pw = lgb.LGBMRegressor(**LGB_CFG)
        m_pw.fit(Xtr, log_power[tr])
        pw_pred_log = m_pw.predict(Xte)
        pw_oofs[te] = pw_pred_log
        mape_pw, _, _, _ = eval_absolute(log_power[te], pw_pred_log)
        pw_folds.append(mape_pw)

        # WL
        m_wl = xgb.XGBRegressor(**XGB_CFG)
        m_wl.fit(Xtr, log_wl[tr])
        wl_pred_log = m_wl.predict(Xte)
        wl_oofs[te] = wl_pred_log
        mape_wl, _, _, _ = eval_absolute(log_wl[te], wl_pred_log)
        wl_folds.append(mape_wl)

        print(f"  [{label[:20]}] {held:12s}  power_MAPE={mape_pw:.2f}%  wl_MAPE={mape_wl:.2f}%")

    pw_mean = np.mean(pw_folds)
    wl_mean = np.mean(wl_folds)
    print(f"  LODO mean: power={pw_mean:.2f}%  WL={wl_mean:.2f}%")
    print()

    pw_results[label] = {'folds': pw_folds, 'mean': pw_mean, 'oofs': pw_oofs}
    wl_results[label] = {'folds': wl_folds, 'mean': wl_mean, 'oofs': wl_oofs}


# ---------------------------------------------------------------------------
# XGB for power too (compare)
# ---------------------------------------------------------------------------

print("=" * 65)
print("XGB COMPARISON FOR POWER")
print("=" * 65)

pw_xgb_folds = []
pw_xgb_oofs = np.zeros(len(pids))
for held in design_list:
    tr = designs != held; te = ~tr
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_abs[tr]); Xte = sc.transform(X_abs[te])
    m = xgb.XGBRegressor(**XGB_CFG)
    m.fit(Xtr, log_power[tr])
    pred = m.predict(Xte)
    pw_xgb_oofs[te] = pred
    mape_val, _, _, _ = eval_absolute(log_power[te], pred)
    pw_xgb_folds.append(mape_val)
    print(f"  XGB {held:12s}  power_MAPE={mape_val:.2f}%")

print(f"  XGB LODO mean power MAPE: {np.mean(pw_xgb_folds):.2f}%")
print()

# ---------------------------------------------------------------------------
# ANALYZE ERRORS BY DESIGN AND COMPARE TO BASELINE RANKING MODEL
# ---------------------------------------------------------------------------

print("=" * 65)
print("COMPARISON: RANKING MODEL (z-score) vs ABSOLUTE MODEL")
print("=" * 65)
print()

best_label = min(pw_results, key=lambda k: pw_results[k]['mean'])
pw_best_oofs = pw_results[best_label]['oofs']
wl_best_oofs = wl_results[best_label]['oofs']

print(f"Best absolute power model: {best_label}")
print()

for held in design_list:
    mask = designs == held
    true_pw = power_raw[mask]
    pred_pw = np.exp(pw_best_oofs[mask])
    true_wl = wl_raw[mask]
    pred_wl = np.exp(wl_best_oofs[mask])

    mape_pw = mape(true_pw, pred_pw)
    mape_wl = mape(true_wl, pred_wl)
    bias_pw = (np.mean(pred_pw) / np.mean(true_pw) - 1) * 100
    bias_wl = (np.mean(pred_wl) / np.mean(true_wl) - 1) * 100

    # Check if ranking is preserved (correlation with ground truth)
    from scipy.stats import spearmanr
    rho_pw = spearmanr(true_pw, pred_pw).correlation
    rho_wl = spearmanr(true_wl, pred_wl).correlation

    print(f"  {held}:")
    print(f"    Power:  MAPE={mape_pw:.2f}%  bias={bias_pw:+.2f}%  rank_rho={rho_pw:.3f}")
    print(f"    WL:     MAPE={mape_wl:.2f}%  bias={bias_wl:+.2f}%  rank_rho={rho_wl:.3f}")

print()
print("Summary:")
print(f"  Absolute power LODO MAPE: {pw_results[best_label]['mean']:.2f}% (GAN-CTS: 3%)")
print(f"  Absolute WL    LODO MAPE: {wl_results[best_label]['mean']:.2f}%")
print()

# ---------------------------------------------------------------------------
# TWO-STAGE MODEL: ABSOLUTE + RANK REFINEMENT
# ---------------------------------------------------------------------------

print("=" * 65)
print("TWO-STAGE MODEL: ABSOLUTE + WITHIN-PLACEMENT RANK REFINEMENT")
print("=" * 65)
print()
print("Stage 1: Predict log(power_abs) with absolute model")
print("Stage 2: Add log-residuals from ranking-aware features")
print()

# Load the z-score ranking model features for comparison
# Use X29 from the z-score experiment (ranking features)
def build_X29(df_in, X_c):
    pids_ = df_in['placement_id'].values
    n = len(pids_)
    knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
    place_cols = ['core_util', 'density', 'aspect_ratio']
    Xraw = df_in[knob_cols].values.astype(np.float32)
    Xplace = df_in[place_cols].values.astype(np.float32)
    Xkz = X_c[:, 72:76]
    raw_max = Xraw.max(axis=0) + 1e-6
    Xrank = np.zeros((n, 4), np.float32)
    Xcentered = np.zeros((n, 4), np.float32)
    Xknob_range = np.zeros((n, 4), np.float32)
    Xknob_mean = np.zeros((n, 4), np.float32)

    def rank_within(vals):
        n_ = len(vals)
        return np.argsort(np.argsort(vals)).astype(float) / max(n_ - 1, 1)

    for pid in np.unique(pids_):
        mask = pids_ == pid
        rows = np.where(mask)[0]
        for ki in range(4):
            z_vals = Xkz[rows, ki]
            Xrank[rows, ki] = rank_within(z_vals)
            Xcentered[rows, ki] = z_vals - z_vals.mean()
            Xknob_range[rows, ki] = Xraw[rows, ki].std()
            Xknob_mean[rows, ki] = Xraw[rows, ki].mean()
    Xplace_norm = Xplace.copy(); Xplace_norm[:, 0] /= 100.0
    Xkp = np.column_stack([
        Xraw[:, 3] * Xplace[:, 0] / 100,
        Xraw[:, 0] * Xplace[:, 1],
        Xraw[:, 3] / np.maximum(Xplace[:, 1], 0.01),
        Xraw[:, 3] * Xplace[:, 2],
        Xrank[:, 3] * (Xplace[:, 0] / 100),
        Xrank[:, 2] * (Xplace[:, 0] / 100),
    ])
    return np.hstack([Xkz, Xrank, Xcentered, Xplace_norm,
                      Xknob_range / raw_max, Xknob_mean / raw_max, Xkp])

X29 = build_X29(df, X_cache)

# Two-stage: concatenate absolute features + ranking features
X_combined = np.hstack([X_abs, X29])

combined_pw_folds = []
combined_wl_folds = []
combined_pw_mape = []
combined_wl_mape = []

for held in design_list:
    tr = designs != held; te = ~tr
    sc = StandardScaler()
    Xtr = sc.fit_transform(X_combined[tr])
    Xte = sc.transform(X_combined[te])

    # Power
    m_pw = lgb.LGBMRegressor(**LGB_CFG)
    m_pw.fit(Xtr, log_power[tr])
    pred_pw_log = m_pw.predict(Xte)
    mape_pw, _, _, _ = eval_absolute(log_power[te], pred_pw_log)
    combined_pw_mape.append(mape_pw)

    # Also check rank MAE for combined model
    from sklearn.metrics import mean_absolute_error as mae_fn
    def rank_within(vals):
        n_ = len(vals)
        return np.argsort(np.argsort(vals)).astype(float) / max(n_ - 1, 1)

    def zscore_pred_to_rank_mae(pred, true, pids_te):
        pr = np.zeros(len(pids_te)); tr_ = np.zeros(len(pids_te))
        for pid in np.unique(pids_te):
            m_ = pids_te == pid; rows = np.where(m_)[0]
            pr[rows] = rank_within(pred[rows]); tr_[rows] = rank_within(true[rows])
        return mae_fn(tr_, pr)

    rank_mae_pw = zscore_pred_to_rank_mae(pred_pw_log, log_power[te], pids[te])
    combined_pw_folds.append(rank_mae_pw)

    # WL
    m_wl = xgb.XGBRegressor(**XGB_CFG)
    m_wl.fit(Xtr, log_wl[tr])
    pred_wl_log = m_wl.predict(Xte)
    mape_wl, _, _, _ = eval_absolute(log_wl[te], pred_wl_log)
    combined_wl_mape.append(mape_wl)
    rank_mae_wl = zscore_pred_to_rank_mae(pred_wl_log, log_wl[te], pids[te])
    combined_wl_folds.append(rank_mae_wl)

    print(f"  {held:12s}  pw_MAPE={mape_pw:.2f}%  wl_MAPE={mape_wl:.2f}%  pw_rankMAE={rank_mae_pw:.4f}  wl_rankMAE={rank_mae_wl:.4f}")

print()
print(f"  Combined LODO: pw_MAPE={np.mean(combined_pw_mape):.2f}%  wl_MAPE={np.mean(combined_wl_mape):.2f}%")
print(f"  Combined rank: pw={np.mean(combined_pw_folds):.4f}  wl={np.mean(combined_wl_folds):.4f}")
print()

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------

print("=" * 65)
print("FINAL SUMMARY — ABSOLUTE PREDICTION CAPABILITY")
print("=" * 65)
print()
print("Model         |   Power MAPE   |   WL MAPE   |  Requires prior runs?")
print("-" * 70)
print(f"Ranking model |   N/A (ranks)  |   N/A       |  YES (μ,σ from 10 runs)")
print(f"Abs (notight) |  {pw_results['X_abs_notight (scale+physics only)']['mean']:>7.2f}%     |  {wl_results['X_abs_notight (scale+physics only)']['mean']:>7.2f}%    |  NO")
print(f"Abs+tight     |  {pw_results['X_abs (scale+physics+tight)']['mean']:>7.2f}%     |  {wl_results['X_abs (scale+physics+tight)']['mean']:>7.2f}%    |  NO (uses pre-CTS timing)")
print(f"Abs+rank feats|  {np.mean(combined_pw_mape):>7.2f}%     |  {np.mean(combined_wl_mape):>7.2f}%    |  NO")
print(f"GAN-CTS ref   |     ~3%        |    ~3%      |  NO (ResNet50 on images)")
print()
print("Deployment: given DEF + SAIF + timing_paths.csv + CTS knobs")
print("→ predict absolute power in Watts and WL in µm, no prior runs needed")
