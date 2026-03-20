"""
Generalizable CTS Predictor v2
================================
Goal: MAE < 0.10 on all 3 tasks (skew, power, wirelength) on UNSEEN designs.
Evaluation: Leave-One-Design-Out (LODO) only.

Key fixes over previous versions:
  1. Per-placement z-score normalization (NOT global z-scores)
  2. Rich features: DEF spatial + SAIF activity + timing paths + CTS knobs
  3. Physics-informed interaction features
  4. Multiple model ensemble: LightGBM + XGBoost + Ridge
  5. Correct LODO split (train on 3 designs, test on 1)

Literature basis:
  - GAN-CTS (TCAD 2022): placement image features + separate skew head
  - PreRoutGNN (AAAI 2024): pre-route timing slack → skew predictor
  - PowPrediCT (DAC 2024): total activity × capacitance → power
  - iCTS (TCAD 2025): cluster_dia × HPWL → WL predictor
  - DME / Cong 1993: WL ≈ 1.1-1.5 × HPWL (Steiner proxy)
  - Elmore delay: skew = max(delay) - min(delay) → tail statistics matter
"""

from __future__ import annotations

import os
import re
import pickle
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy, spearmanr
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset_with_def"
PLACEMENT_DIR = DATASET_DIR / "placement_files"
MANIFEST_CSV = DATASET_DIR / "unified_manifest_normalized.csv"
TEST_CSV = DATASET_DIR / "unified_manifest_normalized_test.csv"

# ─────────────────────────────────────────────────────────────────────────────
#  Per-placement target normalization  (CRITICAL: NOT global z-scores)
# ─────────────────────────────────────────────────────────────────────────────

def compute_per_placement_targets(df: pd.DataFrame) -> np.ndarray:
    """
    For each placement (10 CTS runs), z-score the raw targets within
    the placement group. This makes the task: "given CTS knobs, how does
    this run compare to the placement mean?" — learnable without knowing
    what design this is.

    Sigma floor prevents division by near-zero for designs with low variation.
    """
    Y = np.zeros((len(df), 3), dtype=np.float32)
    df = df.reset_index(drop=True)

    for pid, grp in df.groupby("placement_id"):
        idx = grp.index.tolist()
        for j, col in enumerate(["skew_setup", "power_total", "wirelength"]):
            vals = grp[col].values.astype(np.float64)
            mu = vals.mean()
            sig = vals.std()
            # Floor: at least 1% of mean or 1e-4 — prevents explosion
            sig = max(sig, max(abs(mu) * 0.01, 1e-4))
            Y[idx, j] = (vals - mu) / sig

    return Y


# ─────────────────────────────────────────────────────────────────────────────
#  Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

_FF_PATTERN = re.compile(
    r"sky130_fd_sc_hd__(?:df|sdff|edf|sdfsrn|sdfbbn|dfbbn|dfrtp|dfxbp|dfxtp|dfbbp|df[a-z])",
    re.IGNORECASE,
)

_placement_cache: dict = {}


def _gini(arr: np.ndarray) -> float:
    arr = np.sort(arr.flatten())
    n = len(arr)
    if n < 2 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def parse_def(path: str) -> dict:
    """Parse DEF file → FF positions + die dimensions."""
    result = {"die_w": 1, "die_h": 1, "ff_x": np.array([]), "ff_y": np.array([]),
              "n_cells": 0, "n_ff": 0}
    ff_x, ff_y, n_cells = [], [], 0

    with open(path, "r") as fh:
        in_comp = False
        for line in fh:
            ls = line.strip()
            if ls.startswith("DIEAREA"):
                pts = re.findall(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)", ls)
                if len(pts) >= 2:
                    result["die_w"] = abs(int(pts[1][0]) - int(pts[0][0]))
                    result["die_h"] = abs(int(pts[1][1]) - int(pts[0][1]))
            elif ls.startswith("COMPONENTS") and not ls.startswith("END"):
                in_comp = True
            elif ls.startswith("END COMPONENTS"):
                in_comp = False
            elif in_comp and ls.startswith("- "):
                n_cells += 1
                parts = ls.split()
                if len(parts) < 3:
                    continue
                ct = parts[2]
                if any(k in ct for k in ("decap", "fill", "tap", "tie", "PHY_")):
                    continue
                if _FF_PATTERN.match(ct):
                    m = re.search(r"(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)", ls)
                    if m:
                        ff_x.append(int(m.group(1)))
                        ff_y.append(int(m.group(2)))
    result["ff_x"] = np.array(ff_x, dtype=np.float64)
    result["ff_y"] = np.array(ff_y, dtype=np.float64)
    result["n_cells"] = n_cells
    result["n_ff"] = len(ff_x)
    return result


def extract_def_features(info: dict) -> np.ndarray:
    """
    Extract 36 spatial features from DEF parse result.
    All normalized by die size → design-agnostic.
    """
    die_w = max(info["die_w"], 1)
    die_h = max(info["die_h"], 1)
    n_ff = info["n_ff"]
    n_cells = max(info["n_cells"], 1)

    if n_ff < 2:
        return np.zeros(36, dtype=np.float32)

    fx = info["ff_x"] / die_w
    fy = info["ff_y"] / die_h

    # Basic spatial stats
    hpwl = (fx.max() - fx.min()) + (fy.max() - fy.min())
    cx, cy = fx.mean(), fy.mean()
    sx, sy = fx.std(), fy.std()
    die_ar = die_w / max(die_h, 1)

    # Centroid offset from die center
    centroid_dist = np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2)

    # Grid entropy (8×8)
    bins = np.linspace(0, 1, 9)
    hist2d, _, _ = np.histogram2d(fx, fy, bins=[bins, bins])
    hist2d_flat = hist2d.flatten() + 1e-9
    grid_ent = float(scipy_entropy(hist2d_flat / hist2d_flat.sum()))
    x_hist = np.histogram(fx, bins=bins)[0].astype(float) + 1e-9
    y_hist = np.histogram(fy, bins=bins)[0].astype(float) + 1e-9
    x_ent = float(scipy_entropy(x_hist / x_hist.sum()))
    y_ent = float(scipy_entropy(y_hist / y_hist.sum()))

    # kNN approximation (sorted 1D differences)
    fxs, fys = np.sort(fx), np.sort(fy)
    nn_x = np.diff(fxs).mean() if len(fxs) > 1 else 0.0
    nn_y = np.diff(fys).mean() if len(fys) > 1 else 0.0

    # kNN-4 approximation using 2D
    # Estimate by combining sorted x+y differences (fast proxy for actual kNN)
    combined = np.sqrt(np.diff(fxs) ** 2 + np.diff(np.sort(fy)) ** 2)
    knn4_mean = np.percentile(combined, 25) if len(combined) > 4 else 0.0
    knn4_std = combined.std() if len(combined) > 4 else 0.0
    knn4_max = combined.max() if len(combined) > 0 else 0.0
    knn4_p90 = np.percentile(combined, 90) if len(combined) > 4 else 0.0

    # Quadrant balance
    q1 = ((fx < 0.5) & (fy < 0.5)).sum()
    q2 = ((fx >= 0.5) & (fy < 0.5)).sum()
    q3 = ((fx < 0.5) & (fy >= 0.5)).sum()
    q4 = ((fx >= 0.5) & (fy >= 0.5)).sum()
    q_counts = np.array([q1, q2, q3, q4], dtype=float) + 1
    q_balance = 1 - _gini(q_counts)

    # Gini of spatial coordinates
    x_gini = _gini(fx)
    y_gini = _gini(fy)

    # Launch-capture asymmetry proxy (max quadrant vs min quadrant)
    q_max, q_min = q_counts.max(), q_counts.min()
    lc_asymmetry = (q_max - q_min) / (q_max + q_min + 1e-9)

    # Density at 3 radii (fraction of FFs within r of centroid)
    dists = np.sqrt((fx - cx) ** 2 + (fy - cy) ** 2)
    for_r = sorted(dists)
    r25 = np.percentile(dists, 25) if len(dists) > 0 else 0
    r50 = np.percentile(dists, 50) if len(dists) > 0 else 0
    r90 = np.percentile(dists, 90) if len(dists) > 0 else 0

    # Tail statistics for skew (max distance FFs are the worst-case paths)
    max_dist = dists.max() if len(dists) > 0 else 0
    p99_dist = np.percentile(dists, 99) if len(dists) > 0 else 0
    top1_dist = np.sort(dists)[-max(1, n_ff // 100):].mean()  # top 1% distance

    # Spread ratio (aspect ratio of FF placement)
    spread_ratio = np.clip(np.log((sx + 1e-6) / (sy + 1e-6)), -3, 3)

    feats = np.array([
        np.log1p(n_ff),       # 0  log(n_ff)
        np.log1p(n_cells),    # 1
        n_ff / n_cells,       # 2  FF fraction
        die_ar,               # 3  die aspect ratio
        hpwl,                 # 4  ← KEY: FF bounding box (DME proxy)
        cx, cy,               # 5,6
        sx, sy,               # 7,8
        spread_ratio,         # 9
        centroid_dist,        # 10 clock entry point (Kahng insight)
        grid_ent,             # 11
        x_ent, y_ent,         # 12,13
        x_gini, y_gini,       # 14,15
        nn_x, nn_y,           # 16,17 nearest-neighbor spacing
        knn4_mean,            # 18 kNN-4 mean
        knn4_std,             # 19
        knn4_max,             # 20
        knn4_p90,             # 21
        q_balance,            # 22 quadrant balance
        lc_asymmetry,         # 23 launch-capture asymmetry proxy
        r25, r50, r90,        # 24,25,26 radial density
        max_dist,             # 27 max distance from centroid
        p99_dist,             # 28 p99 distance (worst-case skew proxy)
        top1_dist,            # 29 top 1% distance (tail statistic)
        np.log1p(die_w),      # 30 absolute die scale
        np.log1p(die_h),      # 31
        hpwl * np.log1p(n_ff),  # 32 interaction: spread × count
        np.log1p(n_ff * hpwl),  # 33 log(n_ff × hpwl) → WL proxy
        sx * sy,              # 34 spread area
        (p99_dist - r50) / (r50 + 1e-6),  # 35 tail-to-median ratio (skew proxy)
    ], dtype=np.float32)
    return feats


def parse_saif(path: str) -> dict:
    """Fast SAIF parse: extract all toggle counts."""
    with open(path, "r") as fh:
        content = fh.read()
    tc_vals = np.array([int(x) for x in re.findall(r"\(TC\s+(\d+)\)", content)],
                       dtype=np.float64)
    if len(tc_vals) == 0:
        return {"total": 0, "mean": 0, "std": 0, "max": 0, "log_total": 0,
                "nonzero": 0, "p90": 0, "p50": 0, "gini": 0, "n": 0}
    tc_pos = tc_vals[tc_vals > 0]
    return {
        "total":     float(tc_vals.sum()),
        "mean":      float(tc_vals.mean()),
        "std":       float(tc_vals.std()),
        "max":       float(tc_vals.max()),
        "log_total": float(np.log1p(tc_vals.sum())),
        "nonzero":   len(tc_pos) / max(len(tc_vals), 1),
        "p90":       float(np.percentile(tc_vals, 90)),
        "p50":       float(np.percentile(tc_vals, 50)),
        "gini":      _gini(tc_pos) if len(tc_pos) > 1 else 0.0,
        "n":         float(len(tc_vals)),
    }


def extract_saif_features(act: dict) -> np.ndarray:
    """Extract 10 activity features."""
    return np.array([
        act["log_total"],                                        # 0 log(total TC) ← KEY for power
        act["mean"],                                             # 1
        act["std"],                                              # 2
        np.log1p(act["max"]),                                    # 3
        act["nonzero"],                                          # 4
        act["p90"],                                              # 5
        act["p50"],                                              # 6
        act["gini"],                                             # 7
        np.log1p(act["n"]),                                      # 8
        act["log_total"] / max(np.log1p(act["n"]), 1),          # 9 activity/net
    ], dtype=np.float32)


def extract_timing_features(path: str) -> np.ndarray:
    """
    Extract 22 features from pre-CTS timing paths CSV.
    Based on PreRoutGNN (AAAI 2024): timing slack distribution → skew predictor.
    Extra tail statistics preserved for skew (worst-case metric).
    """
    t = pd.read_csv(path)
    if "slack" not in t.columns or len(t) == 0:
        return np.zeros(22, dtype=np.float32)

    slack = t["slack"].values.astype(np.float64)
    n_paths = len(slack)
    n_launch = t["launch_flop"].nunique() if "launch_flop" in t.columns else 0
    n_capture = t["capture_flop"].nunique() if "capture_flop" in t.columns else 0
    n_ff = len(
        set(t.get("launch_flop", pd.Series([])).tolist()) |
        set(t.get("capture_flop", pd.Series([])).tolist())
    )

    s_min, s_max = slack.min(), slack.max()
    s_mean, s_std = slack.mean(), slack.std()
    s_range = s_max - s_min  # timing window width → skew proxy
    p5 = np.percentile(slack, 5)
    p10 = np.percentile(slack, 10)
    p25 = np.percentile(slack, 25)
    p75 = np.percentile(slack, 75)
    p90 = np.percentile(slack, 90)
    p95 = np.percentile(slack, 95)
    iqr = p75 - p25
    vio_frac = (slack < 0).sum() / max(n_paths, 1)

    # Tail asymmetry: left tail vs right tail (critical for skew)
    tail_left = np.abs(s_mean - p5)
    tail_right = np.abs(p95 - s_mean)
    tail_asym = (tail_left - tail_right) / (tail_left + tail_right + 1e-9)

    # Critical path imbalance
    crit = (slack < p10).sum()
    norm = (slack > p90).sum()
    imbalance = abs(crit - norm) / max(crit + norm, 1)

    # Path fan-out: high fan-out → many capture FFs per launch → more skew
    paths_per_launch = n_paths / max(n_launch, 1)
    paths_per_ff = n_paths / max(n_ff, 1)

    feats = np.array([
        np.log1p(n_paths),    # 0
        np.log1p(n_launch),   # 1
        np.log1p(n_capture),  # 2
        np.log1p(n_ff),       # 3
        s_min,                # 4
        s_max,                # 5
        s_mean,               # 6
        s_std,                # 7
        s_range,              # 8  ← KEY
        p5,                   # 9
        p10,                  # 10
        p25,                  # 11
        p75,                  # 12
        p90,                  # 13
        p95,                  # 14
        iqr,                  # 15
        vio_frac,             # 16
        imbalance,            # 17
        tail_asym,            # 18
        paths_per_launch,     # 19
        paths_per_ff,         # 20
        s_std / max(s_range, 1e-6),  # 21 slack dispersion ratio
    ], dtype=np.float32)
    return feats


def extract_knob_features(row: pd.Series) -> np.ndarray:
    """
    Extract 15 CTS knob features + physics-motivated interactions.
    Based on iCTS (TCAD 2025) and GAN-CTS (TCAD 2022).
    """
    mw = float(row.get("cts_max_wire", 200))
    bd = float(row.get("cts_buf_dist", 110))
    cs = float(row.get("cts_cluster_size", 21))
    cd = float(row.get("cts_cluster_dia", 52))

    z_mw = float(row.get("z_cts_max_wire", 0))
    z_bd = float(row.get("z_cts_buf_dist", 0))
    z_cs = float(row.get("z_cts_cluster_size", 0))
    z_cd = float(row.get("z_cts_cluster_dia", 0))

    # Physics interactions
    # Skew: buf_dist/skip_max proxy → equalization capacity
    bd_mw = z_bd * z_mw
    cs_cd = z_cs * z_cd
    bd_cs = z_bd * z_cs
    mw_cd = z_mw * z_cd
    # WL: cluster_dia × HPWL → captures how clusters reduce routing
    # Power: n_ff/cluster_size → buffer count proxy
    # 3-way interaction
    bd_cs_cd = z_bd * z_cs * z_cd
    mw_cs_cd = z_mw * z_cs * z_cd

    feats = np.array([
        mw / 200, bd / 110, cs / 21, cd / 52,    # 0-3 normalized raw
        z_mw, z_bd, z_cs, z_cd,                   # 4-7 z-scored
        bd_mw,                                     # 8
        cs_cd,                                     # 9
        bd_cs,                                     # 10
        mw_cd,                                     # 11
        bd_cs_cd,                                  # 12
        mw_cs_cd,                                  # 13
        np.log1p(mw) - np.log(200),               # 14 log-normalized
    ], dtype=np.float32)
    return feats


def get_placement_features(pid: str, def_path: str, saif_path: str,
                           timing_path: str) -> tuple:
    """Cache placement-level features (same across 10 CTS runs)."""
    if pid in _placement_cache:
        return _placement_cache[pid]
    t = extract_timing_features(timing_path)
    info = parse_def(def_path)
    d = extract_def_features(info)
    act = parse_saif(saif_path)
    a = extract_saif_features(act)
    _placement_cache[pid] = (t, d, a, info["n_ff"], act["log_total"])
    return _placement_cache[pid]


def extract_row_features(row: pd.Series, base_dir: str) -> np.ndarray:
    """Extract full 83-dim feature vector for one manifest row."""
    def resolve(p):
        p = str(p)
        if p.startswith("../"):
            return str(Path(base_dir) / p[3:])
        return str(Path(base_dir) / p)

    def_path = resolve(row["def_path"])
    saif_path = resolve(row["saif_path"])
    tim_path = resolve(row["timing_path_csv"])
    pid = str(row.get("placement_id", ""))

    t, d, a, n_ff, act_log = get_placement_features(pid, def_path, saif_path, tim_path)
    k = extract_knob_features(row)

    # Cross-feature interaction: geometry × knobs (physics-motivated)
    hpwl = d[4]
    log_nff = d[0]
    spread_x, spread_y = d[7], d[8]
    knn4 = d[18]
    centroid_dist = d[10]
    p99_dist = d[28]
    timing_range = t[8]
    slack_std = t[7]
    z_bd = k[5]
    z_cs = k[6]
    z_cd = k[7]
    z_mw = k[4]

    # Physics-motivated cross features
    # Skew interactions (from CLAUDE.md: iCTS literature)
    f_skew = np.array([
        z_bd / (p99_dist + 1e-6),       # buf_dist / max_path → equalization
        z_cd / (knn4 + 1e-6),            # cluster_dia / kNN → grouping quality
        z_cd * centroid_dist,            # cluster_dia × centroid offset
        timing_range * z_bd,             # timing window × buf_dist
        p99_dist * timing_range,         # worst path × timing spread
    ], dtype=np.float32)

    # Power interactions
    f_power = np.array([
        log_nff - np.log(z_cs + 1e-6) if z_cs > -2 else 0.0,  # log(n_ff/cluster_size)
        z_cd * hpwl,                     # cluster_dia × HPWL
        z_mw * hpwl,                     # max_wire × HPWL
        act_log * z_cs,                  # activity × cluster_size
    ], dtype=np.float32)

    # WL interactions (DME: WL ≈ 1.1-1.5 × HPWL)
    f_wl = np.array([
        z_cd * hpwl,                     # cluster_dia × HPWL ← KEY
        z_cs / (hpwl + 1e-6),            # cluster_size / HPWL
        z_mw / (hpwl + 1e-6),            # max_wire / HPWL
        np.log1p(n_ff) * hpwl,           # n_ff × HPWL (size-aware WL)
    ], dtype=np.float32)

    return np.concatenate([t, d, a, k, f_skew, f_power, f_wl]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

def build_dataset(
    manifest_csv: Path,
    base_dir: str,
    cache_path: Path | None = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build (X, Y_perplacement, df) with per-placement z-score targets."""
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        if verbose:
            print(f"  Loaded from cache: {cache_path} ({d['X'].shape[0]} samples)")
        return d["X"], d["Y"], d["df"]

    df = pd.read_csv(manifest_csv)
    _placement_cache.clear()

    X_list, valid = [], []
    for i, row in df.iterrows():
        try:
            x = extract_row_features(row, base_dir)
            X_list.append(x)
            valid.append(i)
        except Exception as exc:
            if verbose:
                print(f"  [skip {i}] {exc}")

    X = np.stack(X_list).astype(np.float32)
    df = df.iloc[valid].reset_index(drop=True)

    # Per-placement z-score targets
    Y = compute_per_placement_targets(df)

    if verbose:
        n_feat = X.shape[1]
        print(f"  Dataset: {len(X)} samples, {n_feat} features")
        print(f"  Feature dim: {n_feat} "
              f"(timing=22, def=36, saif=10, knobs=15, cross=13)")
        for j, t in enumerate(["skew_pp", "power_pp", "wl_pp"]):
            print(f"  {t}: mean={Y[:,j].mean():.4f} std={Y[:,j].std():.4f}")
        print(f"  Designs: {df['design_name'].value_counts().to_dict()}")

    if cache_path:
        with open(cache_path, "wb") as f:
            pickle.dump({"X": X, "Y": Y, "df": df}, f)
        if verbose:
            print(f"  Cached → {cache_path}")

    return X, Y, df


# ─────────────────────────────────────────────────────────────────────────────
#  Models
# ─────────────────────────────────────────────────────────────────────────────

# LightGBM hyperparams
_LGB_SKEW = dict(
    n_estimators=3000, learning_rate=0.01, num_leaves=31, max_depth=6,
    min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=0.2, n_jobs=-1, verbose=-1,
)
_LGB_POWER = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=63, max_depth=7,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, n_jobs=-1, verbose=-1,
)
_LGB_WL = dict(
    n_estimators=2000, learning_rate=0.02, num_leaves=63, max_depth=7,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.1, n_jobs=-1, verbose=-1,
)
_LGB_PARAMS = [_LGB_SKEW, _LGB_POWER, _LGB_WL]

_XGB_BASE = dict(
    n_estimators=1500, learning_rate=0.02, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
    n_jobs=-1, verbosity=0,
)


class CTSEnsemble:
    """
    LightGBM + XGBoost + Ridge meta-learner ensemble.
    Separate models per task (skew needs different treatment from power/WL).
    """

    def __init__(self):
        self.lgb_ = []
        self.xgb_ = []
        self.meta_ = []
        self.scaler_ = RobustScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose=False) -> "CTSEnsemble":
        X_sc = self.scaler_.fit_transform(X)
        self.lgb_, self.xgb_, self.meta_ = [], [], []
        for j in range(3):
            y = Y[:, j]
            lgb_m = lgb.LGBMRegressor(**_LGB_PARAMS[j])
            lgb_m.fit(X_sc, y)

            xgb_m = xgb.XGBRegressor(**_XGB_BASE)
            xgb_m.fit(X_sc, y)

            meta_X = np.column_stack([lgb_m.predict(X_sc), xgb_m.predict(X_sc)])
            meta = Ridge(alpha=1.0)
            meta.fit(meta_X, y)

            self.lgb_.append(lgb_m)
            self.xgb_.append(xgb_m)
            self.meta_.append(meta)

            if verbose:
                pred = meta.predict(meta_X)
                print(f"    Task {j}: train MAE={mean_absolute_error(y, pred):.4f}")

        self._fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X_sc = self.scaler_.transform(X)
        out = []
        for j in range(3):
            lp = self.lgb_[j].predict(X_sc)
            xp = self.xgb_[j].predict(X_sc)
            out.append(self.meta_[j].predict(np.column_stack([lp, xp])))
        return np.stack(out, axis=1)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"lgb": self.lgb_, "xgb": self.xgb_,
                        "meta": self.meta_, "scaler": self.scaler_}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls()
        obj.lgb_, obj.xgb_, obj.meta_, obj.scaler_ = d["lgb"], d["xgb"], d["meta"], d["scaler"]
        obj._fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
#  LODO Cross-Validation
# ─────────────────────────────────────────────────────────────────────────────

TASK_NAMES = ["skew", "power", "wl"]


def lodo_cv(
    X: np.ndarray,
    Y: np.ndarray,
    df: pd.DataFrame,
    model_cls=CTSEnsemble,
    verbose: bool = True,
) -> pd.DataFrame:
    """Leave-One-Design-Out CV: train on 3 designs, test on 1."""
    designs = df["design_name"].unique()
    results = []

    for test_design in designs:
        mask = (df["design_name"] == test_design).values
        X_tr, Y_tr = X[~mask], Y[~mask]
        X_te, Y_te = X[mask], Y[mask]

        model = model_cls()
        model.fit(X_tr, Y_tr, verbose=False)
        Y_pred = model.predict(X_te)

        row = {"test_design": test_design, "n_test": int(mask.sum())}
        for j, t in enumerate(TASK_NAMES):
            mae = mean_absolute_error(Y_te[:, j], Y_pred[:, j])
            r2 = r2_score(Y_te[:, j], Y_pred[:, j])
            rho = spearmanr(Y_te[:, j], Y_pred[:, j])[0]
            row[f"{t}_mae"] = mae
            row[f"{t}_r2"] = r2
            row[f"{t}_rho"] = rho

        results.append(row)
        if verbose:
            s, p, w = row["skew_mae"], row["power_mae"], row["wl_mae"]
            ok_s = "✓" if s < 0.10 else "✗"
            ok_p = "✓" if p < 0.10 else "✗"
            ok_w = "✓" if w < 0.10 else "✗"
            print(f"  [{test_design}] n={row['n_test']:4d} | "
                  f"{ok_s}skew={s:.4f}  {ok_p}power={p:.4f}  {ok_w}wl={w:.4f}")

    cv_df = pd.DataFrame(results)
    return cv_df


def print_cv_summary(cv_df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  LODO CV Summary (mean MAE per target)")
    print("=" * 60)
    for t in TASK_NAMES:
        m = cv_df[f"{t}_mae"].mean()
        s = cv_df[f"{t}_mae"].std()
        ok = "✓" if m < 0.10 else "✗"
        print(f"  {ok} {t:8s}: {m:.4f} ± {s:.4f}")
    all_ok = all(cv_df[f"{t}_mae"].mean() < 0.10 for t in TASK_NAMES)
    if all_ok:
        print("\n  ★★★ ALL TARGETS < 0.10 — GOAL ACHIEVED! ★★★")
    else:
        missing = [t for t in TASK_NAMES if cv_df[f"{t}_mae"].mean() >= 0.10]
        print(f"\n  Still needs work: {missing}")


# ─────────────────────────────────────────────────────────────────────────────
#  Feature importance
# ─────────────────────────────────────────────────────────────────────────────

FEAT_NAMES = (
    [f"timing_{i}" for i in range(22)] +
    [f"def_{i}" for i in range(36)] +
    [f"saif_{i}" for i in range(10)] +
    [f"knob_{i}" for i in range(15)] +
    [f"x_skew_{i}" for i in range(5)] +
    [f"x_power_{i}" for i in range(4)] +
    [f"x_wl_{i}" for i in range(4)]
)


def print_feature_importance(model: CTSEnsemble, top_k: int = 10):
    print("\n── Top Feature Importances ──")
    for j, t in enumerate(TASK_NAMES):
        imp = model.lgb_[j].feature_importances_
        top = np.argsort(imp)[::-1][:top_k]
        print(f"\n  {t}:")
        for i in top:
            name = FEAT_NAMES[i] if i < len(FEAT_NAMES) else f"feat_{i}"
            print(f"    {name:20s} {imp[i]:6.0f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true", help="Run LODO CV")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--save", default="cts_ensemble_v2.pkl")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 1 fold, 500 trees")
    args = parser.parse_args()

    base_dir = str(BASE)
    cache = None if args.no_cache else BASE / "cache_v2_train.pkl"

    print("=" * 60)
    print("CTS Generalizable Predictor v2")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    print("\nBuilding dataset …")
    X, Y, df = build_dataset(MANIFEST_CSV, base_dir, cache_path=cache, verbose=True)
    print(f"Feature dim: {X.shape[1]}")

    if args.quick:
        # Quick test: 1 fold only
        print("\n── QUICK TEST: 1 LODO fold ──")
        test_design = df["design_name"].unique()[0]
        mask = (df["design_name"] == test_design).values
        m = CTSEnsemble()
        # Use fewer trees
        for p in _LGB_PARAMS:
            p["n_estimators"] = 200
        m.fit(X[~mask], Y[~mask])
        yp = m.predict(X[mask])
        for j, t in enumerate(TASK_NAMES):
            mae = mean_absolute_error(Y[mask, j], yp[:, j])
            r2 = r2_score(Y[mask, j], yp[:, j])
            print(f"  {t}: MAE={mae:.4f}  R²={r2:.4f}  {'OK' if mae<0.10 else 'MISS'}")
        return

    if args.cv:
        print("\n── Leave-One-Design-Out CV ──")
        cv_df = lodo_cv(X, Y, df, verbose=True)
        print_cv_summary(cv_df)

        # Save CV results
        cv_path = BASE / "lodo_cv_results_v2.csv"
        cv_df.to_csv(cv_path, index=False)
        print(f"\nCV results saved → {cv_path}")

    # Train final model
    print("\n── Training final model (all data) ──")
    model = CTSEnsemble()
    model.fit(X, Y, verbose=True)
    model.save(args.save)
    print(f"Model saved → {args.save}")

    print_feature_importance(model)


if __name__ == "__main__":
    main()
