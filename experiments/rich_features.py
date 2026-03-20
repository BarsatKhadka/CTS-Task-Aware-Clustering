"""
Rich feature extraction from DEF + SAIF + timing path CSV.

Based on literature insights:
  - PreRoutGNN (AAAI 2024): timing path slack distribution → best predictor of skew
  - PowPrediCT (DAC 2024): total switching activity × capacitance → predicts power
  - iCTS (DAC 2024): FF spatial spread × cluster params → predicts WL and skew
  - GAN-CTS (TCAD 2022): placement image features + knob interactions

Feature groups:
  [0-16]  Timing path features  (17 feats) – pre-CTS STA slack distribution
  [17-38] DEF spatial features  (22 feats) – FF positions, die geometry
  [39-48] SAIF activity features(10 feats) – toggle counts
  [49-59] CTS knob features     (11 feats) – knob values + interactions
  Total: 60 features
"""

from __future__ import annotations

import os
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
DATASET_DIR = BASE / "dataset_with_def"
PLACEMENT_DIR = DATASET_DIR / "placement_files"

# Sky130 flip-flop cell-type patterns (positive match → is a FF)
_FF_PATTERNS = re.compile(
    r"sky130_fd_sc_hd__(?:df|sdff|edf|sdfsrn|sdfbbn|dfbbn|dfrtp|dfxbp|dfxtp|dfbbp|df[a-z])",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
#  1. DEF parser  (extracts FF positions and die bounding box)
# ─────────────────────────────────────────────────────────────────────────────

def parse_def(def_path: str | Path) -> dict:
    """
    Parse a DEF placement file. Returns dict with:
      - die_w, die_h      : die dimensions (nm)
      - ff_x, ff_y        : np.ndarray of FF x/y positions (nm)
      - n_cells           : total placed cell count
      - n_ff              : number of flip-flops
    """
    def_path = str(def_path)
    result = {"die_w": 1, "die_h": 1, "ff_x": np.array([]), "ff_y": np.array([]),
              "n_cells": 0, "n_ff": 0}

    ff_x, ff_y = [], []
    n_cells = 0

    with open(def_path, "r") as fh:
        in_comp = False
        for line in fh:
            ls = line.strip()

            # Die area
            if ls.startswith("DIEAREA"):
                pts = re.findall(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)", ls)
                if len(pts) >= 2:
                    result["die_w"] = abs(int(pts[1][0]) - int(pts[0][0]))
                    result["die_h"] = abs(int(pts[1][1]) - int(pts[0][1]))
                continue

            # Components section start/end
            if ls.startswith("COMPONENTS") and not ls.startswith("END"):
                in_comp = True
                continue
            if ls.startswith("END COMPONENTS"):
                in_comp = False
                continue

            # Cell instance line  (starts with "- ")
            if in_comp and ls.startswith("- "):
                n_cells += 1
                # Check cell type for FF
                parts = ls.split()
                if len(parts) < 3:
                    continue
                cell_type = parts[2]
                # Skip filler/decap/tap cells
                if any(k in cell_type for k in ("decap", "fill", "tap", "tie",
                                                  "buf_", "PHY_")):
                    continue
                if _FF_PATTERNS.match(cell_type):
                    # Extract position:  PLACED ( x y ) or FIXED ( x y )
                    m = re.search(r"(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)", ls)
                    if m:
                        ff_x.append(int(m.group(1)))
                        ff_y.append(int(m.group(2)))

    result["ff_x"] = np.array(ff_x, dtype=np.float64)
    result["ff_y"] = np.array(ff_y, dtype=np.float64)
    result["n_cells"] = n_cells
    result["n_ff"] = len(ff_x)
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  2. SAIF parser  (extracts toggle counts)
# ─────────────────────────────────────────────────────────────────────────────

def parse_saif_total_activity(saif_path: str | Path) -> dict:
    """
    Fast SAIF parsing: extract total and distribution of toggle counts.
    Returns dict with activity statistics.
    """
    # Use regex to extract all (TC N) values in one pass
    with open(str(saif_path), "r") as fh:
        content = fh.read()

    tc_vals = np.array(
        [int(x) for x in re.findall(r"\(TC\s+(\d+)\)", content)],
        dtype=np.float64,
    )

    if len(tc_vals) == 0:
        return {"tc_total": 0, "tc_mean": 0, "tc_std": 0,
                "tc_max": 0, "tc_log_total": 0, "tc_nonzero_frac": 0,
                "tc_p90": 0, "tc_p50": 0, "tc_gini": 0, "tc_n": 0}

    tc_pos = tc_vals[tc_vals > 0]
    nonzero = len(tc_pos) / max(len(tc_vals), 1)
    tc_gini = _gini(tc_pos) if len(tc_pos) > 1 else 0.0

    return {
        "tc_total":       float(tc_vals.sum()),
        "tc_mean":        float(tc_vals.mean()),
        "tc_std":         float(tc_vals.std()),
        "tc_max":         float(tc_vals.max()),
        "tc_log_total":   float(np.log1p(tc_vals.sum())),
        "tc_nonzero_frac": nonzero,
        "tc_p90":         float(np.percentile(tc_vals, 90)),
        "tc_p50":         float(np.percentile(tc_vals, 50)),
        "tc_gini":        tc_gini,
        "tc_n":           float(len(tc_vals)),
    }


def _gini(arr: np.ndarray) -> float:
    arr = np.sort(arr.flatten())
    n = len(arr)
    if n < 2 or arr.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


# ─────────────────────────────────────────────────────────────────────────────
#  3. Timing path feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_timing_features(timing_csv: str | Path) -> np.ndarray:
    """
    Extract 17 features from pre-CTS timing paths CSV.
    Columns: launch_flop, capture_flop, slack

    Key insight (PreRoutGNN, AAAI 2024): slack distribution is the best
    predictor of post-CTS clock skew.
    """
    t = pd.read_csv(str(timing_csv))
    if "slack" not in t.columns or len(t) == 0:
        return np.zeros(17, dtype=np.float32)

    slack = t["slack"].values.astype(np.float64)

    # Path count features
    n_paths     = len(slack)
    n_launch    = t["launch_flop"].nunique() if "launch_flop" in t.columns else 0
    n_capture   = t["capture_flop"].nunique() if "capture_flop" in t.columns else 0
    n_unique_ff = len(
        set(t["launch_flop"].tolist() if "launch_flop" in t.columns else []) |
        set(t["capture_flop"].tolist() if "capture_flop" in t.columns else [])
    )

    # Slack statistics
    s_min   = slack.min()
    s_max   = slack.max()
    s_mean  = slack.mean()
    s_std   = slack.std()
    s_range = s_max - s_min          # ≈ pre-CTS timing window → predictor of skew
    s_p10   = np.percentile(slack, 10)
    s_p25   = np.percentile(slack, 25)
    s_p75   = np.percentile(slack, 75)
    s_p90   = np.percentile(slack, 90)
    s_iqr   = s_p75 - s_p25

    # Violation fraction
    vio_frac = (slack < 0).sum() / max(n_paths, 1)

    # Critical path imbalance
    critical = slack < np.percentile(slack, 10)
    normal   = slack > np.percentile(slack, 90)
    imbalance = (np.abs(critical.sum() - normal.sum()) /
                 max(critical.sum() + normal.sum(), 1))

    feats = np.array([
        np.log1p(n_paths),               # 0
        np.log1p(n_launch),              # 1
        np.log1p(n_capture),             # 2
        np.log1p(n_unique_ff),           # 3
        s_min,                           # 4
        s_max,                           # 5
        s_mean,                          # 6
        s_std,                           # 7
        s_range,                         # 8  ← KEY: timing window width
        s_p10,                           # 9
        s_p25,                           # 10
        s_p75,                           # 11
        s_p90,                           # 12
        s_iqr,                           # 13
        vio_frac,                        # 14
        imbalance,                       # 15
        n_paths / max(n_unique_ff, 1),   # 16  paths-per-FF ratio
    ], dtype=np.float32)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
#  4. DEF spatial feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_def_features(def_info: dict) -> np.ndarray:
    """
    Extract 22 spatial features from parsed DEF info.
    All dimensions normalized by die size to be design-agnostic.
    """
    die_w = max(def_info["die_w"], 1)
    die_h = max(def_info["die_h"], 1)
    n_ff  = def_info["n_ff"]
    n_cells = max(def_info["n_cells"], 1)
    ff_x  = def_info["ff_x"] / die_w   # normalize to [0,1]
    ff_y  = def_info["ff_y"] / die_h

    if n_ff < 2:
        return np.array(
            [0, np.log1p(n_cells), 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, die_w / max(die_h, 1)],
            dtype=np.float32,
        )

    # Scale features
    die_ar    = die_w / max(die_h, 1)
    ff_frac   = n_ff / n_cells
    log_nff   = np.log1p(n_ff)
    log_ncell = np.log1p(n_cells)

    # FF spatial statistics (normalized)
    hpwl  = (ff_x.max() - ff_x.min()) + (ff_y.max() - ff_y.min())
    cx    = ff_x.mean()
    cy    = ff_y.mean()
    sx    = ff_x.std()
    sy    = ff_y.std()
    spread_ratio = np.clip(np.log((ff_x.std() + 1e-6) /
                                   (ff_y.std() + 1e-6)), -3, 3)

    # 8×8 spatial grid
    bins = np.linspace(0, 1, 9)
    hist2d, _, _ = np.histogram2d(ff_x, ff_y, bins=[bins, bins])
    hist2d_flat = hist2d.flatten() + 1e-9
    grid_ent  = float(scipy_entropy(hist2d_flat / hist2d_flat.sum()))

    x_hist = np.histogram(ff_x, bins=bins)[0].astype(float) + 1e-9
    y_hist = np.histogram(ff_y, bins=bins)[0].astype(float) + 1e-9
    x_ent  = float(scipy_entropy(x_hist / x_hist.sum()))
    y_ent  = float(scipy_entropy(y_hist / y_hist.sum()))
    x_gini = _gini(ff_x)
    y_gini = _gini(ff_y)

    # Nearest-neighbor distances (approximate via sorted 1D)
    ffxs   = np.sort(ff_x)
    nn_x   = np.diff(ffxs).mean() if len(ffxs) > 1 else 0
    ffys   = np.sort(ff_y)
    nn_y   = np.diff(ffys).mean() if len(ffys) > 1 else 0

    # Quadrant balance (how evenly FFs are distributed across 4 quadrants)
    q1 = ((ff_x < 0.5) & (ff_y < 0.5)).sum()
    q2 = ((ff_x >= 0.5) & (ff_y < 0.5)).sum()
    q3 = ((ff_x < 0.5) & (ff_y >= 0.5)).sum()
    q4 = ((ff_x >= 0.5) & (ff_y >= 0.5)).sum()
    q_counts = np.array([q1, q2, q3, q4], dtype=float)
    q_balance = 1 - _gini(q_counts + 1)   # 1 = perfectly balanced

    feats = np.array([
        log_nff,          # 0
        log_ncell,        # 1
        ff_frac,          # 2
        die_ar,           # 3
        hpwl,             # 4  ← KEY: FF bounding box
        cx, cy,           # 5,6  centroid
        sx, sy,           # 7,8  spatial spread
        spread_ratio,     # 9
        grid_ent,         # 10
        x_ent, y_ent,     # 11,12
        x_gini, y_gini,   # 13,14
        nn_x, nn_y,       # 15,16  nn distances
        q_balance,        # 17
        np.log1p(die_w),  # 18  absolute die scale
        np.log1p(die_h),  # 19
        hpwl * log_nff,   # 20  interaction: spread × count (→ WL proxy)
        np.log1p(n_ff * hpwl),  # 21  log(n_ff × hpwl)
    ], dtype=np.float32)

    return feats


# ─────────────────────────────────────────────────────────────────────────────
#  5. SAIF feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_saif_features(act: dict) -> np.ndarray:
    """Extract 10 activity features from parsed SAIF stats."""
    return np.array([
        act["tc_log_total"],     # 0  log(total toggle count) ← KEY for power
        act["tc_mean"],          # 1
        act["tc_std"],           # 2
        np.log1p(act["tc_max"]), # 3
        act["tc_nonzero_frac"],  # 4
        act["tc_p90"],           # 5
        act["tc_p50"],           # 6
        act["tc_gini"],          # 7
        np.log1p(act["tc_n"]),   # 8
        act["tc_log_total"] / max(np.log1p(act["tc_n"]), 1),  # 9  activity per net
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  6. CTS knob feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_knob_features(row: pd.Series) -> np.ndarray:
    """
    Extract 11 CTS knob features + interaction terms.
    Uses raw knob values + z-scored versions (if available).
    """
    mw  = float(row.get("cts_max_wire",    200))
    bd  = float(row.get("cts_buf_dist",    110))
    cs  = float(row.get("cts_cluster_size", 21))
    cd  = float(row.get("cts_cluster_dia",  52))

    # Z-scored versions (from manifest)
    z_mw = float(row.get("z_cts_max_wire",    0))
    z_bd = float(row.get("z_cts_buf_dist",    0))
    z_cs = float(row.get("z_cts_cluster_size", 0))
    z_cd = float(row.get("z_cts_cluster_dia",  0))

    # Interaction terms (literature: iCTS, GAN-CTS)
    # buf_dist × max_wire → controls buffer insertion density
    # cluster_dia × cluster_size → controls sink grouping
    intr_bd_mw  = z_bd * z_mw
    intr_cs_cd  = z_cs * z_cd
    intr_bd_cs  = z_bd * z_cs

    return np.array([
        mw / 200, bd / 110, cs / 21, cd / 52,   # 0-3 normalized raw
        z_mw, z_bd, z_cs, z_cd,                  # 4-7 z-scored
        intr_bd_mw,                               # 8
        intr_cs_cd,                               # 9
        intr_bd_cs,                               # 10
    ], dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  7. Placement-level cache
# ─────────────────────────────────────────────────────────────────────────────

_placement_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def get_placement_features(placement_id: str,
                           def_path: str,
                           saif_path: str,
                           timing_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (timing_feats[17], def_feats[22], saif_feats[10]) for a placement.
    Results are cached in memory (same placement → same result).
    """
    if placement_id in _placement_cache:
        return _placement_cache[placement_id]

    t_feats = extract_timing_features(timing_path)
    d_info  = parse_def(def_path)
    d_feats = extract_def_features(d_info)
    a_info  = parse_saif_total_activity(saif_path)
    a_feats = extract_saif_features(a_info)

    _placement_cache[placement_id] = (t_feats, d_feats, a_feats)
    return t_feats, d_feats, a_feats


# ─────────────────────────────────────────────────────────────────────────────
#  8. Full row feature extractor
# ─────────────────────────────────────────────────────────────────────────────

def extract_row_features(row: pd.Series, base_dir: str | Path = BASE) -> np.ndarray:
    """
    Extract complete 60-dim feature vector for one manifest row.
    Combines placement features (timing+DEF+SAIF) + CTS knob features.
    """
    base_dir = Path(base_dir)

    def resolve(p):
        p = str(p)
        if p.startswith("../"):
            return str(base_dir / p[3:])
        return str(base_dir / p)

    def_path    = resolve(row["def_path"])
    saif_path   = resolve(row["saif_path"])
    timing_path = resolve(row["timing_path_csv"])

    pid = str(row.get("placement_id", ""))
    t_feats, d_feats, a_feats = get_placement_features(
        pid, def_path, saif_path, timing_path
    )
    k_feats = extract_knob_features(row)

    return np.concatenate([t_feats, d_feats, a_feats, k_feats]).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  9. Dataset builder
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_DIM = 60
TARGETS     = ["z_skew_setup", "z_power_total", "z_wirelength"]


def build_dataset(
    manifest_path: str | Path,
    base_dir: str | Path = BASE,
    verbose: bool = True,
    cache_path: Optional[str | Path] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Build (X, Y, df) from a manifest CSV.

    X: [N, 60]  float32
    Y: [N, 3]   float32  (z_skew, z_power, z_wl)
    """
    import pickle

    # Load from cache if available
    if cache_path and Path(cache_path).exists():
        with open(cache_path, "rb") as f:
            d = pickle.load(f)
        if verbose:
            print(f"Loaded from cache: {cache_path}  "
                  f"({d['X'].shape[0]} samples)")
        return d["X"], d["Y"], d["df"]

    df = pd.read_csv(manifest_path)
    X_list, Y_list, valid = [], [], []

    for i, row in df.iterrows():
        try:
            x = extract_row_features(row, base_dir)
            X_list.append(x)
        except Exception as exc:
            if verbose:
                print(f"  [skip row {i}] {exc}")
            continue

        y = np.array([
            float(row["z_skew_setup"]),
            float(row["z_power_total"]),
            float(row["z_wirelength"]),
        ], dtype=np.float32)
        Y_list.append(y)
        valid.append(i)

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(df)} rows processed …")

    X = np.stack(X_list).astype(np.float32)
    Y = np.stack(Y_list).astype(np.float32)
    df = df.iloc[valid].reset_index(drop=True)

    if verbose:
        print(f"Dataset: {len(X)} samples, {X.shape[1]} features")
        for j, t in enumerate(TARGETS):
            print(f"  {t}: mean={Y[:,j].mean():.4f}  std={Y[:,j].std():.4f}")

    # Save cache
    if cache_path:
        import pickle
        with open(cache_path, "wb") as f:
            pickle.dump({"X": X, "Y": Y, "df": df}, f)
        if verbose:
            print(f"Cached to: {cache_path}")

    return X, Y, df
