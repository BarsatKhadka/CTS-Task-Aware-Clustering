"""
Skew Simulation Experiment — Physics-Based CTS Simulation Features

Goal: Break skew MAE below 0.10 using:
  1. skew_hold ranks (direct CTS output, 99% inversely correlated with skew_setup rank)
  2. Simulated CTS clustering features (25 dims, change per run with cluster_dia/size)
  3. Recursive bisection tree path imbalance (physical skew proxy)
  4. Feature ablation to find the best combination

Key physics insight:
  - Grid-based CTS simulation: divide die into cluster_dia × cluster_dia cells
    This mimics how CTS groups FFs spatially. Features CHANGE with cluster_dia.
  - Recursive bisection on cluster centroids: simulates the hierarchical clock tree
    Path length variance = direct proxy for skew.

Quick test: --fold aes only (fast, ~5 min).
Full run:   --full  (all 4 LODO folds).
"""
import pickle
import time
import re
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

BASE = Path(__file__).parent
PLACEMENT_DIR = BASE / "dataset_with_def" / "placement_files"

LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=20,
                  min_child_samples=15, n_jobs=4, verbose=-1)
XGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, max_depth=4,
                  min_child_weight=15, subsample=0.8, colsample_bytree=0.8,
                  n_jobs=4, verbosity=0)


# ── 1. DEF Parser (FF positions in µm) ───────────────────────────────────────

_FF_PAT = re.compile(
    r"sky130_fd_sc_hd__(?:df|sdff|edf|sdfsrn|sdfbbn|dfbbn|dfrtp|dfxbp|dfxtp|dfbbp|df[a-z])",
    re.IGNORECASE,
)
_SKIP_CELLS = ("decap", "fill", "tap", "tie", "buf_", "PHY_")


def parse_def_ff_positions(def_path):
    """Return (ff_xy_um [n,2], die_w_um, die_h_um) or None."""
    ff_x, ff_y = [], []
    die_w = die_h = 1.0
    scale = 1000
    in_comp = False
    try:
        with open(def_path, 'r') as fh:
            for line in fh:
                ls = line.strip()
                if 'UNITS DISTANCE MICRONS' in ls:
                    m = re.search(r'MICRONS\s+(\d+)', ls)
                    if m:
                        scale = int(m.group(1))
                elif ls.startswith('DIEAREA'):
                    pts = re.findall(r'\(\s*(-?\d+)\s+(-?\d+)\s*\)', ls)
                    if len(pts) >= 2:
                        die_w = abs(int(pts[1][0]) - int(pts[0][0])) / scale
                        die_h = abs(int(pts[1][1]) - int(pts[0][1])) / scale
                elif ls.startswith('COMPONENTS') and not ls.startswith('END'):
                    in_comp = True
                elif ls.startswith('END COMPONENTS'):
                    in_comp = False
                elif in_comp and ls.startswith('- '):
                    parts = ls.split()
                    if len(parts) < 3 or any(k in parts[2] for k in _SKIP_CELLS):
                        continue
                    if _FF_PAT.match(parts[2]):
                        m = re.search(
                            r'(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)\s*\)', ls)
                        if m:
                            ff_x.append(int(m.group(1)) / scale)
                            ff_y.append(int(m.group(2)) / scale)
    except Exception as e:
        print(f"  DEF error {def_path}: {e}")
        return None
    if len(ff_x) < 2:
        return None
    return np.column_stack([ff_x, ff_y]), die_w, die_h


# ── 2. Recursive Bisection on Centroids ──────────────────────────────────────

def _bisect_paths(pts, depth=0, max_depth=14):
    """Recursive bisection → list of path lengths from root to each point."""
    if len(pts) <= 1 or depth >= max_depth:
        return [0.0] * len(pts)
    ranges = pts.max(0) - pts.min(0)
    dim = int(ranges.argmax())
    med = float(np.median(pts[:, dim]))
    lmask = pts[:, dim] <= med
    rmask = ~lmask
    if lmask.sum() == 0 or rmask.sum() == 0:
        return [0.0] * len(pts)
    parent_c = pts.mean(0)
    lc = pts[lmask].mean(0)
    rc = pts[rmask].mean(0)
    lw = float(np.linalg.norm(parent_c - lc))
    rw = float(np.linalg.norm(parent_c - rc))
    lp = _bisect_paths(pts[lmask], depth + 1, max_depth)
    rp = _bisect_paths(pts[rmask], depth + 1, max_depth)
    return [p + lw for p in lp] + [p + rw for p in rp]


# ── 3. Grid-based CTS Simulation (per run) ───────────────────────────────────

def simulate_cts_features(ff_xy_um, cluster_dia_um, cluster_size,
                           max_wire_um, buf_dist_um):
    """
    25 physics-grounded features that CHANGE per CTS config.
    Normalised by HPWL → design-invariant.
    """
    n_ff = len(ff_xy_um)
    hpwl = ((ff_xy_um[:, 0].max() - ff_xy_um[:, 0].min()) +
             (ff_xy_um[:, 1].max() - ff_xy_um[:, 1].min()))
    norm = max(hpwl, 1.0)
    cd = max(float(cluster_dia_um), 1.0)
    mw = max(float(max_wire_um), 1.0)
    bd = max(float(buf_dist_um), 1.0)
    cs = max(int(cluster_size), 1)

    # Grid clustering: cells of size cluster_dia × cluster_dia
    x_min, y_min = ff_xy_um[:, 0].min(), ff_xy_um[:, 1].min()
    cx = np.floor((ff_xy_um[:, 0] - x_min) / cd).astype(np.int32)
    cy = np.floor((ff_xy_um[:, 1] - y_min) / cd).astype(np.int32)
    cell_id = cx.astype(np.int64) * 1_000_000 + cy

    unique_ids, inverse = np.unique(cell_id, return_inverse=True)
    n_clusters = len(unique_ids)

    centroids = np.zeros((n_clusters, 2))
    within_radii = np.zeros(n_clusters)
    counts = np.zeros(n_clusters, dtype=np.int32)

    for i in range(n_clusters):
        mask = inverse == i
        pts = ff_xy_um[mask]
        counts[i] = len(pts)
        centroids[i] = pts.mean(0)
        if len(pts) > 1:
            within_radii[i] = np.linalg.norm(pts - centroids[i], axis=1).max()

    # Recursive bisection on cluster centroids
    if n_clusters > 1:
        paths = np.array(_bisect_paths(centroids))
        bisect_skew = paths.max() - paths.min()
        path_std = paths.std()
        path_max = paths.max()
        path_cv = path_std / max(paths.mean(), 1e-6)
        c_hpwl = ((centroids[:, 0].max() - centroids[:, 0].min()) +
                  (centroids[:, 1].max() - centroids[:, 1].min()))
        c_spread = np.linalg.norm(centroids - centroids.mean(0), axis=1).std()
    else:
        bisect_skew = path_std = path_max = path_cv = c_hpwl = c_spread = 0.0

    max_w = within_radii.max()
    std_w = within_radii.std()
    mean_w = within_radii.mean()
    max_cnt = counts.max()
    cnt_cv = counts.std() / max(counts.mean(), 1)

    # Expected n_clusters from cluster_size knob (independent estimate)
    exp_n_clusters = max(1, n_ff / cs)

    feats = np.array([
        n_clusters / max(n_ff, 1),            # 0  cluster density
        np.log1p(n_clusters),                  # 1  log cluster count
        max_w / norm,                          # 2  max within-radius / HPWL
        std_w / norm,                          # 3  within-radius variance
        mean_w / norm,                         # 4  mean within-radius
        max_cnt / max(n_ff, 1),                # 5  largest cluster fraction
        cnt_cv,                                # 6  cluster size imbalance
        max_w / max(cd / 2, 1),               # 7  worst cluster utilization
        bisect_skew / norm,                    # 8  simulated skew / HPWL  ← KEY
        path_std / norm,                       # 9  path std / HPWL
        path_max / norm,                       # 10 max path / HPWL
        path_cv,                               # 11 path CV
        c_hpwl / norm,                         # 12 centroid routing span
        c_spread / norm,                       # 13 centroid spread
        cd / bd,                               # 14 cluster_dia / buf_dist  ← KEY
        cd / mw,                               # 15 cluster_dia / max_wire
        cd / norm,                             # 16 cluster_dia / HPWL
        bd / norm,                             # 17 buf_dist / HPWL
        bisect_skew / max(c_hpwl, 1e-6),      # 18 skew relative to routing span
        np.log1p(hpwl / mw),                  # 19 tree depth estimate
        np.log1p(exp_n_clusters),              # 20 expected n_clusters from cs knob
        bisect_skew * cd / (norm * bd),        # 21 skew × knob interaction
        max_w * np.log1p(n_clusters) / norm,  # 22 within × tree_depth
        n_clusters / max(exp_n_clusters, 1),  # 23 actual/expected cluster ratio
        path_std * cd / (norm * bd),          # 24 path imbalance × knob ratio
    ], dtype=np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=5.0, neginf=-5.0)


# ── 4. Build simulation feature cache ────────────────────────────────────────

def build_sim_features(df, ff_cache):
    """
    Build [n_rows, 25] simulation feature matrix.
    ff_cache: {placement_id: (ff_xy_um, die_w, die_h)} or None for failed DEFs.
    """
    n = len(df)
    out = np.zeros((n, 25), dtype=np.float32)
    for i, (_, row) in enumerate(df.iterrows()):
        pid = row['placement_id']
        if pid not in ff_cache or ff_cache[pid] is None:
            continue
        ff_xy, _, _ = ff_cache[pid]
        out[i] = simulate_cts_features(
            ff_xy,
            float(row['cts_cluster_dia']),
            int(row['cts_cluster_size']),
            float(row['cts_max_wire']),
            float(row['cts_buf_dist']),
        )
    return out


# ── 5. skew_hold rank features ───────────────────────────────────────────────

def build_skew_hold_features(df, pids):
    """2 features: rank_skew_hold + centered_skew_hold within placement."""
    n = len(df)
    rank_sh = np.zeros(n, np.float32)
    cent_sh = np.zeros(n, np.float32)
    sh_vals = df['skew_hold'].values.astype(np.float64)
    for pid in np.unique(pids):
        mask = pids == pid
        rows = np.where(mask)[0]
        vals = sh_vals[rows]
        rank_sh[rows] = np.argsort(np.argsort(vals)).astype(float) / max(len(vals) - 1, 1)
        cent_sh[rows] = vals - vals.mean()
    # Normalize centered by global std to scale-invariant
    gs = cent_sh.std()
    if gs > 1e-8:
        cent_sh /= gs
    return np.column_stack([rank_sh, cent_sh])


# ── 6. X52 feature builder (from train_best_model.py) ────────────────────────

def build_X52(df_cache, df_csv, X_cache):
    """Reproduce X52 from train_best_model.build_features()."""
    designs = df_cache['design_name'].values
    pids    = df_cache['placement_id'].values
    n = len(designs)

    knob_cols  = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
    clock_cols = ['clock_buffers', 'clock_inverters', 'timing_repair_buffers']
    place_cols = ['core_util', 'density', 'aspect_ratio']
    output_cols= ['setup_vio_count', 'hold_vio_count', 'setup_slack', 'hold_slack',
                  'setup_tns', 'hold_tns', 'utilization']

    # Use df_cache directly (it has the full CSV columns since cache_v2_fixed has 38-col df)
    Xraw   = df_cache[knob_cols].values.astype(np.float32)
    Xplace = df_cache[place_cols].values.astype(np.float32)
    Xclock = df_cache[clock_cols].values.astype(np.float32)
    Xouts  = df_cache[output_cols].values.astype(np.float32)
    Xkz    = X_cache[:, 72:76]

    raw_max   = Xraw.max(0) + 1e-6
    clock_max = Xclock.max(0) + 1e-6

    Xrank        = np.zeros((n, 4), np.float32)
    Xcentered    = np.zeros((n, 4), np.float32)
    Xknob_range  = np.zeros((n, 4), np.float32)
    Xknob_mean   = np.zeros((n, 4), np.float32)
    Xclock_rank  = np.zeros_like(Xclock)
    Xclock_cent  = np.zeros_like(Xclock)
    Xouts_rank   = np.zeros_like(Xouts)
    Xouts_cent   = np.zeros_like(Xouts)

    for pid in np.unique(pids):
        mask = pids == pid
        rows = np.where(mask)[0]
        for ki in range(4):
            z_vals = Xkz[rows, ki]
            Xrank[rows, ki]       = np.argsort(np.argsort(z_vals)).astype(float) / max(len(z_vals) - 1, 1)
            Xcentered[rows, ki]   = z_vals - z_vals.mean()
            Xknob_range[rows, ki] = Xraw[rows, ki].std()
            Xknob_mean[rows, ki]  = Xraw[rows, ki].mean()
        for ki in range(3):
            vals = Xclock[rows, ki]
            Xclock_rank[rows, ki] = np.argsort(np.argsort(vals)).astype(float) / max(len(vals) - 1, 1)
            Xclock_cent[rows, ki] = vals - vals.mean()
        for ki in range(7):
            vals = Xouts[rows, ki]
            Xouts_rank[rows, ki] = np.argsort(np.argsort(vals)).astype(float) / max(len(vals) - 1, 1)
            Xouts_cent[rows, ki] = vals - vals.mean()

    Xplace_n = Xplace.copy(); Xplace_n[:, 0] /= 100.0
    Xkp = np.column_stack([
        Xraw[:, 3] * Xplace[:, 0] / 100,
        Xraw[:, 0] * Xplace[:, 1],
        Xraw[:, 3] / np.maximum(Xplace[:, 1], 0.01),
        Xraw[:, 3] * Xplace[:, 2],
        Xrank[:, 3] * (Xplace[:, 0] / 100),
        Xrank[:, 2] * (Xplace[:, 0] / 100),
    ])

    X38  = np.hstack([Xkz, Xrank, Xcentered, Xplace_n, Xkp,
                      Xknob_range / raw_max, Xknob_mean / raw_max,
                      Xclock_rank, Xclock_cent, Xclock / clock_max])
    X52  = np.hstack([X38, Xouts_rank, Xouts_cent])
    return X52, pids, designs


def build_rank_targets(Y, pids):
    n = len(pids)
    Yr = np.zeros((n, 3), np.float32)
    for pid in np.unique(pids):
        mask = pids == pid
        rows = np.where(mask)[0]
        for j in range(3):
            v = Y[mask, j]
            Yr[rows, j] = np.argsort(np.argsort(v)).astype(float) / max(len(v) - 1, 1)
    return Yr


# ── 7. LODO evaluation ────────────────────────────────────────────────────────

def run_lodo(X, Y_rank, designs, label="", folds=None):
    all_designs = sorted(np.unique(designs))
    if folds:
        test_designs = [d for d in all_designs if d in folds]
    else:
        test_designs = all_designs

    results = []
    for held in test_designs:
        tr = designs != held; te = ~tr
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
        fold = []
        for j in range(3):
            m = (xgb.XGBRegressor(**XGB_PARAMS) if j == 0
                 else lgb.LGBMRegressor(**LGB_PARAMS))
            m.fit(Xtr, Y_rank[tr, j])
            fold.append(mean_absolute_error(Y_rank[te, j], m.predict(Xte)))
        results.append((held, fold))

    print(f"\n{'─'*60}")
    print(f"  {label}  [dim={X.shape[1]}]")
    for held, fold in results:
        stat = ' '.join(['PASS✓' if fold[j] < 0.10 else 'FAIL ' for j in range(3)])
        print(f"  {held:12s}: sk={fold[0]:.4f} pw={fold[1]:.4f} wl={fold[2]:.4f}  {stat}")
    if len(results) > 1:
        arr = np.array([f for _, f in results])
        means = arr.mean(0)
        status = ' '.join(['PASS✓' if means[j] < 0.10 else 'FAIL ' for j in range(3)])
        print(f"  {'MEAN':12s}: sk={means[0]:.4f} pw={means[1]:.4f} wl={means[2]:.4f}  {status}")
    return results


# ── 8. Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', action='store_true', help='Run all 4 LODO folds')
    parser.add_argument('--fold', default='aes', help='Single fold to test (default: aes)')
    parser.add_argument('--no-sim', action='store_true', help='Skip simulation features')
    args = parser.parse_args()

    folds = None if args.full else [args.fold]

    # ── Load cache ────────────────────────────────────────────────────────────
    print("Loading cache...")
    with open('cache_v2_fixed.pkl', 'rb') as f:
        d = pickle.load(f)
    X_cache, Y_cache, df_cache = d['X'], d['Y'], d['df']
    df_csv = df_cache  # cache already has all CSV columns

    print(f"Cache: X={X_cache.shape}  designs: {dict(zip(*np.unique(df_cache['design_name'], return_counts=True)))}")

    # ── Build X52 ─────────────────────────────────────────────────────────────
    print("\nBuilding X52 features...")
    X52, pids, designs = build_X52(df_cache, df_csv, X_cache)
    Y_rank = build_rank_targets(Y_cache, pids)
    print(f"X52 shape: {X52.shape}")

    # ── Baseline ──────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("ABLATION STUDY — QUICK TEST")
    print("="*60)
    run_lodo(X52, Y_rank, designs, label="BASELINE X52", folds=folds)

    # ── skew_hold features ────────────────────────────────────────────────────
    print("\nAdding skew_hold rank/centered features...")
    X_sh = build_skew_hold_features(df_cache, pids)
    X52_sh = np.hstack([X52, X_sh])
    run_lodo(X52_sh, Y_rank, designs, label="X52 + skew_hold(2)", folds=folds)

    if args.no_sim:
        print("\n[--no-sim] Skipping simulation features.")
        return

    # ── Parse DEF files ───────────────────────────────────────────────────────
    print("\nParsing DEF files for FF positions...")
    t0 = time.time()
    ff_cache = {}
    unique_pids = df_cache['placement_id'].unique()
    design_map  = df_cache.drop_duplicates('placement_id').set_index('placement_id')['design_name']

    failed = 0
    for i, pid in enumerate(unique_pids):
        design = design_map[pid]
        def_path = PLACEMENT_DIR / pid / f"{design}.def"
        if def_path.exists():
            result = parse_def_ff_positions(def_path)
            ff_cache[pid] = result  # None if failed
            if result is None:
                failed += 1
        else:
            ff_cache[pid] = None
            failed += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(unique_pids)}  ({time.time()-t0:.0f}s)")

    n_ok = sum(v is not None for v in ff_cache.values())
    print(f"DEF parsed: {n_ok}/{len(unique_pids)} ok, {failed} failed  ({time.time()-t0:.1f}s)")

    # ── Build simulation features ─────────────────────────────────────────────
    print("\nBuilding simulated CTS features (25 dims per run)...")
    t1 = time.time()
    X_sim = build_sim_features(df_cache, ff_cache)
    print(f"Simulation features: {X_sim.shape}  ({time.time()-t1:.1f}s)")
    print(f"  Feature stats: mean={X_sim.mean(0)[:5].round(3)}  std={X_sim.std(0)[:5].round(3)}")

    # ── Ablation: add simulation features ─────────────────────────────────────
    X52_sim  = np.hstack([X52, X_sim])
    run_lodo(X52_sim, Y_rank, designs, label="X52 + sim(25)", folds=folds)

    X52_sh_sim = np.hstack([X52, X_sh, X_sim])
    run_lodo(X52_sh_sim, Y_rank, designs, label="X52 + skew_hold(2) + sim(25)", folds=folds)

    # ── Feature importance for skew ───────────────────────────────────────────
    print("\n--- Feature importance for skew (X52+sh+sim) ---")
    sc = StandardScaler()
    test_design = folds[0] if folds else 'aes'
    tr = designs != test_design
    Xtr = sc.fit_transform(X52_sh_sim[tr])
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(Xtr, Y_rank[tr, 0])
    imp = m.feature_importances_

    names = (
        [f"X52_{i}" for i in range(52)] +
        ["sh_rank", "sh_cent"] +
        [f"sim_{i}" for i in range(25)]
    )
    sim_names = [
        "n_clusters/n_ff", "log(n_clusters)", "max_within_r/hpwl",
        "std_within_r/hpwl", "mean_within_r/hpwl", "max_cnt/n_ff",
        "cnt_cv", "worst_cluster_fill", "bisect_skew/hpwl",   # 8 ← KEY
        "path_std/hpwl", "path_max/hpwl", "path_cv",
        "centroid_hpwl/hpwl", "centroid_spread/hpwl",
        "cd/bd", "cd/mw", "cd/hpwl", "bd/hpwl",              # 14-17
        "skew/routing_span", "tree_depth",
        "log_exp_nclusters", "skew×knob", "within×depth",
        "actual/expected_clusters", "imbalance×knob",
    ]
    names = [f"X52_{i}" for i in range(52)] + ["sh_rank", "sh_cent"] + [f"sim_{n}" for n in sim_names]

    top = np.argsort(imp)[::-1][:20]
    for rank_i, fi in enumerate(top):
        nm = names[fi] if fi < len(names) else f"feat_{fi}"
        print(f"  [{rank_i+1:2d}] {nm:35s}: {imp[fi]:.4f}")

    # ── Targeted ablation: sim sub-groups ────────────────────────────────────
    print("\n--- Sim feature sub-group ablation ---")
    # Group 1: cluster geometry (dims 0-7)
    X_cg = np.hstack([X52, X_sh, X_sim[:, :8]])
    run_lodo(X_cg, Y_rank, designs, label="X52+sh + cluster_geom(8)", folds=folds)

    # Group 2: tree path features (dims 8-13)
    X_tp = np.hstack([X52, X_sh, X_sim[:, 8:14]])
    run_lodo(X_tp, Y_rank, designs, label="X52+sh + tree_paths(6)", folds=folds)

    # Group 3: key physics ratios (dims 14-18)
    X_pr = np.hstack([X52, X_sh, X_sim[:, 14:19]])
    run_lodo(X_pr, Y_rank, designs, label="X52+sh + phys_ratios(5)", folds=folds)

    # Group 4: KEY skew feature only (bisect_skew/hpwl = dim 8)
    X_sk = np.hstack([X52, X_sh, X_sim[:, 8:9]])
    run_lodo(X_sk, Y_rank, designs, label="X52+sh + bisect_skew_only(1)", folds=folds)

    # Group 5: best group combinations
    X_best = np.hstack([X52, X_sh, X_sim[:, [8, 9, 11, 14, 15, 16, 18, 22, 24]]])
    run_lodo(X_best, Y_rank, designs, label="X52+sh + top_sim_feats(9)", folds=folds)

    print("\nDone.")


if __name__ == "__main__":
    main()
