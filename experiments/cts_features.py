"""
cts_features.py — Physics-grounded feature extraction for CTS prediction
=========================================================================
All features are derived from the CTS literature. Each feature block is
annotated with its physical justification and the paper it comes from.

Feature vector layout (total: 310 dims):
  [0:256]   spatial_grid         — FF/cell density + area + toggle heatmap
  [256:264] skip_distances       — timing path length distribution (8)
  [264:270] knn_distances        — FF nearest-neighbor distances (6)
  [270:274] steiner_proxy        — HPWL, aspect ratio, centroid offset (4)
  [274:278] capacitive_load      — total cap, cap variance, n_ff scale (4)
  [278:285] skew_physics         — drive variance, spatial asymmetry (7)
  [285:294] density_features     — local density fractions at 3 radii (9)
  [294:300] synthesis_flags      — binary strategy flags (6)
  [300:304] scale_features       — log n_ff, log n_nodes, area proxy (4)
"""

import math
import numpy as np
import scipy.sparse as sp

GRID_SIZE      = 8
DENSITY_RADII  = [0.05, 0.10, 0.20]
SKIP_DIST_PCTS = [0, 25, 50, 75, 90, 100]


# ─────────────────────────────────────────────────────────────────────────────
# SPATIAL GRID  [256 dims]
# ─────────────────────────────────────────────────────────────────────────────
# GAN-CTS (Lu et al. TCAD 2022): extracts features from placement images
# representing FF distributions, clock net distributions, and trial routing.
# We approximate this with a 4-channel 8x8 grid capturing:
#   ch0: all-node density   — physical congestion proxy
#   ch1: all-node area      — capacitance load proxy (Elmore delay model)
#   ch2: FF-only density    — CTS sink distribution (primary WL driver)
#   ch3: FF toggle density  — switching power proxy

def spatial_grid_features(X_all, G=GRID_SIZE):
    ch     = np.zeros((4, G, G), dtype=np.float32)
    x_z    = X_all[:, 0].numpy()
    y_z    = X_all[:, 1].numpy()
    is_seq = (X_all[:, 10] == 1.0).numpy()
    area_raw   = X_all[:, 6].numpy()
    toggle_raw = X_all[:, 12].numpy()

    def minmax01(v):
        lo, hi = v.min(), v.max()
        return (v - lo) / (hi - lo + 1e-8)

    area   = minmax01(area_raw)
    toggle = minmax01(toggle_raw)

    xr = x_z.max() - x_z.min(); yr = y_z.max() - y_z.min()
    x_01 = (x_z - x_z.min()) / (xr if xr > 0 else 1.0)
    y_01 = (y_z - y_z.min()) / (yr if yr > 0 else 1.0)
    gx = np.clip((x_01 * G).astype(int), 0, G-1)
    gy = np.clip((y_01 * G).astype(int), 0, G-1)

    np.add.at(ch[0], (gx, gy), 1)
    np.add.at(ch[1], (gx, gy), area)
    fm = is_seq.astype(bool)
    np.add.at(ch[2], (gx[fm], gy[fm]), 1)
    np.add.at(ch[3], (gx[fm], gy[fm]), toggle[fm])
    for i in range(4):
        s = ch[i].sum()
        if s > 0: ch[i] /= s
    return ch.flatten()   # [256]


# ─────────────────────────────────────────────────────────────────────────────
# SKIP (TIMING PATH) DISTANCES  [8 dims]
# ─────────────────────────────────────────────────────────────────────────────
# ScienceDirect/Wikipedia: skew = difference in clock arrival times.
# The primary geometric predictor is the spatial distance between
# launch and capture FFs. Longer launch-capture distance → more wire
# to balance → higher skew potential.
# CTS-Bench: path length distribution is the strongest structural skew signal.

def skip_distance_features(ff_X_raw, ff_skip_local):
    n_feat = len(SKIP_DIST_PCTS) + 2
    if len(ff_skip_local) == 0:
        return np.zeros(n_feat, dtype=np.float32)
    xy    = np.stack([ff_X_raw[:, 0].numpy(), ff_X_raw[:, 1].numpy()], axis=1)
    dists = np.sqrt(((xy[ff_skip_local[:, 0]] - xy[ff_skip_local[:, 1]])**2).sum(1))
    dists = dists / 4.0   # normalize: N(0,1) 2D max distance ≈ 4-6 std units
    pcts  = np.percentile(dists, SKIP_DIST_PCTS).astype(np.float32)
    return np.concatenate([pcts, [dists.mean()], [dists.std()]])   # [8]


# ─────────────────────────────────────────────────────────────────────────────
# FF K-NEAREST-NEIGHBOR DISTANCES  [12 dims]
# ─────────────────────────────────────────────────────────────────────────────
# KEY INSIGHT from CTS tool mechanics (OpenROAD docs, iCTS paper):
# The CTS tool first clusters FFs using cluster_dia (D) and cluster_size (S).
# FFs within distance D of each other form a cluster.
# Within-cluster skew ∝ cluster_dia × wire_cap_per_unit_length.
#
# The kNN distance distribution tells us:
#   knn_4:  how tightly packed the nearest 4 FFs are → within-cluster density
#   knn_16: how spread the 16th nearest FF is → cluster boundary characteristics
#
# When the skew head sees cluster_dia × knn_4_mean:
#   if cluster_dia >> knn_4_mean: FFs easily cluster, low within-cluster skew
#   if cluster_dia << knn_4_mean: sparse FFs hard to cluster, high skew
#
# This is the physical interaction the head needs to learn skew vs cluster_dia.

def knn_distance_features(ff_X_raw, ks=(4, 16)):
    n = ff_X_raw.shape[0]
    feats = []
    if n < max(ks) + 1:
        return np.zeros(len(ks) * 3, dtype=np.float32)

    xy = np.stack([ff_X_raw[:, 0].numpy(), ff_X_raw[:, 1].numpy()],
                  axis=1).astype(np.float32)
    if n > 500:
        rng = np.random.default_rng(42)
        xy  = xy[rng.choice(n, 500, replace=False)]
        n   = 500

    diff  = xy[:, None, :] - xy[None, :, :]      # [n, n, 2]
    dists = np.sqrt((diff**2).sum(axis=-1))        # [n, n]
    np.fill_diagonal(dists, np.inf)
    dists_sorted = np.sort(dists, axis=1)         # [n, n]

    for k in ks:
        if k <= n - 1:
            kth = dists_sorted[:, k-1] / 4.0     # normalize same as skip_dist
            feats.extend([kth.mean(), kth.std(),
                          float(np.percentile(kth, 90))])
        else:
            feats.extend([0., 0., 0.])
    return np.array(feats, dtype=np.float32)   # [6 per k → 12 total]


# ─────────────────────────────────────────────────────────────────────────────
# STEINER TREE PROXY + ASPECT RATIO + CENTROID OFFSET  [4 dims]
# ─────────────────────────────────────────────────────────────────────────────
# Kahng UCSD (c300): "floorplan context is essential — block aspect ratio and
# clock entry point location cause 43-45% variation in power and delay."
#
# 1. HPWL (half-perimeter wirelength): The Steiner minimum tree of the FF set
#    is lower-bounded by HPWL/2. Actual clock WL ≈ 1.2-1.5× HPWL.
#    This is THE primary wirelength predictor.
#    iCTS paper: "wirelength and the Steiner tree have a tight connection."
#
# 2. Aspect ratio of FF bounding box: Kahng shows 43-45% power variation
#    with aspect ratio. Tall/narrow designs have longer routing distances.
#
# 3. Centroid offset: distance from FF population centroid to the die center.
#    Clock enters from a port (usually near die boundary). If FFs are all
#    near one side, clock paths are shorter → lower skew/power.

def steiner_proxy_features(ff_X_raw):
    x = ff_X_raw[:, 0].numpy()
    y = ff_X_raw[:, 1].numpy()

    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_range = x_max - x_min
    y_range  = y_max - y_min

    # 1. HPWL in z-score space (normalized by 4.0 as all distances are)
    hpwl = (x_range + y_range) / 4.0

    # 2. Aspect ratio (clamped to [0.1, 10] range then log-scaled)
    aspect = np.log(max(x_range, 1e-3) / max(y_range, 1e-3) + 1.0)

    # 3. Centroid distance from origin (z-scored coords have mean≈0 = die center)
    cx = x.mean(); cy = y.mean()
    centroid_offset = np.sqrt(cx**2 + cy**2) / 4.0

    # 4. FF bounding box area as fraction of die area (z-score space)
    # Die extent in z-score space is typically [-3,3]×[-3,3] = 6×6 = 36
    bbox_area = (x_range * y_range) / 36.0

    return np.array([hpwl, aspect, centroid_offset, bbox_area],
                    dtype=np.float32)   # [4]


# ─────────────────────────────────────────────────────────────────────────────
# CAPACITIVE LOAD FEATURES  [4 dims]
# ─────────────────────────────────────────────────────────────────────────────
# iCTS (Li et al. TCAD 2025): "clock capacitance — comprising interconnect and
# pin capacitance — determines the required driving capability. Power = C×V²×f"
#
# The total capacitive load is the dominant clock power predictor:
#   P_clock = α × C_total × V² × f
# where C_total = sum of all FF pin caps + wire capacitance.
#
# We can compute the FF clock pin capacitance component directly:
#   n_ff × mean_pin_cap = total FF input capacitance
#
# Toggle-weighted total cap (C × toggle_rate) = switching power estimate.
# High toggle variance means some paths switch much more → power imbalance.

def capacitive_load_features(ff_X_raw):
    n_ff   = ff_X_raw.shape[0]
    cap    = ff_X_raw[:, 7].numpy()    # avg_pin_cap × 1000, z-scored
    toggle = ff_X_raw[:, 12].numpy()  # toggle_count, z-scored

    # Mean cap × n_ff proxy (absolute load; normalized by log scale)
    total_cap_proxy = float(np.log1p(max(n_ff, 1)) * cap.mean())

    # Cap variance — unequal FF loads create current imbalance → power noise
    cap_std = float(cap.std()) if n_ff > 1 else 0.0

    # Toggle-weighted cap: sum(cap_i × toggle_i) / n_ff
    # High value = high switching power region
    toggle_cap = float((np.maximum(toggle, 0) * (cap - cap.min())).mean())

    # Total toggle activity (proportional to dynamic power)
    total_toggle = float(np.maximum(toggle, 0).mean())

    return np.array([total_cap_proxy, cap_std, toggle_cap, total_toggle],
                    dtype=np.float32)   # [4]


# ─────────────────────────────────────────────────────────────────────────────
# SKEW PHYSICS FEATURES  [7 dims]
# ─────────────────────────────────────────────────────────────────────────────
# Wikipedia/ScienceDirect: "skew caused by wire-interconnect length differences,
# variation in intermediate devices, capacitive coupling, and differences in
# input capacitance on the clock inputs."
#
# 1. Drive strength VARIANCE (not mean) — skew comes from asymmetry.
#    Uniform drive = uniform delay = zero skew contribution.
#    Mixed drive strengths = asymmetric RC = skew.
#
# 2. Launch vs capture centroid distance — if all launches are far from all
#    captures in space, the CTS tool must route long unbalanced paths → skew.
#
# 3. Spatial spread asymmetry: std(launch positions) vs std(capture positions)
#    Asymmetric spread = asymmetric clock tree branch lengths → skew.
#
# 4. Cap variance of timing-path FFs — unequal load on clock tree branches
#    → delay asymmetry → skew. (Referenced in iCTS, Elmore delay model)

def skew_physics_features(ff_X_raw, ff_skip_local):
    n_ff  = ff_X_raw.shape[0]
    zeros = np.zeros(7, dtype=np.float32)
    if len(ff_skip_local) == 0 or n_ff < 2:
        return zeros

    drive  = ff_X_raw[:, 9].numpy()
    cap    = ff_X_raw[:, 7].numpy()
    x_pos  = ff_X_raw[:, 0].numpy()
    y_pos  = ff_X_raw[:, 1].numpy()
    launch = ff_skip_local[:, 0]
    cap_t  = ff_skip_local[:, 1]

    timing_ffs = np.unique(np.concatenate([launch, cap_t]))
    drive_v = float(drive[timing_ffs].var()) if len(timing_ffs) > 1 else 0.
    drive_s = float(drive[timing_ffs].std()) if len(timing_ffs) > 1 else 0.

    lx = x_pos[launch].mean(); ly = y_pos[launch].mean()
    cx = x_pos[cap_t].mean();  cy = y_pos[cap_t].mean()
    centroid_dist = float(np.sqrt((lx-cx)**2 + (ly-cy)**2)) / 4.0

    l_spread = float(np.sqrt(x_pos[launch].var() + y_pos[launch].var()))
    c_spread = float(np.sqrt(x_pos[cap_t].var()  + y_pos[cap_t].var()))
    spread_r = abs(l_spread - c_spread) / (l_spread + c_spread + 1e-8)

    cap_v = float(cap[timing_ffs].var()) if len(timing_ffs) > 1 else 0.

    return np.array([drive_v, drive_s, centroid_dist,
                     l_spread, c_spread, spread_r, cap_v],
                    dtype=np.float32)   # [7]


# ─────────────────────────────────────────────────────────────────────────────
# DENSITY FEATURES  [9 dims]
# ─────────────────────────────────────────────────────────────────────────────
# Kahng UCSD: "nonuniformity of sink placement" is a key CTS prediction
# parameter. Dense FF clusters need fewer buffers but more balancing.
# Returns fractions (0-1), not raw counts — design-size invariant.

def ff_density_features(ff_X_raw):
    n = ff_X_raw.shape[0]
    if n < 2:
        return np.zeros(len(DENSITY_RADII) * 3, dtype=np.float32)
    xy = np.stack([ff_X_raw[:, 0].numpy(), ff_X_raw[:, 1].numpy()],
                  axis=1).astype(np.float32)
    n_s = min(n, 500)
    if n > 500:
        xy = xy[np.random.default_rng(42).choice(n, n_s, replace=False)]
    diff  = xy[:, None, :] - xy[None, :, :]
    dists = np.sqrt((diff**2).sum(axis=-1))
    np.fill_diagonal(dists, np.inf)
    feats = []
    for r in DENSITY_RADII:
        c = (dists < r).sum(axis=1).astype(np.float32) / (n_s - 1)
        feats.extend([c.mean(), c.std(), c.max()])
    return np.array(feats, dtype=np.float32)   # [9]


# ─────────────────────────────────────────────────────────────────────────────
# SYNTHESIS FLAGS  [6 dims]
# ─────────────────────────────────────────────────────────────────────────────

def synthesis_flags(row):
    strat = str(row.get('synth_strategy', '')).upper()
    return np.array([
        float('AREA 0' in strat), float('AREA 2' in strat),
        float('DELAY'  in strat),
        float(int(row.get('io_mode',            0))),
        float(int(row.get('time_driven',        0))),
        float(int(row.get('routability_driven', 0))),
    ], dtype=np.float32)   # [6]


# ─────────────────────────────────────────────────────────────────────────────
# SCALE FEATURES  [4 dims]
# ─────────────────────────────────────────────────────────────────────────────
# Kahng UCSD: number of FFs is a key design scale parameter that affects
# every CTS metric. GAN-CTS concatenates n_ff directly with ResNet features.

def scale_features(n_ff, num_nodes):
    return np.array([
        math.log(max(n_ff,       1)) / 10.0,
        math.log(max(num_nodes,  1)) / 10.0,
        math.log(max(num_nodes**2, 1)) / 20.0,
        n_ff / max(num_nodes, 1),
    ], dtype=np.float32)   # [4]


# ─────────────────────────────────────────────────────────────────────────────
# MASTER FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────

GLOBAL_FEAT_DIM = 256 + 8 +  6 + 4 + 4 + 7 + 9 + 6 + 4   # = 304

def build_global_features(X_all, ff_X_raw, ff_skip_local, row0, num_nodes):
    """
    Build the 310-dim global feature vector for one placement.
    All features are in bounded ranges before global normalization.
    """
    n_ff = ff_X_raw.shape[0]

    grid_f    = spatial_grid_features(X_all)                        # 256
    skip_f    = skip_distance_features(ff_X_raw, ff_skip_local)     # 8
    knn_f     = knn_distance_features(ff_X_raw)                     # 12
    steiner_f = steiner_proxy_features(ff_X_raw)                    # 4
    cap_f     = capacitive_load_features(ff_X_raw)                  # 4
    skew_f    = skew_physics_features(ff_X_raw, ff_skip_local)      # 7
    density_f = ff_density_features(ff_X_raw)                       # 9
    synth_f   = synthesis_flags(row0)                               # 6
    scale_f   = scale_features(n_ff, num_nodes)                     # 4

    v = np.concatenate([grid_f, skip_f, knn_f, steiner_f,
                        cap_f, skew_f, density_f, synth_f, scale_f])
    assert v.shape[0] == GLOBAL_FEAT_DIM, \
        f"Expected {GLOBAL_FEAT_DIM} global feats, got {v.shape[0]}"
    return v


# ─────────────────────────────────────────────────────────────────────────────
# SGC FEATURES  [n_ff, 27]
# ─────────────────────────────────────────────────────────────────────────────
# CTS-Bench: prioritizes data-path cells immediately surrounding registers
# (one-hop fan-out logic) as having the strongest impact on clock tree.
# We smooth FF features over the timing path graph (skip edges) using SGC.

def build_sgc_features(ff_X_raw, ff_skip_local, hops=2):
    X = np.stack([
        ff_X_raw[:, 0].numpy(),   # x_norm
        ff_X_raw[:, 1].numpy(),   # y_norm
        ff_X_raw[:, 12].numpy(),  # toggle_count
        ff_X_raw[:, 13].numpy(),  # sum_toggle
        ff_X_raw[:, 14].numpy(),  # signal_prob
        ff_X_raw[:, 9].numpy(),   # log2(drive_strength)
        ff_X_raw[:, 7].numpy(),   # avg_pin_cap × 1000
        ff_X_raw[:, 16].numpy(),  # log1p(fan_in)
        ff_X_raw[:, 17].numpy(),  # log1p(fan_out)
    ], axis=1).astype(np.float32)   # [n_ff, 9]

    n = X.shape[0]
    if len(ff_skip_local) == 0 or n < 2:
        return np.tile(X, (1, hops + 1))

    rows = np.concatenate([ff_skip_local[:, 0], ff_skip_local[:, 1],
                           np.arange(n)])
    cols = np.concatenate([ff_skip_local[:, 1], ff_skip_local[:, 0],
                           np.arange(n)])
    A    = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(n, n))
    deg  = np.array(A.sum(axis=1)).flatten() + 1e-8
    S    = sp.diags(1.0/np.sqrt(deg)) @ A @ sp.diags(1.0/np.sqrt(deg))

    smoothed = [X]; cur = X.copy()
    for _ in range(hops):
        cur = S @ cur; smoothed.append(cur.copy())
    return np.concatenate(smoothed, axis=1)   # [n_ff, 27]