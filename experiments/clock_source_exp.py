"""
clock_source_exp.py - Clock source location as a physics-grounded feature

Key insight: The CTS tool routes FROM the clock input pin TO each FF.
The distance from the clock source to each FF cluster determines routing cost.
This is a NOVEL feature not previously used in any experiment.

Features:
  - clock_pin (x, y) position from DEF PINS section
  - Per-FF distance to clock pin
  - Radial distribution of FFs around clock pin
  - Asymmetry: fraction of FFs above/below clock pin
  - Interaction with cluster_dia:
    clock_reach = cluster_dia / mean_dist_to_clock_pin
    (how many buffer stages needed to reach all FFs)

Also tests: multi-level tree depth estimate based on clock pin → FF distances.
"""

import pickle, time, warnings, re
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
import os

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Clock Source Location Features (NEW: from DEF PINS section)")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)
with open(f'{BASE}/absolute_v5_def_cache.pkl', 'rb') as f:
    def_cache = pickle.load(f)
with open(f'{BASE}/ff_positions_cache.pkl', 'rb') as f:
    ff_pos = pickle.load(f)

pids = df['placement_id'].values; designs = df['design_name'].values; n = len(df)
y_pw, y_wl = Y_cache[:, 1], Y_cache[:, 2]

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

# ── Build X29T ─────────────────────────────────────────────────────────────
knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
Xraw = df[knob_cols].values.astype(np.float32)
Xkz = X_cache[:, 72:76]
raw_max = Xraw.max(0) + 1e-6
Xrank = np.zeros((n, 4), np.float32); Xcent = np.zeros_like(Xrank)
Xrng = np.zeros_like(Xrank); Xmn = np.zeros_like(Xrank)
for pid in np.unique(pids):
    m = pids == pid; idx = np.where(m)[0]
    for j in range(4):
        v = Xraw[idx, j]; Xrank[idx, j] = rank_within(v)
        Xcent[idx, j] = (v - v.mean()) / raw_max[j]
        Xrng[idx, j] = v.std() / raw_max[j]; Xmn[idx, j] = v.mean() / raw_max[j]
Xplc = df[['core_util', 'density', 'aspect_ratio']].values.astype(np.float32)
Xplc_n = Xplc / (Xplc.max(0) + 1e-9)
cd = Xraw[:, 3]; cs = Xraw[:, 2]; mw = Xraw[:, 0]; bd = Xraw[:, 1]
util = Xplc[:, 0]/100; dens = Xplc[:, 1]; asp = Xplc[:, 2]
Xinter = np.column_stack([cd*util, mw*dens, cd/(dens+0.01), cd*asp,
                           Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrng, Xmn])
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp.get(pid)
    if v is not None: X_tight[i,:20] = np.array(v, np.float32)[:20]
tp_std = X_tight.std(0); tp_std[tp_std<1e-9]=1.0
X29T = np.hstack([X29, X_tight/tp_std])

# ── Extract clock pin locations from DEF ───────────────────────────────────
print(f"\n{T()} Extracting clock pin positions from DEF files...")

def get_clock_pin(def_path):
    """Parse DEF to find clock input pin position."""
    try:
        with open(def_path) as f:
            content = f.read()

        # Look for DIRECTION INPUT + USE CLOCK pins
        # Pattern: - pin_name + NET pin_name + DIRECTION INPUT + USE CLOCK
        # Then PLACED ( x y )
        # First try exact clock pin
        # Try multiple patterns for clock pin
        patterns = [
            r'USE CLOCK.*?PLACED \( (\d+) (\d+) \)',
            r'- clk \+ NET clk.*?PLACED \( (\d+) (\d+) \)',
            r'- CLK \+ NET CLK.*?PLACED \( (\d+) (\d+) \)',
            r'- clock.*?PLACED \( (\d+) (\d+) \)',
        ]
        for pat in patterns:
            m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
            if m:
                x, y = float(m.group(1)), float(m.group(2))
                return x, y

        # Try DIEAREA to get die dimensions
        m = re.search(r'DIEAREA \( (\d+) (\d+) \) \( (\d+) (\d+) \)', content)
        if m:
            dw, dh = float(m.group(3)), float(m.group(4))
            # Default: clock pin at center-bottom
            return dw/2, 0.0

        return None
    except Exception:
        return None

def get_die_info(def_path):
    """Get die dimensions from DEF."""
    try:
        with open(def_path) as f:
            content = f.read()
        m = re.search(r'DIEAREA \( (\d+) (\d+) \) \( (\d+) (\d+) \)', content)
        if m:
            return float(m.group(3)), float(m.group(4))
    except:
        pass
    return None, None

# Extract for each placement
clock_pins = {}  # pid → (x_um, y_um, x_norm, y_norm)

csv_df = df  # already loaded

# Get unique def_paths per placement
pid_to_def = {}
for _, row in df.drop_duplicates('placement_id').iterrows():
    pid = row['placement_id']
    def_path_rel = row['def_path']
    def_path = os.path.join(BASE, def_path_rel.replace('../', ''))
    pid_to_def[pid] = def_path

print(f"  Parsing {len(pid_to_def)} DEF files...")
parsed = 0
for pid, def_path in pid_to_def.items():
    dw, dh = get_die_info(def_path)
    pin_pos = get_clock_pin(def_path)
    if pin_pos and dw and dh:
        x_dbu, y_dbu = pin_pos
        # Convert from DBU to µm (assuming 1 DBU = 1 nm → / 1000 for µm)
        x_um = x_dbu / 1000.0
        y_um = y_dbu / 1000.0
        dw_um = dw / 1000.0
        dh_um = dh / 1000.0
        x_norm = x_um / dw_um if dw_um > 0 else 0.5
        y_norm = y_um / dh_um if dh_um > 0 else 0.5
        clock_pins[pid] = {'x_um': x_um, 'y_um': y_um,
                           'x_norm': x_norm, 'y_norm': y_norm,
                           'dw_um': dw_um, 'dh_um': dh_um}
        parsed += 1

print(f"  Parsed clock pins for {parsed}/{len(pid_to_def)} placements")

# Show some examples
for design in ['aes', 'ethmac', 'picorv32', 'sha256']:
    pids_d = [p for p in clock_pins if p.startswith(design)][:2]
    for pid in pids_d:
        cp = clock_pins[pid]
        print(f"  {pid[:30]}: clk=({cp['x_um']:.1f}, {cp['y_um']:.1f})µm "
              f"norm=({cp['x_norm']:.3f}, {cp['y_norm']:.3f}) "
              f"die=({cp['dw_um']:.0f}×{cp['dh_um']:.0f})µm")

# ── Build clock-source features ────────────────────────────────────────────
print(f"\n{T()} Building clock-source spatial features...")

# Features per placement (constant within placement):
# 1. Clock pin position (x_norm, y_norm) [0,1]
# 2. Mean distance from clock pin to FF centroid
# 3. Max distance from clock pin to any FF cluster
# 4. Spatial asymmetry: std of FF distances to clock pin
# 5. Directional bias: fraction of FFs above/below/left/right of clock pin
# 6. Interaction with cluster_dia: mean_dist / cluster_dia (= expected levels)

from scipy.spatial import cKDTree

clk_features_per_pid = {}  # pid → feature vector

for pid in np.unique(pids):
    cp = clock_pins.get(pid)
    pi = ff_pos.get(pid)
    if cp is None or pi is None or pi.get('ff_norm') is None:
        clk_features_per_pid[pid] = None
        continue

    ff_xy_norm = pi['ff_norm']  # [N, 2] in [0,1]
    dw_um = cp['dw_um']; dh_um = cp['dh_um']
    clk_x_norm = cp['x_norm']; clk_y_norm = cp['y_norm']

    # Convert to µm for proper distance calculation
    ff_x_um = ff_xy_norm[:, 0] * dw_um
    ff_y_um = ff_xy_norm[:, 1] * dh_um
    clk_x_um = cp['x_um']; clk_y_um = cp['y_um']

    # Manhattan distances from clock pin to each FF
    dist_manhattan = np.abs(ff_x_um - clk_x_um) + np.abs(ff_y_um - clk_y_um)
    # Euclidean distances
    dist_eucl = np.sqrt((ff_x_um - clk_x_um)**2 + (ff_y_um - clk_y_um)**2)

    N = len(ff_xy_norm)
    # Statistical features of distance distribution
    mean_dist = dist_manhattan.mean()
    std_dist = dist_manhattan.std()
    max_dist = dist_manhattan.max()
    p50_dist = np.percentile(dist_manhattan, 50)
    p90_dist = np.percentile(dist_manhattan, 90)
    p99_dist = np.percentile(dist_manhattan, 99)

    # Fraction of FFs farther than various thresholds
    frac_far_100 = (dist_manhattan > 100).mean()
    frac_far_200 = (dist_manhattan > 200).mean()
    frac_far_300 = (dist_manhattan > 300).mean()

    # Spatial asymmetry relative to clock pin
    above_clk = (ff_y_um > clk_y_um).mean()  # fraction of FFs above clock pin
    right_of_clk = (ff_x_um > clk_x_um).mean()  # fraction to the right
    # centroid direction from clock pin
    centroid_x = ff_x_um.mean(); centroid_y = ff_y_um.mean()
    centroid_dist = np.sqrt((centroid_x-clk_x_um)**2 + (centroid_y-clk_y_um)**2)

    clk_features_per_pid[pid] = {
        'mean_dist': mean_dist, 'std_dist': std_dist,
        'max_dist': max_dist, 'p50_dist': p50_dist,
        'p90_dist': p90_dist, 'p99_dist': p99_dist,
        'frac_far_100': frac_far_100, 'frac_far_200': frac_far_200,
        'frac_far_300': frac_far_300,
        'above_clk': above_clk, 'right_of_clk': right_of_clk,
        'centroid_dist': centroid_dist,
        'x_norm': clk_x_norm, 'y_norm': clk_y_norm
    }

n_with_clk = sum(v is not None for v in clk_features_per_pid.values())
print(f"  Built clock features for {n_with_clk} placements")

# ── Per-run features (interact with cluster_dia) ────────────────────────────
# For each run: mean_dist / cluster_dia = expected levels in clock tree
mean_dist_arr = np.array([
    (clk_features_per_pid.get(pid) or {}).get('mean_dist', 300.0)
    for pid in pids], dtype=float)
p90_dist_arr = np.array([
    (clk_features_per_pid.get(pid) or {}).get('p90_dist', 500.0)
    for pid in pids], dtype=float)
max_dist_arr = np.array([
    (clk_features_per_pid.get(pid) or {}).get('max_dist', 800.0)
    for pid in pids], dtype=float)
std_dist_arr = np.array([
    (clk_features_per_pid.get(pid) or {}).get('std_dist', 100.0)
    for pid in pids], dtype=float)
centroid_dist_arr = np.array([
    (clk_features_per_pid.get(pid) or {}).get('centroid_dist', 200.0)
    for pid in pids], dtype=float)

# Key interaction: mean_dist_to_clock / cluster_dia → levels in clock tree
# More levels → more buffers → higher power
levels_estimate_mean = mean_dist_arr / cd  # ≈ routing levels using mean dist
levels_estimate_p90 = p90_dist_arr / cd   # ≈ levels for far FFs (worst case)
levels_estimate_max = max_dist_arr / cd   # ≈ max levels in tree

# Per-placement z-scores of these interaction terms
def z_within(arr, pids):
    z = np.zeros(len(arr))
    for pid in np.unique(pids):
        idx = np.where(pids==pid)[0]
        v = arr[idx]; z[idx] = (v-v.mean())/max(v.std(),1e-8)
    return z

z_lev_mean = z_within(levels_estimate_mean, pids)
z_lev_p90 = z_within(levels_estimate_p90, pids)
z_lev_max = z_within(levels_estimate_max, pids)

# Also: absolute distance features (per-placement constants, not z-scored)
# These can help the model distinguish between designs
mean_dist_n = mean_dist_arr / mean_dist_arr.max()
p90_dist_n = p90_dist_arr / p90_dist_arr.max()

print(f"\n  Clock-source interaction feature correlations:")
for name, z in [('z_mean_dist/cd', z_lev_mean),
                ('z_p90_dist/cd', z_lev_p90),
                ('z_max_dist/cd', z_lev_max)]:
    rho_pw, _ = spearmanr(z, y_pw)
    rho_wl, _ = spearmanr(z, y_wl)
    print(f"    {name}: rho_power={rho_pw:.4f}, rho_WL={rho_wl:.4f}")

# Per-placement constant features
above_clk_arr = np.array([
    (clk_features_per_pid.get(pid) or {}).get('above_clk', 0.5)
    for pid in pids], dtype=float)
x_norm_arr = np.array([
    (clk_features_per_pid.get(pid) or {}).get('x_norm', 0.5)
    for pid in pids], dtype=float)
y_norm_arr = np.array([
    (clk_features_per_pid.get(pid) or {}).get('y_norm', 0.5)
    for pid in pids], dtype=float)

for name, arr in [('mean_dist', mean_dist_arr), ('std_dist', std_dist_arr),
                   ('centroid_dist', centroid_dist_arr), ('above_clk', above_clk_arr),
                   ('clk_x_norm', x_norm_arr), ('clk_y_norm', y_norm_arr)]:
    rho_pw, _ = spearmanr(arr, y_pw)
    print(f"    {name} (per-plac const) vs z_power: rho={rho_pw:.4f}")

# ── Build feature matrices ─────────────────────────────────────────────────
# Per-run interaction features (vary with cluster_dia)
X_clk_per_run = np.column_stack([z_lev_mean, z_lev_p90, z_lev_max])

# Per-placement context features (constants)
X_clk_ctx = np.column_stack([mean_dist_n, p90_dist_n,
                               above_clk_arr, x_norm_arr, y_norm_arr,
                               std_dist_arr / std_dist_arr.max()])

# Combined
X_clk_full = np.hstack([X_clk_per_run, X_clk_ctx])
X_aug = np.hstack([X29T, X_clk_full])

for arr in [X_clk_full, X_aug]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:,c])
        if bad.any(): arr[bad,c] = 0.0

# Best from previous experiments
n_ff_arr = np.array([def_cache.get(p, {}).get('n_ff', 2500) for p in pids], dtype=float)
die_area_arr = np.array([def_cache.get(p, {}).get('die_area', 400000) for p in pids], dtype=float)
ff_hpwl_arr = np.array([def_cache.get(p, {}).get('ff_hpwl', 1000) for p in pids], dtype=float)
lambda_arr = n_ff_arr / die_area_arr
n_in_radius = lambda_arr * np.pi * (cd/2)**2
effective_cs = np.minimum(cs, n_in_radius); effective_cs = np.maximum(effective_cs, 1.0)
n_clusters_phys = n_ff_arr / effective_cs
z_nclusters_phys = z_within(n_clusters_phys, pids)
z_hpwl_cd = z_within(ff_hpwl_arr/cd, pids)

X_phys = np.column_stack([z_nclusters_phys, z_hpwl_cd])
X_best_so_far = np.hstack([X29T, X_phys])
X_best_plus_clk = np.hstack([X29T, X_phys, X_clk_full])

for arr in [X_phys, X_best_so_far, X_best_plus_clk]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:,c])
        if bad.any(): arr[bad,c] = 0.0

# ── LODO evaluation ────────────────────────────────────────────────────────
def lodo_lgb(X, y, label, ne=300, nl=20, lr=0.03, mc=15):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        m = LGBMRegressor(n_estimators=ne, num_leaves=nl, learning_rate=lr,
                          min_child_samples=mc, verbose=-1)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(sc.transform(X[te]))))
    mean_mae = np.mean(maes)
    tag = ' ✓' if mean_mae < 0.10 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f} mean={mean_mae:.4f}{tag}")
    return maes

print(f"\n{T()} === LODO POWER MAE ===")
lodo_lgb(X29T, y_pw, "X29T baseline (0.2163)")
lodo_lgb(X_aug, y_pw, "X29T + clock_source_feats")
lodo_lgb(X_best_so_far, y_pw, "X29T + phys_sim (best prev = 0.2098)")
lodo_lgb(X_best_plus_clk, y_pw, "X29T + phys_sim + clock_source")
lodo_lgb(X_best_plus_clk, y_pw, "X29T+phys+clk (nl=15,mc=20)", nl=15, mc=20)

print(f"\n{T()} === LODO WL MAE ===")
lodo_lgb(X29T, y_wl, "X29T baseline (0.2338)")
lodo_lgb(X_aug, y_wl, "X29T + clock_source_feats")
lodo_lgb(X_best_so_far, y_wl, "X29T + phys_sim")
lodo_lgb(X_best_plus_clk, y_wl, "X29T + phys_sim + clock_source")

# ── Ablation: which clock features matter? ────────────────────────────────
print(f"\n{T()} === ABLATION: which clock features matter? ===")
print("  Power:")
lodo_lgb(np.hstack([X29T, X_clk_per_run]), y_pw, "X29T + clk_per_run only")
lodo_lgb(np.hstack([X29T, X_clk_ctx]), y_pw, "X29T + clk_ctx only")

# Specifically: the ratio levels = mean_dist / cluster_dia
lodo_lgb(np.hstack([X29T, z_lev_mean.reshape(-1,1)]), y_pw, "X29T + z(mean_dist/cd)")
lodo_lgb(np.hstack([X29T, z_lev_p90.reshape(-1,1)]), y_pw, "X29T + z(p90_dist/cd)")
lodo_lgb(np.hstack([X29T, z_lev_max.reshape(-1,1)]), y_pw, "X29T + z(max_dist/cd)")

print(f"\n{T()} DONE")
