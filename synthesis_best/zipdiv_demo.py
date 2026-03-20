"""
zipdiv_demo.py — Zero-shot Pareto optimization on zipdiv (truly unseen design)

zipdiv is NOT in the training manifest — it's a brand-new design.
We parse its DEF/SAIF/timing directly, build features, load the production
4-target model, and run the Pareto optimizer.

No ground truth → no accuracy metrics, but the optimizer recommends
optimal CTS knobs from scratch, in milliseconds.
"""

import re, os, sys, time, pickle
import numpy as np
import pandas as pd
from collections import Counter

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

BASE = '/home/rain/CTS-Task-Aware-Clustering'
PLACEMENT_DIR = f'{BASE}/dataset_with_def/placement_files'
MODEL_PATH = f'{BASE}/synthesis_best/saved_models/cts_predictor_4target.pkl'

sys.path.insert(0, f'{BASE}/synthesis_best')
from build_skew_cache import parse_def_ff_positions, compute_skew_features

ZIPDIV_PIDS = ['zipdiv_run_20260312_160558', 'zipdiv_run_20260312_160735']
T_CLK_NS = 10.0   # zipdiv target clock period (assume 100MHz)


# ── Parsers (from absolute_v7.py) ────────────────────────────────────────────

def parse_def(def_path):
    with open(def_path) as f:
        content = f.read()

    units_m = re.search(r'UNITS DISTANCE MICRONS (\d+)', content)
    units = int(units_m.group(1)) if units_m else 1000

    die_m = re.search(
        r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)',
        content)
    if not die_m:
        return None
    x0, y0, x1, y1 = [float(v) / units for v in die_m.groups()]
    die_w, die_h = x1 - x0, y1 - y0
    die_area = die_w * die_h

    ct = Counter(re.findall(r'sky130_fd_sc_hd__(\w+)', content))
    n_total = sum(ct.values())

    filler_keys = ['tap', 'decap', 'fill', 'phy']
    n_tap = sum(v for k, v in ct.items() if any(x in k for x in filler_keys))
    n_active = n_total - n_tap
    n_ff  = sum(v for k, v in ct.items() if k.startswith('df') or k.startswith('ff'))
    n_buf = sum(v for k, v in ct.items() if k.startswith('buf'))
    n_inv = sum(v for k, v in ct.items() if k.startswith('inv'))
    n_xor_xnor = sum(v for k, v in ct.items() if k.startswith('xor') or k.startswith('xnor'))
    n_mux = sum(v for k, v in ct.items() if k.startswith('mux'))
    n_and_or = sum(v for k, v in ct.items() if k.startswith('and') or k.startswith('or'))
    n_nand_nor = sum(v for k, v in ct.items() if k.startswith('nand') or k.startswith('nor'))
    n_comb = max(n_active - n_ff - n_buf - n_inv, 0)

    active_ds = []
    for k, v in ct.items():
        if not any(x in k for x in filler_keys):
            m = re.search(r'_(\d+)$', k)
            if m:
                active_ds.extend([int(m.group(1))] * v)

    avg_ds = np.mean(active_ds) if active_ds else 1.0
    std_ds = np.std(active_ds) if len(active_ds) > 1 else 0.0
    p90_ds = np.percentile(active_ds, 90) if active_ds else 1.0
    frac_ds4plus = sum(1 for d in active_ds if d >= 4) / (len(active_ds) + 1)

    ff_pattern = (r'-\s+\S+\s+(sky130_fd_sc_hd__df\w+)\s+'
                  r'\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)')
    ff_xy = [(float(x) / units, float(y) / units)
             for _, x, y in re.findall(ff_pattern, content)]
    if not ff_xy:
        return None

    xs = np.array([p[0] for p in ff_xy])
    ys = np.array([p[1] for p in ff_xy])
    ff_hpwl = (xs.max() - xs.min()) + (ys.max() - ys.min())
    ff_bbox_area = (xs.max() - xs.min()) * (ys.max() - ys.min()) + 1.0
    ff_spacing = np.sqrt(ff_bbox_area / max(len(ff_xy), 1))

    return {
        'die_area': die_area, 'die_w': die_w, 'die_h': die_h,
        'die_aspect': die_w / (die_h + 1e-6),
        'ff_hpwl': ff_hpwl, 'ff_spacing': ff_spacing,
        'ff_density': len(ff_xy) / die_area,
        'ff_cx': xs.mean() / die_w, 'ff_cy': ys.mean() / die_h,
        'ff_x_std': xs.std() / die_w, 'ff_y_std': ys.std() / die_h,
        'n_ff': len(ff_xy), 'n_active': n_active, 'n_total': n_total, 'n_tap': n_tap,
        'n_buf': n_buf, 'n_inv': n_inv, 'n_comb': n_comb,
        'n_xor_xnor': n_xor_xnor, 'n_mux': n_mux,
        'n_and_or': n_and_or, 'n_nand_nor': n_nand_nor,
        'frac_xor': n_xor_xnor / (n_active + 1),
        'frac_mux': n_mux / (n_active + 1),
        'frac_and_or': n_and_or / (n_active + 1),
        'frac_nand_nor': n_nand_nor / (n_active + 1),
        'frac_ff_active': n_ff / (n_active + 1),
        'frac_buf_inv': (n_buf + n_inv) / (n_active + 1),
        'comb_per_ff': n_comb / (n_ff + 1),
        'avg_ds': avg_ds, 'std_ds': std_ds, 'p90_ds': p90_ds,
        'frac_ds4plus': frac_ds4plus,
        'cap_proxy': n_active * avg_ds,
        'ff_cap_proxy': len(ff_xy) * avg_ds,
    }


def parse_saif(saif_path):
    with open(saif_path) as f:
        lines = f.readlines()

    total_tc = total_t1 = n_nets = max_tc = 0
    tc_vals = []
    duration = None

    for line in lines:
        if '(DURATION' in line:
            m = re.search(r'[\d.]+', line)
            if m:
                duration = float(m.group())
        m = re.search(r'\(TC\s+(\d+)\)', line)
        if m:
            tc = int(m.group(1))
            tc_vals.append(tc)
            n_nets += 1; total_tc += tc; max_tc = max(max_tc, tc)
        m2 = re.search(r'\(T1\s+(\d+)\)', line)
        if m2:
            total_t1 += int(m2.group(1))

    if n_nets == 0 or max_tc == 0:
        return None

    tc_arr = np.array(tc_vals, dtype=float)
    mean_tc = total_tc / n_nets
    rel_act = mean_tc / max_tc
    mean_sig_prob = total_t1 / (n_nets * duration) if duration else 0.0
    return {
        'n_nets': n_nets, 'max_tc': max_tc, 'mean_tc': mean_tc,
        'rel_act': rel_act, 'mean_sig_prob': mean_sig_prob,
        'tc_std_norm': tc_arr.std() / (mean_tc + 1),
        'frac_zero': (tc_arr == 0).mean(),
        'frac_high_act': (tc_arr > mean_tc * 2).mean(),
        'log_n_nets': np.log1p(n_nets),
    }


def parse_timing(tp_path):
    tp = pd.read_csv(tp_path)
    sl = tp['slack'].values
    return {
        'n_paths': len(sl),
        'slack_mean': sl.mean(), 'slack_std': sl.std(),
        'slack_min': sl.min(), 'slack_p10': np.percentile(sl, 10),
        'slack_p50': np.percentile(sl, 50),
        'frac_neg': (sl < 0).mean(),
        'frac_tight': (sl < 0.5).mean(),
        'frac_critical': (sl < 0.1).mean(),
    }


# ── Feature builder (matches multiobjective.py / final_synthesis.py) ─────────

def build_features_for_knobs(df_f, sf, tf, skf, cd, cs, mw, bd):
    """
    Build (X_pw, X_wl, X_sk, X_hv) for a single (cd,cs,mw,bd) knob config.
    skf = skew spatial features dict (from build_skew_cache)
    """
    f_ghz = 1.0 / T_CLK_NS
    t_clk = T_CLK_NS
    # No synth strategy info — use default AREA 2 encoding
    sd, sl_s, sa = 0.0, 0.0, 1.0
    core_util = 0.55  # typical default
    density = 0.50

    n_ff     = df_f['n_ff'];    n_active = df_f['n_active']
    die_area = df_f['die_area']; ff_hpwl = df_f['ff_hpwl']
    ff_spacing = df_f['ff_spacing']; avg_ds = df_f['avg_ds']
    frac_xor = df_f['frac_xor']; frac_mux = df_f['frac_mux']
    comb_per_ff = df_f['comb_per_ff']; n_comb = df_f['n_comb']
    n_nets = sf['n_nets']; rel_act = sf['rel_act']

    pw_norm = max(n_ff * f_ghz * avg_ds, 1e-10)
    wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)

    sm = tf['slack_mean']; fn = tf['frac_neg']; ft = tf['frac_tight']

    # ── Power features (76 dims) ──────────────────────────────────────────
    base_pw = [
        np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
        df_f['die_aspect'], 1.0,
        df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
        frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
        df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
        avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
        np.log1p(df_f['cap_proxy']),
        rel_act, sf['mean_sig_prob'], sf['tc_std_norm'], sf['frac_zero'],
        sf['frac_high_act'], sf['log_n_nets'], n_nets / (n_ff + 1),
        f_ghz, t_clk, sd, sl_s, sa, core_util, density,
        np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd,
        frac_xor * comb_per_ff, rel_act * frac_xor, rel_act * (1 - df_f['frac_ff_active']),
        sd * avg_ds, sa * f_ghz,
        np.log1p(cd * n_ff / die_area), np.log1p(cs * ff_spacing),
        np.log1p(mw * ff_hpwl), np.log1p(n_ff / cs), core_util * density,
        np.log1p(n_active * rel_act * f_ghz), np.log1p(frac_xor * n_active),
        np.log1p(frac_mux * n_active), np.log1p(comb_per_ff * n_ff),
    ]  # 58 dims
    timing_pw = [
        sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'], tf['slack_p50'],
        fn, ft, tf['frac_critical'], tf['n_paths'] / (n_ff + 1),
        sm * frac_xor, sm * comb_per_ff, fn * comb_per_ff, ft * avg_ds,
        float(sm > 1.5), float(sm > 2.0), float(sm > 3.0), np.log1p(sm), sm * f_ghz,
    ]  # 18 dims
    X_pw = np.array(base_pw + timing_pw, dtype=np.float64)  # 76

    # ── WL features (84 dims) ────────────────────────────────────────────
    base_wl = [
        np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
        df_f['die_aspect'], 1.0,
        df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
        frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
        df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
        avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
        np.log1p(df_f['cap_proxy']),
        rel_act, sf['mean_sig_prob'], sf['tc_std_norm'], sf['frac_zero'],
        sf['frac_high_act'], sf['log_n_nets'], n_nets / (n_ff + 1),
        f_ghz, t_clk, core_util, density,
        np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd,
        frac_xor * comb_per_ff, rel_act * frac_xor, rel_act * (1 - df_f['frac_ff_active']),
        np.log1p(cd * n_ff / die_area), np.log1p(cs * ff_spacing),
        np.log1p(mw * ff_hpwl), np.log1p(n_ff / cs), core_util * density,
        np.log1p(n_active * rel_act * f_ghz), np.log1p(frac_xor * n_active),
        np.log1p(frac_mux * n_active), np.log1p(comb_per_ff * n_ff),
    ]  # 53 dims
    # gravity = 0 (no .pt graph for zipdiv)
    gravity = [0.0] * 19
    extra_scale = [
        np.log1p(die_area / (n_ff + 1)), np.log1p(n_comb),
        comb_per_ff * np.log1p(n_ff),
    ]  # 3 dims
    # net features = 0 (no pre-built net cache)
    net_feats = [0.0] * 9
    X_wl = np.array(base_wl + gravity + extra_scale + net_feats, dtype=np.float64)  # 84

    # ── Skew features (63 dims) ──────────────────────────────────────────
    crit_max  = skf.get('crit_max_dist', 0.0)
    crit_mean = skf.get('crit_mean_dist', 0.0)
    crit_p90  = skf.get('crit_p90_dist', 0.0)
    crit_hpwl = skf.get('crit_ff_hpwl', 0.0)
    crit_cx   = skf.get('crit_cx_offset', 0.0)
    crit_cy   = skf.get('crit_cy_offset', 0.0)
    crit_xs   = skf.get('crit_x_std', 0.0)
    crit_ys   = skf.get('crit_y_std', 0.0)
    crit_bnd  = skf.get('crit_frac_boundary', 0.0)
    crit_star = skf.get('crit_star_degree', 0.0)
    crit_chn  = skf.get('crit_chain_frac', 0.0)
    crit_asym = skf.get('crit_asymmetry', 0.0)
    crit_ecc  = skf.get('crit_eccentricity', 1.0)
    crit_dens = skf.get('crit_density_ratio', 1.0)
    crit_max_um  = skf.get('crit_max_dist_um', ff_hpwl)
    crit_mean_um = skf.get('crit_mean_dist_um', ff_hpwl / 2)

    skew_feats = [
        np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl),
        np.log1p(ff_spacing), df_f['die_aspect'],
        df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
        frac_xor, comb_per_ff, avg_ds, rel_act, sf['mean_sig_prob'],
        sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'],
        fn, ft, tf['frac_critical'], np.log1p(tf['n_paths'] / (n_ff + 1)),
        np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
        cd, cs, mw, bd,
        crit_max, crit_mean, crit_p90, crit_hpwl,
        crit_cx, crit_cy, crit_xs, crit_ys,
        crit_bnd, crit_star, crit_chn,
        crit_asym, crit_ecc, crit_dens,
        np.log1p(crit_max_um), np.log1p(crit_mean_um),
        cd / (ff_spacing + 1), bd / (crit_max_um + 1), mw / (crit_max_um + 1),
        crit_star * cd, crit_asym * mw, crit_dens * cs,
        crit_max * cd, crit_asym * crit_max, fn * crit_star, ft * crit_chn,
        crit_hpwl / (cs + 1),
        np.log1p(crit_max_um / (cd + 1)), np.log1p(crit_max_um / (bd + 1)),
        np.log1p(crit_max_um / (mw + 1)),
        crit_cx * cd, crit_cy * mw, np.log1p(n_ff / cs) * crit_hpwl,
    ]  # 63 dims (matches final_synthesis.py)
    X_sk = np.array(skew_feats, dtype=np.float64)

    # ── HoldVio features (66 dims = base_wl53 + hold9 + net4) ───────────
    hold_phys = [
        np.log1p(n_ff / cs), np.log1p(cs * ff_spacing),
        np.log1p(cd / (ff_spacing + 1)), np.log1p(bd / (ff_hpwl + 1)),
        bd / (crit_max_um + 1e-3),
        crit_star * cs, crit_chn * bd,
        crit_asym * cd, np.log1p(crit_max * bd),
    ]  # 9 dims
    net4 = [0.0] * 4  # no net cache for zipdiv
    X_hv = np.array(base_wl + hold_phys + net4, dtype=np.float64)  # 66

    return X_pw, X_wl, X_sk, X_hv, pw_norm, wl_norm


# ── Pareto front ─────────────────────────────────────────────────────────────

def pareto_front(costs):
    n = costs.shape[0]
    lo = costs.min(0); rng = (costs.max(0) - lo) + 1e-10
    c = (costs - lo) / rng
    is_dominated = np.zeros(n, dtype=bool)
    chunk = 500
    for i in range(0, n, chunk):
        ci = c[i:i+chunk]
        dominated_by = (
            np.all(c[:, None, :] <= ci[None, :, :] + 1e-9, axis=2) &
            np.any(c[:, None, :] <  ci[None, :, :] - 1e-9, axis=2)
        )
        is_dominated[i:i+chunk] = dominated_by.any(axis=0)
    return ~is_dominated


# ── Main ─────────────────────────────────────────────────────────────────────

print("=" * 70)
print("Zipdiv Zero-Shot Demo — Unseen Design, Pareto CTS Optimization")
print("=" * 70)
print("  zipdiv: NOT in training set (not in manifest), 142 FFs, ~0.04 mm²")
print()

# Load model
print(f"{T()} Loading production 4-target model...")
with open(MODEL_PATH, 'rb') as f:
    mdl = pickle.load(f)
m_pw   = mdl['model_power'];    sc_pw  = mdl['scaler_power']
lgb_wl = mdl['model_wl_lgb'];  rdg_wl = mdl['model_wl_ridge']
sc_wl  = mdl['scaler_wl']
m_sk   = mdl['model_skew'];    sc_sk  = mdl['scaler_skew']
m_hv   = mdl['model_hold_vio']; sc_hv  = mdl['scaler_hold_vio']
print(f"  LODO validation: {mdl['lodo']}")

# Build per-placement features for both zipdiv placements
print(f"\n{T()} Parsing zipdiv DEF/SAIF/timing...")
placements = {}
for pid in ZIPDIV_PIDS:
    d = f'{PLACEMENT_DIR}/{pid}'
    df_f = parse_def(f'{d}/zipdiv.def')
    sf   = parse_saif(f'{d}/zipdiv.saif')
    tf   = parse_timing(f'{d}/timing_paths.csv')

    # Skew spatial features
    ff_pos, die_w, die_h, origin = parse_def_ff_positions(f'{d}/zipdiv.def')
    timing_df = pd.read_csv(f'{d}/timing_paths.csv')
    skf = compute_skew_features(ff_pos, die_w, die_h, origin, timing_df) or {}

    if df_f and sf and tf:
        placements[pid] = {'df_f': df_f, 'sf': sf, 'tf': tf, 'skf': skf}
        print(f"  {pid}: n_ff={df_f['n_ff']}, "
              f"die={df_f['die_w']:.0f}×{df_f['die_h']:.0f}µm, "
              f"rel_act={sf['rel_act']:.4f}, slack_mean={tf['slack_mean']:.3f}ns")
    else:
        print(f"  {pid}: FAILED to parse")

if not placements:
    print("ERROR: no zipdiv placements parsed"); sys.exit(1)

# Use first placement for Pareto demo
pid0 = ZIPDIV_PIDS[0]
p0 = placements[pid0]

# Compute per-placement skew mu/sig from a few reference knob combos
# (we can't z-score without ground truth — use the model's raw prediction
# but we need to un-z-score; instead, train a rough skew mu/sig proxy
# from 10 reference knobs sampled from the sweep)
rng_seed = np.random.default_rng(0)
ref_cds = rng_seed.uniform(35, 70, 10)
ref_css = rng_seed.integers(12, 31, 10).astype(float)
ref_mws = rng_seed.uniform(130, 280, 10)
ref_bds = rng_seed.uniform(70, 150, 10)
ref_sk_z = []
for cd, cs, mw, bd in zip(ref_cds, ref_css, ref_mws, ref_bds):
    _, _, X_sk_r, _, _, _ = build_features_for_knobs(
        p0['df_f'], p0['sf'], p0['tf'], p0['skf'], cd, cs, mw, bd)
    if len(X_sk_r) == 63:
        ref_sk_z.append(m_sk.predict(sc_sk.transform(X_sk_r.reshape(1,-1)))[0])

# Use mu/sig from a representative training design (aes) as prior for skew denorm
# Since zipdiv is small (142 FFs), skew should be lower than aes (~0.7ns)
# Use a conservative prior: mu_sk=0.5, sig_sk=0.15 (typical small design)
sk_mu_prior  = 0.50
sk_sig_prior = 0.15

print(f"\n{T()} === PARETO OPTIMIZER ON ZIPDIV ===")
print(f"  Placement: {pid0}")
print(f"  Note: No ground truth — optimizer recommends optimal knobs from scratch")
print(f"  Skew denorm prior: mu={sk_mu_prior}ns, sig={sk_sig_prior}ns")
print(f"  (small design with 142 FFs — expect lower skew than AES 2994 FFs)\n")

n_samples = 5000
rng2 = np.random.default_rng(42)
cd_arr = rng2.uniform(20, 65,   n_samples)   # smaller range for small design
cs_arr = rng2.integers(8, 25,   n_samples).astype(float)
mw_arr = rng2.uniform(80, 250,  n_samples)
bd_arr = rng2.uniform(50, 130,  n_samples)

# Build base feature row at median knobs
cd0, cs0, mw0, bd0 = 40.0, 16.0, 165.0, 90.0
X_pw0, X_wl0, X_sk0, X_hv0, pw_norm, wl_norm = build_features_for_knobs(
    p0['df_f'], p0['sf'], p0['tf'], p0['skf'], cd0, cs0, mw0, bd0)

print(f"  Feature dims: X_pw={len(X_pw0)}, X_wl={len(X_wl0)}, "
      f"X_sk={len(X_sk0)}, X_hv={len(X_hv0)}")

if len(X_sk0) != 63:
    print(f"  WARNING: skew feature dim={len(X_sk0)}, expected 63")

KNOB_IDX = {
    'pw': {'log': [36,37,38,39], 'raw': [40,41,42,43],
           'inter': [(49,'cd'),(50,'cs'),(51,'mw'),(52,'cs_inv')]},
    'wl': {'log': [33,34,35,36], 'raw': [37,38,39,40],
           'inter': [(44,'cd'),(45,'cs'),(46,'mw'),(47,'cs_inv')]},
    'sk': {'log': [20,21,22,23], 'raw': [24,25,26,27]},
}

n_ff     = p0['df_f']['n_ff']
die_area = p0['df_f']['die_area']
ff_hpwl  = p0['df_f']['ff_hpwl']
ff_spacing = p0['df_f']['ff_spacing']

def patch(x, ki, cd, cs, mw, bd):
    X = np.tile(x, (n_samples, 1)).astype(np.float64)
    for li, v in zip(ki['log'], [cd, cs, mw, bd]): X[:, li] = np.log1p(v)
    for ri, v in zip(ki['raw'], [cd, cs, mw, bd]):  X[:, ri] = v
    for (ii, kind) in ki.get('inter', []):
        if kind == 'cd':      X[:, ii] = np.log1p(cd * n_ff / die_area)
        elif kind == 'cs':    X[:, ii] = np.log1p(cs * ff_spacing)
        elif kind == 'mw':    X[:, ii] = np.log1p(mw * ff_hpwl)
        elif kind == 'cs_inv':X[:, ii] = np.log1p(n_ff / cs)
    return X

hv_ki = {'log': [33,34,35,36], 'raw': [37,38,39,40],
          'inter': [(44,'cd'),(45,'cs'),(46,'mw'),(47,'cs_inv')]}

t_start = time.time()
Xpw = patch(X_pw0, KNOB_IDX['pw'], cd_arr, cs_arr, mw_arr, bd_arr)
Xwl = patch(X_wl0, KNOB_IDX['wl'], cd_arr, cs_arr, mw_arr, bd_arr)
Xsk = patch(X_sk0, KNOB_IDX['sk'], cd_arr, cs_arr, mw_arr, bd_arr)
Xhv = patch(X_hv0, hv_ki,           cd_arr, cs_arr, mw_arr, bd_arr)

pred_pw  = np.exp(m_pw.predict(sc_pw.transform(Xpw))) * pw_norm
Xwl_s    = sc_wl.transform(Xwl)
pred_wl  = np.exp(0.3 * lgb_wl.predict(Xwl_s) + 0.7 * rdg_wl.predict(Xwl_s)) * wl_norm
pred_sk  = m_sk.predict(sc_sk.transform(Xsk)) * sk_sig_prior + sk_mu_prior
hv_mu_prior  = np.log1p(50.0)   # small design: expect ~50 hold vios
hv_sig_prior = 1.0
pred_hv_z = m_hv.predict(sc_hv.transform(Xhv))
pred_hv  = np.expm1(np.clip(pred_hv_z * hv_sig_prior + hv_mu_prior, 0, 20))

t_ms = (time.time() - t_start) * 1000

df_sweep = pd.DataFrame({
    'cd': cd_arr, 'cs': cs_arr.astype(int),
    'mw': mw_arr.round(0), 'bd': bd_arr.round(0),
    'power_mW': pred_pw * 1000, 'wl_mm': pred_wl / 1000,
    'skew_ns': pred_sk, 'hold_vio': pred_hv,
})

df_feas = df_sweep[df_sweep['skew_ns'] > 0].copy()
costs = df_feas[['power_mW', 'skew_ns', 'hold_vio']].values
pareto_mask = pareto_front(costs)
df_feas['pareto'] = pareto_mask
pareto = df_feas[df_feas['pareto']].sort_values('power_mW')

print(f"  Surrogate sweep: {n_samples} combos in {t_ms:.0f}ms")
print(f"  Pareto-optimal solutions: {len(pareto)} / {n_samples}")
print(f"\n  Top-10 Pareto solutions (sorted by predicted power):")
print(f"  {'cd':>5} {'cs':>4} {'mw':>5} {'bd':>5} | "
      f"{'Power(mW)':>10} {'WL(mm)':>8} {'Skew(ns)':>9} {'HoldVio':>8}")
print(f"  {'-'*5}-{'-'*4}-{'-'*5}-{'-'*5}-+-{'-'*10}-{'-'*8}-{'-'*9}-{'-'*8}")
for _, r in pareto.head(10).iterrows():
    print(f"  {r.cd:>5.0f} {r.cs:>4.0f} {r.mw:>5.0f} {r.bd:>5.0f} | "
          f"{r.power_mW:>9.3f}  {r.wl_mm:>7.2f}  {r.skew_ns:>8.4f}  {r.hold_vio:>8.1f}")

best_pw  = df_sweep.loc[df_sweep['power_mW'].idxmin()]
best_sk  = df_sweep.loc[df_sweep['skew_ns'].idxmin()]
best_hv  = df_sweep.loc[df_sweep['hold_vio'].idxmin()]
print(f"\n  Best power knob: cd={best_pw.cd:.0f} cs={best_pw.cs:.0f} "
      f"mw={best_pw.mw:.0f} bd={best_pw.bd:.0f} → {best_pw.power_mW:.3f}mW")
print(f"  Best skew  knob: cd={best_sk.cd:.0f} cs={best_sk.cs:.0f} "
      f"mw={best_sk.mw:.0f} bd={best_sk.bd:.0f} → {best_sk.skew_ns:.4f}ns")
print(f"  Best hold  knob: cd={best_hv.cd:.0f} cs={best_hv.cs:.0f} "
      f"mw={best_hv.mw:.0f} bd={best_hv.bd:.0f} → {best_hv.hold_vio:.0f} vio")

print(f"\n  Predicted ranges across all 5000 configs:")
print(f"    Power: {df_sweep.power_mW.min():.2f} – {df_sweep.power_mW.max():.2f} mW")
print(f"    WL:    {df_sweep.wl_mm.min():.2f}  – {df_sweep.wl_mm.max():.2f}  mm")
print(f"    Skew:  {df_sweep.skew_ns.min():.4f} – {df_sweep.skew_ns.max():.4f} ns")
print(f"    HoldVio: {df_sweep.hold_vio.min():.0f} – {df_sweep.hold_vio.max():.0f}")

# Also demo second placement
if len(placements) > 1:
    pid1 = ZIPDIV_PIDS[1]
    p1 = placements[pid1]
    X_pw1, X_wl1, X_sk1, X_hv1, pw_norm1, wl_norm1 = build_features_for_knobs(
        p1['df_f'], p1['sf'], p1['tf'], p1['skf'], cd0, cs0, mw0, bd0)

    Xpw1 = patch(X_pw1, KNOB_IDX['pw'], cd_arr, cs_arr, mw_arr, bd_arr)
    Xwl1 = patch(X_wl1, KNOB_IDX['wl'], cd_arr, cs_arr, mw_arr, bd_arr)
    pred_pw1 = np.exp(m_pw.predict(sc_pw.transform(Xpw1))) * pw_norm1
    Xwl1_s   = sc_wl.transform(Xwl1)
    pred_wl1 = np.exp(0.3*lgb_wl.predict(Xwl1_s)+0.7*rdg_wl.predict(Xwl1_s))*wl_norm1

    print(f"\n  Placement 2 ({pid1}):")
    print(f"    n_ff={p1['df_f']['n_ff']}, rel_act={p1['sf']['rel_act']:.4f}")
    print(f"    Predicted power range: {pred_pw1.min()*1000:.2f}–{pred_pw1.max()*1000:.2f} mW")
    print(f"    Predicted WL range:    {pred_wl1.min()/1000:.2f}–{pred_wl1.max()/1000:.2f} mm")

print(f"\n{T()} DONE")
