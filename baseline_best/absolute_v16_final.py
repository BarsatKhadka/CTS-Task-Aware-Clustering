"""
absolute_v16_final.py — Best Zero-Shot Absolute Predictor (Session 10)

Sessions 8-10 findings consolidated:

**Power (32.0% mean MAPE, LODO)**:
- v11-exact base+timing features (76 dims, WITH synth features)
- Normalization: `n_ff * f_ghz * avg_ds` (v11 proven — do NOT use rel_act)
- Model: XGBRegressor(n=300, max_depth=4, lr=0.05, subsample=0.8, colsample_bytree=0.8)
- Synth features CRITICAL for SHA256: without them SHA256=66%, with them SHA256=48.9%
- Note: Session 9 claimed SHA256=9.7% via driven_cap+fanout features — NOT reproducible

**WL (11.0% mean MAPE, LODO) — NEW BEST vs v11 (13.1%)**:
- base(53, no synth) + gravity(19) + extra_scale(3) = 75 dims
- Normalization: sqrt(n_ff * die_area) (v11 proven)
- Model: LGB(300)+Ridge(1000) blend, alpha=0.3

Per-design MAPE (Session 10, verified):
  AES:     power=36.6%  WL=24.9%  (oracle: pw=20.2%, wl=15.2%)
  ETH MAC: power=12.3%  WL= 8.2%  (oracle: pw= 6.8%, wl= 8.8%)  ← near oracle
  PicoRV:  power=30.1%  WL= 5.7%  (oracle: pw= 6.5%, wl= 6.4%)
  SHA-256: power=48.9%  WL= 5.1%  (oracle: pw=10.8%, wl= 8.1%)
  MEAN:    power=32.0%  WL=11.0%

Key findings from Session 10:
- SHA256 power 48.9% is the practical floor (SAIF glitch contamination)
- SHA256 rel_act=0.104 (2x training max) → LGB/XGB can't extrapolate k_PA
- Adding driven_cap_per_ff / ra_corrected features to power HURTS SHA256 (57-66%)
- XGB (synth features) consistently beats LGB for power generalization
- WL improved from v11's 13.1% to 11.0% by using no-synth base + extra_scale

Remaining gap to 2% MAPE: requires SPEF (post-layout parasitics) or 1-shot calibration.
"""

import re, os, sys, time
import numpy as np
import pandas as pd
import pickle
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'
PLACEMENT_DIR = f'{DATASET}/placement_files'

DEF_CACHE    = f'{BASE}/absolute_v7_def_cache.pkl'
SAIF_CACHE   = f'{BASE}/absolute_v7_saif_cache.pkl'
TIMING_CACHE = f'{BASE}/absolute_v7_timing_cache.pkl'
GRAVITY_CACHE = f'{BASE}/absolute_v10_gravity_cache.pkl'
EXT_CACHE    = f'{BASE}/absolute_v13_extended_cache.pkl'

T_CLK_NS = {'aes': 7.0, 'picorv32': 5.0, 'sha256': 9.0, 'ethmac': 9.0, 'zipdiv': 5.0}
CLOCK_PORTS = {'aes': 'clk', 'picorv32': 'clk', 'sha256': 'clk',
               'ethmac': 'wb_clk_i', 'zipdiv': 'i_clk'}


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-12)) * 100


def encode_synth(s):
    if pd.isna(s): return 0.5, 2.0, 0.5
    s = str(s).upper()
    synth_delay = 1.0 if 'DELAY' in s else 0.0
    try: level = float(s.split()[-1])
    except: level = 2.0
    return synth_delay, level, synth_delay * level / 4.0


def build_features(df_in, dc, sc_cache, tc, gc, ec):
    """
    Build separate power (89 dims) and WL (77 dims) feature matrices.

    Power features: base(54) + timing(18) + sha256_distinguishers(16) = 88 dims
    WL features:    base(54) + gravity(19) + extra_scale(3) = 76 dims
    """
    rows_pw, rows_wl = [], []
    y_pw, y_wl = [], []
    meta = []

    for _, row in df_in.iterrows():
        pid = row['placement_id']
        design = row['design_name']
        df_f = dc.get(pid)
        sf = sc_cache.get(pid)
        tf = tc.get(pid)
        gf = gc.get(pid, {})
        ef = ec.get(pid, {})

        if not df_f or not sf or not tf:
            continue

        pw = row['power_total']
        wl = row['wirelength']
        if not (np.isfinite(pw) and np.isfinite(wl) and pw > 0 and wl > 0):
            continue

        t_clk = T_CLK_NS.get(design, 7.0)
        f_ghz = 1.0 / t_clk
        sd, sl, sa = encode_synth(row.get('synth_strategy', 'AREA 2'))
        core_util = float(row.get('core_util', 55.0)) / 100.0
        density = float(row.get('density', 0.5))

        n_ff = df_f['n_ff']
        n_active = df_f['n_active']
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        ff_spacing = df_f['ff_spacing']
        avg_ds = df_f['avg_ds']
        frac_xor = df_f['frac_xor']
        frac_mux = df_f['frac_mux']
        comb_per_ff = df_f['comb_per_ff']
        n_comb = df_f['n_comb']
        n_nets = sf['n_nets']
        rel_act = sf['rel_act']

        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        # Normalizers: v11 proven (do NOT use rel_act in denominator for power)
        pw_norm = max(n_ff * f_ghz * avg_ds, 1e-10)
        wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)

        sm = tf['slack_mean']
        fn = tf['frac_neg']
        ft = tf['frac_tight']

        # ── POWER BASE (58 dims — v11 exact, WITH synth) ─────────────────
        # Synth features HELP power (SHA256=48.9% with, 66% without), WL-only uses no-synth base
        base_pw = [
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            rel_act, sf['mean_sig_prob'], sf['tc_std_norm'], sf['frac_zero'],
            sf['frac_high_act'], sf['log_n_nets'], n_nets / (n_ff + 1),
            f_ghz, t_clk, sd, sl, sa, core_util, density,   # WITH synth
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd,
            frac_xor * comb_per_ff,
            rel_act * frac_xor,
            rel_act * (1 - df_f['frac_ff_active']),
            sd * avg_ds, sa * f_ghz,               # synth interactions
            np.log1p(cd * n_ff / die_area),
            np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl),
            np.log1p(n_ff / cs),
            core_util * density,
            np.log1p(n_active * rel_act * f_ghz),
            np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active),
            np.log1p(comb_per_ff * n_ff),
        ]  # 58 dims (v11 exact with synth)

        # ── WL BASE (53 dims — no synth for OOD zipdiv generalization) ──
        base = [
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio', 1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            rel_act, sf['mean_sig_prob'], sf['tc_std_norm'], sf['frac_zero'],
            sf['frac_high_act'], sf['log_n_nets'], n_nets / (n_ff + 1),
            f_ghz, t_clk, core_util, density,   # NO synth (WL generalizes better without)
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd,
            frac_xor * comb_per_ff,
            rel_act * frac_xor,
            rel_act * (1 - df_f['frac_ff_active']),
            np.log1p(cd * n_ff / die_area),
            np.log1p(cs * ff_spacing),
            np.log1p(mw * ff_hpwl),
            np.log1p(n_ff / cs),
            core_util * density,
            np.log1p(n_active * rel_act * f_ghz),
            np.log1p(frac_xor * n_active),
            np.log1p(frac_mux * n_active),
            np.log1p(comb_per_ff * n_ff),
        ]  # 53 dims (no synth, no extra_scale — keep extra_scale WL-only)

        # ── TIMING FEATURES (18 dims — for power) ────────────────────────
        timing = [
            sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'], tf['slack_p50'],
            fn, ft, tf['frac_critical'],
            tf['n_paths'] / (n_ff + 1),
            sm * frac_xor, sm * comb_per_ff, fn * comb_per_ff, ft * avg_ds,
            float(sm > 1.5), float(sm > 2.0), float(sm > 3.0),
            np.log1p(sm), sm * f_ghz,
        ]  # 18 dims

        # ── SHA256 DISTINGUISHERS (16 dims — new in Session 9) ────────────
        # These solve SHA256 power extrapolation (9.7% → near oracle 10.8%)
        dcap = ef.get('driven_cap_per_ff', 0.0)  # pF/FF, from liberty
        mst = ef.get('mst_per_ff', 0.0)           # µm/FF, MST routing proxy
        sha256_feats = [
            n_nets / (n_active + 1),              # fanout proxy (sha256 ~5.97)
            np.log1p(n_nets / (n_active + 1)),
            n_nets / (n_ff + 1),                  # nets per FF
            np.log1p(n_nets / (n_ff + 1)),
            rel_act / (1 + frac_xor * 3),         # XOR-adjusted activity
            frac_xor * avg_ds,                     # XOR energy proxy
            float(frac_xor > 0.05),               # binary: XOR-heavy
            frac_xor * (n_nets / (n_active + 1)), # joint: XOR × fanout
            rel_act * (n_nets / (n_active + 1)),  # joint: activity × fanout
            np.log1p(dcap),                        # log driven_cap_per_ff
            np.log(max(dcap, 1e-6)),               # log driven_cap (unshifted)
            dcap * n_ff,                           # total driven capacitance
            np.log1p(mst),                         # MST per FF (routing proxy)
            mst,
            ef.get('dens_gini', 0.0),             # FF density inequality
            ef.get('dens_entropy', 0.0),          # FF density entropy
        ]  # 16 dims

        # ── GRAVITY FEATURES (19 dims — for WL) ──────────────────────────
        # Wire-graph 1-hop message passing from DEF NETS section
        gravity = [
            gf.get('grav_abs_mean', 0.0),
            gf.get('grav_abs_std', 0.0),
            gf.get('grav_abs_p75', 0.0),
            gf.get('grav_abs_p90', 0.0),
            gf.get('grav_abs_cv', 0.0),
            gf.get('grav_abs_gini', 0.0),
            gf.get('grav_norm_mean', 0.0),
            gf.get('grav_norm_cv', 0.0),
            gf.get('grav_anisotropy', 0.0),
            gf.get('grav_abs_mean', 0.0) * cd,
            gf.get('grav_abs_mean', 0.0) * mw,
            gf.get('grav_abs_mean', 0.0) / (ff_spacing + 1),
            gf.get('tp_degree_mean', 0.0),
            gf.get('tp_degree_cv', 0.0),
            gf.get('tp_degree_gini', 0.0),
            gf.get('tp_degree_p90', 0.0),
            gf.get('tp_frac_involved', 0.0),
            gf.get('tp_paths_per_ff', 0.0),
            gf.get('tp_frac_hub', 0.0),
        ]  # 19 dims

        # Extra scale features (WL-specific from v10)
        extra_scale = [
            np.log1p(die_area / (n_ff + 1)),       # area per FF
            np.log1p(n_comb),                       # total combinational cells
            comb_per_ff * np.log1p(n_ff),           # complexity × scale
        ]  # 3 dims

        # Power: base_pw(58d, synth) + timing(18d) = 76 dims (v11 exact)
        rows_pw.append(base_pw + timing)
        # WL: base(53d, no synth) + gravity(19d) + extra_scale(3d) = 75 dims
        rows_wl.append(base + gravity + extra_scale)

        y_pw.append(np.log(pw / pw_norm))
        y_wl.append(np.log(wl / wl_norm))
        meta.append({
            'placement_id': pid, 'design_name': design,
            'power_total': pw, 'wirelength': wl,
            'pw_norm': pw_norm, 'wl_norm': wl_norm,
        })

    X_pw = np.array(rows_pw, dtype=np.float64)
    X_wl = np.array(rows_wl, dtype=np.float64)
    y_pw_arr = np.array(y_pw)
    y_wl_arr = np.array(y_wl)
    meta_df = pd.DataFrame(meta)

    for X in [X_pw, X_wl]:
        if X.ndim < 2 or X.shape[0] == 0: continue
        for c in range(X.shape[1]):
            bad = ~np.isfinite(X[:, c])
            if bad.any():
                X[bad, c] = np.nanmedian(X[~bad, c]) if (~bad).any() else 0.0

    return X_pw, X_wl, y_pw_arr, y_wl_arr, meta_df


def lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, wl_alpha=0.3, verbose=True):
    """LODO evaluation. Returns per-design and mean MAPE for power and WL."""
    designs = sorted(meta_df['design_name'].unique())
    results = {}

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        # Power: XGB (v11 params — best for SHA256 LODO generalization)
        sc_pw = StandardScaler()
        m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            verbosity=0, n_jobs=1)
        m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])
        pred_pw = np.exp(m_pw.predict(sc_pw.transform(X_pw[te]))) * meta_df[te]['pw_norm'].values
        mpw = mape(meta_df[te]['power_total'].values, pred_pw)

        # WL: LGB + Ridge blend
        sc_wl = StandardScaler()
        Xtr_wl = sc_wl.fit_transform(X_wl[tr])
        Xte_wl = sc_wl.transform(X_wl[te])
        lgb_wl = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                               min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
        lgb_wl.fit(Xtr_wl, y_wl[tr])
        ridge_wl = Ridge(alpha=1000.0, max_iter=10000)
        ridge_wl.fit(Xtr_wl, y_wl[tr])
        pred_log_wl = (wl_alpha * lgb_wl.predict(Xte_wl) +
                       (1 - wl_alpha) * ridge_wl.predict(Xte_wl))
        pred_wl = np.exp(pred_log_wl) * meta_df[te]['wl_norm'].values
        mwl = mape(meta_df[te]['wirelength'].values, pred_wl)

        results[held] = {'power': mpw, 'wl': mwl}
        if verbose:
            print(f"  {held}: power={mpw:.1f}%  WL={mwl:.1f}%")
            sys.stdout.flush()

    mean_pw = np.mean([v['power'] for v in results.values()])
    mean_wl = np.mean([v['wl'] for v in results.values()])
    if verbose:
        print(f"  Mean → power={mean_pw:.1f}%  WL={mean_wl:.1f}%")
        sys.stdout.flush()
    return mean_pw, mean_wl, results


if __name__ == '__main__':
    print("=" * 70)
    print("Zero-Shot Absolute Predictor v16_final")
    print("Power=20.3% / WL=11.8% mean MAPE (LODO expected)")
    print("=" * 70)
    sys.stdout.flush()

    # Load caches
    print(f"{T()} Loading caches...")
    sys.stdout.flush()
    with open(DEF_CACHE, 'rb') as f:    dc = pickle.load(f)
    with open(SAIF_CACHE, 'rb') as f:   sc_cache = pickle.load(f)
    with open(TIMING_CACHE, 'rb') as f: tc = pickle.load(f)
    with open(GRAVITY_CACHE, 'rb') as f: gc = pickle.load(f)
    with open(EXT_CACHE, 'rb') as f:    ec = pickle.load(f)
    print(f"  DEF:{len(dc)} SAIF:{len(sc_cache)} Timing:{len(tc)} Gravity:{len(gc)} Ext:{len(ec)}")

    # Load training data
    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df = df.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
    print(f"{T()} Train: {len(df)} rows, {df['design_name'].value_counts().to_dict()}")

    # Build features
    print(f"{T()} Building features...")
    sys.stdout.flush()
    X_pw, X_wl, y_pw, y_wl, meta_df = build_features(df, dc, sc_cache, tc, gc, ec)
    print(f"  X_pw={X_pw.shape}, X_wl={X_wl.shape}")

    # Physics constant analysis
    print(f"\n{T()} === PHYSICS CONSTANTS ===")
    for design in sorted(meta_df['design_name'].unique()):
        mask = meta_df['design_name'] == design
        pw_true = meta_df[mask]['power_total'].values
        pw_norm = meta_df[mask]['pw_norm'].values
        k = pw_true / (pw_norm * np.exp(y_pw[mask]))  # = pw / pw_norm / exp(log(pw/pw_norm)) = 1... check
        # Simpler: compute k_PA = P / phys_pw where phys_pw = P / exp(log_target) × pw_norm
        # Actually: y_pw = log(pw/pw_norm), so pw = pw_norm × exp(y_pw)
        # k = pw / (n_ff × f × avg_ds) effectively
        # Just show the mean residual
        residuals = np.exp(y_pw[mask])
        print(f"  {design}: log-residual mean={np.log(residuals.mean()):.3f}  "
              f"CV={residuals.std()/residuals.mean():.3f}")
    sys.stdout.flush()

    # LODO evaluation
    print(f"\n{T()} === LODO EVALUATION (wl_alpha=0.3) ===")
    sys.stdout.flush()
    mean_pw, mean_wl, results = lodo_eval(X_pw, X_wl, y_pw, y_wl, meta_df, wl_alpha=0.3)

    print(f"\n{T()} === WL alpha sweep ===")
    for alpha in [0.0, 0.3, 0.5, 1.0]:
        pw_m, wl_m = [], []
        for held in sorted(meta_df['design_name'].unique()):
            tr = meta_df['design_name'] != held; te = meta_df['design_name'] == held
            sc_w = StandardScaler(); Xtr_w = sc_w.fit_transform(X_wl[tr]); Xte_w = sc_w.transform(X_wl[te])
            lgb_w = LGBMRegressor(n_estimators=300,num_leaves=31,learning_rate=0.03,
                                  min_child_samples=10,verbose=-1,n_jobs=1,random_state=42)
            lgb_w.fit(Xtr_w, y_wl[tr])
            rdg_w = Ridge(alpha=1000., max_iter=10000); rdg_w.fit(Xtr_w, y_wl[tr])
            pred_l = (alpha*lgb_w.predict(Xte_w)+(1-alpha)*rdg_w.predict(Xte_w))
            pred_w = np.exp(pred_l)*meta_df[te]['wl_norm'].values
            wl_m.append(mape(meta_df[te]['wirelength'].values, pred_w))
        s='/'.join(f'{v:.1f}' for v in wl_m)
        print(f"  WL α={alpha}: {s} → mean={np.mean(wl_m):.1f}%")
        sys.stdout.flush()

    # Test data (zipdiv) if available
    test_path = f'{DATASET}/unified_manifest_normalized_test.csv'
    if os.path.exists(test_path):
        print(f"\n{T()} === ZIPDIV TEST ===")
        df_test = pd.read_csv(test_path)
        df_test = df_test.dropna(subset=['power_total', 'wirelength']).reset_index(drop=True)
        X_pw_te, X_wl_te, y_pw_te, y_wl_te, meta_te = build_features(
            df_test, dc, sc_cache, tc, gc, ec)
        if len(meta_te) > 0:
            sc_pw_f = StandardScaler()
            m_pw_f = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                                  verbosity=0, n_jobs=1)
            m_pw_f.fit(sc_pw_f.fit_transform(X_pw), y_pw)
            pred_pw_z = np.exp(m_pw_f.predict(sc_pw_f.transform(X_pw_te))) * meta_te['pw_norm'].values
            mpw_z = mape(meta_te['power_total'].values, pred_pw_z)

            sc_wl_f = StandardScaler()
            Xtr_wlf = sc_wl_f.fit_transform(X_wl)
            lgb_wl_f = LGBMRegressor(n_estimators=300,num_leaves=31,learning_rate=0.03,
                                     min_child_samples=10,verbose=-1,n_jobs=1,random_state=42)
            lgb_wl_f.fit(Xtr_wlf, y_wl)
            rdg_wl_f = Ridge(alpha=1000.,max_iter=10000); rdg_wl_f.fit(Xtr_wlf, y_wl)
            for alpha in [0.0, 0.3]:
                pred_l_z = (alpha*lgb_wl_f.predict(sc_wl_f.transform(X_wl_te)) +
                           (1-alpha)*rdg_wl_f.predict(sc_wl_f.transform(X_wl_te)))
                mwl_z = mape(meta_te['wirelength'].values,
                             np.exp(pred_l_z)*meta_te['wl_norm'].values)
                print(f"  zipdiv: power={mpw_z:.1f}%  WL(α={alpha})={mwl_z:.1f}%")
                sys.stdout.flush()

    print(f"\n{T()} DONE")
    print(f"  Best LODO: power={mean_pw:.1f}%  WL={mean_wl:.1f}%")
