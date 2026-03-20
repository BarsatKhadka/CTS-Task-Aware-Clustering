"""
multiobjective.py — 4-target predictor + Pareto-optimal CTS knob search

Targets:
  1. power_total (W)     — dynamic power
  2. wirelength (µm)     — clock tree routing length
  3. skew_setup (ns)     — setup timing skew
  4. hold_vio_count      — #paths with hold violations (pre-fixing)

Application: Given a new placement, evaluate 10,000 knob combos in <5s
using surrogate models, return the Pareto-optimal (power, skew, hold_vio) frontier.

Usage:
  python3 synthesis_best/multiobjective.py           # full LODO eval + demo
  python3 synthesis_best/multiobjective.py --demo    # Pareto optimizer demo only
"""

import os, sys, time, pickle, argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

BASE = '/home/rain/CTS-Task-Aware-Clustering'
sys.path.insert(0, f'{BASE}/synthesis_best')
from final_synthesis import build_all_features, per_placement_normalize, mape, mae, encode_synth

T_CLK_NS = {'aes':7.0,'picorv32':5.0,'sha256':9.0,'ethmac':9.0,'zipdiv':5.0}

# Known knob positions per feature vector (verified against actual feature vectors)
# Format: (log_indices [cd,cs,mw,bd], raw_indices [cd,cs,mw,bd])
KNOB_IDX = {
    'pw': {'log': [36,37,38,39], 'raw': [40,41,42,43],
           'inter': [(49, 'cd'), (50, 'cs'), (51, 'mw'), (52, 'cs_inv')]},
    'wl': {'log': [33,34,35,36], 'raw': [37,38,39,40],
           'inter': [(44, 'cd'), (45, 'cs'), (46, 'mw'), (47, 'cs_inv'),
                     (82, 'cd_rsmt'), (83, 'cd_rudy')]},
    'sk': {'log': [22,23,24,25], 'raw': [26,27,28,29]},
}


# ─── hold_vio feature builder ─────────────────────────────────────────────────

def build_hold_features(df_in, dc, sc_cache, tc, gc, nc=None):
    """66-dim features for hold_vio_count prediction."""
    rows, y_hv, meta = [], [], []
    for _, row in df_in.iterrows():
        pid = row['placement_id']; design = row['design_name']
        df_f = dc.get(pid); sf = sc_cache.get(pid); tf = tc.get(pid)
        if not df_f or not sf or not tf: continue
        gf = gc.get(pid, {}); nf = (nc or {}).get(pid, {})
        hv = row.get('hold_vio_count', np.nan)
        if not np.isfinite(hv): continue

        t_clk = T_CLK_NS.get(design, 7.0); f_ghz = 1.0/t_clk
        core_util = float(row.get('core_util',55.0))/100.0
        density   = float(row.get('density',0.5))
        n_ff=df_f['n_ff']; n_active=df_f['n_active']; die_area=df_f['die_area']
        ff_hpwl=df_f['ff_hpwl']; ff_spacing=df_f['ff_spacing']; avg_ds=df_f['avg_ds']
        frac_xor=df_f['frac_xor']; frac_mux=df_f['frac_mux']
        comb_per_ff=df_f['comb_per_ff']; n_comb=df_f['n_comb']
        n_nets=sf['n_nets']; rel_act=sf['rel_act']
        cd=row['cts_cluster_dia']; cs=row['cts_cluster_size']
        mw=row['cts_max_wire']; bd=row['cts_buf_dist']
        sm=tf['slack_mean']; fn=tf['frac_neg']; ft=tf['frac_tight']

        # Base geometry — same 53d as WL base (no synth)
        base = [
            np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
            df_f['die_aspect'], float(row.get('aspect_ratio',1.0)),
            df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
            frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
            df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
            avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            rel_act, sf['mean_sig_prob'], sf['tc_std_norm'], sf['frac_zero'],
            sf['frac_high_act'], sf['log_n_nets'], n_nets/(n_ff+1),
            f_ghz, t_clk, core_util, density,
            np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd), cd, cs, mw, bd,  # [33:41]
            frac_xor*comb_per_ff, rel_act*frac_xor, rel_act*(1-df_f['frac_ff_active']),
            np.log1p(cd*n_ff/die_area), np.log1p(cs*ff_spacing),
            np.log1p(mw*ff_hpwl), np.log1p(n_ff/cs), core_util*density,
            np.log1p(n_active*rel_act*f_ghz), np.log1p(frac_xor*n_active),
            np.log1p(frac_mux*n_active), np.log1p(comb_per_ff*n_ff),
        ]
        # Hold-specific physics: buf_dist is the primary hold driver
        hold_phys = [
            bd/(mw+1),                      # buf_dist/max_wire: higher → more hold violations
            np.log1p(bd*n_ff/die_area),     # buffer insertion density
            bd*comb_per_ff,                 # buf stages × logic depth
            cd/(bd+1),                      # cluster_dia/buf_dist
            sm*bd,                          # timing slack × buf_dist
            fn*comb_per_ff,                 # fraction_neg_slack × logic depth
            ft*avg_ds,                      # fraction_tight × drive_strength
            np.log1p(bd*ff_spacing),        # buf_dist × FF spacing
            mw/(bd+1),                      # max_wire/buf_dist ratio
        ]
        rsmt_t = float((nf.get('rsmt_total',0.0) or 0.0))
        net_feats = [
            np.log1p(rsmt_t),
            rsmt_t/max(n_ff*np.sqrt(die_area),1e-3),
            float(nf.get('rudy_mean',0.0) or 0.0),
            float(nf.get('rudy_p90',0.0) or 0.0),
        ]
        rows.append(base + hold_phys + net_feats)   # 53+9+4 = 66 dims
        y_hv.append(np.log1p(hv))
        meta.append({'pid':pid, 'design':design, 'hold_vio':hv})

    return np.array(rows, dtype=np.float32), np.array(y_hv), pd.DataFrame(meta)


# ─── LODO helpers ─────────────────────────────────────────────────────────────

def lodo_hold(X, y, meta_df):
    """LODO evaluation for hold_vio_count."""
    designs = sorted(meta_df['design'].unique())
    results = {}
    for held in designs:
        tr = meta_df['design'] != held
        te = meta_df['design'] == held
        y_norm = np.zeros_like(y); mu_arr = np.zeros_like(y); sig_arr = np.ones_like(y)
        for pid, grp in meta_df.groupby('pid'):
            idx = grp.index.values; vals = y[idx]
            mu=vals.mean(); sig=max(vals.std(), max(abs(mu)*0.01, 1e-4))
            y_norm[idx]=(vals-mu)/sig; mu_arr[idx]=mu; sig_arr[idx]=sig
        sc = StandardScaler()
        m = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                          min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
        m.fit(sc.fit_transform(X[tr]), y_norm[tr])
        pred_z = m.predict(sc.transform(X[te]))
        pred_hv = np.expm1(np.clip(pred_z*sig_arr[te]+mu_arr[te], 0, 20))
        actual_hv = np.expm1(y[te])
        err = np.mean(np.abs(pred_hv-actual_hv)/(actual_hv+1))*100
        results[held] = err
    return results


# ─── Pareto frontier (vectorized) ────────────────────────────────────────────

def pareto_front(costs):
    """
    Returns boolean mask of non-dominated solutions.
    costs: (N, M) array — minimize all objectives.
    Uses chunked O(N²) numpy — fast for N≤5000.
    """
    n = costs.shape[0]
    # Normalize to [0,1] for numerical stability
    lo = costs.min(0); rng = (costs.max(0) - lo) + 1e-10
    c = (costs - lo) / rng

    is_dominated = np.zeros(n, dtype=bool)
    chunk = 500
    for i in range(0, n, chunk):
        ci = c[i:i+chunk]                          # (chunk, M)
        # dominated[j] = any k where c[k] <= c[j] and c[k] < c[j] in at least one dim
        # ci[j] dominated by c[k]: all(c[k] <= ci[j]) and any(c[k] < ci[j])
        dominated_by = (
            np.all(c[:, None, :] <= ci[None, :, :] + 1e-9, axis=2) &   # (N, chunk)
            np.any(c[:, None, :] <  ci[None, :, :] - 1e-9, axis=2)
        )
        # Exclude self-domination
        np.fill_diagonal(dominated_by[i:i+chunk, :].T, False)  # won't work cleanly — handle below
        # j (in chunk) is dominated if any row k != j has dominated_by[k,j]=True
        for j in range(len(ci)):
            row_j = i + j
            col = dominated_by[:, j].copy()
            col[row_j] = False
            if col.any():
                is_dominated[row_j] = True

    return ~is_dominated


# ─── Knob optimizer ──────────────────────────────────────────────────────────

def optimize_knobs(base_row, aux, models, scalers, targets,
                   pw_norm, wl_norm, sk_mu, sk_sig,
                   n_samples=5000, constraints=None):
    """
    Sweep n_samples random knob combinations and return Pareto frontier.

    base_row: feature vector for one placement (with arbitrary knob values)
    aux: dict with {n_ff, die_area, ff_hpwl, ff_spacing, rsmt_t, rudy_mean} for interaction patching
    targets: list of targets to optimize ['power','wl','skew','hold_vio']
    """
    rng = np.random.default_rng(42)
    cd_arr = rng.uniform(35, 70,   n_samples)
    cs_arr = rng.integers(12, 31,  n_samples).astype(float)
    mw_arr = rng.uniform(130, 280, n_samples)
    bd_arr = rng.uniform(70,  150, n_samples)

    results = {}
    for feat_key, (model, scaler) in models.items():
        x = base_row[feat_key]
        ki = KNOB_IDX[feat_key]
        X = np.tile(x, (n_samples, 1))

        # Patch log knobs
        for idx, val in zip(ki['log'], [cd_arr, cs_arr, mw_arr, bd_arr]):
            X[:, idx] = np.log1p(val)
        # Patch raw knobs
        for idx, val in zip(ki['raw'], [cd_arr, cs_arr, mw_arr, bd_arr]):
            X[:, idx] = val
        # Patch interaction terms
        for (idx, kind) in ki.get('inter', []):
            if kind == 'cd':
                X[:, idx] = np.log1p(cd_arr * aux['n_ff'] / aux['die_area'])
            elif kind == 'cs':
                X[:, idx] = np.log1p(cs_arr * aux['ff_spacing'])
            elif kind == 'mw':
                X[:, idx] = np.log1p(mw_arr * aux['ff_hpwl'])
            elif kind == 'cs_inv':
                X[:, idx] = np.log1p(aux['n_ff'] / cs_arr)
            elif kind == 'cd_rsmt':
                X[:, idx] = aux['rsmt_t'] * cd_arr / max(aux['n_ff'] * aux['die_area'], 1.0)
            elif kind == 'cd_rudy':
                X[:, idx] = aux['rudy_mean'] * cd_arr

        X_s = scaler.transform(X.astype(np.float32))
        results[feat_key] = model.predict(X_s)

    pred_pw  = np.exp(results['pw']) * pw_norm
    pred_wl  = np.exp(0.3*results['wl_lgb'] + 0.7*results['wl_ridge']) * wl_norm
    pred_sk_z = results['sk']
    pred_sk  = pred_sk_z * sk_sig + sk_mu
    pred_hv_z = results['hv']
    pred_hv  = np.expm1(np.clip(pred_hv_z * aux['hv_sig'] + aux['hv_mu'], 0, 20))

    df = pd.DataFrame({
        'cd': cd_arr, 'cs': cs_arr.astype(int),
        'mw': mw_arr.round(0), 'bd': bd_arr.round(0),
        'power_mW': pred_pw * 1000,
        'wl_mm':    pred_wl / 1000,
        'skew_ns':  pred_sk,
        'hold_vio': pred_hv,
    })

    if constraints:
        if 'skew_max_ns' in constraints:
            df = df[df['skew_ns'] <= constraints['skew_max_ns']]
        if 'hold_vio_max' in constraints:
            df = df[df['hold_vio'] <= constraints['hold_vio_max']]

    if len(df) < 5:
        print(f"  WARNING: only {len(df)} solutions satisfy constraints")
        return df

    costs = df[['power_mW', 'skew_ns', 'hold_vio']].values
    pareto_mask = pareto_front(costs)
    df['pareto'] = pareto_mask
    return df.sort_values('power_mW').reset_index(drop=True)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='Run Pareto demo only')
    args = parser.parse_args()

    print("=" * 70)
    print("Multi-Objective CTS Predictor — 4 Targets + Pareto Optimizer")
    print("=" * 70)

    # Load caches
    with open(f'{BASE}/absolute_v7_def_cache.pkl','rb') as f:    dc = pickle.load(f)
    with open(f'{BASE}/absolute_v7_saif_cache.pkl','rb') as f:   sc_cache = pickle.load(f)
    with open(f'{BASE}/absolute_v7_timing_cache.pkl','rb') as f: tc = pickle.load(f)
    with open(f'{BASE}/skew_spatial_cache.pkl','rb') as f:       skc = pickle.load(f)
    with open(f'{BASE}/absolute_v10_gravity_cache.pkl','rb') as f: gc = pickle.load(f)
    with open(f'{BASE}/absolute_v13_extended_cache.pkl','rb') as f: ec = pickle.load(f)
    nc = {}
    if os.path.exists(f'{BASE}/net_features_cache.pkl'):
        with open(f'{BASE}/net_features_cache.pkl','rb') as f: nc = pickle.load(f)

    df_all = pd.read_csv(f'{BASE}/dataset_with_def/unified_manifest_normalized.csv')
    df_all = df_all[df_all['design_name'] != 'zipdiv']
    df_all = df_all.dropna(subset=['power_total','wirelength','skew_setup','hold_vio_count'])
    df_all = df_all.reset_index(drop=True)
    designs = sorted(df_all['design_name'].unique())
    print(f"{T()} n={len(df_all)}, designs={designs}")

    # Build features
    print(f"{T()} Building power/WL/skew features...")
    X_pw, X_wl, X_sk, y_pw, y_wl, y_sk, meta_df = build_all_features(
        df_all, dc, sc_cache, tc, skc, gc, ec, nc)
    print(f"  X_pw={X_pw.shape}, X_wl={X_wl.shape}, X_sk={X_sk.shape}")

    print(f"{T()} Building hold_vio features...")
    X_hv, y_hv, meta_hv = build_hold_features(df_all, dc, sc_cache, tc, gc, nc)
    print(f"  X_hv={X_hv.shape}")
    sys.stdout.flush()

    # ── LODO evaluation ───────────────────────────────────────────────────────
    print(f"\n{T()} === LODO: All 4 Targets ===")
    pw_m, wl_m, sk_m, hv_m = {}, {}, {}, {}
    hv_m = lodo_hold(X_hv, y_hv, meta_hv)

    for held in designs:
        tr = meta_df['design_name'] != held
        te = meta_df['design_name'] == held

        sc_pw = StandardScaler()
        m_pw = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8, random_state=42,
                            verbosity=0, n_jobs=1)
        m_pw.fit(sc_pw.fit_transform(X_pw[tr]), y_pw[tr])
        pred_pw = np.exp(m_pw.predict(sc_pw.transform(X_pw[te]))) * meta_df[te]['pw_norm'].values
        pw_m[held] = mape(meta_df[te]['power_total'].values, pred_pw)

        sc_wl = StandardScaler()
        Xwl_tr = sc_wl.fit_transform(X_wl[tr]); Xwl_te = sc_wl.transform(X_wl[te])
        lgb_wl = LGBMRegressor(n_estimators=700, num_leaves=31, learning_rate=0.03,
                               min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
        lgb_wl.fit(Xwl_tr, y_wl[tr])
        rdg_wl = Ridge(alpha=1000.0, max_iter=10000)
        rdg_wl.fit(Xwl_tr, y_wl[tr])
        pred_wl = np.exp(0.3*lgb_wl.predict(Xwl_te)+0.7*rdg_wl.predict(Xwl_te)) * meta_df[te]['wl_norm'].values
        wl_m[held] = mape(meta_df[te]['wirelength'].values, pred_wl)

        y_sk_norm, mu_sk, sig_sk = per_placement_normalize(y_sk, meta_df)
        sc_sk = StandardScaler()
        m_sk = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                             min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
        m_sk.fit(sc_sk.fit_transform(X_sk[tr]), y_sk_norm[tr])
        pred_sk = m_sk.predict(sc_sk.transform(X_sk[te])) * sig_sk[te] + mu_sk[te]
        sk_m[held] = mae(y_sk[te], pred_sk)

    print(f"\n  {'Design':>10} | {'Power':>7} | {'WL':>7} | {'Skew':>8} | {'HoldVio':>9}")
    print(f"  {'-'*10}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*9}")
    for d in designs:
        print(f"  {d:>10} | {pw_m[d]:>6.1f}% | {wl_m[d]:>6.1f}% | "
              f"{sk_m[d]:>7.4f} | {hv_m.get(d,0):>8.1f}%")
    print(f"  {'Mean':>10} | {np.mean(list(pw_m.values())):>6.1f}% | "
          f"{np.mean(list(wl_m.values())):>6.1f}% | "
          f"{np.mean(list(sk_m.values())):>7.4f} | "
          f"{np.mean(list(hv_m.values())):>8.1f}%")
    sys.stdout.flush()

    # ── Train production models (all data) ────────────────────────────────────
    print(f"\n{T()} Training production models on all data...")

    sc_pw_f = StandardScaler()
    m_pw_f = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                           verbosity=0, n_jobs=1)
    m_pw_f.fit(sc_pw_f.fit_transform(X_pw), y_pw)

    sc_wl_f = StandardScaler()
    X_wl_s = sc_wl_f.fit_transform(X_wl)
    lgb_wl_f = LGBMRegressor(n_estimators=700, num_leaves=31, learning_rate=0.03,
                              min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
    lgb_wl_f.fit(X_wl_s, y_wl)
    rdg_wl_f = Ridge(alpha=1000.0, max_iter=10000)
    rdg_wl_f.fit(X_wl_s, y_wl)

    y_sk_norm_f, mu_sk_f, sig_sk_f = per_placement_normalize(y_sk, meta_df)
    sc_sk_f = StandardScaler()
    m_sk_f = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                            min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
    m_sk_f.fit(sc_sk_f.fit_transform(X_sk), y_sk_norm_f)

    y_hv_norm = np.zeros_like(y_hv); mu_hv_f = np.zeros_like(y_hv); sig_hv_f = np.ones_like(y_hv)
    for pid, grp in meta_hv.groupby('pid'):
        idx=grp.index.values; vals=y_hv[idx]
        mu=vals.mean(); sig=max(vals.std(), max(abs(mu)*0.01,1e-4))
        y_hv_norm[idx]=(vals-mu)/sig; mu_hv_f[idx]=mu; sig_hv_f[idx]=sig
    sc_hv_f = StandardScaler()
    m_hv_f = LGBMRegressor(n_estimators=300, num_leaves=31, learning_rate=0.03,
                            min_child_samples=10, verbose=-1, n_jobs=1, random_state=42)
    m_hv_f.fit(sc_hv_f.fit_transform(X_hv), y_hv_norm)
    print(f"  {T()} Done")

    # Save 4-target model
    save_path = f'{BASE}/synthesis_best/saved_models/cts_predictor_4target.pkl'
    os.makedirs(f'{BASE}/synthesis_best/saved_models', exist_ok=True)
    with open(save_path,'wb') as f:
        pickle.dump({
            'model_power': m_pw_f, 'scaler_power': sc_pw_f,
            'model_wl_lgb': lgb_wl_f, 'model_wl_ridge': rdg_wl_f,
            'scaler_wl': sc_wl_f, 'wl_blend_alpha': 0.3,
            'model_skew': m_sk_f, 'scaler_skew': sc_sk_f,
            'model_hold_vio': m_hv_f, 'scaler_hold_vio': sc_hv_f,
            'lodo': {'power':pw_m,'wl':wl_m,'skew':sk_m,'hold_vio':hv_m},
        }, f)
    print(f"  Saved → {save_path} ({os.path.getsize(save_path)/1e6:.1f} MB)")
    sys.stdout.flush()

    # ── Pareto optimizer demo ─────────────────────────────────────────────────
    print(f"\n{T()} === PARETO OPTIMIZER DEMO ===")
    print("  Concept: given one placement, find optimal CTS knobs in milliseconds")
    print("  vs OpenROAD CTS which takes hours for even a fraction of this search\n")

    # Pick an AES placement and use its 10 actual CTS runs as reference
    demo_design = 'aes'
    pid_demo = meta_df[meta_df['design_name']==demo_design]['placement_id'].iloc[0]
    run_idx = meta_df[meta_df['placement_id']==pid_demo].index

    # Actual outcomes for these 10 runs
    actual_rows = df_all.loc[run_idx]
    print(f"  Placement: {pid_demo}  ({len(run_idx)} actual CTS runs)")
    print(f"  Actual outcomes (subset of true CTS runs):")
    print(f"  {'cd':>5} {'cs':>4} {'mw':>5} {'bd':>5} | "
          f"{'Power(mW)':>10} {'WL(mm)':>8} {'Skew(ns)':>9} {'HoldVio':>8}")
    print(f"  {'-'*5}-{'-'*4}-{'-'*5}-{'-'*5}-+-{'-'*10}-{'-'*8}-{'-'*9}-{'-'*8}")
    for _, r in actual_rows.iterrows():
        print(f"  {r.cts_cluster_dia:>5.0f} {r.cts_cluster_size:>4.0f} "
              f"{r.cts_max_wire:>5.0f} {r.cts_buf_dist:>5.0f} | "
              f"{r.power_total*1000:>9.2f}  {r.wirelength/1000:>7.1f}  "
              f"{r.skew_setup:>8.4f}  {r.hold_vio_count:>8.0f}")

    # Now sweep 5000 knob combos using surrogate
    t_start = time.time()
    idx0 = run_idx[0]
    hv_idx0 = meta_hv[meta_hv['pid']==pid_demo].index[0]

    aux = {
        'n_ff':    dc[pid_demo]['n_ff'],
        'die_area':dc[pid_demo]['die_area'],
        'ff_hpwl': dc[pid_demo]['ff_hpwl'],
        'ff_spacing':dc[pid_demo]['ff_spacing'],
        'rsmt_t':  float((nc.get(pid_demo,{}).get('rsmt_total',0.0) or 0.0)),
        'rudy_mean':float((nc.get(pid_demo,{}).get('rudy_mean',0.0) or 0.0)),
        'hv_mu':   mu_hv_f[hv_idx0],
        'hv_sig':  sig_hv_f[hv_idx0],
    }

    base_rows = {
        'pw': X_pw[idx0].copy(),
        'wl': X_wl[idx0].copy(),
        'sk': X_sk[idx0].copy(),
        'hv': X_hv[hv_idx0].copy(),
    }
    pw_norm_v = meta_df.loc[idx0, 'pw_norm']
    wl_norm_v = meta_df.loc[idx0, 'wl_norm']
    sk_mu_v   = mu_sk_f[idx0]
    sk_sig_v  = sig_sk_f[idx0]

    models_dict = {
        'pw':       (m_pw_f,  sc_pw_f),
        'wl_lgb':   (lgb_wl_f, sc_wl_f),
        'wl_ridge': (rdg_wl_f, sc_wl_f),
        'sk':       (m_sk_f,  sc_sk_f),
        'hv':       (m_hv_f,  sc_hv_f),
    }

    n_samples = 5000
    rng = np.random.default_rng(42)
    cd_arr = rng.uniform(35, 70,   n_samples)
    cs_arr = rng.integers(12, 31,  n_samples).astype(float)
    mw_arr = rng.uniform(130, 280, n_samples)
    bd_arr = rng.uniform(70,  150, n_samples)

    def patch(x, ki, cd, cs, mw, bd):
        X = np.tile(x, (n_samples,1)).astype(np.float64)
        for li, v in zip(ki['log'], [cd,cs,mw,bd]): X[:,li] = np.log1p(v)
        for ri, v in zip(ki['raw'], [cd,cs,mw,bd]): X[:,ri] = v
        for (ii, kind) in ki.get('inter',[]):
            if kind=='cd':      X[:,ii] = np.log1p(cd*aux['n_ff']/aux['die_area'])
            elif kind=='cs':    X[:,ii] = np.log1p(cs*aux['ff_spacing'])
            elif kind=='mw':    X[:,ii] = np.log1p(mw*aux['ff_hpwl'])
            elif kind=='cs_inv':X[:,ii] = np.log1p(aux['n_ff']/cs)
            elif kind=='cd_rsmt':X[:,ii] = aux['rsmt_t']*cd/max(aux['n_ff']*aux['die_area'],1)
            elif kind=='cd_rudy':X[:,ii] = aux['rudy_mean']*cd
        return X

    # Patch hold_vio feature (same knob indices as wl base)
    hv_ki = {'log':[33,34,35,36],'raw':[37,38,39,40],
              'inter':[(44,'cd'),(45,'cs'),(46,'mw'),(47,'cs_inv')]}

    Xpw = patch(base_rows['pw'], KNOB_IDX['pw'], cd_arr,cs_arr,mw_arr,bd_arr)
    Xwl = patch(base_rows['wl'], KNOB_IDX['wl'], cd_arr,cs_arr,mw_arr,bd_arr)
    Xsk = patch(base_rows['sk'], KNOB_IDX['sk'], cd_arr,cs_arr,mw_arr,bd_arr)
    Xhv = patch(base_rows['hv'], hv_ki,           cd_arr,cs_arr,mw_arr,bd_arr)

    pred_pw  = np.exp(m_pw_f.predict(sc_pw_f.transform(Xpw))) * pw_norm_v
    Xwl_s    = sc_wl_f.transform(Xwl)
    pred_wl  = np.exp(0.3*lgb_wl_f.predict(Xwl_s)+0.7*rdg_wl_f.predict(Xwl_s)) * wl_norm_v
    pred_sk  = m_sk_f.predict(sc_sk_f.transform(Xsk)) * sk_sig_v + sk_mu_v
    pred_hv_z= m_hv_f.predict(sc_hv_f.transform(Xhv))
    pred_hv  = np.expm1(np.clip(pred_hv_z*aux['hv_sig']+aux['hv_mu'], 0, 20))

    t_elapsed_ms = (time.time()-t_start)*1000

    df_sweep = pd.DataFrame({
        'cd':cd_arr,'cs':cs_arr.astype(int),'mw':mw_arr.round(0),'bd':bd_arr.round(0),
        'power_mW': pred_pw*1000, 'wl_mm': pred_wl/1000,
        'skew_ns': pred_sk, 'hold_vio': pred_hv,
    })

    # Pareto on (power, skew, hold_vio) — feasible only
    df_feas = df_sweep[df_sweep['skew_ns'] > 0].copy()
    costs = df_feas[['power_mW','skew_ns','hold_vio']].values
    pareto_mask = pareto_front(costs)
    df_feas['pareto'] = pareto_mask
    pareto = df_feas[df_feas['pareto']].sort_values('power_mW')

    print(f"\n  Surrogate sweep: {n_samples} combos in {t_elapsed_ms:.0f}ms")
    print(f"  (OpenROAD CTS runtime: ~{n_samples*2//60} hours for same search)")
    print(f"  Pareto-optimal solutions: {len(pareto)} / {n_samples}")
    print(f"\n  Top-10 Pareto solutions (sorted by power):")
    print(f"  {'cd':>5} {'cs':>4} {'mw':>5} {'bd':>5} | "
          f"{'Power(mW)':>10} {'WL(mm)':>8} {'Skew(ns)':>9} {'HoldVio':>8}")
    print(f"  {'-'*5}-{'-'*4}-{'-'*5}-{'-'*5}-+-{'-'*10}-{'-'*8}-{'-'*9}-{'-'*8}")
    for _, r in pareto.head(10).iterrows():
        print(f"  {r.cd:>5.0f} {r.cs:>4.0f} {r.mw:>5.0f} {r.bd:>5.0f} | "
              f"{r.power_mW:>9.2f}  {r.wl_mm:>7.1f}  {r.skew_ns:>8.4f}  {r.hold_vio:>8.0f}")

    best_pw = pareto.sort_values('power_mW').iloc[0]
    best_sk = pareto.sort_values('skew_ns').iloc[0]
    best_hv = pareto.sort_values('hold_vio').iloc[0]
    print(f"\n  Best power  knob: cd={best_pw.cd:.0f} cs={best_pw.cs:.0f} "
          f"mw={best_pw.mw:.0f} bd={best_pw.bd:.0f} → {best_pw.power_mW:.2f}mW")
    print(f"  Best skew   knob: cd={best_sk.cd:.0f} cs={best_sk.cs:.0f} "
          f"mw={best_sk.mw:.0f} bd={best_sk.bd:.0f} → {best_sk.skew_ns:.4f}ns")
    print(f"  Best hold   knob: cd={best_hv.cd:.0f} cs={best_hv.cs:.0f} "
          f"mw={best_hv.mw:.0f} bd={best_hv.bd:.0f} → {best_hv.hold_vio:.0f} violations")

    # zipdiv check
    zip_pids = [k for k in dc.keys() if 'zipdiv' in k]
    print(f"\n  zipdiv in DEF cache: {len(zip_pids)} placements")
    if zip_pids:
        zf = dc[zip_pids[0]]
        print(f"  zipdiv stats: n_ff={zf['n_ff']}, die_area={zf['die_area']:.0f}µm²")
        print(f"  → Run: python3 synthesis_best/multiobjective.py --demo to optimize zipdiv")
        print(f"  → First add zipdiv CTS outcomes to unified_manifest_normalized.csv for eval")

    print(f"\n[{time.time()-t0:.1f}s] DONE")
