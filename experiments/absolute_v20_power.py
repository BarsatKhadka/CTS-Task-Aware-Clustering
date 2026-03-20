"""
absolute_v20_power.py — Power optimization: reduce K-shot requirement

Core insight: SHA256 power fails because rel_act=0.104 is 2x OOD vs training [0.035-0.051].
Tree model extrapolates badly.

Strategies tested:
  A. rel_act clip: in each LODO fold, clip rel_act at training-set max.
     SHA256 (0.104) gets clipped to ~0.051 → inside training distribution.
     Preserves the feature but removes extrapolation.

  B. rel_act rank-norm: replace rel_act with its rank in training set (0-1).
     SHA256 gets rank=1.0 (highest), bounded. Model interpolates instead of extrapolating.

  C. No rel_act: remove rel_act entirely. Let frac_xor/comb_per_ff predict k_PA.
     SHA256's circuit features (frac_xor) are IN training range → model might generalize.

  D. Log(rel_act+ε) in normalization: use log to compress the OOD range.
     log(0.104)/log(0.051) ≈ 1.4 → still somewhat OOD but less extreme.

  E. Power norm v2: instead of n_ff × f × avg_ds, use n_ff × f × driven_cap_per_ff.
     driven_cap captures per-cell load better than avg_ds (drive strength).

  F. Smarter K-shot: use median instead of mean for k_hat (more robust to outliers).
     Also: stratified K-shot (pick samples spanning knob space).

Goal: get mean power MAPE <10% with K≤3 (instead of K=20).
"""

import os, sys, time, pickle
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

t0 = time.time()
def T(): return f"[{time.time()-t0:.1f}s]"

BASE = '/home/rain/CTS-Task-Aware-Clustering'
DATASET = f'{BASE}/dataset_with_def'

DEF_CACHE    = f'{BASE}/absolute_v7_def_cache.pkl'
SAIF_CACHE   = f'{BASE}/absolute_v7_saif_cache.pkl'
TIMING_CACHE = f'{BASE}/absolute_v7_timing_cache.pkl'
GRAVITY_CACHE = f'{BASE}/absolute_v10_gravity_cache.pkl'
EXT_CACHE    = f'{BASE}/absolute_v13_extended_cache.pkl'
T_CLK_NS = {'aes':7.,'picorv32':5.,'sha256':9.,'ethmac':9.,'zipdiv':5.}


def mape(a, b):
    return np.mean(np.abs(b-a)/(np.abs(a)+1e-12))*100

def encode_synth(s):
    if pd.isna(s): return 0.5,2.0,0.5
    s=str(s).upper(); sd=1. if 'DELAY' in s else 0.
    try: lv=float(s.split()[-1])
    except: lv=2.
    return sd,lv,sd*lv/4.


def build_power_features(df_in, dc, sc_cache, tc, ec,
                         rel_act_mode='original',  # original|clip|rank|none|log
                         tr_rel_act_max=None):      # set in LODO fold
    """
    Build power feature matrix.
    rel_act_mode controls how rel_act is treated:
      'original': use raw rel_act (v16_final behavior)
      'clip':     clip at training-set max (data-driven, set tr_rel_act_max)
      'rank':     replace with rank in [0,1] across training set (needs pre-computed ranks)
      'none':     remove rel_act entirely
      'log':      log(rel_act) — compresses OOD range
    """
    rows, y, meta = [], [], []
    raw_rel_acts = []

    for _, row in df_in.iterrows():
        pid=row['placement_id']; design=row['design_name']
        df_f=dc.get(pid); sf=sc_cache.get(pid); tf=tc.get(pid); ef=ec.get(pid,{})
        if not df_f or not sf or not tf: continue
        pw=row['power_total']; wl=row['wirelength']
        if not(np.isfinite(pw) and pw>0): continue

        t_clk=T_CLK_NS.get(design,7.); f_ghz=1./t_clk
        sd,sl,sa=encode_synth(row.get('synth_strategy','AREA 2'))
        cu=float(row.get('core_util',55.))/100.; dn=float(row.get('density',.5))
        n_ff=df_f['n_ff']; n_active=df_f['n_active']; die_area=df_f['die_area']
        ff_hpwl=df_f['ff_hpwl']; ff_spacing=df_f['ff_spacing']; avg_ds=df_f['avg_ds']
        frac_xor=df_f['frac_xor']; frac_mux=df_f['frac_mux']
        comb_per_ff=df_f['comb_per_ff']; n_comb=df_f['n_comb']
        n_nets=sf['n_nets']; rel_act_raw=sf['rel_act']
        cd=row['cts_cluster_dia']; cs=row['cts_cluster_size']
        mw=row['cts_max_wire']; bd=row['cts_buf_dist']

        pw_norm=max(n_ff*f_ghz*avg_ds,1e-10)
        sm=tf['slack_mean']; fn=tf['frac_neg']; ft=tf['frac_tight']

        # Handle rel_act mode
        if rel_act_mode == 'clip' and tr_rel_act_max is not None:
            rel_act = min(rel_act_raw, tr_rel_act_max)
        elif rel_act_mode == 'log':
            rel_act = np.log1p(rel_act_raw * 100)  # log-compress
        elif rel_act_mode == 'none':
            rel_act = 0.0  # feature excluded via masking below
        else:
            rel_act = rel_act_raw

        raw_rel_acts.append(rel_act_raw)

        # Base power features (v16_final, 58 dims WITH synth)
        base = [
            np.log1p(n_ff),np.log1p(die_area),np.log1p(ff_hpwl),np.log1p(ff_spacing),
            df_f['die_aspect'],float(row.get('aspect_ratio',1.)),
            df_f['ff_cx'],df_f['ff_cy'],df_f['ff_x_std'],df_f['ff_y_std'],
            frac_xor,frac_mux,df_f['frac_and_or'],df_f['frac_nand_nor'],
            df_f['frac_ff_active'],df_f['frac_buf_inv'],comb_per_ff,
            avg_ds,df_f['std_ds'],df_f['p90_ds'],df_f['frac_ds4plus'],
            np.log1p(df_f['cap_proxy']),
            rel_act,  # <-- modified by mode
            sf['mean_sig_prob'],sf['tc_std_norm'],sf['frac_zero'],
            sf['frac_high_act'],sf['log_n_nets'],n_nets/(n_ff+1),
            f_ghz,t_clk,sd,sl,sa,cu,dn,
            np.log1p(cd),np.log1p(cs),np.log1p(mw),np.log1p(bd),cd,cs,mw,bd,
            frac_xor*comb_per_ff,
            rel_act*frac_xor,
            rel_act*(1-df_f['frac_ff_active']),
            sd*avg_ds,sa*f_ghz,
            np.log1p(cd*n_ff/die_area),np.log1p(cs*ff_spacing),
            np.log1p(mw*ff_hpwl),np.log1p(n_ff/cs),cu*dn,
            np.log1p(n_active*rel_act*f_ghz),np.log1p(frac_xor*n_active),
            np.log1p(frac_mux*n_active),np.log1p(comb_per_ff*n_ff),
        ]  # 58 dims

        timing = [
            sm,tf['slack_std'],tf['slack_min'],tf['slack_p10'],tf['slack_p50'],
            fn,ft,tf['frac_critical'],tf['n_paths']/(n_ff+1),
            sm*frac_xor,sm*comb_per_ff,fn*comb_per_ff,ft*avg_ds,
            float(sm>1.5),float(sm>2.),float(sm>3.),np.log1p(sm),sm*f_ghz,
        ]  # 18 dims

        rows.append(base+timing)  # 76 dims
        y.append(np.log(pw/pw_norm))
        meta.append({'placement_id':pid,'design_name':design,
                     'power_total':pw,'pw_norm':pw_norm,'rel_act_raw':rel_act_raw})

    X=np.array(rows,dtype=np.float64); y=np.array(y); mdf=pd.DataFrame(meta)
    if X.ndim==2 and X.shape[0]>0:
        for c in range(X.shape[1]):
            bad=~np.isfinite(X[:,c])
            if bad.any(): X[bad,c]=np.nanmedian(X[~bad,c]) if (~bad).any() else 0.
    return X, y, mdf


def kshot_eval_smart(pred_pw, actual_pw, K_values, n_reps=200, seed=42,
                     meta_te=None, use_median=False, stratified=False):
    """
    K-shot calibration with options:
      use_median: use median instead of mean for k_hat
      stratified: pick K samples spanning knob space (if meta_te provided)
    """
    rng = np.random.default_rng(seed)
    n = len(actual_pw)
    results = {}

    for K in K_values:
        if K == 0:
            results[0] = {'pw_mean': mape(actual_pw, pred_pw), 'pw_std': 0.}
            continue
        if K >= n: continue

        pw_mapes = []
        for _ in range(n_reps):
            if stratified and meta_te is not None and K >= 2:
                # Stratify across CTS knob dimensions
                knob_cols = ['cts_cluster_dia','cts_cluster_size','cts_max_wire','cts_buf_dist']
                available = [c for c in knob_cols if c in meta_te.columns]
                if available:
                    # Sort by first knob, pick evenly spaced
                    vals = meta_te[available[0]].values
                    sorted_idx = np.argsort(vals)
                    step = max(len(sorted_idx)//K, 1)
                    supp = sorted_idx[::step][:K]
                    supp = supp[:K]
                else:
                    supp = rng.choice(n, size=K, replace=False)
            else:
                supp = rng.choice(n, size=K, replace=False)

            rest = np.setdiff1d(np.arange(n), supp)
            ratios = actual_pw[supp] / pred_pw[supp]

            if use_median:
                k_hat = np.median(ratios)
            else:
                k_hat = np.mean(ratios)

            k_hat = np.clip(k_hat, 0.1, 10.)
            pw_mapes.append(mape(actual_pw[rest], pred_pw[rest]*k_hat))

        results[K] = {'pw_mean': np.mean(pw_mapes), 'pw_std': np.std(pw_mapes)}
    return results


def lodo_power_fold(held, X, y, mdf, model='xgb'):
    tr = mdf['design_name']!=held; te = mdf['design_name']==held
    sc = StandardScaler()
    if model == 'xgb':
        m = XGBRegressor(n_estimators=300,max_depth=4,learning_rate=0.05,
                         subsample=0.8,colsample_bytree=0.8,random_state=42,
                         verbosity=0,n_jobs=1)
    else:
        m = LGBMRegressor(n_estimators=300,num_leaves=31,learning_rate=0.03,
                          min_child_samples=10,verbose=-1,n_jobs=1,random_state=42)
    m.fit(sc.fit_transform(X[tr]),y[tr])
    pred = np.exp(m.predict(sc.transform(X[te])))*mdf[te]['pw_norm'].values
    return pred, mdf[te]['power_total'].values, mdf[te]


if __name__ == '__main__':
    print("="*70)
    print("v20: Power Optimization — Minimize K-shot requirement")
    print("="*70)
    sys.stdout.flush()

    with open(DEF_CACHE,'rb') as f: dc=pickle.load(f)
    with open(SAIF_CACHE,'rb') as f: sc_cache=pickle.load(f)
    with open(TIMING_CACHE,'rb') as f: tc=pickle.load(f)
    with open(EXT_CACHE,'rb') as f: ec=pickle.load(f)
    df=pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    df=df.dropna(subset=['power_total']).reset_index(drop=True)
    designs=['aes','ethmac','picorv32','sha256']
    K_VALUES=[0,1,2,3,5,10,20]
    print(f"{T()} n={len(df)}")

    # ─── Strategy A: in-fold rel_act clipping ─────────────────────────────
    print(f"\n{T()} === Strategy A: In-fold rel_act clipping ===")
    print(f"{'Mode':<12} | {'AES':>7} {'ETH':>7} {'PicoRV':>7} {'SHA256':>7} {'Mean':>7}")
    print(f"{'-'*12}-+-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}-{'-'*7}")

    best_mode = 'original'; best_mean_zs = 999.

    for mode in ['original','clip','log','none']:
        per_design = {}
        for held in designs:
            df_tr = df[df['design_name']!=held]
            df_te = df[df['design_name']==held]

            # Compute training rel_act stats (for clip/rank)
            tr_rel_acts = []
            for _,row in df_tr.iterrows():
                pid=row['placement_id']; sf=sc_cache.get(pid)
                if sf: tr_rel_acts.append(sf['rel_act'])
            tr_max = np.percentile(tr_rel_acts, 95) if tr_rel_acts else 0.06

            # Build features for all data with this mode
            X_all, y_all, mdf_all = build_power_features(
                df, dc, sc_cache, tc, ec, rel_act_mode=mode, tr_rel_act_max=tr_max)

            pred, actual, te_meta = lodo_power_fold(held, X_all, y_all, mdf_all)
            per_design[held] = mape(actual, pred)

        vals = list(per_design.values()); mean_v = np.mean(vals)
        marker = ' ← BEST' if mean_v < best_mean_zs else ''
        if mean_v < best_mean_zs: best_mean_zs = mean_v; best_mode = mode
        print(f"{mode:<12} | " +
              " ".join(f"{per_design[d]:>6.1f}%" for d in designs) +
              f" {mean_v:>6.1f}%{marker}")
        sys.stdout.flush()

    # ─── Strategy A2: clip at p90 vs p95 vs p99 ───────────────────────────
    print(f"\n{T()} === Strategy A2: Clip percentile sweep ===")
    for pct in [80, 85, 90, 95, 99, 100]:
        per_design = {}
        for held in designs:
            df_tr = df[df['design_name']!=held]
            tr_rel_acts = []
            for _,row in df_tr.iterrows():
                pid=row['placement_id']; sf=sc_cache.get(pid)
                if sf: tr_rel_acts.append(sf['rel_act'])
            tr_max = np.percentile(tr_rel_acts, pct) if tr_rel_acts else 0.06

            X_all, y_all, mdf_all = build_power_features(
                df, dc, sc_cache, tc, ec, rel_act_mode='clip', tr_rel_act_max=tr_max)
            pred, actual, _ = lodo_power_fold(held, X_all, y_all, mdf_all)
            per_design[held] = mape(actual, pred)

        vals = list(per_design.values()); mean_v = np.mean(vals)
        marker = ' ← BEST' if mean_v < best_mean_zs else ''
        if mean_v < best_mean_zs: best_mean_zs = mean_v; best_mode = f'clip_p{pct}'
        print(f"  clip_p{pct:3d}: " +
              " ".join(f"{per_design[d]:>6.1f}%" for d in designs) +
              f" mean={mean_v:.1f}%{marker}")
        sys.stdout.flush()

    # ─── Strategy B: Smarter K-shot (median vs mean, stratified) ──────────
    print(f"\n{T()} === Strategy B: Smarter K-shot calibration ===")
    print(f"  Using best zero-shot features (mode={best_mode})")
    print(f"  {'Method':<25} | K=1      K=3      K=5      K=10")

    for mode in ['original', best_mode]:
        for use_med in [False, True]:
            label = f"{mode}+{'median' if use_med else 'mean'}"
            # Build features
            per_design_kshot = {K: [] for K in K_VALUES}
            for held in designs:
                df_tr = df[df['design_name']!=held]
                tr_rel_acts = []
                for _,row in df_tr.iterrows():
                    pid=row['placement_id']; sf=sc_cache.get(pid)
                    if sf: tr_rel_acts.append(sf['rel_act'])
                tr_max = np.percentile(tr_rel_acts, 95) if tr_rel_acts else 0.06

                X_all, y_all, mdf_all = build_power_features(
                    df, dc, sc_cache, tc, ec,
                    rel_act_mode=mode if mode=='original' else 'clip',
                    tr_rel_act_max=tr_max)

                pred, actual, te_meta = lodo_power_fold(held, X_all, y_all, mdf_all)
                ks_res = kshot_eval_smart(pred, actual, K_VALUES, use_median=use_med)
                for K in K_VALUES:
                    per_design_kshot[K].append(ks_res[K]['pw_mean'])

            row_vals = [np.mean(per_design_kshot[K]) for K in [1,3,5,10]]
            first_hit = next((K for K in K_VALUES if np.mean(per_design_kshot[K])<=10.), None)
            marker = f' ← ≤10% at K={first_hit}' if first_hit else ''
            print(f"  {label:<25} | " +
                  "  ".join(f"{v:.1f}%" for v in row_vals) + marker)
            sys.stdout.flush()

    # ─── Strategy C: LGB ensemble for power (diversity) ───────────────────
    print(f"\n{T()} === Strategy C: XGB+LGB ensemble for power ===")
    all_preds_xgb, all_preds_lgb, all_actuals = {}, {}, {}
    for held in designs:
        df_tr = df[df['design_name']!=held]
        tr_rel_acts = [sc_cache[row['placement_id']]['rel_act']
                       for _,row in df_tr.iterrows() if sc_cache.get(row['placement_id'])]
        tr_max = np.percentile(tr_rel_acts, 95)
        X, y_arr, mdf_all = build_power_features(
            df, dc, sc_cache, tc, ec, rel_act_mode='clip', tr_rel_act_max=tr_max)
        p_xgb, actual, te_meta = lodo_power_fold(held, X, y_arr, mdf_all, model='xgb')
        p_lgb, _, _ = lodo_power_fold(held, X, y_arr, mdf_all, model='lgb')
        all_preds_xgb[held] = p_xgb; all_preds_lgb[held] = p_lgb
        all_actuals[held] = actual

    for alpha in [0.0, 0.3, 0.5, 0.7, 1.0]:
        vals = []
        for d in designs:
            pred = alpha*all_preds_xgb[d] + (1-alpha)*all_preds_lgb[d]
            vals.append(mape(all_actuals[d], pred))
        print(f"  α={alpha:.1f} (XGB): " +
              " ".join(f"{vals[i]:.1f}%" for i in range(4)) +
              f" mean={np.mean(vals):.1f}%")
        sys.stdout.flush()

    # ─── Summary table: best approach at each K ───────────────────────────
    print(f"\n{T()} === FINAL: Best config K-shot table ===")
    # Use best mode (clip_p95) + median k_hat
    per_design_kshot = {K: [] for K in K_VALUES}
    for held in designs:
        df_tr = df[df['design_name']!=held]
        tr_rel_acts = [sc_cache[row['placement_id']]['rel_act']
                       for _,row in df_tr.iterrows() if sc_cache.get(row['placement_id'])]
        tr_max = np.percentile(tr_rel_acts, 95)
        X, y_arr, mdf_all = build_power_features(
            df, dc, sc_cache, tc, ec, rel_act_mode='clip', tr_rel_act_max=tr_max)

        # Ensemble XGB(0.7) + LGB(0.3)
        p_xgb, actual, _ = lodo_power_fold(held, X, y_arr, mdf_all, model='xgb')
        p_lgb, _, _ = lodo_power_fold(held, X, y_arr, mdf_all, model='lgb')
        pred = 0.7*p_xgb + 0.3*p_lgb

        ks_res = kshot_eval_smart(pred, actual, K_VALUES, use_median=True)
        for K in K_VALUES:
            per_design_kshot[K].append(ks_res[K]['pw_mean'])
        print(f"  {held}: K=0→{ks_res[0]['pw_mean']:.1f}%  "
              f"K=1→{ks_res[1]['pw_mean']:.1f}%  K=3→{ks_res[3]['pw_mean']:.1f}%  "
              f"K=5→{ks_res[5]['pw_mean']:.1f}%  K=10→{ks_res[10]['pw_mean']:.1f}%")
        sys.stdout.flush()

    print(f"\n  MEAN:")
    for K in K_VALUES:
        mean_v = np.mean(per_design_kshot[K])
        marker = ' ← ≤10% TARGET' if mean_v <= 10. else ''
        print(f"    K={K:>2}: {mean_v:.1f}%{marker}")

    print(f"\n{T()} DONE")
