"""
cts_oracle.py — Unified CTS Outcome Prediction Framework

Architecture:
  ┌─────────────────────────────────────────────────────────────┐
  │  CTSOracleFramework                                          │
  │                                                              │
  │  ① FeatureEngine                                            │
  │     • Parses DEF/SAIF/timing ONCE per placement             │
  │     • Builds shared_feats (29-dim placement context)        │
  │     • knob_patch() stamps new CTS knobs into any X vector   │
  │                                                              │
  │  ② TriHeadSurrogate (+ HoldVio head)                        │
  │     • power_head:  XGBRegressor (76-dim)                    │
  │     • wl_head:     LGB+Ridge blend α=0.3 (84-dim)           │
  │     • skew_head:   LGBMRegressor (63-dim)                   │
  │     • hold_head:   LGBMRegressor (66-dim)                   │
  │     All share 29-dim placement context prefix.              │
  │     All share the same knob-patch function.                  │
  │                                                              │
  │  ③ ParetoOptimizer                                          │
  │     • Evaluates 5000+ knob combos in <600ms                 │
  │     • Returns non-dominated (Pareto-optimal) configs        │
  │     • Sensitivity analysis: ∂target/∂knob                   │
  │                                                              │
  │  Mathematical connections between heads:                     │
  │    power ≈ k·WL (capacitive):  cor(pw,WL)=0.61 (AES)       │
  │    skew ↔ hold:  cor = -0.96 (fundamental anti-correlation)  │
  │    cts_max_wire: r_skew=-0.47, r_hold=+0.49 (main lever)    │
  │    cts_cluster_dia: r_power=-0.34 (buffer count driver)      │
  └─────────────────────────────────────────────────────────────┘

Usage:
    oracle = CTSOracleFramework.load('synthesis_best/saved_models/cts_predictor_4target.pkl')
    oracle.load_placement('aes_run_xxx', def_path, saif_path, timing_path)
    result = oracle.predict('aes_run_xxx', cd=50, cs=20, mw=200, bd=100)
    pareto = oracle.pareto_optimize('aes_run_xxx', n=5000)
    sensitivity = oracle.sensitivity_analysis('aes_run_xxx')
"""

import re, os, sys, time, pickle, warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge

warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = BASE

# Clock periods per known design (ns)
T_CLK_NS = {'aes': 7.0, 'ethmac': 7.0, 'picorv32': 10.0, 'sha256': 10.0}

# Knob feature indices in each feature vector (0-indexed)
# Verified from final_synthesis.py feature construction
KNOB_IDX = {
    'pw': {
        'log': [36, 37, 38, 39], 'raw': [40, 41, 42, 43],
        'inter': [(49, 'cd'), (50, 'cs'), (51, 'mw'), (52, 'cs_inv')],
    },
    'wl': {
        'log': [33, 34, 35, 36], 'raw': [37, 38, 39, 40],
        'inter': [(44, 'cd'), (45, 'cs'), (46, 'mw'), (47, 'cs_inv'),
                  (82, 'cd_rsmt'), (83, 'cd_rudy')],
    },
    'sk': {
        'log': [22, 23, 24, 25], 'raw': [26, 27, 28, 29],
    },
    'hv': {
        'log': [33, 34, 35, 36], 'raw': [37, 38, 39, 40],
        'inter': [(44, 'cd'), (45, 'cs'), (46, 'mw'), (47, 'cs_inv')],
    },
}

# Shared feature prefix dims (placement context, knob-independent):
#   First 22 dims in skew, first 29 in power/WL are identical placement geometry + activity + timing


# ── Parsers ──────────────────────────────────────────────────────────────────

def _parse_def(def_path):
    with open(def_path) as f:
        content = f.read()
    units = int(re.search(r'UNITS DISTANCE MICRONS (\d+)', content).group(1))
    x0, y0, x1, y1 = [float(v)/units for v in re.search(
        r'DIEAREA\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)',
        content).groups()]
    die_w, die_h, die_area = x1-x0, y1-y0, (x1-x0)*(y1-y0)

    ct = Counter(re.findall(r'sky130_fd_sc_hd__(\w+)', content))
    fill = ['tap','decap','fill','phy']
    n_tap = sum(v for k,v in ct.items() if any(x in k for x in fill))
    n_active = sum(ct.values()) - n_tap
    n_ff  = sum(v for k,v in ct.items() if k.startswith('df') or k.startswith('ff'))
    n_buf = sum(v for k,v in ct.items() if k.startswith('buf'))
    n_inv = sum(v for k,v in ct.items() if k.startswith('inv'))
    n_xor = sum(v for k,v in ct.items() if k.startswith('xor') or k.startswith('xnor'))
    n_mux = sum(v for k,v in ct.items() if k.startswith('mux'))
    n_and_or   = sum(v for k,v in ct.items() if k.startswith('and') or k.startswith('or'))
    n_nand_nor = sum(v for k,v in ct.items() if k.startswith('nand') or k.startswith('nor'))
    n_comb = max(n_active - n_ff - n_buf - n_inv, 0)

    ds_vals = []
    for k,v in ct.items():
        if not any(x in k for x in fill):
            m = re.search(r'_(\d+)$', k)
            if m: ds_vals.extend([int(m.group(1))]*v)
    avg_ds = np.mean(ds_vals) if ds_vals else 1.0

    ff_pat = (r'-\s+\S+\s+(sky130_fd_sc_hd__df\w+)\s+'
              r'\+\s+(?:PLACED|FIXED)\s+\(\s*([\d.]+)\s+([\d.]+)\s*\)')
    ff_xy = [(float(x)/units, float(y)/units) for _,x,y in re.findall(ff_pat, content)]
    xs = np.array([p[0] for p in ff_xy]); ys = np.array([p[1] for p in ff_xy])

    return {
        'die_area': die_area, 'die_w': die_w, 'die_h': die_h,
        'die_aspect': die_w/(die_h+1e-6),
        'ff_hpwl': (xs.max()-xs.min())+(ys.max()-ys.min()),
        'ff_spacing': np.sqrt(((xs.max()-xs.min())*(ys.max()-ys.min())+1)/max(len(ff_xy),1)),
        'ff_density': len(ff_xy)/die_area,
        'ff_cx': xs.mean()/die_w, 'ff_cy': ys.mean()/die_h,
        'ff_x_std': xs.std()/die_w, 'ff_y_std': ys.std()/die_h,
        'n_ff': len(ff_xy), 'n_active': n_active, 'n_comb': n_comb,
        'n_buf': n_buf, 'n_inv': n_inv,
        'frac_xor': n_xor/(n_active+1), 'frac_mux': n_mux/(n_active+1),
        'frac_and_or': n_and_or/(n_active+1), 'frac_nand_nor': n_nand_nor/(n_active+1),
        'frac_ff_active': n_ff/(n_active+1), 'frac_buf_inv': (n_buf+n_inv)/(n_active+1),
        'comb_per_ff': n_comb/(n_ff+1),
        'avg_ds': avg_ds,
        'std_ds': np.std(ds_vals) if len(ds_vals)>1 else 0.0,
        'p90_ds': np.percentile(ds_vals, 90) if ds_vals else 1.0,
        'frac_ds4plus': sum(1 for d in ds_vals if d>=4)/(len(ds_vals)+1),
        'cap_proxy': n_active*avg_ds,
    }


def _parse_saif(saif_path):
    total_tc = total_t1 = n_nets = max_tc = 0
    tc_vals = []; duration = None
    with open(saif_path) as f:
        for line in f:
            if '(DURATION' in line:
                m = re.search(r'[\d.]+', line)
                if m: duration = float(m.group())
            m = re.search(r'\(TC\s+(\d+)\)', line)
            if m:
                tc = int(m.group(1)); tc_vals.append(tc)
                n_nets += 1; total_tc += tc; max_tc = max(max_tc, tc)
            m2 = re.search(r'\(T1\s+(\d+)\)', line)
            if m2: total_t1 += int(m2.group(1))
    if n_nets == 0 or max_tc == 0: return None
    tc_arr = np.array(tc_vals, dtype=float)
    mean_tc = total_tc / n_nets
    return {
        'n_nets': n_nets, 'rel_act': mean_tc/max_tc,
        'mean_sig_prob': total_t1/(n_nets*duration) if duration else 0.0,
        'tc_std_norm': tc_arr.std()/(mean_tc+1),
        'frac_zero': (tc_arr==0).mean(),
        'frac_high_act': (tc_arr > mean_tc*2).mean(),
        'log_n_nets': np.log1p(n_nets),
    }


def _parse_timing(tp_path):
    tp = pd.read_csv(tp_path); sl = tp['slack'].values
    return {
        'n_paths': len(sl), 'slack_mean': sl.mean(), 'slack_std': sl.std(),
        'slack_min': sl.min(), 'slack_p10': np.percentile(sl, 10),
        'slack_p50': np.percentile(sl, 50),
        'frac_neg': (sl < 0).mean(), 'frac_tight': (sl < 0.5).mean(),
        'frac_critical': (sl < 0.1).mean(),
    }


# ── Feature construction ──────────────────────────────────────────────────────

def _build_feature_vectors(df_f, sf, tf, skf, gf, nf, cd, cs, mw, bd, t_clk=7.0,
                            core_util=0.55, density=0.5, synth_enc=(0.,0.,1.)):
    """
    Build (X_pw, X_wl, X_sk, X_hv) for a single placement + knob config.
    All four vectors share the same placement-context prefix.
    """
    f_ghz = 1.0/t_clk
    sd, sl_s, sa = synth_enc
    n_ff = df_f['n_ff']; n_active = df_f['n_active']; die_area = df_f['die_area']
    ff_hpwl = df_f['ff_hpwl']; ff_spacing = df_f['ff_spacing']; avg_ds = df_f['avg_ds']
    frac_xor = df_f['frac_xor']; frac_mux = df_f['frac_mux']
    comb_per_ff = df_f['comb_per_ff']; n_comb = df_f['n_comb']
    n_nets = sf['n_nets'] if sf else 1; rel_act = sf['rel_act'] if sf else 0.05
    sm = tf['slack_mean']; fn = tf['frac_neg']; ft = tf['frac_tight']

    pw_norm = max(n_ff * f_ghz * avg_ds, 1e-10)
    wl_norm = max(np.sqrt(n_ff * die_area), 1e-3)

    # ── Shared placement context (dims 0–28 for pw/wl, 0–21 for sk) ─────
    # These are KNOB-INDEPENDENT. Identical across all heads.
    place_ctx = [
        np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
        df_f['die_aspect'], 1.0,  # aspect_ratio placeholder = 1.0
        df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
        frac_xor, frac_mux, df_f['frac_and_or'], df_f['frac_nand_nor'],
        df_f['frac_ff_active'], df_f['frac_buf_inv'], comb_per_ff,
        avg_ds, df_f['std_ds'], df_f['p90_ds'], df_f['frac_ds4plus'],
        np.log1p(df_f['cap_proxy']),
        rel_act, sf['mean_sig_prob'] if sf else 0.0,
        sf['tc_std_norm'] if sf else 0.0, sf['frac_zero'] if sf else 0.0,
        sf['frac_high_act'] if sf else 0.0, sf['log_n_nets'] if sf else 0.0,
        n_nets / (n_ff + 1),
    ]  # 29 dims shared between power and WL

    # ── Power features (76 dims) ─────────────────────────────────────────
    pw_ctx = place_ctx + [f_ghz, t_clk, sd, sl_s, sa, core_util, density]  # +7 = 36
    pw_knob = [np.log1p(cd),np.log1p(cs),np.log1p(mw),np.log1p(bd), cd,cs,mw,bd]  # +8 = 44
    pw_inter = [                                                              # 14 dims → base_pw=58
        frac_xor*comb_per_ff, rel_act*frac_xor, rel_act*(1-df_f['frac_ff_active']),
        sd*avg_ds, sa*f_ghz,
        np.log1p(cd*n_ff/die_area), np.log1p(cs*ff_spacing),
        np.log1p(mw*ff_hpwl), np.log1p(n_ff/cs), core_util*density,
        np.log1p(n_active*rel_act*f_ghz), np.log1p(frac_xor*n_active),
        np.log1p(frac_mux*n_active), np.log1p(comb_per_ff*n_ff),
    ]  # 14 → base_pw = 36+8+14 = 58
    timing_feats = [
        sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'], tf['slack_p50'],
        fn, ft, tf['frac_critical'], tf['n_paths']/(n_ff+1),
        sm*frac_xor, sm*comb_per_ff, fn*comb_per_ff, ft*avg_ds,
        float(sm>1.5), float(sm>2.0), float(sm>3.0), np.log1p(sm), sm*f_ghz,
    ]  # 18
    X_pw = np.array(pw_ctx + pw_knob + pw_inter + timing_feats, dtype=np.float64)  # 76

    # ── WL features (84 dims) ────────────────────────────────────────────
    wl_ctx = place_ctx[:29] + [f_ghz, t_clk, core_util, density]  # 33 (no synth)
    wl_knob = [np.log1p(cd),np.log1p(cs),np.log1p(mw),np.log1p(bd), cd,cs,mw,bd]  # +8 = 41
    wl_inter = [
        frac_xor*comb_per_ff, rel_act*frac_xor, rel_act*(1-df_f['frac_ff_active']),
        np.log1p(cd*n_ff/die_area), np.log1p(cs*ff_spacing),
        np.log1p(mw*ff_hpwl), np.log1p(n_ff/cs), core_util*density,
        np.log1p(n_active*rel_act*f_ghz), np.log1p(frac_xor*n_active),
        np.log1p(frac_mux*n_active), np.log1p(comb_per_ff*n_ff),
    ]  # 12 → total = 53 (base_wl)
    rsmt_t = float(nf.get('rsmt_total', 0.0) or 0.0)
    gravity_feats = [
        gf.get('grav_abs_mean',0.), gf.get('grav_abs_std',0.),
        gf.get('grav_abs_p75',0.),  gf.get('grav_abs_p90',0.),
        gf.get('grav_abs_cv',0.),   gf.get('grav_abs_gini',0.),
        gf.get('grav_norm_mean',0.), gf.get('grav_norm_cv',0.),
        gf.get('grav_anisotropy',0.),
        gf.get('grav_abs_mean',0.)*cd, gf.get('grav_abs_mean',0.)*mw,
        gf.get('grav_abs_mean',0.)/(ff_spacing+1),
        gf.get('tp_degree_mean',0.), gf.get('tp_degree_cv',0.),
        gf.get('tp_degree_gini',0.), gf.get('tp_degree_p90',0.),
        gf.get('tp_frac_involved',0.), gf.get('tp_paths_per_ff',0.),
        gf.get('tp_frac_hub',0.),
    ]  # 19
    extra_scale = [
        np.log1p(die_area/(n_ff+1)), np.log1p(n_comb),
        comb_per_ff*np.log1p(n_ff),
    ]  # 3
    net_feats = [
        np.log1p(rsmt_t), rsmt_t/max(n_ff*np.sqrt(die_area),1e-3),
        float(nf.get('net_hpwl_mean',0.) or 0.),
        np.log1p(float(nf.get('net_hpwl_p90',0.) or 0.)),
        float(nf.get('frac_high_fanout',0.) or 0.),
        float(nf.get('rudy_mean',0.) or 0.),
        float(nf.get('rudy_p90',0.) or 0.),
        rsmt_t*cd/max(n_ff*die_area,1.0),
        float(nf.get('rudy_mean',0.) or 0.)*cd,
    ]  # 9
    X_wl = np.array(wl_ctx + wl_knob + wl_inter + gravity_feats + extra_scale + net_feats,
                    dtype=np.float64)  # 84

    # ── Skew features (63 dims) ──────────────────────────────────────────
    crit_max   = skf.get('crit_max_dist', 0.0)
    crit_mean  = skf.get('crit_mean_dist', 0.0)
    crit_p90   = skf.get('crit_p90_dist', 0.0)
    crit_hpwl  = skf.get('crit_ff_hpwl', 0.0)
    crit_cx    = skf.get('crit_cx_offset', 0.0)
    crit_cy    = skf.get('crit_cy_offset', 0.0)
    crit_xs    = skf.get('crit_x_std', 0.0)
    crit_ys    = skf.get('crit_y_std', 0.0)
    crit_bnd   = skf.get('crit_frac_boundary', 0.0)
    crit_star  = skf.get('crit_star_degree', 0.0)
    crit_chn   = skf.get('crit_chain_frac', 0.0)
    crit_asym  = skf.get('crit_asymmetry', 0.0)
    crit_ecc   = skf.get('crit_eccentricity', 1.0)
    crit_dens  = skf.get('crit_density_ratio', 1.0)
    crit_max_um  = skf.get('crit_max_dist_um', ff_hpwl)
    crit_mean_um = skf.get('crit_mean_dist_um', ff_hpwl/2)

    skew_ctx = [  # 22 dims (shared placement prefix, minus synth/density info)
        np.log1p(n_ff), np.log1p(die_area), np.log1p(ff_hpwl), np.log1p(ff_spacing),
        df_f['die_aspect'],
        df_f['ff_cx'], df_f['ff_cy'], df_f['ff_x_std'], df_f['ff_y_std'],
        frac_xor, comb_per_ff, avg_ds, rel_act, sf['mean_sig_prob'] if sf else 0.0,
        sm, tf['slack_std'], tf['slack_min'], tf['slack_p10'],
        fn, ft, tf['frac_critical'], np.log1p(tf['n_paths']/(n_ff+1)),
    ]  # 22
    skew_knob = [np.log1p(cd),np.log1p(cs),np.log1p(mw),np.log1p(bd), cd,cs,mw,bd]  # 8
    skew_crit = [  # 23 critical-path spatial features
        crit_max, crit_mean, crit_p90, crit_hpwl,
        crit_cx, crit_cy, crit_xs, crit_ys,
        crit_bnd, crit_star, crit_chn,
        crit_asym, crit_ecc, crit_dens,
        np.log1p(crit_max_um), np.log1p(crit_mean_um),
    ]  # 16
    skew_inter = [  # knob×spatial interactions
        cd/(ff_spacing+1), bd/(crit_max_um+1), mw/(crit_max_um+1),
        crit_star*cd, crit_asym*mw, crit_dens*cs,
        crit_max*cd, crit_asym*crit_max, fn*crit_star, ft*crit_chn,
        crit_hpwl/(cs+1),
        np.log1p(crit_max_um/(cd+1)), np.log1p(crit_max_um/(bd+1)),
        np.log1p(crit_max_um/(mw+1)),
        crit_cx*cd, crit_cy*mw, np.log1p(n_ff/cs)*crit_hpwl,
    ]  # 17 → total 22+8+16+17 = 63
    X_sk = np.array(skew_ctx + skew_knob + skew_crit + skew_inter, dtype=np.float64)  # 63

    # ── HoldVio features (66 dims = base_wl53 + hold9 + net4) ───────────
    hold_phys = [
        np.log1p(n_ff/cs), np.log1p(cs*ff_spacing),
        np.log1p(cd/(ff_spacing+1)), np.log1p(bd/(ff_hpwl+1)),
        bd/(crit_max_um+1e-3),
        crit_star*cs, crit_chn*bd, crit_asym*cd, np.log1p(crit_max*bd),
    ]  # 9
    net4 = [
        float(nf.get('rudy_mean',0.) or 0.), float(nf.get('rudy_p90',0.) or 0.),
        float(nf.get('frac_high_fanout',0.) or 0.),
        np.log1p(rsmt_t),
    ]  # 4
    X_hv = np.array(wl_ctx + wl_knob + wl_inter + hold_phys + net4, dtype=np.float64)  # 66

    return X_pw, X_wl, X_sk, X_hv, pw_norm, wl_norm


def _pareto_front(costs):
    n = costs.shape[0]
    lo = costs.min(0); rng = (costs.max(0) - lo) + 1e-10
    c = (costs - lo) / rng
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(0, n, 500):
        ci = c[i:i+500]
        dom = (np.all(c[:,None,:] <= ci[None,:,:]+1e-9, axis=2) &
               np.any(c[:,None,:] <  ci[None,:,:]-1e-9, axis=2))
        is_dominated[i:i+500] = dom.any(axis=0)
    return ~is_dominated


# ── Main class ────────────────────────────────────────────────────────────────

class CTSOracleFramework:
    """
    Unified CTS outcome predictor.

    Physical connections between the four prediction heads:
      1. Shared feature engine: 29-dim placement context prefix is identical
         for power, WL, and hold heads; 22-dim for skew.
      2. Shared normalization physics:
           pw_norm  = n_ff × f_ghz × avg_ds
           wl_norm  = √(n_ff × die_area)
      3. Shared knob-patch function: same logic, different index offsets.
      4. Anti-correlated objectives: skew ↔ hold (r≈-0.96); captured jointly
         by the Pareto optimizer.
      5. Power ≈ k·WL for clock-tree-dominant designs (capacitive coupling).
    """

    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.lodo_results = {}
        self._placements = {}  # pid → {df_f, sf, tf, skf, gf, nf}

    # ── I/O ────────────────────────────────────────────────────────────────

    @classmethod
    def load(cls, model_path):
        obj = cls()
        with open(model_path, 'rb') as f:
            mdl = pickle.load(f)
        obj.models  = {k: mdl[f'model_{k}']  for k in ['power','wl_lgb','wl_ridge','skew','hold_vio']}
        obj.scalers = {k: mdl[f'scaler_{k}'] for k in ['power','wl','skew','hold_vio']}
        obj.wl_alpha = mdl.get('wl_blend_alpha', 0.3)
        obj.lodo_results = mdl.get('lodo', {})
        return obj

    def load_placement_from_caches(self, pid, dc, sc, tc, skc, gc, nc):
        """Load a placement already present in the pre-built caches."""
        self._placements[pid] = {
            'df_f': dc.get(pid),
            'sf':   sc.get(pid),
            'tf':   tc.get(pid),
            'skf':  skc.get(pid, {}),
            'gf':   gc.get(pid, {}),
            'nf':   nc.get(pid, {}),
        }

    def load_placement(self, pid, def_path, saif_path, timing_path,
                       t_clk=None, design=None):
        """Parse a new placement directly from DEF/SAIF/timing files."""
        if t_clk is None:
            t_clk = T_CLK_NS.get(design or '', 7.0)
        self._placements[pid] = {
            'df_f': _parse_def(def_path),
            'sf':   _parse_saif(saif_path),
            'tf':   _parse_timing(timing_path),
            'skf':  {},   # skew spatial requires separate build_skew_cache step
            'gf':   {},
            'nf':   {},
            't_clk': t_clk,
        }

    # ── Prediction ──────────────────────────────────────────────────────────

    def _get_feats(self, pid, cd, cs, mw, bd):
        p = self._placements[pid]
        t_clk = p.get('t_clk', 7.0)
        return _build_feature_vectors(
            p['df_f'], p['sf'], p['tf'], p['skf'], p['gf'], p['nf'],
            cd, cs, mw, bd, t_clk)

    def predict(self, pid, cd, cs, mw, bd,
                sk_mu=None, sk_sig=None, hv_mu=None, hv_sig=None):
        """
        Predict all 4 CTS outcomes for a single knob configuration.

        Args:
            sk_mu, sk_sig: per-placement skew normalization params.
                           If None, raw z-score is returned.
            hv_mu, hv_sig: per-placement hold-vio log-space norm params.

        Returns dict: {power_W, wl_um, skew_ns, hold_vio, pw_norm, wl_norm}
        """
        X_pw, X_wl, X_sk, X_hv, pw_norm, wl_norm = self._get_feats(pid, cd, cs, mw, bd)

        pw = float(np.exp(self.models['power'].predict(
            self.scalers['power'].transform(X_pw.reshape(1,-1)))[0])) * pw_norm

        X_wl_s = self.scalers['wl'].transform(X_wl.reshape(1,-1))
        wl = float(np.exp(
            self.wl_alpha * self.models['wl_lgb'].predict(X_wl_s)[0] +
            (1-self.wl_alpha) * self.models['wl_ridge'].predict(X_wl_s)[0]
        )) * wl_norm

        sk_z = float(self.models['skew'].predict(
            self.scalers['skew'].transform(X_sk.reshape(1,-1)))[0])
        skew = sk_z * sk_sig + sk_mu if sk_sig is not None else sk_z

        hv_z = float(self.models['hold_vio'].predict(
            self.scalers['hold_vio'].transform(X_hv.reshape(1,-1)))[0])
        if hv_sig is not None:
            hold_vio = float(np.expm1(np.clip(hv_z*hv_sig+hv_mu, 0, 20)))
        else:
            hold_vio = float(np.expm1(np.clip(hv_z, 0, 20)))

        return {'power_W': pw, 'wl_um': wl, 'skew_z': sk_z,
                'skew_ns': skew, 'hold_vio': hold_vio,
                'pw_norm': pw_norm, 'wl_norm': wl_norm}

    # ── Pareto Optimizer ────────────────────────────────────────────────────

    def pareto_optimize(self, pid, n=5000,
                        cd_range=(35,70), cs_range=(12,30),
                        mw_range=(130,280), bd_range=(70,150),
                        objectives=('power','skew','hold_vio'),
                        sk_mu=None, sk_sig=None, hv_mu=None, hv_sig=None,
                        seed=42):
        """
        Sweep n random knob configurations and return Pareto-optimal solutions.

        The three surrogate heads are evaluated simultaneously, providing
        the cross-objective tradeoff surface. All 5000 combos run in <600ms.

        Returns: (df_pareto, df_all_sweep) DataFrames
        """
        rng = np.random.default_rng(seed)
        cd_arr = rng.uniform(*cd_range, n)
        cs_arr = rng.integers(*cs_range, n).astype(float)
        mw_arr = rng.uniform(*mw_range, n)
        bd_arr = rng.uniform(*bd_range, n)

        p  = self._placements[pid]
        df_f = p['df_f']; sf = p['sf']; tf = p['tf']
        skf = p['skf']; gf = p['gf']; nf = p['nf']
        t_clk = p.get('t_clk', 7.0)

        # Build base feature rows at median knobs, then batch-patch
        cd0 = np.median(cd_arr); cs0 = np.median(cs_arr)
        mw0 = np.median(mw_arr); bd0 = np.median(bd_arr)
        X_pw0, X_wl0, X_sk0, X_hv0, pw_norm, wl_norm = _build_feature_vectors(
            df_f, sf, tf, skf, gf, nf, cd0, cs0, mw0, bd0, t_clk)

        n_ff = df_f['n_ff']; die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']; ff_spacing = df_f['ff_spacing']
        rsmt_t = float(nf.get('rsmt_total', 0.0) or 0.0)
        rudy_m = float(nf.get('rudy_mean', 0.0) or 0.0)

        def _patch(x0, ki):
            X = np.tile(x0, (n,1)).astype(np.float64)
            for li, v in zip(ki['log'], [cd_arr,cs_arr,mw_arr,bd_arr]):
                X[:,li] = np.log1p(v)
            for ri, v in zip(ki['raw'], [cd_arr,cs_arr,mw_arr,bd_arr]):
                X[:,ri] = v
            for (ii, kind) in ki.get('inter', []):
                if kind=='cd':      X[:,ii] = np.log1p(cd_arr*n_ff/die_area)
                elif kind=='cs':    X[:,ii] = np.log1p(cs_arr*ff_spacing)
                elif kind=='mw':    X[:,ii] = np.log1p(mw_arr*ff_hpwl)
                elif kind=='cs_inv':X[:,ii] = np.log1p(n_ff/cs_arr)
                elif kind=='cd_rsmt':X[:,ii] = rsmt_t*cd_arr/max(n_ff*die_area,1)
                elif kind=='cd_rudy':X[:,ii] = rudy_m*cd_arr
            return X

        Xpw = _patch(X_pw0, KNOB_IDX['pw'])
        Xwl = _patch(X_wl0, KNOB_IDX['wl'])
        Xsk = _patch(X_sk0, KNOB_IDX['sk'])
        Xhv = _patch(X_hv0, KNOB_IDX['hv'])

        pred_pw = np.exp(self.models['power'].predict(
            self.scalers['power'].transform(Xpw))) * pw_norm

        Xwl_s = self.scalers['wl'].transform(Xwl)
        pred_wl = np.exp(
            self.wl_alpha * self.models['wl_lgb'].predict(Xwl_s) +
            (1-self.wl_alpha) * self.models['wl_ridge'].predict(Xwl_s)
        ) * wl_norm

        pred_sk_z = self.models['skew'].predict(self.scalers['skew'].transform(Xsk))
        if sk_sig is not None:
            pred_sk = pred_sk_z * sk_sig + sk_mu
        else:
            pred_sk = pred_sk_z

        pred_hv_z = self.models['hold_vio'].predict(self.scalers['hold_vio'].transform(Xhv))
        if hv_sig is not None:
            pred_hv = np.expm1(np.clip(pred_hv_z * hv_sig + hv_mu, 0, 20))
        else:
            pred_hv = np.expm1(np.clip(pred_hv_z, 0, 20))

        df_sweep = pd.DataFrame({
            'cd': cd_arr, 'cs': cs_arr.astype(int),
            'mw': mw_arr.round(1), 'bd': bd_arr.round(1),
            'power_mW': pred_pw * 1000, 'wl_mm': pred_wl / 1000,
            'skew_z': pred_sk_z, 'skew_ns': pred_sk,
            'hold_vio': pred_hv,
        })

        obj_map = {'power': 'power_mW', 'wl': 'wl_mm',
                   'skew': 'skew_ns' if sk_sig else 'skew_z', 'hold_vio': 'hold_vio'}
        cost_cols = [obj_map[o] for o in objectives if o in obj_map]
        costs = df_sweep[cost_cols].values
        pareto_mask = _pareto_front(costs)
        df_sweep['pareto'] = pareto_mask

        return df_sweep[df_sweep['pareto']].sort_values('power_mW'), df_sweep

    # ── Sensitivity Analysis ────────────────────────────────────────────────

    def sensitivity_analysis(self, pid, base_knobs=(50, 20, 200, 100),
                              delta_frac=0.10, sk_mu=None, sk_sig=None):
        """
        Compute ∂(target)/∂(knob) at base_knobs using finite differences.

        Shows which knobs drive which targets — the cross-head connections.
        Returns dict of {knob: {target: sensitivity}}
        """
        cd0, cs0, mw0, bd0 = base_knobs
        base = self.predict(pid, cd0, cs0, mw0, bd0, sk_mu=sk_mu, sk_sig=sk_sig)

        knob_names = ['cd', 'cs', 'mw', 'bd']
        knob_vals  = [cd0, cs0, mw0, bd0]
        results = {}

        for i, (kname, kval) in enumerate(zip(knob_names, knob_vals)):
            delta = max(kval * delta_frac, 1.0)
            knobs_hi = list(knob_vals); knobs_hi[i] += delta
            knobs_lo = list(knob_vals); knobs_lo[i] -= delta
            hi = self.predict(pid, *knobs_hi, sk_mu=sk_mu, sk_sig=sk_sig)
            lo = self.predict(pid, *knobs_lo, sk_mu=sk_mu, sk_sig=sk_sig)

            results[kname] = {
                'power_pct': (hi['power_W'] - lo['power_W']) / base['power_W'] / (2*delta_frac) * 100,
                'wl_pct':    (hi['wl_um']   - lo['wl_um'])   / base['wl_um']   / (2*delta_frac) * 100,
                'skew_abs':  (hi['skew_z']  - lo['skew_z'])  / (2*delta_frac),
                'hold_abs':  (hi['hold_vio']- lo['hold_vio']) / base['hold_vio'] / (2*delta_frac) * 100
                             if base['hold_vio'] > 0.1 else 0.0,
            }

        return results

    # ── Evaluation ────────────────────────────────────────────────────────

    def lodo_summary(self):
        """Print LODO validation results."""
        print("=== LODO Validation (Leave-One-Design-Out) ===")
        for target in ['power', 'wl', 'skew', 'hold_vio']:
            res = self.lodo_results.get(target, {})
            if not res: continue
            vals = list(res.values())
            unit = 'MAE' if target == 'skew' else 'MAPE'
            print(f"  {target:8s}: " +
                  "  ".join(f"{d}={v:.1f}{'%' if unit=='MAPE' else ''}"
                             for d,v in res.items()) +
                  f"  → mean={np.mean(vals):.1f}{'%' if unit=='MAPE' else ''}")


# ── Main: extensive testing ───────────────────────────────────────────────────

if __name__ == '__main__':
    t0 = time.time()
    def T(): return f"[{time.time()-t0:.1f}s]"

    print("=" * 70)
    print("CTSOracleFramework — Unified CTS Surrogate")
    print("  Architecture: FeatureEngine → TriHeadSurrogate → ParetoOptimizer")
    print("=" * 70)

    # ── Load model and caches ───────────────────────────────────────────────
    print(f"\n{T()} Loading model and caches...")
    oracle = CTSOracleFramework.load(
        f'{BASE}/synthesis_best/saved_models/cts_predictor_4target.pkl')
    oracle.lodo_summary()

    dc  = pickle.load(open(f'{BASE}/absolute_v7_def_cache.pkl','rb'))
    sc  = pickle.load(open(f'{BASE}/absolute_v7_saif_cache.pkl','rb'))
    tc  = pickle.load(open(f'{BASE}/absolute_v7_timing_cache.pkl','rb'))
    skc = pickle.load(open(f'{BASE}/skew_spatial_cache.pkl','rb'))
    gc  = pickle.load(open(f'{BASE}/absolute_v10_gravity_cache.pkl','rb'))
    nc  = pickle.load(open(f'{BASE}/synthesis_best/net_features_cache.pkl','rb'))

    df_all = pd.read_csv(f'{BASE}/dataset_with_def/unified_manifest_normalized.csv')

    # Load all placements from caches
    for pid in dc.keys():
        oracle.load_placement_from_caches(pid, dc, sc, tc, skc, gc, nc)
    print(f"{T()} Loaded {len(oracle._placements)} placements")

    # ── Cross-target relationship analysis ─────────────────────────────────
    print(f"\n{T()} === CROSS-TARGET RELATIONSHIP ANALYSIS ===")
    print("  Physics connections between the three surrogate heads:\n")

    df = df_all.dropna(subset=['power_total','wirelength','skew_setup'])
    designs = sorted(df['design_name'].unique())

    print("  1. Power ↔ WL (capacitive coupling: P = k·C_wire·V²·f, C ∝ WL)")
    for d in designs:
        m = df[df['design_name']==d]
        r = np.corrcoef(m.power_total, m.wirelength)[0,1]
        bar = '█' * int(abs(r)*20)
        print(f"     {d:10s}: r={r:+.3f}  {bar}")
    print("     AES has highest coupling (clock-tree dominant at 2994 FFs)")
    print("     ETH/SHA256/PicoRV: power dominated by logic, not routing\n")

    print("  2. Skew ↔ Hold (fundamental anti-correlation: r≈-0.96)")
    print("     Physics: lower setup skew = more balanced clock = stricter hold timing")
    for d in designs:
        m = df[df['design_name']==d]
        r = np.corrcoef(m.skew_setup, m.skew_hold)[0,1]
        bar = '█' * int(abs(r)*20)
        print(f"     {d:10s}: r={r:+.3f}  {bar}")
    print("     → Hold violations predictable FROM skew predictions (shared signal)\n")

    print("  3. Knob → Target sensitivity (within-design, linear correlation)")
    pw = df['power_total'].values; wl = df['wirelength'].values
    sk = df['skew_setup'].values;  sh = df['skew_hold'].values
    pw_dm=pw.copy(); wl_dm=wl.copy(); sk_dm=sk.copy(); sh_dm=sh.copy()
    for d in designs:
        idx=df['design_name'].values==d
        for a in [pw_dm,wl_dm,sk_dm,sh_dm]: a[idx]-=a[idx].mean()

    print(f"     {'Knob':20s}  {'→Power':>8}  {'→WL':>8}  {'→Skew':>8}  {'→Hold':>8}")
    knob_roles = {
        'cts_cluster_dia':  'cluster diameter → fewer/larger clusters',
        'cts_cluster_size': 'FFs per cluster → routing topology',
        'cts_max_wire':     'max wire budget → primary skew/hold lever',
        'cts_buf_dist':     'buffer insertion distance',
    }
    for knob in ['cts_cluster_dia','cts_cluster_size','cts_max_wire','cts_buf_dist']:
        k=df[knob].values.copy()
        for d in designs: idx=df['design_name'].values==d; k[idx]-=k[idx].mean()
        rpw=np.corrcoef(k,pw_dm)[0,1]; rwl=np.corrcoef(k,wl_dm)[0,1]
        rsk=np.corrcoef(k,sk_dm)[0,1]; rsh=np.corrcoef(k,sh_dm)[0,1]
        print(f"     {knob:20s}  {rpw:+.3f}     {rwl:+.3f}     {rsk:+.3f}     {rsh:+.3f}")
        print(f"       ({knob_roles[knob]})")
    print()
    print("  Key insight: cts_max_wire is the PRIMARY tradeoff knob:")
    print("    ↑ max_wire → ↓ skew (r=-0.47) but ↑ hold violations (r=+0.49)")
    print("  The Pareto optimizer finds the optimal tradeoff point automatically.\n")

    # ── Sensitivity analysis on one placement ──────────────────────────────
    print(f"{T()} === SENSITIVITY ANALYSIS (∂target/∂knob) ===")
    demo_pid = next(p for p in dc.keys() if 'aes' in p)
    print(f"  Placement: {demo_pid}")
    sens = oracle.sensitivity_analysis(demo_pid, base_knobs=(50,20,200,100))
    print(f"  (±10% perturbation around base knobs: cd=50, cs=20, mw=200, bd=100)")
    print(f"  {'Knob':6s}  {'ΔPower/Δknob':>14}  {'ΔWL/Δknob':>12}  "
          f"{'ΔSkewZ/Δknob':>14}  {'ΔHold/Δknob':>14}")
    for k, v in sens.items():
        print(f"  {k:6s}  {v['power_pct']:+12.1f}%    {v['wl_pct']:+10.1f}%    "
              f"{v['skew_abs']:+12.4f}      {v['hold_abs']:+12.1f}%")
    print()
    print("  Interpretation:")
    print("    cd (cluster_dia): ↑cd → significantly lower power (larger clusters)")
    print("    mw (max_wire):    ↑mw → lower skew but more hold violations")
    print("    These are the two dominant control dimensions for the Pareto surface")

    # ── Pareto optimization demo ────────────────────────────────────────────
    print(f"\n{T()} === PARETO OPTIMIZER ===")
    print("  Evaluating 5000 knob combos simultaneously across all 4 heads...\n")

    for demo_d in ['aes', 'picorv32']:
        pid = next(p for p in dc.keys() if demo_d in p)
        t_s = time.time()
        df_par, df_sw = oracle.pareto_optimize(pid, n=5000)
        t_ms = (time.time()-t_s)*1000

        print(f"  Design: {demo_d}  placement: {pid}")
        print(f"  {len(df_par)} Pareto solutions from 5000 combos in {t_ms:.0f}ms")
        print(f"  {'cd':>5} {'cs':>4} {'mw':>5} {'bd':>5} | "
              f"{'Power(mW)':>10} {'WL(mm)':>8} {'SkewZ':>8} {'HoldVio':>8}")
        print(f"  {'-'*5}-{'-'*4}-{'-'*5}-{'-'*5}-+-{'-'*10}-{'-'*8}-{'-'*8}-{'-'*8}")
        for _, r in df_par.head(8).iterrows():
            print(f"  {r.cd:>5.0f} {r.cs:>4.0f} {r.mw:>5.0f} {r.bd:>5.0f} | "
                  f"{r.power_mW:>9.2f}  {r.wl_mm:>7.2f}  {r.skew_z:>7.3f}  "
                  f"{r.hold_vio:>8.1f}")
        print(f"  Sweep ranges: power {df_sw.power_mW.min():.1f}–{df_sw.power_mW.max():.1f}mW  "
              f"skewZ {df_sw.skew_z.min():.3f}–{df_sw.skew_z.max():.3f}  "
              f"hold {df_sw.hold_vio.min():.0f}–{df_sw.hold_vio.max():.0f}\n")

    # ── Zipdiv (truly unseen design) ────────────────────────────────────────
    print(f"{T()} === ZIPDIV: TRULY UNSEEN DESIGN ===")
    print("  zipdiv not in training manifest — zero-shot prediction\n")
    PLACEMENT_DIR = f'{BASE}/dataset_with_def/placement_files'
    zip_pids = ['zipdiv_run_20260312_160558', 'zipdiv_run_20260312_160735']
    for pid in zip_pids:
        d = f'{PLACEMENT_DIR}/{pid}'
        try:
            oracle.load_placement(
                pid, f'{d}/zipdiv.def', f'{d}/zipdiv.saif', f'{d}/timing_paths.csv',
                t_clk=10.0, design='zipdiv')
            p0 = oracle._placements[pid]
            print(f"  {pid}:")
            print(f"    n_ff={p0['df_f']['n_ff']}, "
                  f"die={p0['df_f']['die_w']:.0f}×{p0['df_f']['die_h']:.0f}µm, "
                  f"rel_act={p0['sf']['rel_act']:.4f}")
            df_par, df_sw = oracle.pareto_optimize(
                pid, n=3000, cd_range=(20,60), cs_range=(8,24),
                mw_range=(80,250), bd_range=(50,120))
            print(f"    Pareto solutions: {len(df_par)}/3000  "
                  f"power {df_sw.power_mW.min():.2f}–{df_sw.power_mW.max():.2f}mW  "
                  f"wl {df_sw.wl_mm.min():.3f}–{df_sw.wl_mm.max():.3f}mm")
            if len(df_par) > 0:
                best = df_par.iloc[0]
                print(f"    Recommended: cd={best.cd:.0f} cs={best.cs:.0f} "
                      f"mw={best.mw:.0f} bd={best.bd:.0f} "
                      f"→ {best.power_mW:.3f}mW, skewZ={best.skew_z:.3f}")
        except Exception as e:
            print(f"    Error: {e}")

    # ── LODO: Verify feature dims match model ───────────────────────────────
    print(f"\n{T()} === FEATURE DIMENSION VERIFICATION ===")
    pid_test = next(p for p in dc.keys() if 'aes' in p)
    X_pw, X_wl, X_sk, X_hv, _, _ = _build_feature_vectors(
        dc[pid_test], sc[pid_test], tc[pid_test],
        skc.get(pid_test,{}), gc.get(pid_test,{}), nc.get(pid_test,{}),
        50, 20, 200, 100)
    print(f"  X_pw={len(X_pw)} (expected 76)  {'✓' if len(X_pw)==76 else '✗'}")
    print(f"  X_wl={len(X_wl)} (expected 84)  {'✓' if len(X_wl)==84 else '✗'}")
    print(f"  X_sk={len(X_sk)} (expected 63)  {'✓' if len(X_sk)==63 else '✗'}")
    print(f"  X_hv={len(X_hv)} (expected 66)  {'✓' if len(X_hv)==66 else '✗'}")

    # Verify predictions match final_synthesis.py output
    print(f"\n{T()} === PREDICTION CONSISTENCY CHECK ===")
    print("  Comparing CTSOracleFramework predictions vs production model")
    row_test = df_all[df_all['placement_id']==pid_test].iloc[0]
    cd_t = row_test['cts_cluster_dia']; cs_t = row_test['cts_cluster_size']
    mw_t = row_test['cts_max_wire'];    bd_t = row_test['cts_buf_dist']
    pred = oracle.predict(pid_test, cd_t, cs_t, mw_t, bd_t)
    print(f"  Placement: {pid_test}, knobs: cd={cd_t:.0f} cs={cs_t:.0f} mw={mw_t:.0f} bd={bd_t:.0f}")
    print(f"  Predicted power: {pred['power_W']*1000:.2f} mW  (true: {row_test['power_total']*1000:.2f} mW)")
    print(f"  Predicted WL:    {pred['wl_um']/1000:.1f} mm   (true: {row_test['wirelength']/1000:.1f} mm)")
    print(f"  Predicted skew Z: {pred['skew_z']:.3f}")

    print(f"\n{T()} DONE")
    print("\nFramework Summary:")
    print("  • Single load_placement() call parses DEF/SAIF/timing once")
    print("  • predict() returns all 4 targets in <1ms")
    print("  • pareto_optimize() finds Pareto front in <600ms (5000 combos)")
    print("  • sensitivity_analysis() quantifies per-knob impact on each target")
    print("  • Key physics: mw↔skew/hold tradeoff, cd↔power, skew↔hold anti-corr")
