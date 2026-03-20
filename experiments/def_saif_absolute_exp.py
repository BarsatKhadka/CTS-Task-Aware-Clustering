"""
def_saif_absolute_exp.py - ICLR-level absolute prediction using DEF+SAIF physics

DEF gives: exact cell composition, die geometry, FF positions/HPWL
SAIF gives: exact per-net switching activity, signal probabilities

Key insight: SAIF has the ACTUAL switching activity for every net in the circuit.
Physics formula: P_dynamic = Σ(α_i × C_i × V² × f) / 2
  = rel_act × total_cap × V² × f

This directly encodes why circuits differ:
  - AES: high XOR/XNOR activity (encryption), high toggle rates
  - ethmac: moderate activity (network MAC), different cell composition
  - picorv32: sequential RISC core, different activity profile
  - sha256: hash function, high activity

For WL: ff_hpwl × CTS_efficiency(cluster_dia, cluster_size) → direct physics

Novel: physics-formula features that normalize out technology parameters:
  power_proxy = rel_act × ff_cap_proxy × f_clk / cluster_size
  wl_proxy = ff_hpwl / cluster_dia^α × cluster_size^β
"""

import pickle, time, warnings
import numpy as np
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("DEF+SAIF Physics-Formula Absolute Prediction")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────
with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/absolute_v5_def_cache.pkl', 'rb') as f:
    def_cache = pickle.load(f)
with open(f'{BASE}/absolute_v5_saif_cache.pkl', 'rb') as f:
    saif_cache = pickle.load(f)
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)

pids = df['placement_id'].values; designs = df['design_name'].values; n = len(pids)

# Absolute targets
y_pw_abs = df['power_total'].values.astype(np.float64)
y_wl_abs = df['wirelength'].values.astype(np.float64)
y_pw_log = np.log(y_pw_abs + 1e-10)
y_wl_log = np.log(y_wl_abs + 1.0)

# Per-placement z-score targets
y_pw_z = Y_cache[:, 1]
y_wl_z = Y_cache[:, 2]

print(f"Power range: [{y_pw_abs.min():.5f}, {y_pw_abs.max():.5f}] W")
print(f"WL range: [{y_wl_abs.min():.0f}, {y_wl_abs.max():.0f}] µm")

def rank_within(v): return np.argsort(np.argsort(v)).astype(float)/max(len(v)-1,1)

# ── Build CTS knob features ───────────────────────────────────────────────
knob_cols = ['cts_max_wire', 'cts_buf_dist', 'cts_cluster_size', 'cts_cluster_dia']
Xraw = df[knob_cols].values.astype(np.float32)
Xkz = X_cache[:, 72:76]
raw_max = Xraw.max(0) + 1e-6
Xrank = np.zeros((n,4),np.float32); Xcent = np.zeros_like(Xrank)
Xrng = np.zeros_like(Xrank); Xmn = np.zeros_like(Xrank)
for pid in np.unique(pids):
    m=pids==pid; idx=np.where(m)[0]
    for j in range(4):
        v=Xraw[idx,j]; Xrank[idx,j]=rank_within(v)
        Xcent[idx,j]=(v-v.mean())/raw_max[j]; Xrng[idx,j]=v.std()/raw_max[j]; Xmn[idx,j]=v.mean()/raw_max[j]
Xplc = df[['core_util','density','aspect_ratio']].values.astype(np.float32)
Xplc_n = Xplc/(Xplc.max(0)+1e-9)
cd=Xraw[:,3]; cs=Xraw[:,2]; mw=Xraw[:,0]; bd=Xraw[:,1]
util=Xplc[:,0]/100; dens=Xplc[:,1]; asp=Xplc[:,2]
Xinter=np.column_stack([cd*util,mw*dens,cd/(dens+0.01),cd*asp,Xrank[:,3]*util,Xrank[:,2]*util])
X29=np.hstack([Xkz,Xrank,Xcent,Xplc_n,Xinter,Xrng,Xmn])
Xt=np.zeros((n,20),np.float32)
for i,pid in enumerate(pids):
    v=tp.get(pid)
    if v is not None: Xt[i,:20]=np.array(v,np.float32)[:20]
tps=Xt.std(0); tps[tps<1e-9]=1.0; X29T=np.hstack([X29,Xt/tps])

# ── Build DEF+SAIF features (per-placement) ──────────────────────────────
print(f"\n{T()} Building DEF+SAIF features...")

def_keys = ['die_area', 'die_w', 'die_h', 'die_aspect', 'ff_hpwl', 'ff_spacing',
            'ff_density', 'ff_cx', 'ff_cy', 'ff_x_std', 'ff_y_std',
            'n_ff', 'n_active', 'n_total', 'n_tap', 'n_buf', 'n_inv', 'n_comb',
            'n_xor_xnor', 'n_mux', 'n_and_or', 'n_nand_nor',
            'frac_xor', 'frac_mux', 'frac_and_or', 'frac_nand_nor',
            'frac_ff_active', 'frac_buf_inv', 'comb_per_ff',
            'avg_ds', 'std_ds', 'p90_ds', 'frac_ds4plus',
            'cap_proxy', 'ff_cap_proxy']
saif_keys = ['n_nets', 'max_tc', 'mean_tc', 'rel_act', 'mean_sig_prob',
             'tc_std_norm', 'frac_zero', 'frac_high_act', 'log_n_nets']

n_def = len(def_keys); n_saif = len(saif_keys)
X_def = np.zeros((n, n_def), np.float64)
X_saif = np.zeros((n, n_saif), np.float64)
has_def = np.zeros(n, bool)

for i, pid in enumerate(pids):
    d = def_cache.get(pid)
    s = saif_cache.get(pid)
    if d is not None:
        for j, k in enumerate(def_keys):
            X_def[i, j] = float(d.get(k, 0) or 0)
        has_def[i] = True
    if s is not None:
        for j, k in enumerate(saif_keys):
            X_saif[i, j] = float(s.get(k, 0) or 0)

print(f"  DEF coverage: {has_def.sum()}/{n} rows ({has_def.sum()/n*100:.0f}%)")

# ── Physics formula features ──────────────────────────────────────────────
# Power physics: P ∝ activity × capacitance × V²f / n_cycles
n_ff = X_def[:, def_keys.index('n_ff')].clip(1)
n_buf = X_def[:, def_keys.index('n_buf')]
n_comb = X_def[:, def_keys.index('n_comb')]
ff_hpwl = X_def[:, def_keys.index('ff_hpwl')].clip(1)
die_area = X_def[:, def_keys.index('die_area')].clip(1)
die_w = X_def[:, def_keys.index('die_w')].clip(1)
die_h = X_def[:, def_keys.index('die_h')].clip(1)
ff_cap_proxy = X_def[:, def_keys.index('ff_cap_proxy')].clip(1)
cap_proxy = X_def[:, def_keys.index('cap_proxy')].clip(1)
avg_ds = X_def[:, def_keys.index('avg_ds')].clip(0.1)
frac_xor = X_def[:, def_keys.index('frac_xor')]

max_tc = X_saif[:, saif_keys.index('max_tc')].clip(1)
mean_tc = X_saif[:, saif_keys.index('mean_tc')].clip(0)
rel_act = X_saif[:, saif_keys.index('rel_act')].clip(0)
n_nets = X_saif[:, saif_keys.index('n_nets')].clip(1)
tc_std = X_saif[:, saif_keys.index('tc_std_norm')].clip(0)

# Key physics interactions with CTS knobs:
# Power = activity × cap × V²f
# CTS knobs modify: cluster_size → n_buffers → extra cap
# WL = HPWL × routing_factor(cluster_dia, n_ff)

# Absolute physics estimate features
total_activity = n_nets * mean_tc           # total toggle events
act_cap = total_activity * cap_proxy / (n_ff + 1)  # activity × cap per FF
clock_freq = max_tc / 2.0                   # clock cycles in simulation

# Power prediction physics
power_proxy = act_cap * avg_ds / clock_freq  # activity-weighted power proxy
power_per_ff = power_proxy / n_ff           # per-FF power
buf_power_proxy = n_ff / (cs + 1) * cap_proxy / n_ff  # buffer contribution

# WL prediction physics
wl_proxy_hpwl = ff_hpwl * np.log1p(n_ff / cs) / np.log1p(cd)  # HPWL × clustering factor
wl_proxy_nff = np.sqrt(n_ff * die_area)     # sqrt(n_ff × die_area)
routing_eff = cd / (ff_hpwl / n_ff.clip(1) * 1000 + 1e-4)  # cluster_dia / mean_ff_spacing

# Log features (scale-invariant)
log_n_ff = np.log1p(n_ff)
log_die_area = np.log1p(die_area)
log_ff_hpwl = np.log1p(ff_hpwl)
log_n_nets = np.log1p(n_nets)
log_act_cap = np.log1p(act_cap.clip(0))
log_rel_act = np.log1p(rel_act)

# Features for absolute prediction
X_phys = np.column_stack([
    # Circuit identity (from DEF+SAIF)
    log_n_ff, log_die_area, log_ff_hpwl, log_n_nets, log_act_cap, log_rel_act,
    frac_xor, X_def[:, def_keys.index('frac_mux')],
    X_def[:, def_keys.index('frac_ff_active')], X_def[:, def_keys.index('comb_per_ff')],
    X_def[:, def_keys.index('ff_density')],
    X_def[:, def_keys.index('avg_ds')], X_def[:, def_keys.index('frac_ds4plus')],
    X_saif[:, saif_keys.index('mean_sig_prob')],
    X_saif[:, saif_keys.index('frac_high_act')], tc_std,
    # CTS knobs (raw)
    np.log1p(cd), np.log1p(cs), np.log1p(mw), np.log1p(bd),
    Xkz,  # 4 z-scored knobs
    # Physics interaction features
    np.log1p(power_proxy.clip(0)),       # activity × cap × drives / freq
    np.log1p(buf_power_proxy.clip(0)),   # buffer contribution proxy
    np.log1p(wl_proxy_nff.clip(0)),      # sqrt(n_ff × die_area)
    routing_eff.clip(0),                  # cluster_dia / FF spacing
    np.log1p(n_ff / cs.clip(1)),          # log(n_clusters approximate)
    log_ff_hpwl - np.log1p(cd),          # log(HPWL / cluster_dia)
    # Rank features (within-placement variation)
    Xrank[:,3], Xrank[:,2], Xcent[:,3], Xcent[:,2],
])

# Fix NaN/Inf
for c in range(X_phys.shape[1]):
    bad = ~np.isfinite(X_phys[:,c])
    if bad.any(): X_phys[bad,c] = np.nanmedian(X_phys[~bad,c]) if (~bad).any() else 0.0
    X_phys[:,c] = np.clip(X_phys[:,c], -1e6, 1e6)

print(f"  Physics feature dim: {X_phys.shape[1]}")

# ── Correlation analysis ───────────────────────────────────────────────────
print(f"\n{T()} Correlation with absolute targets:")
feature_names = ['log_n_ff','log_die_area','log_ff_hpwl','log_n_nets','log_act_cap','log_rel_act',
                 'frac_xor','frac_mux','frac_ff','comb/ff','ff_dens','avg_ds','frac_ds4+',
                 'sig_prob','frac_high_act','tc_std',
                 'log_cd','log_cs','log_mw','log_bd',
                 'kz0','kz1','kz2','kz3',
                 'log_power_prx','log_buf_prx','log_wl_prx','routing_eff',
                 'log_ncl','log_hpwl/cd','rank_cd','rank_cs','cent_cd','cent_cs']

print(f"  {'Feature':<20} {'rho_PW':>8} {'rho_WL':>8}")
for j, name in enumerate(feature_names[:X_phys.shape[1]]):
    rho_pw, _ = pearsonr(X_phys[:,j], y_pw_log)
    rho_wl, _ = pearsonr(X_phys[:,j], y_wl_log)
    if abs(rho_pw) > 0.3 or abs(rho_wl) > 0.3:
        print(f"  {name:<20} {rho_pw:>8.4f} {rho_wl:>8.4f}  ←")

# ── LODO for absolute prediction ──────────────────────────────────────────
def lodo_abs(X, y_log, label, cls=LGBMRegressor, kw=None):
    if kw is None:
        kw = dict(n_estimators=300, num_leaves=20, learning_rate=0.03,
                  min_child_samples=15, verbose=-1)
    dl = sorted(np.unique(designs)); mapes = []
    for held in dl:
        tr=designs!=held; te=designs==held; sc=StandardScaler()
        m=cls(**kw); m.fit(sc.fit_transform(X[tr]), y_log[tr])
        pred_log = m.predict(sc.transform(X[te]))
        pred_abs = np.exp(pred_log); true_abs = np.exp(y_log[te])
        mape = np.mean(np.abs(pred_abs - true_abs) / (true_abs + 1e-10)) * 100
        mapes.append(mape)
    mean_mape = np.mean(mapes)
    tag = ' ✓' if mean_mape < 10 else (' ~' if mean_mape < 20 else '')
    print(f"  {label}: {mapes[0]:.1f}%/{mapes[1]:.1f}%/{mapes[2]:.1f}%/{mapes[3]:.1f}%  mean={mean_mape:.1f}%{tag}")
    return mean_mape

XGB_F = dict(n_estimators=500, max_depth=4, learning_rate=0.03, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.8, verbosity=0)

print(f"\n{T()} === ABSOLUTE POWER MAPE (prior best absolute_v5: 37.8%) ===")
lodo_abs(X_phys, y_pw_log, "X_phys LGB")
lodo_abs(X_phys, y_pw_log, "X_phys XGB_F", XGBRegressor, XGB_F)

# Baseline: just log_n_ff + log_die_area + knobs
X_simple = np.column_stack([log_n_ff, log_die_area, log_ff_hpwl, np.log1p(cd), np.log1p(cs)])
X_simple[~np.isfinite(X_simple)] = 0.0
lodo_abs(X_simple, y_pw_log, "Simple(n_ff+die+hpwl+knobs) Ridge")

print(f"\n{T()} === ABSOLUTE WL MAPE (prior best absolute_v5: 21.2%) ===")
lodo_abs(X_phys, y_wl_log, "X_phys LGB")
lodo_abs(X_phys, y_wl_log, "X_phys XGB_F", XGBRegressor, XGB_F)
lodo_abs(X_simple, y_wl_log, "Simple(n_ff+die+hpwl+knobs)")

# ── DEF+SAIF for Z-SCORE prediction ─────────────────────────────────────
print(f"\n{T()} === Z-SCORE MAE with DEF+SAIF context ===")
print("DEF/SAIF are per-placement constants → add as context to X29T")

# Normalize DEF+SAIF features (per-placement constant, but useful as context)
ds_feats = np.column_stack([
    log_n_ff, log_die_area, log_ff_hpwl, log_act_cap, log_rel_act,
    frac_xor, X_def[:, def_keys.index('frac_mux')],
    X_def[:, def_keys.index('comb_per_ff')],
    X_saif[:, saif_keys.index('mean_sig_prob')],
    X_saif[:, saif_keys.index('frac_high_act')],
])
ds_feats[~np.isfinite(ds_feats)] = 0.0
ds_std = ds_feats.std(0); ds_std[ds_std<1e-9]=1.0
X_ds = ds_feats / ds_std

def lodo_z(X, y, label, cls=LGBMRegressor, kw=None):
    if kw is None:
        kw = dict(n_estimators=300, num_leaves=20, learning_rate=0.03,
                  min_child_samples=15, verbose=-1)
    dl=sorted(np.unique(designs)); maes=[]
    for held in dl:
        tr=designs!=held; te=designs==held; sc=StandardScaler()
        m=cls(**kw); m.fit(sc.fit_transform(X[tr]), y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(sc.transform(X[te]))))
    mean_mae=np.mean(maes)
    tag = ' ✓' if mean_mae<0.10 else (' ~' if mean_mae<0.15 else '')
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f}{tag}")
    return mean_mae

lodo_z(X29T, y_pw_z, "X29T LGB baseline (power)")
lodo_z(np.hstack([X29T, X_ds]), y_pw_z, "X29T+DS LGB (power)")
lodo_z(np.hstack([X29T, X_ds]), y_pw_z, "X29T+DS XGB_F (power)", XGBRegressor, XGB_F)
lodo_z(X29T, y_wl_z, "X29T LGB baseline (WL)")
lodo_z(np.hstack([X29T, X_ds]), y_wl_z, "X29T+DS LGB (WL)")
lodo_z(np.hstack([X29T, X_ds]), y_wl_z, "X29T+DS XGB_F (WL)", XGBRegressor, XGB_F)

# ── Physics formula check ────────────────────────────────────────────────
print(f"\n{T()} === PHYSICS FORMULA DIRECT PREDICTION (MAPE) ===")
print("Using formula: log(power) ≈ log(rel_act × ff_cap_proxy × n_ff/cs)")

# Physics formula prediction (no ML, just formula)
ph_pred_log = np.log(rel_act.clip(1e-10) * ff_cap_proxy.clip(1) * n_ff.clip(1) / cs.clip(1))
ph_pred_log[~np.isfinite(ph_pred_log)] = np.nanmedian(ph_pred_log[np.isfinite(ph_pred_log)])

for held in sorted(np.unique(designs)):
    te = designs==held; tr = designs!=held
    # Fit a scale and offset from training data
    A = np.column_stack([ph_pred_log[tr], np.ones(tr.sum())])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y_pw_log[tr], rcond=None)
        pred = A[:te.sum()] @ coeffs  # wrong — need te
        A_te = np.column_stack([ph_pred_log[te], np.ones(te.sum())])
        pred_te = A_te @ coeffs
        pred_abs = np.exp(pred_te); true_abs = np.exp(y_pw_log[te])
        mape = np.mean(np.abs(pred_abs - true_abs) / (true_abs + 1e-10)) * 100
        print(f"  held={held}: MAPE={mape:.1f}%")
    except: pass

print(f"\n{T()} === WL PHYSICS FORMULA ===")
wl_ph = np.log(ff_hpwl.clip(1) * np.log1p(n_ff / cs.clip(1)) / np.log1p(cd.clip(1)))
wl_ph[~np.isfinite(wl_ph)] = np.nanmedian(wl_ph[np.isfinite(wl_ph)])
for held in sorted(np.unique(designs)):
    te = designs==held; tr = designs!=held
    A = np.column_stack([wl_ph[tr], np.ones(tr.sum())])
    try:
        coeffs, _, _, _ = np.linalg.lstsq(A, y_wl_log[tr], rcond=None)
        A_te = np.column_stack([wl_ph[te], np.ones(te.sum())])
        pred_te = A_te @ coeffs
        pred_abs = np.exp(pred_te); true_abs = np.exp(y_wl_log[te])
        mape = np.mean(np.abs(pred_abs - true_abs) / (true_abs + 1e-10)) * 100
        print(f"  held={held}: MAPE={mape:.1f}%")
    except: pass

print(f"\n{T()} DONE")
