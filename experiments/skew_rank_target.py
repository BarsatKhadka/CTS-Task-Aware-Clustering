"""
skew_rank_target.py — Test RANK targets for skew (the missing piece)

Key finding from RESEARCH_LOG: rank targets are better for noisy skew.
Prior best: X29T + rank targets + XGB_SK (min_child_weight=15) = 0.2369

Tests: X29T rank, X29T+kNN rank, X29TK+phys rank
All with XGB_SK (min_child_weight=15) config.
"""

import pickle, time, warnings
import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Skew Rank Target Test")
print("=" * 70)

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)
with open(f'{BASE}/ff_positions_cache.pkl', 'rb') as f:
    ff = pickle.load(f)

pids = df['placement_id'].values; designs = df['design_name'].values; n = len(pids)

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

def rank_mae(pred, ytrue, pids_):
    pr=np.zeros(len(pids_)); tr=np.zeros(len(pids_))
    for pid in np.unique(pids_):
        m=pids_==pid; idx=np.where(m)[0]
        if len(idx)>1: pr[idx]=rank_within(pred[idx]); tr[idx]=rank_within(ytrue[idx])
        else: pr[idx]=tr[idx]=0.5
    return mean_absolute_error(tr, pr)

def lodo(X, y, label, cls=XGBRegressor, kw=None):
    if kw is None:
        kw = dict(n_estimators=300, learning_rate=0.03, max_depth=4,
                  min_child_weight=15, subsample=0.8, colsample_bytree=0.8, verbosity=0)
    dl=sorted(np.unique(designs)); maes=[]
    for held in dl:
        tr=designs!=held; te=designs==held; sc=StandardScaler()
        m=cls(**kw); m.fit(sc.fit_transform(X[tr]), y[tr])
        pred=m.predict(sc.transform(X[te])); maes.append(rank_mae(pred, y[te], pids[te]))
    mean_mae=np.mean(maes)
    s='NEW_BEST' if mean_mae < 0.2369 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={mean_mae:.4f} {s}")
    return mean_mae

# Build features
knob_cols=['cts_max_wire','cts_buf_dist','cts_cluster_size','cts_cluster_dia']
Xraw=df[knob_cols].values.astype(np.float32); Xkz=X_cache[:,72:76]
raw_max=Xraw.max(0)+1e-6
Xrank=np.zeros((n,4),np.float32); Xcent=np.zeros_like(Xrank)
Xrng=np.zeros_like(Xrank); Xmn=np.zeros_like(Xrank)
for pid in np.unique(pids):
    m=pids==pid; idx=np.where(m)[0]
    for j in range(4):
        v=Xraw[idx,j]; Xrank[idx,j]=rank_within(v); Xcent[idx,j]=(v-v.mean())/raw_max[j]
        Xrng[idx,j]=v.std()/raw_max[j]; Xmn[idx,j]=v.mean()/raw_max[j]
Xplc=df[['core_util','density','aspect_ratio']].values.astype(np.float32)
Xplc_n=Xplc/(Xplc.max(0)+1e-9)
cd=Xraw[:,3]; cs=Xraw[:,2]; mw=Xraw[:,0]; bd=Xraw[:,1]
util=Xplc[:,0]/100; density_=Xplc[:,1]; aspect=Xplc[:,2]
Xinter=np.column_stack([cd*util,mw*density_,cd/(density_+0.01),cd*aspect,
                         Xrank[:,3]*util,Xrank[:,2]*util])
X29=np.hstack([Xkz,Xrank,Xcent,Xplc_n,Xinter,Xrng,Xmn])

X_tight=np.zeros((n,20),np.float32)
for i,pid in enumerate(pids):
    v=tp.get(pid)
    if v is not None: X_tight[i,:20]=np.array(v,np.float32)[:20]
tp_std=X_tight.std(0); tp_std[tp_std<1e-9]=1.0
X29T=np.hstack([X29, X_tight/tp_std])

# kNN features
print(f"{T()} Building kNN features...")
knn_feats={}
for pid in np.unique(pids):
    pi=ff.get(pid)
    if pi is None or pi.get('ff_norm') is None: knn_feats[pid]=np.zeros(20,np.float32); continue
    xy=pi['ff_norm']
    if len(xy)<4: knn_feats[pid]=np.zeros(20,np.float32); continue
    tree=cKDTree(xy); d,_=tree.query(xy,k=min(9,len(xy)))
    k1=d[:,1] if d.shape[1]>1 else np.zeros(len(xy))
    k4=d[:,min(4,d.shape[1]-1)]; cen_=xy.mean(0); cd_=np.linalg.norm(xy-cen_,axis=1)
    knn_feats[pid]=np.array([k1.mean(),k1.std(),np.percentile(k1,90),np.percentile(k1,95),
        np.percentile(k1,99),k1.max(),k4.mean(),k4.std(),np.percentile(k4,90),
        d[:,min(8,d.shape[1]-1)].mean(),cd_.mean(),cd_.std(),np.percentile(cd_,50),
        np.percentile(cd_,90),np.percentile(cd_,95),np.percentile(cd_,99),cd_.max(),
        np.percentile(cd_,90)/(cd_.mean()+1e-8),np.percentile(cd_,99)/(cd_.mean()+1e-8),
        cd_.std()/(cd_.mean()+1e-8)],np.float32)
Xknn=np.array([knn_feats.get(pid,np.zeros(20)) for pid in pids],np.float32)
ks=Xknn.std(0); ks[ks<1e-9]=1.0; Xknn_n=Xknn/ks

# Physics interactions for skew
knn1_mean=Xknn[:,0]; knn1_p99=Xknn[:,4]; knn4_mean=Xknn[:,6]
outlier_r=Xknn[:,18]; tail_r=Xknn[:,17]
raw_phys=np.column_stack([
    cd/(knn1_mean*1000+1e-4), cd/(knn1_p99*1000+1e-4), cd/(knn4_mean*1000+1e-4),
    bd/(knn1_mean*1000+1e-4), Xrank[:,3]*outlier_r, Xrank[:,3]*tail_r,
    outlier_r, tail_r, Xknn[:,19],
])
for c in range(raw_phys.shape[1]):
    bad=~np.isfinite(raw_phys[:,c])
    if bad.any(): raw_phys[bad,c]=np.nanmedian(raw_phys[~bad,c]) if (~bad).any() else 0.0
    raw_phys[:,c]=np.clip(raw_phys[:,c],-1e6,1e6)
phys_rank=np.zeros_like(raw_phys); phys_cent=np.zeros_like(raw_phys)
g=np.abs(raw_phys).max(0)+1e-9
for pid in np.unique(pids):
    m=pids==pid; idx=np.where(m)[0]
    for j in range(raw_phys.shape[1]):
        v=raw_phys[idx,j]; phys_rank[idx,j]=rank_within(v) if v.max()>v.min() else 0.5
        phys_cent[idx,j]=(v-v.mean())/g[j]
Xphys=np.hstack([raw_phys/g, phys_rank, phys_cent])

for arr in [X29, X29T, Xknn_n, Xphys]:
    for c in range(arr.shape[1]):
        bad=~np.isfinite(arr[:,c])
        if bad.any(): arr[bad,c]=np.nanmedian(arr[~bad,c]) if (~bad).any() else 0.0

# RANK targets (key!)
Y_rank=np.zeros((n,3),np.float32)
for pid in np.unique(pids):
    m=pids==pid; idx=np.where(m)[0]
    for j in range(3): Y_rank[idx,j]=rank_within(Y_cache[idx,j])

print(f"{T()} Features: X29={X29.shape[1]}, X29T={X29T.shape[1]}, kNN={Xknn_n.shape[1]}, Xphys={Xphys.shape[1]}")

XGB_SK = dict(n_estimators=300, learning_rate=0.03, max_depth=4,
              min_child_weight=15, subsample=0.8, colsample_bytree=0.8, verbosity=0)
XGB_SK2 = dict(n_estimators=500, learning_rate=0.03, max_depth=4,
               min_child_weight=15, subsample=0.8, colsample_bytree=0.8, verbosity=0)
LGB_SK = dict(n_estimators=300, num_leaves=15, learning_rate=0.03,
              min_child_samples=15, verbose=-1)

print(f"\n{T()} === SKEW: RANK targets (prior best 0.2369) ===")
lodo(X29,  Y_rank[:,0], "X29  rank XGB_SK")
lodo(X29T, Y_rank[:,0], "X29T rank XGB_SK (replicate prior)")
lodo(np.hstack([X29T, Xknn_n]), Y_rank[:,0], "X29T+kNN rank XGB_SK")
lodo(np.hstack([X29T, Xphys]),  Y_rank[:,0], "X29T+phys rank XGB_SK")
lodo(np.hstack([X29T, Xknn_n, Xphys]), Y_rank[:,0], "X29T+kNN+phys rank XGB_SK")
lodo(np.hstack([X29T, Xknn_n, Xphys]), Y_rank[:,0], "X29T+kNN+phys rank XGB_SK2",
     XGBRegressor, XGB_SK2)

print(f"\n{T()} === SKEW: ZSCORE targets (for comparison) ===")
lodo(X29T, Y_cache[:,0], "X29T z XGB_SK")
lodo(np.hstack([X29T, Xknn_n]), Y_cache[:,0], "X29T+kNN z XGB_SK")

print(f"\n{T()} DONE")
