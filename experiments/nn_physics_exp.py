"""
nn_physics_exp.py - Neural network with physics-guided architecture

Key hypothesis: LGB misses nonlinear interactions between cluster_dia and
placement context (n_ff, die_area, FF density). A small MLP can capture these.

Architecture:
  - Input: X29T (49 features, per-run)
  - Hidden: 3 layers with BatchNorm + Dropout (for regularization)
  - Output: z_power, z_wl (multi-task — shared representation)
  - Physics prior: final layer initialized with -0.92 × z_cluster_dia basis

LODO evaluation with same 4-fold as before.

Also tests physics-guided ensemble: NN + LGB + mean → reduced variance.
"""

import pickle, time, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMRegressor

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"

print("=" * 70)
print("Neural Network with Physics Architecture")
print("=" * 70)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)

pids = df['placement_id'].values; designs = df['design_name'].values; n = len(pids)

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

# Also add β_cd × z_cluster_dia as explicit physics feature
cd_z_pp = np.zeros(n, np.float32)  # per-placement z-scored cluster_dia
for pid in np.unique(pids):
    idx = np.where(pids==pid)[0]
    v = Xraw[idx,3]
    cd_z_pp[idx] = (v - v.mean()) / max(v.std(), 1e-8)

# physics_feat: β_nominal × z_cluster_dia (approximation of z_power)
physics_pw = -0.92 * cd_z_pp  # nominal β_cd = -0.92
physics_wl = -0.78 * cd_z_pp  # nominal β for WL (estimated from prior analysis)

X_with_physics = np.hstack([X29T, cd_z_pp.reshape(-1,1),
                              physics_pw.reshape(-1,1), physics_wl.reshape(-1,1)])  # 52-dim

for arr in [X29T, X_with_physics]:
    for c in range(arr.shape[1]):
        bad = ~np.isfinite(arr[:,c])
        if bad.any(): arr[bad,c] = 0.0

y_pw = Y_cache[:, 1]; y_wl = Y_cache[:, 2]

# ── Neural Network ────────────────────────────────────────────────────────
class PhysicsNet(nn.Module):
    def __init__(self, n_in, n_out=2, hidden=128, n_layers=3, dropout=0.3):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train_lodo_nn(X, y_list, label, n_epochs=200, batch_size=256, lr=1e-3,
                  hidden=64, n_layers=2, dropout=0.2, weight_decay=1e-4):
    """Multi-task NN for all targets in y_list."""
    dl = sorted(np.unique(designs))
    target_names = ['power', 'wl'] if len(y_list) == 2 else ['target']
    all_maes = [[] for _ in y_list]

    for held in dl:
        tr_m = designs != held; te_m = designs == held
        sc = StandardScaler()
        X_tr = torch.FloatTensor(sc.fit_transform(X[tr_m])).to(device)
        X_te = torch.FloatTensor(sc.transform(X[te_m])).to(device)
        y_tr = torch.FloatTensor(np.column_stack(y_list)[tr_m]).to(device)
        y_te_np = np.column_stack(y_list)[te_m]

        net = PhysicsNet(X.shape[1], n_out=len(y_list), hidden=hidden,
                         n_layers=n_layers, dropout=dropout).to(device)
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs)

        n_tr = len(X_tr)
        for epoch in range(n_epochs):
            net.train()
            # Shuffle
            perm = torch.randperm(n_tr)
            total_loss = 0.0
            for i in range(0, n_tr, batch_size):
                idx = perm[i:i+batch_size]
                out = net(X_tr[idx])
                loss = torch.mean(torch.abs(out - y_tr[idx]))
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total_loss += loss.item()
            scheduler.step()

        net.eval()
        with torch.no_grad():
            pred_te = net(X_te).cpu().numpy()

        for j, y_te_j in enumerate(y_te_np.T):
            all_maes[j].append(mean_absolute_error(y_te_j, pred_te[:, j]))

    for j, (maes, name) in enumerate(zip(all_maes, target_names)):
        tag = ' ✓' if np.mean(maes) < 0.10 else ''
        print(f"  {label} [{name}]: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  "
              f"mean={np.mean(maes):.4f}{tag}")
    return all_maes

# ── LGB baseline (for comparison) ─────────────────────────────────────────
print(f"\n{T()} === LGB BASELINES ===")
def lodo_lgb(X, y, label):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()
        m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                          min_child_samples=15, verbose=-1)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(sc.transform(X[te]))))
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={np.mean(maes):.4f}")
    return maes

lgb_pw = lodo_lgb(X29T, y_pw, "X29T LGB power")
lgb_wl = lodo_lgb(X29T, y_wl, "X29T LGB wl")

# ── NN experiments ─────────────────────────────────────────────────────────
print(f"\n{T()} === NEURAL NETWORK (multi-task) ===")
train_lodo_nn(X29T, [y_pw, y_wl], "X29T NN(64×2, 200ep)", n_epochs=200, hidden=64, n_layers=2)
train_lodo_nn(X29T, [y_pw, y_wl], "X29T NN(128×3, 200ep)", n_epochs=200, hidden=128, n_layers=3)
train_lodo_nn(X_with_physics, [y_pw, y_wl], "X29T+phys NN(64×2, 200ep)", n_epochs=200, hidden=64, n_layers=2)
train_lodo_nn(X_with_physics, [y_pw, y_wl], "X29T+phys NN(128×3, 300ep)", n_epochs=300, hidden=128, n_layers=3)

# ── Ensemble: LGB + NN ────────────────────────────────────────────────────
print(f"\n{T()} === ENSEMBLE: LGB + NN ===")
def lodo_ensemble(X, y, label, nn_weight=0.5, n_epochs=200, hidden=64, n_layers=2):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs!=held; te = designs==held; sc = StandardScaler()

        # LGB
        m_lgb = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                               min_child_samples=15, verbose=-1)
        m_lgb.fit(sc.fit_transform(X[tr]), y[tr])
        pred_lgb = m_lgb.predict(sc.transform(X[te]))

        # NN
        X_tr = torch.FloatTensor(sc.transform(X[tr])).to(device)
        X_te = torch.FloatTensor(sc.transform(X[te])).to(device)
        y_tr = torch.FloatTensor(y[tr]).to(device)

        net = PhysicsNet(X.shape[1], n_out=1, hidden=hidden, n_layers=n_layers).to(device)
        optimizer = optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        for epoch in range(n_epochs):
            net.train()
            perm = torch.randperm(len(X_tr))
            for i in range(0, len(X_tr), 256):
                idx = perm[i:i+256]
                out = net(X_tr[idx]).squeeze()
                loss = torch.mean(torch.abs(out - y_tr[idx]))
                optimizer.zero_grad(); loss.backward(); optimizer.step()

        net.eval()
        with torch.no_grad():
            pred_nn = net(X_te).cpu().numpy().squeeze()

        pred_ens = (1 - nn_weight) * pred_lgb + nn_weight * pred_nn
        maes.append(mean_absolute_error(y[te], pred_ens))

    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={np.mean(maes):.4f}")
    return maes

lodo_ensemble(X29T, y_pw, "Ensemble(0.5) power", nn_weight=0.5)
lodo_ensemble(X29T, y_pw, "Ensemble(0.3) power", nn_weight=0.3)
lodo_ensemble(X29T, y_wl, "Ensemble(0.5) wl", nn_weight=0.5)
lodo_ensemble(X29T, y_wl, "Ensemble(0.3) wl", nn_weight=0.3)

print(f"\n{T()} DONE")
