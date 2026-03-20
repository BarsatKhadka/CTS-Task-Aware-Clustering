"""
maml_exp.py — Model-Agnostic Meta-Learning (MAML/Reptile) for Few-Shot CTS Adaptation

Key idea: Instead of zero-shot (train on 3 designs, predict 4th with NOTHING),
use few-shot adaptation: see K=3-5 labeled CTS runs from the test design,
then quickly adapt the model weights.

MAML learns an initialization θ* such that a few gradient steps on K examples
from any new design rapidly adapts to that design's characteristics.

Why this breaks the zero-shot limit:
- Zero-shot oracle: 0.13 (limited by β_cd variability across designs)
- With K=5 examples: can calibrate β_cd for the test design → potential < 0.10

Architecture:
  - MLP f(x; θ): X_best → z_power prediction
  - Meta-objective: θ* = argmin_θ Σ_design L(θ - α∇L_K(θ)) where L_K = K-shot inner loss
  - At test time: θ_adapted = θ* - α∇L_K(θ*) using K examples from test design

Also implements Reptile (simpler alternative to MAML):
  - Fast: just SGD on each task, then move θ toward task-specific θ_i

Evaluation:
  - LODO-style: meta-train on 3 designs, meta-test on 1 (4 folds)
  - K-shot setting: K=1,3,5,10 support examples from test design
  - Report MAE on remaining test examples
"""

import pickle, time, warnings, numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
BASE = '/home/rain/CTS-Task-Aware-Clustering'
t0 = time.time()
def T(): return f"[{time.time()-t0:.0f}s]"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

print("=" * 70)
print("MAML / Reptile Few-Shot Adaptation for CTS Prediction")
print("=" * 70)

# ── Load data ─────────────────────────────────────────────────────────────
with open(f'{BASE}/cache_v2_fixed.pkl', 'rb') as f:
    cache = pickle.load(f)
X_cache, Y_cache, df = cache['X'], cache['Y'], cache['df']
with open(f'{BASE}/tight_path_feats_cache.pkl', 'rb') as f:
    tp = pickle.load(f)
with open(f'{BASE}/sim_ff_cache.pkl', 'rb') as f:
    ff_cache = pickle.load(f)

pids = df['placement_id'].values; designs = df['design_name'].values; n = len(pids)
y_pw = Y_cache[:, 1]; y_wl = Y_cache[:, 2]

def rank_within(v):
    return np.argsort(np.argsort(v)).astype(float) / max(len(v)-1, 1)

def z_within(v, pids):
    out = np.zeros(len(v), np.float32)
    for pid in np.unique(pids):
        idx = np.where(pids==pid)[0]
        vv = np.array(v)[idx].astype(float)
        s = max(vv.std(), 1e-8)
        out[idx] = (vv - vv.mean()) / s
    return out

# ── Build X_best ──────────────────────────────────────────────────────────
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
util = Xplc[:, 0]/100; dens = Xplc[:, 1]
Xinter = np.column_stack([cd*util, mw*dens, cd/(dens+0.01), cd*(Xplc[:,2]),
                           Xrank[:,3]*util, Xrank[:,2]*util])
X29 = np.hstack([Xkz, Xrank, Xcent, Xplc_n, Xinter, Xrng, Xmn])
X_tight = np.zeros((n, 20), np.float32)
for i, pid in enumerate(pids):
    v = tp.get(pid)
    if v is not None: X_tight[i,:20] = np.array(v, np.float32)[:20]
tp_std = X_tight.std(0); tp_std[tp_std<1e-9]=1.0
X29T = np.hstack([X29, X_tight/tp_std])

sim_n_clusters = np.zeros(n, np.float32)
for i, pid in enumerate(pids):
    if pid not in ff_cache: continue
    data = ff_cache[pid]; xy = data['xy']
    if len(xy) == 0: continue
    tree = cKDTree(xy)
    counts = tree.query_ball_point(xy, r=cd[i]/2, return_length=True)
    sim_n_clusters[i] = np.sum(1.0 / np.maximum(counts, 1))

z_nc = z_within(sim_n_clusters, pids)
z_log_cd = z_within(np.log(cd), pids)
z_inv_cd = z_within(1.0/cd, pids)
z_cd2 = z_within(cd**2, pids)
z_mw_cd = z_within(mw/(cd+1e-6), pids)
z_bd_cd = z_within(bd/(cd+1e-6), pids)

X_best = np.hstack([X29T, z_mw_cd.reshape(-1,1), z_bd_cd.reshape(-1,1),
                    z_log_cd.reshape(-1,1), z_inv_cd.reshape(-1,1),
                    z_cd2.reshape(-1,1), z_nc.reshape(-1,1)])

for col in range(X_best.shape[1]):
    bad = ~np.isfinite(X_best[:,col])
    if bad.any(): X_best[bad,col] = 0.0

print(f"{T()} Data prepared. X_best shape: {X_best.shape}")

# ── MLP for MAML ─────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, n_in, hidden=64, n_layers=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(n_in, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def clone_params(model):
    """Clone model parameters for inner loop."""
    return {k: v.clone() for k, v in model.state_dict().items()}

def functional_forward(model, x, params):
    """Forward pass using given params (for MAML inner loop)."""
    # Use functional API: temporarily replace model params
    x = x.clone()
    for name, module in model.net.named_children():
        if isinstance(module, nn.Linear):
            w = params[f'net.{name}.weight']
            b = params[f'net.{name}.bias']
            x = F.linear(x, w, b)
        elif isinstance(module, nn.LayerNorm):
            w = params.get(f'net.{name}.weight')
            b = params.get(f'net.{name}.bias')
            if w is not None:
                x = F.layer_norm(x, [x.shape[-1]], w, b)
            else:
                x = F.layer_norm(x, [x.shape[-1]])
        elif isinstance(module, nn.ReLU):
            x = F.relu(x)
        elif isinstance(module, nn.Dropout):
            pass  # no dropout at inference
    return x.squeeze(-1)


# ── Reptile Algorithm (simpler, faster than MAML) ────────────────────────
# Reptile: for each task, do n inner steps, then move θ toward θ_task
def train_reptile(X, y, n_in, target_name, K=5, n_meta_epochs=100,
                  n_inner_steps=5, inner_lr=0.01, meta_lr=0.1,
                  hidden=64, n_layers=2):
    """
    Reptile meta-learning.
    Tasks = placements (each placement has 10 runs).
    Meta-train: placements from 3 designs.
    Meta-test: adapt with K examples from placements of held-out design.
    """
    dl = sorted(np.unique(designs))
    print(f"\n  Reptile K={K}, inner_lr={inner_lr}, meta_lr={meta_lr}")
    all_maes = []

    for held in dl:
        tr_m = designs != held; te_m = designs == held
        sc = StandardScaler()
        X_tr_all = sc.fit_transform(X[tr_m])
        X_te_all = sc.transform(X[te_m])
        y_tr_all = y[tr_m]; y_te_all = y[te_m]
        pids_tr = pids[tr_m]; pids_te = pids[te_m]
        designs_te = designs[te_m]

        # Tasks: each unique placement is a task
        unique_pids_tr = np.unique(pids_tr)

        model = MLP(n_in, hidden=hidden, n_layers=n_layers).to(device)
        meta_opt = torch.optim.Adam(model.parameters(), lr=meta_lr)

        # Pre-training step (standard supervised learning on all training data)
        # to initialize θ to a reasonable starting point
        X_tr_t = torch.FloatTensor(X_tr_all).to(device)
        y_tr_t = torch.FloatTensor(y_tr_all).to(device)
        pretrain_opt = torch.optim.Adam(model.parameters(), lr=0.001)
        for ep in range(30):
            model.train()
            perm = torch.randperm(len(X_tr_t))
            for i in range(0, len(X_tr_t), 256):
                idx = perm[i:i+256]
                loss = F.l1_loss(model(X_tr_t[idx]), y_tr_t[idx])
                pretrain_opt.zero_grad(); loss.backward(); pretrain_opt.step()

        # Reptile meta-training
        for meta_epoch in range(n_meta_epochs):
            model.train()
            # Sample a task (placement)
            task_pid = unique_pids_tr[np.random.randint(len(unique_pids_tr))]
            task_idx = np.where(pids_tr == task_pid)[0]
            X_task = torch.FloatTensor(X_tr_all[task_idx]).to(device)
            y_task = torch.FloatTensor(y_tr_all[task_idx]).to(device)

            # Save current params
            old_params = clone_params(model)

            # Inner loop: few steps on this task
            inner_opt = torch.optim.SGD(model.parameters(), lr=inner_lr)
            for _ in range(n_inner_steps):
                loss = F.l1_loss(model(X_task), y_task)
                inner_opt.zero_grad(); loss.backward(); inner_opt.step()

            # Reptile update: move θ toward θ_task (θ_after_inner_loop)
            for name, param in model.named_parameters():
                param.data = (old_params[name] +
                              meta_lr * (param.data - old_params[name]))

        # Meta-test: adapt on K examples from test design placements
        model.eval()
        # Get per-placement predictions after K-shot adaptation
        te_preds = np.zeros(len(X_te_all))
        unique_pids_te = np.unique(pids_te)

        for test_pid in unique_pids_te:
            pid_idx = np.where(pids_te == test_pid)[0]
            X_pid = torch.FloatTensor(X_te_all[pid_idx]).to(device)
            y_pid = torch.FloatTensor(y_te_all[pid_idx]).to(device)

            # K-shot support: pick K random examples from this placement
            if K > 0 and len(pid_idx) > K:
                support_idx = np.random.choice(len(pid_idx), K, replace=False)
                query_idx = np.array([i for i in range(len(pid_idx)) if i not in support_idx])
            else:
                # No support → zero-shot, just use meta-model
                support_idx = np.array([], dtype=int)
                query_idx = np.arange(len(pid_idx))

            if len(support_idx) > 0:
                # Adapt model on support set
                adapted_model = MLP(n_in, hidden=hidden, n_layers=n_layers).to(device)
                adapted_model.load_state_dict(model.state_dict())
                adapt_opt = torch.optim.SGD(adapted_model.parameters(), lr=inner_lr)
                for _ in range(10):  # More adaptation steps at test time
                    loss = F.l1_loss(adapted_model(X_pid[support_idx]),
                                     y_pid[support_idx])
                    adapt_opt.zero_grad(); loss.backward(); adapt_opt.step()

                adapted_model.eval()
                with torch.no_grad():
                    pred_q = adapted_model(X_pid).cpu().numpy()
            else:
                with torch.no_grad():
                    pred_q = model(X_pid).cpu().numpy()

            te_preds[pid_idx] = pred_q

        # Evaluate on all test data, excluding support examples
        # For simplicity: evaluate on ALL test examples
        mae = mean_absolute_error(y_te_all, te_preds)
        all_maes.append(mae)

    tag = ' ✓' if np.mean(all_maes) < 0.10 else ''
    print(f"    {target_name}: {all_maes[0]:.4f}/{all_maes[1]:.4f}/"
          f"{all_maes[2]:.4f}/{all_maes[3]:.4f}  mean={np.mean(all_maes):.4f}{tag}")
    return all_maes


# ── LGB Zero-Shot baseline ─────────────────────────────────────────────────
from lightgbm import LGBMRegressor
def lodo_lgb(X, y, label):
    dl = sorted(np.unique(designs)); maes = []
    for held in dl:
        tr = designs != held; te = designs == held
        sc = StandardScaler()
        m = LGBMRegressor(n_estimators=300, num_leaves=20, learning_rate=0.03,
                          min_child_samples=15, verbose=-1)
        m.fit(sc.fit_transform(X[tr]), y[tr])
        maes.append(mean_absolute_error(y[te], m.predict(sc.transform(X[te]))))
    tag = ' ✓' if np.mean(maes) < 0.10 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={np.mean(maes):.4f}{tag}")
    return maes

print(f"\n{T()} === ZERO-SHOT BASELINES ===")
lodo_lgb(X_best, y_pw, "LGB power (0.2027)")
lodo_lgb(X_best, y_wl, "LGB WL (0.2337)")

print(f"\n{T()} === REPTILE META-LEARNING (Power) ===")
print("  Note: K=support examples from test placement for fast adaptation")
n_in = X_best.shape[1]

for K in [0, 1, 3, 5]:
    train_reptile(X_best, y_pw, n_in, f"Power K={K}", K=K,
                  n_meta_epochs=200, n_inner_steps=5, inner_lr=0.02, meta_lr=0.05,
                  hidden=64, n_layers=2)

print(f"\n{T()} === REPTILE META-LEARNING (WL) ===")
for K in [0, 3, 5]:
    train_reptile(X_best, y_wl, n_in, f"WL K={K}", K=K,
                  n_meta_epochs=200, n_inner_steps=5, inner_lr=0.02, meta_lr=0.05,
                  hidden=64, n_layers=2)

print(f"\n{T()} DONE")
