"""
dkl_exp.py — Deep Kernel Learning (DKL) + Stochastic Variational GP

Key hypotheses:
1. Power/WL are smooth physical functions (P ∝ 1/cd). LGB creates rigid boundaries.
   A GP with ARD kernel enforces physical smoothness → better generalization.
2. A neural network feature extractor (NN→latent→GP) captures nonlinear interactions
   that LGB misses (Deep Kernel Learning = DKL).

Architecture:
  - Feature extractor: 3-layer MLP → 16-dim latent space
  - GP: SVGP with RBF/ARD kernel in latent space
  - Inducing points: m=200 (for scalability)
  - Training: maximize ELBO (variational evidence lower bound)

Evaluations:
1. Exact GP on X_best (baseline)
2. SVGP on X_best (scalable GP)
3. DKL (NN + SVGP) on X_best

LODO evaluation (4-fold: leave-one-design-out)
Target: beat LGB 0.2027 on power MAE
"""

import pickle, time, warnings, numpy as np
import torch
import torch.nn as nn
import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
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
print("Deep Kernel Learning (SVGP + NN) for CTS Prediction")
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

# Compute simulation features
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

# ── SVGP Model ─────────────────────────────────────────────────────────────
class SVGP(ApproximateGP):
    """Stochastic Variational GP with RBF kernel."""
    def __init__(self, inducing_points):
        vdist = CholeskyVariationalDistribution(inducing_points.size(0))
        vstrat = VariationalStrategy(self, inducing_points, vdist, learn_inducing_locations=True)
        super().__init__(vstrat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=inducing_points.shape[1])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKLModel(nn.Module):
    """Deep Kernel Learning: NN feature extractor + SVGP (wrapped in nn.Module)."""
    def __init__(self, inducing_points, n_in, latent_dim=16, hidden=64):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim)
        )
        # Compute latent inducing points
        with torch.no_grad():
            ip_latent = self.feature_extractor(inducing_points)
        # Build SVGP in latent space
        self.gp = _LatentSVGP(ip_latent, latent_dim)

    def forward(self, x):
        projected = self.feature_extractor(x)
        return self.gp(projected)


class _LatentSVGP(ApproximateGP):
    """GP in latent space (used by DKLModel)."""
    def __init__(self, ip, latent_dim):
        vdist = CholeskyVariationalDistribution(ip.size(0))
        vstrat = VariationalStrategy(self, ip, vdist, learn_inducing_locations=True)
        super().__init__(vstrat)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=latent_dim)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_svgp_lodo(X, y, label, use_dkl=False, latent_dim=16, hidden=64,
                    n_inducing=200, n_epochs=50, batch_size=512, lr=0.01):
    """LODO evaluation with SVGP or DKL."""
    dl = sorted(np.unique(designs)); maes = []

    for held in dl:
        tr_m = designs != held; te_m = designs == held
        sc = StandardScaler()
        X_tr_np = sc.fit_transform(X[tr_m])
        X_te_np = sc.transform(X[te_m])
        y_tr = y[tr_m]; y_te = y[te_m]

        X_tr = torch.FloatTensor(X_tr_np).to(device)
        X_te = torch.FloatTensor(X_te_np).to(device)
        y_tr_t = torch.FloatTensor(y_tr).to(device)

        # Select inducing points (k-means style: random subset)
        idx = torch.randperm(len(X_tr))[:n_inducing]
        ip = X_tr[idx].clone()

        if use_dkl:
            model = DKLModel(ip, X.shape[1], latent_dim=latent_dim, hidden=hidden).to(device)
            params = list(model.feature_extractor.parameters()) + list(model.parameters())
        else:
            model = SVGP(ip).to(device)
            params = list(model.parameters())

        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=len(X_tr))

        optimizer = torch.optim.Adam(params + list(likelihood.parameters()), lr=lr)

        model.train(); likelihood.train()
        n_tr = len(X_tr)
        for epoch in range(n_epochs):
            perm = torch.randperm(n_tr)
            epoch_loss = 0.0
            for i in range(0, n_tr, batch_size):
                idx = perm[i:i+batch_size]
                optimizer.zero_grad()
                output = model(X_tr[idx])
                loss = -mll(output, y_tr_t[idx])
                loss.backward(); optimizer.step()
                epoch_loss += loss.item()

        model.eval(); likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = likelihood(model(X_te)).mean.cpu().numpy()
        maes.append(mean_absolute_error(y_te, pred))

    tag = ' ✓' if np.mean(maes) < 0.10 else ''
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  "
          f"mean={np.mean(maes):.4f}{tag}")
    return maes


# ── LGB baseline for reference ─────────────────────────────────────────────
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
    print(f"  {label}: {maes[0]:.4f}/{maes[1]:.4f}/{maes[2]:.4f}/{maes[3]:.4f}  mean={np.mean(maes):.4f}")
    return maes

print(f"\n{T()} === POWER: LGB BASELINE ===")
lodo_lgb(X_best, y_pw, "LGB X_best (0.2027)")

print(f"\n{T()} === POWER: SVGP (no NN) ===")
train_svgp_lodo(X_best, y_pw, "SVGP(200ip,50ep,lr=0.01)", use_dkl=False,
                n_inducing=200, n_epochs=50)
train_svgp_lodo(X_best, y_pw, "SVGP(300ip,80ep,lr=0.01)", use_dkl=False,
                n_inducing=300, n_epochs=80)

print(f"\n{T()} === POWER: DKL (NN + SVGP) ===")
train_svgp_lodo(X_best, y_pw, "DKL(16dim,200ip,50ep)", use_dkl=True,
                latent_dim=16, hidden=64, n_inducing=200, n_epochs=50)
train_svgp_lodo(X_best, y_pw, "DKL(32dim,200ip,80ep)", use_dkl=True,
                latent_dim=32, hidden=128, n_inducing=200, n_epochs=80)
train_svgp_lodo(X_best, y_pw, "DKL(8dim,100ip,50ep)", use_dkl=True,
                latent_dim=8, hidden=32, n_inducing=100, n_epochs=50)

print(f"\n{T()} === WL: SVGP + DKL ===")
lodo_lgb(X_best, y_wl, "LGB X_best WL (0.2337)")
train_svgp_lodo(X_best, y_wl, "SVGP WL", use_dkl=False,
                n_inducing=200, n_epochs=50)
train_svgp_lodo(X_best, y_wl, "DKL WL (16dim,200ip)", use_dkl=True,
                latent_dim=16, hidden=64, n_inducing=200, n_epochs=50)

print(f"\n{T()} DONE")
