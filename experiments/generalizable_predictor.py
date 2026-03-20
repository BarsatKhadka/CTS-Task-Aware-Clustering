"""
Generalizable CTS Outcome Predictor
====================================
Goal: <0.05 MAE (per-placement z-score) on unseen designs.

Key insight: training CSV uses GLOBAL z-scores; test file uses PER-DESIGN z-scores.
Solution:  re-normalize targets per-placement during training so the model learns
           RELATIVE (within-placement) CTS knob sensitivity — which generalizes
           across design families.

Approaches tried (in order):
  A1 — LightGBM, knob-only features, per-placement z-scores       [quick sanity]
  A2 — LightGBM, rich features (60-dim cache), per-placement z    [main baseline]
  A3 — LightGBM, augmented features + physics interactions         [enhanced]
  A4 — GNN on timing-path FF graph + MLP head                     [structural]
  A5 — Stacked ensemble (LGB + XGB + Ridge)                       [best combo]

Run:
  python generalizable_predictor.py --approach A1   # quick sanity (~2s)
  python generalizable_predictor.py --approach A2   # main baseline
  python generalizable_predictor.py --approach A3   # augmented features
  python generalizable_predictor.py --approach A4   # GNN
  python generalizable_predictor.py --approach A5   # ensemble
  python generalizable_predictor.py --approach all  # everything
  python generalizable_predictor.py --lodo          # leave-one-design-out CV
"""

from __future__ import annotations
import argparse
import os
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE = Path(__file__).parent
TRAIN_MANIFEST = BASE / "dataset_with_def" / "unified_manifest_normalized.csv"
TEST_MANIFEST  = BASE / "dataset_with_def" / "unified_manifest_normalized_test.csv"
TRAIN_CACHE    = BASE / "cache_train_features.pkl"
TEST_CACHE     = BASE / "cache_test_features.pkl"

TARGET_COLS    = ["skew_setup", "power_total", "wirelength"]
TARGET_NAMES   = ["skew", "power", "wl"]
KNOB_Z_COLS    = ["z_cts_max_wire", "z_cts_buf_dist", "z_cts_cluster_size", "z_cts_cluster_dia"]
KNOB_RAW_COLS  = ["cts_max_wire", "cts_buf_dist", "cts_cluster_size", "cts_cluster_dia"]


# ─────────────────────────────────────────────────────────────────────────────
# 1. Normalization
# ─────────────────────────────────────────────────────────────────────────────

def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any target is NaN."""
    return df.dropna(subset=TARGET_COLS).reset_index(drop=True)


def per_placement_z(df: pd.DataFrame) -> np.ndarray:
    """
    Z-score each target within its placement_id.
    Returns [N, 3] array matching test normalization.
    If a placement has std=0 for a target (all configs identical), z=0.
    """
    df = df.reset_index(drop=True)
    Y = np.zeros((len(df), 3), dtype=np.float32)
    for pid, idx in df.groupby("placement_id").groups.items():
        int_idx = list(idx)
        for j, col in enumerate(TARGET_COLS):
            vals = df.loc[int_idx, col].values.astype(np.float64)
            mu, sigma = vals.mean(), vals.std()
            if sigma < 1e-10:
                Y[int_idx, j] = 0.0
            else:
                Y[int_idx, j] = ((vals - mu) / sigma).astype(np.float32)
    return Y


def per_design_z(df: pd.DataFrame) -> np.ndarray:
    """Z-score each target within its design_name (used for test)."""
    df = df.reset_index(drop=True)
    Y = np.zeros((len(df), 3), dtype=np.float32)
    for dname, idx in df.groupby("design_name").groups.items():
        int_idx = list(idx)
        for j, col in enumerate(TARGET_COLS):
            vals = df.loc[int_idx, col].values.astype(np.float64)
            mu, sigma = vals.mean(), vals.std()
            if sigma < 1e-10:
                Y[int_idx, j] = 0.0
            else:
                Y[int_idx, j] = ((vals - mu) / sigma).astype(np.float32)
    return Y


# ─────────────────────────────────────────────────────────────────────────────
# 2. Feature builders
# ─────────────────────────────────────────────────────────────────────────────

def knob_features(df: pd.DataFrame) -> np.ndarray:
    """11-dim CTS knob features (4 raw-norm + 4 z-scored + 3 interactions)."""
    mw = df["cts_max_wire"].values / 200.0
    bd = df["cts_buf_dist"].values / 110.0
    cs = df["cts_cluster_size"].values / 21.0
    cd = df["cts_cluster_dia"].values / 52.0
    z_mw = df["z_cts_max_wire"].values
    z_bd = df["z_cts_buf_dist"].values
    z_cs = df["z_cts_cluster_size"].values
    z_cd = df["z_cts_cluster_dia"].values
    return np.column_stack([
        mw, bd, cs, cd,
        z_mw, z_bd, z_cs, z_cd,
        z_bd * z_mw,   # buffer spacing × wire length
        z_cs * z_cd,   # cluster size × diameter
        z_bd * z_cs,   # spacing × size
    ]).astype(np.float32)


def augmented_knob_features(df: pd.DataFrame) -> np.ndarray:
    """Extended knob features with higher-order interactions."""
    base = knob_features(df)
    z_mw = df["z_cts_max_wire"].values
    z_bd = df["z_cts_buf_dist"].values
    z_cs = df["z_cts_cluster_size"].values
    z_cd = df["z_cts_cluster_dia"].values
    extra = np.column_stack([
        z_mw**2, z_bd**2, z_cs**2, z_cd**2,
        z_mw * z_cs, z_mw * z_cd,
        z_bd * z_cd,
        z_mw * z_bd * z_cs,
    ]).astype(np.float32)
    return np.concatenate([base, extra], axis=1)


def load_cached_features(train=True) -> tuple[np.ndarray, pd.DataFrame]:
    """Load pre-extracted 60-dim placement features from cache."""
    cache = TRAIN_CACHE if train else TEST_CACHE
    with open(cache, "rb") as f:
        d = pickle.load(f)
    return d["X"], d["df"]


def placement_metadata_features(df: pd.DataFrame) -> np.ndarray:
    """
    Design-level meta features from manifest columns.
    These capture absolute design size (helps generalize).
    """
    feats = np.column_stack([
        df.get("aspect_ratio", pd.Series(1.0, index=df.index)).values,
        df.get("core_util", pd.Series(60.0, index=df.index)).values / 100.0,
        df.get("density", pd.Series(0.7, index=df.index)).values,
        (df.get("io_mode", pd.Series(0, index=df.index)).values).astype(float),
        (df.get("time_driven", pd.Series(0, index=df.index)).values).astype(float),
        (df.get("routability_driven", pd.Series(0, index=df.index)).values).astype(float),
    ]).astype(np.float32)
    return feats


# ─────────────────────────────────────────────────────────────────────────────
# 3. Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(preds: np.ndarray, Y_true: np.ndarray, label: str = ""):
    """Print MAE for each target."""
    maes = {}
    for j, name in enumerate(TARGET_NAMES):
        mae = mean_absolute_error(Y_true[:, j], preds[:, j])
        maes[name] = mae
    mean_mae = np.mean(list(maes.values()))
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}MAE — skew:{maes['skew']:.4f}  power:{maes['power']:.4f}  wl:{maes['wl']:.4f}  mean:{mean_mae:.4f}")
    return maes


def lodo_cv(build_model_fn, X_train, Y_train, df_train, label=""):
    """Leave-one-design-out cross-validation."""
    designs = df_train["design_name"].unique()
    all_preds = np.zeros_like(Y_train)
    for d in designs:
        mask = df_train["design_name"].values == d
        X_tr, Y_tr = X_train[~mask], Y_train[~mask]
        X_va, Y_va = X_train[mask],  Y_train[mask]
        model = build_model_fn()
        model.fit(X_tr, Y_tr)
        all_preds[mask] = model.predict(X_va)
    evaluate(all_preds, Y_train, f"LODO {label}")
    return all_preds


# ─────────────────────────────────────────────────────────────────────────────
# 4. Multi-output wrapper
# ─────────────────────────────────────────────────────────────────────────────

class MultiOutputModel:
    """Wraps 3 single-output models for skew/power/wl."""
    def __init__(self, models):
        self.models = models  # list of 3 models

    def fit(self, X, Y):
        for j, m in enumerate(self.models):
            m.fit(X, Y[:, j])
        return self

    def predict(self, X) -> np.ndarray:
        return np.column_stack([m.predict(X) for m in self.models])


# ─────────────────────────────────────────────────────────────────────────────
# 5. Approach A1: LightGBM, knob-only
# ─────────────────────────────────────────────────────────────────────────────

def approach_A1(do_lodo=False):
    """Sanity check: can knob features alone predict per-placement z-scores?"""
    import lightgbm as lgb
    print("\n" + "="*60)
    print("A1: LightGBM — knob features only, per-placement z-scores")
    print("="*60)

    df_tr = drop_nan_rows(pd.read_csv(TRAIN_MANIFEST))
    df_te = drop_nan_rows(pd.read_csv(TEST_MANIFEST))

    Y_tr = per_placement_z(df_tr)
    Y_te = per_design_z(df_te)

    X_tr = knob_features(df_tr)
    X_te = knob_features(df_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    def make_model():
        models = [lgb.LGBMRegressor(
            n_estimators=400, learning_rate=0.05, num_leaves=31,
            min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1,
        ) for _ in range(3)]
        return MultiOutputModel(models)

    if do_lodo:
        lodo_cv(make_model, X_tr_s, Y_tr, df_tr, "A1-knob")

    model = make_model()
    model.fit(X_tr_s, Y_tr)
    preds = model.predict(X_te_s)
    evaluate(preds, Y_te, "A1 test")
    return preds, Y_te


# ─────────────────────────────────────────────────────────────────────────────
# 6. Approach A2: LightGBM + rich features (60-dim cache)
# ─────────────────────────────────────────────────────────────────────────────

def approach_A2(do_lodo=False):
    """Main baseline: knobs + cached 60-dim placement features."""
    import lightgbm as lgb
    print("\n" + "="*60)
    print("A2: LightGBM — knobs + 60-dim rich features, per-placement z")
    print("="*60)

    X_cache_tr, df_tr = load_cached_features(train=True)
    X_cache_te, df_te = load_cached_features(train=False)

    Y_tr = per_placement_z(df_tr)
    Y_te = per_design_z(df_te)

    K_tr = knob_features(df_tr)
    K_te = knob_features(df_te)

    # Cached placement features (49 dims: timing+DEF+SAIF) + knobs (11) = 60 total
    # Note: cache already includes knob features in last 11 dims; we replace them
    # with our own version to avoid duplication
    X_tr = np.concatenate([X_cache_tr[:, :49], K_tr], axis=1)
    X_te = np.concatenate([X_cache_te[:, :49], K_te], axis=1)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    def make_model():
        models = [lgb.LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=63,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        ) for _ in range(3)]
        return MultiOutputModel(models)

    if do_lodo:
        lodo_cv(make_model, X_tr_s, Y_tr, df_tr, "A2-rich")

    model = make_model()
    model.fit(X_tr_s, Y_tr)
    preds = model.predict(X_te_s)
    evaluate(preds, Y_te, "A2 test")
    return preds, Y_te, model, sc


# ─────────────────────────────────────────────────────────────────────────────
# 7. Approach A3: Augmented features + physics interactions
# ─────────────────────────────────────────────────────────────────────────────

def approach_A3(do_lodo=False):
    """
    Enhanced: rich features + augmented knob features + cross-features.
    Physics motivation:
      - skew ∝ max_wire × FF_spread (long wires + spread-out FFs → high skew)
      - power ∝ buf_dist × activity (more buffers × more switching → more power)
      - wl   ∝ cluster_size × HPWL  (big clusters × wide spread → long tree)
    """
    import lightgbm as lgb
    print("\n" + "="*60)
    print("A3: LightGBM — augmented features + physics cross-terms")
    print("="*60)

    X_cache_tr, df_tr = load_cached_features(train=True)
    X_cache_te, df_te = load_cached_features(train=False)

    Y_tr = per_placement_z(df_tr)
    Y_te = per_design_z(df_te)

    K_tr = augmented_knob_features(df_tr)   # 19-dim
    K_te = augmented_knob_features(df_te)

    # Physics cross-terms: knob × placement interaction
    # Features 4 (HPWL from DEF) and 22 (die_area proxy) are at specific indices
    # in the 60-dim cache: [0-16] timing, [17-38] DEF, [39-48] SAIF, [49-59] knobs
    # DEF: index 17=log_nff, 21=hpwl, 24=sx, 25=sy (from rich_features.py)
    # SAIF: index 39=tc_log_total
    def add_cross(X_cache, K):
        z_mw = K[:, 4]   # z_cts_max_wire (col 4 in augmented_knob_features)
        z_bd = K[:, 5]   # z_cts_buf_dist
        z_cs = K[:, 6]   # z_cts_cluster_size
        z_cd = K[:, 7]   # z_cts_cluster_dia
        hpwl     = X_cache[:, 21]   # DEF HPWL
        log_nff  = X_cache[:, 17]   # log(n_ff)
        sx       = X_cache[:, 24]   # FF spread x
        sy       = X_cache[:, 25]   # FF spread y
        act      = X_cache[:, 39]   # log(total toggle count)
        cross = np.column_stack([
            z_mw * hpwl,          # skew physics: wire × spread
            z_mw * log_nff,       # wire × n_ff
            z_bd * act,           # power physics: spacing × activity
            z_cs * hpwl,          # wl physics: cluster × spread
            z_cd * (sx + sy),     # cluster dia × spatial spread
            z_cs * log_nff,       # cluster size × n_ff
            hpwl * log_nff,       # design size proxy (already in cache but explicit)
        ]).astype(np.float32)
        return np.concatenate([X_cache, K, cross], axis=1)

    X_tr = add_cross(X_cache_tr, K_tr)
    X_te = add_cross(X_cache_te, K_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    def make_model():
        models = [lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.02, num_leaves=127,
            min_child_samples=5, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.05, reg_lambda=0.05,
            random_state=42, verbose=-1,
        ) for _ in range(3)]
        return MultiOutputModel(models)

    if do_lodo:
        lodo_cv(make_model, X_tr_s, Y_tr, df_tr, "A3-augmented")

    model = make_model()
    model.fit(X_tr_s, Y_tr)
    preds = model.predict(X_te_s)
    evaluate(preds, Y_te, "A3 test")
    return preds, Y_te, model, sc


# ─────────────────────────────────────────────────────────────────────────────
# 8. Approach A4: GNN on timing-path FF graph
# ─────────────────────────────────────────────────────────────────────────────

def build_ff_graph(timing_csv: str, ff_positions: dict) -> tuple:
    """
    Build FF-to-FF graph from timing paths.
    Returns (edge_index [2, E], node_feats [N, d], node_ids [N]).
    ff_positions: {ff_name: (x_norm, y_norm, toggle_count)}
    """
    import torch
    from torch_geometric.data import Data

    df = pd.read_csv(timing_csv)
    ffs = sorted(set(df["launch_flop"].tolist()) | set(df["capture_flop"].tolist()))
    ff_idx = {ff: i for i, ff in enumerate(ffs)}

    # Node features: [x, y, toggle] or zeros if not in positions
    feats = []
    for ff in ffs:
        pos = ff_positions.get(ff, (0.5, 0.5, 0.0))
        feats.append(pos)
    node_feats = torch.tensor(feats, dtype=torch.float32)

    # Edges
    src, dst = [], []
    for _, row in df.iterrows():
        u, v = row["launch_flop"], row["capture_flop"]
        if u in ff_idx and v in ff_idx:
            src.append(ff_idx[u])
            dst.append(ff_idx[v])
    if not src:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor([src, dst], dtype=torch.long)

    return Data(x=node_feats, edge_index=edge_index, num_nodes=len(ffs))


def _build_gnn_module():
    """Lazily build GNN class after torch is confirmed importable."""
    import torch
    import torch.nn as nn
    from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool

    class GNNModel(nn.Module):
        def __init__(self, in_dim=3, hidden=64, embed_dim=32, knob_dim=11, n_targets=3):
            super().__init__()
            self.conv1 = SAGEConv(in_dim, hidden)
            self.conv2 = SAGEConv(hidden, hidden)
            self.conv3 = SAGEConv(hidden, embed_dim)
            self.bn1 = nn.BatchNorm1d(hidden)
            self.bn2 = nn.BatchNorm1d(hidden)
            self.pool_head = nn.Linear(embed_dim * 2, embed_dim)
            self.head = nn.Sequential(
                nn.Linear(embed_dim + knob_dim, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, n_targets),
            )

        def forward(self, data, knobs):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = torch.relu(self.bn1(self.conv1(x, edge_index)))
            x = torch.relu(self.bn2(self.conv2(x, edge_index)))
            x = self.conv3(x, edge_index)
            # Global pooling: mean + max for richer representation
            x_mean = global_mean_pool(x, batch)
            x_max  = global_max_pool(x, batch)
            x_pool = torch.relu(self.pool_head(torch.cat([x_mean, x_max], dim=1)))
            out = self.head(torch.cat([x_pool, knobs], dim=1))
            return out

    return GNNModel


def approach_A4(do_lodo=False, epochs=100, n_samples=500):
    """
    GNN on timing-path FF graph.
    Due to parsing cost, use a subset of placements for quick validation.
    n_samples: number of training placements to parse (set -1 for all).
    """
    import torch
    import torch.nn as nn
    from torch_geometric.data import DataLoader
    from torch_geometric.data import Batch

    print("\n" + "="*60)
    print("A4: GNN on FF timing-path graph")
    print("="*60)

    GNNModel = _build_gnn_module()

    df_tr = pd.read_csv(TRAIN_MANIFEST)
    df_te = pd.read_csv(TEST_MANIFEST)

    Y_tr_full = per_placement_z(df_tr)
    Y_te      = per_design_z(df_te)

    # Sample training placements
    pids = df_tr["placement_id"].unique()
    if n_samples > 0:
        pids = pids[:n_samples]
        mask = df_tr["placement_id"].isin(pids)
        df_tr = df_tr[mask].reset_index(drop=True)
        Y_tr_full = Y_tr_full[mask.values]

    K_tr = knob_features(df_tr)
    K_te = knob_features(df_te)

    PLACEMENT_DIR = BASE / "dataset_with_def" / "placement_files"

    def parse_placement(placement_id: str, design: str) -> dict | None:
        """Parse DEF+SAIF for a placement, return FF positions."""
        pdir = PLACEMENT_DIR / placement_id
        def_f = pdir / f"{design}.def"
        saif_f = pdir / f"{design}.saif"
        timing_f = pdir / "timing_paths.csv"
        if not def_f.exists():
            return None

        # Quick parse: extract FF positions from DEF
        ff_pos = {}
        die_w, die_h = 1, 1
        with open(def_f) as f:
            in_comp = False
            import re
            ff_pat = re.compile(r"sky130_fd_sc_hd__(?:df|sdff|edf|sdf)", re.I)
            for line in f:
                ls = line.strip()
                if ls.startswith("DIEAREA"):
                    pts = re.findall(r"\(\s*(-?\d+)\s+(-?\d+)\s*\)", ls)
                    if len(pts) >= 2:
                        die_w = max(abs(int(pts[1][0]) - int(pts[0][0])), 1)
                        die_h = max(abs(int(pts[1][1]) - int(pts[0][1])), 1)
                if ls.startswith("COMPONENTS") and "END" not in ls:
                    in_comp = True; continue
                if ls.startswith("END COMPONENTS"):
                    in_comp = False; continue
                if in_comp and ls.startswith("- "):
                    parts = ls.split()
                    if len(parts) < 3: continue
                    if not ff_pat.match(parts[2]): continue
                    inst = parts[1]
                    m = re.search(r"(?:PLACED|FIXED)\s+\(\s*(-?\d+)\s+(-?\d+)", ls)
                    if m:
                        x = int(m.group(1)) / die_w
                        y = int(m.group(2)) / die_h
                        ff_pos[inst] = (x, y, 0.0)

        if not ff_pos or not timing_f.exists():
            return None
        return {"ff_pos": ff_pos, "timing_csv": str(timing_f)}

    # Build graphs for each unique placement
    print("Parsing placement files for GNN graphs...")
    pid_to_graph = {}
    parsed, skipped = 0, 0
    for pid in df_tr["placement_id"].unique():
        design = df_tr[df_tr["placement_id"] == pid]["design_name"].iloc[0]
        info = parse_placement(pid, design)
        if info is None:
            skipped += 1
            continue
        try:
            g = build_ff_graph(info["timing_csv"], info["ff_pos"])
            pid_to_graph[pid] = g
            parsed += 1
        except Exception:
            skipped += 1
        if (parsed + skipped) % 50 == 0:
            print(f"  {parsed} parsed, {skipped} skipped")

    print(f"Total: {parsed} graphs built, {skipped} skipped")

    # Build graph for test placements
    for pid in df_te["placement_id"].unique():
        design = df_te[df_te["placement_id"] == pid]["design_name"].iloc[0]
        info = parse_placement(pid, design)
        if info:
            try:
                g = build_ff_graph(info["timing_csv"], info["ff_pos"])
                pid_to_graph[pid] = g
            except Exception:
                pass

    # Build training dataset (only rows with available graph)
    valid_tr = df_tr["placement_id"].isin(pid_to_graph).values
    df_tr_v = df_tr[valid_tr].reset_index(drop=True)
    Y_tr = Y_tr_full[valid_tr]
    K_tr_v = K_tr[valid_tr]

    print(f"Training samples with graphs: {len(df_tr_v)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Normalize knobs
    sc_k = StandardScaler()
    K_tr_s = sc_k.fit_transform(K_tr_v)

    model = GNNModel(in_dim=3, hidden=64, embed_dim=32, knob_dim=11, n_targets=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    batch_size = 64
    n = len(df_tr_v)
    pids_tr = df_tr_v["placement_id"].values
    Y_tr_t = torch.tensor(Y_tr, dtype=torch.float32).to(device)
    K_tr_t = torch.tensor(K_tr_s, dtype=torch.float32).to(device)

    model.train()
    for ep in range(epochs):
        perm = np.random.permutation(n)
        ep_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            graphs = [pid_to_graph[pids_tr[i]] for i in idx]
            batch_g = Batch.from_data_list(graphs).to(device)
            knobs_b = K_tr_t[idx]
            y_b = Y_tr_t[idx]
            optimizer.zero_grad()
            pred = model(batch_g, knobs_b)
            loss = nn.MSELoss()(pred, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss += loss.item() * len(idx)
        scheduler.step()
        if (ep + 1) % 20 == 0:
            print(f"  Epoch {ep+1}/{epochs}  loss={ep_loss/n:.4f}")

    # Evaluate on test
    model.eval()
    with torch.no_grad():
        valid_te = df_te["placement_id"].isin(pid_to_graph).values
        df_te_v = df_te[valid_te].reset_index(drop=True)
        K_te_v = K_te[valid_te]
        Y_te_v = per_design_z(df_te_v)
        K_te_s = sc_k.transform(K_te_v)
        pids_te = df_te_v["placement_id"].values
        graphs_te = [pid_to_graph[p] for p in pids_te]
        batch_te = Batch.from_data_list(graphs_te).to(device)
        K_te_t = torch.tensor(K_te_s, dtype=torch.float32).to(device)
        preds = model(batch_te, K_te_t).cpu().numpy()

    evaluate(preds, Y_te_v, "A4 test (GNN)")
    return preds, Y_te_v


# ─────────────────────────────────────────────────────────────────────────────
# 9. Approach A5: Stacked ensemble (LGB + XGB + Ridge)
# ─────────────────────────────────────────────────────────────────────────────

def approach_A5(do_lodo=False):
    """
    Stacked ensemble using A3 features.
    Base learners: LightGBM + XGBoost
    Meta learner: Ridge regression (avoids overfitting)
    Uses 5-fold CV for out-of-fold training of meta-learner.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import GroupKFold
    print("\n" + "="*60)
    print("A5: Stacked Ensemble (LGB+XGB → Ridge)")
    print("="*60)

    X_cache_tr, df_tr = load_cached_features(train=True)
    X_cache_te, df_te = load_cached_features(train=False)

    Y_tr = per_placement_z(df_tr)
    Y_te = per_design_z(df_te)

    K_tr = augmented_knob_features(df_tr)
    K_te = augmented_knob_features(df_te)

    def add_cross(X_cache, K):
        z_mw = K[:, 4]; z_bd = K[:, 5]; z_cs = K[:, 6]; z_cd = K[:, 7]
        hpwl = X_cache[:, 21]; log_nff = X_cache[:, 17]
        sx = X_cache[:, 24]; sy = X_cache[:, 25]; act = X_cache[:, 39]
        cross = np.column_stack([
            z_mw * hpwl, z_mw * log_nff, z_bd * act,
            z_cs * hpwl, z_cd * (sx+sy), z_cs * log_nff, hpwl * log_nff,
        ]).astype(np.float32)
        return np.concatenate([X_cache, K, cross], axis=1)

    X_tr = add_cross(X_cache_tr, K_tr)
    X_te = add_cross(X_cache_te, K_te)

    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)

    # Group by design for GroupKFold (prevent leakage)
    groups = df_tr["design_name"].values
    gkf = GroupKFold(n_splits=4)

    def lgb_params():
        return dict(n_estimators=800, learning_rate=0.02, num_leaves=127,
                    min_child_samples=5, subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.05, reg_lambda=0.05, random_state=42, verbose=-1)

    def xgb_params():
        return dict(n_estimators=600, learning_rate=0.02, max_depth=7,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.05, reg_lambda=0.05, random_state=42,
                    tree_method="hist", verbosity=0)

    all_preds = []
    for j, name in enumerate(TARGET_NAMES):
        print(f"\n  Stacking for {name}...")
        oof_lgb = np.zeros(len(X_tr_s))
        oof_xgb = np.zeros(len(X_tr_s))

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_tr_s, Y_tr[:, j], groups)):
            # LightGBM
            m_lgb = lgb.LGBMRegressor(**lgb_params())
            m_lgb.fit(X_tr_s[tr_idx], Y_tr[tr_idx, j])
            oof_lgb[va_idx] = m_lgb.predict(X_tr_s[va_idx])

            # XGBoost
            m_xgb = xgb.XGBRegressor(**xgb_params())
            m_xgb.fit(X_tr_s[tr_idx], Y_tr[tr_idx, j])
            oof_xgb[va_idx] = m_xgb.predict(X_tr_s[va_idx])

        oof_mae_lgb = mean_absolute_error(Y_tr[:, j], oof_lgb)
        oof_mae_xgb = mean_absolute_error(Y_tr[:, j], oof_xgb)
        print(f"    OOF MAE — LGB:{oof_mae_lgb:.4f}  XGB:{oof_mae_xgb:.4f}")

        # Meta-learner on OOF
        meta_X = np.column_stack([oof_lgb, oof_xgb])
        meta = Ridge(alpha=1.0)
        meta.fit(meta_X, Y_tr[:, j])

        # Retrain on full training set
        m_lgb_full = lgb.LGBMRegressor(**lgb_params())
        m_lgb_full.fit(X_tr_s, Y_tr[:, j])
        m_xgb_full = xgb.XGBRegressor(**xgb_params())
        m_xgb_full.fit(X_tr_s, Y_tr[:, j])

        te_lgb = m_lgb_full.predict(X_te_s)
        te_xgb = m_xgb_full.predict(X_te_s)
        te_meta_X = np.column_stack([te_lgb, te_xgb])
        te_pred = meta.predict(te_meta_X)
        all_preds.append(te_pred)

    preds = np.column_stack(all_preds)
    evaluate(preds, Y_te, "A5 test (ensemble)")
    return preds, Y_te


# ─────────────────────────────────────────────────────────────────────────────
# 10. Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--approach", default="A2",
                    choices=["A1", "A2", "A3", "A4", "A5", "all"])
    ap.add_argument("--lodo", action="store_true", help="Run LODO cross-validation")
    ap.add_argument("--gnn-samples", type=int, default=200,
                    help="Placements to parse for GNN (default 200; -1=all)")
    ap.add_argument("--gnn-epochs", type=int, default=80)
    args = ap.parse_args()

    to_run = ["A1", "A2", "A3", "A5"] if args.approach == "all" else [args.approach]
    if "A4" in to_run or args.approach == "A4":
        to_run = [a for a in to_run if a != "A4"]
        to_run.append("A4")

    results = {}
    for a in to_run:
        if a == "A1":
            results["A1"] = approach_A1(do_lodo=args.lodo)
        elif a == "A2":
            results["A2"] = approach_A2(do_lodo=args.lodo)
        elif a == "A3":
            results["A3"] = approach_A3(do_lodo=args.lodo)
        elif a == "A4":
            results["A4"] = approach_A4(
                do_lodo=args.lodo,
                epochs=args.gnn_epochs,
                n_samples=args.gnn_samples,
            )
        elif a == "A5":
            results["A5"] = approach_A5(do_lodo=args.lodo)

    # Need torch for A4 global usage
    try:
        import torch
    except ImportError:
        pass
