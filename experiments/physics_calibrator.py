"""
Progressive Physics Calibration for Absolute CTS Prediction.

Scientific approach:
1. Derive physics formula for WL and power from first principles
2. Fit universal constants from training designs (cross-design calibration)
3. Verify constants hold for unseen design (LODO)
4. As more designs are seen, constants converge to true physical values

Physics models:
  WL  ≈ k_wl × n_ff × sqrt(die_area/n_ff) × g(knobs)
       = k_wl × sqrt(n_ff × die_area) × g(cluster_dia, cluster_size)

  P_clock ≈ k_p × (n_ff/cluster_size) × C_buf × f + k_w × WL × C_wire × f
  P_logic ≈ k_l × n_ff × mean_sig_prob × cap_scale
  P_total ≈ P_logic + P_clock = k_l × activity_metric + k_w × WL_term + k_p × buffer_term

This is an OLS calibration: fit k from training, predict k on test.
"""

import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import RidgeCV, LinearRegression
from scipy.optimize import curve_fit

BASE = '/home/rain/CTS-Task-Aware-Clustering'
CACHE_FILE = f'{BASE}/absolute_v3_cache.pkl'
DATASET = f'{BASE}/dataset_with_def'


def load_and_merge():
    df = pd.read_csv(f'{DATASET}/unified_manifest_normalized.csv')
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    return df, cache


def build_physics_features(df, cache):
    """
    Build INTERPRETABLE physics features.

    For WL (routing length theory):
      phi_WL = sqrt(n_ff × die_area) / cluster_dia
      → WL ≈ k_wl × phi_WL  (calibrate k_wl from training)

    For Power (dynamic power theory):
      phi_PL = n_nets × mean_sig_prob   → logic activity
      phi_PC = n_ff / cluster_size       → buffer count
      phi_PW = n_ff × ff_hpwl / cluster_dia → tree routing area
      → P ≈ k_l × phi_PL + k_c × phi_PC + k_w × phi_PW
    """
    def_feats = cache['def_feats']
    saif_feats = cache['saif_feats']

    records = []
    for _, row in df.iterrows():
        pid = row['placement_id']
        df_f = def_feats.get(pid)
        sf_f = saif_feats.get(pid)
        if not df_f or not sf_f:
            continue

        n_ff = df_f['n_ff']
        die_area = df_f['die_area']
        ff_hpwl = df_f['ff_hpwl']
        n_nets = sf_f['n_nets']
        sig_prob = sf_f['mean_sig_prob']

        cd = row['cts_cluster_dia']
        cs = row['cts_cluster_size']
        mw = row['cts_max_wire']
        bd = row['cts_buf_dist']

        # ---- WL physics terms ----
        # Base: Steiner-tree-like spanning of n_ff points on die
        phi_wl_base = np.sqrt(n_ff * die_area)             # dominant term
        phi_wl_cd   = phi_wl_base / (cd + 1)               # cluster grouping reduces WL
        phi_wl_cs   = phi_wl_base / np.sqrt(cs + 1)        # larger clusters → fewer inter-cluster routes
        phi_wl_mw   = mw * n_ff / die_area                  # wire budget per FF density

        # ---- Power physics terms ----
        # Logic power: P ∝ n_cells × toggle_rate × cap_per_cell
        # Proxy: n_nets × mean_signal_prob (signal_prob ∝ toggle_activity for balanced nets)
        phi_pw_logic  = n_nets * sig_prob                   # net-level switching activity
        phi_pw_buf    = n_ff / (cs + 1)                     # estimated buffer count
        phi_pw_tree   = phi_wl_base / (cd + 1)              # WL proxy × activity
        phi_pw_act_ff = n_ff * sig_prob                     # FF-level activity

        records.append({
            'design_name': row['design_name'],
            'placement_id': pid,
            'power_total': row['power_total'],
            'wirelength': row['wirelength'],

            # WL features
            'phi_wl_base': phi_wl_base,
            'phi_wl_cd': phi_wl_cd,
            'phi_wl_cs': phi_wl_cs,
            'phi_wl_mw': phi_wl_mw,

            # Power features
            'phi_pw_logic': phi_pw_logic,
            'phi_pw_buf': phi_pw_buf,
            'phi_pw_tree': phi_pw_tree,
            'phi_pw_act_ff': phi_pw_act_ff,

            # Raw for reference
            'n_ff': n_ff,
            'die_area': die_area,
            'ff_hpwl': ff_hpwl,
            'n_nets': n_nets,
            'sig_prob': sig_prob,
            'cd': cd,
            'cs': cs,
            'mw': mw,
            'bd': bd,
        })

    return pd.DataFrame(records)


def mape(y_true, y_pred):
    return np.mean(np.abs(y_pred - y_true) / (y_true + 1e-12)) * 100


def lodo_physics_regression(feat_df):
    """
    Fit physics constants k from training designs, predict on held-out design.

    For WL: log(WL) = log(k_wl) + a1*log(phi_wl_base) + a2*log(cd+1) + a3*log(cs+1) ...
    This is a log-linear model → OLS in log space.
    """
    designs = feat_df['design_name'].unique()

    print("\n" + "=" * 65)
    print("PHYSICS CALIBRATION MODEL (log-linear OLS)")
    print("WL = k * phi_wl_base^a × (cd+1)^b × (cs+1)^c × ...")
    print("=" * 65)

    wl_feats = ['phi_wl_base', 'phi_wl_cd', 'phi_wl_cs']
    pw_feats = ['phi_pw_logic', 'phi_pw_buf', 'phi_pw_tree', 'phi_pw_act_ff']

    all_wl_mapes = []
    all_pw_mapes = []

    for held_out in designs:
        train = feat_df[feat_df['design_name'] != held_out]
        test = feat_df[feat_df['design_name'] == held_out]

        # WL: log-linear model
        X_wl_tr = np.log1p(train[wl_feats].values)
        X_wl_te = np.log1p(test[wl_feats].values)
        y_wl_tr = np.log(train['wirelength'].values)
        y_wl_te = test['wirelength'].values

        m_wl = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        m_wl.fit(X_wl_tr, y_wl_tr)
        pred_wl = np.exp(m_wl.predict(X_wl_te))
        mape_wl = mape(y_wl_te, pred_wl)

        # Power: log-linear model
        X_pw_tr = np.log1p(train[pw_feats].values)
        X_pw_te = np.log1p(test[pw_feats].values)
        y_pw_tr = np.log(train['power_total'].values)
        y_pw_te = test['power_total'].values

        m_pw = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        m_pw.fit(X_pw_tr, y_pw_tr)
        pred_pw = np.exp(m_pw.predict(X_pw_te))
        mape_pw = mape(y_pw_te, pred_pw)

        all_wl_mapes.append(mape_wl)
        all_pw_mapes.append(mape_pw)
        print(f"  Held-out {held_out}: WL_MAPE={mape_wl:.1f}%  PW_MAPE={mape_pw:.1f}%")

    print(f"  Mean → WL={np.mean(all_wl_mapes):.1f}%  PW={np.mean(all_pw_mapes):.1f}%")

    # Print calibrated constants on full data
    print("\n--- Calibrated physics constants (full data) ---")
    m_full_wl = LinearRegression()
    X_wl_full = np.log1p(feat_df[wl_feats].values)
    y_wl_full = np.log(feat_df['wirelength'].values)
    m_full_wl.fit(X_wl_full, y_wl_full)
    print(f"WL model: intercept={m_full_wl.intercept_:.3f}")
    for fname, coef in zip(wl_feats, m_full_wl.coef_):
        print(f"  {fname}: {coef:.3f}")

    m_full_pw = LinearRegression()
    X_pw_full = np.log1p(feat_df[pw_feats].values)
    y_pw_full = np.log(feat_df['power_total'].values)
    m_full_pw.fit(X_pw_full, y_pw_full)
    print(f"Power model: intercept={m_full_pw.intercept_:.3f}")
    for fname, coef in zip(pw_feats, m_full_pw.coef_):
        print(f"  {fname}: {coef:.3f}")


def check_design_constants(feat_df):
    """
    Check: does the physics formula have stable constants across designs?
    If yes → universal formula exists.
    If no → need design-level adaptation features.
    """
    print("\n" + "=" * 65)
    print("PHYSICS CONSTANT STABILITY ACROSS DESIGNS")
    print("=" * 65)

    for design in feat_df['design_name'].unique():
        d = feat_df[feat_df['design_name'] == design]
        # Compute empirical k_wl: WL / phi_wl_base
        k_wl = d['wirelength'] / (d['phi_wl_base'] + 1e-9)
        k_pw = d['power_total'] / (d['phi_pw_logic'] + 1e-9)
        print(f"{design}: k_wl={k_wl.mean():.3f}±{k_wl.std():.3f}  "
              f"k_pw={k_pw.mean():.4e}±{k_pw.std():.4e}")


def progressive_calibration_demo(feat_df):
    """
    Demonstrate progressive calibration:
    Train on designs 1, then 1+2, then 1+2+3, test on held-out.
    Shows how adding more designs improves universal formula.
    """
    print("\n" + "=" * 65)
    print("PROGRESSIVE CALIBRATION: Adding designs one by one")
    print("Held-out: ethmac")
    print("=" * 65)

    held_out = 'ethmac'
    other_designs = [d for d in feat_df['design_name'].unique() if d != held_out]
    test = feat_df[feat_df['design_name'] == held_out]

    wl_feats = ['phi_wl_base', 'phi_wl_cd', 'phi_wl_cs']
    pw_feats = ['phi_pw_logic', 'phi_pw_buf', 'phi_pw_tree', 'phi_pw_act_ff']

    for n_designs in range(1, len(other_designs) + 1):
        train_designs = other_designs[:n_designs]
        train = feat_df[feat_df['design_name'].isin(train_designs)]

        X_wl_tr = np.log1p(train[wl_feats].values)
        X_wl_te = np.log1p(test[wl_feats].values)
        m_wl = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        m_wl.fit(X_wl_tr, np.log(train['wirelength'].values))
        pred_wl = np.exp(m_wl.predict(X_wl_te))
        mape_wl = mape(test['wirelength'].values, pred_wl)

        X_pw_tr = np.log1p(train[pw_feats].values)
        X_pw_te = np.log1p(test[pw_feats].values)
        m_pw = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
        m_pw.fit(X_pw_tr, np.log(train['power_total'].values))
        pred_pw = np.exp(m_pw.predict(X_pw_te))
        mape_pw = mape(test['power_total'].values, pred_pw)

        print(f"  Train={train_designs}: WL_MAPE={mape_wl:.1f}%  PW_MAPE={mape_pw:.1f}%  "
              f"(n_rows={len(train)})")


def main():
    if not os.path.exists(CACHE_FILE):
        print(f"Cache not found at {CACHE_FILE}")
        print("Run absolute_v3.py first to build the DEF+SAIF cache.")
        return

    df, cache = load_and_merge()
    print(f"Loaded: {len(df)} rows, {df['design_name'].nunique()} designs")
    print(f"Cache: {len(cache['def_feats'])} DEF, {len(cache['saif_feats'])} SAIF entries")

    feat_df = build_physics_features(df, cache)
    print(f"Physics feature DataFrame: {feat_df.shape}")

    # Check if constants are stable
    check_design_constants(feat_df)

    # Progressive calibration demo
    progressive_calibration_demo(feat_df)

    # Full LODO evaluation
    lodo_physics_regression(feat_df)


if __name__ == '__main__':
    main()
