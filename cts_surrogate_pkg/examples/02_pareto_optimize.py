"""
Example 02 — Pareto optimization
===================================
Compare three multi-objective search strategies over the CTS knob space.
Shows hypervolume and best-objective metrics for each method.

Run:
    python3 examples/02_pareto_optimize.py
"""

import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cts_surrogate import CTSSurrogate
import pandas as pd
import numpy as np

model = CTSSurrogate.from_package()

manifest = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'data', 'manifest.csv'))

# Use one placement per design as representative
DESIGNS = ['aes', 'picorv32', 'ethmac', 'sha256']
pids = {d: manifest[manifest['design_name']==d]['placement_id'].iloc[
            len(manifest[manifest['design_name']==d])//2]
        for d in DESIGNS}

# ── Run all three methods on AES ──────────────────────────────────────────────
pid = pids['aes']
print(f"Design: AES   pid: {pid}")
print()

methods = [
    ('random',   5000, 'Uniform random (legacy baseline)'),
    ('nsga2',    5000, 'NSGA-II — default, 3-6× more Pareto solutions'),
    ('nsga2',     500, 'NSGA-II low-budget (10× fewer evals)'),
    ('bayesian',  500, 'Bayesian (Optuna) — best for minimum-skew'),
]

results = []
for method, n, desc in methods:
    t0 = time.perf_counter()
    df = model.optimize(pid, n=n, method=method)
    elapsed = time.perf_counter() - t0

    results.append(dict(
        label=f'{method}-{n}',
        desc=desc,
        n_pareto=len(df),
        best_pw=df['power_mW'].min(),
        best_sk=df['skew_z'].min(),
        best_hold=df['hold_vio'].min(),
        time_s=elapsed,
    ))
    print(f"[{method}-{n}] {desc}")
    print(f"  {len(df)} Pareto solutions in {elapsed:.2f}s")
    print(f"  Best power: {df['power_mW'].min():.3f} mW  "
          f"Best skew_z: {df['skew_z'].min():.4f}  "
          f"Best hold: {df['hold_vio'].min():.2f}")
    print(f"  Top-3 solutions:")
    for _, r in df.head(3).iterrows():
        print(f"    cd={r.cd:.0f} cs={r.cs:.0f} mw={r.mw:.0f} bd={r.bd:.0f} → "
              f"pw={r.power_mW:.3f}mW  wl={r.wl_mm:.1f}mm  sk={r.skew_z:.4f}  hv={r.hold_vio:.1f}")
    print()

# ── Summary table ──────────────────────────────────────────────────────────────
print("=" * 75)
print(f"{'Method':<18}  {'N':>5}  {'Time':>6}  {'|Pareto|':>8}  "
      f"{'BestPw(mW)':>10}  {'BestSk(z)':>10}")
print("─" * 75)
baseline_sk = results[0]['best_sk']
for r in results:
    sk_gain = r['best_sk'] - baseline_sk
    gain_str = f"({sk_gain:+.3f})" if r['label'] != results[0]['label'] else "  baseline"
    print(f"{r['label']:<18}  {r['n']:>5}  {r['time_s']:>5.1f}s  {r['n_pareto']:>8}  "
          f"{r['best_pw']:>10.3f}  {r['best_sk']:>10.4f} {gain_str}")

# ── Sensitivity analysis ──────────────────────────────────────────────────────
print("\n" + "=" * 75)
print("Sensitivity analysis — ∂(target)/∂(knob) at base knobs (cd=50 cs=20 mw=200 bd=100):")
sens = model.sensitivity(pid)
print(sens.to_string(float_format=lambda x: f"{x:+.3f}"))
print("\nInterpretation:")
print("  mw → skew:  dominant lever (larger max_wire allows better equalization)")
print("  cd → power: buffer count proxy (larger cluster → fewer buffers → lower power)")
print("  cd → hold:  secondary hold-violation driver")
