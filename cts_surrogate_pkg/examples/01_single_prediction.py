"""
Example 01 — Single-call prediction
====================================
Load the surrogate and predict all four CTS outcomes for one
(placement, knob-configuration) pair.

Run from anywhere:
    python3 examples/01_single_prediction.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cts_surrogate import CTSSurrogate
import pandas as pd

# ── Load in one line ──────────────────────────────────────────────────────────
model = CTSSurrogate.from_package()

# ── Pick a known placement to demo ───────────────────────────────────────────
manifest = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                    'data', 'manifest.csv'))
row = manifest[manifest['design_name'] == 'aes'].iloc[0]
pid = row['placement_id']

# ── Predict with four different knob configurations ───────────────────────────
configs = [
    dict(cd=37, cs=27, mw=132, bd=119),   # typical mid-range
    dict(cd=70, cs=28, mw=270, bd=100),   # high cd+mw → low power, low skew
    dict(cd=40, cs=15, mw=140, bd=80),    # low cd → more buffers → higher power
    dict(cd=55, cs=20, mw=220, bd=120),   # balanced
]

print(f"Placement: {pid}")
print(f"{'cd':>4} {'cs':>4} {'mw':>5} {'bd':>4}  "
      f"{'Power(mW)':>10}  {'WL(mm)':>8}  {'Skew(z)':>8}  {'Hold':>6}")
print("─" * 65)
for knobs in configs:
    pred = model.predict(pid, **knobs)
    print(f"{knobs['cd']:>4}  {knobs['cs']:>4}  {knobs['mw']:>5}  {knobs['bd']:>4}  "
          f"{pred.power_mW:>10.3f}  {pred.wl_mm:>8.2f}  "
          f"{pred.skew_z:>8.4f}  {pred.hold_vio:>6.1f}")

# ── What the CTSPrediction dataclass contains ─────────────────────────────────
print("\nFull prediction object:")
pred = model.predict(pid, **configs[0])
print(f"  pred.power_mW  = {pred.power_mW:.3f} mW")
print(f"  pred.wl_mm     = {pred.wl_mm:.3f} mm")
print(f"  pred.skew_z    = {pred.skew_z:.4f}  (per-placement z-score)")
print(f"  pred.skew_ns   = {pred.skew_ns}  (absolute ns, if sk_mu/sig provided)")
print(f"  pred.hold_vio  = {pred.hold_vio:.2f} violations")
print(f"  pred.pw_norm   = {pred.pw_norm:.4f}  (physics normalizer)")
print(f"  pred.wl_norm   = {pred.wl_norm:.4f}")

# ── Compare with ground truth ─────────────────────────────────────────────────
cd0 = int(row['cts_cluster_dia']); cs0 = int(row['cts_cluster_size'])
mw0 = int(row['cts_max_wire']);    bd0 = int(row['cts_buf_dist'])
pred0 = model.predict(pid, cd=cd0, cs=cs0, mw=mw0, bd=bd0)
print(f"\nGround-truth comparison (knobs from manifest):")
print(f"  Predicted: power={pred0.power_mW:.2f}mW  WL={pred0.wl_mm:.1f}mm  skew_z={pred0.skew_z:.4f}")
print(f"  True:      power={row['power_total']*1000:.2f}mW  "
      f"WL={row['wirelength']/1000:.1f}mm  skew={row['skew_setup']:.4f}ns")
