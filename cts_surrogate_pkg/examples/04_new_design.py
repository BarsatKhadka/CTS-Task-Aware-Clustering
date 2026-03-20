"""
Example 04 — Registering a new design from raw files
======================================================
Shows how to add a completely new, unseen design by pointing to its
DEF, SAIF, and timing_paths.csv files.

After add_design(), the placement is available for predict(), optimize(),
and sensitivity() just like any training design.

Run:
    python3 examples/04_new_design.py --def path/to/design.def \
                                       --saif path/to/design.saif \
                                       --timing path/to/timing_paths.csv \
                                       --tclk 10.0

Or without arguments to see the API structure with a training-set demo.
"""

import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cts_surrogate import CTSSurrogate

# ── Parse CLI arguments ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Add a new design to the surrogate.')
parser.add_argument('--def',    dest='def_path',    default=None)
parser.add_argument('--saif',   dest='saif_path',   default=None)
parser.add_argument('--timing', dest='timing_path', default=None)
parser.add_argument('--tclk',   dest='t_clk', type=float, default=10.0,
                    help='Clock period in ns (default 10.0)')
parser.add_argument('--name',   dest='pid', default='my_design_v1',
                    help='Placement identifier (default: my_design_v1)')
args = parser.parse_args()

model = CTSSurrogate.from_package()

if args.def_path:
    # ── Real new design ────────────────────────────────────────────────────────
    print(f"Registering: {args.pid}")
    model.add_design(
        name_or_pid=args.pid,
        def_path=args.def_path,
        saif_path=args.saif_path,
        timing_path=args.timing_path,
        t_clk=args.t_clk,
    )
    print(f"  Parsed DEF + SAIF + timing. Ready for prediction.")

    # Default sweep to explore the design
    print(f"\nPareto optimization (NSGA-II, 1000 evals):")
    pareto = model.optimize(args.pid, n=1000)
    print(f"  Found {len(pareto)} non-dominated solutions")
    print(f"  Best power:  {pareto['power_mW'].min():.3f} mW")
    print(f"  Best skew_z: {pareto['skew_z'].min():.4f}")
    print(f"\nTop 5 Pareto configs:")
    print(pareto.head(5).to_string(index=False))

    print(f"\nSensitivity at base knobs (cd=50 cs=20 mw=200 bd=100):")
    print(model.sensitivity(args.pid).to_string(float_format=lambda x: f"{x:+.3f}"))

else:
    # ── Demo mode: show the API structure with a known design ─────────────────
    import pandas as pd
    PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    manifest = pd.read_csv(os.path.join(PKG, 'data', 'manifest.csv'))
    pid = manifest[manifest['design_name'] == 'picorv32']['placement_id'].iloc[5]

    print("Demo mode (no --def provided). Using picorv32 from cache.")
    print()
    print("API to add a new design:")
    print()
    print("  model = CTSSurrogate.from_package()")
    print("  model.add_design(")
    print("      name_or_pid = 'my_chip_v2',")
    print("      def_path    = 'path/to/my_chip.def',")
    print("      saif_path   = 'path/to/my_chip.saif',")
    print("      timing_path = 'path/to/timing_paths.csv',")
    print("      t_clk       = 10.0,   # ns")
    print("  )")
    print("  pred   = model.predict('my_chip_v2', cd=55, cs=20, mw=220, bd=100)")
    print("  pareto = model.optimize('my_chip_v2', n=2000)")
    print()
    print("─" * 55)
    print(f"Demo on known placement: {pid}")
    pred = model.predict(pid, cd=55, cs=20, mw=220, bd=100)
    print(f"  predict(cd=55, cs=20, mw=220, bd=100):")
    print(f"    power = {pred.power_mW:.3f} mW")
    print(f"    WL    = {pred.wl_mm:.3f} mm")
    print(f"    skew  = {pred.skew_z:.4f} z  (per-placement normalized)")
    print(f"    hold  = {pred.hold_vio:.2f} violations")

    print(f"\n  optimize(n=500, method='nsga2'):")
    pareto = model.optimize(pid, n=500, method='nsga2')
    print(f"    {len(pareto)} Pareto solutions")
    print(f"    Best power: {pareto['power_mW'].min():.3f} mW")
    print(f"    Best skew:  {pareto['skew_z'].min():.4f} z")

    print(f"\n  sensitivity():")
    print(model.sensitivity(pid).to_string(float_format=lambda x: f"{x:+.3f}"))
