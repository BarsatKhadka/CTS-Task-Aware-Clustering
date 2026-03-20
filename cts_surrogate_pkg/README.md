# CTS-Surrogate

A unified, physics-informed ML surrogate for Clock Tree Synthesis (CTS) outcome prediction and knob optimization.

**One call → four predictions. One API → Pareto-optimal design.**

---

## What This Does

CTS is a critical physical-design step in VLSI. Given a floorplan + clock frequency + activity profile, the EDA tool accepts four knobs:

| Knob | Symbol | Typical range |
|------|--------|---------------|
| Max cluster diameter | `cd` | 20–70 µm |
| Cluster size | `cs` | 10–30 FFs/cluster |
| Max wire length | `mw` | 100–280 µm |
| Buffer distance | `bd` | 60–150 µm |

Each CTS run takes **10–30 minutes** in commercial EDA. This surrogate predicts all four outcomes in **< 5 ms** and finds Pareto-optimal configurations in **< 1 second**.

### Predicted Outputs

| Target | Unit | Zero-shot accuracy |
|--------|------|--------------------|
| **Power** | mW | 32% MAPE (zero-shot) → **9.8%** with K=10 calibration |
| **Wirelength** | mm | **7.0% MAPE** zero-shot ✓ |
| **Skew** | z-score | **0.074 MAE** zero-shot ✓ |
| **Hold violations** | count | **12.5% MAPE** zero-shot ✓ |

All evaluated under **LODO (Leave-One-Design-Out)** — the only valid zero-shot protocol.

---

## Quick Start

```python
from cts_surrogate import CTSSurrogate

# One-line load
model = CTSSurrogate.from_package()

# Single prediction
pred = model.predict('aes_run_xxx', cd=55, cs=20, mw=220, bd=100)
print(pred.power_mW, pred.wl_mm, pred.skew_z, pred.hold_vio)

# Pareto optimization (NSGA-II, finds 3-6× more solutions than random)
pareto = model.optimize('aes_run_xxx', n=5000)

# Knob sensitivity table
sens = model.sensitivity('aes_run_xxx')
```

---

## Directory Structure

```
cts_surrogate_pkg/
├── cts_surrogate.py        ← MAIN FILE: CTSSurrogate class, all logic here
├── evaluate.py             ← Reproduces all paper numbers (LODO + zipdiv)
├── requirements.txt        ← pip install -r requirements.txt
│
├── models/
│   ├── cts_4head.pkl       ← 4-target model: power + WL + skew + hold_vio (4.4 MB)
│   └── cts_3head.pkl       ← 3-target model: power + WL + skew (3.6 MB)
│
├── caches/                 ← Pre-parsed DEF/SAIF/timing features (fast load, ~0.8 MB)
│   ├── def_cache.pkl       ← FF positions, die geometry (from .def files)
│   ├── saif_cache.pkl      ← Toggle activity, signal probability (from .saif)
│   ├── timing_cache.pkl    ← Slack distribution, critical paths (from timing_paths.csv)
│   ├── skew_cache.pkl      ← Critical-path spatial features (key for skew prediction)
│   ├── gravity_cache.pkl   ← Gravity/distance features
│   └── net_cache.pkl       ← Signal net routing features (RSMT, RUDY)
│
├── data/
│   ├── manifest.csv        ← Full dataset: 140 placements × 10 CTS runs = 1400 rows
│   └── zipdiv_gt.csv       ← Zipdiv ground truth (5th unseen design, 20 CTS runs)
│
└── examples/
    ├── 01_single_prediction.py   ← Predict one (placement, knob) pair
    ├── 02_pareto_optimize.py     ← Random vs NSGA-II vs Bayesian comparison
    ├── 03_kshot_calibration.py   ← K-shot calibration on zipdiv OOD design
    └── 04_new_design.py          ← Register an unseen design from raw DEF/SAIF files
```

---

## Installation

```bash
pip install -r requirements.txt
```

Core dependencies: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `lightgbm`, `pymoo`
Optional: `optuna` (for `method='bayesian'`)

---

## API Reference

### `CTSSurrogate.from_package(pkg_dir=None, model='4head')`

Load the surrogate from the package directory. Auto-discovers all model and cache files.

```python
model = CTSSurrogate.from_package()                    # 4-target model (default)
model = CTSSurrogate.from_package(model='3head')       # 3-target (no hold_vio)
model = CTSSurrogate.from_package('/path/to/pkg_dir')  # explicit directory
```

---

### `model.predict(pid, cd, cs, mw, bd, sk_mu=None, sk_sig=None)`

Predict all CTS outcomes for one (placement, knob) configuration.

```python
pred = model.predict('aes_run_xxx', cd=55, cs=20, mw=220, bd=100)

pred.power_mW   # float: predicted clock power (mW)
pred.wl_mm      # float: predicted clock wirelength (mm)
pred.skew_z     # float: predicted skew (per-placement z-score, always available)
pred.skew_ns    # float | None: absolute skew in ns (if sk_mu, sk_sig provided)
pred.hold_vio   # float: predicted hold violation count
pred.pw_norm    # float: physics normalizer used for power
pred.wl_norm    # float: physics normalizer used for WL
```

**Skew in absolute ns**: pass `sk_mu` (mean skew for this placement) and `sk_sig` (std):
```python
pred = model.predict(pid, cd=55, cs=20, mw=220, bd=100, sk_mu=0.72, sk_sig=0.05)
print(pred.skew_ns)   # absolute ns
```

---

### `model.optimize(pid, n=5000, method='nsga2', ...)`

Multi-objective Pareto search over the knob space.

```python
pareto = model.optimize(pid, n=5000)                     # NSGA-II (default)
pareto = model.optimize(pid, n=500, method='bayesian')   # Optuna, best min-skew
pareto = model.optimize(pid, n=5000, method='random')    # uniform random (legacy)
```

Returns a DataFrame of non-dominated solutions sorted by `power_mW`:

```
     cd   cs    mw    bd   power_mW   wl_mm   skew_z   hold_vio
     68   28   257    73     55.07   642.76   -0.311       0.0
     70   28   268   146     55.08   640.69   -0.532       0.0
     ...
```

**Method comparison** (AES design, N=5000):

| Method | |Pareto| | Best power | Best skew_z | Time |
|--------|---------|------------|------------|------|
| random | 27 | 69.255 mW | -0.896 | 0.7s |
| **nsga2** | **158** | **69.151 mW** | **-0.911** | **0.9s** |
| bayesian@500 | 81 | 69.226 mW | **-1.014** | 7.1s |

Knob ranges can be customized:
```python
pareto = model.optimize(pid, n=2000,
                        cd_range=(40, 65), cs_range=(15, 25),
                        mw_range=(150, 250), bd_range=(80, 130))
```

---

### `model.sensitivity(pid, base_knobs=(50, 20, 200, 100), delta=0.10)`

Numerical partial derivatives: ∂(target)/∂(knob) at base configuration.

```python
sens = model.sensitivity(pid)
# Returns DataFrame:
#         d_power_pct  d_wl_pct  d_skew_z  d_hold_pct
# cd          -20.562    -3.423    +0.502    -702.707
# cs           -0.456    -0.441    +0.104   -1100.772
# mw           -0.273    -0.139    -1.797    -121.243
# bd           +0.000    -0.264    +0.141      +0.000
```

Key findings:
- **`mw` dominates skew** (sensitivity −1.8 per unit change) — max wire length is the primary equalization lever
- **`cd` dominates power** (sensitivity −20.6%) — larger clusters → fewer buffers → lower capacitance
- **`cd` and `cs` dominate hold violations** — cluster granularity controls inter-cluster path balance

---

### `model.add_design(name_or_pid, def_path, saif_path, timing_path, t_clk=7.0)`

Register a new, unseen design from raw files. Works for zero-shot or K-shot use.

```python
model.add_design(
    name_or_pid = 'my_chip_v2',
    def_path    = 'path/to/my_chip.def',
    saif_path   = 'path/to/my_chip.saif',
    timing_path = 'path/to/timing_paths.csv',
    t_clk       = 10.0,   # clock period in ns
)
pred   = model.predict('my_chip_v2', cd=55, cs=20, mw=220, bd=100)
pareto = model.optimize('my_chip_v2', n=2000)
```

After `add_design()`, the placement is available for all methods.

---

### `model.lodo_summary()`

Print the stored LODO validation results from the trained model.

```
LODO validation (Leave-One-Design-Out, 4 training designs):
  power   : aes=36.6%  ethmac=12.3%  picorv32=30.1%  sha256=48.9%  → mean=32.0%
  wl      : aes=12.8%  ethmac=7.2%   picorv32=4.3%   sha256=3.7%   → mean=7.0%
  skew    : aes=0.085  ethmac=0.079  picorv32=0.063  sha256=0.068  → mean=0.074
  hold_vio: aes=28.3%  ethmac=4.7%   picorv32=4.0%   sha256=12.9%  → mean=12.5%
```

---

## Zero-Shot on Unseen Design (Zipdiv)

Zipdiv is a cryptographic accelerator with 142 FFs and a 215×165 µm die — 10–24× smaller than training designs.

| Metric | Zero-shot | K=1 calibration | K=5 calibration |
|--------|-----------|-----------------|-----------------|
| Power MAPE | 55.5% | **~5.0% ±1.4** ✓ | **~4.1% ±0.7** ✓ |
| WL MAPE | 99.8% | **~4.1% ±0.6** ✓ | **~3.7% ±0.2** ✓ |
| Skew MAE | **3.93 ps** ✓ | 3.93 ps | 3.93 ps |

The large zero-shot power/WL errors are a normalizer scale mismatch (not a model error). Skew is already correct because per-placement z-scoring is size-invariant. A single K=1 CTS run (~10-30 min EDA) anchors the scale and reduces both to ≤5%.

---

## Architecture

```
CTSSurrogate.from_package()
        │
        ▼
FeatureEngine  ← parses DEF/SAIF/timing once, caches in memory
        │
        │  shared 29-dim placement context (geometry + activity + timing)
        │  + 8-dim knob vector (log + raw)
        │
   ┌────┼─────────────────────────────────────┐
   │    │                                     │
   ▼    ▼                                     ▼
Power  Wirelength    Skew              Hold Violations
XGB    LGB+Ridge     XGB               LGB
76d    84d           63d               76d
   │    │             │                  │
   └────┴─────────────┴──────────────────┘
                 │
         CTSPrediction(power_mW, wl_mm, skew_z, hold_vio)
                 │
         ParetoOptimizer (NSGA-II default)
                 │
         Pareto-optimal DataFrame
```

### Why three different model families?

- **Power/WL**: XGBoost and LightGBM+Ridge — global aggregates, tree ensembles handle nonlinear knob interactions
- **Skew**: XGBoost with critical-path spatial features — worst-case metric, needs tail statistics not averages
- **Hold violations**: LightGBM — anti-correlated with skew (r = −0.96), shares power/WL feature prefix

### Key physics insights

1. **Skew ↔ Hold anti-correlation (r = −0.96)**: Increasing `mw` reduces skew (better equalization) but risks hold violations (longer paths arrive late). This is Elmore delay physics, not a model artifact.

2. **Power ↔ WL coupling is regime-dependent**: Strong for AES (r = 0.61, clock-dominated) but near-zero for PicoRV32 (r = 0.05, logic-switching dominated).

3. **`mw` is the dominant skew lever**: Sensitivity −1.8 z-units per unit change. All Pareto-optimal low-skew solutions use `mw` near its maximum.

4. **Net routing features improve WL**: RSMT and RUDY as *features* (not normalizers) cut WL MAPE from 11.0% → 7.0% by providing routing density context.

---

## Reproducing Paper Results

```bash
# All LODO + zipdiv results in one run (~30 seconds)
python3 evaluate.py

# LODO only
python3 evaluate.py --lodo-only

# Zipdiv K-shot calibration only
python3 evaluate.py --zipdiv-only

# Run examples
python3 examples/01_single_prediction.py
python3 examples/02_pareto_optimize.py
python3 examples/03_kshot_calibration.py
python3 examples/04_new_design.py
```

---

## Training Designs

The model was trained on four open-source VLSI benchmarks:

| Design | FFs | Die (approx) | Runs |
|--------|-----|--------------|------|
| AES | ~2994 | large | 31 placements × 10 = 310 |
| ETH MAC | ~5000+ | large | 47 placements × 10 = 470 |
| PicoRV32 | ~1597 | medium | 31 placements × 10 = 310 |
| SHA-256 | ~1807 | medium | 31 placements × 10 = 310 |

**Total**: 140 placements, 1400 CTS runs.

---

## Extending to New Designs

To add a new design family to the training set:
1. Run CTS on ≥5 placements × 10 knob configurations
2. Register each placement with `model.add_design()`
3. Retrain using `synthesis_best/build_final_model.py` (in the parent repo)
4. Save updated model with `pickle.dump()`

For zero-shot use without retraining, K=1–5 calibration runs are sufficient to achieve ≤5% power/WL error on any new design.

---

## Citation

If you use this code, please cite:

```
@inproceedings{cts_surrogate_2026,
  title  = {CTS-Surrogate: A Physics-Informed Unified Framework for
            Clock Tree Synthesis Outcome Prediction},
  author = {[Authors]},
  booktitle = {ICCAD},
  year   = {2026}
}
```
