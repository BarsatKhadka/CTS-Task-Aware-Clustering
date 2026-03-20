"""
smart_pareto.py — Random vs NSGA-II vs Bayesian Pareto Search

Compares three multi-objective search strategies over CTS knob space:
  1. Random Search          (current baseline, 5000 samples)
  2. NSGA-II via pymoo      (same budget 5000; half budget 500)
  3. Optuna TPE/NSGA-II     (500 trials — Bayesian flavor)

Metrics (all on the same 4-objective space: power, WL, skew_z, hold_vio):
  - Hypervolume (HV↑) of the Pareto front vs fixed reference point
  - |Pareto| — number of non-dominated solutions found
  - Best individual objective values across the front
  - Wall clock time

Run:  python3 synthesis_best/smart_pareto.py
"""

import os, sys, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE / 'synthesis_best'))

from unified_cts import CTSSurrogate  # noqa

# ─── config ─────────────────────────────────────────────────────────────────

MODEL    = BASE / 'synthesis_best/saved_models/cts_predictor_4target.pkl'
DESIGNS  = ['aes', 'picorv32']          # representative large + small design
OBJS     = ['power_mW', 'wl_mm', 'skew_z', 'hold_vio']

CD_RANGE = (35.0, 70.0)
CS_RANGE = (12,   30  )
MW_RANGE = (130.0, 280.0)
BD_RANGE = (70.0,  150.0)

N_RANDOM  = 5000          # random search budget
N_NSGA_LG = 5000          # NSGA-II large budget  (100 pop × 50 gen)
N_NSGA_SM = 500           # NSGA-II small budget   (50 pop × 10 gen)
N_OPTUNA  = 500           # Optuna TPE trials
SEED      = 42

# ─── load model ──────────────────────────────────────────────────────────────

print("Loading surrogate model + caches...")
import pickle
def _pkl(p):
    with open(p, 'rb') as f: return pickle.load(f)

model = CTSSurrogate.load_with_caches(
    str(MODEL),
    def_cache     = _pkl(BASE / 'absolute_v7_def_cache.pkl'),
    saif_cache    = _pkl(BASE / 'absolute_v7_saif_cache.pkl'),
    timing_cache  = _pkl(BASE / 'absolute_v7_timing_cache.pkl'),
    skew_cache    = _pkl(BASE / 'skew_spatial_cache.pkl'),
    gravity_cache = _pkl(BASE / 'absolute_v10_gravity_cache.pkl'),
    net_cache     = _pkl(BASE / 'synthesis_best/net_features_cache.pkl'),
)

# Pick one placement per design as the representative pid
manifest = pd.read_csv(BASE / 'dataset_with_def/unified_manifest_normalized.csv')
PIDS = {}
for d in DESIGNS:
    rows = manifest[manifest['design_name'] == d]
    PIDS[d] = rows['placement_id'].iloc[len(rows)//2]   # middle placement

# ─── helpers ─────────────────────────────────────────────────────────────────

def _pareto_mask(costs: np.ndarray) -> np.ndarray:
    """True for non-dominated rows. O(N² / 500-chunk)."""
    n = len(costs)
    lo = costs.min(0); rng = (costs.max(0) - lo) + 1e-10
    c = (costs - lo) / rng
    dom = np.zeros(n, bool)
    for i in range(0, n, 500):
        ci = c[i:i+500]
        dominated_by_any = (
            np.all(c[:, None, :] <= ci[None, :, :] + 1e-9, axis=2) &
            np.any(c[:, None, :] <  ci[None, :, :] - 1e-9, axis=2)
        ).any(axis=0)
        dom[i:i+500] = dominated_by_any
    return ~dom


def hypervolume(pareto_costs: np.ndarray, ref: np.ndarray) -> float:
    """
    Exact hypervolume via pymoo's HV calculator.
    All objectives are minimise-all; ref must dominate all Pareto points.
    """
    try:
        from pymoo.indicators.hv import HV
        ind = HV(ref_point=ref)
        return float(ind(pareto_costs))
    except Exception:
        # fallback: 2D HV on first two objectives
        pts = pareto_costs[:, :2]
        pts = pts[pts[:, 0].argsort()]
        hv = 0.0
        y_prev = ref[1]
        for x, y in pts:
            hv += (ref[0] - x) * (y_prev - y)
            y_prev = y
        return max(hv, 0.0)


def _batch_eval(pid: str, cd_a, cs_a, mw_a, bd_a) -> np.ndarray:
    """Evaluate objectives for a batch of knob configs. Returns (N, 4) array."""
    t = model._get_t_clk(pid)
    Xpw, Xwl, Xsk, Xhv, pw_n, wl_n = model.features.batch_build(
        pid, cd_a, cs_a, mw_a, bd_a, t)
    pw, wl, sk, hv = model._heads.predict_batch(Xpw, Xwl, Xsk, Xhv, pw_n, wl_n)
    return np.column_stack([pw, wl, sk, hv])


def result_row(method, n_evals, elapsed, costs_all, costs_pareto) -> dict:
    """Compute summary statistics for one method run. HV computed later."""
    return dict(
        method=method,
        n_evals=n_evals,
        time_s=round(elapsed, 2),
        n_pareto=len(costs_pareto),
        best_pw=costs_pareto[:, 0].min(),
        best_wl=costs_pareto[:, 1].min(),
        best_sk=costs_pareto[:, 2].min(),
        best_hv=costs_pareto[:, 3].min(),
        _costs=costs_pareto,      # kept for HV computation
    )


# ─── Method 1: Random Search ─────────────────────────────────────────────────

def run_random(pid: str, n: int = N_RANDOM, seed: int = SEED) -> dict:
    rng = np.random.default_rng(seed)
    cd_a = rng.uniform(*CD_RANGE, n)
    cs_a = rng.integers(*CS_RANGE, n).astype(float)
    mw_a = rng.uniform(*MW_RANGE, n)
    bd_a = rng.uniform(*BD_RANGE, n)

    t0 = time.perf_counter()
    costs = _batch_eval(pid, cd_a, cs_a, mw_a, bd_a)
    mask  = _pareto_mask(costs)
    elapsed = time.perf_counter() - t0

    return result_row(f'Random-{n//1000}k', n, elapsed, costs, costs[mask])


# ─── Method 2: NSGA-II (pymoo) ───────────────────────────────────────────────

def run_nsga2(pid: str, n_total: int, seed: int = SEED) -> dict:
    from pymoo.core.problem import Problem
    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.operators.crossover.sbx import SBX
    from pymoo.operators.mutation.pm import PM
    from pymoo.optimize import minimize as pymoo_minimize
    from pymoo.termination import get_termination

    # pop size and generations balanced for budget
    pop_size = max(50, min(200, n_total // 20))
    n_gen    = max(10, n_total // pop_size)

    class CTSProblem(Problem):
        def __init__(self):
            super().__init__(
                n_var=4,
                n_obj=4,
                xl=np.array([CD_RANGE[0], CS_RANGE[0], MW_RANGE[0], BD_RANGE[0]]),
                xu=np.array([CD_RANGE[1], CS_RANGE[1], MW_RANGE[1], BD_RANGE[1]]),
            )

        def _evaluate(self, X, out, *args, **kwargs):
            cd_a = X[:, 0]
            cs_a = X[:, 1].round().astype(int).astype(float)
            mw_a = X[:, 2]
            bd_a = X[:, 3]
            out['F'] = _batch_eval(pid, cd_a, cs_a, mw_a, bd_a)

    problem = CTSProblem()
    algo = NSGA2(
        pop_size=pop_size,
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )
    term = get_termination('n_gen', n_gen)

    t0 = time.perf_counter()
    res = pymoo_minimize(problem, algo, term, seed=seed, verbose=False)
    elapsed = time.perf_counter() - t0

    n_evals  = res.algorithm.evaluator.n_eval
    pareto_f = res.F   # non-dominated front from NSGA-II

    label = f'NSGA-II-{n_total//1000}k' if n_total >= 1000 else f'NSGA-II-{n_total}'
    return result_row(label, n_evals, elapsed, pareto_f, pareto_f)


# ─── Method 3: Optuna TPE (multi-objective) ───────────────────────────────────

def run_optuna(pid: str, n_trials: int = N_OPTUNA, seed: int = SEED,
               sampler: str = 'nsga2') -> dict:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if sampler == 'nsga2':
        s = optuna.samplers.NSGAIISampler(seed=seed)
        label = f'Optuna-NSGA2-{n_trials}'
    else:
        s = optuna.samplers.TPESampler(seed=seed)
        label = f'Optuna-TPE-{n_trials}'

    study = optuna.create_study(
        directions=['minimize'] * 4,
        sampler=s,
    )

    # Batch evaluations in chunks for speed
    CHUNK = 50
    all_costs = []

    def objective(trial):
        cd = trial.suggest_float('cd', *CD_RANGE)
        cs = float(trial.suggest_int('cs', *CS_RANGE))
        mw = trial.suggest_float('mw', *MW_RANGE)
        bd = trial.suggest_float('bd', *BD_RANGE)
        c  = _batch_eval(pid,
                         np.array([cd]), np.array([cs]),
                         np.array([mw]), np.array([bd]))[0]
        all_costs.append(c)
        return tuple(c)

    t0 = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0

    costs_all = np.array(all_costs)
    # get Pareto from all collected points (Optuna's own filtering can miss some)
    mask = _pareto_mask(costs_all)
    return result_row(label, n_trials, elapsed, costs_all, costs_all[mask])


# ─── Method 4: Scipy DE (scalarized single-objective reference) ───────────────

def run_scipy_de(pid: str, obj_idx: int = 0, seed: int = SEED) -> dict:
    """Minimize one objective (default: power). Pure DE, single-objective reference."""
    from scipy.optimize import differential_evolution

    obj_names = ['power_mW', 'wl_mm', 'skew_z', 'hold_vio']
    bounds = [CD_RANGE, CS_RANGE, MW_RANGE, BD_RANGE]

    call_count = [0]
    all_costs  = []

    def fn(x):
        call_count[0] += 1
        cs = round(x[1])
        c = _batch_eval(pid,
                        np.array([x[0]]), np.array([float(cs)]),
                        np.array([x[2]]), np.array([x[3]]))[0]
        all_costs.append(c)
        return float(c[obj_idx])

    t0 = time.perf_counter()
    res = differential_evolution(fn, bounds, seed=seed, maxiter=50, popsize=15,
                                 tol=1e-4, workers=1)
    elapsed = time.perf_counter() - t0

    costs_all = np.array(all_costs)
    mask = _pareto_mask(costs_all)
    return result_row(f'SciPy-DE-{obj_names[obj_idx]}',
                      call_count[0], elapsed, costs_all, costs_all[mask])


# ─── main comparison loop ────────────────────────────────────────────────────

print()
for design in DESIGNS:
    pid = PIDS[design]
    print(f"{'='*70}")
    print(f"Design: {design.upper()}   pid: {pid}")
    print(f"{'='*70}")

    rows = []

    print(f"  [1/5] Random search {N_RANDOM} ...")
    rows.append(run_random(pid, N_RANDOM))

    print(f"  [2/5] Random search {N_NSGA_SM} (same budget as small NSGA) ...")
    rows.append(run_random(pid, N_NSGA_SM))

    print(f"  [3/5] NSGA-II {N_NSGA_LG} evaluations ...")
    rows.append(run_nsga2(pid, N_NSGA_LG))

    print(f"  [4/5] NSGA-II {N_NSGA_SM} evaluations ...")
    rows.append(run_nsga2(pid, N_NSGA_SM))

    print(f"  [5/5] Optuna NSGA-II {N_OPTUNA} trials ...")
    rows.append(run_optuna(pid, N_OPTUNA, sampler='nsga2'))

    # ── compute hypervolume with unified reference point ──
    # ref = 10% worse than the worst observed value across all methods
    all_pts = np.vstack([r['_costs'] for r in rows])
    ref_pt  = all_pts.max(0) * 1.10

    for r in rows:
        r['HV'] = hypervolume(r['_costs'], ref_pt)

    # ── print table ──
    hdr = (f"{'Method':<22}  {'Evals':>6}  {'Time':>6}  "
           f"{'|Pareto|':>8}  {'HV(↑)':>8}  "
           f"{'BestPw(mW)':>10}  {'BestSk(z)':>10}  {'BestHold':>8}")
    sep = '─' * len(hdr)
    print()
    print(hdr)
    print(sep)
    for r in rows:
        print(f"{r['method']:<22}  {r['n_evals']:>6}  {r['time_s']:>5.1f}s  "
              f"{r['n_pareto']:>8}  {r['HV']:>8.4f}  "
              f"{r['best_pw']:>10.3f}  {r['best_sk']:>10.4f}  {r['best_hv']:>8.2f}")

    # ── NSGA-II vs Random improvement summary ──
    rand5k  = next(r for r in rows if 'Random-5k' in r['method'])
    nsga5k  = next(r for r in rows if 'NSGA-II-5k' in r['method'])
    nsga_sm = next(r for r in rows if f'NSGA-II-{N_NSGA_SM}' in r['method'])
    opt     = next(r for r in rows if 'Optuna' in r['method'])

    print()
    print(f"  Improvements over Random-5k baseline:")
    for cand, tag in [(nsga5k, 'NSGA-II-5k'), (nsga_sm, f'NSGA-II-{N_NSGA_SM}'),
                      (opt, f'Optuna-{N_OPTUNA}')]:
        hv_gain = (cand['HV'] - rand5k['HV']) / rand5k['HV'] * 100
        pw_gain = (rand5k['best_pw'] - cand['best_pw']) / rand5k['best_pw'] * 100
        sk_gain = (rand5k['best_sk'] - cand['best_sk'])  # signed: more negative = better
        print(f"    {tag:<22}: HV +{hv_gain:+.1f}%  BestPw {pw_gain:+.2f}%  BestSkz {sk_gain:+.4f}")

    # ── show top-5 Pareto solutions from each method ──
    print()
    for r in [rand5k, nsga5k, opt]:
        print(f"\n  Top-5 Pareto (sorted by power) — {r['method']}:")
        c = r['_costs']
        idx = c[:, 0].argsort()[:5]
        print(f"    {'Power(mW)':>10}  {'WL(mm)':>8}  {'Skew(z)':>8}  {'Hold':>6}")
        for ci in c[idx]:
            print(f"    {ci[0]:>10.3f}  {ci[1]:>8.3f}  {ci[2]:>8.4f}  {ci[3]:>6.2f}")

    print()

print("Done.")
