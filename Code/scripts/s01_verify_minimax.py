"""Section 3: independent LP checks of the one-step minimax transition, predecessor and
update-window rule, noisy corners, three-stage scenario trees, and the closed-form excess."""
import json
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
from tsmc.control import (adversary, control, cost, excess_closed_form, excess_recursion,  # noqa: E402
                          maximum_upper_step, predecessor, value)

rng = np.random.default_rng(20260927)
errors, predecessor_errors, scheduler_errors = [], [], []
for k in range(6000):
    a = 10 ** rng.uniform(-3, 0)
    b = a * (1 if k % 5 == 0 else rng.uniform(1, 15))
    D = rng.uniform(0, 2) if k % 7 else 0.0
    U = D + rng.uniform(.01, 5)
    s = rng.uniform(-30, 30)
    r = abs(s)
    A, B = [], []
    for h in (a, b):
        for sign in (-1, 1):
            A.append([sign * h, -1])
            B.append(-sign * s - h * D)
    sol = linprog([0, 1], A_ub=A, b_ub=B, bounds=[(-U, U), (0, None)], method='highs')
    assert sol.success
    u = control(s, a, b, D, U)
    v = value(r, a, b, D, U)
    errors.append(max(abs(sol.fun - v), abs(cost(s, u, a, b, D) - v)))
    assert errors[-1] < 2e-8
    h, d = adversary(s, u, a, b, D)
    assert abs(abs(s + h * (u + d)) - v) < 2e-8
    R = b * D + rng.uniform(0, 5)
    p = predecessor(R, a, b, D, U)
    predecessor_errors.append(abs(value(p, a, b, D, U) - R))
    assert predecessor_errors[-1] < 2e-8
    assert value(p + 1e-5, a, b, D, U) > R
    cap = maximum_upper_step(r, a, R, D, U)
    if cap is not None and np.isfinite(cap):
        scheduler_errors.append(max(0, value(r, a, cap, D, U) - R))
        assert scheduler_errors[-1] < 2e-8
        assert value(r, a, cap + 1e-4, D, U) > R - 1e-10

# Noisy observations: enumerate independent state-error and disturbance corners.
for k in range(1000):
    y = rng.normal()
    e = rng.uniform(0, .2)
    a, b, D, U = .03, .2, .4, 2
    u = control(y, a, b, D, U)
    worst = max(abs(y + err + h * (u + d)) for err in (-e, e) for h in (a, b) for d in (-D, D))
    assert abs(worst - (e + value(abs(y), a, b, D, U))) < 1e-12

# Independent three-stage scenario-tree LP with a separate causal input at every node.
tree_errors = []
for case in range(100):
    stages = []
    for j in range(3):
        aa = rng.uniform(.02, .2)
        bb = aa + rng.uniform(0, .4)
        dd = rng.uniform(0, .6)
        stages.append((aa, bb, dd, dd + rng.uniform(.1, 2)))
    s0 = rng.uniform(-3, 3)
    nodes = [(np.zeros(21), s0)]
    bounds = []
    cursor = 0
    for aa, bb, dd, uu in stages:
        children = []
        for coeff, const in nodes:
            idx = cursor
            cursor += 1
            bounds.append((-uu, uu))
            for hh in (aa, bb):
                for dist in (-dd, dd):
                    cc = coeff.copy()
                    cc[idx] += hh
                    children.append((cc, const + hh * dist))
        nodes = children
    A, B = [], []
    for coeff, const in nodes:
        for sign in (-1, 1):
            A.append(np.r_[sign * coeff, -1])
            B.append(-sign * const)
    opt = linprog(np.r_[np.zeros(21), 1], A_ub=A, b_ub=B, bounds=bounds + [(0, None)], method='highs')
    assert opt.success
    expected = abs(s0)
    for params in stages:
        expected = value(expected, *params)
    tree_errors.append(abs(opt.fun - expected))
    assert tree_errors[-1] < 2e-8

# Closed-form excess (Corollary 3.8): e_N = max_k q^{N-k}(e_0 - k a c).
closed_errors = []
for case in range(2000):
    a = rng.uniform(.01, .3)
    b = a * rng.uniform(1, 8)
    D = rng.uniform(0, 1)
    U = D + rng.uniform(.05, 3)
    e0 = rng.uniform(1e-3, 6)
    N = int(rng.integers(1, 80))
    rec = excess_recursion(e0, a, b, D, U, N)
    for n in range(N + 1):
        closed_errors.append(abs(rec[n] - excess_closed_form(e0, a, b, D, U, n)))
assert max(closed_errors) < 1e-11

# Exact counterexample and boundary cases.
s, h, v = .1, 1., -1.
exact = ((s + h * v) ** 2 - s * s) / (2 * h)
assert abs(exact - .4) < 1e-14 and s * v < 0
assert control(0, .1, .4, 0, 1) == 0
assert abs(value(.6, .2, .6, .5, 2) - .45) < 1e-14

result = {
    'seed': 20260927, 'lp_cases': 6000, 'noise_corner_cases': 1000,
    'max_lp_and_formula_error': max(errors), 'max_predecessor_error': max(predecessor_errors),
    'max_scheduler_violation': max(scheduler_errors, default=0), 'scenario_tree_lp_cases': 100,
    'max_scenario_tree_error': max(tree_errors), 'closed_form_excess_comparisons': len(closed_errors),
    'max_closed_form_excess_error': max(closed_errors), 'counterexample_delta_V': exact,
    'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
    'status': 'PASS'}
(ROOT / 'Data' / 'verify_minimax.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
