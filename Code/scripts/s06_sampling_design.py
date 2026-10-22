"""Section 4.2: sampling design on time scales.

Checks, in exact rational arithmetic where stated:
  (a) refinement monotonicity: T subset T'  =>  Phi_T' <= Phi_T at every point of T, and at the new
      points of a gap (tau, sigma(tau)) of T, Phi_T' <= max{Phi_T(tau), D mu_T(tau)};
  (b) one-step contraction limit Phi(sigma) >= (D/U) Phi on random isolated scales, hence
      Phi(t_N) >= (D/U)^N r0;
  (c) the self-triggered scale T_*: Phi(t_k) = (D/U)^k r0 = r0 - c (t_k - t0) exactly, every
      injection term of the supremum formula vanishes at t* = t0 + r0/c;
  (d) numerical minimization over gap sequences cannot beat (D/U)^N r0;
  (e) band-rate rule: Phi stays <= R iff every gap is <= R/D;
  (f) sample counts: self-triggered versus the sparsest uniform clock.
"""
import json
import sys
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
from tsmc.design import (refine, samples_self_triggered, samples_uniform,  # noqa: E402
                         self_triggered, self_triggered_scale)
from tsmc import timescales as tsm  # noqa: E402
from tsmc.timescales import PiecewiseScale  # noqa: E402

rng = np.random.default_rng(20260927)


def rational_scale(rng, t_end):
    """Random right-regular time scale with rational data: dense intervals, isolated points and
    truncated accumulating blocks."""
    pieces, t = [], Q(0)
    while t < t_end:
        kind = rng.choice(['dense', 'iso', 'acc'])
        if kind == 'dense':
            L = Q(int(rng.integers(5, 80)), 100)
            pieces.append(('iv', t, t + L))
            t += L
            mu = Q(int(rng.integers(2, 60)), 100)
            pieces.append(('pt', t, mu))
            t += mu
        elif kind == 'iso':
            for _ in range(int(rng.integers(1, 7))):
                mu = Q(int(rng.integers(1, 70)), 100)
                pieces.append(('pt', t, mu))
                t += mu
        else:
            L, rho = Q(int(rng.integers(10, 70)), 100), Q(int(rng.integers(2, 9)), 10)
            K = int(rng.integers(6, 16))
            for k in range(K):
                pieces.append(('pt', t + L * (1 - rho ** k), L * (1 - rho) * rho ** k))
            pieces.append(('pt', t + L * (1 - rho ** K), L * rho ** K))
            t += L
            mu = Q(int(rng.integers(2, 60)), 100)
            pieces.append(('pt', t, mu))
            t += mu
    return PiecewiseScale('random', pieces)


# ------------------------------------------------------------------ (a) refinement monotonicity
n_ref, n_common, n_new, worst_common, worst_new = 0, 0, 0, Q(0), Q(0)
for case in range(300):
    T = rational_scale(rng, int(rng.integers(3, 9)))
    U = Q(int(rng.integers(30, 300)), 100)
    D = U * Q(int(rng.integers(0, 95)), 100)
    r0 = Q(int(rng.integers(0, 300)), 100)
    rec_T = T.phi(U, D, r0)
    phi_T = {t: r for (t, r, _) in rec_T}
    for rep in range(3):
        Tp = refine(T, rng, exact=True)
        n_ref += 1
        rec_Tp = Tp.phi(U, D, r0)
        phi_Tp = {}
        for t, r, _ in rec_Tp:
            phi_Tp.setdefault(t, r)
        # points of T (listed points, interval ends and the final point)
        for t, r, _ in rec_T:
            assert t in phi_Tp
            n_common += 1
            worst_common = max(worst_common, phi_Tp[t] - r)
        # new points inside gaps of T
        gaps = [(p[1], p[1] + p[2]) for p in T.pieces if p[0] == 'pt']
        for t, r, _ in rec_Tp:
            if t in phi_T:
                continue
            for (a, b) in gaps:
                if a < t < b:
                    n_new += 1
                    worst_new = max(worst_new, r - max(phi_T[a], D * (b - a)))
                    break
assert worst_common <= 0 and worst_new <= 0

# ------------------------------------------------------------------ (b) contraction limit
worst_step, worst_power, n_steps = 0.0, 0.0, 0
for case in range(2000):
    U = rng.uniform(.2, 3)
    D = rng.uniform(.01, .95) * U
    r = rng.uniform(0, 5)
    r0 = r
    for N in range(1, 40):
        mu = rng.exponential(rng.uniform(.01, 2))
        r_next = max(D * mu, r - (U - D) * mu)
        worst_step = min(worst_step, r_next - (D / U) * r)
        r = r_next
        worst_power = min(worst_power, r - (D / U) ** N * r0)
        n_steps += 1
assert worst_step > -1e-12 and worst_power > -1e-12

# ------------------------------------------------------------------ (c) self-triggered scale, exact
U, D, r0 = Q(1), Q(1, 5), Q(1)
c = U - D
pts, t_star = self_triggered(r0, U, D, 40)
exact_equalities = all(r == (D / U) ** k * r0 and r == r0 - c * (t - 0) for k, (t, r, mu) in enumerate(pts))
# supremum formula at t*: every injection term D mu_k - c (t* - sigma(t_k)) equals 0
terms = [D * mu - c * (t_star - (t + mu)) for (t, r, mu) in pts]
all_terms_zero = all(x == 0 for x in terms)
assert exact_equalities and all_terms_zero and t_star == Q(5, 4)
sts = self_triggered_scale(float(r0), float(U), float(D), 40)

# ------------------------------------------------------------------ (d) minimization over gap sequences
min_gap = 0.0
for N in range(1, 5):
    for trial in range(40):
        U_, D_ = rng.uniform(.5, 2), rng.uniform(.05, .9)
        D_ *= U_
        r0_ = rng.uniform(.2, 3)

        def final(mus):
            r = r0_
            for m in np.abs(mus):
                r = max(D_ * m, r - (U_ - D_) * m)
            return r
        best = min(minimize(final, rng.uniform(0, 2 * r0_ / U_, size=N), method='Nelder-Mead',
                            options={'xatol': 1e-12, 'fatol': 1e-14, 'maxiter': 4000}).fun for _ in range(6))
        min_gap = min(min_gap, best - (D_ / U_) ** N * r0_)
assert min_gap > -1e-10

# ------------------------------------------------------------------ (e) band-rate rule
band_ok = True
for case in range(500):
    U = rng.uniform(.3, 3)
    D = rng.uniform(.05, .9) * U
    R = rng.uniform(.01, 1)
    mus = rng.uniform(.001, R / D, size=200)
    r = rng.uniform(0, R)
    for m in mus:
        r = max(D * m, r - (U - D) * m)
        band_ok &= r <= R + 1e-12
    r2 = max(D * (1.01 * R / D), r - (U - D) * (1.01 * R / D))
    band_ok &= r2 > R
assert band_ok

# ------------------------------------------------------------------ (f) sample counts (Table)
U, D, r0 = 1.0, 0.2, 1.0
counts = []
for e in range(1, 7):
    eps = 10.0 ** (-e)
    n_st = samples_self_triggered(r0, eps, U, D)
    # direct check on the self-triggered scale
    pts_f, _ = self_triggered(r0, U, D, n_st + 1)
    assert pts_f[n_st][1] <= eps * (1 + 1e-12) and (n_st == 0 or pts_f[n_st - 1][1] > eps)
    n_un = samples_uniform(r0, eps, U, D)
    # direct check on the uniform clock with gap eps/D
    h = eps / D
    r, k = r0, 0
    while r > eps * (1 + 1e-12):
        r = max(D * h, r - (U - D) * h)
        k += 1
    assert k == n_un
    counts.append({'eps': eps, 'self_triggered': n_st, 'uniform': n_un,
                   'time_self_triggered': pts_f[n_st][0], 'time_uniform': n_un * h,
                   'continuous_time': (r0 - eps) / (U - D)})

# ------------------------------------------------------------------ (g) data for the sampling figure
U, D, r0 = 1.0, 0.2, 1.0
pts_e, t_star_e = self_triggered(r0, U, D, 14)
example = {'U': U, 'D': D, 'r0': r0, 't_star': t_star_e,
           'self_triggered': [[t, r] for (t, r, mu) in pts_e]}
for h in (0.25, 0.125):
    ts_u = tsm.uniform(h, int(round(2.0 / h)) + 1)
    example[f'uniform_{h}'] = [[float(t), float(r)] for (t, r, k) in ts_u.phi(U, D, r0)]
complexity = {'self_triggered': [(D / U) ** k * r0 for k in range(0, 10)]}
for eps in (0.05, 0.01, 0.002):
    h = eps / D
    r, seq = r0, [r0]
    for k in range(samples_uniform(r0, eps, U, D) + 3):
        r = max(D * h, r - (U - D) * h)
        seq.append(r)
    complexity[f'uniform_{eps}'] = seq
example['complexity'] = complexity
T_ex = PiecewiseScale('T', [('iv', 0.0, .3), ('pt', .3, .5), ('pt', .8, .35), ('iv', 1.15, 1.4), ('pt', 1.4, .6),
                            ('pt', 2.0, .25), ('pt', 2.25, .45), ('iv', 2.7, 3.0)])
Tp_ex = PiecewiseScale("T'", [('iv', 0.0, .3), ('pt', .3, .25), ('pt', .55, .25), ('pt', .8, .35),
                              ('iv', 1.15, 1.4), ('pt', 1.4, .2), ('iv', 1.6, 1.75), ('pt', 1.75, .25),
                              ('pt', 2.0, .25), ('pt', 2.25, .15), ('pt', 2.4, .15), ('pt', 2.55, .15),
                              ('iv', 2.7, 3.0)])
UR, DR, rR = 1.0, 0.3, 1.2
phi_T_ex = T_ex.phi(UR, DR, rR)
phi_Tp_ex = Tp_ex.phi(UR, DR, rR)
common = {round(t, 12): r for (t, r, k) in phi_T_ex}
assert all(r <= common[round(t, 12)] + 1e-12 for (t, r, k) in phi_Tp_ex if round(t, 12) in common)
example['refine'] = {'U': UR, 'D': DR, 'r0': rR,
                     'T_pieces': [list(p) for p in T_ex.pieces], 'Tp_pieces': [list(p) for p in Tp_ex.pieces],
                     'phi_T': [list(x) for x in phi_T_ex], 'phi_Tp': [list(x) for x in phi_Tp_ex]}
rng_b = np.random.default_rng(7)
UB, DB, RB = 1.0, 0.2, 0.05
mus_b = rng_b.uniform(.05, .3, size=26)
t_b, phi_b = [0.0], [RB]
for m in mus_b:
    t_b.append(t_b[-1] + m)
    phi_b.append(max(DB * m, phi_b[-1] - (UB - DB) * m))
assert all((p <= RB + 1e-12) == (m <= RB / DB + 1e-12) for p, m in zip(phi_b[1:], mus_b))
example['band_rate'] = {'U': UB, 'D': DB, 'R': RB, 't': t_b, 'mu': mus_b.tolist(), 'phi': phi_b}

out = {'status': 'PASS',
       'refinement_cases': n_ref, 'refinement_common_points': n_common, 'refinement_new_points': n_new,
       'refinement_max_excess_common': float(worst_common), 'refinement_max_excess_new': float(worst_new),
       'contraction_steps': n_steps, 'contraction_min_margin': worst_step, 'power_min_margin': worst_power,
       'self_triggered_exact': bool(exact_equalities), 'self_triggered_terms_zero_at_tstar': bool(all_terms_zero),
       'self_triggered_points': [[float(t), float(r), float(mu)] for (t, r, mu) in pts[:12]],
       't_star': float(t_star),
       'min_search_margin': min_gap, 'band_rate_rule': bool(band_ok), 'sample_counts': counts,
       'example': example}
(ROOT / 'Data' / 'sampling_design.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k not in ('self_triggered_points', 'example')}, indent=1))
