"""Section 4.1: randomized checks of the max-plus graininess-memory formula (Theorem 4.3).

Random right-regular time scales are assembled from dense intervals, isolated points and
left-accumulating blocks.  For each scale we check
  (a) |s(t)| <= Phi(t) for random admissible disturbances (exact closed loop),
  (b) |s(t)| = Phi(t) for d = D sgn s(t0),
  (c) the recursion and the supremum form of Phi agree,
  (d) Phi is max{U - D, D}-Lipschitz on T,
  (e) for random causal policies an adversary enforces |s(t)| >= Phi(t) (lower bound),
  (f) limsup Phi = D limsup mu on long scales with prescribed graininess patterns.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
from tsmc.timescales import PiecewiseScale  # noqa: E402

rng = np.random.default_rng(20260927)


def random_scale(rng, t_end):
    pieces, t = [], 0.0
    while t < t_end:
        kind = rng.choice(['dense', 'iso', 'acc'])
        if kind == 'dense':
            L = rng.uniform(.05, .8)
            pieces.append(('iv', t, t + L))
            t += L
            mu = rng.uniform(.02, .6)
            pieces.append(('pt', t, mu))
            t += mu
        elif kind == 'iso':
            for _ in range(int(rng.integers(1, 7))):
                mu = rng.uniform(.005, .7)
                pieces.append(('pt', t, mu))
                t += mu
        else:
            L, rho = rng.uniform(.1, .7), rng.uniform(.2, .9)
            K = int(min(40, np.floor(np.log(1e-9 / L) / np.log(rho))))   # smallest listed gap >= 1e-9
            for k in range(K):
                pieces.append(('pt', t + L * (1 - rho ** k), L * (1 - rho) * rho ** k))
            pieces.append(('pt', t + L * (1 - rho ** K), L * rho ** K))
            t += L
            mu = rng.uniform(.02, .6)
            pieces.append(('pt', t, mu))
            t += mu
    return PiecewiseScale('random', pieces)


def adversarial_run(ts, U, D, s0, rng):
    """Random causal policy (admissible, arbitrary) against the enforcing adversary."""
    s = s0
    out = [(0.0, s)]
    for p in ts.pieces:
        if p[0] == 'iv':
            a, b = p[1], p[2]
            edges = np.linspace(a, b, 17)
            for j in range(16):
                dt = edges[j + 1] - edges[j]
                v = rng.uniform(-U, U)                      # policy constant on the substep
                d = D * (1.0 if s >= 0 else -1.0)
                # exact solution of s' = v + d with constant right-hand side
                s = s + (v + d) * dt
                out.append((edges[j + 1], s))
        else:
            t, mu = p[1], p[2]
            v = rng.uniform(-U, U) if rng.random() < .5 else -np.sign(s) * min(U, abs(s) / mu)
            nominal = s + mu * v
            d = D * (1.0 if nominal >= 0 else -1.0)
            s = nominal + mu * d
            out.append((t + mu, s))
    return out


viol, attain, formula_gap, lip_ratio, lower_viol, npts = 0.0, 0.0, 0.0, 0.0, 0.0, 0
ncases = 400
for case in range(ncases):
    ts = random_scale(rng, rng.uniform(3, 10))
    U = rng.uniform(.3, 3)
    D = rng.uniform(0, .95 * U)
    s0 = rng.uniform(-3, 3)
    c = U - D
    rec = ts.phi(U, D, abs(s0))
    phi_at = {}
    for t, r, _ in rec:
        phi_at[round(t, 12)] = r
    # (c) recursion versus supremum form at every recorded time
    for t, r, _ in rec:
        formula_gap = max(formula_gap, abs(r - ts.phi_supformula(U, D, abs(s0), t)))
    # (d) Lipschitz constant of Phi on T (recorded points)
    tt = np.array([x[0] for x in rec])
    rr = np.array([x[1] for x in rec])
    dt = np.diff(tt)
    ok = dt > 1e-6          # floating resolution of recorded times
    if ok.any():
        lip_ratio = max(lip_ratio, float(np.max(np.abs(np.diff(rr))[ok] / dt[ok]) / max(c, D)))
    # (a) and (b): exact closed loop
    for mode in ('random', 'max'):
        sim = ts.simulate(U, D, s0, mode, rng=rng)
        for t, s in sim:
            ph = ts.phi_supformula(U, D, abs(s0), t)
            npts += 1
            if mode == 'random':
                viol = max(viol, abs(s) - ph)
            else:
                attain = max(attain, abs(abs(s) - ph))
    # (e) lower bound against arbitrary causal policies
    for t, s in adversarial_run(ts, U, D, s0, rng):
        lower_viol = max(lower_viol, ts.phi_supformula(U, D, abs(s0), t) - abs(s))

# (f) asymptotic band: long isolated scales with graininess patterns of known limsup
band_errors = []
for case in range(60):
    U = rng.uniform(.5, 2)
    D = rng.uniform(.05, .9 * U)
    base = rng.uniform(.05, .5, size=7)          # periodic pattern -> limsup = max(base)
    mus = np.tile(base, 400)
    ts = PiecewiseScale('periodic', [('pt', float(np.sum(mus[:k])), float(mus[k])) for k in range(len(mus))])
    rec = ts.phi(U, D, rng.uniform(0, 5))
    tail = [r for (t, r, kind) in rec[-100:]]
    band_errors.append(abs(max(tail) - D * base.max()))

out = {'status': 'PASS', 'random_time_scales': ncases, 'closed_loop_points': npts,
       'max_upper_bound_violation': viol, 'max_attainment_error': attain,
       'max_recursion_vs_supremum_gap': formula_gap, 'max_lipschitz_ratio': lip_ratio,
       'max_lower_bound_violation_random_policies': lower_viol,
       'asymptotic_band_cases': len(band_errors), 'max_asymptotic_band_error': max(band_errors)}
assert viol < 1e-9 and attain < 1e-9 and formula_gap < 1e-9 and lip_ratio <= 1 + 1e-6
assert lower_viol < 1e-9 and max(band_errors) < 1e-9
(ROOT / 'Data' / 'verify_maxplus.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
