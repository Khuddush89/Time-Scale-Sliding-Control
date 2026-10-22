"""Simulations on time scales: every state is computed by the exact jump map at right-scattered
points and by exact piecewise formulas on dense intervals.  Writes CSV/JSON files to Data/."""
import csv
import json
import math
import sys
from fractions import Fraction as Q
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
from tsmc import strange as st  # noqa: E402
from tsmc import timescales as tsm  # noqa: E402
from tsmc.control import adversary, capture_numbers, control, excess_closed_form, value  # noqa: E402

DATA = ROOT / 'Data'
summary = {}


def export(name, header, rows):
    with (DATA / f'{name}.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ============================================ 1. unrevealed graininess: nature's time scale
a, b, D, U, s0, N = .04, .20, .30, 1.50, 2.0, 55
s, t = s0, 0.0
rows = []
r = s0
for k in range(N):
    u = control(s, a, b, D, U)
    h, d = adversary(s, u, a, b, D)
    closed = b * D + excess_closed_form(s0 - b * D, a, b, D, U, k)
    rows.append([k, t, s, r, closed, h, d, u])
    s = s + h * (u + d)
    r = value(r, a, b, D, U)
    t += h
rows.append([N, t, s, r, b * D + excess_closed_form(s0 - b * D, a, b, D, U, N), '', '', ''])
export('worst_case', ['k', 't', 's', 'minimax_radius', 'closed_form_radius', 'mu', 'd', 'v'], rows)
summary['worst_case'] = capture_numbers(s0, a, b, D, U, 1e-3)
summary['worst_case']['max_closed_form_gap'] = max(abs(x[3] - x[4]) for x in rows)
summary['worst_case']['max_attainment_gap'] = max(abs(abs(x[2]) - x[3]) for x in rows)

# ============================================ 2. comparisons on two clocks
def run_comparison(ts, label):
    t_rs, mus = ts.rs()
    times = np.r_[t_rs, t_rs[-1] + mus[-1]]
    ds = .3 * np.sin(2.3 * np.array(t_rs))
    Uc, phi = 1.5, .08
    names = ['Known graininess (implicit relay)', 'Unrevealed graininess (minimax)',
             'Fixed boundary layer', 'Gao reaching law']
    states, inputs, metrics = [], [], []
    for name in names:
        s = 2.0
        ss, uu = [s], []
        for h, d in zip(mus, ds):
            if name.startswith('Known'):
                u = -np.sign(s) * min(Uc, abs(s) / h)
            elif name.startswith('Unrevealed'):
                u = control(s, .04, .2, .3, Uc)
            elif name.startswith('Fixed'):
                u = -Uc * np.clip(s / phi, -1, 1)
            else:                                   # s^Delta = -q s - eps sgn s, saturated at U
                u = float(np.clip(-2.0 * s - 0.35 * np.sign(s), -Uc, Uc))
            s = s + h * (u + d)
            ss.append(s)
            uu.append(u)
        ss = np.array(ss)
        mask = times > 5
        states.append(ss)
        inputs.append(uu)
        metrics.append(dict(clock=label, controller=name,
                            peak_after_5=float(np.max(np.abs(ss[mask]))),
                            rms_after_5=float(np.sqrt(np.mean(ss[mask] ** 2))),
                            input_variation=float(np.sum(np.abs(np.diff(uu)))),
                            outside_local_band=int(np.sum(np.abs(ss[1:][mask[1:]]) > .3 * np.array(mus)[mask[1:]] + 1e-12)),
                            points_after_5=int(np.sum(mask[1:]))))
    export(f'comparison_{label}', ['t', 'mu'] + [f's_{i}' for i in range(4)],
           [[times[k], mus[k] if k < len(mus) else ''] + [st_[k] for st_ in states] for k in range(len(times))])
    return metrics


rng = np.random.default_rng(20260927)
uni = tsm.PiecewiseScale('uniform-random', [])
tt = 0.0
for h in rng.uniform(.04, .2, 250):
    uni.pieces.append(('pt', tt, float(h)))
    tt += h
metrics = run_comparison(uni, 'uniform') + run_comparison(tsm.chaotic(.04, .2, 250), 'chaotic')
(DATA / 'comparison_metrics.json').write_text(json.dumps(metrics, indent=2))
summary['comparison'] = metrics


# ============================================ 3. mixed scales: exact arcs and tracking
def mixed_run(ts, U, D, s0, z0, lam, dist, fine=40):
    """Exact s (implicit relay, constant disturbance dist) and z^Delta = -lam z + s.
    Dense pieces are sampled at `fine` points for plotting only."""
    s, z = s0, z0
    arcs, pts = [], []
    for p in ts.pieces:
        if p[0] == 'iv':
            a_, b_ = float(p[1]), float(p[2])
            grid = np.linspace(a_, b_, fine + 1)
            arc = [(a_, s, z)]
            for j in range(fine):
                dt = grid[j + 1] - grid[j]
                # piecewise-affine s, split at the zero crossing for exactness
                left = dt
                while left > 1e-15:
                    slope = (-U * np.sign(s) + dist) if abs(s) > 1e-15 else 0.0
                    step = left
                    if slope * s < 0:
                        step = min(step, abs(s / slope))
                    E = math.exp(-lam * step)
                    z = E * z + s * (1 - E) / lam + slope * (step / lam - (1 - E) / lam ** 2)
                    s = s + slope * step
                    if abs(s) < 1e-13:
                        s = 0.0
                    left -= step
                arc.append((grid[j + 1], s, z))
            arcs.append(arc)
        else:
            t_, mu = float(p[1]), float(p[2])
            pts.append((t_, s, z, mu))
            v = -np.sign(s) * min(U, abs(s) / mu)
            z = z + mu * (-lam * z + s)
            s = s + mu * (v + dist)
    pts.append((ts.end(), s, z, 0.0))
    return arcs, pts


for name, ts, pars in [('hybrid', tsm.user_hybrid(5), (1.0, .2, 1.4, .8, 1.2)),
                       ('pab', tsm.pab(.8, .2, 6), (1.0, .2, 1.0, .8, 1.2)),
                       ('thuemorse', tsm.thue_morse(8), (1.0, .2, 1.0, .8, 1.2))]:
    U_, D_, s0_, z0_, lam = pars
    arcs, pts = mixed_run(ts, U_, D_, s0_, z0_, lam, D_)
    phi = {round(t_, 9): r_ for (t_, r_, kind) in ts.phi(U_, D_, s0_)}
    export(f'{name}_points', ['t', 's', 'z', 'mu', 'Phi'],
           [[p[0], p[1], p[2], p[3], phi.get(round(p[0], 9), '')] for p in pts])
    export(f'{name}_arcs', ['arc', 't', 's', 'z'], [[i, *x] for i, arc in enumerate(arcs) for x in arc])
    summary[name] = dict(max_point_minus_phi=max(abs(p[1]) - phi[round(p[0], 9)] for p in pts if round(p[0], 9) in phi),
                         max_abs_deviation=max(abs(abs(p[1]) - phi[round(p[0], 9)]) for p in pts if round(p[0], 9) in phi))

# ============================================ 4. isolated clocks: radii and exponentials
Uc, Dc, lam = 1.0, .2, .5
rows = []
for name, ts, r0 in [('harmonic', tsm.harmonic(.5, 400), 1.0), ('sqrt', tsm.sqrt_clock(400), 1.0),
                     ('geometric', tsm.geometric(1.5, 14), 0.0), ('primes', tsm.prime_clock(3000), 1.0)]:
    rec = ts.phi(Uc, Dc, r0)
    t_rs, mus = ts.rs()
    ev = ts.exp_neg(lam) if name in ('harmonic', 'sqrt') else [(x, float('nan')) for x in t_rs] + [(ts.end(), float('nan'))]
    nu = [float('nan')] + list(mus)
    for k, (t_, r_, kind) in enumerate(rec):
        rows.append([name, k, t_, r_, mus[k] if k < len(mus) else '', nu[k] if k < len(nu) else '',
                     ev[k][1] if k < len(ev) else ''])
export('isolated_clocks', ['clock', 'k', 't', 'Phi', 'mu', 'nu', 'exp_neg_half'], rows)

# memory rays on the prime clock (p <= 230)
ps = tsm.primes_upto(240)
rays = []
for i in range(len(ps) - 1):
    tau, mu = ps[i], ps[i + 1] - ps[i]
    for j in range(i + 1, len(ps)):
        val = Dc * mu - (Uc - Dc) * (ps[j] - ps[i + 1])
        if val < -0.25:
            break
        rays.append([tau, mu, ps[j], val])
export('prime_rays', ['tau', 'mu', 't', 'injection_value'], rays)

# ============================================ 5. accumulating blocks
L, rho = .6, .5
rows, r, s = [], 1.0, 1.0
z, lam_z = .8, 1.2
for n in range(5):
    r_start = s
    for k in range(200):
        tt_ = n + L * (1 - rho ** k)
        h = L * (1 - rho) * rho ** k
        if k < 18:
            rows.append([n, k, tt_, h, s, z, 'point'])
        z = z + h * (-lam_z * z + s)
        s = max(Dc * h, s - (Uc - Dc) * h)
    limit = st.accum_A(r_start, L, rho, Uc, Dc)
    assert abs(limit - s) < 1e-12
    rows.append([n, -1, n + L, 1 - L, limit, z, 'limit'])
    z = z + (1 - L) * (-lam_z * z + limit)
    s = max(Dc * (1 - L), limit - (Uc - Dc) * (1 - L))
export('accumulating', ['block', 'k', 't', 'mu', 's', 'z', 'type'], rows)

# ============================================ 6. Cantor clock (generation <= 8 endpoints)
gaps = st.cantor_gaps(9)
pts = sorted({x for (a_, b_, k) in st.cantor_gaps(8) for x in (a_, b_)} | {Q(0), Q(1)})
level = {}
for (a_, b_, k) in st.cantor_gaps(8):
    level[a_] = k
    level[b_] = k
rows = []
for p in pts:
    for UU in (Q(3, 5), Q(1)):
        phi_ = st.cantor_phi(p, gaps, UU, Q(1, 5))
        rows.append([float(p), str(p), float(UU), level.get(p, 0), float(phi_),
                     float(st.cantor_exp_at(Q(1, 2), st.ternary_digits(p, 40)))])
export('cantor', ['t', 't_exact', 'U', 'level', 'Phi', 'exp_neg_half'], rows)

(DATA / 'experiments_summary.json').write_text(json.dumps(summary, indent=2, default=float))
print(json.dumps(summary, indent=1, default=float)[:3000])
print('PASS: experiments')
