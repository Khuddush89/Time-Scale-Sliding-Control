"""Exact computations on selected time scales.

Exact rationals are used wherever the data are rational; transcendental quantities
(Gamma functions, q-Pochhammer symbols, Cantor products) use 40-digit mpmath.
All claims quoted in the paper are asserted here.
"""
import json
import math
import sys
from fractions import Fraction as Q
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
from tsmc import strange as st  # noqa: E402
from tsmc import timescales as tsm  # noqa: E402

mp.mp.dps = 40
U, D = Q(1), Q(1, 5)
c = U - D
out = {}


def fr(x):
    return f'{x.numerator}/{x.denominator}' if isinstance(x, Q) and x.denominator != 1 else str(x)


# ------------------------------------------------------------------ G1  P_{0.8,0.2}
a, b = Q(4, 5), Q(1, 5)
rows, r = [], Q(1)
for k in range(6):
    zero_hit = k + r / c if r / c <= a else None           # first zero on [k, k + a]
    r_end = max(Q(0), r - c * a)
    r_next = st.G(b, r_end, U, D)
    rows.append(dict(k=k, Phi_k=fr(r), zero_from=fr(zero_hit) if zero_hit is not None else '-',
                     Phi_rs=fr(r_end), Phi_next=fr(r_next)))
    r = r_next
assert r == D * b                    # steady state radius D mu = 1/25 at every integer
out['pab'] = dict(rows=rows, steady=fr(D * b),
                  exp_half_per_cell=float(mp.e ** (-mp.mpf('0.4')) * (1 - mp.mpf('0.1'))))

# ------------------------------------------------------------------ G2  harmonic clock
alpha = Q(1, 2)
rh = st.harmonic_radii(alpha, U, D, Q(1), 400)
assert rh[:8] == [Q(1), Q(3, 5), Q(2, 5), Q(4, 15), Q(1, 6), Q(13, 150), Q(1, 50), Q(1, 70)]
for k in range(7, 401):
    assert rh[k] == Q(1, 10 * k) == D * alpha / k           # = D nu(t_k)
hexp = [(N, float(st.harmonic_exp(Q(1, 2), alpha, N)), float(mp.e ** (-mp.mpf(1) / 2 * alpha * mp.harmonic(N))))
        for N in (1, 2, 5, 10, 100, 1000)]
for N in (1, 2, 5, 10, 50):
    direct = mp.nprod(lambda k: 1 - mp.mpf(1) / 2 * alpha / k, [1, N])
    assert abs(direct - st.harmonic_exp(Q(1, 2), alpha, N)) < mp.mpf(10) ** -30
lim = st.harmonic_exp_limit_ratio(1, alpha)
ratio_big = st.harmonic_exp(1, alpha, 10 ** 6) * mp.e ** (alpha * mp.harmonic(10 ** 6))
assert abs(ratio_big - lim) < 1e-6
out['harmonic'] = dict(first=[fr(x) for x in rh[:12]], exp_half=hexp,
                       limit_ratio_lambda1=float(lim), ratio_at_1e6=float(ratio_big))

# ------------------------------------------------------------------ G3  square-root clock
Dm, Um = mp.mpf(1) / 5, mp.mpf(1)
r = mp.mpf(1)
sq_rows, max_dev = [], mp.mpf(0)
for n in range(0, 20001):
    mu = mp.sqrt(n + 1) - mp.sqrt(n)
    r_next = max(Dm * mu, r - (Um - Dm) * mu)
    if n + 1 >= 1:
        max_dev = max(max_dev, abs(r_next - Dm * mu))          # claim: r(sqrt(n+1)) = D nu = D mu(sqrt n)
    if n < 8:
        sq_rows.append(dict(n=n + 1, t=float(mp.sqrt(n + 1)), radius=float(r_next), D_nu=float(Dm * mu)))
    r = r_next
assert max_dev < mp.mpf(10) ** -35
sq_exp = []
for N in (10, 100, 1000, 10000, 100000):
    e = st.sqrt_exp(Q(1, 2), N)
    t = mp.sqrt(N)
    sq_exp.append(dict(N=N, t=float(t), e=float(e), e_times_exp=float(e * mp.e ** (t / 2)),
                       normalized=float(e * mp.e ** (t / 2) * t ** (mp.mpf(1) / 16))))
norms = [x['normalized'] for x in sq_exp]
assert abs(norms[-1] - norms[-2]) < 2e-4 and all(norms[i] > norms[i + 1] for i in range(len(norms) - 1))
out['sqrt'] = dict(rows=sq_rows, max_dev_D_nu=float(max_dev), exp=sq_exp)

# ------------------------------------------------------------------ G4  geometric clock
qg = Q(3, 2)
r, geo = Q(0), []
for k in range(12):
    t, mu = qg ** k, (qg - 1) * qg ** k
    r_next = st.G(mu, r, U, D)
    assert r_next == D * (qg - 1) * qg ** k == D * (qg * t - t)
    geo.append(dict(k=k + 1, t=fr(qg ** (k + 1)), radius=fr(r_next), v=fr(-min(U, r / mu)) if k else '0'))
    r = r_next
out['geometric'] = dict(rows=geo, power_exponent_kappa1=math.log(0.5) / math.log(1.5))

# ------------------------------------------------------------------ G5  prime clock
ps = tsm.primes_upto(200000)
r = Q(1)
events, prime_rows, recs = [], [], []
maxr = Q(0)
for n in range(len(ps) - 1):
    p, g = ps[n], ps[n + 1] - ps[n]
    r_next = st.G(Q(g), r, U, D)
    if r_next > D * g:
        events.append(dict(p=p, next=ps[n + 1], gap=g, prev_gap=ps[n] - ps[n - 1] if n else None,
                           radius=fr(r_next), D_nu=fr(D * g)))
    if n < 16:
        prime_rows.append(dict(p=p, gap=g, radius_at_p=fr(r), radius_next=fr(r_next)))
    if r_next > maxr:
        maxr = r_next
        recs.append(dict(p=ps[n + 1], radius=fr(r_next), gap=g))
    r = r_next
# first primes: exact values used in Table S2
assert prime_rows[0]['radius_next'] == '1/5' and prime_rows[1]['radius_next'] == '2/5'
out['primes'] = dict(rows=prime_rows, n_primes=len(ps), n_events=len(events),
                     first_events=events[:8], records=recs[-6:], pmax=ps[-1])

# ------------------------------------------------------------------ G6  accumulating blocks
L, rho = Q(3, 5), Q(1, 2)
assert st.accum_A(Q(1), L, rho, U, D) == Q(13, 25) and st.accum_B(Q(1), L, rho, U, D) == Q(1, 5)
assert st.accum_A(Q(1, 5), L, rho, U, D) == 0 and st.accum_B(Q(1, 5), L, rho, U, D) == Q(2, 25)
assert st.accum_A(Q(0), L, Q(1, 10), U, D) == Q(3, 50)
prefix = [fr(st.accum_prefix(Q(1), L, rho, U, D, k)) for k in range(8)]
# prefix -> limit with the Cauchy certificate (U + D) L rho^k
for k in (5, 10, 20, 40):
    gap = abs(st.accum_prefix(Q(1), L, rho, U, D, k) - st.accum_A(Q(1), L, rho, U, D))
    assert gap <= (U + D) * L * rho ** k
blocks, r = [], Q(1)
for n in range(4):
    A = st.accum_A(r, L, rho, U, D)
    B = st.accum_B(r, L, rho, U, D)
    blocks.append(dict(n=n, r_n=fr(r), A=fr(A), B=fr(B)))
    r = B
grid = []
for (LL, rr) in [(Q(3, 5), Q(1, 2)), (Q(3, 5), Q(1, 10)), (Q(1, 2), Q(1, 5)), (Q(9, 10), Q(1, 4)), (Q(1, 5), Q(9, 10))]:
    grid.append(dict(L=fr(LL), rho=fr(rr), A0=fr(st.accum_A(Q(0), LL, rr, U, D)),
                     zero_possible=bool(rr >= D / U), cL=fr(c * LL),
                     exp_block=float(st.accum_block_exp(Q(1, 2), LL, rr) * (1 - (1 - LL) / 2))))
qp_direct = mp.nprod(lambda k: 1 - mp.mpf('0.5') * mp.mpf('0.6') * mp.mpf('0.5') * mp.mpf('0.5') ** k, [0, mp.inf])
assert abs(qp_direct - st.accum_block_exp(Q(1, 2), L, rho)) < mp.mpf(10) ** -30
out['accumulating'] = dict(prefix=prefix, blocks=blocks, grid=grid,
                           tail_certificate_18=float((U + D) * L * rho ** 18),
                           qpochhammer_half=float(st.accum_block_exp(Q(1, 2), L, rho)))

# ------------------------------------------------------------------ G7  Thue-Morse clock
tm, r = [], Q(1)
for n in range(8):
    ell, (r1, r2, r3, r4) = st.thue_morse_cell(r, n, U, D)
    tm.append(dict(n=n, bit=st.thue_morse_bit(n), ell=fr(ell), r_n=fr(r), after_dense=fr(r1),
                   after_first=fr(r2), after_second=fr(r3), r_next=fr(r4),
                   exp_half=float(mp.e ** (-ell / 2) * (1 - (Q(13, 20) - ell) / 2) * (1 - Q(1, 10)) * (1 - Q(3, 40)))))
    r = r4
assert [x['r_next'] for x in tm[:2]] == ['1/5', '3/100']
# cross-check the exact cells against the piecewise-scale recursion
ts = tsm.thue_morse(8, exact=True)
rec = ts.phi(U, D, Q(1))
ints = [x for x in rec if x[2] == 'iv-start']
assert [fr(x[1]) for x in ints[1:]] == [x['r_n'] for x in tm[1:]]
out['thue_morse'] = dict(cells=tm, H='9/20', band=fr(Q(9, 20) * D))

# ------------------------------------------------------------------ G8  Cantor clock
gaps9 = st.cantor_gaps(9)
cant = []
for UU in (Q(1), Q(3, 5), Q(3, 10)):          # U = 5D, 3D, 1.5D with D = 1/5
    row = dict(U=fr(UU), UoverD=fr(UU / D))
    for name, t in [('1/3', Q(1, 3)), ('2/3', Q(2, 3)), ('1/4', Q(1, 4)), ('3/4', Q(3, 4)), ('1', Q(1))]:
        row[name] = fr(st.cantor_phi(t, gaps9, UU, D))
    row['formula_1'] = fr(max(Q(0), (2 * D - UU) / 3))
    assert row['1'] == row['formula_1']
    cant.append(row)
# zero set for 2D <= U < 4D: {0} U {3^-m}; test on all endpoints of generation <= 8 and U in {2D, 3D, 3.9D}
g10 = st.cantor_gaps(10)
pts = sorted({x for (a_, b_, k) in st.cantor_gaps(8) for x in (a_, b_)} | {Q(0), Q(1), Q(1, 4), Q(3, 4), Q(1, 10), Q(3, 10)})
expected = {Q(0)} | {Q(1, 3 ** m) for m in range(0, 12)}
for ratio in (Q(2), Q(3), Q(39, 10)):
    UU = ratio * D
    zeros = {p for p in pts if st.cantor_phi(p, g10, UU, D) == 0}
    assert zeros == (expected & set(pts)), (ratio, sorted(zeros)[:10])
# at U = 4D the point 7/9 = 0.20222..._3 joins the zero set
assert st.cantor_phi(Q(7, 9), g10, 4 * D, D) == 0 and st.cantor_phi(Q(7, 9), g10, Q(39, 10) * D, D) > 0
# supremum over a block after capture is D/3, attained at n + 2/3
assert max(st.cantor_phi(p, g10, 3 * D, D) for p in pts) == D / 3 == st.cantor_phi(Q(2, 3), g10, 3 * D, D)
# right-regular approximants T_m: simulated worst case converges to the exact Phi
conv = []
for m in range(1, 9):
    tsC = tsm.cantor_truncated(m, 1, exact=True)
    rec = tsC.phi(3 * D, D, Q(0))
    val = {x[0]: x[1] for x in rec}
    conv.append(dict(m=m, Phi_m_1=fr(val[Q(1)]), Phi_m_2_3=fr(val[Q(2, 3)]), Phi_m_1_4=float(
        st.cantor_phi(Q(1, 4), st.cantor_gaps(m), 3 * D, D))))
assert conv[-1]['Phi_m_2_3'] == fr(D / 3)
kap = {str(l): float(st.cantor_kappa(l)) for l in (Q(1, 2), Q(1), Q(2))}
for l in (Q(1, 2), Q(1)):
    # T_m approximants: exp(-lam (2/3)^m) prod_{gen<=m} (1 - lam mu) -> kappa(lam)
    m = 30
    logp = -float(l) * (2 / 3) ** m + sum((2 ** (k - 1)) * math.log(1 - float(l) * 3.0 ** (-k)) for k in range(1, m + 1))
    assert abs(math.exp(logp) - kap[str(l)]) < 1e-5
    # digit formula at t = 1 (all digits 2) reproduces kappa
    assert abs(float(st.cantor_exp_at(l, [2] * 60)) - kap[str(l)]) < 1e-12
cexp = {name: float(st.cantor_exp_at(Q(1, 2), st.ternary_digits(t, 40))) for name, t in
        [('1/3', Q(1, 3)), ('2/3', Q(2, 3)), ('1/4', Q(1, 4)), ('1', Q(1))]}
out['cantor'] = dict(table=cant, zero_set='{0} U {3^-m}', convergence=conv, kappa=kap, exp_half=cexp)

# ------------------------------------------------------------------ G9  dyadic clock (Lyapunov)
dy = st.dyadic_saturated(Q(1, 2), Q(1), Q(1, 10), 9)
for k, row in enumerate(dy):
    assert row['s'] == (-1) ** k * Q(2 ** k, 3) + Q(1, 6)
    assert row['s_sdelta'] < 0
    if k >= 1:
        assert row['VDelta'] > 0
out['dyadic'] = [dict(k=r_['k'], t=r_['t'], s=fr(r_['s']), s_sdelta=fr(r_['s_sdelta']),
                      corr=fr(r_['corr']), VDelta=fr(r_['VDelta'])) for r_ in dy]

# ------------------------------------------------------------------ G10 chaotic clock statistics
tsc = tsm.chaotic(0.04, 0.2, 250)
mus = np.array(tsc.rs()[1])
x = (mus - 0.04) / 0.16
frac_edges = float(np.mean((x < 0.1) | (x > 0.9)))
out['chaotic'] = dict(n=len(mus), mean=float(mus.mean()), frac_within_10pct_of_endpoints=frac_edges,
                      arcsine_prediction=float(2 * (2 / math.pi) * math.asin(math.sqrt(0.1))))

(ROOT / 'Data' / 'gallery.json').write_text(json.dumps(out, indent=2, default=str))
print(json.dumps({k: ('...' if isinstance(v, (list, dict)) else v) for k, v in out.items()}, indent=1))
print('primes: events', out['primes']['n_events'], 'first', out['primes']['first_events'][:3])
print('cantor table', out['cantor']['table'])
print('cantor conv', out['cantor']['convergence'])
print('kappa', out['cantor']['kappa'], out['cantor']['exp_half'])
print('sqrt exp', out['sqrt']['exp'])
print('chaotic', out['chaotic'])
print('PASS: gallery')
