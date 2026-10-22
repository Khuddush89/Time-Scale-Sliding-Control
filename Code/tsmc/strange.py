"""Exact computations on the unusual time scales of Section 5 and Supplementary Sections S1-S2.

Every function returns exact rationals (fractions.Fraction) when the inputs are rational,
and high-precision floats (mpmath) for transcendental quantities.
"""
from fractions import Fraction as Q
import math

import mpmath as mp

mp.mp.dps = 40


def _m(x):
    """Convert int/float/Fraction/mpf to an mpmath number."""
    if isinstance(x, Q):
        return mp.mpf(x.numerator) / x.denominator
    return mp.mpf(x)


# ---------------------------------------------------------------- radius recursions
def G(h, r, U, D):
    """Scattered transfer G_h(r) = max{D h, r - c h}."""
    return max(D * h, r - (U - D) * h)


def Qdense(ell, r, U, D):
    """Dense transfer Q_ell(r) = (r - c ell)_+."""
    return max(0 * r, r - (U - D) * ell)


# ---------------------------------------------------------------- harmonic clock
def harmonic_radii(alpha, U, D, r0, n):
    """Worst-case radii r_k at t_k = alpha H_k (exact if inputs are Fractions)."""
    r = [r0]
    for k in range(n):
        r.append(G(alpha / (k + 1), r[-1], U, D))
    return r


def harmonic_exp(lam, alpha, N):
    """e_{-lam}(t_N, 0) = Gamma(N + 1 - x) / (Gamma(1 - x) Gamma(N + 1)), x = lam alpha."""
    x = _m(lam) * _m(alpha)
    return mp.gamma(N + 1 - x) / (mp.gamma(1 - x) * mp.gamma(N + 1))


def harmonic_exp_limit_ratio(lam, alpha):
    """lim e_{-lam}(t_N,0) e^{lam t_N} = e^{x gamma} / Gamma(1 - x), x = lam alpha."""
    x = _m(lam) * _m(alpha)
    return mp.e ** (x * mp.euler) / mp.gamma(1 - x)


# ---------------------------------------------------------------- square-root clock
def sqrt_exp(lam, N):
    """e_{-lam}(sqrt N, 0) = prod_{n<N} (1 - lam (sqrt(n+1) - sqrt n))."""
    lam = _m(lam)
    return mp.e ** mp.fsum(mp.log(1 - lam * (mp.sqrt(n + 1) - mp.sqrt(n))) for n in range(N))


# ---------------------------------------------------------------- accumulating blocks
def accum_A(r, L, rho, U, D):
    """Exact worst-case radius at the accumulation point l_n = n + L."""
    c = U - D
    return max(r - c * L, L * (D - U * rho), 0 * r)


def accum_B(r, L, rho, U, D):
    """Exact block transfer r(n) -> r(n+1)."""
    c = U - D
    return max(D * (1 - L), accum_A(r, L, rho, U, D) - c * (1 - L))


def accum_prefix(r, L, rho, U, D, k):
    """Exact worst-case radius at t_{n,k} (Section 5.1)."""
    out = r
    for j in range(k):
        out = G(L * (1 - rho) * rho ** j, out, U, D)
    return out


def accum_block_exp(lam, L, rho):
    """e_{-lam}(l_n, n) = (lam L (1 - rho); rho)_infinity (q-Pochhammer symbol)."""
    return mp.qp(_m(lam) * _m(L) * (1 - _m(rho)), _m(rho))


# ---------------------------------------------------------------- Thue-Morse clock
def thue_morse_bit(n):
    return bin(n).count('1') % 2


def thue_morse_cell(r, n, U, D):
    """Exact cell transfer r_n -> r_{n+1} = G_{3/20} G_{1/5} G_{13/20 - l_n} Q_{l_n}(r_n).
    Returns the four intermediate radii."""
    ell = Q(1, 5) + thue_morse_bit(n) * Q(1, 10)
    r1 = Qdense(ell, r, U, D)
    r2 = G(Q(13, 20) - ell, r1, U, D)
    r3 = G(Q(1, 5), r2, U, D)
    r4 = G(Q(3, 20), r3, U, D)
    return ell, [r1, r2, r3, r4]


# ---------------------------------------------------------------- Cantor clock
def cantor_gaps(m, offset=Q(0)):
    """All removed intervals (a, b, level) of generation <= m inside [offset, offset + 1]."""
    gaps, intervals = [], [(Q(0), Q(1))]
    for k in range(1, m + 1):
        new = []
        for (x, y) in intervals:
            L = (y - x) / 3
            gaps.append((offset + x + L, offset + x + 2 * L, k))
            new += [(x, x + L), (x + 2 * L, y)]
        intervals = new
    gaps.sort()
    return gaps


def cantor_phi(t, gaps, U, D, r0=Q(0), t0=Q(0)):
    """Phi(t) = max{0, r0 - c(t - t0), sup_gaps [D mu - c(t - b)]} over the supplied gaps."""
    c = U - D
    best = max(Q(0), r0 - c * (t - t0))
    for (a, b, _) in gaps:
        if b <= t:
            best = max(best, D * (b - a) - c * (t - b))
    return best


def ternary_digits(t, n):
    """First n ternary digits of t in [0, 1] using the {0, 2} expansion for points of C
    (left endpoints of removed intervals end in ...0222...)."""
    digits = []
    x = Q(t)
    for _ in range(n):
        x *= 3
        d = int(x)
        if d == 1 and x == 1:        # x = 1 exactly: write ...0222...
            d = 0
        elif d == 3:
            d = 2
        digits.append(d)
        x -= d
    return digits


def cantor_kappa(lam, K=400):
    """kappa(lam) = e_{-lam}(n+1, n) on the Cantor clock = prod_k (1 - lam 3^-k)^(2^(k-1))."""
    lam = _m(lam)
    return mp.e ** mp.fsum((2 ** (k - 1)) * mp.log(1 - lam * mp.mpf(3) ** (-k)) for k in range(1, K))


_KAPPA_CACHE = {}


def _kappa_level(lam, i):
    """Cached kappa(lam 3^-i) (1 - lam 3^-i): factor of a digit 2 at position i."""
    key = (str(lam), i)
    if key not in _KAPPA_CACHE:
        x = _m(lam) * mp.mpf(3) ** (-i)
        _KAPPA_CACHE[key] = cantor_kappa(x) * (1 - x)
    return _KAPPA_CACHE[key]


def cantor_exp_at(lam, digits):
    """e_{-lam}(t, 0) for t = sum d_i 3^-i in C (finite digit list, d_i in {0, 2}):
    each digit 2 at position i contributes kappa(lam 3^-i) (the full sub-block to the left)
    times (1 - lam 3^-i) (the level-i gap)."""
    val = mp.mpf(1)
    for i, d in enumerate(digits, start=1):
        if d == 2:
            val *= _kappa_level(lam, i)
    return val


# ---------------------------------------------------------------- dyadic clock
def dyadic_saturated(s0, U, phi, n):
    """Saturated law v = -U sat(s/phi) (Baby-Nivas form) on T = 2^N0, D = 0."""
    s, t, rows = s0, 1, []
    for k in range(n):
        mu = t
        v = -U * max(-1, min(1, s / phi))
        s_next = s + mu * v
        sd = (s_next - s) / mu
        rows.append(dict(k=k, t=t, mu=mu, s=s, v=v, s_next=s_next, s_sdelta=s * sd,
                         VDelta=(s_next ** 2 - s ** 2) / (2 * mu), corr=mu * sd ** 2 / 2))
        s, t = s_next, 2 * t
    return rows
