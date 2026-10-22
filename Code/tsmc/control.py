"""Scalar sliding channel s^sigma = s + mu (v + d) on a time scale.

Formulas of Section 3 (unrevealed graininess mu(t) in [a, b]) and of the
known-graininess implicit relay of Section 4.  Assumes 0 < a <= b and U >= D >= 0.
"""
import math

import numpy as np


def control(s, a, b, D, U):
    """Minimax feedback v_*(s) of Theorem 3.2 (unrevealed next graininess in [a, b])."""
    r = abs(s)
    w = r / b if r <= b * D else (2 * r - (b - a) * D) / (a + b)
    return -np.sign(s) * min(U, w)


def implicit_relay(s, mu, U):
    """Known-graininess law v in -U Sign(s + mu v); at a right-scattered point (mu > 0)
    it equals -sgn(s) min{U, |s|/mu}."""
    if mu <= 0:
        raise ValueError("use the relay inclusion on dense intervals")
    return -np.sign(s) * min(U, abs(s) / mu)


def value(r, a, b, D, U):
    """Exact one-step minimax radius F(r) of Theorem 3.2."""
    r = abs(r)
    if r <= b * D:
        return b * D
    return max(((b - a) * r + 2 * a * b * D) / (a + b), r - a * (U - D))


def known_value(r, h, D, U):
    """F_h(r) = max{hD, r - h(U - D)} (known graininess h)."""
    return max(h * D, abs(r) - h * (U - D))


def cost(s, u, a, b, D):
    """Worst-case successor radius of a fixed input u (Lemma 3.1)."""
    return max(abs(s + a * u) + a * D, abs(s + b * u) + b * D)


def adversary(s, u, a, b, D):
    """Endpoint adversary of Lemma 3.1: returns (h, d) maximizing |s + h(u + d)|."""
    candidates = [(abs(s + h * (u + d)), h, d) for h in (a, b) for d in (-D, D)]
    _, h, d = max(candidates)
    return h, d


def predecessor(R, a, b, D, U):
    """Exact robust predecessor radius P(R) of Theorem 3.6; None if empty."""
    if R < b * D:
        return None
    radius = R + a * (U - D)
    if b > a:
        radius = min(radius, ((a + b) * R - 2 * a * b * D) / (b - a))
    return radius


def maximum_upper_step(r, a, R, D, U):
    """Largest admissible upper graininess b >= a guaranteeing |s^sigma| <= R (Theorem 3.10)."""
    r = abs(r)
    if R < a * D or r > R + a * (U - D):
        return None
    cap = R / D if D > 0 else np.inf
    den = r + 2 * a * D - R
    if den > 0:
        cap = min(cap, a * (r + R) / den)
    return cap if cap >= a - 1e-12 else None


def excess_recursion(e0, a, b, D, U, N):
    """Excess e_j = F^j(R_* + e0) - R_* by iterating e -> max{q e, e - a c}."""
    q = (b - a) / (a + b)
    c = U - D
    out = [e0]
    for _ in range(N):
        e = out[-1]
        out.append(max(q * e, e - a * c))
    return out


def excess_closed_form(e0, a, b, D, U, N):
    """Closed form e_N = max_{0<=k<=N} q^{N-k} (e0 - k a c) (Corollary 3.8)."""
    q = (b - a) / (a + b)
    c = U - D
    return max((q ** (N - k) if N - k > 0 else 1.0) * (e0 - k * a * c) for k in range(N + 1))


def capture_numbers(r0, a, b, D, U, eps):
    """Numbers of Theorem 3.7: R_*, q, c, E_c, m, e_m and the update bound for R_* + eps."""
    Rstar = b * D
    q = (b - a) / (a + b)
    c = U - D
    Ec = (a + b) * c / 2
    e0 = r0 - Rstar
    m = max(0, math.ceil((e0 - Ec) / (a * c)))
    em = e0 - m * a * c
    ell = 0 if q == 0 else max(0, math.ceil(math.log(em / eps) / math.log(1 / q)))
    return dict(Rstar=Rstar, q=q, c=c, Ec=Ec, e0=e0, m=m, em=em, ell=ell, total=m + ell)
