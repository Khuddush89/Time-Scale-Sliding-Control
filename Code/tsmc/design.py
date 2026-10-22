"""Sampling design on time scales (Section 4.2): refinement and optimal time scales.

For the implicit relay with known graininess the exact worst-case radius obeys
    Phi(sigma(t)) = max{D mu(t), Phi(t) - c mu(t)},   c = U - D,
at right-scattered points and decreases at rate c on dense intervals (Theorem 4.3).
This module provides
  * self_triggered : the time scale T_* with gaps mu_k = Phi(t_k)/U, which attains the
                     continuous-time radius [r0 - c(t - t0)]_+ at every point;
  * refine         : random refinements T' of a piecewise time scale T (T subset of T');
  * sample counts  : fewest samples on any isolated time scale versus a uniform clock.
All routines accept floats or fractions.Fraction.
"""
from fractions import Fraction as Q
import math

from .timescales import PiecewiseScale


def self_triggered(r0, U, D, K, t0=0):
    """First K points of T_* = {t0 + (r0/c)(1 - (D/U)^k)} with Phi(t_k) = (D/U)^k r0.

    Returns (points, t_star) where points is a list of (t_k, Phi(t_k), mu_k)."""
    c = U - D
    rho = D / U
    pts, t, r = [], t0, r0
    for _ in range(K):
        mu = r / U
        pts.append((t, r, mu))
        t, r = t + mu, rho * r
    return pts, t0 + r0 / c


def self_triggered_scale(r0, U, D, K, t0=0, tail=1):
    """Piecewise representation: K points of T_*, the lumped remaining gap up to t*, and a dense
    tail [t*, t* + tail] (so that the scale is right-regular)."""
    pts, t_star = self_triggered(r0, U, D, K, t0)
    pieces = [('pt', t, mu) for (t, _, mu) in pts]
    t_last = pts[-1][0] + pts[-1][2]
    pieces.append(('pt', t_last, t_star - t_last))
    pieces.append(('iv', t_star, t_star + tail))
    return PiecewiseScale('self-triggered', pieces)


def refine(scale, rng, p_points=.45, p_dense=.25, exact=False):
    """A random refinement T' of T: inside some gaps insert isolated points, inside others a dense
    interval.  Every listed point of T remains a point of T'."""
    def draw(a, b):
        if exact:
            k = int(rng.integers(1, 999))
            return a + (b - a) * Q(k, 1000)
        return a + (b - a) * float(rng.uniform(.001, .999))

    pieces = []
    for p in scale.pieces:
        if p[0] == 'iv':
            pieces.append(p)
            continue
        t, mu = p[1], p[2]
        u = rng.random()
        if u < p_points:
            cuts = sorted({draw(t, t + mu) for _ in range(int(rng.integers(1, 4)))})
            prev = t
            for x in cuts:
                pieces.append(('pt', prev, x - prev))
                prev = x
            pieces.append(('pt', prev, t + mu - prev))
        elif u < p_points + p_dense:
            x, y = sorted((draw(t, t + mu), draw(t, t + mu)))
            if y > x:
                pieces += [('pt', t, x - t), ('iv', x, y), ('pt', y, t + mu - y)]
            else:
                pieces.append(p)
        else:
            pieces.append(p)
    return PiecewiseScale(scale.name + "'", pieces)


def samples_self_triggered(r0, eps, U, D):
    """Fewest right-scattered points after which the worst-case radius is <= eps (any isolated T)."""
    return max(0, math.ceil(math.log(r0 / eps) / math.log(U / D) - 1e-12))


def samples_uniform(r0, eps, U, D):
    """Points needed by the sparsest uniform clock that keeps the radius <= eps (gap h = eps/D)."""
    c = U - D
    return max(0, math.ceil(D * (r0 - eps) / (c * eps) - 1e-12))
