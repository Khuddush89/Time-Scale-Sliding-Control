"""Time scales used in the paper.

A time scale on a finite window is stored as an ordered list of pieces
    ('pt', t, mu)   a right-scattered point t with graininess mu, sigma(t) = t + mu,
    ('iv', a, b)    a closed dense interval [a, b] contained in T (points of [a, b) are right-dense).
The right end b of a dense interval is listed separately as a 'pt' when it is right-scattered.
Accumulation points and Cantor dust are handled by the exact routines in
``strange.py``; the piecewise representation is used for exact simulation.

All routines work with floats or with fractions.Fraction (exact arithmetic).
"""
from dataclasses import dataclass, field
from fractions import Fraction as Q
import math

import numpy as np


@dataclass
class PiecewiseScale:
    name: str
    pieces: list = field(default_factory=list)   # ('pt', t, mu) or ('iv', a, b)

    # ------------------------------------------------------------------ geometry
    def rs(self):
        """Arrays (t, mu) of the listed right-scattered points."""
        t = [p[1] for p in self.pieces if p[0] == 'pt']
        mu = [p[2] for p in self.pieces if p[0] == 'pt']
        return t, mu

    def intervals(self):
        return [(p[1], p[2]) for p in self.pieces if p[0] == 'iv']

    def mu_bar(self):
        _, mu = self.rs()
        return max(mu) if mu else 0

    def end(self):
        last = self.pieces[-1]
        return last[1] + last[2] if last[0] == 'pt' else last[2]

    def jump_measure(self):
        """J = sum of graininess over the listed right-scattered points."""
        return sum(self.rs()[1])

    def dense_measure(self):
        return sum(b - a for a, b in self.intervals())

    # ------------------------------------------------------------ exponentials
    def exp_neg(self, lam):
        """Values of e_{-lam}(t, t0) at every listed point and at interval ends.

        Across a dense interval the factor is exp(-lam*length); across a right-scattered
        point the factor is 1 - lam*mu.  Returns a list of (t, value)."""
        val = 1.0
        out = []
        for p in self.pieces:
            if p[0] == 'iv':
                a, b = p[1], p[2]
                out.append((a, val))
                val *= math.exp(-lam * float(b - a))
                out.append((b, val))
            else:
                t, mu = p[1], p[2]
                out.append((t, val))
                val *= (1 - lam * mu)
        out.append((self.end(), val))
        return out

    # --------------------------------------------------- max-plus radius (Thm 4.3)
    def phi(self, U, D, r0):
        """Worst-case radius Phi at every listed point (recursion form of Theorem 4.3).

        Returns a list of (t, Phi(t), kind) including dense-interval endpoints and the
        final point sigma(t_last)."""
        c = U - D
        r = r0
        out = []
        for p in self.pieces:
            if p[0] == 'iv':
                a, b = p[1], p[2]
                out.append((a, r, 'iv-start'))
                r = max(r - c * (b - a), 0 * r)
                out.append((b, r, 'iv-end'))
            else:
                t, mu = p[1], p[2]
                out.append((t, r, 'pt'))
                r = max(D * mu, r - c * mu)
        out.append((self.end(), r, 'end'))
        return out

    def phi_supformula(self, U, D, r0, t, t0=0):
        """Phi(t) = max{0, r0 - c(t - t0), sup_tau [D mu(tau) - c(t - sigma(tau))]} (direct form)."""
        c = U - D
        best = max(0 * r0, r0 - c * (t - t0))
        for p in self.pieces:
            if p[0] == 'pt' and p[1] + p[2] <= t + 1e-12 * (1 + abs(t)):
                best = max(best, D * p[2] - c * (t - (p[1] + p[2])))
        return best

    # ------------------------------------------------------------ exact closed loop
    def simulate(self, U, D, s0, disturbance, dense_substeps=16, rng=None):
        """Exact closed loop of the implicit relay v in -U Sign(s + mu v).

        disturbance: 'max' (d = D sgn s0 throughout) or 'random' (random admissible values;
        piecewise constant on dense substeps).  Returns (t, s) samples at every listed point,
        interval end and dense substep."""
        rng = np.random.default_rng(0) if rng is None else rng
        sign0 = 1.0 if s0 >= 0 else -1.0
        s = s0
        out = [(self.pieces[0][1], s)]
        for p in self.pieces:
            if p[0] == 'iv':
                a, b = float(p[1]), float(p[2])
                edges = np.linspace(a, b, dense_substeps + 1)
                for j in range(dense_substeps):
                    dt = edges[j + 1] - edges[j]
                    d = D * sign0 if disturbance == 'max' else rng.uniform(-D, D)
                    if s > 0:
                        s = max(0.0, s - (U - d) * dt)
                    elif s < 0:
                        s = min(0.0, s + (U + d) * dt)
                    out.append((edges[j + 1], s))
            else:
                t, mu = float(p[1]), float(p[2])
                v = -np.sign(s) * min(U, abs(s) / mu)
                if disturbance == 'max':
                    d = D * sign0
                else:
                    d = rng.choice([-D, D]) if rng.random() < 0.6 else rng.uniform(-D, D)
                s = s + mu * (v + d)
                out.append((t + mu, s))
        return out


# ====================================================================== builders
def uniform(h, n, t0=0):
    return PiecewiseScale(f'{h}Z', [('pt', t0 + k * h, h) for k in range(n)])


def pab(a, b, n):
    """P_{a,b} = union_k [k(a+b), k(a+b)+a]; Wan et al. use a = 0.8, b = 0.2."""
    pieces = []
    for k in range(n):
        s = k * (a + b)
        pieces += [('iv', s, s + a), ('pt', s + a, b)]
    return PiecewiseScale(f'P_{{{a},{b}}}', pieces)


def harmonic(alpha, n):
    """t_k = alpha * H_k, mu(t_k) = alpha/(k+1)."""
    pieces = []
    t = alpha * 0
    for k in range(n):
        mu = alpha / (k + 1)
        pieces.append(('pt', t, mu))
        t = t + mu
    return PiecewiseScale('harmonic', pieces)


def sqrt_clock(n):
    """t_k = sqrt(k), mu(t_k) = sqrt(k+1) - sqrt(k)."""
    return PiecewiseScale('sqrt', [('pt', math.sqrt(k), math.sqrt(k + 1) - math.sqrt(k)) for k in range(n)])


def geometric(q, n, exact=False):
    """t_k = q^k (k >= 0), mu = (q-1) t."""
    q = Q(q) if exact else q
    return PiecewiseScale('geometric', [('pt', q ** k, (q - 1) * q ** k) for k in range(n)])


def primes_upto(pmax):
    sieve = np.ones(pmax + 1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(pmax ** 0.5) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    return [int(p) for p in np.flatnonzero(sieve)]


def prime_clock(pmax):
    ps = primes_upto(pmax)
    return PiecewiseScale('primes', [('pt', ps[k], ps[k + 1] - ps[k]) for k in range(len(ps) - 1)])


def chaotic(a, b, n, x0=0.3):
    """Logistic clock: mu_k = a + (b - a) x_k, x_{k+1} = 4 x_k (1 - x_k)."""
    pieces, t, x = [], 0.0, x0
    for _ in range(n):
        mu = a + (b - a) * x
        pieces.append(('pt', t, mu))
        t += mu
        x = 4 * x * (1 - x)
    return PiecewiseScale('chaotic', pieces)


def accumulating_truncated(L, rho, nblocks, K):
    """T_{L,rho} with the first K points of each block; the tail (length L rho^K) is lumped into
    the last listed gap, which is itself a time scale that differs from T_{L,rho} only there."""
    pieces = []
    for n in range(nblocks):
        for k in range(K):
            pieces.append(('pt', n + L * (1 - rho ** k), L * (1 - rho) * rho ** k))
        pieces.append(('pt', n + L * (1 - rho ** K), L * rho ** K))       # lumped tail
        pieces.append(('pt', n + L, 1 - L))                                # accumulation point
    return PiecewiseScale('accumulating', pieces)


def thue_morse_bit(n):
    return bin(n).count('1') % 2


def thue_morse(ncells, exact=False):
    """T_m = union_n ([n, n + l_n] U {n + 13/20, n + 17/20}), l_n = 1/5 + tau_n/10."""
    F = (lambda p, q: Q(p, q)) if exact else (lambda p, q: p / q)
    pieces = []
    for n in range(ncells):
        ell = F(1, 5) + thue_morse_bit(n) * F(1, 10)
        pieces += [('iv', n, n + ell), ('pt', n + ell, F(13, 20) - ell),
                   ('pt', n + F(13, 20), F(1, 5)), ('pt', n + F(17, 20), F(3, 20))]
    return PiecewiseScale('thue-morse', pieces)


def user_hybrid(cycles=5):
    """Dense interval of length 0.45 followed by gaps 0.12, 0.30, 0.08, repeated."""
    pieces, t = [], 0.0
    for _ in range(cycles):
        pieces.append(('iv', t, t + 0.45))
        t += 0.45
        for h in (0.12, 0.30, 0.08):
            pieces.append(('pt', t, h))
            t += h
    return PiecewiseScale('hybrid', pieces)


def cantor_truncated(m, nblocks=1, exact=False):
    """Right-regular approximant T_m of the Cantor clock: 2^m dense intervals of length 3^-m per
    block and all removed intervals of generation <= m as right-scattered gaps."""
    one = Q(1) if exact else 1.0
    pieces = []
    for n in range(nblocks):
        intervals = [(n * one, (n + 1) * one)]
        gaps = []
        for _ in range(m):
            new = []
            for (x, y) in intervals:
                Lh = (y - x) / 3
                gaps.append((x + Lh, Lh))
                new += [(x, x + Lh), (x + 2 * Lh, y)]
            intervals = new
        items = [('iv', x, y) for x, y in intervals] + [('pt', a, mu) for a, mu in gaps]
        items.sort(key=lambda p: (p[1], 0 if p[0] == 'iv' else 1))
        pieces += items
    return PiecewiseScale(f'cantor_m{m}', pieces)
