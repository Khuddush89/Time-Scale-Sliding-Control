"""Leader-follower networks on heterogeneous clocks (Section 6).

Agents i = 1..N move in physical time, x_i' = u_i + d_i, |d_i| <= D.  The leader x_0 has a
piecewise constant velocity with |x_0'| <= gamma.  Agent i listens to in-neighbours j in N_i
(all earlier in the topological order) with convex weights w_ij and uses its own clock T_i
(a PiecewiseScale with t0 = 0):
    z_i = x_i - sum_j w_ij x_j                                   (local sliding variable)
    right-scattered t in T_i : u_i = -sgn z_i(t) min{U_i, |z_i(t)|/mu_i(t)} held on [t, sigma_i(t))
    dense part of T_i        : u_i in -U_i Sign(z_i)             (relay; sliding at z_i = 0)
Between events all velocities are constant, so the simulation below is exact (event driven):
events are clock instants, dense-interval ends, zero hits of z_i on dense parts and leader
velocity switches.

The a priori budget is W_0 = gamma, W_i = U_i + D, D_i = D + sum_j w_ij W_j, c_i = U_i - D_i > 0.
"""
from dataclasses import dataclass, field
import math

import numpy as np


@dataclass
class Agent:
    name: str
    clock: object            # PiecewiseScale starting at t0 = 0
    parents: list            # [(j, w_ij)], j < i, weights sum to 1
    U: float
    x0: float
    extra: dict = field(default_factory=dict)


class Network:
    def __init__(self, agents, D, gamma, leader_segments, leader_x0=0.0):
        """agents: list of Agent (index i = position + 1); leader_segments: [(t_start, velocity)]."""
        self.agents = agents
        self.N = len(agents)
        self.D = D
        self.gamma = gamma
        self.leader_segments = sorted(leader_segments)
        self.leader_x0 = leader_x0
        for i, a in enumerate(agents, start=1):
            assert all(j < i for j, _ in a.parents), 'parents must precede the agent'
            assert abs(sum(w for _, w in a.parents) - 1) < 1e-12
            assert all(abs(v) <= gamma + 1e-15 for _, v in leader_segments)
        self.W, self.Dk, self.ck = self.budgets()

    # ------------------------------------------------------------------ budgets
    def budgets(self):
        W = [self.gamma]
        Dk, ck = [None], [None]
        for a in self.agents:
            d = self.D + sum(w * W[j] for j, w in a.parents)
            Dk.append(d)
            ck.append(a.U - d)
            W.append(a.U + self.D)
        return W, Dk, ck

    def depth(self, i):
        if i == 0:
            return 0
        return 1 + max(self.depth(j) for j, _ in self.agents[i - 1].parents)

    def path_weights(self, i):
        """P_ik: total weight of directed paths from k to i (P_ii = 1)."""
        P = {i: 1.0}
        if i == 0:
            return P
        for j, w in self.agents[i - 1].parents:
            for k, v in self.path_weights(j).items():
                P[k] = P.get(k, 0.0) + w * v
        return P

    # ------------------------------------------------------------------ leader
    def leader_velocity(self, t):
        v = 0.0
        for ts, vel in self.leader_segments:
            if ts <= t + 1e-15:
                v = vel
        return v

    def leader_switch_after(self, t):
        for ts, _ in self.leader_segments:
            if ts > t + 1e-13:
                return ts
        return math.inf

    # ------------------------------------------------------------------ simulation
    def simulate(self, t_end, dist='push', rng=None, record_every_event=True):
        """Exact event-driven closed loop.

        dist = 'push'   : at every clock instant d_i = D sgn z_i (pushes the link away from 0),
               'random' : d_i = +-D or uniform in [-D, D], drawn at clock instants,
               a number : constant disturbance d_i = dist for every agent.
        Returns dict with event times, positions (N+1 columns) and per-agent own-clock records."""
        rng = np.random.default_rng(1) if rng is None else rng
        N, D = self.N, self.D
        x = np.array([self.leader_x0] + [a.x0 for a in self.agents], dtype=float)
        v = np.zeros(N + 1)
        mode = [None] * (N + 1)
        u = np.zeros(N + 1)
        d = np.zeros(N + 1)
        piece = [-1] * (N + 1)
        next_clock = [0.0] * (N + 1)
        own = {i: [] for i in range(1, N + 1)}        # (t, x_i, z_i, kind) at own clock events
        t = 0.0
        times, X = [], []

        def z_of(i):
            return x[i] - sum(w * x[j] for j, w in self.agents[i - 1].parents)

        def draw_d(zi):
            if not isinstance(dist, str):
                return float(dist)                      # constant disturbance
            if dist == 'push':
                return D * (1.0 if zi >= 0 else -1.0)
            if rng.random() < .6:
                return D * rng.choice([-1.0, 1.0])
            return D * rng.uniform(-1, 1)

        guard = 0
        while t < t_end - 1e-13:
            guard += 1
            assert guard < 5_000_000
            # --- clock events at time t (topological order: parents' positions already final)
            for i in range(1, N + 1):
                if next_clock[i] <= t + 1e-12:
                    pieces = self.agents[i - 1].clock.pieces
                    piece[i] += 1
                    if piece[i] >= len(pieces):
                        raise RuntimeError(f'clock of agent {i} exhausted at t = {t}')
                    p = pieces[piece[i]]
                    zi = z_of(i)
                    if p[0] == 'pt':
                        mu = float(p[2])
                        mode[i] = 'hold'
                        u[i] = -math.copysign(1.0, zi) * min(self.agents[i - 1].U, abs(zi) / mu) if zi != 0 else 0.0
                        d[i] = draw_d(zi)
                        next_clock[i] = float(p[1]) + mu
                        own[i].append((t, x[i], zi, 'pt'))
                    else:
                        a_, b_ = float(p[1]), float(p[2])
                        next_clock[i] = b_
                        own[i].append((t, x[i], zi, 'iv-start'))
                        if abs(zi) <= 1e-13:
                            mode[i] = 'slide'
                            x[i] -= zi
                        else:
                            mode[i] = 'approach'
                            u[i] = -math.copysign(self.agents[i - 1].U, zi)
                            d[i] = draw_d(zi)
            # --- velocities in topological order
            v[0] = self.leader_velocity(t)
            for i in range(1, N + 1):
                if mode[i] == 'slide':
                    v[i] = sum(w * v[j] for j, w in self.agents[i - 1].parents)
                else:
                    v[i] = u[i] + d[i]
            # --- next event
            t_next = min(min(next_clock[1:]), self.leader_switch_after(t), t_end)
            zero_hit = [math.inf] * (N + 1)
            for i in range(1, N + 1):
                if mode[i] == 'approach':
                    zi = z_of(i)
                    zdot = v[i] - sum(w * v[j] for j, w in self.agents[i - 1].parents)
                    if zi * zdot < 0:
                        zero_hit[i] = t + abs(zi) / abs(zdot)
            t_next = min(t_next, min(zero_hit))
            if record_every_event:
                times.append(t)
                X.append(x.copy())
            # --- advance
            x = x + v * (t_next - t)
            t = t_next
            for i in range(1, N + 1):
                if zero_hit[i] <= t + 1e-13 and mode[i] == 'approach':
                    x[i] -= z_of(i)           # remove rounding: exact zero
                    mode[i] = 'slide'
                    own[i].append((t, x[i], 0.0, 'zero'))
            # dense-interval ends are recorded as 'iv-end'
            for i in range(1, N + 1):
                if next_clock[i] <= t + 1e-12 and mode[i] in ('slide', 'approach'):
                    own[i].append((t, x[i], z_of(i), 'iv-end'))
        times.append(t)
        X.append(x.copy())
        return {'t': np.array(times), 'x': np.array(X), 'own': own}

    # ------------------------------------------------------------------ bounds
    def link_phi(self, i, r0):
        """Max-plus radius of link i on its own clock: list of (t, Phi, kind)."""
        a = self.agents[i - 1]
        return a.clock.phi(a.U, self.Dk[i], r0)

    def phi_hat(self, i, r0, tau):
        """Physical-time envelope of |z_i| at times tau (array)."""
        a = self.agents[i - 1]
        Di, ci = self.Dk[i], self.ck[i]
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        out = np.full(tau.shape, np.nan)
        r = r0
        for p in a.clock.pieces:
            if p[0] == 'pt':
                t0_, mu = float(p[1]), float(p[2])
                m = (tau >= t0_ - 1e-12) & (tau < t0_ + mu - 1e-12)
                out[m] = np.where(np.abs(tau[m] - t0_) <= 1e-12, r, max(r, Di * mu))
                r = max(Di * mu, r - ci * mu)
            else:
                a_, b_ = float(p[1]), float(p[2])
                m = (tau >= a_ - 1e-12) & (tau < b_ - 1e-12)
                out[m] = np.maximum(r - ci * (tau[m] - a_), 0.0)
                r = max(r - ci * (b_ - a_), 0.0)
        m = np.isnan(out)
        out[m] = r
        return out

    def consensus_bound(self, tau, z0):
        """B_i(tau) = Phi_hat_i(tau) + sum_j w_ij B_j(tau); z0[i] = |z_i(0)|."""
        tau = np.atleast_1d(np.asarray(tau, dtype=float))
        B = [np.zeros_like(tau)]
        for i, a in enumerate(self.agents, start=1):
            b = self.phi_hat(i, z0[i], tau)
            for j, w in a.parents:
                b = b + w * B[j]
            B.append(b)
        return B

    def asymptotic_bands(self, limsup_mu):
        """Link bands D_i limsup mu_i and consensus bands sum_k P_ik D_k limsup mu_k."""
        link = [0.0] + [self.Dk[i] * limsup_mu[i] for i in range(1, self.N + 1)]
        cons = [0.0]
        for i, a in enumerate(self.agents, start=1):
            cons.append(link[i] + sum(w * cons[j] for j, w in a.parents))
        lower = [0.0] + [(self.D + self.gamma) * limsup_mu[i] for i in range(1, self.N + 1)]
        return link, cons, lower
