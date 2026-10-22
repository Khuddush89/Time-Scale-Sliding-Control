"""Section 6: leader-follower networks on heterogeneous clocks.

(a) Random verification: random acyclic networks (convex weights), random clocks (uniform, random
    isolated, mixed dense/scattered, accumulating, logistic, P_{a,b}, Thue-Morse, Cantor
    approximants), random leader motion and disturbances.  Exact event-driven simulation; checks
      |z_i(t)| <= Phi_hat_i(t)  and  |x_i(t) - x_0(t)| <= B_i(t)  at every event time.
(b) Attainment for agents listening only to the leader (adversarial leader and disturbance).
(c) Refinement: refining one clock never increases any bound B_i.
(d) Speed transmission on isolated clocks: observed tail speeds <= D + min{U, theta D_i}.
(e) The seven-agent example on seven clocks (data for the figures and tables of Section 6).
(f) Cascade depth guaranteed by Proposition 6.4 for a common input bound U.
"""
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'Code'))
from tsmc import timescales as tsm  # noqa: E402
from tsmc.design import refine  # noqa: E402
from tsmc.network import Agent, Network  # noqa: E402

rng = np.random.default_rng(20260927)
DATA = ROOT / 'Data'


# ------------------------------------------------------------------ random clocks
def random_isolated(rng, t_end):
    a, b = sorted(rng.uniform(.02, .6, size=2))
    pieces, t = [], 0.0
    while t < t_end + 2:
        mu = rng.uniform(a, b + 1e-9)
        pieces.append(('pt', t, mu))
        t += mu
    return tsm.PiecewiseScale('iso', pieces)


def random_mixed(rng, t_end):
    pieces, t = [], 0.0
    while t < t_end + 2:
        if rng.random() < .45:
            L = rng.uniform(.05, .8)
            pieces.append(('iv', t, t + L))
            t += L
        mu = rng.uniform(.02, .5)
        pieces.append(('pt', t, mu))
        t += mu
    return tsm.PiecewiseScale('mixed', pieces)


def random_clock(rng, t_end):
    kind = rng.choice(['uniform', 'iso', 'mixed', 'acc', 'logistic', 'pab', 'tm', 'cantor'])
    n = int(t_end) + 3
    if kind == 'uniform':
        h = rng.uniform(.05, .5)
        return tsm.uniform(h, int((t_end + 2) / h) + 2), kind
    if kind == 'iso':
        return random_isolated(rng, t_end), kind
    if kind == 'mixed':
        return random_mixed(rng, t_end), kind
    if kind == 'acc':
        return tsm.accumulating_truncated(rng.uniform(.3, .8), rng.uniform(.3, .8), n, 14), kind
    if kind == 'logistic':
        return tsm.chaotic(.04, .2, int((t_end + 2) / .04) + 2, x0=rng.uniform(.1, .9)), kind
    if kind == 'pab':
        a = rng.uniform(.3, .9)
        return tsm.pab(a, 1 - a, n), kind
    if kind == 'tm':
        return tsm.thue_morse(n), kind
    return tsm.cantor_truncated(int(rng.integers(3, 6)), nblocks=n), kind


def random_network(rng, t_end):
    N = int(rng.integers(3, 9))
    D = rng.uniform(0, .3)
    gamma = rng.uniform(0, .6)
    W = [gamma]
    agents = []
    for i in range(1, N + 1):
        k = int(rng.integers(1, min(i, 3) + 1))
        js = sorted(rng.choice(i, size=k, replace=False).tolist())
        w = rng.dirichlet(np.ones(k))
        parents = [(int(j), float(wj)) for j, wj in zip(js, w)]
        Di = D + sum(wj * W[j] for j, wj in parents)
        U = Di + rng.uniform(.05, .6)
        W.append(U + D)
        clock, _ = random_clock(rng, t_end)
        agents.append(Agent(f'a{i}', clock, parents, U, float(rng.uniform(-3, 3))))
    segs, t = [], 0.0
    while t < t_end:
        segs.append((t, float(rng.uniform(-gamma, gamma)) if rng.random() < .5 else float(gamma * rng.choice([-1, 1]))))
        t += rng.uniform(.2, 2)
    return Network(agents, D, gamma, segs, leader_x0=float(rng.uniform(-1, 1)))


def z0_of(net):
    x = [net.leader_x0] + [a.x0 for a in net.agents]
    return [0.0] + [abs(x[i] - sum(w * x[j] for j, w in net.agents[i - 1].parents)) for i in range(1, net.N + 1)]


def check(net, sim):
    t, X = sim['t'], sim['x']
    z0 = z0_of(net)
    B = net.consensus_bound(t, z0)
    wl, wc = -np.inf, -np.inf
    for i in range(1, net.N + 1):
        zi = X[:, i] - sum(w * X[:, j] for j, w in net.agents[i - 1].parents)
        wl = max(wl, float(np.max(np.abs(zi) - net.phi_hat(i, z0[i], t))))
        wc = max(wc, float(np.max(np.abs(X[:, i] - X[:, 0]) - B[i])))
    return wl, wc, len(t)


# ------------------------------------------------------------------ (a) random verification
worst_link, worst_cons, n_events, n_agents, kinds = -np.inf, -np.inf, 0, 0, {}
for case in range(160):
    t_end = float(rng.uniform(4, 9))
    net = random_network(rng, t_end)
    sim = net.simulate(t_end, dist=rng.choice(['push', 'random']), rng=rng)
    wl, wc, ne = check(net, sim)
    worst_link, worst_cons = max(worst_link, wl), max(worst_cons, wc)
    n_events += ne
    n_agents += net.N
assert worst_link < 1e-9 and worst_cons < 1e-9

# ------------------------------------------------------------------ (b) attainment for leader listeners
attain_err = 0.0
for case in range(60):
    t_end = 6.0
    clock, _ = random_clock(rng, t_end)
    D, gamma = rng.uniform(0, .3), rng.uniform(0, .6)
    U = D + gamma + rng.uniform(.05, .8)
    x0 = float(rng.uniform(-3, 3))
    s = 1.0 if x0 >= 0 else -1.0
    net = Network([Agent('a1', clock, [(0, 1.0)], U, x0)], D, gamma, [(0.0, -s * gamma)])
    sim = net.simulate(t_end, dist=s * D)
    phi = {round(t, 12): r for (t, r, _) in net.link_phi(1, abs(x0))}
    for (t, xi, zi, kind) in sim['own'][1]:
        key = round(t, 12)
        if key in phi and kind in ('pt', 'iv-start'):
            attain_err = max(attain_err, abs(abs(zi) - phi[key]))
assert attain_err < 1e-9

# ------------------------------------------------------------------ (c) refinement never increases bounds
ref_worst = -np.inf
for case in range(60):
    t_end = 6.0
    net = random_network(rng, t_end)
    k = int(rng.integers(1, net.N + 1))
    agents2 = [Agent(a.name, refine(a.clock, rng) if i == k else a.clock, a.parents, a.U, a.x0)
               for i, a in enumerate(net.agents, start=1)]
    net2 = Network(agents2, net.D, net.gamma, net.leader_segments, net.leader_x0)
    tau = np.linspace(0, t_end, 3001)
    B1, B2 = net.consensus_bound(tau, z0_of(net)), net2.consensus_bound(tau, z0_of(net2))
    for i in range(1, net.N + 1):
        ref_worst = max(ref_worst, float(np.max(B2[i] - B1[i])))
assert ref_worst < 1e-12

# ------------------------------------------------------------------ (d) speed transmission, isolated clocks
speed_margin = np.inf
for case in range(40):
    D, gamma = .1, .4
    theta_clock = rng.choice(['uniform', 'two-gap', 'logistic'])
    if theta_clock == 'uniform':
        h = .2
        clock = tsm.uniform(h, 200)
        theta = 1.0
    elif theta_clock == 'two-gap':
        a_, b_ = .2 / 3, .2
        pieces, t = [], 0.0
        for k in range(300):
            mu = b_ if k % 2 == 0 else a_
            pieces.append(('pt', t, mu))
            t += mu
        clock = tsm.PiecewiseScale('two-gap', pieces)
        theta = b_ / a_
    else:
        clock = tsm.chaotic(.04, .2, 500, x0=rng.uniform(.1, .9))
        theta = 5.0
    U = float(rng.uniform(.8, 3))
    net = Network([Agent('a1', clock, [(0, 1.0)], U, float(rng.uniform(-2, 2)))], D, gamma,
                  [(float(t), float(gamma * rng.choice([-1, 1]))) for t in np.arange(0, 20, .7)])
    sim = net.simulate(18.0, dist='push')
    t, X = sim['t'], sim['x']
    v = np.diff(X[:, 1]) / np.maximum(np.diff(t), 1e-15)
    tail = t[:-1] > 10
    bound = D + min(U, theta * (D + gamma))
    speed_margin = min(speed_margin, bound - float(np.max(np.abs(v[tail & (np.diff(t) > 1e-9)]))))
assert speed_margin > -1e-9

# ------------------------------------------------------------------ (e) seven agents on seven clocks
T_END = 9.0
D, gamma = 0.1, 0.4
CLOCKS = {
    1: ('uniform', r'$\frac14\mathbb Z$', tsm.uniform(0.25, 60), 0.25),
    2: ('pab', r'$\mathbb P_{0.8,0.2}$', tsm.pab(0.8, 0.2, 14), 0.2),
    3: ('logistic', r'$\mathbb T_6$ (logistic)', tsm.chaotic(0.04, 0.2, 400), 0.2),
    4: ('thue-morse', r'$\mathbb T_8$ (Thue--Morse)', tsm.thue_morse(14), 0.45),
    5: ('sqrt', r'$\mathbb T_3=\{\sqrt k\}$', tsm.sqrt_clock(200), 0.0),
    6: ('cantor', r'$\mathbb T_9$ (Cantor, $m=6$)', tsm.cantor_truncated(6, nblocks=14), 1 / 3),
    7: ('accumulating', r'$\mathbb T_{3/5,1/2}$', tsm.accumulating_truncated(0.6, 0.5, 14, 18), 0.4),
}
PARENTS = {1: [(0, 1.0)], 2: [(0, 1.0)], 3: [(1, 1.0)], 4: [(1, .5), (2, .5)], 5: [(2, 1.0)],
           6: [(3, 1.0)], 7: [(4, 2 / 3), (5, 1 / 3)]}
X0 = {1: 1.5, 2: -1.0, 3: 2.0, 4: 0.5, 5: -1.8, 6: 1.2, 7: -0.6}
MARGIN = 0.3
W = {0: gamma}
agents = []
for i in range(1, 8):
    Di = D + sum(w * W[j] for j, w in PARENTS[i])
    U = round(Di + MARGIN, 10)
    W[i] = U + D
    agents.append(Agent(f'agent {i}', CLOCKS[i][2], PARENTS[i], U, X0[i]))
SEGS = [(0.0, 0.4), (1.3, -0.4), (2.9, 0.25), (4.1, 0.4), (5.2, -0.3), (6.6, 0.4), (7.7, -0.4)]
net = Network(agents, D, gamma, SEGS)
sim = net.simulate(T_END, dist='push')
t, X = sim['t'], sim['x']
z0 = z0_of(net)
B = net.consensus_bound(t, z0)
limsup_mu = [0.0] + [CLOCKS[i][3] for i in range(1, 8)]
link_band, cons_band, lower = net.asymptotic_bands(limsup_mu)
# For the fixed logistic seed, b is an upper bound, not a proved limsup.
lower[3] = (D + gamma) * 0.04
wl, wc, _ = check(net, sim)
assert wl < 1e-9 and wc < 1e-9
rows = []
tail = t >= 6.0
for i in range(1, 8):
    zi = X[:, i] - sum(w * X[:, j] for j, w in PARENTS[i])
    Ei = np.abs(X[:, i] - X[:, 0])
    P = net.path_weights(i)
    rows.append({'agent': i, 'clock': CLOCKS[i][0], 'parents': PARENTS[i], 'U': agents[i - 1].U,
                 'D_i': net.Dk[i], 'c_i': net.ck[i], 'mu_upper_limit': limsup_mu[i], 'mu_lower_limit': (0.04 if i == 3 else limsup_mu[i]), 'depth': net.depth(i),
                 'link_band': link_band[i], 'consensus_band': cons_band[i], 'lower_bound': lower[i],
                 'observed_link_tail': float(np.max(np.abs(zi[tail]))),
                 'observed_consensus_tail': float(np.max(Ei[tail])),
                 'bound_tail_max': float(np.max(B[i][tail])),
                 'path_weights': {str(k): v for k, v in P.items() if k != 0}})
# events (global) and own-clock records for the figures
with (DATA / 'network_events.csv').open('w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(['t'] + [f'x{i}' for i in range(8)] + [f'B{i}' for i in range(1, 8)] +
                 [f'phihat{i}' for i in range(1, 8)])
    PH = [net.phi_hat(i, z0[i], t) for i in range(1, 8)]
    for k in range(len(t)):
        wtr.writerow([f'{t[k]:.12g}'] + [f'{X[k, i]:.12g}' for i in range(8)] +
                     [f'{B[i][k]:.12g}' for i in range(1, 8)] + [f'{PH[i - 1][k]:.12g}' for i in range(1, 8)])
with (DATA / 'network_own.csv').open('w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(['agent', 't', 'x', 'z', 'kind', 'E', 'B', 'phihat'])
    for i in range(1, 8):
        rec = sim['own'][i]
        ti = np.array([r[0] for r in rec])
        Bi = net.consensus_bound(ti, z0)[i]
        ph = net.phi_hat(i, z0[i], ti)
        x0t = np.interp(ti, t, X[:, 0])            # the leader is piecewise linear between events
        for k, (tk, xk, zk, kind) in enumerate(rec):
            wtr.writerow([i, f'{tk:.12g}', f'{xk:.12g}', f'{zk:.12g}', kind, f'{abs(xk - x0t[k]):.12g}',
                          f'{Bi[k]:.12g}', f'{ph[k]:.12g}'])
# clock pieces for the strips
with (DATA / 'network_clocks.csv').open('w', newline='') as f:
    wtr = csv.writer(f)
    wtr.writerow(['agent', 'kind', 'a', 'b_or_mu'])
    for i in range(1, 8):
        for p in CLOCKS[i][2].pieces:
            if float(p[1]) > T_END + 1:
                break
            wtr.writerow([i, p[0], f'{float(p[1]):.12g}', f'{float(p[2]):.12g}'])

# ------------------------------------------------------------------ (f) cascade depth (Prop. 6.4)
def guaranteed_depth(U, theta, D=0.1, gamma=0.4, mmax=200):
    Dm = D + gamma
    m = 0
    while Dm < U and m < mmax:
        m += 1
        Dm = 2 * D + min(U, theta * Dm)
    return m


depth_table = []
for theta in (1, 2, 3, 5):
    depth_table.append({'theta': theta, 'depths': {str(U): guaranteed_depth(U, theta) for U in (1, 2, 3, 5, 10)}})

out = {'status': 'PASS', 'random_networks': 160, 'random_agents': n_agents, 'random_events': n_events,
       'max_link_excess': worst_link, 'max_consensus_excess': worst_cons,
       'attainment_cases': 60, 'max_attainment_error': attain_err,
       'refinement_cases': 60, 'refinement_max_increase': ref_worst,
       'speed_cases': 40, 'speed_min_margin': speed_margin,
       'example': {'D': D, 'gamma': gamma, 'margin': MARGIN, 'events': int(len(t)),
                   'max_link_excess': wl, 'max_consensus_excess': wc, 'agents': rows},
       'depth_table': depth_table}
(DATA / 'network.json').write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != 'example'}, indent=1))
for r in rows:
    print(r['agent'], r['clock'], 'U', r['U'], 'D_i', round(r['D_i'], 4), 'link', round(r['link_band'], 4),
          'cons', round(r['consensus_band'], 4), 'lower', round(r['lower_bound'], 4),
          'obs', round(r['observed_consensus_tail'], 4))
