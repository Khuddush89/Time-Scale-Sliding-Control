"""Hand identities from the manuscript, evaluated without importing the tsmc package."""
import json
import math
from fractions import Fraction as Q
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def value(r, a, b, D, U):
    r = abs(r)
    if r <= b * D:
        return b * D
    return max(((b - a) * r + 2 * a * b * D) / (a + b), r - a * (U - D))


def control(s, a, b, D, U):
    r = abs(s)
    w = r / b if r <= b * D else (2 * r - (b - a) * D) / (a + b)
    return -(0 if s == 0 else math.copysign(1.0, s)) * min(U, w)


def cost(s, u, a, b, D):
    return max(abs(s + a * u) + a * D, abs(s + b * u) + b * D)


checks = []
s, h, v = 0.1, 1.0, -1.0
checks.append(("delta V two-cycle", ((s + h * v) ** 2 - s * s) / (2 * h), 0.4))
checks.append(("scheduler b=0.5", value(0.6, 0.2, 0.5, 0.5, 2), 0.4))
checks.append(("scheduler b=0.6", value(0.6, 0.2, 0.6, 0.5, 2), 0.45))
checks.append(("attaining input", cost(0.6, control(0.6, 0.2, 0.5, 0.5, 2), 0.2, 0.5, 0.5), 0.4))
a, b, D, U, s0 = 0.04, 0.20, 0.30, 1.50, 2.0
Rstar, c, q = b * D, U - D, (b - a) / (a + b)
Ec = (a + b) * c / 2
e0 = s0 - Rstar
m = max(0, math.ceil((e0 - Ec) / (a * c)))
em = e0 - m * a * c
ell = math.ceil(math.log(em / 0.001) / math.log(1 / q))
checks += [("Rstar", Rstar, 0.06), ("q", q, 2 / 3), ("m", m, 38), ("em", em, 0.116), ("ell", ell, 12)]
# closed-form excess at N = m + ell equals q^ell e_m
eN = max(q ** (m + ell - k) * (e0 - k * a * c) for k in range(m + ell + 1))
checks.append(("closed-form excess", eN, q ** ell * em))
checks.append(("hybrid threshold", (1.4 - 0.06) / 0.8, 1.675))
# timing-uncertainty arithmetic F(0.6) = 0.2 + 0.6 eta - 0.2 eta^2
for eta in (0.0, 0.25, 0.5, 0.75):
    checks.append((f"F(0.6), eta={eta}", value(0.6, 0.4 * (1 - eta), 0.4 * (1 + eta), 0.5, 2.0),
                   0.2 + 0.6 * eta - 0.2 * eta ** 2))
# P_{0.8,0.2} with U = 1, D = 1/5, s0 = 1: radius at t = 1 is 1/5, steady radius 1/25
r = max(0.0, 1 - 0.8 * 0.8)
r = max(0.2 * 0.2, r - 0.8 * 0.2)
checks.append(("P_{0.8,0.2}: Phi(1)", r, 0.2))
# harmonic clock exponential: prod_{k<=N} (1 - x/k) = Gamma(N+1-x)/(Gamma(1-x) N!)
x = 0.25
prod = 1.0
for k in range(1, 31):
    prod *= 1 - x / k
checks.append(("harmonic Gamma product", prod, math.gamma(31 - x) / (math.gamma(1 - x) * math.factorial(30))))
# Cantor clock: Phi(2/3) = D/3 and Phi(1) = (2D - U)_+/3 with U = 3/10, D = 1/5
Dc, Uc = Q(1, 5), Q(3, 10)
checks.append(("Cantor Phi(1), U=1.5D", float(max(Q(0), (2 * Dc - Uc) / 3)), 1 / 30))
# accumulation block: A(0) with L = 3/5, rho = 1/10 equals 3/50
checks.append(("accumulation A(0)", float(Q(3, 5) * (Q(1, 5) - Q(1, 10))), 0.06))
# dyadic clock: s(2^k) = (-1)^k 2^k/3 + 1/6 for the saturated law
s = Q(1, 2)
for k in range(8):
    assert s == (-1) ** k * Q(2 ** k, 3) + Q(1, 6)
    s = s - 2 ** k * max(Q(-1), min(Q(1), s / Q(1, 10)))
checks.append(("dyadic s(2^8)", float(s), 256 / 3 + 1 / 6))

failed = [(n, got, exp) for n, got, exp in checks if abs(got - exp) > 1e-12]
if failed:
    raise SystemExit(failed)
(ROOT / 'Data' / 'independent_identities.json').write_text(
    json.dumps({n: got for n, got, _ in checks}, indent=2))
print("independent identities: PASS")
for n, got, exp in checks:
    print(f"  {n}: {got}")
