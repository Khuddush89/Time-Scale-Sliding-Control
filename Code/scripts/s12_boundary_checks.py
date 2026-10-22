"""Exact-arithmetic regression checks at boundary cases (Cantor threshold U = 4D, q = 0,
band counterexamples, logistic lower bound)."""
from fractions import Fraction as Q
from pathlib import Path
from itertools import product
import sys,json
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'Code'))
from tsmc.strange import cantor_gaps,cantor_phi
# Compare finite prefixes followed by all 2s with the exact digit criterion at U=4D.
D=Q(1,5);U=4*D;gaps=cantor_gaps(9); cases=0
for n in range(1,8):
 for digits in product((0,2),repeat=n):
  t=sum(Q(d,3**(i+1)) for i,d in enumerate(digits))+Q(1,3**n)
  tails=[sum(Q(digits[j],3**(j-i)) for j in range(i+1,n))+Q(1,3**(n-i-1)) for i in range(n)]
  expected=all(tails[i]>=Q(1,3) for i,d in enumerate(digits) if d==2)
  assert (cantor_phi(t,gaps,U,D)==0)==expected
  cases+=1
assert cantor_phi(Q(7,9),gaps,4*D,D)==0
assert cantor_phi(Q(7,9),gaps,Q(39,10)*D,D)>0
# Gao two-cycle amplitude is not an invariant interval: near zero maps outside it.
h=Q(1);q=Q(1,2);eps=Q(1);r=eps*h/(2-q*h);s=Q(1,100)
snext=(1-q*h)*s-eps*h
assert abs(snext)>r
# The claimed moving D*mu(t) tube need not hold at a short successor gap.
D=Q(1,5);U=Q(1);h0=Q(1);h1=Q(1,10)
s1=D*h0
assert s1>D*h1
# q=0 closed form, including post-capture iterations and 0^0 convention.
q0cases=0
for e in (Q(1,4),Q(1),Q(11,5)):
 for ac in (Q(1,5),Q(1),Q(3)):
  r=e
  for N in range(12):
   formula=max((Q(0)**(N-k))*(e-k*ac) for k in range(N+1))
   assert r==formula
   r=max(Q(0),r-ac);q0cases+=1
# Upper/lower graininess limits must be distinguished in network data.
nw=json.loads((ROOT/'Data/network.json').read_text());row=nw['example']['agents'][2]
assert row['mu_upper_limit']==.2 and row['mu_lower_limit']==.04
assert abs(row['lower_bound']-.02)<1e-15
result={'status':'PASS','cantor_threshold_cases':cases,'q_zero_cases':q0cases,
'gao_counterexample':{'two_cycle_amplitude':str(r if False else Q(2,3)), 's0':'1/100','s1':str(snext)},
'moving_tube_counterexample':{'s1':str(s1),'D_mu1':str(D*h1)},
'logistic_network_lower_bound':row['lower_bound'],
'note':'Finite checks supplement the proofs and do not certify all claims.'}
(ROOT/'Data/boundary_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
