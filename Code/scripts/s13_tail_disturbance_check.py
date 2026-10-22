"""Exact check: vanishing upstream speed and unbounded gaps can yield growing link error."""
from fractions import Fraction as Q
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
z=Q(0);U=Q(2);rows=[]
for k in range(20):
 t=Q(4**k);h=3*t
 u=-z/h
 assert abs(u)<=U
 leader_displacement=Q(2**(k+1)-2**k)
 zn=z+h*u-leader_displacement
 assert abs(zn)==2**k
 rows.append({'k':k,'t':str(t),'gap':str(h),'input':str(u),'next_link_error':str(zn)})
 z=zn
result={'status':'PASS','exact_steps':len(rows),'rows':rows,
 'conclusion':'A zero limiting upstream speed budget does not imply a zero asymptotic link error on unbounded-gap scales.'}
(ROOT/'Data/tail_disturbance_check.json').write_text(json.dumps(result,indent=2)+'\n')
print('PASS: 20 exact rational steps; |z(4^k)| = 2^(k-1) for k>=1.')
