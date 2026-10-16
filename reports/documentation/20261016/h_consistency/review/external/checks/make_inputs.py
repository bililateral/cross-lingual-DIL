#!/usr/bin/env python3
"""Write deterministic HANDWRITTEN mathematical fixtures; no project import or data."""
from pathlib import Path
import json, math
out = Path(__file__).resolve().parent / 'inputs.json'
components = [[2*i, 2*i+1] for i in range(8)] + [[16+3*i+j for j in range(3)] for i in range(4)]
owner = {a:c for c, comp in enumerate(components) for a in comp}
view0 = [[2*a,2*a+1] for a in range(28)]
view1 = [x[:] for x in view0]
for comp in components:
    records = [r for a in comp for r in view0[a]]
    rotated = records[1:] + records[:1]
    for i,a in enumerate(comp): view1[a] = rotated[2*i:2*i+2]
swap = [x[:] for x in view0]
swap[17], swap[18] = swap[18], swap[17]
pairs = [(i,j) for i in range(56) for j in range(i+1,56)]
k_general, k_perfect, k_invariant = [], [], []
for i,j in pairs:
    a,b = i//2,j//2
    k_general.append(0.7*math.sin((i+1)*0.17+(j+1)*0.31) + 0.4*math.cos((i+1)*(j+1)*0.013))
    if a == b:
        x = 1.0
    elif 16 in (a,b):
        other = b if a == 16 else a
        x = math.log(0.8 if other == 17 else 0.15 if other == 18 else 0.002)
    else:
        x = math.log(0.4 if owner[a] == owner[b] else 0.002)
    k_perfect.append(x)
    k_invariant.append(2.0 if owner[a] == owner[b] else -2.0)
obj = {
    'scope':'Reviewer-isolated handwritten algebra only. Not production implementation, qualification, training or formal data.',
    'accounts':28, 'records':56, 'components':components, 'view0':view0, 'view1_rotated':view1,
    'view1_swap_17_18':swap, 'record_pair_order':pairs,
    'K_general':k_general, 'K_AP1_nonzero':k_perfect, 'K_correct_invariant':k_invariant,
    'binary':{'p':[0.9,0.1], 'r':[0.1,0.9]},
    'AP1_query':{'p':[0.8,0.15]+[0.002]*25,'r':[0.15,0.8]+[0.002]*25},
    'correct_vs_wrong':{'p':[0.51,0.49], 'r':[0.01,0.99], 'positive_index':0,'algebraic_logit_step':0.2},
    'finite_difference_indices':[0,1,29,100,199,387,599,1000,1300,1539],
    'tolerances':{'analytical_autograd_atol':2e-12,'finite_difference_atol':2e-9,
                  'zero_atol':2e-13,'finite_difference_step':1e-5}
}
out.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'output':out.name,'bytes':out.stat().st_size,'accounts':28,'records':56,'record_pairs':len(pairs)},ensure_ascii=False))
