from pathlib import Path
import json,hashlib,difflib
r=Path('/mnt/data/bge_continual_external_audit');p=r/'scripts/independent_checks_v2.py';s=p.read_text()
old=''' term=torch.stack([torch.logaddexp(z.new_zeros(()),v)-int(y)*v for v,y in zip(z,Y)])
 return float(term.detach().float().mean())-float(term.detach().mean())'''
new=''' # Two explicit float32 stores are necessary for mixed-target BCE values.
 # The first rounds (1-y)*z; the second rounds its sum with softplus(-z).
 # Measured per-element equality in diagnose_reference_v2, not a fitted offset.
 y=torch.tensor(Y,dtype=z.dtype)
 exact=torch.logaddexp(torch.zeros_like(z),z)-y*z
 first_store=((1-y)*z).float().double()
 second_store=(first_store+torch.logaddexp(torch.zeros_like(z),-z)).float()
 return float(second_store.detach().mean())-float(exact.detach().mean())'''
assert old in s;s=s.replace(old,new)
s=s.replace('BCE values are per-element float32 before reduction.', 'BCE values have two float32 stores before reduction.')
p2=r/'scripts/independent_checks_v3.py';p2.write_text(s)
(r/'scripts/independent_v2_to_v3.diff').write_text(''.join(difflib.unified_diff(p.read_text().splitlines(True),s.splitlines(True),fromfile=p.name,tofile=p2.name)))
record={'revision':'v2 -> v3','project_sources_modified':False,'metric_tolerance_unchanged':1e-12,'diagnosis':'outputs/reference_v2_diagnosis.json','changes':['The diagnostic measured the mixed-dtype BCE per-element intermediate float32 rounding; independent scalar reference now explicitly includes both stores. Gradient reference remains exact Bernoulli analytic graph. No empirical fitted offset and no change to input, project algorithm, losses or test acceptance thresholds.'],'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (p,p2)}}
(r/'outputs/reference_revision_v3.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
