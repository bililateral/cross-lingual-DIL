"""One-off CPU checks for the approved reference-reading change."""
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

started = time.monotonic()
sys.path.insert(0, str(Path(sys.argv[1])/'scripts'))
import numpy as np
import torch
from torch import nn
from step35_vendorlink_reference import reference_vectors, score_reference_vectors

torch.set_num_threads(2)
torch.manual_seed(41)
model = SimpleNamespace(settings=SimpleNamespace(method='H_SEQ'),
                        head=nn.Sequential(nn.Linear(10, 8), nn.ReLU(), nn.Linear(8, 1)))
query, reference = torch.randn(3, 2, 5), torch.randn(3, 12, 5)
before = {k: v.detach().clone() for k, v in model.head.state_dict().items()}
actual = score_reference_vectors(model, query, reference)
error = 0.
for size in (4, 12):
    expected = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            values = []
            for u in query[a]:
                for v in reference[b, :size]:
                    with torch.no_grad():
                        values.append(float(model.head(torch.cat((torch.abs(u-v), u*v)))))
            expected[a, b] = .5*sum(values)/len(values)+.5*math.log(sum(math.exp(x) for x in values)/len(values))
    error = max(error, float(np.max(np.abs(expected-actual['H_FIT'+str(size)+':R11']))))
    assert np.allclose(expected, actual['H_FIT'+str(size)+':R11'], rtol=1e-6, atol=1e-6)
perturbed = reference.clone()
perturbed[:, 4:] += 2
changed = score_reference_vectors(model, query, perturbed)
assert np.array_equal(actual['H_FIT4:R11'], changed['H_FIT4:R11'])
assert not np.allclose(actual['H_FIT12:R11'], changed['H_FIT12:R11'])
assert all(torch.equal(v, before[k]) for k, v in model.head.state_dict().items())
assert all(p.grad is None for p in model.head.parameters())

accounts = [dict(account_uid=str(i), records=[dict(uid=f'f{i}_{j:02d}') for j in range(12)]) for i in range(3)]
ids = [a['account_uid'] for a in accounts]
fit, base, identity = {'fit': 'handwritten'}, {'base': 'handwritten'}, {'test': True}
cache = dict(classes=ids, records=[r['uid'] for a in accounts for r in a['records']],
             fit=fit, base=base, identity=identity, encoder_frozen=True, encoder_mode='eval', vectors=reference)
completion = dict(fit=fit, base=base, identity=identity)
views = [SimpleNamespace(records=tuple(SimpleNamespace(uid=f'q{i}_{j}') for j in range(2))) for i in range(3)]
assert torch.equal(reference_vectors(cache, completion, fit, accounts, ids, views), reference)
rejections = 0
for kind in ('query_overlap', 'cache_order', 'class_order'):
    c, v, qids = copy.deepcopy(cache), copy.deepcopy(views), ids.copy()
    if kind == 'query_overlap':
        v[0].records[0].uid = accounts[0]['records'][11]['uid']
    elif kind == 'cache_order':
        c['records'][0], c['records'][1] = c['records'][1], c['records'][0]
    else:
        qids.reverse()
    try:
        reference_vectors(c, completion, fit, accounts, qids, v)
    except ValueError:
        rejections += 1
assert rejections == 3
elapsed = time.monotonic()-started
source = Path(__file__).read_bytes()
print(json.dumps(dict(status='PASS', failures=0, errors=0, independent_score_max_abs_error=error,
    extra_eight_references_affect_fit12_only=True, parameters_unchanged=True, no_gradients=True,
    isolation_rejections=rejections, actual_seconds=elapsed, charged_seconds=math.ceil(elapsed),
    source_bytes=len(source), source_sha256=hashlib.sha256(source).hexdigest())))
