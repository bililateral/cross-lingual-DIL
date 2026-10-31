"""One authorized saved-input CPU check; no training, inference or retry."""
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback
from unittest import mock

started = time.monotonic()
deadline = started + 60
base = Path('/home/yongpeng/cross-lingual/reports/documentation/20261030/comparison_resume')
sys.path.insert(0, str(base/'workspace/scripts'))
out = Path(sys.argv[1])
out.mkdir(exist_ok=False)
result = {'scope': 'One newly authorized train CSV parse, retaining only original C calibration 12 groups; one ftol=0 fit; 60s CPU; no training.',
          'authorization_quote': '批准这一次限定核验', 'formal_fits': 0,
          'development_reads': 0, 'heldout_reads': 0, 'owners_reads': 0,
          'model_loads': 0, 'gpu_updates': 0}


def check():
    if time.monotonic() >= deadline:
        raise TimeoutError('Authorized 60-second check exhausted')


def identity(path):
    return {'path': str(path), 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def independent(theta, values, labels):
    a, b = map(float, theta)
    terms, da, db = [], [], []
    for score, label in zip(values.flat, labels.flat, strict=True):
        x, y = float(score), float(label)
        z = a*x+b
        p = 1/(1+math.exp(-z)) if z >= 0 else math.exp(z)/(1+math.exp(z))
        terms.append(max(z, 0)+math.log1p(math.exp(-abs(z)))-y*z)
        da.append((p-y)*x)
        db.append(p-y)
    n = len(terms)
    return math.fsum(terms)/n, [math.fsum(da)/n, math.fsum(db)/n]


try:
    import numpy as np
    import scipy
    import step28_continual_comparison as method
    calibration = method.parent.calibration
    run = base/'formal/CAB_s6_L2R_text_fixed/run'
    startup_path, score_path, fit_path = (run/'startup.json', run/'scores/stage1_diagnostic.npy',
                                         run/'maps/stage1_diagnostic.json')
    startup, failed = (json.loads(p.read_text()) for p in (startup_path, fit_path))
    assert startup['spec']['order'] == 'CAB' and startup['spec']['seed'] == 6
    spec = method.Spec.from_dict(startup['spec'])
    config = method.config(spec)
    all_ids = {r['group_uid'] for rows in startup['partition'].values() for r in rows}
    selected = [r['group_uid'] for r in startup['partition']['calibration'] if r['domain'] == 'C']
    assert len(selected) == len(set(selected)) == 12
    scores = np.load(score_path, allow_pickle=False).astype(np.float64)
    assert scores.shape == (12, 378) and np.isfinite(scores).all()
    pairs = method.data.ROOT/config['data_root']/'train/supervision/pairs.csv'
    expected = startup['public_inputs']['train/supervision/pairs.csv']
    method.data.verify(pairs, expected)
    # Exclusive access receipt before parsing. Do not preserve other groups' labels.
    with (out/'access.json').open('x') as stream:
        json.dump({'train_parse_attempts': 1, 'retained_groups': selected,
                   'valid': 0, 'heldout': 0, 'owners': 0}, stream)
    kept = {uid: {} for uid in selected}
    rows_seen = 0
    with pairs.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            check()
            assert row['group_uid'] in all_ids
            rows_seen += 1
            if row['group_uid'] not in kept:
                continue
            key = (row['seller_uid_left'], row['seller_uid_right'])
            assert key not in kept[row['group_uid']] and row['label'] in ('0', '1')
            kept[row['group_uid']][key] = int(row['label'])
    labels = []
    for uid in selected:
        sellers = sorted({s for pair in kept[uid] for s in pair})
        order = list(itertools.combinations(sellers, 2))
        assert len(sellers) == 28 and len(kept[uid]) == 378 and set(order) == set(kept[uid])
        labels.append([kept[uid][pair] for pair in order])
    labels = np.asarray(labels, dtype=np.uint8)
    assert rows_seen == 180*378 and int(labels.sum()) == 240
    before_loss, before_grad = independent([failed['a'], failed['b']], scores, labels)
    initial_loss, _ = independent([1., 0.], scores, labels)
    # The saved solution is interior, so its projected and ordinary gradients agree.
    assert .001 < failed['a'] < 100 and -100 < failed['b'] < 100
    assert abs(before_loss-failed['final_nll']) <= 1e-12
    assert abs(initial_loss-failed['initial_nll']) <= 1e-12
    assert abs(max(map(abs, before_grad))-failed['projected_gradient_max']) <= 1e-11
    result.update(inputs=[identity(p) for p in (startup_path, score_path, fit_path, pairs)],
                  source=identity(Path(calibration.__file__)), verifier=identity(Path(__file__)),
                  scipy=scipy.__version__, numpy=np.__version__, selected_groups=selected,
                  original_independent={'nll': before_loss, 'gradient': before_grad},
                  old_gradient_limit_failed=max(map(abs, before_grad)) > 1e-6)
    result['formal_fits'] = 1
    with mock.patch.dict(calibration.OPTIONS, ftol=0.):
        fitted = calibration.fit(scores, labels, role='calibration', check=check)
    result['candidate'] = fitted
    after_loss, after_grad = independent([fitted['a'], fitted['b']], scores, labels)
    result['candidate_independent'] = {'nll': after_loss, 'gradient': after_grad}
    assert abs(after_loss-fitted['final_nll']) <= 1e-12
    assert .001 < fitted['a'] < 100 and -100 < fitted['b'] < 100
    assert fitted['status'] == 'PASS_CALIBRATION_FIT' and max(map(abs, after_grad)) <= 1e-6
    result['order'] = calibration.preserve_order(scores, calibration.transform(scores, fitted))
    result['status'] = 'PASS_SAVED_INPUT_CANDIDATE_CHECK_NOT_FORMAL_ENDPOINT'
except BaseException as error:
    result.update(status='FAILED_NO_RETRY', error=repr(error), traceback=traceback.format_exc())
finally:
    result['wall_seconds'] = time.monotonic()-started
    result['charged_seconds'] = math.ceil(result['wall_seconds'])
    (out/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ('status', 'formal_fits', 'wall_seconds', 'charged_seconds')}))
if 'candidate' in result:
    print(json.dumps({k: result['candidate'][k] for k in ('status', 'optimizer_message',
          'projected_gradient_max', 'optimizer_iterations', 'objective_calls', 'a', 'b', 'final_nll')}))
sys.exit(result['status'] != 'PASS_SAVED_INPUT_CANDIDATE_CHECK_NOT_FORMAL_ENDPOINT')
