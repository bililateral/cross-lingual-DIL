"""Inspect saved calibration geometry and a predeclared analytic CPU example.

No formal labels, model loading, formal refit or production source changes.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from unittest import mock

started = time.monotonic()
base = Path('/home/yongpeng/cross-lingual/reports/documentation/20261030/comparison_resume')
sys.path.insert(0, str(base / 'workspace/scripts'))
import numpy as np
import scipy
import step28_alias_calibration as calibration

run = base / 'formal/CAB_s6_L2R_text_fixed/run'
score_path = run / 'scores/stage1_diagnostic.npy'
fit_path = run / 'maps/stage1_diagnostic.json'
scores = np.load(score_path, allow_pickle=False).astype(np.float64)
fit = json.loads(fit_path.read_text())
assert scores.shape == (12, 378) and np.isfinite(scores).all()
z = fit['a'] * scores.ravel() + fit['b']
p = np.exp(-np.logaddexp(0., -z))
weight = p * (1-p)
design = np.column_stack((scores.ravel(), np.ones(scores.size)))
hessian = design.T @ (weight[:, None] * design) / scores.size
weighted_mean = np.average(scores.ravel(), weights=weight)
weighted_variance = np.average((scores.ravel()-weighted_mean)**2, weights=weight)
eigenvalues = np.linalg.eigvalsh(hessian)

# Two bins with exactly 1/40 and 3/40 positives: the unique unconstrained
# optimum is a=.4, b=-50, inside the unchanged box. No formal labels used.
logits = np.log(np.array([1/40, 3/40]) / (1-np.array([1/40, 3/40])))
x = np.repeat((logits+50)/.4, 40)
y = np.array([1]+[0]*39+[1]*3+[0]*37, dtype=np.uint8)
original = calibration.fit(x, y, role='calibration')
with mock.patch.dict(calibration.OPTIONS, ftol=0.):
    tightened = calibration.fit(x, y, role='calibration')

result = {
    'scope': 'Saved score geometry only; two fits on one fixed handwritten example. No formal labels/refit.',
    'numpy_version': np.__version__, 'scipy_version': scipy.__version__,
    'sources': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), Path(calibration.__file__), score_path, fit_path)},
    'score_summary': {'shape': list(scores.shape), 'minimum': float(scores.min()),
                      'maximum': float(scores.max()), 'mean': float(scores.mean()),
                      'std': float(scores.std()), 'all_positive': bool((scores>0).all())},
    'hessian_at_saved_fit': hessian.tolist(), 'hessian_eigenvalues': eigenvalues.tolist(),
    'hessian_condition_number': float(eigenvalues[-1]/eigenvalues[0]),
    'weighted_score_mean': float(weighted_mean),
    'weighted_score_std': float(np.sqrt(weighted_variance)),
    'column_weighted_cosine': float(hessian[0, 1]/np.sqrt(hessian[0, 0]*hessian[1, 1])),
    'handwritten_example': {'count': len(x), 'positive_count': int(y.sum()),
                            'analytic_optimum': [.4, -50.],
                            'original': original, 'ftol_zero': tightened},
    'wall_seconds': time.monotonic()-started,
    'formal_train_label_reads': 0, 'formal_fits': 0, 'gpu_updates': 0,
}
result['charged_seconds'] = math.ceil(result['wall_seconds'])
Path(sys.argv[1]).write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
print(json.dumps({k: result[k] for k in ('score_summary', 'hessian_condition_number',
      'weighted_score_mean', 'weighted_score_std', 'column_weighted_cosine', 'charged_seconds')}))
print(json.dumps({name: {k: value[k] for k in ('status','optimizer_message','a','b',
      'projected_gradient_max','optimizer_iterations','objective_calls')}
      for name, value in [('original',original),('ftol_zero',tightened)]}))
