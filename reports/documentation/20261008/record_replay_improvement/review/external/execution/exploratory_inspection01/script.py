from pathlib import Path
import json
r=Path('/workspace/scratch/fa11b6ce03cf/review_background/original')
for x in ['result/job/completion.json','result/job/run/updates/ABC_shared.json','qualification/gpu/attempt01/result.json']:
 d=json.loads((r/x).read_text());print(x);print(list(d));print({k:v for k,v in d.items() if k in ['budget','training_seconds','status','projected_training_seconds','projected_total_seconds','worst_shape_projected_seconds']})
print('metricslist sample')
m=json.loads(Path('/workspace/scratch/fa11b6ce03cf/replay_review/calculation/saved_history.json').read_text())['read_matrices']
print(m[:4]); print(type(m[0]))
