"""Review-only text reader. Does not import or execute submitted source."""
from pathlib import Path
import argparse, hashlib, json

p = argparse.ArgumentParser()
p.add_argument('path')
p.add_argument('--start', type=int, default=1)
p.add_argument('--end', type=int)
p.add_argument('--reader', default='root')
a = p.parse_args()
root = Path('/workspace/scratch/3e46faac4c35/optuna_review_input')
out = Path('/workspace/scratch/3e46faac4c35/optuna_static_review')
f = root / a.path
b = f.read_bytes()
lines = b.decode('utf-8-sig').splitlines()
end = min(a.end or len(lines), len(lines))
record = {'path': a.path, 'start': a.start, 'end': end, 'total_lines': len(lines), 'sha256': hashlib.sha256(b).hexdigest(), 'kind': 'text_read', 'reader': a.reader}
with (out / ('reads_' + a.reader + '.jsonl')).open('a') as log:
    log.write(json.dumps(record, ensure_ascii=False) + '\n')
print('FILE ' + a.path + ' | lines ' + str(a.start) + '-' + str(end) + '/' + str(len(lines)))
for n in range(a.start, end + 1):
    print(f'{n:5d} {lines[n - 1]}')
