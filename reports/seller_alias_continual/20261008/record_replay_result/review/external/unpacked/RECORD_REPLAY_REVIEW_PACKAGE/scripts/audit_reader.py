"""Read-only evidence viewer; logs exact displayed line intervals for review scope."""
from pathlib import Path
import sys, json, hashlib, datetime, zipfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'input'
LOG = ROOT / 'reading_events.jsonl'

def bytes_for(path):
    if '!/' in path:
        archive, member = path.split('!/', 1)
        with zipfile.ZipFile(BASE/archive) as z: return z.read(member)
    return (BASE/path).read_bytes()

def record(path, mode, detail):
    data = bytes_for(path)
    obj = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'path': path, 'mode': mode, 'bytes': len(data),
           'sha256': hashlib.sha256(data).hexdigest(), **detail}
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def show(spec):
    parts = spec.rsplit('::', 1)
    path = parts[0]
    lines = bytes_for(path).decode('utf-8-sig').splitlines()
    ranges = parts[1] if len(parts) == 2 else f'1-{len(lines)}'
    intervals = []
    print(f'\nFILE {path} | total_lines={len(lines)}')
    for r in ranges.split(','):
        a, _, b = r.partition('-'); a=int(a); b=int(b or a)
        b=min(b,len(lines)); intervals.append([a,b])
        for i in range(a,b+1): print(f'{i:05d}: {lines[i-1]}')
    record(path,'displayed_text',{'total_lines':len(lines),'intervals':intervals})

if __name__ == '__main__':
    for arg in sys.argv[1:]: show(arg)
