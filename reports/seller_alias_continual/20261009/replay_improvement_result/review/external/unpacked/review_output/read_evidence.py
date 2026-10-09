"""Print selected attachment text and record the exact human-review range.

This utility only reads the extracted review package; it never imports project
modules, loads models, or reads project datasets/labels.
"""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = Path(__file__).resolve().parent / 'reading_log.jsonl'

def record(path, mode, detail):
    p = Path(path)
    row = {'path': str(p.relative_to(ROOT)), 'bytes': p.stat().st_size,
           'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
           'mode': mode, 'detail': detail}
    with LOG.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('paths', nargs='+')
    ap.add_argument('--start', type=int, default=1)
    ap.add_argument('--end', type=int)
    args = ap.parse_args()
    for rel in args.paths:
        p = ROOT / rel
        lines = p.read_text(encoding='utf-8-sig').splitlines()
        end = min(args.end or len(lines), len(lines))
        print('\nFILE', rel, 'LINES', args.start, end, 'OF', len(lines))
        for i in range(args.start-1, end):
            print(f'{i+1:5d} {lines[i]}')
        record(p, 'full_text_read' if args.start == 1 and end == len(lines) else 'selected_text_read',
               {'start_line': args.start, 'end_line': end, 'total_lines': len(lines)})

if __name__ == '__main__':
    main()
