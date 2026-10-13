#!/usr/bin/env python3
"""Record human-visible reads of the supplied evidence; never executes it."""
from pathlib import Path
import argparse, datetime, hashlib, json

BASE = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('paths', nargs='+')
    p.add_argument('--start', type=int, default=1)
    p.add_argument('--end', type=int)
    args = p.parse_args()
    for name in args.paths:
        target = BASE / 'input' / name
        data = target.read_bytes()
        lines = data.decode('utf-8-sig').splitlines()
        end = min(args.end or len(lines), len(lines))
        event = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 'path': name, 'sha256': hashlib.sha256(data).hexdigest(),
                 'bytes': len(data), 'line_count': len(lines),
                 'read_start': args.start, 'read_end': end,
                 'mode': 'human_text_read'}
        with (BASE / 'independent' / 'reading_events.jsonl').open('a') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
        block = '\n### ' + name + '\n' + '\n'.join(
            f'{i+1}: {lines[i]}' for i in range(args.start-1, end)) + '\n'
        with (BASE / 'independent' / 'reading_transcript.txt').open('a') as f:
            f.write(block)
        print(block)

if __name__ == '__main__':
    main()
