#!/usr/bin/env python3
"""Display specified source lines and retain the displayed evidence, without execution."""
import pathlib,sys,json
root=pathlib.Path('/mnt/data/s_review_input'); out=pathlib.Path('/mnt/data/S_contribution_review'); rel=sys.argv[1]
p=root/rel; lines=p.read_text(encoding='utf-8-sig').splitlines(); a=int(sys.argv[2]) if len(sys.argv)>2 else 1; b=min(int(sys.argv[3]),len(lines)) if len(sys.argv)>3 else len(lines)
s='\n=== '+rel+f' [lines {a}-{b}; total {len(lines)}] ===\n'+'\n'.join(f'{i}: {lines[i-1]}' for i in range(a,b+1))+'\n'
print(s)
with (out/'audit/reading_log.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(dict(file=rel,start=a,end=b,total=len(lines)),ensure_ascii=False)+'\n')
with (out/'audit/read_excerpts.txt').open('a',encoding='utf-8') as f:f.write(s)
