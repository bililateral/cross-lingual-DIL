from pathlib import Path
import sys,json,hashlib
root=Path('/mnt/data/cq_input'); out=Path('/mnt/data/cq_review')
p=sys.argv[1]; text=(root/p).read_text(encoding='utf-8'); lines=text.splitlines(); s=int(sys.argv[2]) if len(sys.argv)>2 else 1; e=min(int(sys.argv[3]),len(lines)) if len(sys.argv)>3 else len(lines)
with (out/'evidence/reading_events.jsonl').open('a') as f:f.write(json.dumps({'file':p,'start':s,'end':e,'total_lines':len(lines),'level':'text_read','sha256':hashlib.sha256((root/p).read_bytes()).hexdigest()},ensure_ascii=False)+'\n')
print('FILE',p,'LINES',s,e,'TOTAL',len(lines))
for i in range(s-1,e):print(f'{i+1:04d} {lines[i]}')
