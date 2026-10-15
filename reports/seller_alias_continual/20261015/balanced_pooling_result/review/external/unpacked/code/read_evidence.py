"""Read only authorized extracted files; record displayed line ranges."""
import pathlib,sys,json,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[2]/'input'
OUT=pathlib.Path(__file__).resolve().parents[1]
for spec in sys.argv[1:]:
 pth,_,rg=spec.partition('::'); p=ROOT/pth; data=p.read_bytes(); lines=data.decode('utf-8').splitlines()
 a,b=(map(int,rg.split('-')) if rg else (1,len(lines)))
 text='\n'.join(f'{i+1:5d} {x}' for i,x in enumerate(lines) if a<=i+1<=b)
 header=f'### {pth} lines {a}-{min(b,len(lines))} / {len(lines)} SHA256 {hashlib.sha256(data).hexdigest()}'
 print(header+'\n'+text+'\n')
 with (OUT/'raw'/'manual_reads.jsonl').open('a') as f: f.write(json.dumps({'path':pth,'start':a,'end':min(b,len(lines)),'total_lines':len(lines),'sha256':hashlib.sha256(data).hexdigest()},ensure_ascii=False)+'\n')
 with (OUT/'raw'/'manual_reads.log').open('a') as f:f.write(header+'\n'+text+'\n')
