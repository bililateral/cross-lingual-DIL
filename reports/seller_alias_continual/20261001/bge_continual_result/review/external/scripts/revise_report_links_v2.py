from pathlib import Path
import hashlib,json
p=Path('/mnt/data/bge_review_evidence/scripts/05_report_crosschecks_v1.py');text=p.read_text()
text=text.replace('import csv, hashlib, json, math, re, time','import csv, hashlib, json, math, re, time, os')
text=text.replace("start=time.perf_counter()", "start=time.perf_counter()\nREPORT_OUT=O/'report_crosscheck_v2'\nREPORT_OUT.mkdir(exist_ok=False)")
text=text.replace("def save(n,x):(O/n).write_text", "def save(n,x):(REPORT_OUT/n).write_text")
old="""        if not present and mapped and (R/mapped).exists():present=True
        links.append"""
new="""        if not present and mapped and (R/mapped).exists():present=True
        # source_inventory is file-granular. A directory link can represent a
        # complete mapped subtree even when the Windows directory is not in ZIP.
        if not present:
            descendants=[v for k,v in source_to_archive.items() if k.startswith(relative.rstrip('/')+'/')]
            if descendants and all((R/v).is_file() for v in descendants):
                present=True
                mapped=os.path.commonpath(descendants)
        links.append"""
assert old in text;text=text.replace(old,new)
q=p.with_name('05_report_crosschecks_v2.py');assert not q.exists();q.write_text(text)
print(json.dumps({'reason':'Own v1 navigation diagnostic handled manifest-mapped files but not directory links. User explicitly declared returned/job mapping. v2 recognizes its complete manifest subtree. Numeric verification unchanged; v1 source, outputs and logs preserved.','source_before':str(p),'sha256_before':hashlib.sha256(p.read_bytes()).hexdigest(),'source_after':str(q),'sha256_after':hashlib.sha256(q.read_bytes()).hexdigest()},indent=2))
