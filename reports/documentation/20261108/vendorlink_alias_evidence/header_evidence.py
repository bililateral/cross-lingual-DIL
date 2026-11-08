"""Inspect authorized identity headers for missing signature/verification evidence."""
from collections import Counter
from pathlib import Path
import argparse
import hashlib
import html
import json
import os
import re
import resource
import signal
import time

from step33_vendorlink_alias_audit import digest, write_json, write_jsonl

p=argparse.ArgumentParser(); p.add_argument('--prior',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
a=p.parse_args(); start=time.monotonic(); os.umask(0o077)
resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3)); resource.setrlimit(resource.RLIMIT_CPU,(300,300)); signal.alarm(300)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2])
a.output.mkdir(parents=True,exist_ok=False); private=a.output/'evidence'; private.mkdir()
stats=Counter(); candidates=[]
for entry in map(json.loads,(a.prior/'evidence/html_sources.jsonl').open()):
    path=Path(entry['path'])
    if path.parent != Path('/home/yongpeng/cross-lingual/html-rips'): raise ValueError('Unexpected source path')
    data=path.read_bytes()
    if hashlib.sha256(data).hexdigest()!=entry['sha256']: raise ValueError('HTML source changed')
    source=data.decode('utf-8'); stats['pages']+=1
    boundaries=[source.find(x) for x in ['panel-title text-primary strong">Listings','<!--end of grams reviews-->','<h2>Recent Market Reviews</h2>']]
    boundaries=[i for i in boundaries if i>=0]
    if not boundaries:
        stats['missing_identity_boundary']+=1; continue
    header=source[:min(boundaries)]
    header=re.sub(r'(?is)<(?:script|style)\b.*?</(?:script|style)>','',header)
    text=html.unescape(re.sub('<[^>]+>',' ',header))
    signed=bool(re.search(r'BEGIN\s+PGP\s+SIGNED\s+MESSAGE',text,re.I))
    complete=bool(re.search(r'BEGIN\s+PGP\s+SIGNED\s+MESSAGE.*?END\s+PGP\s+SIGNATURE',text,re.S|re.I))
    stats['headers_with_signed_message_begin']+=signed
    stats['headers_with_complete_signed_message_words']+=complete
    proof=[]
    for m in re.finditer(r'\b(?:verified|unverified|verification|challenge|claimed|signature|impost[eo]r)\b',text,re.I):
        proof.append(text[max(0,m.start()-120):m.end()+160])
    stats['headers_with_verification_or_claim_words']+=bool(proof)
    if signed or proof:
        candidates.append({'path':str(path),'sha256':entry['sha256'],'signature_begin':signed,'complete_signature_words':complete,
            'contexts':proof,'identity_header':text if signed else None,'label':'REQUIRES_CONTEXT_NOT_AUTOMATIC_VERIFICATION'})
write_jsonl(private/'header_candidates.jsonl',candidates)
summary={'status':'AUTHORIZED_IDENTITY_HEADER_SCAN_COMPLETE','counts':dict(stats),
    'candidate_headers':len(candidates),'source_manifest_sha256':digest(a.prior/'evidence/html_sources.jsonl'),
    'script_sha256':digest(Path(__file__)),'wall_seconds':time.monotonic()-start,
    'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    'scope':'Previously matched three-market identity headers only; listings and feedback bodies excluded.',
    'limitations':'Keyword absence is not proof that no verification ever occurred; original curation provenance remains separate.'}
write_json(a.output/'summary.json',summary); print(json.dumps(summary),flush=True)
