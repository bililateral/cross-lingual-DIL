"""User-approved custodian-only metadata preparation, before formal training."""
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import sys
import time

started = time.monotonic()
root = Path('/home/yongpeng/cross-lingual')
base = root/'reports/documentation/20261030/vendorlink/formal'
workspace = base/'workspace'
job = base/'dependency_preflight'
if sys.platform != 'linux' or job.exists():
    raise ValueError('One fresh Linux metadata attempt only')
job.mkdir()
def timeout(signum, frame):
    raise TimeoutError('600-second preparation sublimit; within the 3900-second bucket')
signal.signal(signal.SIGALRM, timeout)
signal.alarm(600)
receipt = dict(scope='Approved canonical-to-group metadata, no model, qrels or owner access',
               formal_training=False, canonical_parse_attempts=1, qrels_reads=0)
try:
    sources = json.loads((workspace/'cpu_sources.json').read_text())
    for row in sources:
        content = (workspace/row['path']).read_bytes()
        if len(content) != row['bytes'] or hashlib.sha256(content).hexdigest() != row['sha256']:
            raise ValueError('Qualified source differs')
    sys.path.insert(0, str(workspace/'scripts'))
    import step29_vendorlink_evaluate as evaluate
    config = json.loads((root/'reports/documentation/20261030/vendorlink/native/config.json').read_text())
    allowed = dict(dependency_metadata_access=True, canonical=config['canonical'],
                   partition=config['partition'], sources=sources)
    evaluate.prepare_dependencies(job, allowed)
    value = json.loads((job/'dependencies.json').read_text())
    receipt.update(status='COMPLETE_METADATA_ONLY', groups=len(value['query_counts']),
        queries=sum(value['query_counts'].values()), components=len(value['components']),
        component_group_sizes=sorted(map(len, value['components']), reverse=True), sources=sources)
except BaseException as error:
    receipt.update(status='STOP_NO_AUTOMATIC_RETRY', error=repr(error))
    raise
finally:
    receipt.update(wall_seconds=time.monotonic()-started, charged_seconds=math.ceil(time.monotonic()-started),
                   peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    (job/'receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='sources'}), flush=True)
