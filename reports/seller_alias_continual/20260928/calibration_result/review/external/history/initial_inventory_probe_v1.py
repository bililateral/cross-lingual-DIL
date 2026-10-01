# Exact Python body of the initial inventory probe; it assumed a root inventory.
from pathlib import Path
import hashlib,zipfile,json
p=Path('/mnt/data/calibration_result_review.zip'); b=p.read_bytes(); print(len(b),hashlib.sha256(b).hexdigest())
with zipfile.ZipFile(p) as z:
 print('members',len(z.namelist()),'CRC',z.testzip())
 print('\n'.join(z.namelist()[:35]))
 inv=json.loads(z.read('source_inventory.json')); print(type(inv));print(str(inv)[:2400])
