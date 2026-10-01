from pathlib import Path,PurePosixPath
import hashlib,zipfile,json,sys
src=Path('/mnt/data/calibration_result_review.zip');root=Path('/mnt/data/calibration_result_audit_20260928/project');root.mkdir(exist_ok=False)
inv_path='reports/seller_alias_continual/20260928/calibration_result/review/source_inventory.json'
with zipfile.ZipFile(src) as z:
 print('members',len(z.infolist()),'CRC',z.testzip())
 inv=json.loads(z.read(inv_path)); print('inventory fields',list(inv) if isinstance(inv,dict) else 'list');print(str(inv)[:1600])
 for inf in z.infolist():
  p=PurePosixPath(inf.filename);assert not p.is_absolute() and '..' not in p.parts
 z.extractall(root)
 Path('/mnt/data/calibration_result_audit_20260928/evidence/outputs/zip_members.txt').write_text('\n'.join(z.namelist())+'\n')
 print('zip_bytes',src.stat().st_size,'sha256',hashlib.sha256(src.read_bytes()).hexdigest())
