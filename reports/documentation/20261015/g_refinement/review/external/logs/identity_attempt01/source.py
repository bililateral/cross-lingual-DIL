from pathlib import Path
import zipfile,json,hashlib
p=Path('/mnt/data/review_input(20261015-132357).zip'); r=Path('/mnt/data/g_half_review')
with zipfile.ZipFile(p) as z:
 for i in z.infolist():
  assert not Path(i.filename).is_absolute() and '..' not in Path(i.filename).parts
 z.extractall(r/'input')
 records=[{'path':i.filename,'bytes':i.file_size,'sha256':hashlib.sha256(z.read(i)).hexdigest()} for i in z.infolist()]
(r/'return/input_identity.json').write_text(json.dumps({'original_path':str(p),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'members':len(records),'zip_crc_test':z.testzip(),'inventory':records},ensure_ascii=False,indent=2)+'\n')
print((r/'input/proposal.zh.txt').read_text())
