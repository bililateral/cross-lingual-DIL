"""Package the completed review and its actual evidence; no scientific run."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import zipfile

root = Path(__file__).resolve().parent
input_source = root.parent / 'upload/review_input(8).zip'
(root / 'input').mkdir(exist_ok=True)
shutil.copyfile(input_source, root / 'input/review_input.zip')

archive = root / 'REPLAY_IMPROVEMENT_REVIEW_PACKAGE.zip'
excluded = {'MANIFEST.json', 'delivery_validation.json', archive.name, 'library_save_result.json'}
payloads = sorted(p for p in root.rglob('*') if p.is_file() and p.name not in excluded and '__pycache__' not in p.parts)
utf8, syntax = [], []
for path in payloads:
    if path.suffix in ('.txt', '.json', '.csv', '.py'):
        content = path.read_text(encoding='utf-8')
        utf8.append(str(path.relative_to(root)))
        if path.suffix == '.py':
            ast.parse(content)
            syntax.append(str(path.relative_to(root)))
scope = json.loads((root / 'READING_SCOPE.json').read_text(encoding='utf-8'))
assert len(scope['files']) == 600
assert hashlib.sha256((root / 'input/review_input.zip').read_bytes()).hexdigest() == 'f24e5c147561fef9a37cc05842c0843207c95c859b1f076fb47d10f2033fedb7'
assert json.loads((root/'calculation/execution.json').read_text())['production_Torch_updates_executed'] == 0
report = (root/'REPLAY_IMPROVEMENT_REVIEW.zh.txt').read_text(encoding='utf-8')
validation = {
    'validation_scope':'Delivery UTF-8/syntax/inventory checks only; not scientific or native qualification.',
    'utf8_files':utf8, 'python_AST_syntax_files':syntax,
    'reading_ledger_members':600,
    'input_archive_exact_copy':True,
    'report_characters':len(report),
    'report_utf8_bytes':len(report.encode('utf-8')),
    'native_verification_executed_here':False,
}
(root/'delivery_validation.json').write_text(json.dumps(validation, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
payloads.append(root/'delivery_validation.json')
manifest = []
for path in sorted(payloads):
    value = path.read_bytes()
    manifest.append({'path':str(path.relative_to(root)), 'bytes':len(value), 'sha256':hashlib.sha256(value).hexdigest()})
(root/'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
    for path in sorted([*payloads, root/'MANIFEST.json']):
        z.write(path, str(path.relative_to(root)))
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    members = len(z.infolist())
print(json.dumps({'status':'DELIVERY_VALIDATED', 'report_utf8_bytes':validation['report_utf8_bytes'],
                  'report_characters':validation['report_characters'], 'reading_scope_members':600,
                  'archive_members':members, 'archive_bytes':archive.stat().st_size,
                  'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
                  'input_copy_sha256':hashlib.sha256((root/'input/review_input.zip').read_bytes()).hexdigest(),
                  'files':[str(root/'REPLAY_IMPROVEMENT_REVIEW.zh.txt'), str(root/'READING_SCOPE.zh.txt'), str(archive)]},
                 ensure_ascii=False, indent=2))
