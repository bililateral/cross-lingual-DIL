"""Create and verify the current review deliverable; no project execution.

This preserves the actual packing script. Paths below identify this session's
already verified files; this is not a project runner or a training command.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

WORK = Path(__file__).resolve().parent
STAGE = WORK / 'relation_revision_external_audit_bundle'
DEST = WORK / 'SELLER_ALIAS_RELATION_REVISION.audit_evidence.zip'
INPUT = WORK / 'upload/relation_revision_review.zip'
PRIOR = WORK / 'relation_memory_result_review/reports/seller_alias_continual/20261006/relation_result/source'
ROOT = WORK / 'relation_revision_review'


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def copy(source: Path, target: str) -> None:
    destination = STAGE / target
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


STAGE.mkdir(exist_ok=False)
assert INPUT.stat().st_size == 5_257_725
assert sha(INPUT.read_bytes()) == '480371343ea0be50e4f8fcc614ec1aee5b35b20593151b7fbb3932e75669a9b5'
copy(INPUT, 'input/relation_revision_review.zip')
copy(WORK / 'SELLER_ALIAS_RELATION_REVISION.external_review.zh.md',
     'SELLER_ALIAS_RELATION_REVISION.external_review.zh.md')
for name in ('README.zh.md', 'primary_sources.zh.md'):
    copy(WORK / 'revision_review_bundle_notes' / name, name)
copy(Path(__file__), 'packaging/build_revision_review_bundle.py')

evidence = {
    'revision_identity_audit': ['verify_revision_input.py', 'result.json', 'run.stdout.json', 'run.stderr.log'],
    'revision_math_audit': ['independent_revision_math.py', 'result.json', 'run.stdout.json', 'run.stderr.log'],
    'revision_code_audit': ['static_and_optimizer_probe.py', 'static_and_optimizer_result.json'],
    'revision_lifecycle_audit': ['source_and_coverage.py', 'source_and_coverage.json', 'source_and_coverage.stdout.json'],
    'revision_resource_audit': ['identity/check_identity.py', 'identity/identity_checks.json',
                               'check_supervisor_control.py', 'supervisor_control_checks.json'],
}
for folder, names in evidence.items():
    for name in names:
        copy(WORK / folder / name, f'{folder}/{name}')

prior_records = []
gate = json.loads((ROOT / 'reports/seller_alias_continual/20261006/relation_result/gate.json').read_text())
for descriptor in gate['source_files']:
    source = PRIOR / descriptor['path']
    data = source.read_bytes()
    assert len(data) == descriptor['bytes'] and sha(data) == descriptor['sha256']
    assert data == (ROOT / 'reports/seller_alias_continual/20261006/relation_result/source' / descriptor['path']).read_bytes()
    target = 'input/prior_result_frozen_source/' + descriptor['path']
    copy(source, target)
    prior_records.append({'path': target, 'bytes': len(data), 'sha256': sha(data),
                          'original_path': str(source), 'matches_current_frozen_copy': True})
assert len(prior_records) == 23
write_json(STAGE / 'input_provenance.json', {
    'scope': 'Current revision review; original bytes and prior frozen sources only',
    'current_input': {'path': 'input/relation_revision_review.zip', 'bytes': INPUT.stat().st_size,
                      'sha256': sha(INPUT.read_bytes()), 'original_path': str(INPUT)},
    'previously_reviewed_input_identity': {
        'name': 'relation_result_context_review.zip', 'bytes': 5_077_487,
        'sha256': '3c11a9640c13438d9eb7ace1005183d83ccafbdca7036fa39e0689b06731625a',
        'scope': 'Historical verified attachment identity; this packing step only copies its 23 frozen source files',
    },
    'prior_frozen_sources': prior_records,
    'execution_boundary': 'No Torch, BGE, formal labels, training, original checkpoint load or project Linux execution',
})

payloads = []
for path in sorted(STAGE.rglob('*')):
    if path.is_file():
        data = path.read_bytes()
        payloads.append({'path': path.relative_to(STAGE).as_posix(), 'bytes': len(data), 'sha256': sha(data)})
write_json(STAGE / 'manifest.json', {
    'scope': 'Independent relation revision design and implementation review deliverable',
    'manifest_excludes_itself': True,
    'payload_count': len(payloads),
    'files': payloads,
})

with ZipFile(DEST, 'x', compression=ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(p for p in STAGE.rglob('*') if p.is_file()):
        item = ZipInfo(path.relative_to(STAGE).as_posix(), date_time=(2026, 10, 6, 0, 0, 0))
        item.compress_type = ZIP_DEFLATED
        item.external_attr = 0o100644 << 16
        archive.writestr(item, path.read_bytes())

with ZipFile(DEST) as archive:
    assert archive.testzip() is None
    assert len(archive.namelist()) == len(set(archive.namelist())) == len(payloads) + 1
    saved_manifest = json.loads(archive.read('manifest.json'))
    assert saved_manifest['files'] == payloads
    assert set(archive.namelist()) == {'manifest.json', *(p['path'] for p in payloads)}
    for p in payloads:
        data = archive.read(p['path'])
        assert len(data) == p['bytes'] and sha(data) == p['sha256']
    assert archive.read('input/relation_revision_review.zip') == INPUT.read_bytes()

report = WORK / 'SELLER_ALIAS_RELATION_REVISION.external_review.zh.md'
result = {
    'status': 'PASS',
    'zip': str(DEST), 'zip_bytes': DEST.stat().st_size, 'zip_sha256': sha(DEST.read_bytes()),
    'zip_file_count': len(payloads) + 1, 'manifest_payload_count': len(payloads),
    'report': str(report), 'report_bytes': report.stat().st_size, 'report_sha256': sha(report.read_bytes()),
    'original_input_preserved': True, 'prior_frozen_files': len(prior_records),
    'all_zip_crc_and_payload_checks': True,
}
write_json(WORK / 'revision_review_bundle_integrity.json', result)
print(json.dumps(result, ensure_ascii=False, indent=2))
