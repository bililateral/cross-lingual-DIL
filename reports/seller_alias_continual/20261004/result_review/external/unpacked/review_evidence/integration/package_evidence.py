"""Package the unchanged uploaded ZIP plus actual external-review evidence."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', required=True, type=Path)
    args = parser.parse_args()
    root = args.workspace.resolve()
    evidence, output = root / 'review_evidence', root / 'deliverables'
    original = root / 'upload/review(5).zip'
    expected = '0d5e45250361558d6b2d66bca8815103e6f0657eeab9fb9d63a4bbbec50c5053'
    assert original.stat().st_size == 9639622
    assert hashlib.sha256(original.read_bytes()).hexdigest() == expected
    members = [('original/review.zip', original)]
    members += [(name, output / name) for name in
                ('external_review.zh.txt', 'external_verdict.json', 'REPRODUCE.zh.txt')]
    members += [('review_evidence/' + p.relative_to(evidence).as_posix(), p)
                for p in sorted(evidence.rglob('*')) if p.is_file()]
    assert len({name for name, _ in members}) == len(members)
    manifest = {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
                'scope': 'Original authorized attachment, independent saved-result reference code, inputs, outputs and review deliverables',
                'original_attachment_sha256': expected,
                'self_in_manifest': False, 'files': []}
    zip_path = output / 'external_review_evidence.zip'
    with zipfile.ZipFile(zip_path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in members:
            payload = path.read_bytes()
            manifest['files'].append({'path': name, 'bytes': len(payload),
                                      'sha256': hashlib.sha256(payload).hexdigest()})
            archive.writestr(name, payload, compress_type=zipfile.ZIP_STORED if name == 'original/review.zip'
                             else zipfile.ZIP_DEFLATED)
        archive.writestr('MANIFEST.json', json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    with zipfile.ZipFile(zip_path) as archive:
        assert archive.testzip() is None
        assert len(archive.infolist()) == len(members) + 1
        assert len(set(archive.namelist())) == len(archive.namelist())
        saved_manifest = json.loads(archive.read('MANIFEST.json'))
        for rec in saved_manifest['files']:
            payload = archive.read(rec['path'])
            assert len(payload) == rec['bytes']
            assert hashlib.sha256(payload).hexdigest() == rec['sha256']
        assert hashlib.sha256(archive.read('original/review.zip')).hexdigest() == expected
        member_count = len(archive.infolist())
    identity = {'path': str(zip_path), 'bytes': zip_path.stat().st_size,
                'members': member_count, 'payloads': len(members),
                'sha256': hashlib.sha256(zip_path.read_bytes()).hexdigest(),
                'crc_and_manifest_verified': True, 'original_attachment_preserved_exactly': True}
    (output / 'delivery_identity.json').write_text(json.dumps(identity, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(identity, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
