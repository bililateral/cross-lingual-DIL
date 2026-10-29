"""Verify this review ZIP, nested cpu07 sources, current sources and frozen config.

Usage: python3 -B verify_package.py INPUT_ZIP OUTPUT_JSON
Standard library only. No supplied module, model or original test is executed.
"""
from __future__ import annotations
import ast
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import platform
import stat
import sys
import zipfile

EXPECTED_SHA = '0d2124d490d4048f635f6b15f2538a883a20d0c2995084364849b3d1bc9280ad'
sys.dont_write_bytecode = True


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    input_zip, output = map(Path, sys.argv[1:])
    raw = input_zip.read_bytes()
    assert len(raw) == 136445 and sha(raw) == EXPECTED_SHA
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) == 35
        for member in z.infolist():
            p = PurePosixPath(member.filename)
            assert not p.is_absolute() and '..' not in p.parts and not member.is_dir()
            assert not stat.S_ISLNK(member.external_attr >> 16)
        contents = {name: z.read(name) for name in names}
    manifest = json.loads(contents['MANIFEST.json'])
    assert len(manifest) == len({r['path'] for r in manifest}) == 34
    assert set(contents) == {r['path'] for r in manifest} | {'MANIFEST.json'}
    verified = []
    for row in manifest:
        blob = contents[row['path']]
        assert len(blob) == row['bytes'] and sha(blob) == row['sha256']
        verified.append({**row, 'actual_bytes': len(blob), 'actual_sha256': sha(blob), 'match': True})
    j = lambda name: json.loads(contents[name])
    current_sources = j('native/cpu_sources.json')
    cpu07, cpu08 = j('native/cpu07/receipt.json'), j('native/cpu08/receipt.json')
    assert len(current_sources) == 9 and current_sources == cpu08['sources']
    for row in current_sources:
        blob = contents[row['path']]
        assert len(blob) == row['bytes'] and sha(blob) == row['sha256']
        ast.parse(blob)
    nested_raw = contents['native/cpu07_sources.zip']
    assert len(nested_raw) == 39032
    assert sha(nested_raw) == 'ebbf4f403c6dc8c965bd083e68ef35e9fe9249a7046e5fa3be4e818d85a1dc2f'
    nested = []
    with zipfile.ZipFile(io.BytesIO(nested_raw)) as z:
        old_sources = json.loads(z.read('cpu_sources.json'))
        assert old_sources == cpu07['sources'] and len(old_sources) == 9
        assert len(z.namelist()) == len(set(z.namelist())) == 10
        assert set(z.namelist()) == {r['path'] for r in old_sources} | {'cpu_sources.json'}
        for name in z.namelist():
            blob = z.read(name)
            nested.append(dict(path=name, bytes=len(blob), sha256=sha(blob)))
        changed = []
        for row in old_sources:
            blob = z.read(row['path'])
            assert len(blob) == row['bytes'] and sha(blob) == row['sha256']
            if blob != contents[row['path']]: changed.append(row['path'])
    assert set(changed) == {'scripts/step29_vendorlink_export.py', 'tests/test_step29_vendorlink.py'}
    unchanged_core = []
    for row in j('three_market/cpu06/receipt.json')['sources']:
        if row['path'].startswith('scripts/'):
            blob = contents[row['path']]
            assert len(blob) == row['bytes'] and sha(blob) == row['sha256']
            unchanged_core.append(row)
    assert len(unchanged_core) == 5
    transfers = j('native/transfer.json')
    assert len(transfers) == 8
    for row in transfers:
        blob = contents['native/' + row['path']]
        assert len(blob) == row['bytes'] and sha(blob) == row['sha256']
    config, permit = j('native/config.json'), j('native/permit.json')
    assert permit['config_sha256'] == sha(contents['native/config.json'])
    assert permit['approved'] is True and permit['implementation_review_closed'] is False
    inv, source_cfg = j('three_market/inventory.json'), j('three_market/source_config.json')
    assert config['canonical'] == inv['canonical_cache']
    assert config['partition'] == source_cfg['proposed_partition']
    assert config['expected_partitions'] == inv['partitions']
    for row in config['sources']:
        blob = contents['scripts/' + Path(row['path']).name]
        assert len(blob) == row['bytes'] and sha(blob) == row['sha256']
    ready = j('data_qualification/model_ready.json')
    declared = {row['name']: (row['bytes'], row['sha256']) for row in ready['files']}
    assert {Path(row['path']).name: (row['bytes'], row['sha256']) for row in config['model_files']} == declared
    assert {str(Path(row['path']).parent) for row in config['model_files']} == {config['model_path']}
    job = Path(permit['job'])
    assert Path(config['export']).parent == job and Path(config['custody']).parent == job
    assert Path(config['export_manifest']) == Path(config['export']) / 'manifest.json'
    assert sha(contents['three_market/review/external/report.zh.txt']) == '72d0b9b0ee0e537e32e353ba379aea7caf618f530622031aa989c5a42bfb83c2'
    assert sha(contents['three_market/review/external/disposition.json']) == 'b79e5b9544beac10dd5b3eb1090d93ef64740a5847f75676186073b901bcd959'
    assert math.ceil(cpu07['wall_seconds']) == cpu07['charged_seconds'] == 15
    assert math.ceil(cpu08['wall_seconds']) == cpu08['charged_seconds'] == 5
    assert 30 + 15 == cpu07['cumulative_seconds'] == 45
    assert 45 + 5 == cpu08['cumulative_seconds'] == 50
    g1 = j('native/cpu07/native_g1.json')
    counts = {name: value['updates'] for name, value in g1.items()}
    assert counts == {'VendorLink_SEQ': 4, 'H_SEQ': 3, 'H_ER': 3, 'H_full': 5}
    assert sum(counts.values()) == 15
    assert all(row['classes'] == [35, 35, 35] for row in g1.values())
    for name in ('H_ER', 'H_full'):
        assert [row['groups'] for row in g1[name]['retention']] == [3, 6]
        assert all(row['bytes'] <= 1048576 for row in g1[name]['retention'])
    limits = config['limits']
    assert limits['g1_seconds'] + limits['g2_cumulative_seconds'] + limits['auxiliary_seconds'] == limits['total_seconds'] == 2100
    assert limits['g1_updates'] == 15 and limits['g2_updates_maximum'] == 9 and limits['g2_implemented_updates'] == 6
    resource = j('native/resource.json')
    assert resource['free'] / 2**20 == 15190
    assert resource['free'] < limits['initial_free_bytes'] == 17 * 2**30
    evidence = {
        'scope': 'Package-only identity, source bridge, AST syntax and arithmetic; no project execution',
        'environment': {'python': sys.version, 'platform': platform.platform()},
        'zip': {'name': input_zip.name, 'bytes': len(raw), 'sha256': sha(raw), 'members': 35, 'payloads': 34},
        'manifest': {'bytes': len(contents['MANIFEST.json']), 'sha256': sha(contents['MANIFEST.json']), 'all_payloads_match': True},
        'payload_verification': verified,
        'cpu07_nested': {'bytes': len(nested_raw), 'sha256': sha(nested_raw), 'members': nested, 'all_sources_match_receipt': True},
        'cpu08_sources': current_sources, 'cpu08_all_sources_match': True,
        'changed_cpu07_to_cpu08': changed, 'unchanged_core_from_cpu06': unchanged_core,
        'nine_python_files_ast_parse': True, 'eight_transfer_records_match': True,
        'config_canonical_partition_inventory_match': True, 'model_metadata_consistent_not_model_bytes_rehashed': True,
        'permit_config_sha256_matches': True, 'permit_approved': True, 'implementation_review_closed': False,
        'configured_export_and_custody_inside_job': True,
        'previous_report_identity_matches': True,
        'cpu_ledger': {'prior': 30, 'cpu07_wall': cpu07['wall_seconds'], 'cpu07_charge': 15,
                       'cpu07_exit_code': cpu07['exit_code'], 'cpu08_wall': cpu08['wall_seconds'],
                       'cpu08_charge': 5, 'cpu08_exit_code': cpu08['exit_code'], 'total': 50},
        'g1_provided_native_update_counts': counts,
        'g2_implemented_updates': 6, 'g1_plus_g2_implemented_updates': 21, 'maximum_allowed_updates': 24,
        'provided_resource': {'observed_server': resource['observed_server'], 'free_mib': 15190,
                              'required_g1_free_mib': 17408, 'below_g1_proposed_gate': True},
        'original_data_models_qrels_read': False, 'original_tests_training_or_gpu_executed': False,
        'does_not_resolve_implementation_findings': ['NI01', 'NI02', 'NI03'],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'status': 'identity_and_arithmetic_checks_passed', 'payloads': 34, 'output': str(output)}))


if __name__ == '__main__':
    main()
