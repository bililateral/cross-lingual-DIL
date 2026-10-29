"""Package-only source bridge and handwritten export checks; no model/test-suite execution.

Usage: python3 -B export_bridge_checks.py INPUT_ROOT OUTPUT_JSON
Only stdlib data module and selected AST functions are executed. No run/native import.
"""
from __future__ import annotations
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import platform
import stat
import sys
import tempfile
import types
import zipfile

sys.dont_write_bytecode = True


def sha(data):
    return hashlib.sha256(data).hexdigest()


def selected_functions(source, names, environment):
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in nodes} == set(names)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<reviewed_function_ast>', 'exec'), environment)


def symbols(source):
    result = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            result[node.name] = ast.dump(node, include_attributes=False)
            if isinstance(node, ast.ClassDef):
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        result[node.name + '.' + sub.name] = ast.dump(sub, include_attributes=False)
    return result


def main():
    root, output = map(Path, sys.argv[1:])
    receipt7 = json.loads((root/'native/cpu07/receipt.json').read_text())
    receipt8 = json.loads((root/'native/cpu08/receipt.json').read_text())
    current_manifest = json.loads((root/'native/cpu_sources.json').read_text())
    archive_bytes = (root/'native/cpu07_sources.zip').read_bytes()
    with zipfile.ZipFile(root/'native/cpu07_sources.zip') as archive:
        assert len(archive.infolist()) == 10
        assert len(set(archive.namelist())) == 10
        old_manifest = json.loads(archive.read('cpu_sources.json'))
        assert old_manifest == receipt7['sources']
        assert set(archive.namelist()) == {r['path'] for r in old_manifest} | {'cpu_sources.json'}
        nested_entries = []
        for name in archive.namelist():
            data = archive.read(name)
            nested_entries.append(dict(path=name, bytes=len(data), sha256=sha(data)))
        old_sources = {r['path']: archive.read(r['path']) for r in old_manifest}
    for r in old_manifest:
        data = old_sources[r['path']]
        assert len(data) == r['bytes'] and sha(data) == r['sha256']
    assert current_manifest == receipt8['sources'] and len(current_manifest) == 9
    for r in current_manifest:
        data = (root/r['path']).read_bytes()
        assert len(data) == r['bytes'] and sha(data) == r['sha256']
    changed = [p for p, data in old_sources.items() if data != (root/p).read_bytes()]
    assert set(changed) == {'scripts/step29_vendorlink_export.py', 'tests/test_step29_vendorlink.py'}
    old_symbols = symbols(old_sources['tests/test_step29_vendorlink.py'])
    new_symbols = symbols((root/'tests/test_step29_vendorlink.py').read_bytes())
    stable = [k for k in old_symbols if k != 'Qualification']
    assert all(old_symbols[k] == new_symbols[k] for k in stable)
    extra_tests = sorted(set(new_symbols)-set(old_symbols))
    assert extra_tests == ['Qualification.test_native_supervisor_charges_and_stops_real_children']
    for r in json.loads((root/'native/transfer.json').read_text()):
        data = (root/'native'/r['path']).read_bytes()
        assert len(data) == r['bytes'] and sha(data) == r['sha256']
    config_bytes = (root/'native/config.json').read_bytes()
    config = json.loads(config_bytes)
    permit = json.loads((root/'native/permit.json').read_text())
    inventory = json.loads((root/'three_market/inventory.json').read_text())
    partition = json.loads((root/'three_market/source_config.json').read_text())['proposed_partition']
    assert config['canonical'] == inventory['canonical_cache']
    assert config['partition'] == partition
    assert config['expected_partitions'] == inventory['partitions']
    assert permit['config_sha256'] == sha(config_bytes)
    assert permit['approved'] is True and permit['implementation_review_closed'] is False
    job = Path(permit['job'])
    assert Path(config['export']).parent == job and Path(config['custody']).parent == job
    assert Path(config['export_manifest']).parent == Path(config['export'])
    for r in config['sources']:
        local = root/'scripts'/Path(r['path']).name
        data = local.read_bytes()
        assert len(data) == r['bytes'] and sha(data) == r['sha256']
    module = types.ModuleType('review_stdlib_vendorlink_data')
    sys.modules[module.__name__] = module
    data_source = (root/'scripts/step29_vendorlink_data.py').read_bytes()
    exec(compile(data_source, str(root/'scripts/step29_vendorlink_data.py'), 'exec'), module.__dict__)
    environment = dict(Counter=Counter, json=json, Path=Path, hashlib=hashlib,
                       Group=module.Group, build_views=module.build_views)
    selected_functions((root/'scripts/step29_vendorlink_run.py').read_bytes(), ['write_json', 'file_sha256'], environment)
    selected_functions((root/'scripts/step29_vendorlink_export.py').read_bytes(), ['export'], environment)
    selected_functions((root/'scripts/step29_vendorlink_native.py').read_bytes(), ['stage_loader'], environment)
    old_environment = dict(environment)
    selected_functions(old_sources['scripts/step29_vendorlink_export.py'], ['export'], old_environment)
    rows = [dict(market=market, vendor=f'hand_seller_{i}', title=f'item {market} {i} {j}',
                 description=f'description {market} {i} {j}')
            for market in ('AlphaBay', 'Dream', 'Agora') for i in range(80) for j in range(20)]
    expected = module.preview_counts(rows, **partition)['partitions']
    actual_payloads, _, _ = module.build_views(rows, **partition)
    checked_samples = {}
    with tempfile.TemporaryDirectory(prefix='vendorlink_export_review_') as tmp:
        base = Path(tmp)
        try:
            old_environment['export'](rows, partition, base/'old_export', base/'old_custody', expected)
        except ValueError as error:
            assert str(error).startswith('Export counts differ:')
        else:
            raise AssertionError('Old integer-key reference should reproduce cpu07 failure')
        assert not (base/'old_export').exists() and not (base/'old_custody').exists()
        result = environment['export'](rows, partition, base/'export', base/'custody', expected)
        json_result = environment['export'](rows, partition, base/'json_export', base/'json_custody',
                                            json.loads(json.dumps(expected)))
        assert result['counts'] == json_result['counts']
        assert len(result['counts']) == 9
        assert stat.S_IMODE((base/'custody').stat().st_mode) == 0o700
        for key, values in actual_payloads.items():
            domain, split = key.split(':')
            emitted = json.loads((base/'export'/f'{domain}_{split}.json').read_text())
            # JSON serialization intentionally converts dataclass tuple fields to arrays.
            assert emitted == json.loads(json.dumps(values))
            if split == 'fit':
                source = result['qualification_samples'][domain]
                content = Path(source['path']).read_bytes()
                assert len(content) == source['bytes'] and sha(content) == source['sha256']
                sample = json.loads(content)
                expected_groups = [g for g in values if len(g['views']) == 16][:3]
                expected_classes = sorted({v['identity'] for g in values for v in g['views']})
                assert sample['groups'] == json.loads(json.dumps(expected_groups)) and sample['classes'] == expected_classes
                assert len(sample['classes']) > len({v['identity'] for g in sample['groups'] for v in g['views']}) == 24
                checked_samples[domain] = dict(full_classes=len(sample['classes']), small_group_classes=24,
                                               chosen_groups=[g['uid'] for g in sample['groups']])
            else:
                private = base/'custody'/f'{domain}_{split}_qrels.json'
                assert stat.S_IMODE(private.stat().st_mode) == 0o600
                assert not (base/'export'/private.name).exists()
                qrels = json.loads(private.read_text())
                for g in emitted:
                    views = g['views']
                    assert all(v['identity'] is None for v in views)
                    assert qrels[g['uid']]['view_ids'] == [v['uid'] for v in views]
                    # Private handwritten fixture owner is derived independently of exported IDs/order.
                    owner = lambda v: v['records'][0]['title'].split()[2]
                    labels = [int(owner(a) == owner(b)) for i,a in enumerate(views) for b in views[i+1:]]
                    assert qrels[g['uid']]['labels'] == labels
        for item in result['files']:
            content = Path(item['path']).read_bytes()
            assert len(content) == item['bytes'] and sha(content) == item['sha256']
        loader = environment['stage_loader'](result['qualification_samples'])
        try:
            loader('Dream', base/'access.jsonl')
        except ValueError as error:
            assert 'Future/repeated' in str(error)
        else:
            raise AssertionError('Future access must fail')
        for domain in ('AlphaBay', 'Dream', 'Agora'):
            groups, classes = loader(domain, base/'access.jsonl')
            assert len(groups) == 3 and len(classes) == checked_samples[domain]['full_classes']
        bad_reference = json.loads(json.dumps(expected))
        bad_reference['Agora:fit']['accounts'] += 1
        try:
            environment['export'](rows, partition, base/'bad_export', base/'bad_custody', bad_reference)
        except ValueError as error:
            assert 'counts differ' in str(error)
        else:
            raise AssertionError('Mismatched fixed preview must fail')
        assert not (base/'bad_export').exists() and not (base/'bad_custody').exists()
    assert 'torch' not in sys.modules and 'transformers' not in sys.modules
    assert math.ceil(receipt7['wall_seconds']) == receipt7['charged_seconds'] == 15
    assert math.ceil(receipt8['wall_seconds']) == receipt8['charged_seconds'] == 5
    assert 30 + receipt7['charged_seconds'] == receipt7['cumulative_seconds'] == 45
    assert 45 + receipt8['charged_seconds'] == receipt8['cumulative_seconds'] == 50
    output.parent.mkdir(parents=True, exist_ok=True)
    evidence = dict(
        environment=dict(python=sys.version, platform=platform.platform()),
        scope='Web isolated stdlib/AST and handwritten export only; not project Linux qualification',
        no_torch_transformers_or_training=True, no_formal_data_or_models_read=True,
        no_project_server_or_network_access=True,
        nested_cpu07=dict(bytes=len(archive_bytes), sha256=sha(archive_bytes), members=nested_entries,
                         all_nine_sources_match_receipt=True),
        cpu08_all_nine_sources_and_manifest_match=True, transfer_all_eight_match=True,
        changed_cpu07_to_current=changed, unchanged_previous_test_symbols=stable, only_added_test=extra_tests,
        config_canonical_partition_and_nine_preview_partitions_match=True,
        config_source_hashes_and_permit_binding_match=True,
        configured_export_and_custody_within_job=True, permit_approved=True, implementation_review_closed=False,
        handwritten_checks=dict(rows=len(rows), partitions=9, cpu07_integer_key_failure_reproduced=True,
             current_accepts_integer_and_json_string_histogram_keys=True,
             payloads_equal_builder=True, samples=checked_samples,
             all_dev_and_heldout_blind_qrels_match_private_handwritten_truth=True,
             custody_mode_0700_qrels_mode_0600=True, future_stage_rejected=True,
             fixed_preview_mismatch_fails_before_writing=True),
        cpu_ledger=dict(prior=30,cpu07_wall=receipt7['wall_seconds'],cpu07_charge=15,
                        cpu08_wall=receipt8['wall_seconds'],cpu08_charge=5,total=50),
        reviewer_harness_note='Initial reviewer comparison was corrected to normalize tuple-to-JSON-array representation; no project source was changed.',
        limitations=['No original test suite or model code executed.',
                     'Export/run/native main functions and supervisor were not executed here.',
                     'CPU07/08 are provided native evidence, not reviewer-native reproductions.'])
    output.write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='scoped assertions passed',output=str(output))))


if __name__ == '__main__':
    main()
