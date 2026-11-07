"""One-off Linux CPU evidence for the five custodian review repairs.

All data are handwritten. No official archive, model, real text or labels open.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
from unittest.mock import patch
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import step30_evolution_control as control
import step30_evolution_data as data
import step30_evolution_io as eio
from step30_evolution_policy import DOMAINS, LIMITS


def rejects(fn, error, contains=''):
    try:
        fn()
    except error as exc:
        assert contains in str(exc), str(exc)
        return
    raise AssertionError('Expected rejection: '+contains)


def atomic_count(root):
    temporary, final = root/'payload.tmp', root/'payload.bin'
    temporary.write_bytes(b'atomic payload')
    original = Path.stat
    moved = False

    class Scan:
        def rglob(self, pattern):
            return iter((temporary, final))

    def racing_stat(path, *args, **kwargs):
        nonlocal moved
        if path == temporary and not moved:
            moved = True
            temporary.replace(final)
        return original(path, *args, **kwargs)

    with patch.object(Path, 'stat', racing_stat):
        assert eio.directory_bytes(Scan()) == len(b'atomic payload')
    assert moved
    with patch.object(Path, 'stat', side_effect=PermissionError('keep permission failure')):
        rejects(lambda: eio.directory_bytes(Scan()), PermissionError, 'permission')


def large_tsv(root, qualification):
    archive = root/'long_field.zip'
    long_body = 'The text is for a handwritten test with '+('x'*140000)
    tables = {
        'categories': [['cid', 'category', 'parent_cid'], ['1', 'Example', '']],
        'scrapes': [['mscrape_id', 'scrape_year', 'scrape_month', 'scrape_day'], ['1', '2014', '7', '1']],
        'vendors': [['vid', 'username'], ['1', 'fixture_user']],
        'listings': [['vid', 'lid', 'mscrape_id', 'title', 'description', 'cid'],
                     ['1', '1', '1', 'Handwritten title', long_body, '1']]}
    with zipfile.ZipFile(archive, 'w') as z:
        for name, rows in tables.items():
            stream = io.StringIO(newline='')
            csv.writer(stream, delimiter='\t').writerows(rows)
            z.writestr('market/'+name+'.tsv', stream.getvalue())
    csv.field_size_limit(131072)
    # Reach the intended real-count mismatch, not csv.Error at 128 KiB.
    rejects(lambda: data.clean_archive(archive, qualification, {}, lambda: None),
            ValueError, 'Qualification base counts changed')
    assert csv.field_size_limit() == 16*1024**2


def build_and_language(root, qualification):
    workspace = root/'workspace'
    destination = workspace/'qualification'
    destination.mkdir(parents=True)
    for name in ('audit.py', 'followup.py'):
        shutil.copy2(qualification/name, destination/name)
    archive, summary = root/'placeholder.zip', root/'aggregate.json'
    archive.write_bytes(b'handwritten scanner is substituted below')
    eio.write(summary, {'analysis': {}})
    allowed = dict(workspace=str(workspace), qualification_directory=str(destination),
                   qualification_sources=[eio.record(destination/name) for name in ('audit.py', 'followup.py')],
                   archive=eio.record(archive), qualification_summary=eio.record(summary),
                   build_output=str(root/'dataset'), experiment_root=str(root),
                   prior_seconds=dict(training=0, gpu=0, auxiliary=622), cpu_test_seconds=22)
    assert eio.qualification_directory(allowed) == destination.resolve()
    rejects(lambda: eio.qualification_directory(dict(allowed, qualification_sources=[])),
            ValueError, 'exact audit/followup')
    wrong = dict(allowed, qualification_directory=str(root/'old_qualification'), build_output=str(root/'wrong_source'))
    with patch.object(data, 'clean_archive', side_effect=AssertionError('must not enter scanner')):
        rejects(lambda: control.build(wrong), ValueError, 'exact audit/followup')
    assert not (root/'wrong_source_accounting/access/official_four_table_build.json').exists()
    assert eio.read(root/'wrong_source_accounting/budget.json')['events'][-1]['success'] is False

    records = [data.Record(f'{d}_v{v}', f'{d}_{v}_{r}', '1', day, r+1, 'fixture',
                           'Handwritten title', f'The text is for a fixture with record {d} {v} {r}.', f'{d}-{v}-{r}')
               for d, day in zip(DOMAINS, ('2014-07-01', '2014-10-01', '2015-01-01'))
               for v in range(320) for r in range(4)]
    checks = []
    original_budget = control.Budget

    class TracedBudget(original_budget):
        def check(self, force=False):
            checks.append(force)
            return super().check(force=force)

    with patch.object(data, 'clean_archive', return_value=(records, {'handwritten': True})), patch.object(control, 'Budget', TracedBudget):
        control.build(allowed)
    assert len(checks) > 1900 and checks[-1] is True
    manifest_path = root/'dataset/manifest.json'
    manifest = eio.read(manifest_path)
    groups = [g for key, row in manifest['files'].items() if not key.endswith(':qrels') for g in eio.pinned(row)]
    assert len(groups) == 120
    assert sum(len(v['records']) for g in groups for v in g['views']) == 3840
    assert manifest['language_samples'] == eio.record(root/'dataset/private_custody/language_samples.json')
    budget = eio.read(root/'dataset_accounting/budget.json')
    assert budget['events'][-1]['success'] is True and budget['used_seconds']['auxiliary'] >= 652

    clean_done = False

    class StopAfterCleaning(original_budget):
        def check(self, force=False):
            if clean_done:
                raise TimeoutError('post-clean resource stop')
            return super().check(force=force)

    def fixture_scanner(*args):
        nonlocal clean_done
        clean_done = True
        return records, {'handwritten': True}

    stopped = dict(allowed, build_output=str(root/'resource_stop'))
    with patch.object(data, 'clean_archive', side_effect=fixture_scanner), patch.object(control, 'Budget', StopAfterCleaning):
        rejects(lambda: control.build(stopped), TimeoutError, 'post-clean resource stop')
    assert not (root/'resource_stop/manifest.json').exists()
    assert eio.read(root/'resource_stop_accounting/budget.json')['events'][-1]['success'] is False

    probe = original_budget(root, time.monotonic()+60)
    probe.last = time.monotonic()
    with patch.object(eio, 'directory_bytes', return_value=LIMITS['disk_bytes']):
        rejects(lambda: probe.check(force=True), RuntimeError, 'disk cap')

    samples = eio.pinned(manifest['language_samples'])
    decisions_path = root/'decisions.json'
    eio.write(decisions_path, {d: {r['id']: 'english' for r in samples[d]} for d in DOMAINS})
    language = dict(allowed, dataset_manifest=eio.record(manifest_path), language_samples=manifest['language_samples'],
                    language_decisions=eio.record(decisions_path), language_output=str(root/'language_ok'))
    control.adjudicate_language(language)
    assert eio.read(root/'language_ok/language_receipt.json')['clearly_english'] == {d: 32 for d in DOMAINS}
    dataset_allowed = dict(language, language_receipt=eio.record(root/'language_ok/language_receipt.json'))
    assert eio.dataset(dataset_allowed) == manifest
    altered = root/'wrong_language_sample.json'
    samples[DOMAINS[0]][0]['body'] = 'Changed body with the same old ID.'
    eio.write(altered, samples)
    wrong_language = dict(language, language_samples=eio.record(altered), language_output=str(root/'language_rejected'))
    rejects(lambda: control.adjudicate_language(wrong_language), ValueError, 'exact dataset manifest')
    assert not (root/'language_rejected/access/manual_fit_language_judgments.json').exists()
    assert eio.read(root/'language_rejected/budget.json')['events'][-1]['success'] is False
    wrong_receipt = eio.read(root/'language_ok/language_receipt.json')
    wrong_receipt['language_samples'] = eio.record(altered)
    wrong_receipt_path = root/'wrong_language_receipt.json'
    eio.write(wrong_receipt_path, wrong_receipt)
    rejects(lambda: eio.dataset(dict(dataset_allowed, language_receipt=eio.record(wrong_receipt_path))),
            ValueError, 'Same frozen dataset')
    return dict(groups=len(groups), records=3840, resource_checks=len(checks), final_check_forced=checks[-1],
                manual_fixture_counts={d: 32 for d in DOMAINS})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if sys.platform != 'linux' or Path(sys.executable).resolve() != Path(eio.PYTHON).resolve() or os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Existing Linux py310 with CUDA hidden required')
    if args.output.exists():
        raise FileExistsError('Fresh one-off CPU output required')
    args.output.mkdir(parents=True)
    start, result = time.monotonic(), dict(status='RUNNING', checks=[])
    try:
        sources = eio.read(args.sources)
        for row in sources:
            eio.verify(row)
        assert Path(__file__).resolve() in {Path(row['path']).resolve() for row in sources}
        with tempfile.TemporaryDirectory(prefix='custodian-fixture-', dir=args.output) as tmp:
            root = Path(tmp)
            atomic_count(root)
            result['checks'].append('R01_atomic_rename_and_permission_failure')
            large_tsv(root, args.qualification)
            result['checks'].append('R02_actual_clean_entry_long_TSV')
            result['integration'] = build_and_language(root, args.qualification)
            result['checks'] += ['R03_actual_qualification_path_rejected_before_archive',
                                 'R04_manifest_language_body_binding_and_success',
                                 'R05_actual_build_partition_export_forced_final_and_failure_ledger']
        result['status'] = 'PASS'
    except BaseException:
        result.update(status='FAIL', traceback=traceback.format_exc())
        raise
    finally:
        result['test_function_seconds'] = time.monotonic()-start
        result['scope'] = 'Handwritten sources only; large TSV parsed by real cleaner; full quota export uses substituted scanner, not official evidence.'
        eio.write(args.output/'receipt.json', result)
        print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
