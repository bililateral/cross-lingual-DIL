"""One authorized CPU qualification; no cohort selection, labels or model imports."""
from __future__ import annotations

import os
import time

START = time.monotonic()
os.environ.update(OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', MKL_NUM_THREADS='2',
                  CUDA_VISIBLE_DEVICES='')

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import date, datetime
import hashlib
import itertools
import json
from pathlib import Path
import resource
import signal
import sys
import traceback

import psutil


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def roles(rows):
    ordered = sorted(rows, key=lambda r: (r.observed, sha(('evolution-registered-v1:'+r.vid+':'+r.lid).encode())))
    if len(ordered) < 8:
        return None, 'fewer_than_eight'
    fit = ordered[:4]
    gallery = [r for r in ordered if r.observed > fit[-1].observed][:2]
    if len(gallery) != 2:
        return None, 'no_two_later_gallery'
    query = [r for r in ordered if r.observed > gallery[-1].observed][:2]
    if len(query) != 2:
        return None, 'no_two_later_query'
    chosen = fit+gallery+query
    if len({r.lid for r in chosen}) != 8 or len({r.body_key for r in chosen}) != 8:
        return None, 'role_record_or_body_overlap'
    return dict(fit=fit, gallery=gallery, query=query), None


def category(vid, previous):
    return previous.get(vid, 'unused')


def role_near_copy(parts, follow):
    rows = parts['fit']+parts['gallery']+parts['query']
    grams = [follow.shingles(follow.body_signature(r.body.casefold())) for r in rows]
    # Always evaluate all 28 declared pairs, including when an early match exists.
    similarities = [follow.jaccard(a, b) for a, b in itertools.combinations(grams, 2)]
    return any(value >= .8 for value in similarities), len(similarities)


def handwritten(record_type, follow):
    def rows(days):
        return [record_type('fixture', str(i), 'fixture', day, i, 'fixture', 'title',
                            ' '.join([chr(97+i)*j for j in range(1, 15)]), str(i))
                for i, day in enumerate(days)]
    good = rows(['2020-01-01']*4+['2020-01-02']*2+['2020-01-03']*2)
    parts, reason = roles(good)
    assert reason is None and [len(parts[k]) for k in ('fit', 'gallery', 'query')] == [4, 2, 2]
    assert not role_near_copy(parts, follow)[0]
    assert roles(rows(['2020-01-01']*8))[1] == 'no_two_later_gallery'
    assert roles(rows(['2020-01-01']*6+['2020-01-02']*2))[1] == 'no_two_later_query'
    copied = dict(parts)
    copied['query'] = list(parts['query'])
    copied['query'][0] = record_type(**{**asdict(copied['query'][0]), 'body': parts['fit'][0].body})
    assert role_near_copy(copied, follow) == (True, 28)
    duplicate = list(good)
    duplicate[-1] = record_type(**{**asdict(duplicate[-1]), 'body_key': good[0].body_key})
    assert roles(duplicate)[1] == 'role_record_or_body_overlap'
    previous = {'a': 'fit', 'b': 'dev', 'c': 'test'}
    assert [v for v in ('a', 'b', 'c', 'd') if category(v, previous) in ('test', 'unused')] == ['c', 'd']
    assert [v for v in ('a', 'b', 'c', 'd') if category(v, previous) in ('fit', 'dev', 'unused')] == ['a', 'b', 'd']
    return ['strict_4_2_2_dates', 'same_day_not_later', 'no_role_replacement',
            'all_28_near_copy_pairs', 'body_overlap_rejected', 'old_fit_dev_not_confirmation']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.project.resolve(), args.output.resolve()
    expected_output = root/'reports/documentation/20261108/evolution_registered/qualification'
    if output != expected_output or not output.is_dir() or (output/'started.json').exists():
        raise ValueError('Exact fresh qualification destination required')
    os.umask(0o077)
    process = psutil.Process()
    affinity = process.cpu_affinity()[:2]
    if len(affinity) != 2 or psutil.virtual_memory().available < 8*2**30:
        raise ValueError('CPU or memory admission unavailable')
    process.cpu_affinity(affinity)
    private = output/'private_custody'
    private.mkdir(mode=0o700)
    sources = output/'sources'
    bound = json.loads((output/'source_manifest.json').read_text(encoding='utf-8'))
    receipts, accesses = {}, []
    phase, peak_rss, last_disk = 'source_binding', 0, 0.

    def check(*unused):
        nonlocal peak_rss, last_disk
        now = time.monotonic()
        if now-START > 1620:
            raise TimeoutError('Qualification time cap')
        peak_rss = max(peak_rss, process.memory_info().rss)
        if peak_rss > 8*2**30:
            raise MemoryError('Qualification RSS cap')
        if now-last_disk > 2:
            size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if size > 4*2**30:
                raise OSError('Qualification directory cap')
            last_disk = now

    def access(name):
        accesses.append(dict(object=name, purpose=phase, at=datetime.now().astimezone().isoformat()))
        write(output/'access.json', accesses)

    def receipt(path):
        digest = hashlib.sha256()
        size = 0
        with path.open('rb') as stream:
            while chunk := stream.read(2**20):
                digest.update(chunk)
                size += len(chunk)
                check()
        return dict(path=str(path), bytes=size, sha256=digest.hexdigest())

    write(output/'started.json', dict(at=datetime.now().astimezone().isoformat(), pid=os.getpid(),
          owner=process.username(), process_create_time=process.create_time(), command=process.cmdline(),
          cwd=str(Path.cwd()), affinity=affinity, preparation_reserve_seconds=180,
          total_cap_seconds=1800, execution_cap_seconds=1620, training_allowed=False))
    signal.signal(signal.SIGALRM, check)
    signal.setitimer(signal.ITIMER_REAL, .5, .5)
    success = False
    try:
        for name, wanted in bound['sources'].items():
            path = sources/name
            got = receipt(path)
            if (got['bytes'], got['sha256']) != (wanted['bytes'], wanted['sha256']):
                raise ValueError('Bound source mismatch')
            receipts[name] = got
        if receipt(Path(__file__))['sha256'] != bound['entry_sha256']:
            raise ValueError('Qualification entry mismatch')
        sys.path.insert(0, str(sources))
        from step30_evolution_data import Record, clean_archive, qualification_modules
        _, follow = qualification_modules(sources, check)
        phase = 'handwritten_role_checks'
        tests = handwritten(Record, follow)
        write(output/'handwritten.json', dict(status='PASS', checks=tests, real_data=False, gpu=False))

        old_data = root/'reports/documentation/20261107/evolution_pilot/data_preparation/dataset'
        manifest_path = old_data/'manifest.json'
        receipts['old_manifest'] = receipt(manifest_path)
        if receipts['old_manifest']['sha256'] != bound['old_manifest_sha256']:
            raise ValueError('Old manifest identity mismatch')
        expected = json.loads(manifest_path.read_text(encoding='utf-8'))['cleaning']['cleaning']
        phase = 'old_mapping_usage_mask_only'
        mapping_path = old_data/'private_custody/source_mapping.json'
        access('old source_mapping.json: vid/split usage mask; no qrels')
        raw = mapping_path.read_bytes()
        receipts['old_mapping'] = dict(path=str(mapping_path), bytes=len(raw), sha256=sha(raw))
        mapping = json.loads(raw)
        previous, record_counts = {}, Counter()
        if len(mapping) != 3840:
            raise ValueError('Old mapping record count mismatch')
        for row in mapping:
            v, split = row['vid'], row['split']
            if split not in ('fit', 'dev', 'test') or (v in previous and previous[v] != split):
                raise ValueError('Old account partition mismatch')
            previous[v] = split
            record_counts[v] += 1
        if Counter(previous.values()) != {'fit': 480, 'dev': 192, 'test': 288} or set(record_counts.values()) != {4}:
            raise ValueError('Old identity count mismatch')
        del raw, mapping, row, record_counts

        phase = 'same_cleaning_once'
        archive = root/'reports/documentation/20261107/evolution_qualification/received_windows/data-and-readme.zip'
        receipts['archive'] = receipt(archive)
        if receipts['archive']['sha256'] != bound['archive_sha256']:
            raise ValueError('Official archive identity mismatch')
        access('official archive: existing four market tables only')
        records, cleaning = clean_archive(archive, sources, expected, check)
        write(output/'cleaning.json', cleaning)
        phase = 'strict_role_capacity'
        by_account = defaultdict(list)
        for row in records:
            by_account[row.vid].append(row)
        counters = {name: Counter() for name in ('fit', 'dev', 'test', 'unused')}
        registration_days, months, lag_bins = defaultdict(set), defaultdict(Counter), Counter()
        pairs = 0
        cache = private/'eligible.jsonl'
        with cache.open('x', encoding='utf-8') as stream:
            for vid in sorted(by_account):
                check()
                prior, rows = category(vid, previous), by_account[vid]
                count = counters[prior]
                count['strict_pool_accounts'] += 1
                count['strict_pool_records'] += len(rows)
                if len(rows) >= 8:
                    count['at_least_eight'] += 1
                parts, reason = roles(rows)
                if reason:
                    count[reason] += 1
                    continue
                count['strict_date_roles'] += 1
                near_copy, compared = role_near_copy(parts, follow)
                pairs += compared
                if near_copy:
                    count['role_near_copy_rejected'] += 1
                    continue
                count['eligible'] += 1
                reg = parts['fit'][-1].observed
                registration_days[prior].add(reg)
                months[prior][reg[:7]] += 1
                lag = (date.fromisoformat(parts['query'][0].observed)-date.fromisoformat(reg)).days
                lag_bins['1_to_7' if lag <= 7 else '8_to_30' if lag <= 30 else '31_to_90' if lag <= 90 else 'over_90'] += 1
                stream.write(json.dumps(dict(vid=vid, prior_usage=prior,
                    roles={k: [asdict(r) for r in rs] for k, rs in parts.items()}), ensure_ascii=False)+'\n')
        check()
        eligible = sum(c.get('eligible', 0) for c in counters.values())
        unused = counters['unused'].get('eligible', 0)
        old_test = counters['test'].get('eligible', 0)
        confirmation = unused+old_test
        development = eligible-old_test
        passed = eligible >= 576 and confirmation >= 288 and development >= 288
        result = dict(status='CAPACITY_PASS_NOT_DATA_READY' if passed else 'STOP_CAPACITY',
            counters=counters, eligible_accounts=eligible, unused_eligible_accounts=unused,
            old_test_eligible_accounts=old_test, confirmation_candidate_accounts=confirmation,
            development_candidate_accounts=development,
            required=dict(total=576, confirmation=288, development=288),
            shortfall=dict(total=max(0, 576-eligible), confirmation=max(0, 288-confirmation),
                           development=max(0, 288-development)),
            exact_within_role_pairs=pairs, registration_distinct_days={k: len(v) for k, v in registration_days.items()},
            registration_month_counts=months, fit_to_query_first_day_lag=lag_bins,
            candidate_cache=receipt(cache), sources=receipts,
            cohorts_selected=0, training_exports=0, qrels_opened=0, models_loaded=0, trainings=0,
            old_test_role='confirmation_candidate_only_under_explicit_supplement',
            old_fit_dev_excluded_from_confirmation=True,
            global_cross_account_near_copy_recall='approximate_inherited',
            first_observation_not_creation=True, global_cleaning_uses_future_text=True)
        write(output/'result.json', result)
        success = True
        print(json.dumps({k: result[k] for k in ('status', 'eligible_accounts', 'confirmation_candidate_accounts', 'shortfall')}))
    except BaseException as exc:
        # Do not serialize exception values that could contain real input text.
        write(output/'failure.json', dict(phase=phase, error_type=type(exc).__name__,
              frames=[dict(file=f.filename, line=f.lineno, name=f.name) for f in traceback.extract_tb(exc.__traceback__)],
              automatic_retry=False))
        raise SystemExit(1) from None
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        peak_rss = max(peak_rss, process.memory_info().rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        elapsed = time.monotonic()-START
        write(output/'budget.json', dict(completed=success, measured_body_seconds=elapsed,
            preparation_reserve_seconds=180, charged_auxiliary_seconds=180+__import__('math').ceil(elapsed)+1,
            cap_seconds=1800, peak_rss_bytes=peak_rss, cpu_affinity=affinity, gpu_seconds=0,
            training_seconds=0, failure_retained=not success,
            note='180 seconds conservatively includes transport, setup and closeout; external time receipt checked on completion.'))


if __name__ == '__main__':
    main()
