#!/usr/bin/env python3
"""Comparator/before-start binding supplement; standard library only."""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import sys
import traceback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {'checks': [], 'files': [], 'scope': 'Saved comparator/authorization bindings, no project imports/model/labels.'}

    def read(path):
        b = path.read_bytes()
        report['files'].append({'path': str(path.relative_to(root)), 'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()})
        return b

    def jr(path):
        return json.loads(read(path))

    def ck(name, okay):
        report['checks'].append({'name': name, 'pass': bool(okay)})
        if not okay:
            raise AssertionError(name)

    try:
        risk = root / 'reports/seller_alias_continual/20261004/risk_result'
        policy = jr(risk / 'source/schema/step28_risk_policy.json')
        execution = jr(risk / 'job/execution.json')
        auth = jr(risk / 'authorization.json')
        started = dt.datetime.fromisoformat(read(risk / 'job/started.txt').decode().strip())
        ck('authorization_predates_start', dt.datetime.fromisoformat(auth['created_at']) < started)
        ck('authorized_fixed_study', auth['status'] == 'AUTHORIZED_RISK_REPLAY_AFTER_REVIEW'
           and auth['review_and_primary_passed'] is True)
        ck('authorized_runtime_matches_policy', auth['runtime'] == policy['runtime'])
        ck('authorized_supervision_matches_policy', auth['supervision'] == policy['supervision'])
        ck('single_cpu_runtime', len(execution['cpu_affinity']) == 1)
        ck('actual_risk_environment_matches_baseline', all(execution[key] == value
            for key, value in policy['baseline']['environment'].items()))
        baseline = root / 'dependencies/shared'
        bm = jr(baseline / 'run/manifest.json')
        partition = jr(baseline / 'run/partition.json')
        shared_collection = jr(baseline / 'evaluation/collected.json')
        mapping = jr(root / 'dependencies/mapping.json')
        binding = jr(risk / 'job/evaluation/logit_reference_binding.json')
        ck('preselected_logit_job_matches_attachment_map', policy['pending_logit_reference']['linux_job'] == mapping['logit_tenth']['linux_job'])
        specifications = dict(policy['continuation_references'])
        specifications['logit_tenth'] = {
            'policy_sha256': policy['pending_logit_reference']['policy_sha256'],
            'memory_arm': 'logit', 'evidence_tag': 'LOGIT_WEIGHT',
            'records': binding['completed_records']}
        report['comparator_details'] = {}
        for arm, spec in specifications.items():
            dep = root / mapping[arm]['attachment']
            records = {}
            for name, rec in spec['records'].items():
                blob = read(dep / name)
                ck(arm + ':original_bound_record:' + name,
                   len(blob) == rec['bytes'] and hashlib.sha256(blob).hexdigest() == rec['sha256'])
                records[name] = json.loads(blob)
            em, rm, rc, re = [records[name] for name in ('execution.json', 'run/manifest.json',
                                                       'evaluation/collected.json', 'evaluation/evaluation.json')]
            ck(arm + ':fixed_policy_everywhere', all(data['policy_sha256'] == spec['policy_sha256'] for data in (rm, rc, re)))
            ck(arm + ':sources_same_execution_to_result', all(data['source_files'] == em['source_files'] for data in (rm, rc, re)))
            ck(arm + ':environment_pairing', all(em[key] == value for key, value in policy['baseline']['environment'].items()))
            ck(arm + ':baseline_records_pairing', rm['baseline_records'] == policy['baseline']['records'])
            ck(arm + ':partition_pairing', records['run/partition.json'] == partition)
            ck(arm + ':collection_group_order', all(rc[key] == shared_collection[key] for key in ('group_ids', 'domains', 'metric_columns')))
            expected = {f'{order}_{arm}_stage{stage}' for order in policy['orders'] for stage in (2, 3)}
            ck(arm + ':six_complete_endpoints', set(rm['points']) == set(rc['points']) == expected
               and rm['physical_updates'] == 1728 and rm['gradient_group_presentations'] == 3456)
            ck(arm + ':completed_states', rm['status'] == f"COMPLETE_1728_{spec['evidence_tag']}_UPDATES_VALID_BLIND"
               and rc['status'] == f"ALL_18_{spec['evidence_tag']}_MATRICES_SAVED_BEFORE_COMPARISONS"
               and re['status'] == f"COMPLETE_{spec['evidence_tag']}_DEVELOPMENT_COMPARISON")
            for order in policy['orders']:
                sr = bm['points'][order + '_shared']
                shared = jr(baseline / 'run' / sr['path'])
                restored = rm['restored_starts'][order + '_' + arm]
                for key, value in [('full_checkpoint', shared['full_checkpoint']),
                                   ('model_state_sha256', shared['model_state_sha256']),
                                   ('first_map', shared['first_map_parameters']),
                                   ('memory_source', bm['memories'][order + '_' + spec['memory_arm'] + '_after1']['file']),
                                   ('adam_step', 288), ('first_scores_replayed_exactly', True)]:
                    ck(arm + ':' + order + ':shared_start:' + key, restored[key] == value)
            report['comparator_details'][arm] = {'frozen_policy_sha256': spec['policy_sha256'],
                                               'source_files': len(em['source_files']),
                                               'same_shared_full_checkpoint_model_adam_rng_descriptor': True,
                                               'same_first_map': True, 'matrix_count_sets': 18}
        report['status'] = 'PASS_COMPARATOR_AND_PRESTART_BINDING'
    except Exception as error:
        report['status'] = 'FAILED_REFERENCE_ATTEMPT_PRESERVE'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
        print(report['error']['traceback'], file=sys.stderr)
    report['argv'] = sys.argv
    report['checks_count'] = len(report['checks'])
    report['failed_checks'] = sum(not x['pass'] for x in report['checks'])
    report['project_modules_imported'] = [name for name in sys.modules if name.startswith('step28_')]
    report['torch_imported'] = 'torch' in sys.modules
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'checks_count', 'failed_checks', 'comparator_details') if key in report}, ensure_ascii=False, indent=2))
    return int('error' in report)


if __name__ == '__main__':
    raise SystemExit(main())
