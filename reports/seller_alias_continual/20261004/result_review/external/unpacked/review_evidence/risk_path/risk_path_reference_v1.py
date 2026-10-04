#!/usr/bin/env python3
"""Independent saved-record review. No project imports, Torch, models or labels.

Only original saved metrics/scores, update logs and descriptors are read. UID-only
schedule and reservoir simulations never access a training group payload.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import sys
import traceback

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    inputs, checks, report = {}, [], {}

    def read(path):
        path = path.resolve()
        payload = path.read_bytes()
        inputs[str(path.relative_to(root))] = {
            'path': str(path.relative_to(root)), 'bytes': len(payload),
            'sha256': hashlib.sha256(payload).hexdigest()}
        return payload

    def jread(path):
        return json.loads(read(path))

    def check(name, condition, detail=None):
        checks.append({'name': name, 'pass': bool(condition), 'detail': detail})
        if not condition:
            raise AssertionError(name + (': ' + repr(detail) if detail is not None else ''))

    def identity(path, rec, name):
        raw = read(path)
        check(name, len(raw) == rec['bytes'] and hashlib.sha256(raw).hexdigest() == rec['sha256'])

    def array(path, shape=None):
        import io
        result = np.load(io.BytesIO(read(path)), allow_pickle=False)
        check('finite_array:' + str(path.relative_to(root)), np.isfinite(result).all())
        if shape is not None:
            check('array_shape:' + str(path.relative_to(root)), result.shape == shape)
        return result

    def seed(*parts):
        payload = (json.dumps(list(parts), ensure_ascii=False, sort_keys=True,
                              separators=(',', ':'), allow_nan=False) + '\n').encode()
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], 'big') % (2**63 - 1)

    def timestamp(path):
        return dt.datetime.fromisoformat(read(path).decode().strip())

    try:
        risk = root / 'reports/seller_alias_continual/20261004/risk_result'
        source, job = risk / 'source', risk / 'job'
        run = job / 'run'
        logit = root / 'reports/seller_alias_continual/20261004/logit_low_result/job'
        shared = root / 'dependencies/shared'
        policy = jread(source / 'schema/step28_risk_policy.json')
        policy_hash = hashlib.sha256(read(source / 'schema/step28_risk_policy.json')).hexdigest()
        old_policy = jread(source / 'schema/step28_bge_continual_policy.json')
        execution = jread(job / 'execution.json')
        manifest = jread(run / 'manifest.json')
        auth = jread(risk / 'authorization.json')
        audit = jread(root / 'direct_reviews/risk_cpu/audit.json')
        collection = jread(job / 'evaluation/collected.json')
        evaluation = jread(job / 'evaluation/evaluation.json')
        inventory = jread(risk / 'inventory.json')
        check('risk_frozen_source_count', len(execution['source_files']) == 32)
        check('risk_source_list_unique', len({r['path'] for r in execution['source_files']}) == 32)
        for owner, data in [('manifest', manifest), ('authorization', auth), ('audit', audit),
                            ('collection', collection), ('evaluation', evaluation)]:
            check('risk_sources_match_' + owner, data['source_files'] == execution['source_files'])
        for rec in execution['source_files']:
            identity(source / rec['path'], rec, 'source:' + rec['path'])
        for rec in inventory['files']:
            identity(risk / rec['path'], rec, 'return_inventory:' + rec['path'])
        for owner, data in [('manifest', manifest), ('authorization', auth),
                            ('collection', collection), ('evaluation', evaluation)]:
            check('risk_policy_' + owner, data['policy_sha256'] == policy_hash)
        check('risk_policy_sha_literal', policy_hash == '5706629736dfe9c598f1d59cc61f04bb65e9fd83b4275e391dfc3d2ebf2e5548')
        check('risk_fixed_arm_and_coefficient', policy['arms'] == {'risk': .1}
              and policy['loss']['risk_retention'] == .5)
        identity(risk / 'authorization.json', execution['authorization'], 'executed_authorization_binding')
        identity(root / 'direct_reviews/risk_cpu/audit.json', execution['audit'], 'executed_native_audit_binding')
        native = jread(root / 'direct_reviews/risk_cpu/native.json')
        identity(root / 'direct_reviews/risk_cpu/native.json', audit['native']['risk'], 'native_evidence_binding')
        check('native_fixture_scope', native['actual_native_updates'] == 3
              and 'counter_fixture' in native and 'reference_fixture' in native)
        check('native_two_parameter_slices_increment', all(
            item['different_actual_parameter_update'] and item['increment_norm'] > 0
            for item in native['retention_gradient_decomposition'].values()))
        check('native_same_base_objectives', all(native['updates']['risk'][name] == native['updates']['zero'][name]
            for name in ('current_total', 'history_total', 'weighted_history_total')))
        native_gap = native['updates']['risk']['total'] - native['updates']['zero']['total']
        check('native_total_increment_half_retention', abs(native_gap - .5 * native['updates']['risk']['retention']) < 1e-12)
        report['reused_native_evidence'] = {
            'scope': 'Linux handmade activation/counter fixture, 3 native updates; not rerun here',
            'retention_gradient_decomposition': native['retention_gradient_decomposition'],
            'total_increment': native_gap}

        shared_manifest = jread(shared / 'run/manifest.json')
        partition = jread(run / 'partition.json')
        check('risk_same_partition', partition == jread(shared / 'run/partition.json'))
        for role, size in [('fit', 48), ('calibration', 12), ('development', 20)]:
            check('partition_cardinality_' + role,
                  collections.Counter(row['domain'] for row in partition[role]) == {d: size for d in 'ABC'})
        check('partition_disjoint', len({row['group_uid'] for role in ('fit', 'calibration', 'development')
                                        for row in partition[role]}) == 240)
        fitting = {row['group_uid']: row['domain'] for row in partition['fit']}
        all_points = {f'{order}_risk_stage{stage}' for order in policy['orders'] for stage in (2, 3)}
        check('six_complete_points', set(manifest['points']) == all_points == set(manifest['training'])
              == set(collection['points']))
        check('training_budget', manifest['physical_updates'] == 1728
              and manifest['gradient_group_presentations'] == 3456)
        check('manifest_before_valid_state', manifest['status'] == 'COMPLETE_1728_RISK_REPLAY_UPDATES_VALID_BLIND')
        before = jread(job / 'before_valid.json')
        identity(run / 'manifest.json', before['manifest'], 'blind_gate_bound_manifest')
        check('blind_gate_access', before['label_parses'] == {'train': 1, 'valid': 0, 'heldout': 0, 'owners': 0})
        after = jread(job / 'access.json')
        check('final_access', after == {'train': 1, 'valid': 1, 'heldout': 0, 'owners': 0})
        completion = jread(job / 'completion.json')
        check('completion_access', completion['label_parses'] == after)
        check('completion_success', completion['status'] == 'COMPLETE_RISK_REPLAY_DEVELOPMENT_COMPARISON'
              and read(job / 'exit_status.txt').decode().strip() == '0' and not (job / 'failure.json').exists())
        identity(job / 'evaluation/evaluation.json', completion['evaluation'], 'completion_evaluation_binding')
        check('program_budget', completion['budget']['elapsed_seconds'] < 43200
              and completion['budget']['peak_observed_bytes'] < 16 * 2**30)

        weights = jread(risk / 'verification/payload_custody.json')
        check('six_remote_weight_descriptors', len(weights['weights_retained_linux']) == 6)
        check('prior_audit_did_not_load_model', weights['models_loaded'] is False
              and weights['formal_labels_parsed'] is False)
        weights_by_name = {Path(rec['path']).name: rec for rec in weights['weights_retained_linux']}
        report['stages'], report['starts'], report['retention_lifecycle'] = {}, {}, {}
        cache_sizes, totals = [], []
        sum_retention = []
        for order in policy['orders']:
            start = manifest['restored_starts'][order + '_risk']
            initial = shared_manifest['memories'][order + '_er_after1']
            rec = shared_manifest['points'][order + '_shared']
            identity(shared / 'run' / rec['path'], rec, order + ':shared_point_identity')
            first = jread(shared / 'run' / rec['path'])
            for name, expected in [('full_checkpoint', first['full_checkpoint']),
                                   ('model_state_sha256', first['model_state_sha256']),
                                   ('first_map', first['first_map_parameters']),
                                   ('memory_source', initial['file']), ('adam_step', 288)]:
                check(order + ':shared_start:' + name, start[name] == expected)
            check(order + ':first_replay_record', start['first_scores_replayed_exactly'] is True)
            sm = start['memory_summary']
            check(order + ':six_initial_members', sm['members'] == initial['members'] and len(sm['members']) == 6)
            check(order + ':initial_reference_domains', set(sm['risk_references']) == set(sm['members'])
                  == set(sm['risk_origins']) and all(o == 1 and fitting[uid] == order[0]
                                                     for uid, o in sm['risk_origins'].items()))
            check(order + ':initial_counters', sm['seen'] == 48 and sm['draw_count'] == 0 and sm['draw_stage'] == 0)
            retained = manifest['memories'][f'{order}_risk_stage2']
            rng_retention = random.Random(seed(old_policy['memory_seed'], order, 'retention'))
            reservoir, seen = [], 0
            reservoir_at_stage = {}
            for source_stage in (1, 2):
                for uid in sorted(uid for uid, domain in fitting.items() if domain == order[source_stage - 1]):
                    seen += 1
                    if seen <= 6:
                        reservoir.append(uid)
                    else:
                        slot = rng_retention.randrange(seen)
                        if slot < 6:
                            reservoir[slot] = uid
                reservoir_at_stage[source_stage] = reservoir.copy()
            check(order + ':uid_only_algorithm_R_stage1', reservoir_at_stage[1] == sm['members'])
            check(order + ':uid_only_algorithm_R_stage2', reservoir_at_stage[2] == retained['members'])
            survivors = set(sm['members']) & set(retained['members'])
            newcomers = set(retained['members']) - set(sm['members'])
            check(order + ':surviving_references_not_refreshed', all(
                retained['risk_references'][uid] == sm['risk_references'][uid]
                and retained['risk_origins'][uid] == 1 for uid in survivors))
            check(order + ':new_reference_domains', set(retained['risk_references']) == set(retained['members'])
                  == set(retained['risk_origins']) and all(retained['risk_origins'][uid] == 2
                    and fitting[uid] == order[1] for uid in newcomers))
            check(order + ':retained_budget_counter', retained['seen'] == 96 and retained['draw_count'] == 0
                  and retained['draw_stage'] == 0 and retained['serialized_bytes'] == retained['file']['bytes'])
            report['starts'][order] = {'model_state_sha256': start['model_state_sha256'],
                                      'adam_step': start['adam_step'], 'first_map': start['first_map'],
                                      'initial_recorded_memory_bytes': sm['serialized_bytes']}
            report['retention_lifecycle'][order] = {'survivors': len(survivors), 'newcomers': len(newcomers),
                'members_and_rng_recreated_from_uids': True,
                'reference_hashes_bound_and_survivors_unchanged': True,
                'saved_memory_recorded_bytes': retained['serialized_bytes'],
                'real_cache_payload_deserialized_here': False}
            cache_sizes.extend([sm['serialized_bytes'], retained['serialized_bytes']])
            for stage in (2, 3):
                name = f'{order}_risk_stage{stage}'
                point = jread(run / manifest['points'][name]['path'])
                identity(run / manifest['points'][name]['path'], manifest['points'][name], name + ':point')
                log = jread(run / manifest['training'][name]['path'])
                identity(run / manifest['training'][name]['path'], manifest['training'][name], name + ':training')
                original_log = jread(shared / 'run/updates' / f'{order}_er_stage{stage}.json')
                for field in ('current_ids', 'history_ids', 'current_dropout_stream', 'adam_step', 'updates'):
                    check(name + ':paired_original_' + field, log[field] == original_log[field])
                current = sorted(uid for uid, domain in fitting.items() if domain == order[stage - 1])
                stream = seed(old_policy['schedule_seed'], order, stage, 'current')
                rng_current = random.Random(stream)
                scheduled = []
                for epoch in range(6):
                    shuffled = current.copy()
                    rng_current.shuffle(shuffled)
                    scheduled.extend(shuffled)
                check(name + ':independent_current_schedule', log['current_ids'] == scheduled
                      and log['current_dropout_stream'] == stream)
                check(name + ':six_passes_each_current_group', collections.Counter(log['current_ids'])
                      == {uid: 6 for uid in current})
                live = sm if stage == 2 else retained
                draw_rng = random.Random(seed(old_policy['memory_seed'], order, stage, 'history_draws'))
                independent_draws = [live['members'][draw_rng.randrange(6)] for _ in range(288)]
                check(name + ':independent_history_draws', log['history_ids'] == independent_draws)
                check(name + ':no_current_history_overlap', not set(current) & set(log['history_ids']))
                check(name + ':reference_origins_old_for_every_draw', all(
                    live['risk_origins'][uid] < stage for uid in log['history_ids']))
                consumed = log['memory_after_training']
                check(name + ':consumed_reference_hashes', all(consumed[k] == live[k]
                    for k in ('members', 'risk_references', 'risk_origins', 'seen')))
                check(name + ':no_logit_reference', consumed['with_logits'] is False
                      and consumed['references'] == {} and consumed['reference_origins'] == {})
                check(name + ':draw_counter', consumed['draw_count'] == 288 and consumed['draw_stage'] == stage)
                budget_rec = jread(run / 'memory' / f'{name}_budget.json')
                for key in ('members', 'risk_references', 'risk_origins', 'seen', 'draw_stage', 'draw_count'):
                    check(name + ':checkpoint_kept_references:' + key, budget_rec[key] == consumed[key])
                cache_sizes.extend([consumed['serialized_bytes'], budget_rec['serialized_bytes']])
                for obj in (live, consumed, budget_rec):
                    check(name + ':cache_complete_budget:' + str(obj['serialized_bytes']),
                          0 < obj['auxiliary_serialized_bytes'] < obj['serialized_bytes'] <= 2**20)
                values = array(run / log['update_file']['path'], (288, 19))
                identity(run / log['update_file']['path'], log['update_file'], name + ':update_array')
                fields = {key: values[:, index] for index, key in enumerate(log['update_columns'])}
                all_base_residuals = []
                for role in ('current', 'history'):
                    residual = fields[role + '_total'] - (fields[role + '_bce']
                               + fields[role + '_rank'] + .5 * fields[role + '_hard'])
                    all_base_residuals.extend(residual.tolist())
                    check(name + ':base_objective:' + role, np.allclose(residual, 0, atol=3e-6, rtol=0))
                independent_retention = sum(fields['retention_' + channel] for channel in ('rank', 'positive', 'negative')) / 3
                independent_objective = fields['current_total'] + .1 * fields['history_total'] + .5 * independent_retention
                check(name + ':lambda_all_history', np.array_equal(fields['weighted_history_total'], .1 * fields['history_total']))
                check(name + ':retention_channel_mean', np.allclose(fields['retention'], independent_retention, atol=1e-8, rtol=2e-6))
                check(name + ':whole_objective_from_channels', np.allclose(fields['total'], independent_objective, atol=3e-6, rtol=3e-6))
                check(name + ':fixed_lambda_gamma', np.all(fields['history_weight'] == .1) and np.all(fields['retention_weight'] == .5))
                check(name + ':nonnegative_channel_penalties', all(np.all(fields['retention_' + channel] >= 0)
                    for channel in ('rank', 'positive', 'negative')))
                expected_rates = [1e-5 * (step / 29 if step <= 29 else (288 - step) / 259) for step in range(1, 289)]
                check(name + ':entire_lr_schedule', np.array_equal(fields['encoder_lr'], expected_rates)
                      and np.all(fields['head_lr'] == .001))
                check(name + ':continuous_adam_endpoint', log['adam_step'] == point['completed_updates'] == stage * 288)
                check(name + ':observation_steps', set(log['observations']) == {'1', '29', '30', '288'})
                for step, observation in log['observations'].items():
                    for module in ('encoder', 'head'):
                        check(name + ':actual_gradient_update:' + step + ':' + module,
                              observation[module]['finite_nonzero_combined_gradient'] is True
                              and observation[module]['parameters_changed'] == (module == 'head' or step != '288'))
                check(name + ':point_restoration_metadata', point['full_model_adam_and_rng_restore_verified'] is True
                      and point['first_map_parameters'] == first['first_map_parameters']
                      and point['actual_domain'] == order[stage - 1] and point['policy_sha256'] == policy_hash)
                record_weight = weights_by_name[Path(point['model']['path']).name]
                check(name + ':remote_weight_descriptor_matches', all(point['model'][k] == record_weight[k] for k in ('bytes', 'sha256')))
                check(name + ':remote_weight_not_in_attachment', not (run / point['model']['path']).exists())
                score = {}
                for role, shape in [('calibration', (12, 378)), ('development', (60, 378)),
                                    ('stage-cal', (60, 378)), ('first-cal', (60, 378))]:
                    identity(run / point['scores'][role]['path'], point['scores'][role], name + ':score_identity:' + role)
                    score[role] = array(run / point['scores'][role]['path'], shape)
                mapping = jread(run / point['mapping']['path'])
                identity(run / point['mapping']['path'], point['mapping'], name + ':map_identity')
                cal_uids = [row['group_uid'] for row in partition['calibration'] if row['domain'] == order[stage - 1]]
                check(name + ':current_only_calibration', mapping['calibration_group_ids'] == cal_uids
                      and mapping['group_count'] == 12 and mapping['pair_count'] == 4536
                      and mapping['positive_count'] == 240 and mapping['actual_domain'] == order[stage - 1]
                      and mapping['score_source'] == point['scores']['calibration']
                      and mapping['model_state_sha256'] == point['model_state_sha256'])
                for role, affine in [('stage-cal', mapping), ('first-cal', first['first_map_parameters'])]:
                    # Parameters are saved in mapping itself. No fitting or label access.
                    a, b = affine['a'], affine['b']
                    check(name + ':positive_calibration_slope:' + role, a > 0)
                    expected = a * score['development'].astype(np.float64) + b
                    check(name + ':affine_full_scores:' + role, np.array_equal(score[role], expected))
                    order_idx = np.argsort(score['development'], axis=1, kind='stable')
                    raw_ordered = np.take_along_axis(score['development'], order_idx, axis=1)
                    cal_ordered = np.take_along_axis(score[role], order_idx, axis=1)
                    check(name + ':affine_order_and_ties:' + role,
                          np.all(np.diff(cal_ordered, axis=1) >= 0)
                          and np.array_equal(np.diff(raw_ordered, axis=1) == 0,
                                             np.diff(cal_ordered, axis=1) == 0))
                report['stages'][name] = {
                    'updates': 288, 'adam_step': log['adam_step'],
                    'current_stream': stream, 'unique_current_groups': len(set(scheduled)),
                    'unique_history_groups': len(set(independent_draws)),
                    'current_each_group_times': 6,
                    'retention_positive_steps': int(np.sum(fields['retention'] > 0)),
                    'retention_mean': float(fields['retention'].mean()),
                    'retention_weighted_mean': float((.5 * fields['retention']).mean()),
                    'channel_means': {channel: float(fields['retention_' + channel].mean()) for channel in ('rank', 'positive', 'negative')},
                    'maximum_base_decomposition_residual': max(map(abs, all_base_residuals)),
                    'maximum_retention_mean_residual': float(np.max(np.abs(fields['retention'] - independent_retention))),
                    'maximum_objective_from_channels_residual': float(np.max(np.abs(fields['total'] - independent_objective))),
                    'all_288_gradients_above_clip_one': bool(np.all(fields['gradient_norm'] > 1)),
                    'current_calibration_group_count': 12, 'current_calibration_pair_count': 4536,
                    'affine_score_arrays_verified': 2}
                totals.append(len(values))
                sum_retention.extend(fields['retention'].tolist())

        # All five preselected comparator records and each reused set bind to the
        # actual original collections. This never infers comparator choice by effect.
        binding = jread(job / 'evaluation/logit_reference_binding.json')
        check('preselected_logit_spec_preserved', binding['preselected_job'] == policy['pending_logit_reference'])
        check('no_result_based_comparator_selection_record', binding['selection_by_results'] is False)
        identity(logit / 'execution.json', policy['pending_logit_reference']['execution'], 'preselected_logit_execution')
        expected_record_names = {'execution.json', 'run/manifest.json', 'run/partition.json',
                                 'evaluation/collected.json', 'evaluation/evaluation.json'}
        check('all_five_completed_logit_records', set(binding['completed_records']) == expected_record_names)
        for name, rec in binding['completed_records'].items():
            identity(logit / name, rec, 'bound_logit_record:' + name)
        logit_m = jread(logit / 'run/manifest.json')
        check('bound_logit_policy', logit_m['policy_sha256'] == policy['pending_logit_reference']['policy_sha256'])
        check('bound_logit_complete', jread(logit / 'completion.json')['status'] == 'COMPLETE_LOGIT_WEIGHT_DEVELOPMENT_COMPARISON')
        reference = jread(job / 'evaluation/reference/collected.json')
        check('new_18_sets_and_reused_81', len(collection['points']) * 3 == 18
              and len(reference['points']) * 3 == 81
              and evaluation['new_metric_count_sets'] == 18 and evaluation['reused_metric_count_sets'] == 81)
        check('same_group_order_all_collections', collection['group_ids'] == reference['group_ids']
              == [row['group_uid'] for row in partition['development']]
              and collection['domains'] == reference['domains'] == [row['domain'] for row in partition['development']])
        original_collections = {'shared': jread(shared / 'evaluation/collected.json')}
        for method_name in ('tenth', 'logit_quarter'):
            dep = root / 'dependencies' / method_name
            for name, rec in policy['continuation_references'][method_name]['records'].items():
                identity(dep / name, rec, method_name + ':policy_fixed_record:' + name)
            original_collections[method_name] = jread(dep / 'evaluation/collected.json')
        for name, rec in policy['baseline']['records'].items():
            identity(shared / name, rec, 'shared:policy_fixed_record:' + name)
        original_collections['logit_tenth'] = jread(logit / 'evaluation/collected.json')
        for name, roles in reference['points'].items():
            match = ('shared' if name.endswith('_shared') or '_seq_stage' in name else
                     next(method_name for method_name in ('logit_tenth', 'logit_quarter', 'tenth')
                          if '_' + method_name + '_stage' in name))
            check('reused_original_records:' + name, roles == original_collections[match]['points'][name])
            for role, pair in roles.items():
                for payload_type, rec in pair.items():
                    identity(job / 'evaluation/reference' / rec['path'], rec,
                             'reused_payload:' + name + ':' + role + ':' + payload_type)

        # One start, complete 24-step progress ladder per stage, final success.
        log_lines = read(job / 'train.log').decode().splitlines()
        log_events = [json.loads(line) for line in log_lines if line.startswith('{')]
        progress = [row for row in log_events if row.get('event') == 'updates']
        check('72_progress_events', len(progress) == 72)
        check('unique_training_completion', sum(row.get('status') == 'COMPLETE_RISK_REPLAY_DEVELOPMENT_COMPARISON' for row in log_events) == 1)
        for order in policy['orders']:
            for stage in (2, 3):
                rows = [row for row in progress if row['order'] == order and row['stage'] == stage]
                check(f'{order}{stage}:progress_ladder', [row['stage_updates'] for row in rows] == list(range(24, 289, 24))
                      and all(row['logical_updates'] == (stage - 1) * 288 + row['stage_updates'] for row in rows))
        check('monotonic_training_time', all(b['elapsed_seconds'] > a['elapsed_seconds'] for a, b in zip(progress, progress[1:])))
        listener = risk / 'listener'
        samples = []
        for line in read(listener / 'events.log').decode().splitlines():
            match = re.fullmatch(r'(\S+) gpu_free_mib=(\d+) host_free_kib=(\d+) disk_free_bytes=(\d+) cpu=(\d+) idle=(\d+)/(\d+)', line)
            if match:
                t, gpu, host, disk, cpu, idle, all_ticks = match.groups()
                samples.append({'time': t, 'gpu_free_bytes': int(gpu) * 2**20,
                                'host_free_bytes': int(host) * 1024, 'disk_free_bytes': int(disk),
                                'cpu': int(cpu), 'idle_fraction': int(idle) / int(all_ticks)})
        def ready(sample):
            return (sample['gpu_free_bytes'] >= 24 * 2**30 and sample['host_free_bytes'] >= 16 * 2**30
                    and sample['disk_free_bytes'] >= 16 * 2**30 and sample['idle_fraction'] >= .9)
        check('resource_samples_exist', len(samples) > 1)
        check('wait_until_resource_gate', not any(ready(sample) for sample in samples[:-1]) and ready(samples[-1]))
        risk_start, risk_finish = timestamp(job / 'started.txt'), timestamp(job / 'finished.txt')
        listener_start = timestamp(listener / 'started.txt')
        listener_finish = timestamp(listener / 'finished.txt')
        delete_time = timestamp(listener / 'script_deleted_after_updates.txt')
        logit_finish = timestamp(logit / 'finished.txt')
        check('resource_gate_at_training_start', dt.datetime.fromisoformat(samples[-1]['time']) == risk_start)
        check('one_listener_clean_exit', read(listener / 'exit_status.txt').decode().strip() == '0'
              and listener_finish == risk_finish
              and read(listener / 'events.log').decode().count('training_exit=0 cleanup_exit=0; no automatic retry') == 1)
        check('script_deleted_after_real_update', delete_time > risk_start + dt.timedelta(seconds=progress[0]['elapsed_seconds'])
              and delete_time < risk_finish and inventory['listener_script_exists'] is False)
        listener_hash = read(listener / 'script.sha256').decode().split()[0]
        check('listener_uses_closed_revision_sha', listener_hash == '1a20eaa71a53a8a36074055d7c674ff5507df21dc357f339160f8acee849ac1f')
        check('logit_completed_before_risk_last_training_update', logit_finish < risk_start + dt.timedelta(seconds=progress[-1]['elapsed_seconds']))
        check('fixed_risk_execute_argv', 'execute --study risk' in read(job / 'resource_usage.log').decode())
        report['lifecycle'] = {
            'risk_started': risk_start.isoformat(), 'risk_finished': risk_finish.isoformat(),
            'risk_elapsed_timestamp_seconds': (risk_finish - risk_start).total_seconds(),
            'risk_elapsed_internal_seconds': completion['budget']['elapsed_seconds'],
            'listener_wait_seconds': (risk_start - listener_start).total_seconds(),
            'listener_resource_samples': len(samples), 'launch_resource_sample': samples[-1],
            'deletion_receipt': delete_time.isoformat(), 'logit_finished': logit_finish.isoformat(),
            'risk_start_minus_logit_wrapper_finish_seconds': (risk_start - logit_finish).total_seconds(),
            'failure_recovery_finalize_exercised': False,
            'risk_and_listener_exit': 0,
            'listener_script_body_reused_from_closed_review_not_reexecuted_here': True}
        report['counts'] = {'sources': 32, 'inventory_files': len(inventory['files']), 'new_updates': sum(totals),
                            'gradient_group_presentations': 2 * sum(totals), 'stage_endpoints': len(all_points),
                            'score_arrays': 24, 'new_metric_sets': 18, 'reused_metric_sets': 81,
                            'retention_nonzero_steps': sum(x > 0 for x in sum_retention),
                            'recorded_memory_bytes_min': min(cache_sizes), 'recorded_memory_bytes_max': max(cache_sizes),
                            'retained_weight_descriptor_bytes': sum(rec['bytes'] for rec in weights['weights_retained_linux'])}
        report['limits'] = [
            'Frozen code inspected; independent standard-library/NumPy calculations use saved allowed records only.',
            'No Torch, training modules, models, labels, training cache, remote server, test or owners accessed.',
            'Actual formal gradients and native restore cannot be reproduced from omitted weights; reuse bound prior native evidence and execution records.',
            'Recorded cache bytes and reference digests checked; real serialized cache was not re-deserialized here.',
            'No claim of exhaustive event provenance beyond the attached source and execution records.',
            'Metric/AP/MAP statistical recalculation is delegated separately; this script binds their input sets and role identities only.']
        report['status'] = 'PASS_INDEPENDENT_RISK_SAVED_RECORDS_AND_EXECUTION_PATH'
        report['issues'] = {'SCIENTIFIC_BLOCKER': [], 'REPRODUCIBILITY_DEFECT': [],
                            'FOLLOWUP_RESEARCH_SUGGESTION': [], 'OUT_OF_SCOPE_OVERDESIGN': []}
    except Exception as error:
        report['status'] = 'FAILED_REFERENCE_ATTEMPT_PRESERVE'
        report['error'] = {'type': type(error).__name__, 'message': str(error), 'traceback': traceback.format_exc()}
    report['checks'] = checks
    report['checked_assertions'] = len(checks)
    report['failed_assertions'] = sum(not row['pass'] for row in checks)
    report['runtime'] = {'python': platform.python_version(), 'numpy': np.__version__,
                          'allowed_cpu_affinity': sorted(os.sched_getaffinity(0)),
                          'torch_imported': 'torch' in sys.modules,
                          'project_modules_imported': sorted(name for name in sys.modules if name.startswith('step28_')),
                          'argv': sys.argv}
    (out / 'inputs.json').write_text(json.dumps({'files': sorted(inputs.values(), key=lambda rec: rec['path'])}, ensure_ascii=False, indent=2) + '\n')
    (out / 'result.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('status', 'checked_assertions', 'failed_assertions', 'counts', 'lifecycle') if key in report}, ensure_ascii=False, indent=2))
    if 'error' in report:
        print(report['error']['traceback'], file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
