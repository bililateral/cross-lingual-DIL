#!/usr/bin/env python3
"""只读两份用户已提供ZIP，核对身份并做普通算术；不导入项目模块。

用法：python3 verify_mechanical.py NEW_INPUT.zip PRIOR_INPUT.zip OUTPUT_DIRECTORY
不会训练、推理、重采样、生成科研数据、读取外部材料或连接网络。
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

CURRENT_SHA = '0c95fd669fc41cc7423c36038fb2c6d356a607d0676ad69cf8f070aeb63a8300'
PRIOR_SHA = '17d1bfd3214d6b9f164274527925ef2d14774967edd0ae2d22947e620097f487'
PROPOSAL = 'reports/documentation/20261023/continual_baselines/proposal.zh.txt'
BUDGET = 'reports/documentation/20261023/continual_baselines/budget.json'
SELECTED = 'reports/seller_alias_continual/20261023/optuna_result/job/selected.json'
NATIVE = 'reports/documentation/20261018/optuna_implementation/repair/native_completed/qualification/native/result.json'
OLD_REPORT = 'reports/seller_alias_continual/20261011/record_attribution_result/report.zh.txt'
GIB = 2**30


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def identity(path, expected_size, expected_sha, expected_members):
    raw = path.read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    assert len(raw) == expected_size and actual_sha == expected_sha
    payloads = {}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) == expected_members
        assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts for n in names)
        m = json.loads(z.read('input_manifest.json'))
        expected_names = {r['path'] for r in m['files']} | {'input_manifest.json'}
        assert set(names) == expected_names
        rows = []
        for r in m['files']:
            b = z.read(r['path'])
            digest = hashlib.sha256(b).hexdigest()
            assert len(b) == r['size_bytes'] and digest == r['sha256']
            rows.append({'path': r['path'], 'expected_bytes': r['size_bytes'],
                         'actual_bytes': len(b), 'expected_sha256': r['sha256'],
                         'actual_sha256': digest, 'matches': True})
            payloads[r['path']] = b
        payloads['input_manifest.json'] = z.read('input_manifest.json')
    result = {'input_name': path.name, 'bytes': len(raw), 'sha256': actual_sha,
              'zip_members': len(names), 'manifest_payloads': len(rows),
              'all_payloads_match': True, 'missing_members': [], 'extra_members': [],
              'duplicate_members': [], 'files': rows}
    return result, payloads


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('new_input', type=Path)
    parser.add_argument('prior_input', type=Path)
    parser.add_argument('output_directory', type=Path)
    args = parser.parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    check, new = identity(args.new_input, 42232, CURRENT_SHA, 8)
    old_check, old = identity(args.prior_input, 233582, PRIOR_SHA, 35)
    check['prior_input_reuse'] = {
        k: v for k, v in old_check.items() if k != 'files'
    }
    check['prior_input_reuse']['scope_note'] = '原输入可取得且身份一致；哈希核对不等于本轮重审34载荷语义。'
    check['selected_json_same_as_prior_package'] = new[SELECTED] == old[SELECTED]
    assert check['selected_json_same_as_prior_package']
    save(args.output_directory/'input_identity_check.json', check)

    b = json.loads(new[BUDGET])
    selected = json.loads(new[SELECTED])
    native = json.loads(old[NATIVE])
    methods, orders, seeds = len(b['arms']), len(b['orders']), len(b['seeds'])
    paths = methods*orders*seeds
    seq_paths = orders*seeds
    memory_paths = paths-seq_paths
    updates_per_path = 3*6*48
    groups_memory_path = 288+2*288*2
    updates = paths*updates_per_path
    gradients = seq_paths*updates_per_path + memory_paths*groups_memory_path
    states = paths*3
    core_forwards = paths*6*40
    seq_extra = seq_paths*2*40
    forwards = core_forwards+seq_extra
    assert (paths, updates, gradients, states, forwards) == (45, 38880, 59616, 135, 11520)
    for k, v in {'new_full_trajectories': paths, 'optimizer_updates': updates,
                 'gradient_group_presentations': gradients, 'stage_checkpoints': states}.items():
        assert b[k] == v
    assert b['new_hpo_runs'] == 0 and b['confirmation']['group_forwards'] == forwards
    buckets = b['budget_seconds']
    total = sum(buckets.values())
    other = sum(v for k, v in buckets.items() if k not in ('training_shared_pool', 'confirmation_shared_pool'))
    assert total == b['total_seconds'] == 1764000
    assert total/3600 == b['total_hours'] == 490 and other == 14400
    limits = b['limits']
    assert limits['full_run_outer_max_seconds']/3600 == 24
    assert limits['full_run_average_planning_seconds']/3600 == 10

    selected_fields = {}
    for arm, name in [('S','S_tuned'), ('H_pos','H_pos_tuned')]:
        row = next(r for r in b['arms'] if r['name'] == name)
        ref = selected[arm]
        for key in ('lambda_rank','lr_multiplier'):
            assert row[key] == ref['spec'][key]
        assert row['selected_trial'] == ref['number']
        selected_fields[name] = {k: row[k] for k in ('selected_trial','lambda_rank','lr_multiplier')}

    reduced = {}
    for arm in ('S','H_pos'):
        m = native['arms'][arm]['measurements']
        components = {
            'training_proxy_seconds': 288*(m['t_A']+2*m['t_B']),
            'endpoint_and_birth_forward_proxy_seconds': 660*m['v'],
            'load_proxy_seconds': m['L'],
            'three_endpoint_io_proxy_seconds': 3*(m['s_F']+m['r_F']+m['s_I']+m['r_I']),
            'unmeasured_allowance_seconds': 2700
        }
        seconds = 1.25*sum(components.values())
        reduced[arm] = {
            'source': {'package_sha256': PRIOR_SHA, 'file': NATIVE,
                       'json_pointer': '/arms/'+arm+'/measurements'},
            'stored_measurements': m, 'components': components,
            'factor': 1.25, 'single_run_serial_proxy_seconds': seconds,
            'single_run_serial_proxy_hours': seconds/3600,
            'three_times_elapsed_proxy_hours': 3*seconds/3600,
            'status': 'OLD_FIELDS_MECHANICAL_PROXY_NOT_NEW_QUALIFICATION'
        }
    v = max(r['measurements']['v'] for r in native['arms'].values())
    confirm_h = forwards*v*1.25/3600
    assert abs(confirm_h-b['forecast']['serial_confirmation_forward_proxy_hours']) < 1e-10
    assert abs(3*confirm_h-b['forecast']['three_times_confirmation_task_proxy_hours']) < 1e-10

    storage = b['storage']
    storage_components = {
        'permanent_135_inference_GiB': states*storage['inference_file_proxy_bytes']/GIB,
        'concurrent_6_full_GiB': storage['max_concurrent_full_files']*storage['full_file_proxy_bytes']/GIB,
        'concurrent_3_temporary_inference_GiB': storage['max_concurrent_temporary_inference_files']*storage['inference_file_proxy_bytes']/GIB,
        'small_evidence_GiB': storage['small_evidence_max_GiB']
    }
    stored_full_max = max(r['measurements']['W_F'] for r in native['arms'].values())
    assert stored_full_max == storage['full_file_proxy_bytes'] == 3904680901
    named_storage = sum(storage_components.values())
    old_report = old[OLD_REPORT].decode('utf-8')
    assert '9,466,544,128' in old_report and '27小时25分28秒' in old_report
    old_reserved = 9466544128/GIB
    snapshot_free = b['resources']['gpu_free_MiB']/1024
    w = b['forecast']['old_9path_training_seconds']*(paths/9)/3600
    assert abs(w-b['forecast']['serial_equivalent_training_hours']) < 1e-10
    forecasts = []
    for q in (1.,1.5,2.):
        wall = w/q
        forecasts.append({'q_assumption':q, 'training_wall_hours_proxy':wall,
                          'training_cumulative_task_hours_if_three_active':3*wall,
                          'calendar_proxy_with_confirmation_10_to_13h_and_other_full_4h':[wall+14,wall+17]})
    checks = {
        'scope':'仅提供文档/保存字段与机械算术；没有研究执行或新资源测量。',
        'current_input_sha256':CURRENT_SHA, 'prior_input_sha256':PRIOR_SHA,
        'sources':{'proposal':PROPOSAL,'budget':BUDGET,'selected':SELECTED,
                   'old_native':NATIVE,'old_resource_report':OLD_REPORT},
        'source_lines':{'scope_and_formulas':[29,55],'confirmation':[63,81],
                        'budget_resources_forecast':[93,105],'old_gpu_peak':[17,17]},
        'selected_parameters_match':selected_fields,
        'counts':{'methods':methods,'orders':orders,'seeds':seeds,'full_paths':paths,
                  'SEQ_paths':seq_paths,'memory_paths':memory_paths,'updates_per_path':updates_per_path,
                  'updates':updates,'gradient_groups_SEQ':seq_paths*updates_per_path,
                  'gradient_groups_memory':memory_paths*groups_memory_path,
                  'gradient_groups_total':gradients,'stage_states':states,
                  'confirmation_core_sixcell_forwards':core_forwards,
                  'confirmation_SEQ_extra_forwards':seq_extra,'confirmation_total_forwards':forwards,
                  'independent_confirmation_groups':120,
                  'excluded_from_confirmation_count':'训练端点的旧dev/cal评分、恢复、教师出生前向计入训练池，不是11520的一部分。'},
        'budget':{'buckets_seconds':buckets,'other_seconds':other,'other_hours':other/3600,
                  'total_seconds':total,'total_hours':total/3600,
                  'training_pool_hours':buckets['training_shared_pool']/3600,
                  'run_max_hours':limits['full_run_outer_max_seconds']/3600,
                  'average_planning_hours':limits['full_run_average_planning_seconds']/3600,
                  '45_times_24h_arithmetic_only_not_authorized':paths*24,
                  'native_optimizer_calls_cap':limits['native_optimizer_calls_max']},
        'reduced_training_proxy':{'formula':'1.25*[288*(t_A+2*t_B)+660*v+L+3*(s_F+r_F+s_I+r_I)+2700]',
                                  'forward_count':{'three_endpoints_three_passes_cal_dev':3*3*(12+60),'birth_teacher_upper':12,'sum':660},
                                  'removed':'2088轮诊断前向、60未训练起点评分、72次旧诊断历史教师投影；训练更新与必要端点不删。',
                                  'arms':reduced},
        'confirmation_proxy':{'old_max_v_seconds':v,'single_route_forward_only_hours':confirm_h,
                              'three_times_task_forward_only_hours':3*confirm_h,
                              '36h_minus_three_times_forward_proxy_hours':36-3*confirm_h,
                              'not_included_in_forward_proxy':'新载模/I/O/标签指标/失败尾部；三倍不是已证实上界。'},
        'storage_proxy':{'components':storage_components,'named_total_GiB':named_storage,
                         'headroom_to_224GiB':224-named_storage,'cap_GiB':224,'required_free_GiB':288,
                         'headroom_free_requirement_over_cap_GiB':64,
                         'full_file_proxy_matches_old_native_max':True,
                         'status':'生命周期及新文件峰值未执行验证'},
        'gpu_proxy':{'snapshot_source':'本次proposal101及budget/resources转述；raw server_resources.json未提供，未独立核验。',
                     'snapshot_total_GiB':b['resources']['gpu_total_MiB']/1024,
                     'snapshot_free_GiB':snapshot_free,'old_single_torch_reserved_GiB':old_reserved,
                     'three_times_old_torch_reserved_GiB':3*old_reserved,
                     'snapshot_free_minus_three_old_reserved_GiB':snapshot_free-3*old_reserved,
                     'not_a_concurrent_admission_proof':True,
                     'old_native_resources_note':'旧native中部分reserved=0属于释放后字段，不能作为运行峰值。'},
        'forecast':{'old_9path_exact_seconds_source':'本次budget/forecast和历史主审转述；没有独立取得精确原计时字段。',
                    'old_9path_exact_seconds_as_provided':b['forecast']['old_9path_training_seconds'],
                    'old_report_integer_wall_seconds':27*3600+25*60+28,
                    'difference_seconds':27*3600+25*60+28-b['forecast']['old_9path_training_seconds'],
                    'serial_equivalent_work_hours':w,'conditions':forecasts,
                    'q_for_450h_full_three_proxy_only_not_a_new_gate':3*w/450,
                    'exact_task_accounting_relation':'sum_i elapsed_i = integral(active_worker_count(t) dt); 满三路才约等于3*训练墙钟。',
                    'q_measured':False,'current_server_checked':False,'calendar_guarantee':False},
        'accounting_semantics':{
            'time_units':'累计任务外层elapsed，不是CPU user/system时间、墙钟或物理GPU小时。',
            'reservation':'已结算U+活跃run未结算整段预留+未覆盖尾部<=共享池；预留含已发生但尚未结算部分。',
            'child_limit':'实际分配r_i<=24h须传入父launcher和child Budget/watchdog，不能留旧21600s。',
            'failure':'失败与恢复耗时保留，同run不能清零；不完整45条不得用子集开结果。',
            'other_pools':'确认36h、资格5400s也按共享累计口径，不各worker再获完整一份。'}
    }
    save(args.output_directory/'mechanical_checks.json', checks)
    print(json.dumps({'identity':'PASS_7_OF_7','prior_identity':'PASS_34_OF_34',
                      'counts':checks['counts'],'total_hours':total/3600,
                      'storage_GiB':named_storage,'storage_headroom_GiB':224-named_storage,
                      'serial_confirmation_hours':confirm_h},ensure_ascii=False))


if __name__ == '__main__':
    main()
