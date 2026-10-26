"""Reviewer-only stdlib file/JSON/AST/arithmetic checks; NEVER imports project code.

Usage:
  python3 mechanical_review.py --input-dir EXTRACTED_INPUT --output-dir CHECK_OUTPUT
The input is the provided archive's extracted text files, not a project workspace.
All apparent counterexamples below are written arithmetic scenarios, not GPU tests.
"""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import math
import sys

P = 'reports/documentation/20261026/continual_defaults/'
C = 'reports/documentation/20261026/continual_comparison/'
GIB = 2**30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.input_dir.resolve(), args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    def read(name):
        return json.loads((root/name).read_text(encoding='utf-8'))
    def record(name):
        data = (root/name).read_bytes()
        return dict(path=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
    manifest = read('input_manifest.json')
    assert len(manifest['files']) == 79
    assert len({r['path'] for r in manifest['files']}) == 79
    assert all(record(r['path']) == r for r in manifest['files'])

    ast_read = set()
    def closure(extra=()):
        names = {'scripts/step28_continual_defaults_run.py',
                 'tests/test_step28_continual_defaults.py', *extra}
        todo = list(names)
        while todo:
            name = todo.pop()
            ast_read.add(name)
            tree = ast.parse((root/name).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                imports = ([x.name for x in node.names] if isinstance(node, ast.Import)
                           else [node.module] if isinstance(node, ast.ImportFrom) else [])
                for module in imports:
                    if module and (module.startswith('step28_') or module.startswith('test_step28_')):
                        child = ('tests/' if module.startswith('test_') else 'scripts/')+module+'.py'
                        if child not in names:
                            names.add(child)
                            todo.append(child)
        names.update((P+'implementation.zh.txt', P+'authorization.json'))
        names.update('schema/'+n+'_policy.json' for n in
                     ('step28_chinese_base','step28_alias_ranking','step28_record_replay','step28_bge_continual'))
        return [record(name) for name in sorted(names)]

    scientific = closure()
    orchestration = closure(('scripts/step28_defaults_control.py', 'tests/test_step28_defaults_control.py'))
    orchestration.append(record('run_step28_defaults_linux_20261026.sh'))
    orchestration.sort(key=lambda x: x['path'])
    draft = read(P+'waiting/draft_gate.json')
    cpu = read(P+'qualification/cpu08/receipt.json')
    native = read(P+'qualification/native_h02/H_default/result.json')
    status = read(P+'qualification/native_h02/status.json')
    def bypath(rows):
        assert len(rows) == len({r['path'] for r in rows})
        return {r['path']:r for r in rows}
    cm, nm = bypath(cpu['sources']), bypath(native['source_files'])
    assert len(scientific) == 44 and scientific == draft['sources']
    assert len(orchestration) == 57 and orchestration == draft['orchestration_sources']
    assert all(nm.get(r['path']) == r for r in scientific)
    assert bypath(status['sources']) == nm
    assert len(cm) == 58 and all(record(p) == r for p,r in cm.items())
    assert set(cm)-{r['path'] for r in orchestration} == {'scripts/step28_defaults_cpu.py'}
    present = [r for p,r in nm.items() if (root/p).is_file()]
    absent = sorted(p for p in nm if not (root/p).is_file())
    assert all(record(r['path']) == r for r in present)

    expected_runs=[]
    for order in ('ABC','BCA','CAB'):
        for seed in (4,5,6):
            for name in ('S_default','H_default'):
                expected_runs.append(dict(name=name,order=order,seed=seed,
                    roots=dict(initialization=2026101900+seed,schedule=2026102900+seed,
                               memory=2026103900+seed,partition=20260918,diagnostic=20261008),
                    lambda_rank=1,lr_multiplier=1))
    assert draft['runs'] == expected_runs
    assert draft['budget_seconds'] == dict(training=108*3600,control=2*3600)
    assert draft['per_run_seconds'] == 6*3600
    assert draft['status']=='PENDING_ORCHESTRATION_REVIEW' and draft['review_ready'] is None

    old_gate=read(C+'formal_gate.json'); launch=read(C+'launch.json')
    for label,name in [('gate',C+'formal_gate.json'),('launch',C+'launch.json')]:
        actual=record(name); supplied=draft['predecessor'][label]
        assert all(actual[k]==supplied[k] for k in ('bytes','sha256'))
    assert launch['formal_gate']==draft['predecessor']['gate']
    assert old_gate['campaign']==draft['predecessor']['campaign']
    assert old_gate['workspace']==launch['cwd']==draft['predecessor']['workspace']
    predecessor_ids=sorted({f"{r['order']}_s{r['seed']}_{r['name']}" for r in old_gate['new_runs']}
                           | {'generation','confirmation','analysis'})
    assert len(predecessor_ids)==21
    old_cpu=read(C+'qualification/cpu01/receipt.json')
    old_cpu_missing=[]
    for r in old_cpu['orchestration_sources']:
        if (root/r['path']).is_file(): assert record(r['path'])==r
        else: old_cpu_missing.append(r['path'])

    provenance=dict(check_mode='stdlib_bytes_JSON_AST_only_no_project_imports_or_tests',
        mechanical_python=sys.version.split()[0], payload_hashes_matched=79,
        run_source_count=len(scientific),orchestration_source_count=len(orchestration),
        cpu_source_count=len(cm),draft_run_sources_match_current=True,
        draft_orchestration_sources_match_current=True,cpu_sources_match_current=True,
        cpu_only_vs_orchestration=['scripts/step28_defaults_cpu.py'],
        native_supports_current_run_sources=True,native_controller_worker_sources_equal=True,
        native_source_count=len(nm),native_present_count=len(present),native_absent_from_this_input=absent,
        native_present_changed=[],scientific_sources=scientific,orchestration_sources=orchestration,
        AST_import_closure_files=sorted(ast_read),exact_18_draft_identities=True,expected_runs=expected_runs,
        predecessor_metadata_identity_match=True,predecessor_expected_complete_ids=predecessor_ids,
        inherited_dispatcher_cpu_present_sources_match=True,
        inherited_dispatcher_cpu_sources_absent_here=old_cpu_missing,
        review_ready_is_null=True,status_intentionally_non_executable=True,
        no_runtime_proof_from_hashes=True)
    (out/'source_identity_checks.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')

    cap=9.25*GIB; slot=10.25*GIB
    race=dict(kind='static_arithmetic_counterexample_not_an_execution',active=1,
        old_free_GiB=16,old_own_GiB=0,new_own_GiB=8,new_actual_free_GiB=8,
        unmaterialized_reserve_GiB=10.25-8,required_GiB=10.25+(10.25-8),
        current_code_would_admit=(16>=10.25+(10.25-8)),
        consistent_snapshot_would_admit=(8>=10.25+(10.25-8)))
    aggregation=dict(kind='conditional_invariant_counterexample_not_observed_usage',active=2,
        own_GiB=[11.25,0],free_GiB=20,
        aggregate_required_GiB=10.25+max(0,2*10.25-11.25),
        per_worker_required_GiB=10.25+sum(max(0,10.25-x) for x in (11.25,0)))
    startup=dict(kind='static_control_flow_counterexample_not_an_execution',
        initial_startup_check_passed=True,first_dispatch_temporarily_waited_for_gpu=True,
        later_ram_GiB=80,later_disk_GiB=100,
        default_refill_threshold_would_admit=True,required_first_launch_threshold_would_admit=False)
    cpu_charge=math.ceil(cpu['wall_seconds']); native_charge=math.ceil(status['charged_seconds'])
    assert cpu_charge==cpu['charged_seconds']==5 and cpu['cumulative_seconds']==56
    assert native_charge==167
    prior=56+171+120
    assert draft['prior_used_seconds']==dict(training=0,control=prior)
    arithmetic=dict(mode='ordinary_arithmetic_only',units=dict(GiB=GIB,MiB=2**20,hour=3600),
        slot_bytes=int(slot),allocator_bytes=int(cap),three_slots_GiB=30.75,
        gpu_total_MiB_from_supplied_snapshot=32607,gpu_total_GiB=32607/1024,
        remaining_MiB_after_three_slots=32607-3*10496,
        gpu_snapshot_free_MiB=11220,one_slot_surplus_MiB_in_old_snapshot=11220-10496,
        old_snapshot_is_not_current_resource_verification=True,
        race_witness=race,aggregate_witness=aggregation,startup_witness=startup,
        runs=18,updates=18*864,gradient_group_presentations=18*(288+576+576),endpoints=18*3,
        primary_blind_dev_group_scores=54*60,calibration_group_scores=54*12,
        endpoint_scoring_for_12cal_plus_60dev_before_and_two_restores=54*(12+60)*3,
        scoring_count_limit='This subtotal excludes training and Memory/teacher forwards; transforms do not add model forwards.',
        training_seconds=18*6*3600,auxiliary_seconds=2*3600,total_seconds=110*3600,
        cpu08_wall_seconds=cpu['wall_seconds'],cpu08_ceiled=cpu_charge,cpu_prior_inferred=56-cpu_charge,
        cpu_current_cumulative=56,cpu_limit=120,cpu_remaining=64,
        native_controller_seconds=status['charged_seconds'],native_controller_ceiled=native_charge,
        native_prior_failed_seconds_from_primary=4,native_current_cumulative=171,native_limit=900,
        native_remaining_not_a_request_for_more_work=729,
        prior_control_breakdown=dict(cpu=56,native=171,preparation_conservative=120),
        prior_control_seconds=prior,script_old_prior_floor_seconds=222,old_floor_gap_seconds=prior-222,
        precharged_control_seconds=60,auxiliary_balance_after_prior_and_precharge=7200-prior-60,
        metadata_and_prior_durations_are_supplied_records_not_remeasured=True,
        inference_file_bytes_proxy=native['inference_bytes'],full_file_bytes_proxy=native['full_bytes'],
        retained_54_inference_plus_3_full_proxy_GiB=(54*native['inference_bytes']+3*native['full_bytes'])/GIB,
        storage_proxy_excludes_memory_scores_logs=True,
        all_max_shape_update_proxy_hours=(288*native['measurements']['current_update']+
                                         576*native['measurements']['replay_update'])/3600,
        max_shape_proxy_is_not_formal_trajectory_time=True,
        capped_query_plus_old_child_termination_seconds={str(a):20*(3-a)+15*a for a in (1,2,3)},
        tail_arithmetic_excludes_other_control_IO_and_is_not_hard_realtime=True,
        confirmed_scientific_blockers=0,project_execution_performed=False)
    (out/'resource_arithmetic.json').write_text(json.dumps(arithmetic,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status='MECHANICAL_CHECKS_MATCH',project_code_executed=False,
                         scientific_sources=44,orchestration_sources=57,cpu_sources=58),ensure_ascii=False))


if __name__=='__main__':
    main()
