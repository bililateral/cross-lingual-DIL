"""Sequential, three-seed FIT12 development for the shared last-block FFN."""
from pathlib import Path

from step30_evolution_io import read, record, verify, write
from step35_market_queue import train_block
from step35_vendorlink_account_policy import CELLS, DOMAINS, SEEDS, decide


PROTOCOL = 'vendorlink-fit12-shared-ffn-staged'
METHODS = ('H_SEQ', 'H_ER', 'H_full')
CAPS = dict(training=102*3600, gpu=4*3600, auxiliary=4*3600)
REFERENCE_SHA = '954213a120a2a799c76f4736f33b6974c92d158fb1fa58624148e9b7f22d9305'


def batch_tasks(tasks, method):
    if method not in METHODS:
        raise ValueError('Unknown shared FFN stage')
    batch = [t for t in tasks if t['method'] == method]
    if [t['id'] for t in batch] != [f'{method}_s{s}' for s in SEEDS] or [t['seed'] for t in batch] != list(SEEDS):
        raise ValueError('A stage requires seeds 51, 52 and 53 in order')
    return batch


def vendor_reference(g):
    binding = g['vendorlink_reference']
    if binding['sha256'] != REFERENCE_SHA:
        raise ValueError('Expected the frozen four-arm FIT12 reference')
    ref = read(verify(binding))
    previous = read(verify(ref['gate']))
    closed = read(verify(ref['completion']))
    if (g['manifest'] != previous['manifest'] or g['model_binding'] != previous['model_binding']
            or closed['status'] != 'COMPLETE' or closed['dev_opening'] != 'whole_batch_once'
            or closed['confirm_open'] is not False):
        raise ValueError('Reference requires the matching completed FIT12 development protocol')
    aggregate = read(verify(ref['results']))
    readers = {m: aggregate[m] for m in ref['readers']}
    if any([r['seed'] for r in rows] != list(SEEDS) for rows in readers.values()):
        raise ValueError('VendorLink reader seeds differ')
    return readers, previous, ref


def validate_gate(g, sources, gate):
    from step35_market_adapters import SHARED_FFN_FORMAT
    if Path(__file__).resolve() not in sources or Path(g['experiment_root']).resolve() != gate.parent:
        raise ValueError('Bind the staged dispatcher and its independent experiment directory')
    _, previous, _ = vendor_reference(g)
    if Path(previous['campaign']).resolve() == Path(g['campaign']).resolve():
        raise ValueError('The previous campaign is closed')
    script_dir = Path(__file__).resolve().parent
    previous_sources = {Path(b['path']).name: b['sha256'] for b in previous['sources']}
    if any(script_dir/name not in sources for name in previous_sources):
        raise ValueError('Missing a bound FIT12 dependency from the staged deployment')
    bound_sources = {Path(b['path']).name: b['sha256'] for b in g['sources']}
    for name in ('step35_vendorlink_account_evaluate.py', 'step35_vendorlink_account_policy.py'):
        if bound_sources[name] != previous_sources[name]:
            raise ValueError('The reused readers require the identical FIT12 evaluator and metric policy')
    qualified = g['gpu_qualification']
    if qualified not in g['qualification']:
        raise ValueError('Missing candidate CUDA qualification')
    gpu = read(verify(qualified))
    model_sources = {Path(b['path']).name: b['sha256'] for b in g['sources']
                     if Path(b['path']).name in ('step35_market_adapters.py', 'step35_vendorlink_account_run.py')}
    if (gpu['status'] != 'PASS' or gpu['device'] != 'cuda' or gpu['market_adapters'] != SHARED_FFN_FORMAT
            or len(model_sources) != 2 or gpu['model_sources'] != model_sources):
        raise ValueError('CUDA qualification must match the candidate model sources')


def bound_at(binding, path):
    if Path(binding['path']) != path:
        raise ValueError('Evidence belongs to a different task, stage or attempt')
    return verify(binding)


def succeeded(path):
    receipt = read(path)
    if receipt['success'] is not True or receipt['released'] is not True:
        raise ValueError('A successful, released job is required')
    return receipt


def check_winners(root, frozen, tasks, gate_id, manifest_binding):
    winners = frozen['winners']
    if frozen['tasks'] != tasks or set(winners) != {t['id'] for t in tasks}:
        raise ValueError('All three fixed training winners are required before scoring')
    for task in tasks:
        name = task['id']
        winner = winners[name]
        attempt = winner['attempt']
        if attempt not in (1, 2):
            raise ValueError('Unknown training attempt')
        job = root/'training'/name/f'attempt-{attempt}'
        receipt = succeeded(bound_at(winner['receipt'], job/'dispatch_receipt.json'))
        if (receipt['task_id'], receipt['attempt']) != (name, attempt):
            raise ValueError('Training receipt differs from the winning task')
        if attempt == 2:
            first_path = job.parent/'attempt-1'/'dispatch_receipt.json'
            first = read(bound_at(winner['first_oom_receipt'], first_path))
            reopen = read(first_path.parent/'slot_reopened.json')
            if ((first['task_id'], first['attempt'], first['success'], first['released'], first['failure_kind'],
                    first['slot']) != (name, 1, False, True, 'CUDA_OOM', receipt['slot'])
                    or reopen['previous']['receipt'] != winner['first_oom_receipt']
                    or (reopen['next_task'], reopen['next_attempt']) != (name, 2)):
                raise ValueError('Retry lacks a matching released first OOM and reopening')
        completion = read(bound_at(winner['completion'], job/'run/completion.json'))
        identity = dict(protocol=PROTOCOL, task=task, manifest=manifest_binding, gate=gate_id)
        if (completion['task'] != task or completion['identity'] != identity
                or completion['status'] != 'TRAINING_COMPLETE_EVALUATION_CLOSED'):
            raise ValueError('Wrong training identity or incomplete training')


def require_prior(root, method, gate_id):
    """Read only controller-issued permissions. Training must never open prior development values."""
    permits = []
    for index in range(METHODS.index(method)):
        previous, following = METHODS[index:index+2]
        directory = root/'stages'/previous
        stage = read(directory/'closed.json')
        if stage['gate'] != gate_id or stage['method'] != previous:
            raise ValueError('Prior stage belongs to another campaign')
        if stage['status'] != 'PASS_'+previous or stage['next_method'] != following:
            raise ValueError('The prior stage did not authorize its successor')
        permits.append(stage)
    return permits


def stage_decision(method, results, historical=None):
    from step35_market_worker import er_pressure, paired_report
    if method == 'H_SEQ':
        original = decide(results)
        success = original['status'] == 'PASS_P'
        return dict(method=method, checks=original['checks'], next_method='H_ER' if success else None,
                    status='PASS_H_SEQ' if success else original['status'])
    if method == 'H_ER':
        seq, er = results['H_SEQ'], results['H_ER']
        checks = er_pressure(er)
        same_start = all(a[k] == b[k] for a,b in zip(seq, er, strict=True) for k in ('R01','R11','R12'))
        if not same_start:
            status = 'HOLD_PREHISTORY_IDENTITY'
        elif not checks['er_learning']:
            status = 'STOP_ER_LEARNING'
        elif not checks['er_forgetting']:
            status = 'STOP_ER_FORGETTING'
        else:
            status = 'PASS_H_ER'
        return dict(method=method, checks=checks, prehistory_integrity=same_start, status=status,
                    next_method='H_full' if status == 'PASS_H_ER' else None)
    if method != 'H_full':
        raise ValueError('Unknown stage decision')
    final = paired_report(results, historical)
    status = ('HOLD_PREHISTORY_IDENTITY' if final['full_application'] is None else
              'PASS_FULL_APPLICATION' if final['full_application'] else 'STOP_FULL_APPLICATION')
    return dict(method=method, status=status, next_method=None, **final)


def check_scores(root, task, score, manifest, gate_id, manifest_binding):
    from step35_vendorlink_account_evaluate import validate_scores
    identity = dict(protocol=PROTOCOL, task=task, manifest=manifest_binding, gate=gate_id)
    expected_sources = {d+':'+r: manifest['files'][d+':'+r] for d in DOMAINS for r in ('dev','fit')}
    if (score['task'] != task or score['training_identity'] != identity or score['role'] != 'dev'
            or score['sources'] != expected_sources):
        raise ValueError('Wrong development score identity or role sources')
    validate_scores(score)
    import numpy as np
    for t, j in CELLS:
        cell = f'R{t}{j}'
        binding = score['files'][task['method']+':'+cell]
        path = root/'scoring'/task['id']/'result'/f"{task['method']}_{cell}.npy"
        if Path(binding['path']) != path:
            raise ValueError('Score matrix does not belong to its fixed task and cell')
        n = manifest['counts'][DOMAINS[j-1]]['eligible_accounts']
        if np.load(path, mmap_mode='r', allow_pickle=False).shape != (n, n):
            raise ValueError('Incomplete Q2/FIT12 candidate population')


def analyze_stage(root, method, tasks, g, gate_id, manifest):
    from step35_vendorlink_account_evaluate import analyze
    from step35_market_worker import market_tasks, write_metric_tables
    directory = root/'stages'/method
    dest = directory/'analysis/result'
    batch = read(directory/'scoring_complete.json')
    if batch['tasks'] != tasks or len(batch['completions']) != len(tasks):
        raise ValueError('All three blind score sets are required before development opening')
    completed = []
    for task, binding in zip(tasks, batch['completions'], strict=True):
        job = root/'scoring'/task['id']
        succeeded(job/'dispatch_receipt.json')
        score = read(bound_at(binding, job/'result/completion.json'))
        check_scores(root, task, score, manifest, gate_id, g['manifest'])
        completed.append(score)
    earlier, metric_bindings = {}, {}
    for permit in require_prior(root, method, gate_id):
        previous = permit['method']
        saved = root/'stages'/previous/'analysis/result'
        prior_policy = read(bound_at(permit['decision'], saved/'decision.json'))
        if any(prior_policy[k] != permit[k] for k in ('method','status','next_method')):
            raise ValueError('Published permission and saved development decision differ')
        earlier[previous] = read(bound_at(permit['results'], saved/'results.json'))[previous]
        metric_bindings[previous] = permit['metrics']
        bound_at(permit['metrics'], saved/'metrics.json')
    readers, original, ref = vendor_reference(g)
    # analyze creates a fresh result directory and opens precisely these three dev sets once.
    current = analyze(completed, tasks, dest)
    results = {**readers, **earlier, **current}
    historical = None
    if method == 'H_full':
        historical = {k: read(verify(r['binding']))[r['method']] for k,r in original['historical'].items()}
        vendor_tasks = [t for t in market_tasks() if t['method'] == 'VendorLink_SEQ']
        vendor_ids = {t['id'] for t in vendor_tasks}
        metrics = {k:v for k,v in read(verify(ref['metrics'])).items() if k.split(':',1)[0] in vendor_ids}
        for binding in metric_bindings.values():
            metrics.update(read(verify(binding)))
        metrics.update(read(dest/'metrics.json'))
        write_metric_tables(vendor_tasks+market_tasks(shared_ffn=True), metrics, dest)
    decision = stage_decision(method, results, historical)
    write(dest/'decision.json', decision)


def dispatch(runner, tasks, completed, gate_id):
    from step35_market_worker import market_tasks, validate_streams
    if tasks != market_tasks(shared_ffn=True):
        raise ValueError('Only the nine fixed shared-FFN H trajectories may be dispatched')
    root, g = runner.root, runner.allowed
    winners, finished = {}, {}
    for index, method in enumerate(METHODS):
        require_prior(root, method, gate_id)
        group = batch_tasks(tasks, method)
        directory = root/'stages'/method
        directory.mkdir(parents=True, exist_ok=False)
        batch = dict(tasks=group, winners=train_block(runner, group, 3, completed))
        check_winners(root, batch, group, gate_id, g['manifest'])
        winners.update(batch['winners'])
        validate_streams(winners, METHODS[:index+1])
        write(directory/'training_complete.json', batch)
        scores = [dict(action='score', task_id=t['id'], target=root/'scoring'/t['id'], bucket='gpu',
                       seconds=1200, gpu_family='H', reserved_bytes=runner.limits['free_gpu_bytes']) for t in group]
        manifest = read(verify(g['manifest']))
        by_id = {t['id']: t for t in group}
        def scored(job):
            score = read(job['target']/'result/completion.json')
            check_scores(root, by_id[job['task_id']], score, manifest, gate_id, g['manifest'])
        runner.run_many(scores, 1, scored)
        for job in scores:
            succeeded(job['target']/'dispatch_receipt.json')
        write(directory/'scoring_complete.json', dict(tasks=group,
              completions=[record(j['target']/'result/completion.json') for j in scores]))
        analysis = directory/'analysis'
        result = analysis/'result'
        def analyzed(job):
            decision = read(result/'decision.json')
            following = METHODS[index+1] if index+1 < len(METHODS) and decision['status'] == 'PASS_'+method else None
            if decision['method'] != method or decision['next_method'] != following:
                raise ValueError('Wrong stage decision')
        runner.run_many([dict(action='analyze', target=analysis, bucket='auxiliary', seconds=600,
                              operation='analyze_'+method)], 1, analyzed)
        succeeded(analysis/'dispatch_receipt.json')
        decision = read(result/'decision.json')
        write(directory/'closed.json', dict(method=method, gate=gate_id, status=decision['status'],
              next_method=decision['next_method'], decision=record(result/'decision.json'),
              results=record(result/'results.json'), metrics=record(result/'metrics.json')))
        finished[method] = record(directory/'closed.json')
        if decision['next_method'] is None:
            break
    write(root/'completion.json', dict(status=decision['status'], stages=finished, decisions_by_stage=True,
                                      dev_opening='once_per_complete_three_seed_stage', confirm_open=False))
