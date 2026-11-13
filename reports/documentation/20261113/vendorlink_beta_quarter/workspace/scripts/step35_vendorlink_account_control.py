"""Finite reviewed D/P/M/C execution with source, access, and cumulative gates."""
import argparse
from contextlib import contextmanager, nullcontext
import json
import math
from pathlib import Path
import sys
import time

from step30_evolution_control import Runner
from step30_evolution_io import Ledger
from step35_vendorlink_account_io import Check, SharedDiskLimit, PYTHON, pinned, record, write
from step35_vendorlink_account_policy import (PROTOCOL, DOMAINS, CAPS, LIMITS, tasks, decide, task_domains,
    DETERMINISTIC_PROTOCOL, DETERMINISTIC_CAPS, DETERMINISTIC_LIMITS,
    deterministic_tasks, deterministic_comparison, BETA_PROTOCOL, BETA_LIMITS, beta_tasks, beta_comparison)

SOURCES = tuple('step35_vendorlink_account_'+n+'.py' for n in ('io', 'policy', 'methods', 'run', 'evaluate', 'control'))+(
    'step29_vendorlink_data.py', 'step29_vendorlink_methods.py', 'step29_vendorlink_run.py',
    'step28_formal_queue.py', 'step30_evolution_policy.py', 'step30_evolution_io.py', 'step30_evolution_control.py')


def read(path):
    return json.loads(Path(path).read_text())


def phase_tasks(g, selected):
    if g['protocol'] == BETA_PROTOCOL:
        return beta_tasks(selected)
    return (deterministic_tasks(selected) if g['protocol'] == DETERMINISTIC_PROTOCOL else tasks('M', selected))


def configure_compute():
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Deliberately no allocator quota or fixed process-GPU termination threshold.


def make_jobs(root, phase, specs, action):
    out = []
    for task in specs:
        vendor = task['method'] == 'VendorLink_SEQ'
        hours = ((1.5 if vendor else 3) if phase == 'D' else
                 (4 if vendor else (6 if task['method'] == 'H_SEQ' else 10)))
        if task['settings'].get('deterministic_history'):
            hours = 13
        target = root/phase/('training' if action == 'train' else 'scoring')/task['id']
        out.append(dict(action=action, target=target, bucket='training' if action == 'train' else 'gpu',
            seconds=hours*3600 if action == 'train' else 900, extra=['--phase', phase, '--task', task['id']],
            gpu_family='V' if vendor else 'H',
            reserved_bytes=LIMITS['vendor_reserve_bytes'] if vendor else LIMITS['free_gpu_bytes']))
    return out


def permit(path):
    g = read(path)
    deterministic = g['protocol'] in (DETERMINISTIC_PROTOCOL, BETA_PROTOCOL)
    caps, limits = (DETERMINISTIC_CAPS, DETERMINISTIC_LIMITS) if deterministic else (CAPS, LIMITS)
    if g['protocol'] == BETA_PROTOCOL:
        limits = BETA_LIMITS
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve()
            or g['protocol'] not in (PROTOCOL, DETERMINISTIC_PROTOCOL, BETA_PROTOCOL)
            or g['approved'] is not True or g['review_closed'] is not True
            or g['caps'] != caps or g['limits'] != limits or g['automatic_retry'] is not False
            or g['worker_failure_policy'] != 'retire_oom_continue_current_phase_no_retry'):
        raise ValueError('Reviewed source-bound approved account protocol required')
    names = SOURCES + (('step35_vendorlink_beta_wait.py',) if g['protocol'] == BETA_PROTOCOL else ())
    expected = {Path(g['workspace']).resolve()/'scripts'/name for name in names}
    if {Path(b['path']).resolve() for b in g['sources']} != expected:
        raise ValueError('Actual entry and every direct runtime dependency must be bound')
    for b in g['sources']+[g['scope'], g['protocol_source'], g['review_disposition'], g['manifest'], g['cpu'], g['gpu'], g['baseline']]:
        pinned(b)
    if Path(__file__).resolve().parent != Path(g['workspace']).resolve()/'scripts':
        raise ValueError('Use the isolated bound workspace')
    if not any(Path(b['path']).resolve() == Path(__file__).resolve() for b in g['sources']):
        raise ValueError('Actual entry must be bound')
    root = Path(g['experiment_root']).resolve()
    if any(not Path(g[k]).resolve().is_relative_to(root) for k in ('workspace', 'campaign')):
        raise ValueError('New outputs must stay inside this experiment')
    cpu = read(g['cpu']['path'])
    if (read(g['gpu']['path'])['status'] != 'PASS' or cpu['status'] != 'PASS'
            or cpu['tests'] < 1 or cpu['errors'] or cpu['failures']):
        raise ValueError('Actual necessary CPU/GPU evidence must pass')
    manifest = read(g['manifest']['path'])
    if manifest['status'] != 'QUALIFIED' or any(manifest['counts'][d]['eligible_accounts'] < 128 for d in DOMAINS):
        raise ValueError('One-shot data qualification must pass')
    continued_baseline(g)
    if g['protocol'] == BETA_PROTOCOL:
        beta_reference(g)
    return g


def continued_baseline(g):
    """Bind the completed P and its costs without rewriting its failed decision."""
    if g['protocol'] in (DETERMINISTIC_PROTOCOL, BETA_PROTOCOL):
        return deterministic_baseline(g)
    if g.get('continuation_after_p_stop') is not True:
        raise ValueError('Explicit six-task mechanism continuation required')
    prior = g['prior_P']
    values = {k: read(pinned(prior[k])) for k in ('gate', 'batch', 'decision', 'completion', 'budget', 'results')}
    old = values['gate']
    if (values['completion']['status'] != 'STOP_LEARNING'
            or values['decision'] != values['completion']['P']
            or values['decision']['checks'] != dict(learning=False, meaningful_forgetting=False)
            or values['batch']['tasks'] != tasks('P', dict(H='H_CE_RMS', V='V1'))
            or values['batch']['gate'] != prior['gate']
            or old['manifest'] != g['manifest'] or old['model_binding'] != g['model_binding']
            or old['caps'] != g['caps'] or old['limits'] != g['limits']
            or Path(old['experiment_root']).resolve() != Path(g['experiment_root']).resolve()
            or old['campaign'] != g['prior_campaign']
            or Path(g['campaign']).resolve() == Path(old['campaign']).resolve()
            or any(g['prior_seconds'][k] < values['budget']['used_seconds'][k] for k in CAPS)
            or g['baseline'] != prior['results']):
        raise ValueError('Fixed P identity, unchanged configuration and cumulative costs required')
    baseline = values['results']
    if not decide(baseline)['checks']['learning']:
        raise ValueError('The fixed P must meet the newly approved four-point learning requirement')
    return baseline


def deterministic_baseline(g):
    """Read only already-opened P/M evidence; new budget does not erase old costs."""
    p, m = g['prior_P'], g['prior_M']
    pv = {k: read(pinned(p[k])) for k in ('gate', 'batch', 'decision', 'completion', 'budget', 'results')}
    mv = {k: read(pinned(m[k])) for k in ('gate', 'batch', 'completion', 'budget', 'results')}
    selected = dict(H='H_CE_RMS', V='V1')
    if (pv['completion']['status'] != 'STOP_LEARNING'
            or pv['completion']['P'] != pv['decision']
            or pv['batch']['tasks'] != tasks('P', selected) or pv['batch']['gate'] != p['gate']
            or mv['completion']['status'] != 'EXPLORATORY_M_COMPLETE'
            or mv['completion']['confirmation'] is not False
            or mv['batch']['tasks'] != tasks('M', selected) or mv['batch']['gate'] != m['gate']
            or mv['gate']['prior_P'] != p or g['baseline'] != p['results']
            or any(v['gate']['manifest'] != g['manifest'] or v['gate']['model_binding'] != g['model_binding']
                   for v in (pv, mv))
            or any(Path(v['gate']['experiment_root']).resolve() == Path(g['experiment_root']).resolve()
                   for v in (pv, mv))):
        raise ValueError('Completed unchanged P/M and identical scientific inputs required')
    baseline = {**pv['results'], **mv['results']}
    if decide(baseline, True) != mv['completion']['M']:
        raise ValueError('Old M decision or complete baseline differs')
    return baseline


def beta_reference(g):
    prior = g['prior_deterministic']
    v = {k: read(pinned(prior[k])) for k in ('gate', 'batch', 'completion', 'budget', 'results')}
    old = v['gate']
    if (old['protocol'] != DETERMINISTIC_PROTOCOL or old['prior_P'] != g['prior_P']
            or old['prior_M'] != g['prior_M'] or old['manifest'] != g['manifest']
            or old['model_binding'] != g['model_binding']
            or v['batch']['tasks'] != deterministic_tasks(dict(H='H_CE_RMS', V='V1'))
            or v['batch']['gate'] != prior['gate']
            or Path(old['experiment_root']).resolve() == Path(g['experiment_root']).resolve()
            or v['completion'] != deterministic_comparison(deterministic_baseline(g), v['results'])):
        raise ValueError('Complete deterministic reference and identical fixed scientific inputs required')
    return v['results']


def validate_training(task, result, manifest, identity):
    if (result['task'] != task or result['identity'] != identity
            or result['status'] != 'TRAINING_COMPLETE_EVALUATION_CLOSED'
            or set(result['endpoints']) != {str(i) for i in range(task['stages']+1)}):
        raise ValueError('Training identity/endpoints incomplete')
    ads = sum(manifest['counts'][d]['eligible_accounts']*12*task['epochs'] for d in task_domains(task))
    from step29_vendorlink_data import group_chunks
    updates = sum((((manifest['counts'][d]['eligible_accounts']*12+31)//32) if task['method'] == 'VendorLink_SEQ'
                   else 3*len(group_chunks(list(range(manifest['counts'][d]['eligible_accounts'])), 8)))
                  *task['epochs'] for d in task_domains(task))
    if result['updates'] != updates or result['workload']['current_ads'] != ads:
        raise ValueError('Actual exposure/update count incomplete')
    if task['method'] in ('H_ER', 'H_full'):
        history_updates = sum(3*len(group_chunks(list(range(manifest['counts'][d]['eligible_accounts'])), 8))
                              *task['epochs'] for d in task_domains(task)[1:])
        if (result['workload']['history_groups'] != history_updates
                or result['workload']['history_ads'] <= 0 or result['workload']['teacher_groups'] <= 0):
            raise ValueError('Every later-domain update must actually consume retained history')
        if (task['settings'].get('deterministic_history') and
                result['workload'].get('deterministic_history_groups') != history_updates):
            raise ValueError('Every history update must execute the deterministic birth branch')
    for b in result['endpoints'].values():
        pinned(b)
    if task['settings'].get('diagnostics'):
        if set(result.get('diagnostics', {})) != {str(i) for i in range(1, task['stages']+1)}:
            raise ValueError('Diagnostic archive missing before latest retirement')
        for b in result['diagnostics'].values():
            pinned(b)


def pipeline(run_phase, save, baseline, measure=lambda name: nullcontext(), deterministic=False, reference=None):
    admission = decide(baseline)
    if not admission['checks']['learning']:
        raise ValueError('Fixed completed P learning requirement failed')
    selected = dict(H='H_CE_RMS', V='V1')
    with measure('fixed_selection'):
        save('selection', selected)
        save('continuation_basis', dict(learning_gain_min=.04, original_P_unchanged=True,
             baseline_checks=admission['checks'], meaningful_forgetting_required_for_this_M=False))
    m = run_phase('M', selected)
    if reference is not None:
        with measure('beta_comparison'):
            result = beta_comparison(baseline, reference, m)
            save('beta_comparison', result)
            save('comparison_results', dict(baseline=baseline, reference=reference, candidate=m))
        return result
    if deterministic:
        with measure('deterministic_comparison'):
            result = deterministic_comparison(baseline, m)
            save('deterministic_comparison', result)
            save('comparison_results', dict(baseline=baseline, candidate=m))
        return result
    with measure('decide_M'):
        combined = {**baseline, **m}
        m_gate = decide(combined, True)
        save('M', m_gate)
        save('comparison_results', combined)
    return dict(status='EXPLORATORY_M_COMPLETE', selected=selected, M=m_gate,
                substantive_checks={k: v for k, v in m_gate['checks'].items() if k != 'meaningful_forgetting'},
                original_P_unchanged=True, confirmation=False, application=True)


def worker(g, args):
    from step35_vendorlink_account_run import make_model, execute
    from step35_vendorlink_account_evaluate import score, analyze
    root, campaign = Path(g['experiment_root']), Path(g['campaign'])
    check = Check(root, max(1, args.deadline-time.monotonic()), g['limits'])
    manifest = read(pinned(g['manifest']))
    if args.phase != 'M':
        raise ValueError('This continuation only runs M; P is reused and confirmation remains closed')
    batch = read(campaign/args.phase/'batch.json')
    expected = phase_tasks(g, batch['selected'])
    if batch['tasks'] != expected or batch['manifest'] != g['manifest'] or batch['gate'] != record(args.gate):
        raise ValueError('Frozen batch identity mismatch')
    if args.action in ('train', 'score', 'application'):
        configure_compute()
        for b in g['model_binding']['files']:
            pinned(b)
    if args.action in ('train', 'score'):
        task = next(t for t in expected if t['id'] == args.task)
        origin = campaign/task['phase']/'training'/task['id']/'run'
        identity = dict(protocol=g['protocol'], task=task, manifest=g['manifest'], gate=record(args.gate))
        if args.action == 'train':
            requested = []
            def load_fit(domain):
                if domain != task_domains(task)[len(requested)]:
                    raise ValueError('Only the current market may be loaded')
                b = manifest['files'][domain+':fit']
                write(args.target/f'access_stage{len(requested)+1}.json', dict(domain=domain, source=b))
                requested.append(domain)
                return read(pinned(b))
            if args.target/'run' != origin:
                raise ValueError('Wrong fresh training destination')
            result = execute(make_model(g['model_binding'], task), load_fit, origin, task, identity, g['model_binding'], check)
            validate_training(task, result, manifest, identity)
        else:
            if args.target != campaign/args.phase/'scoring'/task['id']:
                raise ValueError('Wrong one-shot scoring destination')
            # The phase freeze proves every required trajectory finished before scoring.
            frozen = read(campaign/args.phase/'training_complete.json')
            if frozen['tasks'] != expected:
                raise ValueError('Incomplete whole training batch')
            for b in frozen['completions']:
                pinned(b)
            completion = read(origin/'completion.json')
            role = 'dev'
            validate_training(task, completion, manifest, identity)
            score(task, completion, manifest, g['model_binding'], args.target/'result', role, check)
    elif args.action == 'analyze':
        if args.target != campaign/args.phase/'analysis':
            raise ValueError('Wrong one-shot result release destination')
        completed = read(campaign/args.phase/'scoring_complete.json')
        if completed['tasks'] != expected:
            raise ValueError('Incomplete whole scoring batch')
        analyze([read(pinned(b)) for b in completed['completions']], expected, args.target/'result')
    elif args.action == 'application':
        if g['protocol'] in (DETERMINISTIC_PROTOCOL, BETA_PROTOCOL):
            raise ValueError('This intervention does not authorize another application batch')
        if args.target != campaign/'application':
            raise ValueError('Wrong application destination')
        from step35_vendorlink_account_evaluate import application
        if not read(campaign/'quantitative_completion.json')['application']:
            raise ValueError('No Full application output exists for a P stop')
        application(g, batch['selected'], args.target/'result', check)
    else:
        raise ValueError('Unknown action')


def campaign(g, gate_path, started=None):
    started = time.monotonic() if started is None else started
    root = Path(g['campaign'])
    if root.exists():
        raise ValueError('No duplicate controller or automatic campaign restart')
    prior_seconds = dict(g['prior_seconds'])
    if g['protocol'] == BETA_PROTOCOL:
        from step35_vendorlink_beta_wait import dependency_observation
        ready = read(Path(g['experiment_root'])/'wait'/'ready.json')
        if (ready['gate'] != record(gate_path) or ready['status'] != 'READY'
                or not dependency_observation(g)['ready']):
            raise ValueError('Chinese experiment must finish before the unique waiting controller starts training')
        prior_seconds['auxiliary'] += ready['charged_seconds']
    root.mkdir(parents=True)
    ledger = Ledger(root, prior_seconds, g['cpu_test_seconds'], caps=g['caps'])
    runner = Runner(g, gate_path, root, ledger, worker_entry=Path(__file__), caps=g['caps'], limits=g['limits'],
                    halt_dispatch_on_failure=True, continue_after_oom=True)
    manifest = read(pinned(g['manifest']))
    ledger.charge('auxiliary', 'controller_setup', time.monotonic()-started, True)
    @contextmanager
    def measure(name):
        # Only small parent segments; never wrap worker execution or waiting.
        ledger.reserve('auxiliary', 60)
        with ledger.active(name):
            yield
    def complete(job):
        if job['action'] == 'train':
            c = read(job['target']/'run/completion.json')
            task = c['task']
            identity = dict(protocol=g['protocol'], task=task, manifest=g['manifest'], gate=record(gate_path))
            validate_training(task, c, manifest, identity)
            latest = job['target']/'run/latest.pt'
            old = record(latest)
            latest.unlink()  # Explicitly approved retirement after verified compact endpoints.
            write(job['target']/'latest_retired.json', old)
        elif job['action'] in ('score', 'application'):
            if not (job['target']/'result/completion.json').is_file():
                raise ValueError('Missing complete scoring output')
            if job['action'] == 'score':
                from step35_vendorlink_account_evaluate import validate_scores
                validate_scores(read(job['target']/'result/completion.json'))
        elif not (job['target']/'result/results.json').is_file():
            raise ValueError('Missing complete analysis')
    def run_phase(phase, selected):
        if phase != 'M':
            raise ValueError('Only the six fixed mechanism tasks may be dispatched')
        with measure(phase+'_freeze'):
            specs = phase_tasks(g, selected)
            write(root/phase/'batch.json', dict(tasks=specs, selected=selected, manifest=g['manifest'], gate=record(gate_path)))
        runner.run_many(make_jobs(root, phase, specs, 'train'), g['limits']['workers'], complete)
        with measure(phase+'_training_manifest'):
            training = [record(root/t['phase']/'training'/t['id']/'run/completion.json') for t in specs]
            write(root/phase/'training_complete.json', dict(tasks=specs, completions=training))
        runner.run_many(make_jobs(root, phase, specs, 'score'),
                        1 if g['protocol'] in (DETERMINISTIC_PROTOCOL, BETA_PROTOCOL) else 2, complete)
        with measure(phase+'_scoring_manifest'):
            scoring = [record(root/phase/'scoring'/t['id']/'result/completion.json') for t in specs]
            write(root/phase/'scoring_complete.json', dict(tasks=specs, completions=scoring))
        job = dict(action='analyze', target=root/phase/'analysis', bucket='auxiliary', seconds=600,
                   extra=['--phase', phase])
        runner.run_many([job], 1, complete)
        with measure(phase+'_analysis_read'):
            return read(root/phase/'analysis/result/results.json')
    try:
        result = pipeline(run_phase, lambda name, value: write(root/'decisions'/f'{name}.json', value),
                          continued_baseline(g), measure, deterministic=g['protocol'] == DETERMINISTIC_PROTOCOL,
                          reference=beta_reference(g) if g['protocol'] == BETA_PROTOCOL else None)
        with measure('quantitative_completion'):
            write(root/'quantitative_completion.json', result)
        if result['application']:
            # Reuse M's frozen specs/selection; no new fit, labels, or training.
            job = dict(action='application', target=root/'application', bucket='gpu', seconds=3600,
                       extra=['--phase', 'M'], gpu_family='V', reserved_bytes=LIMITS['vendor_reserve_bytes'])
            runner.run_many([job], 1, complete)
        with measure('completion'):
            write(root/'completion.json', result)
    except BaseException as e:
        write(root/'failure.json', dict(error=repr(e), automatic_retry=False,
              released_phases=[p for p in ('D', 'P', 'M', 'C') if (root/p/'analysis/result/results.json').exists()]))
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gate', type=Path, required=True)
    p.add_argument('--action', choices=['campaign', 'train', 'score', 'analyze', 'application'], required=True)
    p.add_argument('--target', type=Path)
    p.add_argument('--deadline', type=float)
    p.add_argument('--phase', choices=('M',))
    p.add_argument('--task')
    args = p.parse_args()
    started = time.monotonic()
    if args.action == 'campaign':
        g = permit(args.gate)
        if g['protocol'] == BETA_PROTOCOL:
            raise ValueError('Start this batch only through its source-bound once_wait.py')
        campaign(g, args.gate, started)
    else:
        # Bind the marker destination to this isolated deployment before permit:
        # source failures inside permit must reach the parent as shared failures.
        expected_campaign = Path(__file__).resolve().parents[2]/'campaign'
        if args.target is None or not args.target.resolve().is_relative_to(expected_campaign):
            raise ValueError('Worker target outside approved campaign')
        try:
            g = permit(args.gate)
        except (ValueError, OSError) as e:
            write(args.target/'shared_failure.json', dict(error=repr(e), status='SOURCE_VALIDITY_FAILURE'))
            raise
        try:
            worker(g, args)
        except (ValueError, SharedDiskLimit) as e:
            status = 'SHARED_DISK_FAILURE' if isinstance(e, SharedDiskLimit) else 'DATA_OR_PROTOCOL_VALIDITY_FAILURE'
            write(args.target/'shared_failure.json', dict(error=repr(e), status=status))
            raise
        except RuntimeError as e:
            import torch
            oom = isinstance(e, torch.cuda.OutOfMemoryError) or 'CUDA out of memory' in str(e)
            write(args.target/'worker_failure.json', dict(kind='CUDA_OOM' if oom else 'OTHER',
                  error=repr(e), automatic_retry=False))
            raise


if __name__ == '__main__':
    main()
