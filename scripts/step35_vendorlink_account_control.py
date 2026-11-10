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
from step35_vendorlink_account_policy import PROTOCOL, DOMAINS, CAPS, LIMITS, tasks, select, decide, task_domains

SOURCES = tuple('step35_vendorlink_account_'+n+'.py' for n in ('io', 'policy', 'methods', 'run', 'evaluate', 'control'))+(
    'step29_vendorlink_data.py', 'step29_vendorlink_methods.py', 'step29_vendorlink_run.py',
    'step28_formal_queue.py', 'step30_evolution_policy.py', 'step30_evolution_io.py', 'step30_evolution_control.py')


def read(path):
    return json.loads(Path(path).read_text())


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
        target = root/phase/('training' if action == 'train' else 'scoring')/task['id']
        out.append(dict(action=action, target=target, bucket='training' if action == 'train' else 'gpu',
            seconds=hours*3600 if action == 'train' else 3600, extra=['--phase', phase, '--task', task['id']],
            gpu_family='V' if vendor else 'H',
            reserved_bytes=LIMITS['vendor_reserve_bytes'] if vendor else LIMITS['free_gpu_bytes']))
    return out


def permit(path):
    g = read(path)
    if (sys.platform != 'linux' or Path(sys.executable).resolve() != Path(PYTHON).resolve()
            or g['protocol'] != PROTOCOL or g['approved'] is not True or g['review_closed'] is not True
            or g['caps'] != CAPS or g['limits'] != LIMITS or g['automatic_retry'] is not False
            or g['worker_failure_policy'] != 'retire_oom_continue_current_phase_no_retry'):
        raise ValueError('Reviewed source-bound approved account protocol required')
    expected = {Path(g['workspace']).resolve()/'scripts'/name for name in SOURCES}
    if {Path(b['path']).resolve() for b in g['sources']} != expected:
        raise ValueError('Actual entry and every direct runtime dependency must be bound')
    for b in g['sources']+[g['scope'], g['protocol_source'], g['review_disposition'], g['manifest'], g['cpu'], g['gpu']]:
        pinned(b)
    if Path(__file__).resolve().parent != Path(g['workspace']).resolve()/'scripts':
        raise ValueError('Use the isolated bound workspace')
    if not any(Path(b['path']).resolve() == Path(__file__).resolve() for b in g['sources']):
        raise ValueError('Actual entry must be bound')
    root = Path(g['experiment_root']).resolve()
    if any(not Path(g[k]).resolve().is_relative_to(root) for k in ('workspace', 'campaign')):
        raise ValueError('New outputs must stay inside this experiment')
    if read(g['gpu']['path'])['status'] != 'PASS' or read(g['cpu']['path'])['exit_code'] != 0:
        raise ValueError('Actual necessary CPU/GPU evidence must pass')
    manifest = read(g['manifest']['path'])
    if manifest['status'] != 'QUALIFIED' or any(manifest['counts'][d]['eligible_accounts'] < 128 for d in DOMAINS):
        raise ValueError('One-shot data qualification must pass')
    return g


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
    for b in result['endpoints'].values():
        pinned(b)


def pipeline(run_phase, save, measure=lambda name: nullcontext()):
    d = run_phase('D', None)
    with measure('select_D'):
        d_gate = select(d)
        save('D', d_gate)
        selected = d_gate['selected']
        save('selection', selected)
    if d_gate['status'] != 'PASS_D':
        return dict(status=d_gate['status'], D=d_gate, selected=selected, application=False)
    p = run_phase('P', selected)
    with measure('decide_P'):
        p_gate = decide(p)
        save('P', p_gate)
    if p_gate['status'] != 'PASS_P':
        return dict(status=p_gate['status'], selected=selected, P=p_gate, application=False)
    m = run_phase('M', selected)
    with measure('decide_M'):
        combined = {**p, **m}
        m_gate = decide(combined, True)
        save('M', m_gate)
    if m_gate['status'] != 'PASS_SYSTEM':
        return dict(status=m_gate['status'], selected=selected, P=p_gate, M=m_gate, application=True)
    c = run_phase('C', selected)
    with measure('decide_C'):
        c_gate = decide(c, True)
        c_gate['old_score_increment'] &= m_gate['old_score_increment']
        save('C', c_gate)
    return dict(status='CONFIRMATION_PASS' if c_gate['status'] == 'PASS_SYSTEM' else c_gate['status'],
                selected=selected, P=p_gate, M=m_gate, C=c_gate, application=True)


def worker(g, args):
    from step35_vendorlink_account_run import make_model, execute
    from step35_vendorlink_account_evaluate import score, analyze
    root, campaign = Path(g['experiment_root']), Path(g['campaign'])
    check = Check(root, max(1, args.deadline-time.monotonic()))
    manifest = read(pinned(g['manifest']))
    batch = read(campaign/args.phase/'batch.json')
    if args.phase in ('M', 'C') and read(campaign/'decisions/P.json')['status'] != 'PASS_P':
        raise ValueError('Historical mechanisms are closed until P passes')
    if args.phase == 'C' and read(campaign/'decisions/M.json')['status'] != 'PASS_SYSTEM':
        raise ValueError('Confirmation is closed until M passes')
    expected = tasks(args.phase, batch['selected']) if args.phase != 'C' else tasks('P', batch['selected'])+tasks('M', batch['selected'])
    if batch['tasks'] != expected or batch['manifest'] != g['manifest'] or batch['gate'] != record(args.gate):
        raise ValueError('Frozen batch identity mismatch')
    if args.action in ('train', 'score', 'application'):
        configure_compute()
        for b in g['model_binding']['files']:
            pinned(b)
    if args.action in ('train', 'score'):
        task = next(t for t in expected if t['id'] == args.task)
        origin = campaign/task['phase']/'training'/task['id']/'run'
        identity = dict(protocol=PROTOCOL, task=task, manifest=g['manifest'], gate=record(args.gate))
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
            role = 'confirm' if args.phase == 'C' else 'dev'
            if role == 'confirm' and read(campaign/'decisions/M.json')['status'] != 'PASS_SYSTEM':
                raise ValueError('Confirmation is closed')
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
    root.mkdir(parents=True)
    ledger = Ledger(root, g['prior_seconds'], g['cpu_test_seconds'], caps=CAPS)
    runner = Runner(g, gate_path, root, ledger, worker_entry=Path(__file__), caps=CAPS, limits=LIMITS,
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
            identity = dict(protocol=PROTOCOL, task=task, manifest=g['manifest'], gate=record(gate_path))
            validate_training(task, c, manifest, identity)
            latest = job['target']/'run/latest.pt'
            old = record(latest)
            latest.unlink()  # Explicitly approved retirement after verified compact endpoints.
            write(job['target']/'latest_retired.json', old)
        elif job['action'] in ('score', 'application'):
            if not (job['target']/'result/completion.json').is_file():
                raise ValueError('Missing complete scoring output')
        elif not (job['target']/'result/results.json').is_file():
            raise ValueError('Missing complete analysis')
    def run_phase(phase, selected):
        with measure(phase+'_freeze'):
            specs = tasks(phase, selected) if phase != 'C' else tasks('P', selected)+tasks('M', selected)
            write(root/phase/'batch.json', dict(tasks=specs, selected=selected, manifest=g['manifest'], gate=record(gate_path)))
        if phase != 'C':
            runner.run_many(make_jobs(root, phase, specs, 'train'), 2, complete)
        with measure(phase+'_training_manifest'):
            training = [record(root/t['phase']/'training'/t['id']/'run/completion.json') for t in specs]
            write(root/phase/'training_complete.json', dict(tasks=specs, completions=training))
        runner.run_many(make_jobs(root, phase, specs, 'score'), 2, complete)
        with measure(phase+'_scoring_manifest'):
            scoring = [record(root/phase/'scoring'/t['id']/'result/completion.json') for t in specs]
            write(root/phase/'scoring_complete.json', dict(tasks=specs, completions=scoring))
        job = dict(action='analyze', target=root/phase/'analysis', bucket='auxiliary', seconds=1800,
                   extra=['--phase', phase])
        runner.run_many([job], 1, complete)
        with measure(phase+'_analysis_read'):
            return read(root/phase/'analysis/result/results.json')
    try:
        result = pipeline(run_phase, lambda name, value: write(root/'decisions'/f'{name}.json', value), measure)
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
    p.add_argument('--phase')
    p.add_argument('--task')
    args = p.parse_args()
    started = time.monotonic()
    if args.action == 'campaign':
        g = permit(args.gate)
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
