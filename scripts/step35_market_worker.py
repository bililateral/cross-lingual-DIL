"""Fixed FIT12 market-adapter training, blind scoring and paired development policy."""
import argparse
import os
from pathlib import Path
import statistics
import time

from step30_evolution_io import read, record, verify, write
from step35_market_queue import H_ORDER, PROTOCOL, V_ORDER
from step35_vendorlink_account_policy import BETA, DOMAINS, SEEDS, decide, settings


def market_tasks(*, shared_ffn=False):
    from step35_market_adapters import FORMAT, SHARED_FFN_FORMAT
    result = []
    order = ([(method, seed) for method in ('H_SEQ', 'H_ER', 'H_full') for seed in SEEDS]
             if shared_ffn else V_ORDER+H_ORDER)
    for method, seed in order:
        vendor = method == 'VendorLink_SEQ'
        candidate = 'V1' if vendor else 'H_CE_RMS'
        hyper = settings(candidate, method)
        task = dict(id=f'{method}_s{seed}', method=method, seed=seed, epochs=40, stages=3,
                    phase='M', domains=list(DOMAINS), candidate=candidate, settings=hyper, stage_end_only=True)
        if not vendor:
            task['market_adapters'] = SHARED_FFN_FORMAT if shared_ffn else FORMAT
            hyper.update(beta=BETA, deterministic_history=method == 'H_full')
        result.append(task)
    return result


def validate_streams(winners, methods=('H_SEQ', 'H_ER', 'H_full')):
    for seed in SEEDS:
        rows = {m: read(verify(winners[f'{m}_s{seed}']['completion'])) for m in methods}
        seq = rows['H_SEQ']
        if (len({row['current_stream'] for row in rows.values()}) != 1 or seq['survivors']
                or seq['workload']['history_groups'] or seq['workload']['teacher_groups']):
            raise ValueError('Unpaired current groups or unintended SEQ replay')
        if 'H_ER' in rows:
            er = rows['H_ER']
            if set(er['survivors']) != set(DOMAINS[:2]) or any(len(set(v)) != 6 for v in er['survivors'].values()):
                raise ValueError('Incomplete six-group replay state')
        if 'H_full' in rows:
            full = rows['H_full']
            if er['history_stream'] != full['history_stream'] or er['survivors'] != full['survivors']:
                raise ValueError('Unpaired Full and ER replay')
        for row in rows.values():
            sizes = row['history_bytes']
            if set(sizes) != {'end_1','enter_2','end_2','enter_3','end_3'} or not all(0 < n <= 2**20 for n in sizes.values()):
                raise ValueError('Incomplete or oversized historical state')


def er_pressure(rows):
    def values(key):
        return [r[key] for r in rows]
    absolute = all(statistics.mean(values(k)) >= .70 and min(values(k)) >= .60 for k in ('R11','R22','R33'))
    gain = all(statistics.mean(values(k)) >= .04 and min(values(k)) >= 0 for k in ('GB','GC'))
    return dict(er_learning=absolute and gain,
                er_forgetting=statistics.mean(values('F')) >= .02 and sum(x > 0 for x in values('F')) >= 2)


def paired_report(results, historical):
    mean = statistics.mean
    ref = decide(results, mechanism=True)
    full = results['H_full']
    def calc(rows, key):
        return [r[key] for r in rows]
    def delta(rows, key):
        return [a[key]-b[key] for a,b in zip(full, rows, strict=True)]
    def guard(rows, keys, average, single):
        return all(mean(v) >= average and min(v) >= single for v in (delta(rows,k) for k in keys))
    def absolute(rows):
        return all(mean(calc(rows,k)) >= .70 and min(calc(rows,k)) >= .60 for k in ('R11','R22','R33'))
    def gain(rows):
        return all(mean(calc(rows,k)) >= .04 and min(calc(rows,k)) >= 0 for k in ('GB','GC'))
    er, seq = results['H_ER'], results['H_SEQ']
    old, forget = delta(er,'O'), [-x for x in delta(er,'F')]
    gain_protection = {}
    for method, rows in (('H_ER', er), ('H_SEQ', seq)):
        for domain, g, p, a in (('Dream','GB','R12','R22'), ('Agora','GC','R23','R33')):
            net, start, end = [mean(delta(rows,k)) for k in (g,p,a)]
            gain_protection[method+'/'+domain] = dict(net=net, start=start, end=end,
                                                      passed=net >= -.02 or (start > 0 and end >= 0))
    guards = dict(effect=mean(old) >= .02 and mean(forget) >= .02,
        paired_direction=sum(o>0 and f>0 for o,f in zip(old,forget)) >= 2,
        worst_seed=min(old+forget) >= -.01, each_old_final=all(mean(delta(er,k)) > 0 for k in ('R31','R32')),
        acquisition=all(guard(r,('R11','R22'),-.01,-.03) for r in (er,seq)),
        absolute=absolute(full), new_gain=gain(full),
        new_final=all(guard(r,('R22','R33'),-.02,-.05) for r in (er,seq)),
        relative_gain=all(item['passed'] for item in gain_protection.values()))
    pressure = dict(seq_learning=ref['checks']['learning'], seq_forgetting=ref['checks']['meaningful_forgetting'],
                    **er_pressure(er))
    extern = (ref['checks']['external'] and guard(results['V_CLS'],('R11','R22','R33'),-.02,-.05)
              and guard(results['V_SET'],('R22','R33'),-.02,-.05))
    joint, scenario = all(guards.values()), all(pressure.values())
    internal = (ref['checks']['retention'] and guard(seq,('R11','R22'),-.01,-.03)
                and guard(seq,('R22','R33'),-.02,-.05) and absolute(full) and gain(full)
                and all(v['passed'] for k,v in gain_protection.items() if k.startswith('H_SEQ/')))
    def summary(values):
        return dict(seeds=list(SEEDS), values=values, mean=mean(values), sd=statistics.stdev(values),
                    minimum=min(values), maximum=max(values))
    summaries, paired = {}, {}
    for name, rows in {**results, **historical}.items():
        if [r['seed'] for r in rows] != list(SEEDS):
            raise ValueError('Expected the three fixed paired development seeds')
        decorated = [dict(r, FA=r['R11']-r['R31'], FB=r['R22']-r['R32']) for r in rows]
        summaries[name] = {k:summary(calc(decorated,k)) for k in decorated[0] if k != 'seed'}
        if name != 'H_full':
            paired[name] = {k:summary([s*t for t in delta(rows,k)]) for k,s in
                            [(key,-1 if key=='F' else 1) for key in full[0] if key!='seed']}
            for k,a,b in (('FA','R11','R31'),('FB','R22','R32')):
                paired[name][k] = summary([(r[a]-r[b])-(f[a]-f[b]) for r,f in zip(rows,full)])
    prehistory={m:{key:delta(results[m],key) for key in ('R01','R11','R12')} for m in ('H_SEQ','H_ER')}
    integrity=all(d==0 for cells in prehistory.values() for values in cells.values() for d in values)
    scientific=joint and scenario and internal and extern
    return dict(full_er=joint, full_er_checks=guards, new_learning=gain_protection, pressure=pressure,
                meaningful_scenario=scenario, full_seq=internal, external=extern,
                scientific_gates_joint=scientific,full_application=scientific if integrity else None,
                prehistory_integrity=integrity,interpretation_requires_identity_review=not integrity,
                prehistory_H_macro_f1_differences=prehistory,original_fit12=ref,
                summaries=summaries, full_minus_comparator=paired,
                difference_convention='F, FA and FB are comparator minus Full; abilities are Full minus comparator')


def worker(gate, action, target, name, deadline):
    from step30_evolution_control import kernel_identity
    from step35_vendorlink_account_control import configure_compute, validate_training
    from step35_vendorlink_account_evaluate import analyze, score, validate_scores
    from step35_vendorlink_account_io import Check
    from step35_vendorlink_account_run import make_model, execute
    g = read(gate)
    root = Path(g['campaign'])
    staged = g['protocol'] != PROTOCOL
    if staged:
        import step35_ffn_stages as stages
    if (gate.resolve() != Path(__file__).resolve().parents[2]/'formal_gate.json'
            or root != gate.parent/'campaign' or g['training_authorized'] is not True or g['review_closed'] is not True
            or (staged and g['protocol'] != stages.PROTOCOL)):
        raise ValueError('Wrong deployment or authorization')
    controller = read(root/'controller.json')
    if (controller['protocol'] != g['protocol'] or os.getppid() != controller['pid'] or
            kernel_identity(controller['pid']) != {k:controller[k] for k in ('start_ticks','boot_id')}):
        raise ValueError('Only the single live campaign controller may dispatch a worker')
    tasks = read(verify(g['tasks']))
    if tasks != market_tasks(shared_ffn=staged):
        raise ValueError('Changed frozen scientific task/configuration')
    gate_id = record(gate)
    batch_root = root
    if staged:
        method = target.parent.name if action == 'analyze' else next(t['method'] for t in tasks if t['id'] == name)
        tasks = stages.batch_tasks(tasks, method)
        batch_root = root/'stages'/method
        stages.require_prior(root, method, gate_id)
        if not batch_root.is_dir() or (batch_root/'closed.json').exists():
            raise ValueError('This training stage is not active')
        if ((action == 'train' and (batch_root/'training_complete.json').exists())
                or (action == 'score' and (batch_root/'scoring_complete.json').exists())):
            raise ValueError('This stage operation is already complete')
    if action != 'train':
        frozen = read(batch_root/'training_complete.json')
        if frozen['tasks'] != tasks or set(frozen['winners']) != {t['id'] for t in tasks}:
            raise ValueError('Complete training batch required')
        if staged:
            stages.check_winners(root, frozen, tasks, gate_id, g['manifest'])
    manifest = read(verify(g['manifest']))
    check = Check(gate.parent, deadline-time.monotonic(), g['limits'])
    check()
    if action in ('train','score'):
        configure_compute()
        for binding in g['model_binding']['files']:
            verify(binding)
        task = next(t for t in tasks if t['id'] == name)
        identity = dict(protocol=g['protocol'], task=task, manifest=g['manifest'], gate=gate_id)
    if action == 'train':
        if target not in [root/'training'/name/f'attempt-{i}' for i in (1,2)]:
            raise ValueError('Wrong training attempt directory')
        domains = iter(DOMAINS)
        def load_fit(domain):
            if domain != next(domains):
                raise ValueError('Out-of-order fit access')
            src = manifest['files'][domain+':fit']
            write(target/f'access_{domain}.json', dict(role='fit', task=name, source=src))
            return read(verify(src))
        out = execute(make_model(g['model_binding'],task), load_fit, target/'run', task, identity, g['model_binding'], check)
        validate_training(task, out, manifest, identity)
        return
    if action == 'score':
        if target != root/'scoring'/name:
            raise ValueError('Wrong blind score destination')
        winner = frozen['winners'][name]
        if not read(verify(winner['receipt']))['success']:
            raise ValueError('Unsuccessful attempt cannot be scored')
        out = read(verify(winner['completion']))
        validate_training(task, out, manifest, identity)
        scored = score(task, out, manifest, g['model_binding'], target/'result', 'dev', check)
        validate_scores(scored)
        return
    if action != 'analyze' or target != batch_root/'analysis':
        raise ValueError('Invalid result operation')
    if staged:
        stages.analyze_stage(root, method, tasks, g, gate_id, manifest)
        return
    batch = read(root/'scoring_complete.json')
    if batch['tasks'] != tasks or len(batch['completions']) != len(tasks):
        raise ValueError('Complete blind score batch required')
    completions = [read(verify(b)) for b in batch['completions']]
    for task, c in zip(tasks,completions,strict=True):
        if c['role']!='dev' or c['training_identity'] != dict(protocol=PROTOCOL, task=task, manifest=g['manifest'], gate=gate_id):
            raise ValueError('Mismatched blind scoring release')
    results = analyze(completions, tasks, target/'result')
    historical = {key:read(verify(row['binding']))[row['method']] for key,row in g['historical'].items()}
    decision = paired_report(results,historical)
    metrics = read(target/'result/metrics.json')
    write_metric_tables(tasks, metrics, target/'result')
    write(target/'result/decision.json', decision)


def write_metric_tables(tasks, metrics, destination):
    groups = {}
    for task in tasks:
        start = task['id']+':'
        for name,row in metrics.items():
            if name.startswith(start):
                groups.setdefault(name[len(start):],[]).append((task['seed'],row))
    table = {}
    for name, rows in groups.items():
        rows.sort(key=lambda pair:pair[0])
        if [s for s,_ in rows] != list(SEEDS):
            raise ValueError('Incomplete metric seed table')
        table[name] = {}
        for k in rows[0][1]:
            v = [r[k] for _,r in rows]
            table[name][k] = dict(values=v, mean=statistics.mean(v), sd=statistics.stdev(v), range=[min(v),max(v)])
    write(destination/'metric_summary.json',table)
    paired_metrics = {}
    for method in ('H_ER','H_SEQ','V_SET','V_CLS'):
        for cell in ('R01','R11','R12','R21','R22','R23','R31','R32','R33'):
            left = table['H_full:'+cell]
            right = table[method+':'+cell]
            paired_metrics[method+':'+cell] = {}
            for metric in left:
                differences = [a-b for a,b in zip(left[metric]['values'],right[metric]['values'],strict=True)]
                paired_metrics[method+':'+cell][metric] = dict(values=differences,mean=statistics.mean(differences),
                    sd=statistics.stdev(differences),range=[min(differences),max(differences)])
    write(destination/'metric_pairs.json',dict(seeds=list(SEEDS),full_minus_comparator=paired_metrics))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--gate',required=True,type=Path)
    p.add_argument('--action',required=True,choices=('train','score','analyze'))
    p.add_argument('--target',required=True,type=Path)
    p.add_argument('--task')
    p.add_argument('--deadline',required=True,type=float)
    args = p.parse_args()
    expected = Path(__file__).resolve().parents[2]/'campaign'
    if not args.target.resolve().is_relative_to(expected):
        raise ValueError('Target outside this campaign')
    try:
        worker(args.gate,args.action,args.target,args.task,args.deadline)
    except Exception as error:
        import torch
        from step35_vendorlink_account_io import SharedDiskLimit
        oom = isinstance(error,torch.cuda.OutOfMemoryError) or 'CUDA out of memory' in str(error)
        if isinstance(error,(ValueError,SharedDiskLimit)) and not oom:
            write(args.target/'shared_failure.json',dict(error=repr(error)))
        write(args.target/'worker_failure.json',dict(kind='CUDA_OOM' if oom else 'OTHER',error=repr(error)))
        raise
