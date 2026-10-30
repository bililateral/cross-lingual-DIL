"""One complete frozen trial, or the single nine-run saved-score opening."""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from functools import partial
import math
import os
from pathlib import Path
import sys
import threading
import traceback

import numpy as np

import step28_optuna as method
import step28_optuna_control as control
import step28_positive_ranking_run as old

single, data, core, parent, diag, previous = old.single, old.data, old.core, old.parent, old.diag, old.previous
ARMS=method.ARMS
POLICY = data.ROOT/"schema/step28_optuna_policy.json"
CONTRACT = data.ROOT/"docs/SELLER_ALIAS_OPTUNA.zh.md"
SCOPE = data.ROOT/"reports/documentation/20261018/optuna_plan/implementation_scope.zh.txt"
AUTHORIZATION = data.ROOT/"reports/documentation/20261018/optuna_implementation/authorization.json"
REPAIR_SCOPE = data.ROOT/"reports/documentation/20261018/optuna_implementation/review/repair_scope.zh.txt"
REPAIR_AUTHORIZATION = data.ROOT/"reports/documentation/20261018/optuna_implementation/repair/authorization.json"
EVIDENCE_LIMIT=2*2**30
COMPLETE = single.COMPLETE
CPU_TESTS=("test_actual_objective_vjp_lr_and_rejected_omissions", "test_seeds_own_memory_capture_and_restore",
    "test_real_optuna_selection_and_stop", "test_disjoint_budgets_and_real_launcher_failure",
    "test_entry_orchestration_and_access_negative_controls", "test_nine_saved_evaluation_and_independent_analysis",
    "test_cpu_six_call_fixture_orchestration", "test_repair_resources_and_anchor",
    "test_repair_controller_boundaries", "test_budget_amendment_gate",
    "test_resource_wait_and_admission", "test_specific_resource_recovery", "test_specific_reboot_recovery")


def sources():
    paths = {data.ROOT/p for p in ("scripts/step28_optuna.py", "scripts/step28_optuna_control.py",
        "scripts/step28_optuna_run.py", "scripts/step28_optuna_search.py", "scripts/step28_optuna_verify.py",
        "tests/test_step28_optuna.py", "tests/test_step28_positive_ranking.py", "tests/test_step28_g_half.py",
        "tests/test_step28_record_replay.py")}
    todo = list(paths)
    while todo:
        for node in ast.walk(ast.parse(todo.pop().read_text(encoding="utf-8"))):
            names = [n.name for n in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) else []
            for name in names:
                if name and name.startswith("step28_"):
                    child = data.ROOT/"scripts"/(name+".py")
                    if child not in paths:
                        paths.add(child); todo.append(child)
    paths.update(data.ROOT/"schema"/n for n in ("step28_g_half_policy.json", "step28_record_replay_policy.json",
        "step28_bge_continual_policy.json", "step28_alias_ranking_policy.json", "step28_chinese_base_policy.json"))
    paths.update((POLICY, CONTRACT, SCOPE, AUTHORIZATION, REPAIR_SCOPE, REPAIR_AUTHORIZATION))
    paths.update(data.ROOT/p for p in ("scripts/run_step28_optuna_cpu_linux_20261018.sh",
        "scripts/step28_optuna_formal_wait.py",
        "reports/documentation/20261019/optuna_recovery/scope.zh.txt",
        "reports/documentation/20261021/optuna_reboot/scope.zh.txt",
        "reports/documentation/20261019/optuna_budget/proposal.zh.txt",
        "reports/documentation/20261019/optuna_budget/authorization.json"))
    return [data.record(p, data.ROOT) for p in sorted(paths)]


def policy():
    p = data.read_json(POLICY)
    if p != dict(arms=list(method.ARMS), search_trials_per_arm=12, replication_seeds=[1, 2, 3],
        cpu_seconds=300, native_seconds=1800, native_calls_per_arm=6, per_arm_native_seconds=600,
        run_seconds=21600, search_seconds=777600, replication_seconds=194400,
        shared_seconds=control.SHARED, total_seconds=979200, train_parse_attempts=45, valid_parse_attempts=37,
        maximum_output_bytes=64*2**30, maximum_evidence_bytes=2*2**30):
        raise ValueError("Frozen finite campaign policy differs")
    return p


def runtime():
    return {**single.policy()["runtime"], "maximum_gpu_stage_seconds": 21600,
            "maximum_output_bytes": 64*2**30, "minimum_free_gpu_bytes": 9*2**30}


class Budget(single.legacy.Budget):
    def __init__(self, root, seconds=21600, *, campaign=None, started=None):
        super().__init__(root)
        if started is not None:
            self.started = started
        self.limits = {**runtime(), "maximum_gpu_stage_seconds": seconds}
        self.campaign = campaign if campaign is not None else root.parent

    def output_files(self):
        return list(self.campaign.rglob("*"))

    def check(self,reserve=0,*,force=False):
        with self.lock:
            if force:self.last_disk=-float("inf")
            previous_scan=self.last_disk
            super().check(reserve)
            if self.last_disk!=previous_scan and sum(p.stat().st_size for p in self.output_files() if p.is_file() and p.suffix!=".pt")>EVIDENCE_LIMIT:
                raise RuntimeError("Campaign evidence exceeds 2 GiB")


@contextmanager
def shared_resources(campaign,kind):
    """Check the actual CPU evaluator/analyzer and the complete campaign output."""
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="":raise ValueError("Shared stages require hidden GPU")
    budget=Budget(campaign,control.SHARED[kind],campaign=campaign)
    budget.phase=kind
    budget.check(force=True)
    stop=threading.Event()
    def watch():
        while not stop.wait(.2):
            try:budget.check()
            except BaseException:
                data.write_json(campaign/(kind+"_resource_failure.json"),dict(traceback=traceback.format_exc(),budget=budget.snapshot()))
                os._exit(2)
    thread=threading.Thread(target=watch,daemon=True);thread.start()
    try:
        yield budget
        budget.check(force=True)
    finally:
        stop.set();thread.join()


class Entry:
    POLICY, BASELINE, COMPLETE = POLICY, old.BASELINE, COMPLETE
    sources = staticmethod(sources)
    baseline_inputs, baseline_runs = staticmethod(old.baseline_inputs), staticmethod(old.baseline_runs)
    fresh, pretrained_archive = staticmethod(single.fresh), staticmethod(single.pretrained_archive)
    public_inputs, attach_labels, partition = staticmethod(single.public_inputs), staticmethod(single.attach_labels), staticmethod(single.partition)

    def __init__(self, spec):
        self.spec, self.roots, self.method = spec, spec.roots, method.Method(spec)
        self.POINTS = tuple(d+"_"+spec.arm for d in "AB")
        for name in ("train_stage", "train", "restore_point", "open_valid"):
            setattr(self, name, partial(getattr(single, name), entry=self))
        self.blind_gate = partial(old.blind_gate, entry=self)

    def config(self):
        return {**method.config(self.spec), "runtime": runtime()}

    def expected_head(self):
        import torch
        c = self.config()
        # The pinned loader reseeds immediately before head construction. No BGE is constructed here.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []):
            torch.manual_seed(self.roots["initialization"])
            head = core.build_model(torch.nn.Identity(), c["interventions"]["split_rank"]["account_dim"], c["input"]["head_hidden"]).head
            return core.state_digest(head.state_dict())

    def validate_initial(self, model, optimizer, initial, baseline):
        if optimizer.state or parent.adam_step(optimizer) != 0:
            raise ValueError("Fresh optimizer moments must be empty")
        initial.update(seed=self.roots["initialization"], head=core.state_digest(model.head.state_dict()), empty_moments=True)
        return self.validate_saved_initial(initial, baseline)

    def validate_saved_initial(self, initial, baseline):
        if (initial.get("seed") != self.roots["initialization"] or initial.get("empty_moments") is not True
                or initial.get("head") != self.expected_head()):
            return False
        if self.spec.seed == 0 and initial["model"] != baseline["initial_identity"]["model"]:
            return False
        if self.spec.seed == 0 and self.spec.lr_multiplier == 1 and initial["adam"] != baseline["initial_identity"]["adam"]:
            return False
        return True

    def checkpoint(self, root, name, model, optimizer, memory, current, calibration, valid, c, arm, stage, budget):
        return single.checkpoint(root, name, model, optimizer, memory if memory is not None else method.Memory(self.spec),
            current, calibration, valid, c, arm, stage, budget, entry=self)

    def memory_from(self, payload, arm):
        self.method.check_arm(arm)
        memory = method.Memory.from_bytes(payload, self.spec)
        if any(memory.maps.get(k) != v for k, v in self.method.metadata(arm).items()):
            raise ValueError("Own Memory scoring identity differs")
        return memory

    def validate_diagnostic(self, value, arm):
        if value.get("seed") != f"s{self.spec.seed}" or any(value.get(k) != v for k, v in self.method.metadata(arm).items()):
            raise ValueError("Diagnostic trial identity differs")

    def loss_total(self, row, stage, arm):
        if any(row.get(k) != v for k, v in self.method.metadata(arm).items()):
            raise ValueError("Update trial identity differs")
        def close(a, b):
            if not np.isfinite(a) or not np.isfinite(b) or not np.isclose(a, b, atol=2e-6, rtol=2e-6):
                raise ValueError("Trial loss decomposition differs")
        total = 0.
        for role, weight in (("current", 1.), ("history", .1)):
            if role == "history" and stage == 1:
                if role in row: raise ValueError("A cannot use history")
                continue
            r = row[role]
            if r["mse1"] is not None or len(r["views"]) != 2 or role == "current" and r["mse0"] != 0:
                raise ValueError("Wrong views/teacher")
            for v in r["views"]:
                close(v["total"], v["bce"]+self.spec.lambda_rank*v["rank_used"]+.5*v["hard"])
            close(r["supervised"], sum(v["total"] for v in r["views"])/2)
            close(r["distillation"], r["mse0"])
            close(r["supervised_contribution"], weight*r["supervised"])
            close(r["teacher_contribution"], .5*r["mse0"])
            close(r["weighted"], weight*r["supervised"]+.5*r["mse0"])
            total += r["weighted"]
        close(row["total"], total)
        return total


def anchor_compatibility(entry, scores):
    if entry.spec.phase != "search" or entry.spec.number != 0:
        return None
    root = (entry.baseline_runs()["S"][0] if entry.spec.arm == "S" else
            data.ROOT/"reports/seller_alias_continual/20261018/positive_ranking_result/job")
    manifest_path=root/"run/manifest.json"
    if entry.spec.arm!="S":
        data.verify(manifest_path,dict(bytes=210304,sha256="ef6416de2070376c4e45613493a23bd92083293f9b5b8bad8fabf430d5742d35"))
    manifest = data.read_json(manifest_path)
    for name in entry.POINTS:
        prior = previous.array(root/"run", manifest["points"][name]["scores"]["raw"], (40, 378), np.float32)
        if not np.array_equal(scores[name]["raw"], prior):
            raise ValueError("Default anchor raw scores differ from retained original")
    return True


def load_arrays(root, col):
    result = {}
    for name, roles in col["points"].items():
        result[name] = {}
        for role, rec in roles.items():
            saved = data.read_json(data.verify(root/rec["matrix"]["path"], rec["matrix"]))
            values = np.asarray(saved["values"], np.float64)
            counts = data.read_json(data.verify(root/rec["counts"]["path"], rec["counts"]))
            queries = previous.array(root, rec["queries"], (40,28,10), np.float64)
            if (saved["columns"] != list(parent.metrics.COLUMNS) or values.shape != (40,22)
                or np.isinf(values).any() or not np.array_equal(np.isnan(values), old.undefined_metrics(counts))
                or not np.allclose(queries.mean(1), values[:,12:], atol=1e-12, rtol=1e-12)):
                raise ValueError("Full saved metrics/counts/query correspondence differs")
            result[name][role] = values
    return result


def statistics(root, col, arm):
    arrays = load_arrays(root, col)
    rows = [np.flatnonzero(np.asarray(col["domains"]) == d) for d in "AB"]
    if any(len(r) != 20 for r in rows): raise ValueError("Fixed AB populations required")
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(20, size=(5000,2,20))
    result = dict(absolute={}, old_changes={}, columns=col["columns"], query_positive_count_MAP={})
    degrees=previous.array(root,col["query_positive_counts"],(40,28),np.uint8)
    for name, roles in arrays.items():
        result["absolute"][name] = {}
        queries=previous.array(root,col["points"][name]["raw"]["queries"],(40,28,10),np.float64)
        result["query_positive_count_MAP"][name]={}
        for count in (1,2):
            subgroup=(queries[:,:,0]*(degrees==count)).sum(1)/(degrees==count).sum(1)
            result["query_positive_count_MAP"][name][str(count)]={d:dict(mean=float(subgroup[r].mean()),
                conditional_95pct_interval=np.quantile(subgroup[r][draws[:,index,:]].mean(1),[.025,.975]).tolist()) for index,(d,r) in enumerate(zip("AB",rows))}
        for role, values in roles.items():
            rec = col["points"][name][role]
            counts = data.read_json(root/rec["counts"]["path"])
            ct = np.asarray([[r[k] for k in ("tp","fp","fn","tn")] for r in counts], np.int64)
            result["absolute"][name][role] = dict(by_domain=old.summarize(values, rows, draws),
                macro_all={k: None if np.isnan(values[:,j]).any() else float(values[:,j].mean()) for j,k in enumerate(col["columns"])},
                macro_undefined_groups=dict(zip(col["columns"], np.isnan(values).sum(0).tolist())),
                fixed_half_classification={"all": old.pooled_rates(ct), **{d:old.pooled_rates(ct[r]) for d,r in zip("AB",rows)}})
    result["old_changes"] = {role: old.summarize(arrays["B_"+arm][role]-arrays["A_"+arm][role],rows,draws)["A"] for role in parent.ROLES}
    result["domains"] = {d: float(arrays["B_"+arm]["raw"][r,12].mean(dtype=np.float64)) for d,r in zip("AB",rows)}
    result.update(status="FULL_TRIAL_STATISTICS_COMPLETE", source_files=sources(),
        limits="Repeated development; fixed whole-group conditional intervals; no success or parameter selection from auxiliary metrics.")
    data.write_json(root/"evaluation.json", result)
    return result


def evaluate_run(job, entry, manifest, scores, labelled, split):
    col = old.collect(job/"evaluation", scores, labelled, split, entry=entry)
    col["status"] = "ALL_6_TRIAL_METRIC_COUNT_SETS_SAVED"
    data.write_json(job/"evaluation/collected.json", col)
    diag.evaluate(job/"run/diagnostics", manifest, labelled, split, job/"diagnostics_evaluation", scope="AB")
    return statistics(job/"evaluation", col, entry.spec.arm)


def execute_trial(job, spec, gate_path):
    import torch
    from step28_optuna_recovery import formal_preflight
    from step28_optuna_search import validate_gate
    validate_gate(gate_path)
    if spec.phase not in ("search", "replication") or job.exists():
        raise ValueError("Fresh formal run required")
    resources = formal_preflight(data.ROOT)
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(28*2**30/torch.cuda.get_device_properties(0).total_memory,0)
    job.mkdir(parents=True)
    entry, budget = Entry(spec), Budget(job)
    data.write_json(job/"access.json", dict(train=0,valid=0,heldout=0,owners=0))
    data.write_json(job/"execution.json", dict(trial=spec.identity(), source_files=sources(),resources=resources))
    stop = threading.Event()
    def watchdog():
        while not stop.wait(1):
            try: budget.check()
            except BaseException:
                data.write_json(job/"failure.json", dict(status="BUDGET_STOP_NO_RETRY",traceback=traceback.format_exc()))
                os._exit(2)
    watcher=threading.Thread(target=watchdog,daemon=True);watcher.start()
    try:
        manifest,split,valid,metadata=entry.train(job,budget)
        scores,split=entry.blind_gate(job/"run",manifest)
        compatible=anchor_compatibility(entry,scores)
        data.write_json(job/"before_valid.json",dict(status="ALL_BLIND_GATES_PASS",trial=spec.identity(),source_files=sources()))
        result=dict(status="REPLICATION_BLIND_COMPLETE",trial=spec.identity(),anchor_compatible=compatible)
        if spec.phase == "search":
            labelled=previous.parse_once(job,valid,entry.config(),"development",loader=lambda gs,c,s:entry.attach_labels(gs,c,s,metadata))
            result.update(status="SEARCH_COMPLETE",domains=evaluate_run(job,entry,manifest,scores,labelled,split)["domains"])
        budget.check()
        result.update(access=data.read_json(job/"access.json"),budget=budget.snapshot(),source_files=sources())
        data.write_json(job/"completion.json",result)
    except BaseException:
        data.write_json(job/"failure.json",dict(status="FAIL_NO_RETRY",traceback=traceback.format_exc(),budget=budget.snapshot()))
        raise
    finally:
        stop.set();watcher.join()


def joint_evaluate(campaign, specs, records):
    with shared_resources(campaign,"evaluation") as budget:
        return _joint_evaluate(campaign,specs,records,budget)


def _joint_evaluate(campaign, specs, records, budget):
    control.joint_opening(records,specs)
    prepared=[]
    for spec in specs:
        job=campaign/spec.run_id;entry=Entry(spec)
        rec=next(r for r in records if r["trial"]==spec.identity())
        data.verify(data.ROOT/rec["completion_identity"]["path"],rec["completion_identity"])
        if data.read_json(job/"access.json")!=rec["access"]:raise ValueError("Own actual access ledger changed before joint opening")
        manifest=data.read_json(job/"run/manifest.json")
        scores,split=entry.blind_gate(job/"run",manifest)
        prepared.append((job,entry,manifest,scores,split))
    if any(p[4]!=prepared[0][4] for p in prepared): raise ValueError("Nine fixed partitions differ")
    # Public technical input reads are distinct from the one attempted supervision parse.
    groups,metadata,checked=single.public_inputs(prepared[0][1].config())
    if checked!=prepared[0][2]["public_inputs"]:raise ValueError("Joint public identities differ")
    out=campaign/"joint";out.mkdir()
    data.write_json(out/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
    labelled=previous.parse_once(out,groups["development"],prepared[0][1].config(),"development",
        loader=lambda gs,c,s:single.attach_labels(gs,c,s,metadata))
    for job,entry,manifest,scores,split in prepared:
        evaluate_run(job,entry,manifest,scores,labelled,split)
        budget.check(force=True)
    return replication_statistics(campaign,specs)


def replication_statistics(campaign,specs):
    if len(specs)!=9 or {(s.arm,s.seed) for s in specs}!={(a,s) for a in ARMS for s in (1,2,3)}:
        raise ValueError("All three fixed seeds required for each arm")
    cols={};arrays={}
    for spec in specs:
        root=campaign/spec.run_id/"evaluation";col=data.read_json(root/"collected.json")
        cols[spec.arm,spec.seed]=col;arrays[spec.arm,spec.seed]=load_arrays(root,col)
    first=next(iter(cols.values()))
    if any(any(col[k]!=first[k] for k in ("group_ids","domains","columns")) for col in cols.values()):
        raise ValueError("Paired group correspondence differs")
    rows=[np.flatnonzero(np.asarray(first["domains"])==d) for d in "AB"]
    draws=np.random.Generator(np.random.PCG64(20260930)).integers(20,size=(5000,2,20))
    signs=np.array([-1 if k in ("brier","log_loss") else 1 for k in first["columns"]])
    result=dict(comparisons={},development_direction_consistency={},source_files=sources(),three_seed_absolute={},query_subgroup_MAP_comparisons={})
    for arm in ARMS:
        result["three_seed_absolute"][arm]={phase:{role:old.summarize(np.mean([arrays[arm,s][phase+"_"+arm][role] for s in (1,2,3)],axis=0),rows,draws)
            for role in parent.ROLES} for phase in "AB"}
    for arm,opponent in (("S_pos","S"),("H_pos","S"),("H_pos","S_pos")):
        each={};aggregate={}
        for seed in (1,2,3):
            a,b=arrays[arm,seed],arrays[opponent,seed]
            each[str(seed)]={phase:{role:old.summarize(a[phase+"_"+arm][role]-b[phase+"_"+opponent][role],rows,draws)
                for role in parent.ROLES} for phase in "AB"}
            each[str(seed)]["forgetting"]={role:old.summarize(((a["A_"+arm][role]-a["B_"+arm][role])-
                (b["A_"+opponent][role]-b["B_"+opponent][role]))*signs,rows,draws)["A"] for role in parent.ROLES}
        for phase in "AB":
            aggregate[phase]={role:old.summarize(np.mean([arrays[arm,s][phase+"_"+arm][role]-
                arrays[opponent,s][phase+"_"+opponent][role] for s in (1,2,3)],axis=0),rows,draws) for role in parent.ROLES}
        aggregate["forgetting"]={role:old.summarize(np.mean([((arrays[arm,s]["A_"+arm][role]-arrays[arm,s]["B_"+arm][role])-
            (arrays[opponent,s]["A_"+opponent][role]-arrays[opponent,s]["B_"+opponent][role]))*signs for s in (1,2,3)],axis=0),rows,draws)["A"] for role in parent.ROLES}
        result["comparisons"][arm+"-"+opponent]=dict(by_seed=each,three_seed_mean=aggregate)
        if opponent=="S":
            result["development_direction_consistency"][arm]=all(each[str(s)]["B"]["raw"][d]["map"]["mean"]>0 for s in (1,2,3) for d in "AB")
        sub={}
        for count in (1,2):
            deltas=[];by_seed={}
            for seed in (1,2,3):
                pair=[]
                for candidate in (arm,opponent):
                    spec=next(s for s in specs if s.arm==candidate and s.seed==seed)
                    root=campaign/spec.run_id/"evaluation";col=cols[candidate,seed]
                    degrees=previous.array(root,col["query_positive_counts"],(40,28),np.uint8)
                    q=previous.array(root,col["points"]["B_"+candidate]["raw"]["queries"],(40,28,10),np.float64)
                    pair.append((q[:,:,0]*(degrees==count)).sum(1)/(degrees==count).sum(1))
                delta=pair[0]-pair[1];deltas.append(delta)
                by_seed[str(seed)]={d:dict(mean=float(delta[r].mean()),conditional_95pct_interval=np.quantile(delta[r][draws[:,i,:]].mean(1),[.025,.975]).tolist()) for i,(d,r) in enumerate(zip("AB",rows))}
            delta=np.mean(deltas,axis=0)
            sub[str(count)]=dict(by_seed=by_seed,three_seed_mean={d:dict(mean=float(delta[r].mean()),conditional_95pct_interval=np.quantile(delta[r][draws[:,i,:]].mean(1),[.025,.975]).tolist()) for i,(d,r) in enumerate(zip("AB",rows))})
        result["query_subgroup_MAP_comparisons"][arm+"-"+opponent]=sub
    result["diagnostics"]=replication_diagnostics(campaign,specs)
    result.update(status="NINE_REPLICATIONS_EVALUATED",limits="Development only; all nine retained regardless of six-cell result. No seed-population or final independent confirmation claim.")
    data.write_json(campaign/"joint/evaluation.json",result)
    return result


def replication_diagnostics(campaign,specs):
    """Describe fixed-seed point curves; retain conditional intervals per seed only."""
    result={}
    for arm in ARMS:
        per_seed={}
        for seed in (1,2,3):
            spec=next(s for s in specs if s.arm==arm and s.seed==seed)
            job=campaign/spec.run_id
            path=job/"diagnostics_evaluation/overfitting.json"
            report=data.read_json(path);manifest=data.read_json(job/"run/manifest.json")
            populations={}
            for phase in "AB":
                rec=manifest["diagnostics"][phase+"_"+arm][0]
                captured=data.read_json(data.verify(job/"run/diagnostics"/rec["path"],rec))
                populations[phase]={k:captured[k] for k in ("fit_ids","fit_domains","cache_ids","cache_domains","valid_ids")}
            per_seed[str(seed)]=dict(source=data.record(path,campaign),populations=populations,points=report["points"])
        means={}
        for phase in "AB":
            means[phase]={}
            for role,domains in (("fit",phase),("cache","A" if phase=="B" else "")):
                means[phase][role]={}
                for domain in domains:
                    rows=[per_seed[str(s)]["points"][phase+"_"+arm][role][domain] for s in (1,2,3)]
                    if any(r["status"]!="DESCRIPTIVE_CONDITIONAL_DEVELOPMENT_DIAGNOSIS" for r in rows):
                        raise ValueError("Complete seed diagnostics required")
                    means[phase][role][domain]={k:np.mean([r[k] for r in rows],axis=0).tolist()
                        for k in ("train_raw_curve","valid_raw_curve","benefit_gap_curve")}
                    means[phase][role][domain].update({k:np.mean([r[k]["mean"] for r in rows],axis=0).tolist()
                        for k in ("epoch1_to_6_train_benefit","epoch1_to_6_valid_benefit","epoch1_to_6_gap_widening")})
        result[arm]=dict(by_seed=per_seed,three_fixed_seed_point_means=means,
            limits="Equal-weight descriptive points only; populations stay per seed, no pooled cache sample, averaged CI or seed-population inference.")
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--job",type=Path,required=True);parser.add_argument("--gate",type=Path,required=True)
    parser.add_argument("--spec",type=Path);parser.add_argument("--joint",action="store_true")
    args=parser.parse_args()
    if os.name!="posix":parser.error("Scientific execution is Linux-only")
    if args.joint:
        from step28_optuna_search import validate_gate
        validate_gate(args.gate)
        records=data.read_json(args.job/"replications.json")
        joint_evaluate(args.job,[method.TrialSpec.from_dict(r["trial"]) for r in records],records)
    else:
        execute_trial(args.job,method.TrialSpec.from_dict(data.read_json(args.spec)),args.gate)
