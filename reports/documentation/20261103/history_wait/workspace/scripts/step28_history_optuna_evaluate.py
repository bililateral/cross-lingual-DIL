"""Finite development feedback and one frozen, independent confirmation batch."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import step28_history_optuna_run as run
import step28_continual_five_evaluate as prior
import step28_continual_comparison_evaluate as shared
from step28_bge_continual_run import save_array

m, data = run.m, run.data
NAMESPACE = "h_history_confirmation_20261102"
CONFIRM_SEED, BOOTSTRAP_SEED = 2026110201, 2026110202
NAMES = ("H_pos_tuned", m.NAME)
CELLS, FIELDS = shared.CELLS, shared.FIELDS


def identity(item):
    spec = item["spec"]
    return f'{spec.order}_s{spec.seed}_{item["name"]}'


def complete_batch(batch_path, allowed, *, reference=False):
    _, specs = run.read_batch(batch_path, allowed)
    dispatch = data.read_json(batch_path.parent/"dispatch.json")
    if (dispatch["status"] != "ALL_COMPLETE_RELEASED" or dispatch["sources"] != allowed["sources"]
            or [r["id"] for r in sorted(dispatch["completed"], key=lambda r: r["index"])] != [s.run_id for s in specs]
            or not all(r["success"] and r["released"] for r in dispatch["completed"])):
        raise ValueError("Whole batch must finish and release all worker processes before evaluation")
    records = []
    for spec in specs:
        root = batch_path.parent/spec.run_id/"run"
        row = run.validate_completion(root, spec, allowed, verify_models=False)
        records.append(dict(name=m.NAME, spec=spec, root=root, record=row))
    if reference:
        old = run.references(allowed, verify_models=False)
        by_pair = {(r["spec"].order, r["spec"].seed): r for r in old}
        for item in records:
            original = by_pair[item["spec"].order, item["spec"].seed]["record"]
            if (item["record"]["partition"] != original["partition"]
                    or item["record"]["pretrained_archive"] != original["pretrained_archive"]):
                raise ValueError("Paired H fit/calibration/development or initialization archive differs")
        records = old+records
    return records


def metric_matrix(job, record, count):
    saved = data.read_json(data.verify(job/record["matrix"]["path"], record["matrix"]))
    values = np.asarray(saved["values"], np.float64)
    queries = np.load(data.verify(job/record["queries"]["path"], record["queries"]), allow_pickle=False)
    if (saved["columns"] != list(m.parent.metrics.COLUMNS) or values.shape != (count, 22)
            or queries.shape != (count, 28, 10)
            or not np.allclose(values[:, 12:], queries.mean(1), atol=1e-12, rtol=1e-12)):
        raise ValueError("Saved metric/query correspondence differs")
    return values, saved


def development(campaign, job, records, budget, allowed, phase, batch_path):
    c = m.config(records[-1]["spec"])
    groups, metadata, checked = m.base.public.public_inputs(c)
    _, partition = m.base.partition(groups, metadata, c)
    if any(r["record"]["partition"] != partition or r["record"]["public_inputs"] != checked for r in records):
        raise ValueError("Common old development identity differs")
    # All completion/restore/score hashes have been checked above, before this one opening.
    prior.access(campaign, batch_path.parent.name+"_dev_joint", ("old_development_pairs",))
    labelled = m.base.public.attach_labels(groups["development"], c, "development")
    truth = np.asarray([g.labels for g in labelled], np.uint8)
    result = dict(status="ALL_DEVELOPMENT_METRICS_COMPLETE", sources=allowed["sources"],
        groups=[g.uid for g in labelled], domains=[r["domain"] for r in partition["development"]], runs={})
    for item in records:
        spec, row, root = item["spec"], item["record"], item["root"]
        cells = result["runs"][identity(item)] = dict(spec=spec.identity(), name=item["name"], cells={})
        for stage, position in CELLS:
            budget.check()
            point = row["points"][str(stage)]
            if point["development_ids"] != result["groups"]:
                raise ValueError("Saved development scores/group order differs")
            domain = spec.order[position-1]
            indices = [i for i, d in enumerate(result["domains"]) if d == domain]
            key = f"{stage}_{position}"
            cell = cells["cells"][key] = dict(domain=domain, ids=[result["groups"][i] for i in indices], metrics={}, scores={})
            for role in m.parent.ROLES:
                rec = point["scores"][role]
                values = np.load(data.verify(root/rec["path"], rec), allow_pickle=False)
                if values.shape != (60, 378) or not np.isfinite(values).all():
                    raise ValueError("Incomplete saved development scores")
                cell["scores"][role] = run.queue.record(root/rec["path"])
                cell["metrics"][role] = prior.metric_set(job/f'{identity(item)}_{key}_{role}', truth[indices], values[indices])
    data.write_json(job/"result.json", result)
    summarize(job, result, budget, confirmation=False)
    if phase == "search":
        feedback = verified_feedback(job, batch_path, allowed)
        data.write_json(job/"feedback.json", feedback)


def verified_feedback(job, batch_path, allowed):
    """Compute U from the actual saved final RAW query matrices, never supplied U."""
    _, specs = run.read_batch(batch_path, allowed)
    records = complete_batch(batch_path, allowed)
    saved = data.read_json(job/"result.json")
    if (specs[0].phase != "search" or saved["status"] != "ALL_DEVELOPMENT_METRICS_COMPLETE"
            or saved["sources"] != allowed["sources"]
            or set(saved["runs"]) != {identity(item) for item in records}):
        raise ValueError("Only the exact completed candidate may feed Optuna")
    values = {}
    map_index = list(m.parent.metrics.COLUMNS).index("map")
    for item in records:
        spec, row = item["spec"], item["record"]
        cells = saved["runs"][identity(item)]
        if cells["spec"] != spec.identity() or saved["groups"] != row["points"]["3"]["development_ids"]:
            raise ValueError("Feedback identity differs from actual final endpoint")
        domain_means = []
        for position in (1, 2, 3):
            cell = cells["cells"][f"3_{position}"]
            domain = spec.order[position-1]
            ids = [r["group_uid"] for r in row["partition"]["development"] if r["domain"] == domain]
            score = row["points"]["3"]["scores"]["raw"]
            if (cell["domain"] != domain or cell["ids"] != ids or len(ids) != 20
                    or cell["scores"]["raw"] != run.queue.record(item["root"]/score["path"])):
                raise ValueError("Feedback domain/RAW score binding differs")
            matrix, _ = metric_matrix(job, cell["metrics"]["raw"], 20)
            domain_means.append(float(matrix[:, map_index].mean()))
        values[spec.order] = sum(domain_means)/3
    return dict(status="THREE_ORDERS_VERIFIED_DEV_COMPLETE", specs=[s.identity() for s in specs],
        final_map_by_order=values, result_identity=data.record(job/"result.json", job), all_workers_released=True)


def generate(campaign, job, records, budget, allowed):
    generator, validator = shared.generator, shared.validator
    policy, catalog = data.read_json(generator.POLICY), data.read_json(generator.CATALOG)
    generator.check_contract(policy, catalog)
    if CONFIRM_SEED in (policy["root_seed"], prior.CONFIRM_SEED, shared.CONFIRM_SEED):
        raise ValueError("Confirmation root must be new")
    policy = {**policy, "root_seed": CONFIRM_SEED}
    checker = validator.TextChecker(catalog)
    old = {r["group_uid"] for item in records for rows in item["record"]["partition"].values() for r in rows}
    seen, accounts, controllers, items_seen = set(), set(), set(), set()
    paths = [job/name for name in ("groups.jsonl", "owners.jsonl", "pairs.jsonl")]
    prior.access(campaign, "history_generation", ("new_generated_owners_and_pairs",))
    with paths[0].open("x") as public, paths[1].open("x") as owners, paths[2].open("x") as pairs:
        for domain in "ABC":
            for index in range(40):
                budget.check()
                meta, items, identities = generator.make_group(policy, catalog, domain, NAMESPACE, index)
                pair_rows = list(generator.pair_rows(identities))
                checked = validator.check_group(items, identities, pair_rows, checker)
                expected = generator.opaque_id(CONFIRM_SEED, domain, NAMESPACE, index, "group")
                ids = {generator.opaque_id(CONFIRM_SEED, domain, NAMESPACE, index, "account", i) for i in range(28)}
                control = {generator.opaque_id(CONFIRM_SEED, domain, NAMESPACE, index, "controller", i) for i in range(12)}
                if (meta["group_uid"] != expected or expected in old | seen or checked["accounts"] != ids
                        or {r["controller_uid"] for r in identities} != control or accounts & ids
                        or controllers & control or items_seen & checked["item_ids"]):
                    raise ValueError("New generated identity coverage differs")
                seen.add(expected)
                accounts |= ids
                controllers |= control
                items_seen |= checked["item_ids"]
                public.write(generator.json_line(dict(metadata=meta, items=items)))
                owners.write(generator.json_line(identities))
                pairs.write(generator.json_line(pair_rows))
    if (len(seen), len(accounts), len(controllers)) != (120, 3360, 1440):
        raise ValueError("New confirmation incomplete")
    files = {p.name: data.record(p, job) for p in paths}
    paths[1].chmod(0)
    data.write_json(job/"result.json", dict(status="GENERATED_VALIDATED_OWNERS_SEALED", root_seed=CONFIRM_SEED,
        namespace=NAMESPACE, groups=120, files=files, sources=allowed["sources"]))


def validate_cells(result, records):
    if (set(result["runs"]) != {identity(item) for item in records} or len(records) != 18
            or result["forwards"] != 4320 or len(result["groups"]) != 120
            or len(set(result["groups"])) != 120 or result["domains"] != [d for d in "ABC" for _ in range(40)]):
        raise ValueError("18 trajectories/108 cells/4320 forwards must all be present")
    for item in records:
        saved = result["runs"][identity(item)]
        if saved["spec"] != item["spec"].identity() or set(saved["cells"]) != {f"{s}_{p}" for s, p in CELLS}:
            raise ValueError("Confirmation endpoint identity differs")
        for stage, position in CELLS:
            cell, domain = saved["cells"][f"{stage}_{position}"], item["spec"].order[position-1]
            if (cell["domain"] != domain or cell["ids"] != [uid for uid, d in zip(result["groups"], result["domains"], strict=True) if d == domain]
                    or set(cell["scores"]) != set(m.parent.ROLES)):
                raise ValueError("Confirmation cell/domain/group alignment differs")


def confirmation(campaign, job, records, budget, allowed):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(run.LIMITS["allocator_bytes"]/torch.cuda.get_device_properties(0).total_memory)
    generated = data.read_json(campaign/"generation/result.json")
    if (generated["namespace"] != NAMESPACE or generated["root_seed"] != CONFIRM_SEED or generated["sources"] != allowed["sources"]):
        raise ValueError("Fresh confirmation source differs")
    groups, domains = prior.confirmation_public(campaign/"generation")
    by_domain = {d: [g for g, actual in zip(groups, domains, strict=True) if actual == d] for d in "ABC"}
    result = dict(groups=[g.uid for g in groups], domains=domains, runs={}, forwards=0, sources=allowed["sources"])
    prior.access(campaign, "history_blind_inference", ("new_public_text", "54_inference_states"))
    for item in records:
        spec, root = item["spec"], item["root"]
        method = m.five.Method(spec) if item["name"] == NAMES[0] else m.Method(spec)
        c = m.config(spec)
        model, optimizer = method.fresh(c)
        del optimizer
        cells = result["runs"][identity(item)] = dict(spec=spec.identity(), name=item["name"], cells={})
        for stage in (1, 2, 3):
            budget.check()
            point = item["record"]["points"][str(stage)]
            path = data.verify(root/point["model"]["path"], point["model"])
            actual = method.restore(path, point["model"], model, None)
            if actual["stage"] != stage or actual["maps"] != point["maps"]:
                raise ValueError("Restored stage/calibration differs")
            for _, position in (cell for cell in CELLS if cell[0] == stage):
                domain, key = spec.order[position-1], f"{stage}_{position}"
                with m.diag.neutral(model):
                    raw = method.score(model, by_domain[domain], c, budget.check)
                roles, preserved = shared.calibrated_roles(raw, point["maps"])
                cells["cells"][key] = dict(domain=domain, ids=[g.uid for g in by_domain[domain]], preserved=preserved,
                    scores={role: save_array(job/f'{identity(item)}_{key}_{role}.npy', value, job) for role, value in roles.items()})
                result["forwards"] += 40
        del model
        gc.collect()
        torch.cuda.empty_cache()
        data.write_json(job/"blind_progress.json", result)
    validate_cells(result, records)
    for saved in result["runs"].values():
        for cell in saved["cells"].values():
            values = {role: np.load(data.verify(job/rec["path"], rec), allow_pickle=False) for role, rec in cell["scores"].items()}
            if any(v.shape != (40, 378) or not np.isfinite(v).all() for v in values.values()):
                raise ValueError("Complete finite blind matrices required before pairs")
            for role in ("stage-cal", "first-cal"):
                m.parent.calibration.preserve_order(values["raw"], values[role])
    data.write_json(job/"all_blind_complete.json", result)
    prior.access(campaign, "history_confirmation_joint", ("new_pairs",))
    truth = {}
    with data.verify(campaign/"generation/pairs.jsonl", generated["files"]["pairs.jsonl"]).open() as stream:
        for group, line in zip(groups, stream, strict=True):
            labels = data.align_labels(group, [dict(r, label=str(r["label"])) for r in json.loads(line)])
            if sum(labels) != 20:
                raise ValueError("Confirmation label structure differs")
            truth[group.uid] = labels
    for name, saved in result["runs"].items():
        for key, cell in saved["cells"].items():
            labels = np.asarray([truth[uid] for uid in cell["ids"]], np.uint8)
            cell["metrics"] = {}
            for role, rec in cell["scores"].items():
                budget.check()
                values = np.load(data.verify(job/rec["path"], rec), allow_pickle=False)
                cell["metrics"][role] = prior.metric_set(job/f"metrics_{name}_{key}_{role}", labels, values)
    data.write_json(job/"result.json", {**result, "status": "ALL_HISTORY_CONFIRMATION_METRICS_COMPLETE"})


def summarize(job, saved, budget, *, confirmation):
    count = 40 if confirmation else 20
    per_run, cells_saved, grouped = {}, {}, {name: [] for name in NAMES}
    for name, item in saved["runs"].items():
        spec = m.five.Spec.from_dict(item["spec"]) if item["name"] == NAMES[0] else m.Spec.from_dict(item["spec"])
        grouped[item["name"]].append(name)
        per_run[name] = {}
        for role in m.parent.ROLES:
            cells = {}
            for key, point in item["cells"].items():
                budget.check()
                values, row = metric_matrix(job, point["metrics"][role], count)
                cells[key] = values
                totals = {k: sum(r[k] for r in row["counts"]) for k in ("tp", "tn", "fp", "fn")}
                tp, tn, fp, fn = (totals[k] for k in ("tp", "tn", "fp", "fn"))
                denominator = ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
                cells_saved[f"{name}_{role}_{key}"] = dict(domain=point["domain"], mean=prior.nullable(values.mean(0)),
                    defined_groups=np.isfinite(values).sum(0).tolist(), groups=count, counts=totals,
                    pooled_precision=tp/(tp+fp) if tp+fp else None,
                    pooled_mcc=(tp*tn-fp*fn)/denominator if denominator else None)
            fields = {}
            for field in FIELDS:
                terms = ((3, 1, 1/3), (3, 2, 1/3), (3, 3, 1/3)) if field == "final_mean" else prior.field_terms(field)
                out = np.zeros((3, count, 22))
                for stage, position, coefficient in terms:
                    out["ABC".index(spec.order[position-1])] += coefficient*cells[f"{stage}_{position}"]
                fields[field] = out
            per_run[name][role] = fields
    averaged = {name: {role: {field: np.mean([per_run[identity][role][field] for identity in ids], axis=0)
        for field in FIELDS} for role in m.parent.ROLES} for name, ids in grouped.items() if ids}
    result = dict(status="FIXED_METRIC_SUMMARY", metric_columns=list(m.parent.metrics.COLUMNS),
        per_run={name: {role: {f: prior.nullable(v.mean(1).sum(0)) for f, v in fields.items()} for role, fields in roles.items()} for name, roles in per_run.items()},
        absolute={name: {role: {f: prior.nullable(v.mean(1).sum(0)) for f, v in fields.items()} for role, fields in roles.items()} for name, roles in averaged.items()},
        per_cell=cells_saved, confirmation=confirmation,
        limits="Old development and seeds 4/5/6 have been seen. New confirmation is same-mechanism synthetic data. Group intervals condition on nine paired paths; not seed/population uncertainty. Undefined metrics stay N/A. Original H-S gate unchanged.")
    if set(averaged) == set(NAMES):
        result["difference_new_minus_original"] = {role: {f: prior.nullable((averaged[NAMES[1]][role][f]-averaged[NAMES[0]][role][f]).mean(1).sum(0)) for f in FIELDS} for role in m.parent.ROLES}
    if confirmation:
        if any(len(ids) != 9 for ids in grouped.values()):
            raise ValueError("Nine paired trajectories per method required")
        draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(count, size=(5000, 3, count))
        result["paired_intervals"] = {role: {field: shared.interval(averaged[NAMES[1]][role][field]-averaged[NAMES[0]][role][field], draws, (.025, .975), budget.check) for field in FIELDS} for role in m.parent.ROLES}
        result["primary_final_map_improved"] = result["paired_intervals"]["raw"]["final_mean"]["map"]["conditional_interval"][0] > 0
        result["bootstrap"] = dict(seed=BOOTSTRAP_SEED, replicates=5000, paired_groups_per_domain=count)
    data.write_json(job/"summary.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=run.ACTIONS[1:])
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    allowed = run.gate(args.gate, args.action)
    allowed["_gate_sha256"] = data.sha256(args.gate)
    campaign = Path(allowed["campaign"])
    _, specs = run.read_batch(args.batch, allowed)
    search = args.action == "search-evaluate"
    if (search != (specs[0].phase == "search") or not args.job.is_dir()
            or set(p.name for p in args.job.iterdir()) - {"console.txt", "process.json"}):
        raise ValueError("Correct phase and fresh owned evaluation job required")
    expected = args.batch.parent/"evaluation" if args.action.endswith("evaluate") else campaign/{"generate": "generation", "confirmation": "confirmation", "analyze": "analysis"}[args.action]
    if args.job.resolve() != expected.resolve():
        raise ValueError("Evaluation destination outside its fixed phase")
    records = complete_batch(args.batch, allowed, reference=not search)
    budget = run.Budget(campaign, args.deadline)
    if args.action.endswith("evaluate"):
        development(campaign, args.job, records, budget, allowed, specs[0].phase, args.batch)
    elif args.action == "generate":
        if data.read_json(args.batch.parent/"evaluation/result.json")["status"] != "ALL_DEVELOPMENT_METRICS_COMPLETE":
            raise ValueError("Replication evaluation must finish first")
        generate(campaign, args.job, records, budget, allowed)
    elif args.action == "confirmation":
        confirmation(campaign, args.job, records, budget, allowed)
    else:
        saved = data.read_json(campaign/"confirmation/result.json")
        if saved["status"] != "ALL_HISTORY_CONFIRMATION_METRICS_COMPLETE" or saved["sources"] != allowed["sources"]:
            raise ValueError("Complete frozen confirmation required")
        validate_cells(saved, records)
        summarize(campaign/"confirmation", saved, budget, confirmation=True)
        data.write_json(args.job/"result.json", dict(status="ANALYSIS_COMPLETE", summary=run.queue.record(campaign/"confirmation/summary.json")))
    data.write_json(args.job/"resources.json", budget.snapshot())


if __name__ == "__main__":
    main()
