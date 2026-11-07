"""Finite development feedback and one frozen, independent confirmation batch."""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np

import step28_h_ablation_run as run
import step28_continual_five_evaluate as prior
import step28_continual_comparison_evaluate as shared
from step28_bge_continual_run import save_array

m, data = run.m, run.data
NAMESPACE = "h_history_ablation_confirmation_20261106"
CONFIRM_SEED, BOOTSTRAP_SEED = 2026110601, 2026110602
NAMES = m.NAMES
CELLS, FIELDS = shared.CELLS, shared.FIELDS


def identity(item):
    spec = item["spec"]
    return f'{spec.order}_s{spec.seed}_{item["name"]}'


def complete_batch(batch_path, allowed, *, reference=True):
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
        records.append(dict(name=spec.name, spec=spec, root=root, record=row))
    if reference:
        old = run.references(allowed, verify_models=False)
        by_pair = {(r["spec"].order, r["spec"].seed): r for r in old}
        for item in records:
            original = by_pair[item["spec"].order, item["spec"].seed]["record"]
            if (item["record"]["partition"] != original["partition"]
                    or item["record"]["pretrained_archive"] != original["pretrained_archive"]
                    or item["record"]["initial"] != original["initial"]
                    or item["record"]["public_inputs"] != original["public_inputs"]):
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
    prior.access(campaign, "ablation_generation", ("new_generated_owners_and_pairs",))
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
    if (set(result["runs"]) != {identity(item) for item in records} or len(records) != 72
            or result["forwards"] != 17280 or len(result["groups"]) != 120
            or len(set(result["groups"])) != 120 or result["domains"] != [d for d in "ABC" for _ in range(40)]):
        raise ValueError("72 trajectories/432 cells/17280 forwards must all be present")
    if {(r["name"], r["spec"].order, r["spec"].seed) for r in records} != {
            (n, o, s) for n in NAMES for o in m.ORDERS for s in m.SEEDS}:
        raise ValueError("Every fixed arm/order/seed is required")
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
    if (generated["status"] != "GENERATED_VALIDATED_OWNERS_SEALED" or generated["namespace"] != NAMESPACE
            or generated["root_seed"] != CONFIRM_SEED or generated["sources"] != allowed["sources"]):
        raise ValueError("Fresh confirmation source differs")
    groups, domains = prior.confirmation_public(campaign/"generation")
    by_domain = {d: [g for g, actual in zip(groups, domains, strict=True) if actual == d] for d in "ABC"}
    result = dict(groups=[g.uid for g in groups], domains=domains, runs={}, forwards=0, sources=allowed["sources"])
    prior.access(campaign, "ablation_blind_inference", ("new_public_text", "216_inference_states"))
    for item in records:
        spec, root = item["spec"], item["root"]
        method = m.history.Method(spec) if item["name"] == "Full" else m.Method(spec)
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
    prior.access(campaign, "ablation_confirmation_joint", ("new_pairs",))
    truth = {}
    with data.verify(campaign/"generation/pairs.jsonl", generated["files"]["pairs.jsonl"]).open() as stream:
        for group, line in zip(groups, stream, strict=True):
            labels = data.align_labels(group, [dict(r, label=str(r["label"])) for r in json.loads(line)])
            if sum(labels) != 20:
                raise ValueError("Confirmation label structure differs")
            truth[group.uid] = labels
    left, right = np.triu_indices(28, 1)
    counts = []
    for group in groups:
        matrix = np.zeros((28, 28), np.uint8)
        matrix[left, right] = matrix[right, left] = truth[group.uid]
        counts.append(matrix.sum(1))
    counts = np.asarray(counts, np.uint8)
    if not np.all((counts == 1).sum(1) == 16) or not np.all((counts == 2).sum(1) == 12):
        raise ValueError("Expected eight K2 and four K3 components")
    result["query_positive_counts"] = save_array(job/"query_positive_counts.npy", counts, job)
    for name, saved in result["runs"].items():
        for key, cell in saved["cells"].items():
            labels = np.asarray([truth[uid] for uid in cell["ids"]], np.uint8)
            cell["metrics"] = {}
            for role, rec in cell["scores"].items():
                budget.check()
                values = np.load(data.verify(job/rec["path"], rec), allow_pickle=False)
                cell["metrics"][role] = prior.metric_set(job/f"metrics_{name}_{key}_{role}", labels, values)
    data.write_json(job/"result.json", {**result, "status": "ALL_ABLATION_CONFIRMATION_METRICS_COMPLETE"})


CONTRASTS = (("Full", "ER"), ("Full", "KD"), ("ER", "SEQ"), ("KD", "SEQ"),
             ("Full", "SEQ"), ("Full", "NoRegroup"), ("Full", "MeanPool"),
             ("Full", "StandardRank"), ("Full", "AccountPool"))


def field_values(spec, cells, field):
    terms = ((3, 1, 1/3), (3, 2, 1/3), (3, 3, 1/3)) if field == "final_mean" else prior.field_terms(field)
    out = np.zeros((3, 40, 22))
    for stage, position, coefficient in terms:
        out["ABC".index(spec.order[position-1])] += coefficient*cells[f"{stage}_{position}"]
    if field in ("F", "F_first"):
        # Positive forgetting means deterioration, also for lower-is-better losses.
        for name in ("brier", "log_loss"):
            out[:, :, list(m.parent.metrics.COLUMNS).index(name)] *= -1
    return out


def summarize(job, saved, budget):
    count = 40
    positive_record = saved["query_positive_counts"]
    positives = np.load(data.verify(job/positive_record["path"], positive_record), allow_pickle=False)
    if positives.shape != (120, 28) or not np.all((positives == 1).sum(1) == 16) or not np.all((positives == 2).sum(1) == 12):
        raise ValueError("Same-opening query grouping is missing")
    per_run, cells_saved, subgroups = {}, {}, {}
    grouped = {name: [] for name in NAMES}
    for name, item in saved["runs"].items():
        spec = m.history.Spec.from_dict(item["spec"]) if item["name"] == "Full" else m.Spec.from_dict(item["spec"])
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
                cell_id = f"{name}_{role}_{key}"
                cells_saved[cell_id] = dict(domain=point["domain"], mean=prior.nullable(values.mean(0)),
                    defined_groups=np.isfinite(values).sum(0).tolist(), groups=count, counts=totals,
                    pooled_precision=tp/(tp+fp) if tp+fp else None,
                    pooled_mcc=(tp*tn-fp*fn)/denominator if denominator else None)
                if item["name"] in ("Full", "StandardRank"):
                    rec = point["metrics"][role]["queries"]
                    queries = np.load(data.verify(job/rec["path"], rec), allow_pickle=False)
                    indices = [saved["groups"].index(uid) for uid in point["ids"]]
                    subgroups[cell_id] = {str(p): dict(queries_per_group=16 if p == 1 else 12,
                        columns=["map", "mrr", "recall_at_5"],
                        group_values=[q[mask == p][:, [0, 1, 4]].mean(0).tolist()
                                      for q, mask in zip(queries, positives[indices], strict=True)]) for p in (1, 2)}
            per_run[name][role] = {field: field_values(spec, cells, field) for field in FIELDS}
    if any(len(ids) != 9 for ids in grouped.values()):
        raise ValueError("Nine complete paired trajectories per arm required")
    averaged = {name: {role: {field: np.mean([per_run[uid][role][field] for uid in ids], axis=0)
        for field in FIELDS} for role in m.parent.ROLES} for name, ids in grouped.items()}
    def points(roles):
        return {role: {f: prior.nullable(v.mean(1).sum(0)) for f, v in fields.items()} for role, fields in roles.items()}
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(count, size=(5000, 3, count))
    contrasts = {f"{a}-{b}": {role: {f: shared.interval(averaged[a][role][f]-averaged[b][role][f],
        draws, (.025, .975), budget.check) for f in FIELDS} for role in m.parent.ROLES} for a, b in CONTRASTS}
    interaction = {role: {f: shared.interval(averaged["Full"][role][f]-averaged["ER"][role][f]
        -averaged["KD"][role][f]+averaged["SEQ"][role][f], draws, (.025, .975), budget.check)
        for f in FIELDS} for role in m.parent.ROLES}
    primary = contrasts["Full-ER"]["raw"]
    checks = dict(old_retention=primary["O"]["map"]["conditional_interval"][0] > 0,
        forgetting=primary["F"]["map"]["conditional_interval"][1] < 0,
        **{f: primary[f]["map"]["mean"] >= 0 for f in ("A_acq", "N", "Z")})
    stratified = {}
    for arm, ids in grouped.items():
        stratified[arm] = {}
        for label, groups in (("seed", m.SEEDS), ("order", m.ORDERS)):
            stratified[arm][label] = {str(value): points({role: {f: np.mean([
                per_run[uid][role][f] for uid in ids if saved["runs"][uid]["spec"][label] == value], axis=0)
                for f in FIELDS} for role in m.parent.ROLES}) for value in groups}
        stratified[arm]["seed_sample_sd"] = {role: {f: prior.nullable(np.std(np.asarray([
            stratified[arm]["seed"][str(seed)][role][f] for seed in m.SEEDS], dtype=np.float64), axis=0, ddof=1))
            for f in FIELDS} for role in m.parent.ROLES}
    result = dict(status="FIXED_ABLATION_METRIC_SUMMARY", metric_columns=list(m.parent.metrics.COLUMNS),
        per_run={name: points(roles) for name, roles in per_run.items()},
        absolute={name: points(roles) for name, roles in averaged.items()}, per_cell=cells_saved,
        stratified=stratified, ranking_query_subgroups=subgroups, contrasts=contrasts,
        exploratory_interaction=interaction, primary_checks=checks, primary_pass=all(checks.values()),
        forgetting_orientation="F/F_first: acquisition-minus-final for higher-is-better metrics; reversed for brier/log_loss",
        bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=5000, paired_groups_per_actual_domain=40,
            same_indices_all_arms_roles_endpoints_metrics=True),
        limits="Full-ER is the sole joint confirmatory claim. Other contrasts/interaction are exploratory. "
            "Group intervals condition on the fixed nine paths and synthetic mechanism; not seed/population uncertainty. "
            "A_acq/N/Z are point protections, not statistical noninferiority. N/A propagates; pooled metrics are separate. "
            "Selected mother and training world have been seen; previous results and failed gates remain unchanged.")
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
    run.read_batch(args.batch, allowed)
    expected = campaign/{"generate": "generation", "confirmation": "confirmation", "analyze": "analysis"}[args.action]
    if (args.job.resolve() != expected.resolve() or not args.job.is_dir()
            or set(p.name for p in args.job.iterdir()) - {"console.txt", "process.json"}):
        raise ValueError("Fresh owned evaluation job required")
    records = complete_batch(args.batch, allowed)
    budget = run.Budget(campaign, args.deadline)
    if args.action == "generate":
        generate(campaign, args.job, records, budget, allowed)
    elif args.action == "confirmation":
        confirmation(campaign, args.job, records, budget, allowed)
    else:
        saved = data.read_json(campaign/"confirmation/result.json")
        if saved["status"] != "ALL_ABLATION_CONFIRMATION_METRICS_COMPLETE" or saved["sources"] != allowed["sources"]:
            raise ValueError("Complete frozen confirmation required")
        validate_cells(saved, records)
        summarize(campaign/"confirmation", saved, budget)
        data.write_json(args.job/"result.json", dict(status="ANALYSIS_COMPLETE",
            summary=run.queue.record(campaign/"confirmation/summary.json")))
    data.write_json(args.job/"resources.json", budget.snapshot())


if __name__ == "__main__":
    main()


