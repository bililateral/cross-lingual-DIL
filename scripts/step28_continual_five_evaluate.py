"""All-run gated old-dev/new-confirmation evaluation and fixed paired statistics."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

import numpy as np

import step28_continual_five as m
import step28_continual_five_run as run
import step28_continual_five_control as ctl
import step28_continual_expression_data as generator
import step28_continual_expression_data_verify as validator
from step28_balanced_pooling_run import query_metrics, undefined_metrics

data = m.data
NAMESPACE, CONFIRM_SEED, BOOTSTRAP_SEED = "continual_confirmation_20261023", 2026102301, 2026102302
CORE_CELLS = ((1, 1), (2, 1), (2, 2), (3, 1), (3, 2), (3, 3))
COMPARISONS = (("H_pos_tuned", "S_tuned"), ("ER0.1_ref", "SEQ_ref"),
               ("LOGIT0.1_ref", "ER0.1_ref"), ("S_tuned", "LOGIT0.1_ref"),
               ("H_pos_tuned", "LOGIT0.1_ref"), ("H_pos_tuned", "ER0.1_ref"), ("H_pos_tuned", "SEQ_ref"))


def access(campaign: Path, role: str, keys: tuple[str, ...]) -> None:
    path = campaign/"evaluation_access.json"
    row = data.read_json(path) if path.exists() else dict(events=[])
    if any(event["role"] == role for event in row["events"]):
        raise ValueError("Logical access already attempted; no reread/retry")
    row["events"].append(dict(role=role, keys=list(keys), time=time.time()))
    data.write_json(path, row)  # Charge the attempted access before opening.


def full_gate(campaign: Path, gate: dict) -> list[dict]:
    if gate["sources"] != run.sources() or gate["status"] != "APPROVED_IMPLEMENTATION_AND_RESOURCES":
        raise ValueError("Actual frozen implementation/statistics differ")
    complete = data.read_json(campaign/"training_complete.json")
    if complete["status"] != "ALL_45_BLIND_COMPLETE" or complete["sources"] != gate["sources"]:
        raise ValueError("Entire training campaign incomplete")
    records = [data.read_json(campaign/s.run_id/"run/completion.json") for s in m.specifications()]
    ctl.joint_opening(records, gate["sources"])
    return records


def metric_set(root: Path, labels: np.ndarray, logits: np.ndarray) -> dict:
    matrix, counts = m.parent.metrics.group_metrics(labels, logits)
    queries = query_metrics(labels, logits)
    if (matrix.shape != (len(labels), 22) or queries.shape != (len(labels), 28, 10)
            or not np.allclose(matrix[:, 12:], queries.mean(1), rtol=1e-12, atol=1e-12)
            or any(v["tp"]+v["fn"] != 20 or v["tn"]+v["fp"] != 358 for v in counts)):
        raise ValueError("Full metric/query/confusion correspondence differs")
    matrix[undefined_metrics(counts)] = np.nan
    nullable = matrix.astype(object)
    nullable[np.isnan(matrix)] = None
    data.write_json(root.with_suffix(".json"), dict(columns=list(m.parent.metrics.COLUMNS),
        values=nullable.tolist(), counts=counts, na_reason="Undefined precision/MCC denominator; never zero-fill"))
    query_rec = run.save_array(root.with_suffix(".npy"), queries, root.parent)
    return dict(matrix=data.record(root.with_suffix(".json"), root.parent), queries=query_rec)


def old_dev(campaign: Path, job: Path, records: list, budget: run.Budget) -> None:
    # Revalidate all blind matrices before the single joint label opening.
    ctl.validate_blind_files(campaign, records)
    c = m.config(m.specifications()[0])
    groups, metadata, checked = m.base.public.public_inputs(c)
    _, partition = m.base.partition(groups, metadata, c)
    if any(r["partition"] != partition or r["public_inputs"] != checked for r in records):
        raise ValueError("Common old-development identity differs")
    access(campaign, "old_dev_joint", ("old_development_pairs",))
    labelled = m.base.public.attach_labels(groups["development"], c, "development")
    truth = np.asarray([g.labels for g in labelled], np.uint8)
    result = dict(group_ids=[g.uid for g in labelled], domains=[r["domain"] for r in partition["development"]], runs={})
    for row in records:
        spec = m.Spec.from_dict(row["spec"])
        result["runs"][spec.run_id] = {}
        for stage, point in row["points"].items():
            if point["development_ids"] != result["group_ids"]:
                raise ValueError("Development score/group order differs")
            for role in m.parent.ROLES:
                budget.check()
                rec = point["scores"][role]
                values = np.load(data.verify(campaign/spec.run_id/"run"/rec["path"], rec), allow_pickle=False)
                name = f"{spec.run_id}_{stage}_{role}"
                result["runs"][spec.run_id][f"{stage}_{role}"] = metric_set(job/name, truth, values)
    data.write_json(job/"result.json", {**result, "status": "OLD_DEV_ALL_45_EVALUATED_NO_SELECTION"})


def generate(campaign: Path, job: Path, records: list, budget: run.Budget) -> None:
    old_policy = data.read_json(generator.POLICY)
    catalog = data.read_json(generator.CATALOG)
    generator.check_contract(old_policy, catalog)
    policy = {**old_policy, "root_seed": CONFIRM_SEED}
    checker = validator.TextChecker(catalog)
    public_path, owner_path, pair_path = job/"groups.jsonl", job/"owners.jsonl", job/"pairs.jsonl"
    with public_path.open("x", encoding="utf-8") as public, owner_path.open("x", encoding="utf-8") as owners, pair_path.open("x", encoding="utf-8") as pairs:
        for domain in "ABC":
            for index in range(40):
                budget.check()
                group, items, identities = generator.make_group(policy, catalog, domain, NAMESPACE, index)
                public.write(generator.json_line(dict(metadata=group, items=items)))
                owners.write(generator.json_line(identities))
                pairs.write(generator.json_line(list(generator.pair_rows(identities))))
    access(campaign, "new_generation_validation", ("new_owners", "new_pairs"))
    old_ids = {r["group_uid"] for role in records[0]["partition"].values() for r in role}
    seen_groups, seen_sellers, seen_items = set(), set(), set()
    with public_path.open(encoding="utf-8") as public, owner_path.open(encoding="utf-8") as owners, pair_path.open(encoding="utf-8") as pairs:
        for i, (p, o, q) in enumerate(zip(public, owners, pairs, strict=True)):
            budget.check()
            p, o, q = [__import__("json").loads(x) for x in (p, o, q)]
            meta = p["metadata"]
            checked = validator.check_group(p["items"], o, q, checker)
            if (meta["domain"] != "ABC"[i//40] or meta["group_index"] != i % 40 or meta["accounts"] != 28
                    or meta["items"] != len(p["items"]) or {v["group_uid"] for v in o} != {meta["group_uid"]}
                    or meta["split"] != NAMESPACE or meta["group_uid"] in seen_groups | old_ids
                    or seen_sellers & checked["accounts"] or seen_items & checked["item_ids"]):
                raise ValueError("New confirmation group/domain/namespace separation differs")
            seen_groups.add(meta["group_uid"])
            seen_sellers |= checked["accounts"]
            seen_items |= checked["item_ids"]
    if len(seen_groups) != 120:
        raise ValueError("Exactly 120 independent confirmation groups required")
    files = {p.name: data.record(p, job) for p in (public_path, owner_path, pair_path)}
    owner_path.chmod(0)
    data.write_json(job/"result.json", dict(status="GENERATED_VALIDATED_OWNERS_SEALED", root_seed=CONFIRM_SEED,
        namespace=NAMESPACE, groups=120, files=files, sources=run.sources(),
        mechanism="Same pinned generator and catalog, new root and namespace; no new training stream"))


def confirmation_public(job: Path) -> tuple[list, list]:
    import json
    manifest = data.read_json(job/"result.json")
    if manifest["status"] != "GENERATED_VALIDATED_OWNERS_SEALED":
        raise ValueError("Qualified new confirmation missing")
    groups, domains = [], []
    with data.verify(job/"groups.jsonl", manifest["files"]["groups.jsonl"]).open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            by_seller = defaultdict(list)
            for item in row["items"]:
                by_seller[item["seller_uid"]].append((item["item_uid"], item["title"], item["description"]))
            sellers = tuple(sorted(by_seller))
            group = data.Group(row["metadata"]["group_uid"], sellers, tuple(tuple(sorted(by_seller[s])) for s in sellers))
            group.validate()
            groups.append(group)
            domains.append(row["metadata"]["domain"])
    if len(groups) != 120 or domains != [d for d in "ABC" for _ in range(40)]:
        raise ValueError("Confirmation public coverage differs")
    return groups, domains


def confirmation(campaign: Path, job: Path, records: list, budget: run.Budget, gate: dict) -> None:
    import torch
    import json
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    m.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(gate["allocator_bytes"]/torch.cuda.get_device_properties(0).total_memory)
    groups, domains = confirmation_public(campaign/"generation")
    by_domain = {d: [g for g, gd in zip(groups, domains) if gd == d] for d in "ABC"}
    result = dict(groups=[g.uid for g in groups], domains=domains, runs={}, forwards=0)
    for row in records:
        spec = m.Spec.from_dict(row["spec"])
        method, c = m.Method(spec), m.config(spec)
        model, optimizer = method.fresh(c)
        del optimizer
        cells = CORE_CELLS+(((1, 2), (2, 3)) if spec.name == "SEQ_ref" else ())
        result["runs"][spec.run_id] = {}
        for stage in (1, 2, 3):
            budget.check()
            point = row["points"][str(stage)]
            rec = point["model"]
            restored = method.restore(data.verify(campaign/spec.run_id/"run"/rec["path"], rec), rec, model, None)
            if restored["stage"] != stage or restored["maps"] != point["maps"] or restored["adam_step"] != 288*stage:
                raise ValueError("Actual restored confirmation stage/calibration differs")
            for _, position in [cell for cell in cells if cell[0] == stage]:
                domain = spec.order[position-1]
                with m.diag.neutral(model):
                    raw = method.score(model, by_domain[domain], c, budget.check)
                if raw.shape != (40, 378):
                    raise ValueError("Confirmation cell shape differs")
                roles = dict(raw=raw, **{"stage-cal": m.parent.calibration.transform(raw, point["maps"]["stage"]),
                                        "first-cal": m.parent.calibration.transform(raw, point["maps"]["first"])})
                name = f"{stage}_{position}"
                result["runs"][spec.run_id][name] = dict(domain=domain, ids=[g.uid for g in by_domain[domain]],
                    scores={role: run.save_array(job/f"{spec.run_id}_{name}_{role}.npy", v, job) for role, v in roles.items()})
                result["forwards"] += 40
        del model
        __import__("gc").collect()
        torch.cuda.empty_cache()
        data.write_json(job/"blind_progress.json", result)
    expected_forwards = sum(40*(len(CORE_CELLS)+(2 if s.name == "SEQ_ref" else 0)) for s in m.specifications())
    if result["forwards"] != expected_forwards or set(result["runs"]) != {s.run_id for s in m.specifications()}:
        raise ValueError("All planned confirmation cells must finish before labels")
    for spec in m.specifications():
        cells = CORE_CELLS+(((1, 2), (2, 3)) if spec.name == "SEQ_ref" else ())
        if set(result["runs"][spec.run_id]) != {f"{i}_{j}" for i, j in cells}:
            raise ValueError("Missing or duplicate confirmation cell")
    data.write_json(job/"all_blind_complete.json", {**result, "sources": run.sources()})
    access(campaign, "new_confirmation_joint", ("new_pairs",))
    generated = data.read_json(campaign/"generation/result.json")
    path = data.verify(campaign/"generation/pairs.jsonl", generated["files"]["pairs.jsonl"])
    truth = {}
    with path.open(encoding="utf-8") as stream:
        for group, line in zip(groups, stream, strict=True):
            # The unchanged generator emits numeric JSON labels; the shared
            # audited aligner consumes the original CSV-style string schema.
            rows = [dict(row, label=str(row["label"])) for row in json.loads(line)]
            labels = data.align_labels(group, rows)
            if sum(labels) != 20:
                raise ValueError("Confirmation positive count differs")
            truth[group.uid] = labels
    for spec in m.specifications():
        for cell, point in result["runs"][spec.run_id].items():
            labels = np.asarray([truth[uid] for uid in point["ids"]], np.uint8)
            point["metrics"] = {}
            for role, rec in point["scores"].items():
                budget.check()
                values = np.load(data.verify(job/rec["path"], rec), allow_pickle=False)
                point["metrics"][role] = metric_set(job/f"metrics_{spec.run_id}_{cell}_{role}", labels, values)
    data.write_json(job/"result.json", {**result, "status": "ALL_CONFIRMATION_METRICS_COMPLETE", "sources": run.sources()})


def field_terms(field: str) -> tuple:
    terms = {"O": ((3, 1, .5), (3, 2, .5)), "N": ((2, 2, .5), (3, 3, .5)), "Z": ((3, 3, 1.),),
        "A_acq": ((1, 1, .5), (2, 2, .5)), "F_first": ((1, 1, 1.), (3, 1, -1.)),
        "F": ((1, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)),
        "D2": ((1, 1, 1.), (2, 1, -1.)), "G2": ((2, 2, 1.), (1, 2, -1.)),
        "D3": ((2, 1, .5), (3, 1, -.5), (2, 2, .5), (3, 2, -.5)), "G3": ((3, 3, 1.), (2, 3, -1.))}
    return terms[field]


def contributions(spec: m.Spec, cells: dict, field: str, count: int) -> np.ndarray:
    out = np.zeros((3, count, 22), np.float64)
    for stage, position, coefficient in field_terms(field):
        out["ABC".index(spec.order[position-1])] += coefficient*cells[f"{stage}_{position}"]
    return out


def interval(values: np.ndarray, draws: np.ndarray) -> dict:
    # values: actual-domain x group x metric. Domains are paired everywhere;
    # coefficients already encode the endpoint's domain weights.
    estimate = values.mean(1).sum(0)
    samples = sum(values[d][draws[:, d]].mean(1) for d in range(3))
    result = {}
    for j, name in enumerate(m.parent.metrics.COLUMNS):
        missing = bool(np.isnan(values[:, :, j]).any())
        result[name] = dict(mean=None if missing else float(estimate[j]),
            conditional_95pct_interval=None if missing else np.quantile(samples[:, j], [.025, .975]).tolist())
    return result


def analyze(campaign: Path, job: Path, budget: run.Budget) -> None:
    confirmation_result = data.read_json(campaign/"confirmation/result.json")
    if confirmation_result["status"] != "ALL_CONFIRMATION_METRICS_COMPLETE":
        raise ValueError("Cannot analyze a partial confirmation")
    draws = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(40, size=(5000, 3, 40))
    fields = ("O", "N", "Z", "A_acq", "F_first", "F")
    by_run, summaries, cell_points = {}, {}, {}
    for spec in m.specifications():
        by_run[spec.run_id] = {}
        for role in m.parent.ROLES:
            cells = {}
            for cell, point in confirmation_result["runs"][spec.run_id].items():
                rec = point["metrics"][role]["matrix"]
                saved = data.read_json(data.verify(campaign/"confirmation"/rec["path"], rec))
                cells[cell] = np.asarray(saved["values"], np.float64)
                if saved["columns"] != list(m.parent.metrics.COLUMNS) or cells[cell].shape != (40, 22):
                    raise ValueError("Saved confirmation matrix schema differs")
                cell_points[f"{spec.run_id}_{role}_{cell}"] = dict(domain=point["domain"],
                    values=nullable(cells[cell].mean(0)), counts=saved["counts"])
            selected_fields = fields+(("D2", "G2", "D3", "G3") if spec.name == "SEQ_ref" else ())
            by_run[spec.run_id][role] = {field: contributions(spec, cells, field, 40) for field in selected_fields}
            summaries[spec.run_id+"_"+role] = {field: nullable(v.mean(1).sum(0)) for field, v in by_run[spec.run_id][role].items()}
    averaged = {name: {role: {field: np.mean([by_run[s.run_id][role][field] for s in m.specifications() if s.name == name], axis=0)
                              for field in fields} for role in m.parent.ROLES} for name in m.METHODS}
    comparisons = {}
    for a, b in COMPARISONS:
        comparisons[a+"-"+b] = {}
        for role in m.parent.ROLES:
            comparisons[a+"-"+b][role] = {}
            for field in fields:
                budget.check()
                delta = averaged[a][role][field]-averaged[b][role][field]
                comparisons[a+"-"+b][role][field] = interval(delta, draws)
    primary = comparisons["H_pos_tuned-S_tuned"]["raw"]
    passed = (primary["O"]["map"]["conditional_95pct_interval"][0] > 0
              and primary["F"]["map"]["conditional_95pct_interval"][1] < 0
              and primary["N"]["map"]["mean"] >= 0 and primary["Z"]["map"]["mean"] >= 0)
    for role in m.parent.ROLES:
        for name in m.METHODS:
            old = averaged[name][role]["O"]
            decomposed = averaged[name][role]["A_acq"]-averaged[name][role]["F"]
            finite = np.isfinite(old) & np.isfinite(decomposed)
            np.testing.assert_allclose(old[finite], decomposed[finite], rtol=1e-10, atol=1e-10)
    data.write_json(job/"result.json", dict(status="FROZEN_ANALYSIS_COMPLETE", primary_pass=bool(passed),
        primary="H_pos_tuned-S_tuned: raw MAP O lower>0 AND F upper<0 AND N/Z point>=0", comparisons=comparisons,
        columns=list(m.parent.metrics.COLUMNS), per_run_endpoint_points=summaries,
        per_run_actual_domain_cell_points=cell_points,
        absolute_endpoint_points={name: {role: {field: nullable(v.mean(1).sum(0)) for field, v in fields_by_role.items()}
                                         for role, fields_by_role in roles.items()} for name, roles in averaged.items()},
        sign_convention="All reported fields retain the literal R formulas. For Brier/log_loss lower levels are better; positive acquisition-minus-final differences mean loss improvement, not forgetting. Only raw MAP defines the primary gate.",
        bootstrap=dict(seed=BOOTSTRAP_SEED, replicates=5000, actual_domains="ABC", clusters_per_domain=40),
        limits="Only fixed systems, three circular orders/three seeds and this generator; all non-H-S comparisons exploratory. N/Z are point guards, not statistical noninferiority."))


def nullable(values) -> list:
    return [None if not np.isfinite(v) else float(v) for v in values]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("old-dev", "generate", "confirmation", "analyze"))
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--deadline", type=float, required=True)
    args = parser.parse_args()
    gate = data.read_json(args.gate)
    records = full_gate(args.campaign, gate)
    budget = run.Budget(args.campaign, args.deadline)
    if args.action == "old-dev":
        old_dev(args.campaign, args.job, records, budget)
    elif args.action == "generate":
        generate(args.campaign, args.job, records, budget)
    elif args.action == "confirmation":
        confirmation(args.campaign, args.job, records, budget, gate)
    else:
        analyze(args.campaign, args.job, budget)
