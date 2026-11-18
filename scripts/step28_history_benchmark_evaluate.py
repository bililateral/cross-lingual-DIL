"""One fresh cohort, endpoint-specific inference, joint opening and paired raw MAP."""
from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np

import step28_history_benchmark as bench
import step28_continual_comparison_evaluate as shared
import step28_continual_five_evaluate as prior
from step28_bge_continual_run import save_array

data, five = bench.data, bench.five
CELLS, ROLES = shared.CELLS, five.parent.ROLES
FIELDS = ("O", "F", "A_acq", "N", "Z", "final_mean")


def validate_cells(result):
    rows = bench.plan()
    ids, domains = result["groups"], result["domains"]
    if (len(set(ids)) != 120 or len(ids) != 120 or domains != [d for d in "ABC" for _ in range(40)] or
            result["forwards"] != 12960 or set(result["runs"]) != {r["id"] for r in rows}):
        raise ValueError("54 trajectories and 120 unique aligned groups are required")
    for row in rows:
        saved = result["runs"][row["id"]]
        cells = saved["cells"]
        if saved["native"] != row["native"] or set(cells) != {f"{s}_{p}" for s, p in CELLS}:
            raise ValueError("Missing or misidentified endpoint cell")
        for stage, position in CELLS:
            key = f"{stage}_{position}"
            cell, domain = cells[key], row["order"][position-1]
            if (cell["domain"] != domain or cell["ids"] != [uid for uid, d in zip(ids, domains) if d == domain]
                    or set(cell["scores"]) != set(ROLES) or any(
                        rec["path"] != f'{row["id"]}_{key}_{role}.npy' for role, rec in cell["scores"].items())):
                raise ValueError("Wrong cohort, group order or saved score role")


def load_blind(job, rec):
    values = np.load(data.verify(job/rec["path"], rec), allow_pickle=False)
    if values.shape != (40, 378) or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise ValueError("Expected 40 finite groups of 378 account-pair scores")
    return values


def blind_trajectory(item, groups, job, budget):
    import torch
    name = item["name"]
    spec = bench.native(name, item["order"], item["seed"])
    if name == bench.forever.NAME:
        c = bench.forever.config(spec)
        model, optimizer = bench.forever.fresh(spec, c)
        score = bench.forever.record.score
        def restore(path, point):
            actual = bench.forever.core.restore_state(path, model, None, point["state_sha256"])
            if actual["spec"] != spec.identity() or actual["counts"] != point["counts"]:
                raise ValueError("Wrong FOREVER trajectory or consolidation point")
            return actual
    else:
        runtime = bench.history if name == bench.H else bench.comparison if name in bench.comparison.NEW_METHODS else five
        method, c = runtime.Method(spec), runtime.config(spec)
        model, optimizer = method.fresh(c)
        score = method.score
        def restore(path, point):
            return method.restore(path, point["model"], model, None)
    del optimizer
    cells = {}
    for stage in (1, 2, 3):
        budget.check()
        point = item["record"]["points"][str(stage)]
        rec = point["model"]
        actual = restore(data.verify(item["root"]/rec["path"], rec), point)
        if actual["stage"] != stage or actual["maps"] != point["maps"]:
            raise ValueError("Loaded endpoint identity or fit-only mapping differs")
        for position in range(1, stage+1):
            key, domain = f"{stage}_{position}", item["order"][position-1]
            with five.diag.neutral(model):
                raw = score(model, groups[domain], c, budget.check)
            roles, preserved = shared.calibrated_roles(raw, actual["maps"])
            cells[key] = dict(domain=domain, ids=[g.uid for g in groups[domain]], preserved=preserved,
                              scores={role: save_array(job/f'{item["id"]}_{key}_{role}.npy', value, job)
                                      for role, value in roles.items()})
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return dict(native=item["native"], cells=cells)


def open_pairs(campaign, generated, groups):
    path = campaign/"generation/pairs.jsonl"
    rec = generated["files"]["pairs.jsonl"]
    if rec["path"] != path.name or path.stat().st_size != rec["bytes"]:
        raise ValueError("Pair-file identity differs")
    digest, truth = hashlib.sha256(), {}
    prior.access(campaign, "six_joint_pairs", ("new_pairs",))
    with path.open("rb") as stream:
        for group, line in zip(groups, stream, strict=True):
            digest.update(line)
            pairs = [dict(row, label=str(row["label"])) for row in json.loads(line)]
            labels = data.align_labels(group, pairs)
            if sum(labels) != 20:
                raise ValueError("Wrong confirmation label structure")
            truth[group.uid] = labels
    if digest.hexdigest() != rec["sha256"]:
        raise ValueError("Pair-file digest differs")
    return truth


def confirm(campaign, job, records, allowed, budget):
    import torch
    generated = data.read_json(campaign/"generation/result.json")
    if (generated["status"] != "GENERATED_VALIDATED_OWNERS_SEALED" or generated["groups"] != 120 or
            generated["root_seed"] != bench.CONFIRM_SEED or generated["namespace"] != bench.NAMESPACE or
            generated["sources"] != allowed["sources"]):
        raise ValueError("Expected this separately generated common cohort")
    prior.access(campaign, "six_blind_inference", ("new_public_text", "162_inference_states"))
    groups, domains = prior.confirmation_public(campaign/"generation")
    by_domain = {d: [g for g, gd in zip(groups, domains, strict=True) if gd == d] for d in "ABC"}
    result = dict(groups=[g.uid for g in groups], domains=domains, runs={}, forwards=0,
                  cohort=bench.cohort(), sources=allowed["sources"])
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    five.parent.configure_numerics()
    torch.cuda.set_per_process_memory_fraction(allowed["limits"]["allocator_bytes"]/torch.cuda.get_device_properties(0).total_memory)
    for item in records:
        saved = blind_trajectory(item, by_domain, job, budget)
        result["runs"][item["id"]] = saved
        result["forwards"] += sum(len(cell["ids"]) for cell in saved["cells"].values())
        data.write_json(job/"blind_progress.json", dict(completed=list(result["runs"]), forwards=result["forwards"]))
    validate_cells(result)
    for item in records:
        for key, cell in result["runs"][item["id"]]["cells"].items():
            budget.check()
            raw = load_blind(job, cell["scores"]["raw"])
            expected, _ = shared.calibrated_roles(raw, item["record"]["points"][key[0]]["maps"])
            for role in ("stage-cal", "first-cal"):
                if not np.array_equal(expected[role], load_blind(job, cell["scores"][role])):
                    raise ValueError("Saved blind matrix differs from the restored mapping")
    data.write_json(job/"all_blind_complete.json", result)
    truth = open_pairs(campaign, generated, groups)
    for identity, saved in result["runs"].items():
        for key, cell in saved["cells"].items():
            labels = np.asarray([truth[uid] for uid in cell["ids"]], np.uint8)
            cell["metrics"] = {}
            for role, rec in cell["scores"].items():
                budget.check()
                cell["metrics"][role] = prior.metric_set(job/f"metrics_{identity}_{key}_{role}",
                                                        labels, load_blind(job, rec))
    data.write_json(job/"result.json", {**result, "status": "ALL_54_BLIND_AND_SCORED"})


def field_means(cells):
    terms = {field: prior.field_terms(field) for field in FIELDS if field != "final_mean"}
    terms["final_mean"] = ((3, 1, 1/3), (3, 2, 1/3), (3, 3, 1/3))
    return {field: sum(weight*cells[f"{stage}_{position}"] for stage, position, weight in indexes)
            for field, indexes in terms.items()}


def paired_table(per_run):
    systems, pairs = {}, {}
    for name in bench.NAMES:
        rows = [per_run[f"{order}_s{seed}_{name}"] for order in bench.ORDERS for seed in bench.SEEDS]
        systems[name] = {k: float(np.mean([r[k] for r in rows])) for k in FIELDS}
        if name != bench.H:
            matched = {}
            for order in bench.ORDERS:
                for seed in bench.SEEDS:
                    key = f"{order}_s{seed}"
                    h, base = per_run[key+"_"+bench.H], per_run[key+"_"+name]
                    matched[key] = {k: h[k]-base[k] for k in FIELDS}
            pairs[name] = dict(by_order_seed=matched,
                               mean={k: float(np.mean([r[k] for r in matched.values()])) for k in FIELDS})
    return dict(systems=systems, h_minus_baseline=pairs)


def analyze(campaign, job, allowed, budget):
    root = campaign/"confirmation"
    saved = data.read_json(root/"result.json")
    if saved["status"] != "ALL_54_BLIND_AND_SCORED" or saved["sources"] != allowed["sources"] or saved["cohort"] != bench.cohort():
        raise ValueError("Complete scoring on the approved cohort is required")
    validate_cells(saved)
    values = {}
    for identity, row in saved["runs"].items():
        cells = {}
        for key, cell in row["cells"].items():
            budget.check()
            rec = cell["metrics"]["raw"]["matrix"]
            if rec["path"] != f"metrics_{identity}_{key}_raw.json":
                raise ValueError("Raw MAP matrix identity differs")
            matrix = data.read_json(data.verify(root/rec["path"], rec))
            a = np.asarray(matrix["values"], dtype=np.float64)
            if matrix["columns"] != list(five.parent.metrics.COLUMNS) or a.shape != (40, 22):
                raise ValueError("Invalid per-group metric matrix")
            raw_map = a[:, matrix["columns"].index("map")]
            if not np.isfinite(raw_map).all():
                raise ValueError("Incomplete raw MAP")
            cells[key] = float(raw_map.mean())
        values[identity] = field_means(cells)
    data.write_json(job/"result.json", dict(status="SIX_METHOD_PAIRED_TABLE", cohort=bench.cohort(),
                    metric="raw MAP", unit="fraction", difference="H minus baseline; negative F means less forgetting",
                    by_trajectory=values, **paired_table(values), sources=allowed["sources"]))


def run(action, campaign, job, allowed, budget):
    if action == "analysis":
        analyze(campaign, job, allowed, budget)
        return
    # Protect new group generation with a fresh on-disk hash of every endpoint.
    # During blind inference each payload is verified again on actual restoration.
    records = bench.completed(campaign, allowed, verify_models=action == "generation", check=budget.check)
    if action == "generation":
        shared.generate(campaign, job, records, budget, root_seed=bench.CONFIRM_SEED, namespace=bench.NAMESPACE,
                        sources=allowed["sources"], access_role="six_generation")
    elif action == "confirmation":
        confirm(campaign, job, records, allowed, budget)
    else:
        raise ValueError("Unknown post-training action")
