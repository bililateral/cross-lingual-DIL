"""Fixed-model expression interventions; Linux blind scoring, Windows valid audit.

No training, owner reconstruction, heldout access, or changes to source data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import platform
import time
from typing import Any

import numpy as np

import step28_continual_population as core
import step28_continual_population_data as data
import step28_continual_population_evaluate as metrics

RUN = data.ROOT / "reports/seller_alias_continual/20260909/population_execution/20260909_162607/run"
RUN_SHA = "dc4bc45cd91cb6e22a46fc91240b358a26516960c58dd52a0115f02ac0454b1f"
CATALOG = data.ROOT / "schema/step28_continual_catalog.json"
VIEWS = ("original", "renamed", "reassigned")
ORDERS = ("ABC", "BCA", "CAB")
ROLES = ("frozen", "sequential")
SEED = 20260910
MAX_SECONDS = 2700
MAX_BYTES = 64 * 1024**2
COMPLETE = "COMPLETE_FIXED_MODEL_INTERVENTION_VALID_LABELS_UNREAD"


def sources() -> list[dict]:
    paths = [Path(__file__), CATALOG,
             data.ROOT / "tests/test_step28_continual_cause_probe_contracts.py"]
    paths += [data.ROOT / "scripts" / (name + ".py") for name in
              ("step28_continual_population", "step28_continual_population_data",
               "step28_continual_population_run", "step28_continual_population_evaluate")]
    return [data.record(p, data.ROOT) for p in paths]


def original_manifest() -> dict:
    if data.sha256(RUN / "manifest.json") != RUN_SHA:
        raise ValueError("Original run manifest differs")
    result = data.read_json(RUN / "manifest.json")
    for record in result["source_files"]:
        data.verify(data.ROOT / record["path"], record)
    return result


def decode(title: str, description: str, catalog: dict) -> tuple[tuple[int, int, int], dict]:
    products = [p for topic in catalog["topics"] for sub in topic["subcategories"] for p in sub["products"]]
    matches = [(i, p, q) for p in products if p in title
               for i, t in enumerate(catalog["title_templates"]) for q in catalog["qualifiers"]
               if t.format(product=p, qualifier=q) == title]
    if len(matches) != 1:
        raise ValueError("Title does not have one exact catalog parse")
    title_axis, product, qualifier = matches[0]
    sub = next(s for t in catalog["topics"] for s in t["subcategories"] if product in s["products"])
    fields = {"product": product, "qualifier": qualifier}
    for key, candidates in (
        ("intro", [s.format(product=product) for s in catalog["intros"]]),
        ("detail", sub["details"]), ("packaging", catalog["packaging"]),
        ("delivery", catalog["delivery"]),
        ("service", [s for family in catalog["service_phrases"] for s in family]),
    ):
        found = [s for s in candidates if s in description]
        if not found:
            raise ValueError("Description component missing")
        # E.g. 提供... is also inside 主要提供...; exact reconstruction below is mandatory.
        fields[key] = max(found, key=len)
    desc_axes = [i for i, t in enumerate(catalog["description_templates"]) if t.format(**fields) == description]
    services = [(i, j) for i, family in enumerate(catalog["service_phrases"])
                for j, text in enumerate(family) if text == fields["service"]]
    if len(desc_axes) != 1 or len(services) != 1:
        raise ValueError("Description not exactly reconstructable")
    family, position = services[0]
    fields["service_position"] = position
    return (title_axis, desc_axes[0], family), fields


def render(style: tuple[int, int, int], fields: dict, catalog: dict) -> tuple[str, str]:
    values = dict(fields)
    values["service"] = catalog["service_phrases"][style[2]][fields["service_position"]]
    return (catalog["title_templates"][style[0]].format(**values),
            catalog["description_templates"][style[1]].format(**values))


def views(group: data.Group, catalog: dict) -> tuple[dict[str, data.Group], dict]:
    parsed = [[decode(title, description, catalog) for _, title, description in rows] for rows in group.items]
    styles = [rows[0][0] for rows in parsed]
    if any(any(style != styles[i] for style, _ in rows) for i, rows in enumerate(parsed)):
        raise ValueError("Account has inconsistent observed style")
    renamed = [tuple((x + 1) % n for x, n in zip(style, (3, 3, 4))) for style in styles]
    offset = data.seed_for(SEED, group.uid, "style_cycle") % (len(styles) - 1) + 1
    assigned = [renamed[(i + offset) % len(styles)] for i in range(len(styles))]
    assert Counter(assigned) == Counter(renamed)
    equal = lambda ss: np.asarray([[a == b for a, b in zip(ss[i], ss[j])]
                                  for i in range(len(ss)) for j in range(i + 1, len(ss))])
    if not np.array_equal(equal(styles), equal(renamed)):
        raise ValueError("Global rename changed equality relations")
    result = {"original": group}
    for name, selected in (("renamed", renamed), ("reassigned", assigned)):
        items = []
        for i, records in enumerate(group.items):
            changed = []
            for (uid, title, desc), (style, fields) in zip(records, parsed[i]):
                if render(style, fields, catalog) != (title, desc):
                    raise ValueError("Original exact roundtrip failed")
                new_title, new_desc = render(selected[i], fields, catalog)
                decoded_style, decoded_fields = decode(new_title, new_desc, catalog)
                preserved = ("product", "qualifier", "intro", "detail", "packaging", "delivery", "service_position")
                if decoded_style != selected[i] or any(fields[k] != decoded_fields[k] for k in preserved):
                    raise ValueError("Intervention altered retained content")
                changed.append((uid, new_title, new_desc))
            items.append(tuple(changed))
        result[name] = data.Group(group.uid, group.sellers, tuple(items), None)
        result[name].validate()
    return result, {"group_uid": group.uid, "offset": offset,
                    "pair_axis_relations_changed": int(np.count_nonzero(equal(renamed) != equal(assigned))),
                    "pair_axis_relations_total": len(styles) * (len(styles) - 1) // 2 * 3}


def inputs(manifest: dict) -> tuple[dict[str, list[data.Group]], list[dict], dict]:
    root = data.ROOT / manifest["config"]["data_root"]
    if data.sha256(root / "manifest.json") != manifest["config"]["data_manifest_sha256"]:
        raise ValueError("Data source differs")
    original = data.read_json(root / "manifest.json")
    if (root / "sources/step28_continual_catalog.json").read_bytes() != CATALOG.read_bytes():
        raise ValueError("Catalog differs from original generation")
    verified = {}
    for relative in ("groups.csv", "development/items.jsonl"):
        data.verify(root / relative, original["files"][relative])
        verified[relative] = original["files"][relative]
    with (root / "groups.csv").open(encoding="utf-8", newline="") as handle:
        metadata = sorted((r for r in csv.DictReader(handle) if r["split"] == "development"),
                          key=lambda r: (r["domain"], int(r["group_index"])))
    if [r["group_uid"] for r in metadata] != manifest["evaluation_group_ids"]["development"]:
        raise ValueError("Original valid row order differs")
    collected: dict = defaultdict(lambda: defaultdict(list))
    with (root / "development/items.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            r = json.loads(line)
            if set(r) != {"group_uid", "seller_uid", "item_uid", "title", "description"}:
                raise ValueError("Unexpected public field")
            collected[r["group_uid"]][r["seller_uid"]].append((r["item_uid"], r["title"], r["description"]))
    if set(collected) != {r["group_uid"] for r in metadata}:
        raise ValueError("Public groups differ")
    catalog = data.read_json(CATALOG)
    output = {name: [] for name in VIEWS}
    transformations = []
    for row in metadata:
        sellers = collected[row["group_uid"]]
        ids = tuple(sorted(sellers))
        group = data.Group(row["group_uid"], ids, tuple(tuple(sorted(sellers[s])) for s in ids))
        group.validate()
        changed, record = views(group, catalog)
        for name in VIEWS:
            output[name].append(changed[name])
        transformations.append(record)
    digest = {name: hashlib.sha256(data.json_bytes([g.payload() for g in rows])).hexdigest()
              for name, rows in output.items()}
    return output, metadata, {"files": verified, "view_sha256": digest, "transformations": transformations}


def score_run(out: Path) -> dict:
    import torch
    if platform.system() != "Linux" or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Requires the resumed Linux stage and one idle visible GPU")
    if not torch.cuda.is_bf16_supported() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("Insufficient free eligible GPU; wait")
    if out.exists() or not out.is_relative_to((data.ROOT / "reports").resolve()):
        raise ValueError("Use a new reports directory")
    manifest = original_manifest()
    groups, metadata, input_record = inputs(manifest)
    snapshot = sources()
    out.mkdir(parents=True)
    start = time.monotonic()

    def budget() -> None:
        if time.monotonic() - start > MAX_SECONDS:
            raise RuntimeError("Diagnostic time limit exceeded")
        if sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) > MAX_BYTES:
            raise RuntimeError("Diagnostic output limit exceeded")

    result: dict[str, Any] = {"status": "RUNNING", "source_files": snapshot, "run_sha256": RUN_SHA,
                              "inputs": input_record, "groups": metadata, "points": {},
                              "label_parses": 0, "optimizer_updates": 0,
                              "environment": {"python": platform.python_version(), "torch": torch.__version__,
                                              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}
    data.write_json(out / "startup.json", result)
    try:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        model = core.load_model(manifest["config"])
        for order in manifest["orders"]:
            for role in ROLES:
                key = order["order"] + "_" + role
                record = order["models"][role]
                path = data.verify(RUN / record["path"], record)
                budget()
                core.restore_state(path, model, None, record["state_sha256"])
                old_name = "shared" if role == "frozen" else "sequential_stage3"
                old_point = order["points"][old_name]
                if core.state_digest(model.state_dict()) != old_point["model_state_sha256"]:
                    raise ValueError("Loaded model differs from original score point")
                point_result = {"model": record, "scores": {}, "original_exact_replay": False}
                for view in VIEWS:
                    before = time.monotonic()
                    values = core.score(model, groups[view], manifest["config"], budget)
                    if view == "original":
                        old_score = old_point["scores"]["development"]
                        original = np.load(data.verify(RUN / old_score["path"], old_score), allow_pickle=False)
                        if not np.array_equal(values, original):
                            raise ValueError("Original full valid scores failed exact replay")
                        point_result["original_exact_replay"] = True
                    target = out / f"{key}_{view}.npy"
                    np.save(target, values, allow_pickle=False)
                    point_result["scores"][view] = {**data.record(target, out), "seconds": time.monotonic() - before}
                    print(data.json_bytes({"point": key, "view": view, "elapsed_seconds": time.monotonic()-start}).decode(), flush=True)
                result["points"][key] = point_result
        if sources() != snapshot:
            raise ValueError("Scientific sources changed")
        budget()
        result.update(status=COMPLETE, total_seconds=time.monotonic()-start,
                      maximum_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                      maximum_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
        data.write_json(out / "manifest.json", result)
        return result
    except Exception as error:
        data.write_json(out / "failure.json", {"error_type": type(error).__name__, "error": str(error),
                                               "elapsed_seconds": time.monotonic()-start})
        raise


def validated_scores(out: Path, original: dict, input_record: dict, metadata: list[dict]) -> dict:
    if (out / "failure.json").exists():
        raise ValueError("Failed run cannot be evaluated")
    record = data.read_json(out / "manifest.json")
    expected = {o + "_" + r for o in ORDERS for r in ROLES}
    if (record["status"] != COMPLETE or record["source_files"] != sources()
            or record["run_sha256"] != RUN_SHA or record["inputs"] != input_record
            or record["groups"] != metadata or record["label_parses"] != 0
            or record["optimizer_updates"] != 0 or set(record["points"]) != expected
            or record["total_seconds"] > MAX_SECONDS):
        raise ValueError("Incomplete or mismatched probe contract")
    values = {}
    for order in original["orders"]:
        for role in ROLES:
            key = order["order"] + "_" + role
            point = record["points"][key]
            if (point["model"] != order["models"][role] or not point["original_exact_replay"]
                    or set(point["scores"]) != set(VIEWS)):
                raise ValueError("Model/view mapping differs")
            for view in VIEWS:
                file = point["scores"][view]
                if file["path"] != f"{key}_{view}.npy":
                    raise ValueError("Unexpected score path")
                matrix = np.load(data.verify(out / file["path"], file), allow_pickle=False)
                if matrix.dtype != np.float32 or matrix.shape != (60, 378) or not np.isfinite(matrix).all():
                    raise ValueError("Incomplete/nonfinite blind scores")
                values[(key, view)] = matrix
            old_name = "shared" if role == "frozen" else "sequential_stage3"
            file = order["points"][old_name]["scores"]["development"]
            old = np.load(data.verify(RUN / file["path"], file), allow_pickle=False)
            if not np.array_equal(values[(key, "original")], old):
                raise ValueError("Original score replay not independently confirmed")
    return values


def contrast_rows(arrays: dict, column: int) -> dict[str, np.ndarray]:
    def rows(role: str, view: str) -> np.ndarray:
        # Match the first trained domain, not the order's position in ABC/BCA/CAB.
        return np.stack([arrays[(order + "_" + role, view)]
                         ["ABC".index(order[0])*20:("ABC".index(order[0])+1)*20, column]
                         for order in ORDERS])
    result = {}
    for view in VIEWS:
        result["retention_" + view] = rows("sequential", view) - rows("frozen", view)
    for name, left, right in (("surface", "renamed", "original"),
                              ("reassignment", "reassigned", "renamed")):
        for role in ROLES:
            result[name + "_" + role] = rows(role, left) - rows(role, right)
        result[name + "_retention_interaction"] = result[name + "_sequential"] - result[name + "_frozen"]
    return result


def manipulation_counts(groups: dict[str, list[data.Group]], labels: np.ndarray, catalog: dict) -> dict:
    """Check the actual expression/label association, without another truth read.

    Changed edges alone do not imply weakened association: a permutation can
    preserve the label graph. Keep this fixed intervention even in that case.
    """
    left, right = np.triu_indices(28, 1)
    bit_values = np.asarray([[int(char) for char in f"{i:03b}"] for i in range(8)])
    output = {}
    tables = {}
    for view in VIEWS:
        if labels.shape != (len(groups[view]), 378) or not np.isin(labels, [0, 1]).all():
            raise ValueError("Manipulation counts need aligned binary pairs")
        rows, matrices = [], []
        for group, truth in zip(groups[view], labels):
            styles = np.asarray([decode(items[0][1], items[0][2], catalog)[0] for items in group.items])
            codes = (styles[left] == styles[right]).astype(np.int64) @ np.asarray([4, 2, 1])
            counts = np.zeros((2, 8), dtype=np.int64)
            np.add.at(counts, (truth, codes), 1)
            if counts[0].sum() != 358 or counts[1].sum() != 20:
                raise ValueError("Manipulation count class totals differ")
            matrices.append(counts)
            rows.append({"group_uid": group.uid, "negative_pattern_counts": counts[0].tolist(),
                         "positive_pattern_counts": counts[1].tolist(),
                         "negative_axis_equal_rates": (counts[0] @ bit_values / 358).tolist(),
                         "positive_axis_equal_rates": (counts[1] @ bit_values / 20).tolist()})
        output[view] = rows
        tables[view] = np.stack(matrices)
    if not np.array_equal(tables["original"], tables["renamed"]):
        raise ValueError("Global rename changed label-conditioned equality patterns")
    if not np.array_equal(tables["renamed"].sum(1), tables["reassigned"].sum(1)):
        raise ValueError("Account permutation changed full-graph equality-pattern marginals")
    return {"axis_order": ["title_template", "description_template", "service_family"],
            "patterns": [f"{i:03b}" for i in range(8)], "by_view": output,
            "limit": "Inspect whether positive/negative association weakened; never redraw for a favorable manipulation."}


def evaluate(out: Path, destination: Path) -> dict:
    if destination.exists() or not destination.is_relative_to((data.ROOT / "reports").resolve()):
        raise ValueError("Use a new evaluation directory")
    original = original_manifest()
    public, metadata, input_record = inputs(original)
    values = validated_scores(out, original, input_record, metadata)
    destination.mkdir(parents=True)
    data.write_json(destination / "access.json", {"status": "COMPLETE_SCORE_GATE_PASSED_VALID_PARSE_STARTING",
                                                  "valid_parse_attempts": 1, "train": 0, "heldout": 0, "owners": 0})
    root = data.ROOT / original["config"]["data_root"]
    dm = data.read_json(root / "manifest.json")
    path = data.verify(root / "development/supervision/pairs.csv", dm["files"]["development/supervision/pairs.csv"])
    grouped: dict = defaultdict(list)
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            grouped[row["group_uid"]].append(row)
    if set(grouped) != {g.uid for g in public["original"]}:
        raise ValueError("Valid supervision groups differ; do not automatically retry")
    labels = np.asarray([data.align_labels(g, grouped[g.uid]) for g in public["original"]], dtype=np.uint8)
    if labels.shape != (60, 378) or not np.all(labels.sum(1) == 20):
        raise ValueError("Valid supervision structure differs")
    manipulation = manipulation_counts(public, labels, data.read_json(CATALOG))
    arrays, summaries = {}, {}
    for key, scores in values.items():
        classification = [metrics.classification(y, s) for y, s in zip(labels, scores)]
        matrix = np.column_stack((np.asarray([[row[c] for c in metrics.CLASS_KEYS] for row in classification]),
                                  metrics.retrieval(labels, scores, 28)))
        arrays[key] = matrix
        name = "_".join(key)
        file = destination / (name + "_metrics.npy")
        np.save(file, matrix, allow_pickle=False)
        summaries[name] = {"file": data.record(file, destination),
                           "by_domain": {d: dict(zip(metrics.COLUMNS, matrix[i*20:(i+1)*20].mean(0).tolist()))
                                         for i, d in enumerate("ABC")},
                           "confusion": [r["confusion"] for r in classification]}
    draws = np.random.default_rng(SEED).integers(0, 20, size=(5000, 3, 20))
    contrasts = {}
    for j, c in enumerate(metrics.COLUMNS):
        contrasts[c] = {}
        for name, delta in contrast_rows(arrays, j).items():
            sampled = np.stack([delta[i][draws[:, i]] for i in range(3)], axis=1).mean((1, 2))
            contrasts[c][name] = {"mean": float(delta.mean()), "by_first_domain": dict(zip("ABC", delta.mean(1).tolist())),
                                  "conditional_95pct_interval": np.quantile(sampled, [.025, .975]).tolist()}
    result = {"status": "FIXED_MODEL_INTERVENTION_EVALUATED_CAUSAL_INTERPRETATION_REQUIRED",
              "label_parses": {"development": 1, "train": 0, "heldout": 0, "owners": 0},
              "score_manifest_sha256": data.sha256(out / "manifest.json"), "columns": list(metrics.COLUMNS),
              "points": summaries, "contrasts": contrasts, "manipulation": manipulation,
              "bootstrap": {"seed": SEED, "draws": 5000, "unit": "paired group within first domain"},
              "limit": "Fixed-model intervention, single chosen rewrite, seen valid; not unique training-cause identification or mediation percentage."}
    data.write_json(destination / "evaluation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("score", "evaluate"))
    parser.add_argument("--out", type=Path, required=True, help="New blind run for score; completed run for evaluate")
    parser.add_argument("--evaluation", type=Path, help="New evaluation directory, Windows only")
    args = parser.parse_args()
    if args.action == "evaluate" and (platform.system() != "Windows" or args.evaluation is None):
        parser.error("Evaluation requires Windows and --evaluation")
    result = score_run(args.out.resolve()) if args.action == "score" else evaluate(args.out.resolve(), args.evaluation.resolve())
    print(data.json_bytes({"status": result["status"], "output": str(args.out)}).decode())


if __name__ == "__main__":
    main()
