"""Read the five approved FIT12 dev cells per trajectory and release aggregates once."""
import argparse
import hashlib
import io
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np


FAMILIES = {
    "ER": ("20261110/vendorlink_redesign/fit_reference/mechanism", "M_H_ER", "H_ER"),
    "det": ("20261112/vendorlink_deterministic", "M_H_full_det", "H_full"),
    "quarter": ("20261113/vendorlink_beta_quarter", "M_H_full_beta_quarter", "H_full"),
}
SEEDS = (51, 52, 53)
MARKETS = {"AlphaBay": 546, "Dream": 157}
CELLS = {"R11": "AlphaBay", "R21": "AlphaBay", "R31": "AlphaBay", "R22": "Dream", "R32": "Dream"}
TRANSITIONS = (("R11", "R21"), ("R21", "R31"), ("R11", "R31"), ("R22", "R32"))


def save(path, obj):
    tmp = path.with_suffix(path.suffix + ".part")
    with tmp.open("x", encoding="utf-8") as file:
        json.dump(obj, file, ensure_ascii=False, allow_nan=False, indent=2)
        file.write("\n")
    tmp.replace(path)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def receipt_path(root, family, seed):
    campaign, prefix, _ = FAMILIES[family]
    return root / "reports/documentation" / campaign / "campaign/M/scoring" / f"{prefix}_s{seed}" / "result/completion.json"


def inspect_inputs(root, bindings):
    """Read named completion receipts and stat selected matrices. Do not open scores."""
    selected, errors = {}, []
    expected = {(method, seed) for method in FAMILIES for seed in SEEDS}
    rows = bindings["trajectories"]
    require(len(rows) == len(expected) and {(r["family"], r["seed"]) for r in rows} == expected,
            "Incomplete trajectory binding")
    for row in rows:
        family, seed = row["family"], row["seed"]
        path = receipt_path(root, family, seed)
        _, prefix, reader = FAMILIES[family]
        try:
            require(path.resolve() == path, "Unexpected receipt location")
            data = path.read_bytes()
            require(len(data) == row["bytes"] and digest(data) == row["sha256"], "Receipt identity mismatch")
            receipt = json.loads(data)
            task = receipt["task"]
            require(task["id"] == f"{prefix}_s{seed}" and task["seed"] == seed and task["method"] == reader
                    and task["domains"] == ["AlphaBay", "Dream", "Agora"]
                    and receipt["training_identity"]["task"] == task
                    and receipt["training_identity"]["manifest"]["sha256"] == bindings["manifest_sha256"],
                    "Trajectory or original manifest mismatch")
            require(receipt["status"] == "BLIND_SCORES_COMPLETE" and receipt["role"] == "dev"
                    and receipt["reference_role"] == "fit" and receipt["reference_records"] == 12,
                    "Incomplete or different scoring protocol")
            require(set(bindings["roles"]) == {f"{d}:{r}" for d in MARKETS for r in ("dev", "fit")}
                    and all(receipt["sources"][k]["sha256"] == v for k, v in bindings["roles"].items()),
                    "Query/reference row identity mismatch")
            require(set(CELLS) == set(row["metrics"]), "Incomplete endpoint anchors")
        except (OSError, KeyError, TypeError, ValueError) as exc:
            errors.append({"family": family, "seed": seed, "reason": str(exc)})
            continue
        for cell in CELLS:
            try:
                require(cell in receipt["cells"], "Cell absent")
                record = receipt["files"][f"{reader}:{cell}"]
                score = path.parent / f"{reader}_{cell}.npy"
                require(record["path"] == str(score) and score.resolve() == score, "Unexpected matrix location")
                require(score.is_file() and score.stat().st_size == record["bytes"], "Missing matrix or wrong size")
                selected[(family, seed, cell)] = (score, record, row["metrics"][cell])
            except (OSError, KeyError, TypeError, ValueError) as exc:
                errors.append({"family": family, "seed": seed, "cell": cell, "reason": str(exc)})
    return selected, errors


def read_endpoint(item, size):
    path, record, anchors = item
    data = path.read_bytes()
    require(len(data) == record["bytes"] and digest(data) == record["sha256"], "Matrix identity mismatch")
    scores = np.load(io.BytesIO(data), allow_pickle=False)
    require(scores.shape == (size, size) and scores.dtype.kind == "f" and np.isfinite(scores).all(),
            "Invalid score matrix")
    scores = scores.astype(np.float64, copy=False)
    truth = np.arange(size)
    order = np.argsort(-scores, axis=1, kind="stable")
    ranks = (order == truth[:, None]).argmax(axis=1) + 1
    winners = order[:, 0]
    hits = (winners == truth)
    predicted = np.bincount(winners, minlength=size)
    metrics = {"Top1": float(hits.mean()), "MAP": float((1.0 / ranks).mean()),
               "macro_F1": float((2.0 * hits / (1 + predicted)).mean())}
    require(set(metrics) == set(anchors) and all(abs(v - anchors[k]) <= 1e-12 for k, v in metrics.items()),
            "Saved endpoint metric or stable ordering mismatch")
    return scores, ranks


def tied_ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def spearman(x, y):
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    return float(np.corrcoef(tied_ranks(x), tied_ranks(y))[0, 1])


def transition(before, after):
    scores, rank0 = before
    new_scores, rank1 = after
    delta = new_scores - scores
    mean = delta.mean(axis=1)
    shift = mean ** 2
    relative = ((delta - mean[:, None]) ** 2).mean(axis=1)
    absolute = (delta ** 2).mean(axis=1)
    require(np.allclose(absolute, shift + relative, rtol=1e-11, atol=1e-13), "Rowwise MSE decomposition failed")
    first, last = rank0 == 1, rank1 == 1
    groups = {"all": np.ones(len(rank0), dtype=bool), "kept_first": first & last,
              "lost_first": first & ~last, "gained_first": ~first & last, "neither_first": ~first & ~last}
    result = {}
    for name, selected in groups.items():
        count = int(selected.sum())
        a, b = rank0[selected], rank1[selected]
        mse = float(absolute[selected].mean()) if count else None
        offset = float(shift[selected].mean()) if count else None
        result[name] = dict(n=count, abs_mse=mse, shift_mse=offset,
                            relative_mse=float(relative[selected].mean()) if count else None,
                            shift_share=offset / mse if mse else None,
                            rank_before=float(a.mean()) if count else None, rank_after=float(b.mean()) if count else None,
                            fell=int((b > a).sum()), unchanged=int((b == a).sum()), rose=int((b < a).sum()))
    severity = {}
    for name, mask in (("initially_first", first), ("lost_first", groups["lost_first"])):
        loss = rank1[mask] - 1
        severity[name] = {"n": int(mask.sum()), "rho_absolute": spearman(absolute[mask], loss),
                          "rho_relative": spearman(relative[mask], loss)}
    return {"groups": result, "rank_loss_association": severity}


def common_risk(left_start, left_end, right_start, right_end):
    common = (left_start == 1) & (right_start == 1)
    left_loss = common & (left_end > 1)
    right_loss = common & (right_end > 1)
    return dict(common_start_correct=int(common.sum()), left_lost=int(left_loss.sum()),
                right_lost=int(right_loss.sum()), both_lost=int((left_loss & right_loss).sum()))


def run(root, bindings, output, mode, sizes):
    output.mkdir(exist_ok=False)
    opened = []
    current = None
    try:
        selected, errors = inspect_inputs(root, bindings)
        save(output / "inventory.json", dict(trajectories=9, logical_cells=len(selected), errors=errors,
                                             payloads_opened=0, stage="metadata"))
        require(not errors and len(selected) == 45, "Incomplete inventory; see inventory.json")
        if mode == "inventory":
            return
        transitions, paired = [], []
        with (output / "access.jsonl").open("x", encoding="utf-8", buffering=1) as access:
            for seed in SEEDS:
                endpoints = {}
                for family in FAMILIES:
                    for cell, domain in CELLS.items():
                        current = {"method": family, "seed": seed, "cell": cell}
                        item = selected[family, seed, cell]
                        access.write(json.dumps({**current, "planned_sha256": item[1]["sha256"]}) + "\n")
                        opened.append(current)
                        endpoints[family, cell] = read_endpoint(item, sizes[domain])
                for start, end in TRANSITIONS:
                    key = dict(seed=seed, domain=CELLS[start], start=start, end=end)
                    for family in FAMILIES:
                        desc = transition(endpoints[family, start], endpoints[family, end])
                        transitions.append(dict(**key, method=family, **desc))
                    for left, right in itertools.combinations(FAMILIES, 2):
                        counts = common_risk(endpoints[left, start][1], endpoints[left, end][1],
                                             endpoints[right, start][1], endpoints[right, end][1])
                        paired.append(dict(**key, left=left, right=right, **counts))
        require(len(opened) == 45 and len(transitions) == 36 and len(paired) == 36, "Incomplete batch")
        save(output / "aggregates.json", dict(status="COMPLETE", opened=45, verified_anchor_values=135,
                                              transitions=transitions, paired=paired))
    except Exception as exc:
        save(output / "failure.json", dict(status="STOP", reason=str(exc), last_endpoint_attempt=current,
                                           payload_attempts=len(opened)))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inventory", "analyze"))
    args = parser.parse_args()
    require(sys.platform == "linux" and Path(sys.executable).resolve() ==
            Path("/home/yongpeng/miniconda3/envs/py310/bin/python").resolve(), "Use the project Linux py310")
    require(0 < len(os.sched_getaffinity(0)) <= 2 and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Use at most two bound CPU cores and no GPU")
    work = Path(__file__).resolve().parent
    pins = json.loads((work / "bindings.json").read_text())
    run(work.parents[3], pins, work / args.mode, args.mode, MARKETS)
