"""Count actual late winners among five rivals nominated at market exit."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import sys
import time

import numpy as np
import psutil


WORK = Path(__file__).resolve().parent
PREVIOUS = WORK.parent / "vendorlink_score_drift"
PROJECT = WORK.parents[3]
SOURCE = "fdf8f1ced64e519861098134638b814e8ce33829b5e2b3ddf56cbf1fe6abb8e5"
BINDINGS = "95ab26273d0cca0b3dbb90301b4cb6e301cac2c51d10673d403e0c3f99509fe6"
AGGREGATES = "42dada4d390cdbb27e61aed72797d4fbc28abf5fffd124ec6caaf3b708500c03"
MARKETS = (("AlphaBay", 546, "R11", "R31"), ("Dream", 157, "R22", "R32"))
SEEDS = (51, 52, 53)
K = 5


def pinned(path, expected):
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError(f"Source binding mismatch: {path.name}")
    return data


pinned(PREVIOUS / "diagnose.py", SOURCE)
sys.path.insert(0, str(PREVIOUS))
from diagnose import read_endpoint, receipt_path, require, save


def count_challengers(start, finish):
    initial, initial_ranks = start
    final, final_ranks = finish
    n = len(initial)
    require(initial.shape == final.shape == (n, n) and n - 1 >= K, "Invalid pool")
    correct = np.flatnonzero(initial_ranks == 1)
    ordered = np.argsort(-initial[correct], axis=1, kind="stable")
    require(np.array_equal(ordered[:, 0], correct), "Initial truth or tie order mismatch")
    top_wrong = ordered[:, 1:K + 1]
    winners = final.argmax(axis=1)
    require(np.array_equal(winners == np.arange(n), final_ranks == 1), "Final tie order mismatch")
    lost = winners[correct] != correct
    losses = int(lost.sum())
    hits = int(np.any(top_wrong[lost] == winners[correct[lost], None], axis=1).sum())
    return dict(accounts=n, initially_first=len(correct), lost_first=losses, top5_hits=hits,
                outside_top5=losses - hits, hit_rate=hits / losses if losses else None,
                random_draws=K, random_wrong_pool=n - 1, random_hit_probability=K / (n - 1),
                random_expected_hits=losses * K / (n - 1))


def hand_check():
    # Start with four steady correct rows and two previously wrong rows (one later gains).
    initial = np.tile(np.arange(7, -1, -1, dtype=float), (8, 1))
    np.fill_diagonal(initial, 10)
    initial[0:2, 5:7] = 2
    initial[2, 0], initial[3, 0] = 12, 10
    final = initial.copy()
    final[0, 5:7], final[1, 6], final[2, 2], final[3, 7] = 15, 15, 13, 14

    # Compute truth positions independently, including lower-index ties.
    def with_ranks(scores):
        truth = np.diag(scores)[:, None]
        earlier_column = np.arange(8)[None, :] < np.arange(8)[:, None]
        rank = 1 + ((scores > truth) | ((scores == truth) & earlier_column)).sum(axis=1)
        return scores, rank

    def run(before, after):
        return count_challengers(with_ranks(before), with_ranks(after))

    sample = run(initial, final)
    require((sample["initially_first"], sample["lost_first"], sample["top5_hits"],
             sample["outside_top5"], sample["hit_rate"]) == (6, 2, 1, 1, .5), "Hand tally mismatch")
    require(sample["random_hit_probability"] == 5 / 7 and sample["random_expected_hits"] == 10 / 7,
            "Wrong random denominator")
    swapped = initial.copy()
    swapped[0, 6] += .5
    require(run(swapped, final)["top5_hits"] == 0, "Nomination must use initial scores only")
    recovered = final.copy()
    recovered[1, 5] = 16
    require(run(initial, recovered)["top5_hits"] == 2, "Count the final winner's identity")
    empty = run(initial, initial)
    require(empty["lost_first"] == 0 and empty["hit_rate"] is None
            and empty["random_expected_hits"] == 0, "Empty denominator")
    return dict(status="PASS", hand_tally=sample, changed_nomination_hits=0, changed_winner_hits=2,
                no_loss_rate=None, input="handwritten_arrays_only")


def inspect_er():
    pins = json.loads(pinned(PREVIOUS / "bindings.json", BINDINGS))
    expected_rows = [r for r in pins["trajectories"] if r["family"] == "ER"]
    require(len(expected_rows) == 3 and {r["seed"] for r in expected_rows} == set(SEEDS), "ER binding")
    selected = {}
    for row in expected_rows:
        seed = row["seed"]
        receipt = receipt_path(PROJECT, "ER", seed)
        require(receipt.resolve() == receipt and receipt.stat().st_size == row["bytes"], "Receipt location/size")
        data = json.loads(pinned(receipt, row["sha256"]))
        require(data["task"]["id"] == f"M_H_ER_s{seed}" and data["task"]["seed"] == seed
                and data["task"]["method"] == "H_ER" and data["role"] == "dev"
                and data["reference_role"] == "fit" and data["reference_records"] == 12
                and data["status"] == "BLIND_SCORES_COMPLETE"
                and data["training_identity"]["manifest"]["sha256"] == pins["manifest_sha256"]
                and all(data["sources"][k]["sha256"] == v for k, v in pins["roles"].items()), "Scoring identity")
        for _, _, start, end in MARKETS:
            for cell in (start, end):
                record = data["files"][f"H_ER:{cell}"]
                path = receipt.parent / f"H_ER_{cell}.npy"
                require(cell in data["cells"] and str(path) == record["path"] and path.resolve() == path
                        and path.is_file() and path.stat().st_size == record["bytes"], "Missing or unexpected matrix")
                selected[seed, cell] = (path, record, row["metrics"][cell])
    require(len(selected) == 12, "Incomplete preflight")
    return selected


def analyze(output):
    require(json.loads((WORK / "check.json").read_text())["status"] == "PASS", "Synthetic check missing")
    old = json.loads(pinned(PREVIOUS / "analyze/aggregates.json", AGGREGATES))
    anchors = {(r["seed"], r["start"], r["end"]): r for r in old["transitions"] if r["method"] == "ER"}
    selected = inspect_er()
    rows = []
    with (output / "access.jsonl").open("x", encoding="utf-8", buffering=1) as access:
        for market, size, start, end in MARKETS:
            for seed in SEEDS:
                pair = []
                for cell in (start, end):
                    item = selected[seed, cell]
                    access.write(json.dumps(dict(seed=seed, market=market, cell=cell,
                                                 bytes=item[1]["bytes"], sha256=item[1]["sha256"])) + "\n")
                    pair.append(read_endpoint(item, size))
                counts = count_challengers(*pair)
                previous = anchors[seed, start, end]
                require(counts["initially_first"] == previous["rank_loss_association"]["initially_first"]["n"]
                        and counts["lost_first"] == previous["groups"]["lost_first"]["n"], "Denominator mismatch")
                rows.append(dict(market=market, seed=seed, start=start, end=end, **counts))
    require(len(rows) == 6, "Incomplete batch")
    return dict(status="COMPLETE", method="H_ER", role="dev", reference="FIT12", reads=12,
                verified_original_metric_values=36, verified_previous_counts=12, rows=rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "analyze"))
    mode = parser.parse_args().mode
    require(sys.platform == "linux" and Path(sys.executable).resolve() ==
            Path("/home/yongpeng/miniconda3/envs/py310/bin/python").resolve()
            and len(os.sched_getaffinity(0)) == 1 and os.environ.get("CUDA_VISIBLE_DEVICES") == "",
            "Use the project Linux py310, one CPU core and no GPU")
    if mode == "analyze":
        destination = WORK / "result"
        destination.mkdir(mode=0o700, exist_ok=False)
    else:
        destination = WORK
        require(not (WORK / "check.json").exists(), "Check already recorded")
    clock = time.monotonic()
    started = datetime.now().astimezone().isoformat()
    resource = dict(owner=pwd.getpwuid(os.getuid()).pw_name, pid=os.getpid(), cwd=str(Path.cwd()),
                    cpus=sorted(os.sched_getaffinity(0)), available_ram_bytes=psutil.virtual_memory().available,
                    available_disk_bytes=psutil.disk_usage(WORK).free, gpu=0)
    try:
        result = hand_check() if mode == "check" else analyze(destination)
    except Exception as exc:
        save(destination / f"{mode}_failure.json", dict(reason=str(exc), started=started, **resource,
                                                       elapsed_seconds=round(time.monotonic() - clock, 6)))
        raise
    elapsed = time.monotonic() - clock
    result.update(started=started, finished=datetime.now().astimezone().isoformat(),
                  process_body_seconds=round(elapsed, 6), body_seconds_rounded=math.ceil(elapsed), **resource)
    result["source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    save(destination / ("check.json" if mode == "check" else "summary.json"), result)
    print(result["status"])
