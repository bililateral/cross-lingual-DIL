"""Independent hand cases and the production batch with invented score files."""
import copy
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

import diagnose as d


def anchors(scores):
    """Independent Python ordering and per-class confusion counts."""
    size = len(scores)
    orders = [sorted(range(size), key=lambda j: (-float(scores[i, j]), j)) for i in range(size)]
    wins = [row[0] for row in orders]
    ranks = [row.index(i) + 1 for i, row in enumerate(orders)]
    return {"Top1": sum(rank == 1 for rank in ranks) / size,
            "MAP": sum(1 / rank for rank in ranks) / size,
            "macro_F1": sum(2 * int(wins[i] == i) / (1 + wins.count(i)) for i in range(size)) / size}


def fake(root):
    roles = {f"{market}:{role}": d.digest(f"{market}:{role}".encode())
             for market in d.MARKETS for role in ("fit", "dev")}
    source_hash = d.digest(b"fixture manifest")
    sizes = {"AlphaBay": 4, "Dream": 3}
    pins = {"manifest_sha256": source_hash, "roles": roles, "trajectories": []}
    for idx, (family, (_, prefix, reader)) in enumerate(d.FAMILIES.items()):
        for seed in d.SEEDS:
            path = d.receipt_path(root, family, seed)
            path.parent.mkdir(parents=True)
            task = dict(id=f"{prefix}_s{seed}", seed=seed, method=reader, domains=["AlphaBay", "Dream", "Agora"])
            receipt = dict(task=task, training_identity=dict(task=task, manifest=dict(sha256=source_hash)),
                           status="BLIND_SCORES_COMPLETE", role="dev", reference_role="fit", reference_records=12,
                           sources={k: {"sha256": v} for k, v in roles.items()}, files={}, cells=list(d.CELLS))
            expected = {}
            for cell, market in d.CELLS.items():
                n = sizes[market]
                scores = np.eye(n) * 2.0 + np.arange(n)[:, None] + idx + seed
                if cell in ("R21", "R31", "R32"):
                    scores[1, 0] += 2  # Tied with the true column: earlier column wins.
                if cell == "R31" and family == "quarter":
                    scores[2, 0] += 3
                dest = path.parent / f"{reader}_{cell}.npy"
                np.save(dest, scores)
                receipt["files"][f"{reader}:{cell}"] = dict(path=str(dest), bytes=dest.stat().st_size,
                                                           sha256=d.digest(dest.read_bytes()))
                expected[cell] = anchors(scores)
            d.save(path, receipt)
            pins["trajectories"].append(dict(family=family, seed=seed, bytes=path.stat().st_size,
                                             sha256=d.digest(path.read_bytes()), metrics=expected))
    return pins, sizes


def main():
    assert sys.platform == "linux" and Path(sys.executable).resolve() == Path("/home/yongpeng/miniconda3/envs/py310/bin/python").resolve()
    assert 0 < len(os.sched_getaffinity(0)) <= 2 and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    before = np.array([[2., 2., 0.], [0., 3., 0.], [0., 0., 2.]])
    after = np.array([[5., 5., 3.], [3., 3., 0.], [0., 0., 2.]])
    rank0, rank1 = np.array([1, 1, 1]), np.array([1, 2, 1])
    hand = d.transition((before, rank0), (after, rank1))["groups"]
    assert anchors(after) == {"Top1": 2 / 3, "MAP": 5 / 6, "macro_F1": (2 / 3 + 1) / 3}
    assert hand["kept_first"]["n"] == 2 and hand["kept_first"]["shift_mse"] == 4.5
    assert hand["kept_first"]["relative_mse"] == 0 and hand["kept_first"]["shift_share"] == 1
    assert hand["lost_first"]["abs_mse"] == 3 and hand["lost_first"]["shift_mse"] == 1
    assert hand["lost_first"]["relative_mse"] == 2 and hand["lost_first"]["fell"] == 1
    assert hand["gained_first"]["n"] == 0 and hand["gained_first"]["relative_mse"] is None
    reverse = d.transition((after, rank1), (before, rank0))["groups"]
    assert reverse["gained_first"]["n"] == 1 and reverse["gained_first"]["rose"] == 1
    zero = d.transition((before, rank0), (before, rank0))
    assert zero["groups"]["all"]["shift_share"] is None and zero["rank_loss_association"]["initially_first"]["rho_relative"] is None
    assert np.isclose(d.spearman(np.array([1, 1, 9]), np.array([2, 2, 0])), -1.)
    pair = d.common_risk(np.array([1, 1, 1, 2]), np.array([2, 1, 2, 2]),
                         np.array([1, 1, 2, 1]), np.array([2, 2, 2, 2]))
    assert pair == dict(common_start_correct=2, left_lost=1, right_lost=2, both_lost=1)
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp).resolve()
        pins, sizes = fake(root)
        d.run(root, pins, root / "inventory", "inventory", sizes)
        assert not (root / "inventory/access.jsonl").exists()
        d.run(root, pins, root / "success", "analyze", sizes)
        released = json.loads((root / "success/aggregates.json").read_text())
        assert len(released["transitions"]) == len(released["paired"]) == 36 and released["opened"] == 45
        assert len((root / "success/access.jsonl").read_text().splitlines()) == 45
        old_to_last = next(r for r in released["transitions"] if r["method"] == "quarter" and r["seed"] == 51
                           and r["start"] == "R11" and r["end"] == "R31")
        assert old_to_last["groups"]["lost_first"]["n"] == 2
        try:
            d.run(root, pins, root / "success", "analyze", sizes)
        except FileExistsError:
            pass
        else:
            raise AssertionError("Repeated opening allowed")
        incorrect_role = copy.deepcopy(pins)
        incorrect_role["roles"]["Dream:dev"] = "wrong"
        _, errors = d.inspect_inputs(root, incorrect_role)
        assert len(errors) == 9
        wrong_anchor = copy.deepcopy(pins)
        wrong_anchor["trajectories"][0]["metrics"]["R11"]["Top1"] = -1
        last = d.receipt_path(root, "quarter", 53).parent / "H_full_R32.npy"
        for failure, config, count in (("anchor", wrong_anchor, 1), ("corrupt", pins, 45), ("missing", pins, 0)):
            if failure == "corrupt":
                payload = last.read_bytes()
                last.write_bytes(payload[:-1] + bytes([payload[-1] ^ 1]))
            if failure == "missing":
                last.unlink()  # Invented fixture inside TemporaryDirectory.
            try:
                d.run(root, config, root / failure, "analyze", sizes)
            except ValueError:
                receipt = json.loads((root / failure / "failure.json").read_text())
                assert receipt["payload_attempts"] == count and not (root / failure / "aggregates.json").exists()
            else:
                raise AssertionError("Unexpected release")
    print(json.dumps(dict(status="PASS", runtime=sys.executable, affinity=sorted(os.sched_getaffinity(0)),
                          real_payloads=0, fixture_success_cells=45, complete_method_transitions=36, paired=36,
                          hand_lost_row={"absolute": 3, "shift": 1, "relative": 2, "rank_before": 1, "rank_after": 2},
                          hand_kept_rows={"n": 2, "shift": 4.5, "relative": 0}, common_risk=pair,
                          guarded=["wrong_source", "wrong_anchor", "last_hash", "missing_file", "duplicate_release"])))


if __name__ == "__main__":
    main()
