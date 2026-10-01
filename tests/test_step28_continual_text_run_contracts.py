"""New loader/scheduler/publication boundaries; all fixtures are hand-created."""
from __future__ import annotations

import copy
import csv
import hashlib
import itertools
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import step28_continual_data as shared
import step28_continual_text as core
import step28_continual_text_data as data
import step28_continual_text_evaluate as evaluate
import step28_continual_text_run as runner
from test_step28_continual_text_contracts import CharacterTokenizer, toy_encoder, torch


def make_archives(root: Path, training_worlds: int = 5, sellers: int = 4) -> tuple[data.WorldArchive, data.WorldArchive]:
    """Complete hand graphs and harmless text, NEVER a new research dataset."""
    archives = []
    for split, count in (("train", training_worlds), ("development", 2)):
        path = root / split
        path.mkdir()
        pairs = sellers * (sellers - 1) // 2
        items = 99 if sellers == 28 else sellers
        rng = np.random.default_rng(805 + (split == "development"))
        np.save(path / "base.npy", rng.normal(size=(count * pairs, 24)))
        np.save(path / "identity.npy", rng.normal(size=(count * pairs, 33)))
        labels = []
        with (path / "rows.csv").open("w", encoding="utf-8", newline="") as rh, (
                path / "texts.jsonl").open("w", encoding="utf-8", newline="\n") as th:
            writer = csv.writer(rh)
            writer.writerow(shared.ROW_FIELDS)
            for ordinal in range(count):
                uid = f"{split}_world_{ordinal:03d}"
                accounts = [f"{uid}_a{i:02d}" for i in range(sellers)]
                for left, right in itertools.combinations(accounts, 2):
                    writer.writerow([split, ordinal, uid, left + "||" + right, left, right])
                    a, b = int(left[-2:]), int(right[-2:])
                    group = lambda i: i // 3 if sellers == 28 and i < 12 else (4 + (i - 12) // 2 if sellers == 28 else i // 2)
                    labels.append(int(group(a) == group(b)))
                for i in range(items):
                    a = i % sellers
                    row = {"world_uid": uid, "seller_uid": accounts[a], "item_uid": f"{uid}_item{i:03d}",
                           "title": f"普通商品{a}，样例{ordinal}。", "description": f"包装说明{a}，核验文本。"}
                    th.write(json.dumps(row, ensure_ascii=False) + "\n")
        label_path = path / "labels.npy" if split == "train" else None
        if label_path is not None:
            np.save(label_path, np.asarray(labels, dtype=np.uint8))
        archives.append(data.WorldArchive(split, path / "base.npy", path / "identity.npy",
                                         path / "rows.csv", path / "texts.jsonl", world_count=count,
                                         sellers=sellers, items=items, labels=label_path))
    return tuple(archives)


def assert_order_result(result: dict, output: Path, *, stages: int, worlds_per_stage: int,
                        development_worlds: int, memory_budget: int) -> None:
    evaluate.validate_order(result, output, stages=stages, worlds_per_stage=worlds_per_stage,
                            development_worlds=development_worlds, memory_budget=memory_budget,
                            arrival_stages=[[3], [0], [4], [1], [2]])
    assert result["physical_updates"] == 46
    assert len(result["points"]) == 14
    assert len(list(output.rglob("model.pt"))) == 4
    assert len(list(output.rglob("logits.npy"))) == 14
    assert not list(output.rglob("state.pt"))


def fake_order(base: Path, seed: int, arrival: list[list[int]], *, development_worlds: int = 2,
               budget: int = 8192) -> dict:
    """Artificial metadata/files for fail-before-truth tests, not training evidence."""
    base.mkdir()
    n = len(arrival[0])
    def memory(stage):
        return {"seen_unique_rows": stage * n * 378, "last_stage": stage, "records": 4,
                "positive_records": 1, "retained_accounts": 8, "total_bytes": 4096,
                "sha256": str(stage) * 64, "budget": budget}
    logs = {arm: [] for arm in runner.ARMS}
    first = {**runner.expected_counts(n), "new_arrival_presentations": n * 756, "prior_arrival_presentations": 0,
             "mean_update_loss": .7, "current_order_sha256": "a" * 64,
             "history_before_training": None, "shared_physical_fit": True}
    for arm in runner.ARMS:
        logs[arm].append(copy.deepcopy(first))
        for stage in range(2, 6):
            old = memory(stage - 1) if arm == "er" else None
            frozen = arm == "frozen"
            count = 0 if frozen else n * (stage if arm == "cumulative" else 1)
            logs[arm].append({**runner.expected_counts(count, history_edges=4 if old else 0),
                             "new_arrival_presentations": 0 if frozen else n * 756,
                             "prior_arrival_presentations": (stage - 1) * n * 756 if arm == "cumulative" else 0,
                             "mean_update_loss": .6, "current_order_sha256": str(stage) * 64,
                             "history_before_training": old})
    points = {}
    names = ["initial", "shared"] + [f"{arm}_stage{s}" for arm in runner.ARMS[1:] for s in range(2, 6)]
    for name in names:
        dest = base / name
        dest.mkdir()
        np.save(dest / "logits.npy", np.zeros(development_worlds * 378, dtype=np.float32))
        stage = 0 if name == "initial" else 1 if name == "shared" else int(name[-1])
        arm = 0 if stage < 2 else runner.ARMS.index(name.split("_stage")[0])
        model = None
        if name == "shared" or stage == 5:
            (dest / "model.pt").write_bytes(b"HAND_FIXTURE_NOT_A_MODEL")
            model = shared.record(dest / "model.pt", base)
        value = {"scores": shared.record(dest / "logits.npy", base), "inference_model": model,
                 "state_signature": "b" * 64, "temporary_state": {"path": "state.pt", "size_bytes": 1, "sha256": "c" * 64},
                 "progress": {"order_seed": seed, "stage": stage, "arm": arm},
                 "full_state_and_all_scores_exact": True, "temporary_state_removed": True,
                 "score_passes": 2, "scored_world_presentations": 2 * development_worlds,
                 "memory": memory(stage) if name == "shared" or name.startswith("er_stage") else None}
        shared.write_json(dest / "point.json", value)
        points[name] = value
    np.savez(base / "scaler.npz", medians=np.zeros(57), means=np.zeros(57), scales=np.ones(57))
    access = [{"arm": "shared", "stage": 1, "scope": "current", "world_ordinals": arrival[0], "rows": n * 378}]
    for arm in runner.ARMS[1:]:
        for stage in range(2, 6):
            worlds = [w for g in arrival[:stage] for w in g] if arm == "cumulative" else arrival[stage - 1]
            access.append({"arm": arm, "stage": stage, "scope": "arrived_prefix" if arm == "cumulative" else "current",
                           "world_ordinals": worlds, "rows": len(worlds) * 378})
    order = {"seed": seed, "points": points, "trajectory": runner.trajectory(), "training": logs,
             "physical_updates": 46 * n, "first_memory": memory(1), "stage_accesses": access,
             "scaler": shared.record(base / "scaler.npz", base), "first_stage_shared_model_and_adam": True,
             "cumulative_resets_original_and_empty_adam": True, "frozen_artifact_unchanged": True}
    shared.write_json(base / "order.json", order)
    return order


class TextRunContracts(unittest.TestCase):
    def test_raw_columns_and_complete_text_align_without_stale_six_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, dev = make_archives(Path(temporary))
            data.disjoint_archives(train, dev)
            batch = train.read(3)
            expected_base = np.load(train.base)[18:24, :18]
            expected_identity = np.load(train.identity)[18:24]
            np.testing.assert_array_equal(batch.numeric, np.column_stack((expected_base, expected_identity)))
            self.assertEqual(batch.edges, tuple(itertools.combinations(train.worlds[3][1], 2)))
            self.assertEqual(batch.accounts[batch.edges[0][0]]["title"], ("普通商品0，样例3。",))
            self.assertIsNone(dev.read(0).labels)
            self.assertFalse(any(isinstance(v, np.ndarray) for v in vars(train).values()))
            batch.numeric[:] = -999
            self.assertFalse(np.all(train.read(3).numeric == -999))

    def test_only_current_or_explicit_cumulative_prefix_is_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, _ = make_archives(Path(temporary))
            source = data.StageSource(train, [[3], [0], [4], [1], [2]])
            with mock.patch.object(train, "read", wraps=train.read) as read:
                source.load("er", 2)
                self.assertEqual(read.call_args_list, [mock.call(0)])
                read.reset_mock()
                source.load("cumulative", 2)
                self.assertEqual(read.call_args_list, [mock.call(3), mock.call(0)])
            for arm, stage in (("shared", 2), ("er", 1), ("frozen", 2), ("er", 6)):
                with self.assertRaises(ValueError):
                    source.load(arm, stage)
            with self.assertRaises(ValueError):
                data.StageSource(train, [[0], [0], [2], [3], [4]])

    def test_text_endpoint_mismatch_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, _ = make_archives(Path(temporary))
            raw = train.texts.read_bytes().replace(b"train_world_000_a00", b"train_world_000_x00")
            train.texts.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "endpoints"):
                train.read(0)

    def test_pair_permutation_and_noncontiguous_text_worlds_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train, _ = make_archives(root)
            rows = root / "train/rows.csv"
            original = rows.read_text(encoding="utf-8")
            lines = original.splitlines()
            lines[1], lines[2] = lines[2], lines[1]
            rows.write_text("\n".join(lines) + "\n", encoding="utf-8")
            arguments = ("train", train.base, train.identity, rows, train.texts)
            with self.assertRaisesRegex(ValueError, "Pair order"):
                data.WorldArchive(*arguments, world_count=5, sellers=4, items=4)
            rows.write_text(original, encoding="utf-8")
            text = train.texts.read_text(encoding="utf-8")
            train.texts.write_text(text.replace('"world_uid": "train_world_000"', '"world_uid": "train_world_001"', 1), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Text schema or world order"):
                data.WorldArchive(*arguments, world_count=5, sellers=4, items=4)

    def test_development_labels_and_overlapping_public_ids_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            train, dev = make_archives(Path(temporary))
            with self.assertRaisesRegex(ValueError, "label-free"):
                data.WorldArchive("development", train.base, train.identity, train.base, train.texts, labels=train.labels)
            dev.worlds[0] = train.worlds[0]
            with self.assertRaises(ValueError):
                data.disjoint_archives(train, dev)

    def test_expected_budget_counts_world_updates_and_actual_replay(self):
        self.assertEqual(runner.expected_counts(100), {"updates": 200, "current_presentations": 75600, "replay_presentations": 0})
        self.assertEqual(runner.expected_counts(100, history_edges=7)["replay_presentations"], 1400)
        self.assertEqual(runner.expected_counts(100, history_edges=50)["replay_presentations"], 3200)
        self.assertEqual(3 * (200 + 2 * 4 * 200 + sum(200 * k for k in range(2, 6))), 13800)
        self.assertEqual(len({p for v in runner.trajectory().values() for p in v}), 14)

    def test_valid_artificial_receipts_are_accepted_then_budget_and_access_mutations_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "order11"
            stages = [[3], [0], [4], [1], [2]]
            order = fake_order(base, 11, stages)
            validate = lambda value: evaluate.validate_order(value, base, worlds_per_stage=1,
                        development_worlds=2, memory_budget=8192, arrival_stages=stages)
            validate(order)
            for mutation in ("future", "updates", "shared", "old_rng", "trajectory"):
                bad = copy.deepcopy(order)
                if mutation == "future": bad["stage_accesses"][1]["world_ordinals"] = [4]
                if mutation == "updates": bad["training"]["cumulative"][2]["updates"] -= 1
                if mutation == "shared": bad["training"]["er"][0]["shared_physical_fit"] = False
                if mutation == "old_rng": bad["training"]["er"][2]["history_before_training"]["sha256"] = "f" * 64
                if mutation == "trajectory": bad["trajectory"]["frozen"][3] = "er_stage3"
                with self.assertRaises(ValueError, msg=mutation): validate(bad)

    def test_last_score_or_retained_model_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "order11"
            stages = [[3], [0], [4], [1], [2]]
            order = fake_order(base, 11, stages)
            for name in ("cumulative_stage5/logits.npy", "shared/model.pt"):
                path = base / name
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaises(ValueError):
                    evaluate.validate_order(order, base, worlds_per_stage=1, development_worlds=2,
                                            memory_budget=8192, arrival_stages=stages)
                path.write_bytes(original)

    def test_unexpected_retained_cache_or_score_mapping_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary) / "order11"
            stages = [[3], [0], [4], [1], [2]]
            order = fake_order(base, 11, stages)
            for mode in ("cache", "score", "stage", "retention"):
                bad = copy.deepcopy(order)
                point = bad["points"]["sequential_stage2"]
                if mode == "cache": point["memory"] = order["first_memory"]
                if mode == "score": point["scores"] = order["points"]["initial"]["scores"]
                if mode == "stage": point["progress"]["stage"] = 4
                if mode == "retention": point["inference_model"] = order["points"]["shared"]["inference_model"]
                path = base / "sequential_stage2/point.json"
                path.write_text(json.dumps(point), encoding="utf-8")
                with self.assertRaises(ValueError, msg=mode):
                    evaluate.validate_order(bad, base, worlds_per_stage=1, development_worlds=2,
                                            memory_budget=8192, arrival_stages=stages)

    def test_missing_manifest_blocks_development_truth(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            shared.write_json(root / "manifest.json", {"status": "PARTIAL"})
            with mock.patch.object(data, "load_settings", return_value={}), mock.patch.object(data, "verify_public"), mock.patch.object(shared, "aligned_labels") as labels:
                with self.assertRaises(ValueError): evaluate.evaluate(root, root / "evaluation")
                labels.assert_not_called()

    def test_complete_42_point_gate_rejects_last_missing_score_before_truth(self):
        settings = data.load_settings()
        arrival = shared.read_json(data.ROOT / settings["benchmark"]["arrival"]["path"])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            orders = [fake_order(root / f"order{o['seed']}", o["seed"], o["train_world_ordinals_by_stage"],
                                 development_worlds=500, budget=524288) for o in arrival["orders"]]
            manifest = {"status": "ALL_TEXT_FOUR_ARM_SCORES_SAVED_AND_REPLAYED_NO_DEVELOPMENT_LABELS",
                        "settings": settings, "code": data.code_records(), "orders": orders,
                        "label_reads": {"train_csv_offline_packaging": 1, "development": 0, "qrels": 0, "audit_a": 0, "audit_b": 0},
                        "temporary_work_removed": True, "physical_updates": 13800}
            shared.write_json(root / "manifest.json", manifest)
            self.assertEqual(evaluate.validate_run(root, settings), manifest)
            original_verify = shared.verify
            def public_only_verify(spec, base=shared.ROOT):
                if spec == settings["benchmark"]["development_labels"]:
                    return base / spec["path"]  # No private bytes opened in this fixture.
                return original_verify(spec, base)
            with mock.patch.object(data, "verify_public"), mock.patch.object(shared, "verify", side_effect=public_only_verify), mock.patch.object(
                    shared, "aligned_labels", side_effect=RuntimeError("reached label parser")) as labels:
                with self.assertRaisesRegex(RuntimeError, "reached label parser"):
                    evaluate.evaluate(root, root / "evaluation")
                labels.assert_called_once()
            (root / "order37/cumulative_stage5/logits.npy").unlink()
            with mock.patch.object(data, "verify_public"), mock.patch.object(shared, "aligned_labels") as labels:
                with self.assertRaises(FileNotFoundError): evaluate.evaluate(root, root / "evaluation")
                labels.assert_not_called()
            self.assertFalse((root / "evaluation").exists())

    def test_failure_receipt_keeps_starting_and_observed_code_separate(self):
        """Exercise failure publication only; GPU and private inputs are simulated."""
        settings = data.load_settings()
        environment = shared.read_json(data.ROOT / settings["run"]["component_runtime"]["path"])["environment"]
        started = data.code_records()
        changed = copy.deepcopy(started)
        changed[0]["sha256"] = "f" * 64
        fake_torch = mock.MagicMock()
        fake_torch.__version__ = environment["torch"]
        fake_torch.version.cuda = environment["cuda"]
        fake_torch.backends.cudnn.version.return_value = environment["cudnn"]
        fake_torch.cuda.mem_get_info.return_value = (25 * 1024**3, 32 * 1024**3)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            check = root / "check.json"
            shared.write_json(check, {"status": "PASSED_REAL_CPU_TEXT_ORCHESTRATION_CHECK", "code": started,
                                      "environment": {"torch": environment["torch"], "numpy": np.__version__}})
            with mock.patch.object(runner.platform, "system", return_value="Linux"), mock.patch.object(
                    data, "load_settings", return_value=settings), mock.patch.object(data, "verify_public"), mock.patch.object(
                    core, "torch_module", return_value=fake_torch), mock.patch.object(
                    runner, "version", side_effect=environment["packages"].__getitem__), mock.patch.object(
                    data, "code_records", side_effect=[started, changed]), mock.patch.object(
                    shared, "verify", return_value=root / "UNUSED_PRIVATE_PATH"), mock.patch.object(
                    shared, "aligned_labels", side_effect=RuntimeError("simulated packaging failure")) as labels:
                with self.assertRaisesRegex(RuntimeError, "simulated packaging failure"):
                    runner.run(root / "run", check)
                labels.assert_called_once()
            failure = shared.read_json(root / "run/failure.json")
            self.assertEqual(failure["code"], started)
            self.assertEqual(failure["observed_code_at_failure"], changed)
            self.assertFalse((root / "run/manifest.json").exists())

    def test_summary_preserves_fixed_world_pairing_and_comparison_directions(self):
        columns = ["average_precision", "no_strong_average_precision"]
        orders = [{"seed": s, "trajectory": runner.trajectory()} for s in (11, 23, 37)]
        arrays = {}
        for i, order in enumerate(orders):
            for name in {v for path in order["trajectory"].values() for v in path}:
                stage = 0 if name == "initial" else 1 if name == "shared" else int(name[-1])
                bonus = .02 if name.startswith("er_") else .04 if name.startswith("cumulative_") else 0
                arrays[f"order{order['seed']}_{name}"] = np.array([[.1, .05], [.3, .2]]) + stage * .01 + i * .001 + bonus
        comparisons, changes, bootstrap = evaluate.summarize({"orders": orders}, arrays, columns, bootstrap_seed=7, replicates=40)
        for key, expected in (("er_minus_sequential", .02), ("cumulative_minus_sequential", .04), ("sequential_minus_frozen", .04)):
            self.assertAlmostEqual(comparisons[key]["average_precision"]["mean"], expected)
            np.testing.assert_allclose(comparisons[key]["average_precision"]["conditional_95pct_interval"], expected)
        self.assertAlmostEqual(changes["frozen"]["stage5_minus_stage1"]["average_precision"]["mean"], 0)
        self.assertEqual(sum(len(v) for v in changes.values()), 28)
        self.assertEqual(bootstrap["replicates"], 40)

    @unittest.skipIf(torch is None, "PyTorch absent locally; new real CPU orchestration check remains required")
    def test_real_cpu_production_scheduler_and_inference_reload(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = runner.runtime_check(Path(temporary) / "check.json")
            self.assertEqual(result["verification"]["physical_updates"], 46)
            self.assertEqual(result["verification"]["retained_inference_models"], 4)
            self.assertEqual(result["project_label_reads"], 0)


if __name__ == "__main__":
    unittest.main()
