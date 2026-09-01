from __future__ import annotations

import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_v9_4_1_train_development_v2 as runner
import step28_v13_v1_13_v9_4_1_model_training_core_v3 as core


class TrainDevelopmentV2Contracts(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path, dict[str, object]]:
        sellers = ["seller_a", "seller_b", "seller_c", "seller_d"]
        pairs = [
            (sellers[left], sellers[right])
            for left in range(len(sellers))
            for right in range(left + 1, len(sellers))
        ]
        row_key_path = root / "row_keys.csv"
        with row_key_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(runner.ROW_KEY_FIELDS)
            for left, right in pairs:
                writer.writerow(
                    ["development", 0, "world_0", f"{left}||{right}", left, right]
                )
        rows = runner._read_row_keys(
            row_key_path,
            "development",
            expected_rows=6,
            expected_worlds=1,
            expected_rows_per_world=6,
        )
        label_path = root / "pair_labels.csv"
        with label_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(runner.LABEL_FIELDS)
            for left, right in pairs:
                writer.writerow(
                    [f"{left}||{right}", "world_0", int({left, right} in ({"seller_a", "seller_b"}, {"seller_c", "seller_d"}))]
                )
        qrel_path = root / "qrels.jsonl"
        relevant = {
            "seller_a": ["seller_b"],
            "seller_b": ["seller_a"],
            "seller_c": ["seller_d"],
            "seller_d": ["seller_c"],
        }
        with qrel_path.open("w", encoding="utf-8") as handle:
            for seller in sellers:
                handle.write(
                    json.dumps(
                        {
                            "world_uid": "world_0",
                            "query_seller_uid": seller,
                            "relevant_seller_uids": relevant[seller],
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        return row_key_path, label_path, qrel_path, rows

    def test_live_policy_and_public_contract_validate_without_truth_read(self) -> None:
        result = runner.validate_contract()
        self.assertEqual(
            result["status"],
            "PASSED_TRAIN_DEVELOPMENT_CONTRACT_VALIDATION_NO_TRUTH_READ",
        )
        self.assertFalse(result["supervision_or_audit_truth_read"])
        self.assertFalse(result["model_training_performed"])

    def test_execution_policy_authorizes_only_train_and_development_truth(self) -> None:
        policy = runner.load_execution_policy()
        self.assertEqual(
            set(policy["authorized_private_inputs"]),
            {"train_labels", "development_labels", "development_qrels"},
        )
        self.assertEqual(policy["truth_read_budget"]["train_qrels"], 0)
        self.assertEqual(policy["truth_read_budget"]["audit_a_labels_or_qrels"], 0)
        self.assertEqual(policy["truth_read_budget"]["audit_b_labels_or_qrels"], 0)

    def test_labels_and_qrels_align_to_public_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _, labels_path, qrels_path, rows = self._fixture(Path(temp))
            labels = runner._read_labels(
                labels_path,
                rows,
                expected_rows_per_world=6,
                expected_positive_per_world=2,
            )
            relevance = runner._read_qrels_relevance(qrels_path, rows)
            np.testing.assert_array_equal(labels, relevance)

    def test_label_row_reordering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _, labels_path, _, rows = self._fixture(Path(temp))
            lines = labels_path.read_text(encoding="utf-8").splitlines()
            lines[1], lines[2] = lines[2], lines[1]
            labels_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(runner.TrainDevelopmentError, "alignment drift"):
                runner._read_labels(
                    labels_path,
                    rows,
                    expected_rows_per_world=6,
                    expected_positive_per_world=2,
                )

    def test_asymmetric_qrels_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            _, _, qrels_path, rows = self._fixture(Path(temp))
            records = [json.loads(line) for line in qrels_path.read_text(encoding="utf-8").splitlines()]
            records[1]["relevant_seller_uids"] = ["seller_c"]
            qrels_path.write_text(
                "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(runner.TrainDevelopmentError, "not symmetric"):
                runner._read_qrels_relevance(qrels_path, rows)

    def test_fast_threshold_search_matches_the_original_exhaustive_rule(self) -> None:
        generator = np.random.default_rng(20260901)
        for _ in range(100):
            world_sizes = generator.integers(2, 8, size=generator.integers(1, 6))
            worlds = np.repeat(np.arange(len(world_sizes)), world_sizes)
            labels = generator.integers(0, 2, size=len(worlds), dtype=np.int8)
            scores = generator.choice(np.asarray([0.1, 0.2, 0.5, 0.8]), size=len(worlds))
            expected = -np.inf
            best_f1 = -np.inf
            for threshold in np.r_[-np.inf, np.unique(scores), np.inf]:
                confusion = core.world_equal_confusion(labels, scores, worlds, threshold)
                f1 = core._threshold_metrics_from_confusion(confusion)["f1"]
                if f1 > best_f1 or (f1 == best_f1 and threshold > expected):
                    best_f1 = f1
                    expected = float(threshold)
            self.assertEqual(
                core.select_development_threshold(labels, scores, worlds), expected
            )

    def test_m3_models_are_saved_as_reloadable_lf_only_bytes(self) -> None:
        import lightgbm as lgb

        matrix = np.asarray(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype="<f8"
        )
        labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        model = lgb.LGBMClassifier(
            n_estimators=3,
            num_leaves=2,
            min_child_samples=1,
            deterministic=True,
            force_col_wise=True,
            num_threads=1,
            verbose=-1,
        ).fit(matrix, labels)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            artifacts = {}
            predictions = {}
            matrices = {}
            for model_id in ("m3_base", "m3_joint"):
                model_root = root / "models" / model_id
                runner._write_m3_model(model_root / "model.txt", model)
                runner._save_array(model_root / "medians.npy", np.zeros(2, dtype="<f8"))
                artifacts[model_id] = {"medians": np.zeros(2, dtype="<f8")}
                matrices[model_id] = matrix
                predictions[model_id] = np.asarray(
                    model.predict_proba(matrix)[:, 1], dtype="<f8"
                )
                payload = (model_root / "model.txt").read_bytes()
                self.assertNotIn(b"\r\n", payload)
                self.assertEqual(lgb.Booster(model_file=str(model_root / "model.txt")).num_trees(), 3)
            replay = runner._validate_reloaded_m3_models(
                root, artifacts, matrices, predictions
            )
            self.assertTrue(replay["m3_base"]["probability_byte_match"])
            self.assertTrue(replay["m3_joint"]["probability_byte_match"])


if __name__ == "__main__":
    unittest.main()
