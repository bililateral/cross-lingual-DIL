from __future__ import annotations

import copy
import hashlib
import itertools
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import sklearn
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_build_metadata_shortcut_lock as lock_builder  # noqa: E402
import step28_v13_common as dataset_common  # noqa: E402
import step28_v13_metadata_shortcut_common as shortcut_common  # noqa: E402
import step28_v13_project_null_nuisance as projector  # noqa: E402
import step28_v13_run_metadata_shortcut_audit as audit_runner  # noqa: E402
import step28_v13_seal_classification_labels as label_sealer  # noqa: E402
import step28_v13_validate_label_formula as formula_validator  # noqa: E402


class Step28V13MetadataShortcutContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.lock_path = (
            ROOT
            / "schema"
            / "step28_v13_metadata_shortcut_audit_lock.json"
        )
        try:
            cls.lock = shortcut_common.load_lock(cls.lock_path)
        except shortcut_common.ShortcutAuditError as exc:
            if str(exc) not in {
                "parent_contract byte pin drift",
                "parent_policy byte pin drift",
            }:
                raise
            raise unittest.SkipTest(
                "The immutable, execution-blocked metadata-shortcut lock "
                "belongs to the superseded dataset_smoke_v3 parent contract "
                "and policy. "
                "Current training_ready shortcut execution is covered by "
                "test_step28_v13_training_ready_builder_contracts.py."
            ) from exc
        (
            cls.candidates,
            cls.memberships,
            cls.redacted_items,
            cls.history_items,
        ) = cls._build_fixture(world_count=5)
        cls.projections = projector.build_projection(
            candidate_rows=cls.candidates,
            redacted_items=cls.redacted_items,
            history_item_rows=cls.history_items,
            expected_world_count=5,
        )
        cls.labels = label_sealer.build_labels(
            candidate_rows=cls.candidates,
            membership_rows=cls.memberships,
            expected_world_count=5,
        )

    @staticmethod
    def _build_fixture(
        *,
        world_count: int,
    ) -> tuple[
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
        list[dict[str, str]],
    ]:
        candidates: list[dict[str, str]] = []
        memberships: list[dict[str, str]] = []
        redacted_items: list[dict[str, str]] = []
        history_items: list[dict[str, str]] = []
        for world_index in range(world_count):
            world_uid = f"world_{world_index:03d}"
            sellers = [
                f"seller_{world_index:03d}_{seller_index:02d}"
                for seller_index in range(28)
            ]
            for seller_index, seller_uid in enumerate(sellers):
                if seller_index < 16:
                    controller_ordinal = seller_index // 2
                else:
                    controller_ordinal = (
                        8 + (seller_index - 16) // 3
                    )
                memberships.append(
                    {
                        "world_uid": world_uid,
                        "controller_uid": (
                            f"controller_{world_index:03d}_"
                            f"{controller_ordinal:02d}"
                        ),
                        "seller_uid": seller_uid,
                    }
                )
                item_count = 2 + seller_index % 3
                for item_index in range(item_count):
                    item_uid = (
                        f"item_{world_index:03d}_"
                        f"{seller_index:02d}_{item_index:02d}"
                    )
                    history_items.append(
                        {
                            "world_uid": world_uid,
                            "seller_uid": seller_uid,
                            "item_uid": item_uid,
                            "time_bucket": str(
                                (seller_index + item_index) % 4
                            ),
                        }
                    )
                    redacted_items.append(
                        {
                            "description": (
                                ""
                                if (seller_index + item_index) % 5 == 0
                                else f"描述 {seller_index} {item_index}"
                            ),
                            "item_uid": item_uid,
                            "seller_uid": seller_uid,
                            "title": (
                                ""
                                if (seller_index + item_index) % 7 == 0
                                else f"标题 {seller_index} {item_index}"
                            ),
                            "world_uid": world_uid,
                        }
                    )
            for left, right in list(
                itertools.combinations(sellers, 2)
            )[:40]:
                candidates.append(
                    {
                        "canonical_pair_uid": (
                            dataset_common.canonical_pair_uid(
                                left,
                                right,
                            )
                        ),
                        "world_uid": world_uid,
                        "seller_uid_left": left,
                        "seller_uid_right": right,
                    }
                )
        return (
            candidates,
            memberships,
            redacted_items,
            history_items,
        )

    def _write_test_lock(
        self,
        root: Path,
        *,
        train_world_count: int = 5,
    ) -> tuple[dict[str, object], Path]:
        body = copy.deepcopy(
            shortcut_common.canonical_without_self(self.lock)
        )
        body["formal_world_counts"]["train"] = train_world_count
        lock = shortcut_common.add_self_hash(body)
        path = root / "test_metadata_shortcut_lock.json"
        dataset_common.write_json(path, lock)
        return lock, path

    def test_lock_payload_and_source_closure_are_exact(self) -> None:
        self.assertEqual(
            dataset_common.load_json(self.lock_path),
            lock_builder.build_payload(),
        )
        self.assertFalse(self.lock["formal_execution"]["enabled"])
        self.assertTrue(
            self.lock["engineering_smoke_release"][
                "numeric_shortcut_execution_forbidden"
            ]
        )
        self.assertFalse(
            self.lock["claim_boundary"]["pass_dataset_only_granted"]
        )
        self.assertEqual(
            self.lock["bootstrap_draw_hash_dtype"],
            ">u8",
        )
        smoke = dataset_common.load_json(
            dataset_common.repo_path(
                self.lock["engineering_smoke_release"]["path"]
            )
        )
        self.assertEqual(
            self.lock["engineering_smoke_release"][
                "content_sha256"
            ],
            smoke["canonical_self_hash"],
        )
        identity = shortcut_common.manifest_identity(
            self.lock,
            lock_path=self.lock_path,
            stage="test",
            producer_relative_path=(
                "tests/"
                "test_step28_v13_metadata_shortcut_audit_contracts.py"
            ),
        )
        self.assertEqual(
            identity["lock_content_sha256"],
            self.lock["canonical_self_hash"],
        )
        sorted_identity = shortcut_common.manifest_identity(
            self.lock,
            lock_path=self.lock_path,
            stage="test",
            producer_relative_path=(
                "tests/"
                "test_step28_v13_metadata_shortcut_audit_contracts.py"
            ),
            additional_parent_manifests=[
                {
                    "role": "z_parent",
                    "file_sha256": "1" * 64,
                    "content_sha256": "2" * 64,
                },
                {
                    "role": "a_parent",
                    "file_sha256": "3" * 64,
                    "content_sha256": "4" * 64,
                },
            ],
        )
        parent_roles = [
            row["role"]
            for row in sorted_identity["parent_manifests"]
        ]
        self.assertEqual(
            parent_roles,
            sorted(parent_roles, key=lambda value: value.encode("utf-8")),
        )
        drifted_lock = copy.deepcopy(self.lock)
        drifted_lock["objective"] = "in-memory-only drift"
        with self.assertRaises(shortcut_common.ShortcutAuditError):
            shortcut_common.manifest_identity(
                drifted_lock,
                lock_path=self.lock_path,
                stage="test",
                producer_relative_path=(
                    "tests/"
                    "test_step28_v13_metadata_shortcut_audit_contracts.py"
                ),
            )
        self.assertIn(
            "tests/test_step28_v13_metadata_shortcut_audit_contracts.py",
            self.lock["source_files_sha256"],
        )
        for split in shortcut_common.SPLITS:
            self.assertFalse(
                self.lock["formal_execution"][
                    "split_authorizations"
                ][split]["authorized"]
            )

    def test_config_preflights_pass_but_execution_fails_before_input(self) -> None:
        scripts = (
            "step28_v13_project_null_nuisance.py",
            "step28_v13_seal_classification_labels.py",
            "step28_v13_validate_label_formula.py",
            "step28_v13_run_metadata_shortcut_audit.py",
        )
        for name in scripts:
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPTS / name),
                    "--split",
                    "train",
                    "--validate-config-only",
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("formal execution remains blocked", result.stdout)
        blocked = subprocess.run(
            [
                sys.executable,
                str(
                    SCRIPTS
                    / "step28_v13_project_null_nuisance.py"
                ),
                "--split",
                "train",
                "--candidate-pairs",
                "definitely_missing/candidate_pairs.csv",
                "--redacted-items",
                "definitely_missing/redacted_items.jsonl",
                "--history-item-index",
                "definitely_missing/history_item_index.csv",
                "--output-dir",
                "definitely_missing/output",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(blocked.returncode, 0)
        self.assertIn(
            "Formal metadata-shortcut execution is blocked",
            blocked.stderr,
        )
        self.assertNotIn("FileNotFoundError", blocked.stderr)
        filled = copy.deepcopy(self.lock)
        execution = filled["formal_execution"]
        execution["enabled"] = True
        execution["missing_prerequisites"] = []
        execution["formal_release_content_sha256"] = "1" * 64
        execution["custody_access_manifest_content_sha256"] = "2" * 64
        execution["execution_environment_content_sha256"] = "3" * 64
        execution["exact_input_bindings"] = {"train": {}}
        filled["version"] = "forged-formal-version"
        filled["status"] = "FORGED_FORMAL_ENABLED"
        with self.assertRaises(shortcut_common.ShortcutAuditError):
            shortcut_common.require_formal_execution_envelope(filled)

    def test_public_release_writers_cannot_bypass_blocked_core(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing"
            forged = copy.deepcopy(self.lock)
            forged["version"] = "forged-formal-version"
            forged["status"] = "FORGED_FORMAL_ENABLED"
            forged_execution = forged["formal_execution"]
            forged_execution["enabled"] = True
            forged_execution["missing_prerequisites"] = []
            forged_execution["formal_release_content_sha256"] = (
                "1" * 64
            )
            forged_execution[
                "custody_access_manifest_content_sha256"
            ] = "2" * 64
            forged_execution[
                "execution_environment_content_sha256"
            ] = "3" * 64
            forged_execution["exact_input_bindings"] = {"train": {}}
            calls = (
                lambda: projector.write_projection_release(
                    lock=forged,
                    lock_path=self.lock_path,
                    split="train",
                    candidate_path=missing / "candidate_pairs.csv",
                    redacted_path=missing / "redacted_items.jsonl",
                    history_item_path=(
                        missing / "history_item_index.csv"
                    ),
                    output_dir=root / "forged_projection",
                ),
                lambda: projector.write_projection_release(
                    lock=self.lock,
                    lock_path=self.lock_path,
                    split="train",
                    candidate_path=missing / "candidate_pairs.csv",
                    redacted_path=missing / "redacted_items.jsonl",
                    history_item_path=(
                        missing / "history_item_index.csv"
                    ),
                    output_dir=root / "projection",
                ),
                lambda: label_sealer.write_label_release(
                    lock=self.lock,
                    lock_path=self.lock_path,
                    split="train",
                    candidate_path=missing / "candidate_pairs.csv",
                    membership_path=(
                        missing / "controller_membership.csv"
                    ),
                    output_dir=root / "labels",
                ),
                lambda: formula_validator.write_validation_release(
                    lock=self.lock,
                    lock_path=self.lock_path,
                    split="train",
                    candidate_path=missing / "candidate_pairs.csv",
                    membership_path=(
                        missing / "controller_membership.csv"
                    ),
                    labels_path=(
                        missing / shortcut_common.LABEL_FILENAME
                    ),
                    label_manifest_path=(
                        missing
                        / shortcut_common.LABEL_MANIFEST_FILENAME
                    ),
                    output_dir=root / "formula",
                ),
                lambda: audit_runner.write_audit_release(
                    lock=self.lock,
                    lock_path=self.lock_path,
                    split="train",
                    projection_path=(
                        missing / shortcut_common.PROJECTION_FILENAME
                    ),
                    labels_path=(
                        missing / shortcut_common.LABEL_FILENAME
                    ),
                    label_formula_receipt_path=(
                        missing
                        / shortcut_common.LABEL_VALIDATION_FILENAME
                    ),
                    output_dir=root / "audit",
                ),
            )
            for call in calls:
                with self.subTest(call=call):
                    with self.assertRaises(
                        shortcut_common.ShortcutAuditError
                    ):
                        call()

    def test_projection_is_exact_order_independent_and_label_free(self) -> None:
        replay = projector.build_projection(
            candidate_rows=list(reversed(self.candidates)),
            redacted_items=list(reversed(self.redacted_items)),
            history_item_rows=list(reversed(self.history_items)),
            expected_world_count=5,
        )
        self.assertEqual(replay, self.projections)
        self.assertEqual(len(self.projections), 200)
        self.assertEqual(
            tuple(self.projections[0]),
            shortcut_common.PROJECTION_FIELDS,
        )
        self.assertNotIn("label", self.projections[0])
        self.assertNotIn("controller_uid", self.projections[0])
        first_pair = self.projections[0]
        candidate = next(
            row
            for row in self.candidates
            if row["canonical_pair_uid"]
            == first_pair["canonical_pair_uid"]
        )

        def independent_seller_vector(
            seller_uid: str,
        ) -> tuple[float, ...]:
            selected = [
                row
                for row in self.redacted_items
                if row["world_uid"] == candidate["world_uid"]
                and row["seller_uid"] == seller_uid
            ]
            bucket_by_item = {
                row["item_uid"]: int(row["time_bucket"])
                for row in self.history_items
                if row["world_uid"] == candidate["world_uid"]
                and row["seller_uid"] == seller_uid
            }
            count = len(selected)
            bucket_counts = [
                sum(
                    bucket_by_item[row["item_uid"]] == bucket
                    for row in selected
                )
                for bucket in range(4)
            ]
            return (
                float(count),
                sum(row["title"] == "" for row in selected) / count,
                sum(
                    row["description"] == "" for row in selected
                )
                / count,
                *(value / count for value in bucket_counts),
            )

        left = independent_seller_vector(
            candidate["seller_uid_left"]
        )
        right = independent_seller_vector(
            candidate["seller_uid_right"]
        )
        independent_values = tuple(
            abs(left[index] - right[index]) for index in range(7)
        ) + tuple(left[index] + right[index] for index in range(7))
        self.assertEqual(
            [
                first_pair[name]
                for name in shortcut_common.PAIR_FEATURES
            ],
            [format(value, ".12f") for value in independent_values],
        )
        for row in self.projections:
            for name in shortcut_common.PAIR_FEATURES:
                self.assertEqual(
                    row[name],
                    format(float(row[name]), ".12f"),
                )

    def test_projection_rejects_unknown_field_and_illegal_bucket(self) -> None:
        redacted = copy.deepcopy(self.redacted_items)
        redacted[0]["label"] = "1"
        with self.assertRaises(shortcut_common.ShortcutAuditError):
            projector.build_projection(
                candidate_rows=self.candidates,
                redacted_items=redacted,
                history_item_rows=self.history_items,
                expected_world_count=5,
            )
        extra_history = copy.deepcopy(self.history_items)
        extra_redacted = copy.deepcopy(self.redacted_items)
        extra_history.extend(
            {
                "world_uid": "extra_world",
                "seller_uid": f"extra_seller_{seller_index:02d}",
                "item_uid": (
                    f"extra_item_{seller_index:02d}_{item_index}"
                ),
                "time_bucket": str(item_index),
            }
            for seller_index in range(28)
            for item_index in range(2)
        )
        extra_redacted.extend(
            {
                "description": "描述",
                "item_uid": (
                    f"extra_item_{seller_index:02d}_{item_index}"
                ),
                "seller_uid": f"extra_seller_{seller_index:02d}",
                "title": "标题",
                "world_uid": "extra_world",
            }
            for seller_index in range(28)
            for item_index in range(2)
        )
        with self.assertRaises(shortcut_common.ShortcutAuditError):
            projector.build_projection(
                candidate_rows=self.candidates,
                redacted_items=extra_redacted,
                history_item_rows=extra_history,
                expected_world_count=5,
            )
        history = copy.deepcopy(self.history_items)
        history[0]["time_bucket"] = "04"
        with self.assertRaises(shortcut_common.ShortcutAuditError):
            projector.build_projection(
                candidate_rows=self.candidates,
                redacted_items=self.redacted_items,
                history_item_rows=history,
                expected_world_count=5,
            )

    def test_label_formula_has_independent_recalculation(self) -> None:
        result = formula_validator.validate_formula(
            candidate_rows=self.candidates,
            membership_rows=self.memberships,
            label_rows=self.labels,
            expected_world_count=5,
        )
        self.assertTrue(result["validated"])
        self.assertTrue(result["class_counts_withheld"])
        damaged = copy.deepcopy(self.labels)
        damaged[0]["label"] = (
            "0" if damaged[0]["label"] == "1" else "1"
        )
        with self.assertRaises(shortcut_common.ShortcutAuditError):
            formula_validator.validate_formula(
                candidate_rows=self.candidates,
                membership_rows=self.memberships,
                label_rows=damaged,
                expected_world_count=5,
            )

    def test_world_fold_assignment_is_deterministic_and_grouped(self) -> None:
        worlds = [row["world_uid"] for row in self.projections]
        first = audit_runner.assign_world_folds(
            worlds,
            seed=2026072707,
            fold_count=5,
        )
        second = audit_runner.assign_world_folds(
            list(reversed(worlds)),
            seed=2026072707,
            fold_count=5,
        )
        self.assertEqual(first, second)
        self.assertEqual(set(first.values()), set(range(5)))
        for world_uid in set(worlds):
            self.assertEqual(
                {
                    first[value]
                    for value in worlds
                    if value == world_uid
                },
                {first[world_uid]},
            )
        with self.assertRaises(shortcut_common.ShortcutAuditError):
            audit_runner.assign_world_folds(
                ["world_0", "world_1", "world_2"],
                seed=2026072707,
                fold_count=5,
            )

    def test_symmetric_auc_treats_inverse_prediction_as_signal(self) -> None:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        auc, symmetric = audit_runner.symmetric_auc(labels, scores)
        inverse_auc, inverse_symmetric = audit_runner.symmetric_auc(
            labels,
            -scores,
        )
        self.assertEqual(auc, 1.0)
        self.assertEqual(inverse_auc, 0.0)
        self.assertEqual(symmetric, 1.0)
        self.assertEqual(inverse_symmetric, 1.0)

    def test_world_bootstrap_matches_independent_expansion_oracle(self) -> None:
        worlds = [
            f"w{world_index}"
            for world_index in range(6)
            for _row in range(4)
        ]
        labels = np.asarray(
            [value for _world in range(6) for value in (0, 0, 1, 1)],
            dtype=np.int64,
        )
        scores = {
            "logistic_l2": np.asarray(
                [
                    -1.0,
                    -0.4,
                    1.3,
                    0.2,
                    0.9,
                    0.6,
                    -0.6,
                    0.5,
                    -0.3,
                    -0.3,
                    0.1,
                    -1.5,
                    1.2,
                    -0.7,
                    1.0,
                    0.1,
                    1.5,
                    -0.7,
                    -0.3,
                    0.3,
                    -2.2,
                    0.8,
                    1.5,
                    1.1,
                ]
            ),
            "gradient_tree": np.asarray(
                [
                    0.8,
                    -0.1,
                    1.3,
                    1.1,
                    0.4,
                    0.0,
                    -0.4,
                    -1.2,
                    1.2,
                    -2.2,
                    -0.4,
                    0.2,
                    0.9,
                    1.8,
                    1.0,
                    -0.3,
                    0.7,
                    -1.3,
                    1.4,
                    -0.2,
                    -0.7,
                    -0.6,
                    -0.1,
                    -0.4,
                ]
            ),
            "rbf_svm": np.asarray(
                [
                    2.3,
                    -0.7,
                    0.0,
                    0.0,
                    0.0,
                    0.1,
                    -0.5,
                    -0.6,
                    -0.9,
                    -1.5,
                    0.2,
                    1.0,
                    -0.5,
                    -0.6,
                    -0.3,
                    -0.5,
                    -1.4,
                    1.4,
                    0.4,
                    -0.7,
                    0.3,
                    -0.4,
                    -0.2,
                    0.4,
                ]
            ),
        }
        replicates = 99
        observed_upper, observed_hash = (
            audit_runner.world_bootstrap_upper(
                y=labels,
                world_uids=worlds,
                score_by_model=scores,
                split="train",
                replicates=replicates,
                base_seed=2026072711,
            )
        )
        ordered_worlds = sorted(
            set(worlds),
            key=lambda value: value.encode("utf-8"),
        )
        digest = hashlib.sha256(
            b"2026072711\x1ftrain"
        ).digest()
        seed = int.from_bytes(digest[:16], "big", signed=False)
        generator = np.random.Generator(np.random.PCG64DXSM(seed))
        draws = generator.integers(
            0,
            len(ordered_worlds),
            size=(replicates, len(ordered_worlds)),
            dtype=np.int64,
        )
        expected_hash = hashlib.sha256(
            np.ascontiguousarray(
                draws.astype(">u8", copy=False)
            ).tobytes(order="C")
        ).hexdigest()
        statistics = []
        for draw in draws:
            expanded_indices = []
            for world_ordinal in draw.tolist():
                selected_world = ordered_worlds[world_ordinal]
                expanded_indices.extend(
                    index
                    for index, value in enumerate(worlds)
                    if value == selected_world
                )
            expanded = np.asarray(expanded_indices, dtype=np.int64)
            model_values = []
            for model_key in (
                "logistic_l2",
                "gradient_tree",
                "rbf_svm",
            ):
                auc = roc_auc_score(
                    labels[expanded],
                    scores[model_key][expanded],
                )
                model_values.append(max(auc, 1.0 - auc))
            statistics.append(max(model_values))
        self.assertGreater(len(set(statistics)), 2)
        expected_upper = float(
            np.quantile(
                np.asarray(statistics),
                0.95,
                method="higher",
            )
        )
        linear_upper = float(
            np.quantile(
                np.asarray(statistics),
                0.95,
                method="linear",
            )
        )
        self.assertEqual(observed_hash, expected_hash)
        self.assertEqual(observed_upper, expected_upper)
        self.assertEqual(
            observed_hash,
            "a842a67d73d39acb1fdb8aa70a204fca63aaade959f0863d3d2a10b690d6fcfd",
        )
        self.assertGreater(observed_upper, 0.5)
        self.assertLess(observed_upper, 1.0)
        self.assertNotEqual(observed_upper, linear_upper)
        _development_upper, development_hash = (
            audit_runner.world_bootstrap_upper(
                y=labels,
                world_uids=worlds,
                score_by_model=scores,
                split="development",
                replicates=replicates,
                base_seed=2026072711,
            )
        )
        self.assertNotEqual(observed_hash, development_hash)

    def test_projection_semantic_range_tamper_is_rejected(self) -> None:
        damaged = copy.deepcopy(self.projections)
        damaged[0]["absdiff__item_count"] = "-1.000000000000"
        with self.assertRaises(shortcut_common.ShortcutAuditError):
            audit_runner.run_audit(
                projection_rows=damaged,
                label_rows=self.labels,
                split="train",
                expected_world_count=5,
                bootstrap_replicates=3,
            )

    def test_convergence_and_nonfinite_scores_fail_closed(self) -> None:
        x_train = np.asarray(
            [[0.0] * 14, [1.0] * 14],
            dtype=np.float64,
        )
        y_train = np.asarray([0, 1], dtype=np.int64)
        x_test = np.asarray([[0.5] * 14], dtype=np.float64)
        with mock.patch.object(
            audit_runner.LogisticRegression,
            "fit",
            autospec=True,
            side_effect=ConvergenceWarning("no convergence"),
        ):
            with self.assertRaises(
                shortcut_common.ShortcutAuditError
            ):
                audit_runner._fit_and_score_fold(
                    "logistic_l2",
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                )
        with mock.patch.object(
            audit_runner.SVC,
            "decision_function",
            autospec=True,
            return_value=np.asarray([np.nan]),
        ):
            with self.assertRaises(
                shortcut_common.ShortcutAuditError
            ):
                audit_runner._fit_and_score_fold(
                    "rbf_svm",
                    x_train=x_train,
                    y_train=y_train,
                    x_test=x_test,
                )

    def test_snapshot_record_uses_exact_bytes_read_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "input.csv"
            path.write_bytes(b"a,b\n1,2\n")
            snapshot = shortcut_common.snapshot_regular_file(path)
            path.write_bytes(b"a,b\n9,9\n")
            self.assertEqual(snapshot.payload, b"a,b\n1,2\n")
            self.assertEqual(
                snapshot.sha256,
                hashlib.sha256(b"a,b\n1,2\n").hexdigest(),
            )
            self.assertNotEqual(
                snapshot.sha256,
                hashlib.sha256(path.read_bytes()).hexdigest(),
            )

    def test_access_isolation_is_not_hardcoded_as_zero(self) -> None:
        self.assertEqual(
            shortcut_common.BLOCKED_ACCESS_STATUS,
            "UNVERIFIED_FORMAL_EXECUTION_BLOCKED",
        )
        for name in (
            "step28_v13_project_null_nuisance.py",
            "step28_v13_seal_classification_labels.py",
            "step28_v13_validate_label_formula.py",
        ):
            source = (SCRIPTS / name).read_text(encoding="utf-8")
            self.assertNotIn(
                '"forbidden_successful_open_count": 0',
                source,
            )
            self.assertIn(
                "BLOCKED_ACCESS_STATUS",
                source,
            )
        runner_source = (
            SCRIPTS / "step28_v13_run_metadata_shortcut_audit.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            '"formal_scientific_claim_authorized": False',
            runner_source,
        )
        self.assertIn("BLOCKED_ACCESS_STATUS", runner_source)

    def test_label_manifest_parent_is_fully_validated(self) -> None:
        self.assertIn(
            "validate_label_manifest_release",
            (
                SCRIPTS / "step28_v13_validate_label_formula.py"
            ).read_text(encoding="utf-8"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            label_dir = root / "labels"
            label_dir.mkdir()
            lock, test_lock_path = self._write_test_lock(root)
            labels_path = (
                label_dir / shortcut_common.LABEL_FILENAME
            )
            dataset_common.write_csv(
                labels_path,
                self.labels,
                shortcut_common.LABEL_FIELDS,
            )
            label_snapshot = shortcut_common.snapshot_regular_file(
                labels_path
            )
            manifest = shortcut_common.add_self_hash(
                {
                    "version": (
                        shortcut_common.LABEL_MANIFEST_VERSION
                    ),
                    "status": (
                        "SEALED_PRIVATE_CLASSIFICATION_LABELS"
                    ),
                    "mode": "formal",
                    "split": "train",
                    "row_count": 200,
                    "world_count": 5,
                    "rows_per_world": 40,
                    "label_schema": list(
                        shortcut_common.LABEL_FIELDS
                    ),
                    "formula": (
                        "int(controller(left)==controller(right))"
                    ),
                    "formula_equality_required": True,
                    "class_counts_withheld": True,
                    "label_content_sha256": (
                        dataset_common.canonical_sha256(self.labels)
                    ),
                    **shortcut_common.manifest_identity(
                        lock,
                        lock_path=test_lock_path,
                        stage="seal_classification_labels",
                        producer_relative_path=(
                            "scripts/"
                            "step28_v13_seal_classification_labels.py"
                        ),
                    ),
                    "input_allowlist": [
                        {
                            "role": "candidate_pairs",
                            "basename": "candidate_pairs.csv",
                            "size_bytes": 100,
                            "sha256": "1" * 64,
                        },
                        {
                            "role": "controller_membership",
                            "basename": "controller_membership.csv",
                            "size_bytes": 200,
                            "sha256": "2" * 64,
                        },
                    ],
                    "access_isolation_status": (
                        shortcut_common.BLOCKED_ACCESS_STATUS
                    ),
                    "forbidden_open_count_not_self_asserted": True,
                    "files": [
                        shortcut_common.file_record(
                            labels_path,
                            role="private_classification_labels",
                            root=label_dir,
                        )
                    ],
                }
            )
            manifest_path = (
                label_dir
                / shortcut_common.LABEL_MANIFEST_FILENAME
            )
            dataset_common.write_json(manifest_path, manifest)
            manifest_snapshot = (
                shortcut_common.snapshot_regular_file(manifest_path)
            )
            observed_inputs = (
                shortcut_common.validate_label_manifest_release(
                    lock=lock,
                    lock_path=test_lock_path,
                    split="train",
                    labels_path=labels_path,
                    label_rows=self.labels,
                    label_snapshot=label_snapshot,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    manifest_snapshot=manifest_snapshot,
                )
            )
            self.assertEqual(
                set(observed_inputs),
                {"candidate_pairs", "controller_membership"},
            )
            damaged = copy.deepcopy(
                shortcut_common.canonical_without_self(manifest)
            )
            damaged["version"] = "obsolete-label-manifest"
            damaged_manifest = shortcut_common.add_self_hash(damaged)
            damaged_dir = root / "labels_version_drift"
            damaged_dir.mkdir()
            damaged_labels_path = (
                damaged_dir / shortcut_common.LABEL_FILENAME
            )
            dataset_common.write_csv(
                damaged_labels_path,
                self.labels,
                shortcut_common.LABEL_FIELDS,
            )
            damaged_label_snapshot = (
                shortcut_common.snapshot_regular_file(
                    damaged_labels_path
                )
            )
            damaged_manifest_path = (
                damaged_dir
                / shortcut_common.LABEL_MANIFEST_FILENAME
            )
            dataset_common.write_json(
                damaged_manifest_path,
                damaged_manifest,
            )
            damaged_snapshot = (
                shortcut_common.snapshot_regular_file(
                    damaged_manifest_path
                )
            )
            with self.assertRaises(
                shortcut_common.ShortcutAuditError
            ):
                shortcut_common.validate_label_manifest_release(
                    lock=lock,
                    lock_path=test_lock_path,
                    split="train",
                    labels_path=damaged_labels_path,
                    label_rows=self.labels,
                    label_snapshot=damaged_label_snapshot,
                    manifest_path=damaged_manifest_path,
                    manifest=damaged_manifest,
                    manifest_snapshot=damaged_snapshot,
                )

    def test_projection_manifest_version_is_enforced(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            projection_dir = root / "projection"
            projection_dir.mkdir()
            lock, test_lock_path = self._write_test_lock(root)
            projection_path = (
                projection_dir / shortcut_common.PROJECTION_FILENAME
            )
            dataset_common.write_csv(
                projection_path,
                self.projections,
                shortcut_common.PROJECTION_FIELDS,
            )
            manifest = shortcut_common.add_self_hash(
                {
                    "version": (
                        shortcut_common.PROJECTION_MANIFEST_VERSION
                    ),
                    "status": "SEALED_LABEL_FREE_PROJECTION",
                    "mode": "formal",
                    "split": "train",
                    "row_count": 200,
                    "world_count": 5,
                    "rows_per_world": 40,
                    "projection_schema": list(
                        shortcut_common.PROJECTION_FIELDS
                    ),
                    "projection_content_sha256": (
                        dataset_common.canonical_sha256(
                            self.projections
                        )
                    ),
                    **shortcut_common.manifest_identity(
                        lock,
                        lock_path=test_lock_path,
                        stage="project_null_nuisance",
                        producer_relative_path=(
                            "scripts/"
                            "step28_v13_project_null_nuisance.py"
                        ),
                    ),
                    "input_allowlist": [
                        {
                            "role": "candidate_pairs",
                            "basename": "candidate_pairs.csv",
                            "size_bytes": 100,
                            "sha256": "1" * 64,
                        },
                        {
                            "role": "history_item_index",
                            "basename": "history_item_index.csv",
                            "size_bytes": 200,
                            "sha256": "2" * 64,
                        },
                        {
                            "role": "redacted_items",
                            "basename": "redacted_items.jsonl",
                            "size_bytes": 300,
                            "sha256": "3" * 64,
                        },
                    ],
                    "access_isolation_status": (
                        shortcut_common.BLOCKED_ACCESS_STATUS
                    ),
                    "forbidden_open_count_not_self_asserted": True,
                    "files": [
                        shortcut_common.file_record(
                            projection_path,
                            role="null_nuisance_projection",
                            root=projection_dir,
                        )
                    ],
                }
            )
            manifest_path = (
                projection_dir
                / shortcut_common.PROJECTION_MANIFEST_FILENAME
            )
            dataset_common.write_json(manifest_path, manifest)
            observed_rows, _manifest, _snapshots = (
                audit_runner._load_and_validate_projection_release(
                    lock=lock,
                    lock_path=test_lock_path,
                    split="train",
                    projection_path=projection_path,
                )
            )
            self.assertEqual(observed_rows, self.projections)
            damaged = copy.deepcopy(
                shortcut_common.canonical_without_self(manifest)
            )
            damaged["version"] = "obsolete-projection-manifest"
            damaged_dir = root / "projection_version_drift"
            damaged_dir.mkdir()
            damaged_projection_path = (
                damaged_dir / shortcut_common.PROJECTION_FILENAME
            )
            dataset_common.write_csv(
                damaged_projection_path,
                self.projections,
                shortcut_common.PROJECTION_FIELDS,
            )
            dataset_common.write_json(
                damaged_dir
                / shortcut_common.PROJECTION_MANIFEST_FILENAME,
                shortcut_common.add_self_hash(damaged),
            )
            with self.assertRaises(
                shortcut_common.ShortcutAuditError
            ):
                audit_runner._load_and_validate_projection_release(
                    lock=lock,
                    lock_path=test_lock_path,
                    split="train",
                    projection_path=damaged_projection_path,
                )

    def test_runner_requires_formula_receipt_to_bind_exact_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            label_dir = root / "labels"
            receipt_dir = root / "formula"
            label_dir.mkdir()
            receipt_dir.mkdir()
            labels_path = (
                label_dir / shortcut_common.LABEL_FILENAME
            )
            dataset_common.write_csv(
                labels_path,
                self.labels,
                shortcut_common.LABEL_FIELDS,
            )
            label_snapshot = shortcut_common.snapshot_regular_file(
                labels_path
            )
            lock, test_lock_path = self._write_test_lock(root)
            parent_records = [
                {
                    "role": "candidate_pairs",
                    "basename": "candidate_pairs.csv",
                    "size_bytes": 100,
                    "sha256": "1" * 64,
                },
                {
                    "role": "controller_membership",
                    "basename": "controller_membership.csv",
                    "size_bytes": 200,
                    "sha256": "2" * 64,
                },
            ]
            label_manifest = shortcut_common.add_self_hash(
                {
                    "input_allowlist": copy.deepcopy(
                        parent_records
                    )
                }
            )
            label_manifest_path = (
                label_dir
                / shortcut_common.LABEL_MANIFEST_FILENAME
            )
            dataset_common.write_json(
                label_manifest_path,
                label_manifest,
            )
            label_manifest_snapshot = (
                shortcut_common.snapshot_regular_file(
                    label_manifest_path
                )
            )
            receipt = shortcut_common.add_self_hash(
                {
                    "version": (
                        shortcut_common.LABEL_FORMULA_RECEIPT_VERSION
                    ),
                    "status": "PASS_LABEL_FORMULA_ONLY",
                    "mode": "formal",
                    "split": "train",
                    "validated": True,
                    "row_count": 200,
                    "world_count": 5,
                    "rows_per_world": 40,
                    "class_counts_withheld": True,
                    "alternative_derivation": (
                        shortcut_common
                        .LABEL_FORMULA_ALTERNATIVE_DERIVATION
                    ),
                    **shortcut_common.manifest_identity(
                        lock,
                        lock_path=test_lock_path,
                        stage="validate_label_formula",
                        producer_relative_path=(
                            "scripts/"
                            "step28_v13_validate_label_formula.py"
                        ),
                        additional_parent_manifests=[
                            {
                                "role": (
                                    "classification_label_manifest"
                                ),
                                "file_sha256": (
                                    label_manifest_snapshot.sha256
                                ),
                                "content_sha256": label_manifest[
                                    "canonical_self_hash"
                                ],
                            }
                        ],
                    ),
                    "input_allowlist": [
                        *copy.deepcopy(parent_records),
                        label_snapshot.record(
                            role="sealed_labels"
                        ),
                        label_manifest_snapshot.record(
                            role="label_manifest"
                        ),
                    ],
                    "access_isolation_status": (
                        shortcut_common.BLOCKED_ACCESS_STATUS
                    ),
                    "forbidden_open_count_not_self_asserted": True,
                }
            )
            receipt_path = (
                receipt_dir
                / shortcut_common.LABEL_VALIDATION_FILENAME
            )
            dataset_common.write_json(receipt_path, receipt)
            observed, _snapshot = (
                audit_runner._load_and_validate_formula_receipt(
                    lock=lock,
                    lock_path=test_lock_path,
                    split="train",
                    receipt_path=receipt_path,
                    labels_path=labels_path,
                    label_snapshot=label_snapshot,
                    label_manifest_snapshot=(
                        label_manifest_snapshot
                    ),
                    label_manifest=label_manifest,
                )
            )
            self.assertTrue(observed["validated"])
            damaged = copy.deepcopy(
                shortcut_common.canonical_without_self(receipt)
            )
            damaged["input_allowlist"][-1]["sha256"] = "3" * 64
            damaged_dir = root / "formula_damaged"
            damaged_dir.mkdir()
            damaged_path = (
                damaged_dir
                / shortcut_common.LABEL_VALIDATION_FILENAME
            )
            dataset_common.write_json(
                damaged_path,
                shortcut_common.add_self_hash(damaged),
            )
            with self.assertRaises(
                shortcut_common.ShortcutAuditError
            ):
                audit_runner._load_and_validate_formula_receipt(
                    lock=lock,
                    lock_path=test_lock_path,
                    split="train",
                    receipt_path=damaged_path,
                    labels_path=labels_path,
                    label_snapshot=label_snapshot,
                    label_manifest_snapshot=(
                        label_manifest_snapshot
                    ),
                    label_manifest=label_manifest,
                )
            version_drift = copy.deepcopy(
                shortcut_common.canonical_without_self(receipt)
            )
            version_drift["version"] = "obsolete-formula-receipt"
            version_dir = root / "formula_version_drift"
            version_dir.mkdir()
            version_path = (
                version_dir
                / shortcut_common.LABEL_VALIDATION_FILENAME
            )
            dataset_common.write_json(
                version_path,
                shortcut_common.add_self_hash(version_drift),
            )
            with self.assertRaises(
                shortcut_common.ShortcutAuditError
            ):
                audit_runner._load_and_validate_formula_receipt(
                    lock=lock,
                    lock_path=test_lock_path,
                    split="train",
                    receipt_path=version_path,
                    labels_path=labels_path,
                    label_snapshot=label_snapshot,
                    label_manifest_snapshot=(
                        label_manifest_snapshot
                    ),
                    label_manifest=label_manifest,
                )

    def test_audit_b_cannot_skip_required_prior_receipts(self) -> None:
        audit_b_requirements = shortcut_common.prior_requirements(
            "audit_b"
        )
        self.assertIn(
            {
                "role": "audit_a_overall_pass",
                "split": "audit_a",
                "operation": "overall_dataset_gate",
                "status": "PASS_A_ONLY",
            },
            audit_b_requirements,
        )
        self.assertNotIn(
            "AUDIT_A_COMPLETED",
            {row["status"] for row in audit_b_requirements},
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            lock = copy.deepcopy(self.lock)
            execution = lock["formal_execution"]
            execution["formal_release_content_sha256"] = "4" * 64
            execution["custody_access_manifest_content_sha256"] = (
                "5" * 64
            )
            authorization = execution["split_authorizations"][
                "audit_b"
            ]
            authorization["authorized"] = True
            receipt_value = shortcut_common.add_self_hash(
                {
                    "version": "test-v1",
                    "status": "AUTHORIZED_SPLIT_OPERATION",
                    "authorized": True,
                    "split": "audit_b",
                    "operation": "metadata_shortcut_audit",
                    "run_id": lock["formal_run_id"],
                    "implementation_lock_content_sha256": (
                        lock["canonical_self_hash"]
                    ),
                    "formal_release_content_sha256": "4" * 64,
                    "custody_access_manifest_content_sha256": (
                        "5" * 64
                    ),
                    "issuer": "test-custody",
                    "one_shot_nonce_sha256": "6" * 64,
                }
            )
            receipt_path = root / "authorization.json"
            dataset_common.write_json(receipt_path, receipt_value)
            authorization["authorization_receipts"][
                "metadata_shortcut_audit"
            ] = {
                "role": (
                    "audit_b_metadata_shortcut_audit_authorization"
                ),
                "path": "authorization.json",
                "file_sha256": dataset_common.sha256_file(
                    receipt_path
                ),
                "content_sha256": receipt_value[
                    "canonical_self_hash"
                ],
            }
            authorization["required_prior_receipts"] = []
            with mock.patch.object(
                dataset_common,
                "repo_path",
                side_effect=lambda relative: root / relative,
            ):
                with self.assertRaises(
                    shortcut_common.ShortcutAuditError
                ):
                    shortcut_common.require_split_supervision_authorization(
                        lock,
                        split="audit_b",
                        operation="metadata_shortcut_audit",
                    )

    def test_small_artificial_audit_is_deterministic(self) -> None:
        first_report, first_oof = audit_runner.run_audit(
            projection_rows=self.projections,
            label_rows=self.labels,
            split="train",
            expected_world_count=5,
            bootstrap_replicates=31,
        )
        second_report, second_oof = audit_runner.run_audit(
            projection_rows=list(reversed(self.projections)),
            label_rows=list(reversed(self.labels)),
            split="train",
            expected_world_count=5,
            bootstrap_replicates=31,
        )
        self.assertEqual(first_report, second_report)
        self.assertEqual(first_oof, second_oof)
        self.assertEqual(len(first_oof), 200)
        self.assertIn(
            first_report["status"],
            {
                "PASS_METADATA_SHORTCUT_ONLY",
                "FAIL_METADATA_SHORTCUT_ONLY",
            },
        )
        self.assertFalse(first_report["pass_dataset_only_granted"])
        self.assertEqual(first_report["bootstrap_replicates"], 31)

    def test_exact_environment_gate_is_explicit(self) -> None:
        exact = (
            np.__version__
            == self.lock["statistics"]["numpy_version"]
            and sklearn.__version__
            == self.lock["statistics"]["scikit_learn_version"]
        )
        if exact:
            audit_runner.require_exact_environment(self.lock)
        else:
            with self.assertRaises(
                shortcut_common.ShortcutAuditError
            ):
                audit_runner.require_exact_environment(self.lock)

    def test_atomic_release_helper_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "release"
            shortcut_common.publish_directory(
                target,
                writer=lambda stage: dataset_common.write_json(
                    stage / "receipt.json",
                    {"ok": True},
                ),
            )
            with self.assertRaises(FileExistsError):
                shortcut_common.publish_directory(
                    target,
                    writer=lambda stage: dataset_common.write_json(
                        stage / "receipt.json",
                        {"ok": False},
                    ),
                )
            self.assertEqual(
                dataset_common.load_json(target / "receipt.json"),
                {"ok": True},
            )
            failure_path = shortcut_common.publish_stage_failure(
                lock=self.lock,
                lock_path=self.lock_path,
                split="train",
                stage="test_no_replace_failure",
                producer_relative_path=(
                    "tests/"
                    "test_step28_v13_metadata_shortcut_audit_contracts.py"
                ),
                output_dir=target,
                error=FileExistsError("target exists"),
            )
            self.assertNotEqual(failure_path, target)
            self.assertTrue(
                failure_path.name.startswith(
                    "release.failure-"
                )
            )
            failure = dataset_common.load_json(
                failure_path / shortcut_common.FAILURE_MANIFEST_FILENAME
            )
            shortcut_common.validate_self_hash(
                failure,
                label="test failure receipt",
            )
            self.assertEqual(
                failure["status"],
                "INVALID_STAGE_EXECUTION",
            )


if __name__ == "__main__":
    unittest.main()
