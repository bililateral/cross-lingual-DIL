from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import step28_v13_exact_candidate_preflight as preflight  # noqa: E402


class ExactPreflightCheckpointContracts(unittest.TestCase):
    def test_npz_checkpoint_is_exact_and_no_replace(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path = Path(directory) / "checkpoint.npz"
            expected = np.asarray([[1.0, 2.0]], dtype=np.float64)
            preflight._write_npz_no_replace(path, x=expected)
            with np.load(path, allow_pickle=False) as observed:
                np.testing.assert_array_equal(observed["x"], expected)
            with self.assertRaises(FileExistsError):
                preflight._write_npz_no_replace(path, x=expected)

    def test_shortcut_stage_persists_report_and_replay_arrays(self) -> None:
        shortcut_module = preflight.shortcut_audit
        builder_module = preflight.builder
        originals = {
            "shortcut": shortcut_module.run_audit,
            "identity33": builder_module._identity33_matrix_audit,
            "retrieval": builder_module._build_retrieval,
            "validate": builder_module._validate_payload,
        }

        def fake_shortcut(**kwargs):
            kwargs["evidence_sink"](
                {
                    "folds": np.asarray([0], dtype=np.int64),
                    "score_logistic_l2": np.asarray(
                        [0.1],
                        dtype=np.float64,
                    ),
                    "score_gradient_tree": np.asarray(
                        [0.2],
                        dtype=np.float64,
                    ),
                    "score_rbf_svm": np.asarray(
                        [0.3],
                        dtype=np.float64,
                    ),
                    "bootstrap_statistics": np.asarray(
                        [0.5, 0.5, 0.5],
                        dtype=np.float64,
                    ),
                }
            )
            return {"status": "PASS_METADATA_SHORTCUT_ONLY"}, [
                {"canonical_pair_uid": "pair-1"}
            ]

        shortcut_module.run_audit = fake_shortcut
        try:
            with tempfile.TemporaryDirectory(dir=ROOT) as directory:
                prefix = Path(directory) / "audit_b"
                targets = preflight._assert_fresh_checkpoint_targets(prefix)
                overlay_path = ROOT / (
                    "schema/step28_v13_training_ready_dataset_policy.json"
                )
                restore = preflight._install_checkpoint_wrappers(
                    targets=targets,
                    split="audit_b",
                    overlay_path=overlay_path,
                )
                try:
                    row = {
                        "world_uid": "world-1",
                        "canonical_pair_uid": "pair-1",
                        **{
                            name: float(index)
                            for index, name in enumerate(
                                preflight.shortcut_common.PAIR_FEATURES
                            )
                        },
                    }
                    report, oof = shortcut_module.run_audit(
                        projection_rows=[row],
                        label_rows=[
                            {
                                "canonical_pair_uid": "pair-1",
                                "label": "1",
                            }
                        ],
                    )
                finally:
                    restore()
                self.assertEqual(
                    report["status"],
                    "PASS_METADATA_SHORTCUT_ONLY",
                )
                self.assertEqual(len(oof), 1)
                self.assertTrue(targets["shortcut"].is_file())
                with np.load(
                    targets["shortcut_arrays"],
                    allow_pickle=False,
                ) as arrays:
                    self.assertEqual(arrays["x"].shape, (1, 14))
                    self.assertEqual(arrays["y"].tolist(), [1])
                    self.assertEqual(
                        arrays["pair_uids"].tolist(),
                        ["pair-1"],
                    )
                    self.assertEqual(
                        arrays["score_rbf_svm"].tolist(),
                        [0.3],
                    )
                    self.assertEqual(
                        arrays["bootstrap_statistics"].shape,
                        (3,),
                    )
                checkpoint = preflight.common.load_json(
                    targets["shortcut"]
                )
                self_hash = checkpoint.pop("canonical_self_hash")
                self.assertEqual(
                    self_hash,
                    preflight.common.canonical_sha256(checkpoint),
                )
                with self.assertRaises(FileExistsError):
                    preflight._assert_fresh_checkpoint_targets(prefix)
                self.assertIs(shortcut_module.run_audit, fake_shortcut)
        finally:
            shortcut_module.run_audit = originals["shortcut"]
            builder_module._identity33_matrix_audit = originals["identity33"]
            builder_module._build_retrieval = originals["retrieval"]
            builder_module._validate_payload = originals["validate"]

    def test_registry_evidence_rejects_bootstrap_array_forgery(self) -> None:
        preflight.builder._EXACT_PREFLIGHT_FULL_REPLAY_CACHE.clear()
        overlay = preflight.common.load_json(
            preflight.builder.DEFAULT_OVERLAY
        )
        split = "train"
        world_count = overlay["world_counts"][split]
        rows_per_world = 40
        row_count = world_count * rows_per_world
        bootstrap_replicates = overlay["shortcut_gate"][
            "bootstrap_replicates"
        ]
        world_uids = np.asarray(
            [
                f"world-{world:04d}"
                for world in range(world_count)
                for _row in range(rows_per_world)
            ]
        )
        pair_uids = np.asarray(
            [f"pair-{index:05d}" for index in range(row_count)]
        )
        y = np.tile(
            np.asarray([1] * 16 + [0] * 24, dtype=np.int8),
            world_count,
        )
        features = np.asarray(
            preflight.shortcut_common.PAIR_FEATURES
        )
        x = np.zeros(
            (row_count, len(features)),
            dtype=np.float64,
        )
        fold_map = preflight.shortcut_audit.assign_world_folds(
            world_uids.tolist(),
            seed=overlay["shortcut_gate"]["fold_seed"],
            fold_count=overlay["shortcut_gate"]["fold_count"],
        )
        folds = np.asarray(
            [fold_map[value] for value in world_uids.tolist()],
            dtype=np.int64,
        )
        score = np.zeros(row_count, dtype=np.float64)
        fold_audit = [
            {
                "fold": fold,
                "train_world_count": len(
                    {
                        world_uid
                        for world_uid, assigned in fold_map.items()
                        if assigned != fold
                    }
                ),
                "test_world_count": len(
                    {
                        world_uid
                        for world_uid, assigned in fold_map.items()
                        if assigned == fold
                    }
                ),
                "train_row_count": int(np.sum(folds != fold)),
                "test_row_count": int(np.sum(folds == fold)),
                "train_class_count": 2,
                "test_class_count": 2,
            }
            for fold in range(overlay["shortcut_gate"]["fold_count"])
        ]
        bootstrap_statistics = np.full(
            bootstrap_replicates,
            0.5,
            dtype=np.float64,
        )
        split_seed = preflight.shortcut_audit._split_bootstrap_seed(
            overlay["shortcut_gate"]["bootstrap_base_seed"],
            split,
        )
        draws = np.random.Generator(
            np.random.PCG64DXSM(split_seed)
        ).integers(
            0,
            world_count,
            size=(bootstrap_replicates, world_count),
            dtype=np.int64,
        )
        draw_sha256 = hashlib.sha256(
            np.ascontiguousarray(
                draws.astype(">u8", copy=False)
            ).tobytes(order="C")
        ).hexdigest()
        arrays = {
            "feature_names": features,
            "x": x,
            "y": y,
            "pair_uids": pair_uids,
            "world_uids": world_uids,
            "folds": folds,
            "score_logistic_l2": score,
            "score_gradient_tree": score.copy(),
            "score_rbf_svm": score.copy(),
            "bootstrap_statistics": bootstrap_statistics,
        }
        projection_rows = [
            {
                "world_uid": str(world_uids[index]),
                "canonical_pair_uid": str(pair_uids[index]),
                **{
                    name: float(x[index, feature_index])
                    for feature_index, name in enumerate(
                        preflight.shortcut_common.PAIR_FEATURES
                    )
                },
            }
            for index in range(row_count)
        ]
        label_rows = [
            {
                "canonical_pair_uid": str(pair_uids[index]),
                "label": str(int(y[index])),
            }
            for index in range(row_count)
        ]
        oof_rows = [
            {
                "canonical_pair_uid": str(pair_uids[index]),
                "world_uid": str(world_uids[index]),
                "label": str(int(y[index])),
                "fold": str(int(folds[index])),
                "score_logistic_l2": "0",
                "score_gradient_tree": "0",
                "score_rbf_svm": "0",
            }
            for index in range(row_count)
        ]
        arrays.update(
            {
                "projection_rows_canonical_json_utf8": np.frombuffer(
                    preflight._canonical_sorted_rows_bytes(
                        projection_rows,
                        order_fields=(
                            "world_uid",
                            "canonical_pair_uid",
                        ),
                    ),
                    dtype=np.uint8,
                ).copy(),
                "label_rows_canonical_json_utf8": np.frombuffer(
                    preflight._canonical_sorted_rows_bytes(
                        label_rows,
                        order_fields=("canonical_pair_uid",),
                    ),
                    dtype=np.uint8,
                ).copy(),
                "oof_rows_canonical_json_utf8": np.frombuffer(
                    preflight.common.canonical_json_bytes(oof_rows),
                    dtype=np.uint8,
                ).copy(),
            }
        )
        shortcut_report = {
            "status": "PASS_METADATA_SHORTCUT_ONLY",
            "model_metrics": {
                model: {
                    "roc_auc": 0.5,
                    "roc_auc_symmetric": 0.5,
                }
                for model in preflight.shortcut_audit.MODEL_ORDER
            },
            "point_statistic_max_auc_symmetric": 0.5,
            "point_gate_pass": True,
            "bootstrap_draw_matrix_sha256": draw_sha256,
            "bootstrap_95_upper": 0.5,
            "bootstrap_upper_gate_pass": True,
            "fold_audit": fold_audit,
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            report_path = root / "formal.json"
            targets = preflight._checkpoint_targets(
                root / "formal.checkpoint"
            )
            preflight._write_checkpoint_json(
                targets["started"],
                {
                    "version": preflight.CHECKPOINT_VERSION,
                    "stage": "preflight_started",
                    "split": split,
                    "world_count": world_count,
                    "bootstrap_replicates": bootstrap_replicates,
                    "design_only": True,
                    "formal_private_structure_key_created_or_read": False,
                    "overlay_sha256": "d" * 64,
                    "implementation_contract_sha256": "a" * 64,
                    "builder_source_closure_sha256": "b" * 64,
                    "builder_implementation_sha256": "c" * 64,
                },
            )
            preflight._write_npz_no_replace(
                targets["shortcut_arrays"],
                **arrays,
            )

            def write_shortcut_checkpoint() -> None:
                preflight._write_checkpoint_json(
                    targets["shortcut"],
                    {
                        "version": preflight.CHECKPOINT_VERSION,
                        "stage": "metadata_shortcut_audit",
                        "stage_completed": True,
                        "split": split,
                        "design_only": True,
                        "formal_private_structure_key_created_or_read": False,
                        "elapsed_seconds": 0.0,
                        "projection_row_count": row_count,
                        "label_row_count": row_count,
                        "oof_row_count": row_count,
                        "oof_rows_sha256": (
                            preflight.common.canonical_sha256(oof_rows)
                        ),
                        "projection_rows_sha256": (
                            preflight.common.canonical_rows_sha256(
                                projection_rows,
                                order_fields=(
                                    "world_uid",
                                    "canonical_pair_uid",
                                ),
                            )
                        ),
                        "label_rows_sha256": (
                            preflight.common.canonical_rows_sha256(
                                label_rows,
                                order_fields=("canonical_pair_uid",),
                            )
                        ),
                        "array_checkpoint": {
                            "path": targets["shortcut_arrays"]
                            .relative_to(ROOT)
                            .as_posix(),
                            "sha256": preflight.common.sha256_file(
                                targets["shortcut_arrays"]
                            ),
                        },
                        "array_schema": {
                            name: {
                                "shape": list(value.shape),
                                "dtype": value.dtype.str,
                            }
                            for name, value in sorted(arrays.items())
                        },
                        "metadata_shortcut_audit": shortcut_report,
                        "overlay_sha256": "d" * 64,
                    },
                )

            write_shortcut_checkpoint()
            identity_audit = {"feature_count": 33}
            preflight._write_checkpoint_json(
                targets["identity33"],
                {
                    "version": preflight.CHECKPOINT_VERSION,
                    "stage": "identity33_matrix_audit",
                    "stage_completed": True,
                    "split": split,
                    "elapsed_seconds": 0.0,
                    "audit": identity_audit,
                },
            )
            preflight._write_checkpoint_json(
                targets["aggregate"],
                {
                    "version": preflight.CHECKPOINT_VERSION,
                    "stage": "aggregate_payload_validation",
                    "stage_completed": True,
                    "split": split,
                    "elapsed_seconds": 0.0,
                    "audit": {
                        "all_keysets_and_foreign_keys_exact": True,
                        "all_source_dataset_names_training_ready": True,
                        "identity_values_replayed_exactly": True,
                        "world_uid_count": world_count,
                        "seller_uid_count": (
                            world_count
                            * overlay["sellers_per_world"]
                        ),
                        "pair_uid_count": (
                            world_count
                            * overlay["complete_pairs_per_world"]
                        ),
                    },
                },
            )
            report = {
                "candidate_count": row_count,
                "positive_count": world_count * 16,
                "implementation_contract_sha256": "a" * 64,
                "builder_source_closure_sha256": "b" * 64,
                "builder_implementation_sha256": "c" * 64,
                "overlay_sha256": "d" * 64,
                "metadata_shortcut_audit": shortcut_report,
                "identity33_matrix_audit": identity_audit,
                "checkpoint_manifest": preflight._checkpoint_manifest(
                    targets=targets,
                    split=split,
                ),
            }
            replayed_scores = {
                model: score.copy()
                for model in preflight.shortcut_audit.MODEL_ORDER
            }

            def fake_compute_oof_scores(**_kwargs):
                return replayed_scores, folds.copy(), fold_audit

            def fake_world_bootstrap_upper(**kwargs):
                kwargs["statistics_sink"](
                    bootstrap_statistics.copy()
                )
                return 0.5, draw_sha256

            with (
                mock.patch.object(
                    preflight.shortcut_audit,
                    "compute_oof_scores",
                    side_effect=fake_compute_oof_scores,
                ) as compute_mock,
                mock.patch.object(
                    preflight.shortcut_audit,
                    "world_bootstrap_upper",
                    side_effect=fake_world_bootstrap_upper,
                ) as bootstrap_mock,
            ):
                preflight.builder._validate_exact_preflight_checkpoint_evidence(
                    report=report,
                    report_path=report_path,
                    split=split,
                    overlay=overlay,
                )
                self.assertEqual(compute_mock.call_count, 1)
                self.assertEqual(bootstrap_mock.call_count, 1)

            arrays["bootstrap_statistics"] = (
                bootstrap_statistics.copy()
            )
            arrays["bootstrap_statistics"][123] = 0.51
            targets["shortcut_arrays"].unlink()
            preflight._write_npz_no_replace(
                targets["shortcut_arrays"],
                **arrays,
            )
            targets["shortcut"].unlink()
            write_shortcut_checkpoint()
            report["checkpoint_manifest"] = (
                preflight._checkpoint_manifest(
                    targets=targets,
                    split=split,
                )
            )
            with (
                mock.patch.object(
                    preflight.shortcut_audit,
                    "compute_oof_scores",
                    side_effect=fake_compute_oof_scores,
                ),
                mock.patch.object(
                    preflight.shortcut_audit,
                    "world_bootstrap_upper",
                    side_effect=fake_world_bootstrap_upper,
                ),
                self.assertRaisesRegex(
                    preflight.common.ContractError,
                    "full bootstrap replay drift",
                ),
            ):
                preflight.builder._validate_exact_preflight_checkpoint_evidence(
                    report=report,
                    report_path=report_path,
                    split=split,
                    overlay=overlay,
                )

            arrays["bootstrap_statistics"] = (
                bootstrap_statistics.copy()
            )
            forged_logistic = score.copy()
            forged_logistic[0] = 0.1
            arrays["score_logistic_l2"] = forged_logistic
            oof_rows[0]["score_logistic_l2"] = format(0.1, ".17g")
            arrays["oof_rows_canonical_json_utf8"] = np.frombuffer(
                preflight.common.canonical_json_bytes(oof_rows),
                dtype=np.uint8,
            ).copy()
            forged_auc, forged_symmetric = (
                preflight.shortcut_audit.symmetric_auc(
                    y.astype(np.int64, copy=False),
                    forged_logistic,
                )
            )
            shortcut_report["model_metrics"]["logistic_l2"] = {
                "roc_auc": forged_auc,
                "roc_auc_symmetric": forged_symmetric,
            }
            shortcut_report["point_statistic_max_auc_symmetric"] = (
                forged_symmetric
            )
            targets["shortcut_arrays"].unlink()
            preflight._write_npz_no_replace(
                targets["shortcut_arrays"],
                **arrays,
            )
            targets["shortcut"].unlink()
            write_shortcut_checkpoint()
            report["checkpoint_manifest"] = (
                preflight._checkpoint_manifest(
                    targets=targets,
                    split=split,
                )
            )
            with (
                mock.patch.object(
                    preflight.shortcut_audit,
                    "compute_oof_scores",
                    side_effect=fake_compute_oof_scores,
                ),
                mock.patch.object(
                    preflight.shortcut_audit,
                    "world_bootstrap_upper",
                    side_effect=fake_world_bootstrap_upper,
                ),
                self.assertRaisesRegex(
                    preflight.common.ContractError,
                    "frozen OOF score replay drift",
                ),
            ):
                preflight.builder._validate_exact_preflight_checkpoint_evidence(
                    report=report,
                    report_path=report_path,
                    split=split,
                    overlay=overlay,
                )


if __name__ == "__main__":
    unittest.main()
