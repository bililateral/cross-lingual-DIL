from __future__ import annotations

import copy
import inspect
import io
import json
import sys
import unittest
import warnings
from pathlib import Path

import joblib
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step7_v4_1_select_style_free_m0 as selection


class StaticFactory:
    def __init__(self, values: dict[str, list[float]]) -> None:
        self.values = values
        self.calls: list[tuple[list[dict], list[dict]]] = []

    def design(
        self,
        fit_rows: list[dict],
        hold_rows: list[dict],
        feature_names: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[float], dict]:
        self.calls.append((list(fit_rows), list(hold_rows)))
        fit = np.asarray(
            [self.values[row["pair_uid"]] for row in fit_rows],
            dtype=np.float64,
        )
        hold = np.asarray(
            [self.values[row["pair_uid"]] for row in hold_rows],
            dtype=np.float64,
        )
        medians = np.median(fit, axis=0)
        return (
            fit,
            hold,
            medians.tolist(),
            {
                "fit_pair_count": len(fit_rows),
                "target_pair_count": len(hold_rows),
                "feature_reference_sha256": selection.canonical_hash(
                    sorted(row["pair_uid"] for row in fit_rows)
                ),
            },
        )


def synthetic_rows() -> tuple[list[dict], dict[str, list[float]]]:
    rows = []
    values = {}
    for index in range(18):
        label = "positive" if index % 3 == 0 else "negative"
        pair_uid = f"pair_{index:03d}"
        rows.append(
            {
                "pair_uid": pair_uid,
                "component_id": f"component_{index:03d}",
                "review_label": label,
                "seller_uid_left": f"left_{index:03d}",
                "seller_uid_right": f"right_{index:03d}",
            }
        )
        values[pair_uid] = [
            float(index) / 17.0,
            float(index % 5) / 4.0,
        ]
    return rows, values


class Step7V41StyleFreeClassifierContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = selection.load_policy(require_frozen=False)
        cls.parent_policy = selection.parent_common.load_json(
            selection.resolve(
                cls.policy["parent_contract"]["policy_path"]
            )
        )

    def test_candidate_universe_is_exactly_style_free(self) -> None:
        specs = selection.candidate_specs(
            self.policy, self.parent_policy
        )
        self.assertEqual(len(specs), 22)
        self.assertEqual(specs[0]["id"], selection.NULL_CANDIDATE_ID)
        forbidden = tuple(
            self.policy["forbidden_feature_name_prefixes"]
        )
        non_null = specs[1:]
        self.assertEqual(
            {item["classifier_id"] for item in non_null},
            set(selection.CLASSIFIER_IDS),
        )
        self.assertEqual(
            {item["feature_set_id"] for item in non_null},
            set(selection.FEATURE_SET_IDS),
        )
        for item in non_null:
            self.assertFalse(
                any(
                    name.startswith(forbidden)
                    for name in item["feature_names"]
                )
            )
            self.assertNotIn("pcm6", item["blocks"])
            self.assertNotIn("mstyle6", item["blocks"])
            self.assertNotIn("stylometry22", item["blocks"])

    def test_each_classifier_receives_the_same_feature_sets(self) -> None:
        specs = selection.candidate_specs(
            self.policy, self.parent_policy
        )[1:]
        by_classifier = {
            classifier_id: {
                item["feature_set_id"]: item["feature_names"]
                for item in specs
                if item["classifier_id"] == classifier_id
            }
            for classifier_id in selection.CLASSIFIER_IDS
        }
        reference = by_classifier[selection.CLASSIFIER_IDS[0]]
        for classifier_id in selection.CLASSIFIER_IDS[1:]:
            self.assertEqual(by_classifier[classifier_id], reference)

    def test_classifier_grids_are_closed_and_deterministic(self) -> None:
        self.assertEqual(
            len(
                selection.hyperparameter_grid(
                    self.policy, "l2_logistic"
                )
            ),
            7,
        )
        self.assertEqual(
            len(
                selection.hyperparameter_grid(
                    self.policy, "rbf_svm"
                )
            ),
            9,
        )
        self.assertEqual(
            len(
                selection.hyperparameter_grid(
                    self.policy, "lightgbm"
                )
            ),
            8,
        )
        fixed = self.policy["classifiers"]["lightgbm"][
            "fixed_parameters"
        ]
        self.assertTrue(fixed["deterministic"])
        self.assertTrue(fixed["force_col_wise"])
        self.assertEqual(fixed["num_threads"], 1)

    def test_corrected_float64_solver_is_explicitly_pinned(self) -> None:
        record = self.policy["implementation"][
            "corrected_float64_logistic_solver"
        ]
        self.assertEqual(
            selection.sha256_file(selection.resolve(record["path"])),
            record["sha256"],
        )
        source_text = inspect.getsource(
            selection.fit_corrected_logistic
        )
        self.assertIn(
            "corrected_logistic_solver.fit_logistic", source_text
        )
        self.assertNotIn(
            "parent_selector.fit_logistic", source_text
        )

    def test_style_free_loader_does_not_call_parent_four_model_loader(
        self,
    ) -> None:
        source_text = inspect.getsource(
            selection.load_style_free_parent_data
        )
        self.assertNotIn("verify_gpu_outputs", source_text)
        fixed_loader = inspect.getsource(
            selection.load_style_free_fixed_features
        )
        self.assertIn('"multilingual_e5_large"', fixed_loader)
        self.assertIn('"labse"', fixed_loader)
        self.assertNotIn('"pcm_multilingual_authorship"', fixed_loader)
        self.assertNotIn('"mstyledistance"', fixed_loader)

    def test_pinned_inputs_match_current_bytes(self) -> None:
        audit = selection.verify_pinned_inputs(self.policy)
        self.assertEqual(set(audit), set(self.policy["pinned_inputs"]))
        self.assertEqual(len(audit), 10)

    def test_weighted_standardizer_uses_only_supplied_rows(self) -> None:
        matrix = np.asarray([[0.0, 2.0], [2.0, 6.0]])
        weights = np.asarray([1.0, 3.0])
        standardized, mean, scale = selection.weighted_standardizer(
            matrix, weights
        )
        np.testing.assert_allclose(mean, [1.5, 5.0])
        np.testing.assert_allclose(
            np.average(standardized, axis=0, weights=weights),
            [0.0, 0.0],
            atol=1e-15,
        )
        self.assertTrue(np.all(scale > 0.0))

    def test_svm_calibration_refits_features_inside_component_folds(
        self,
    ) -> None:
        rows, values = synthetic_rows()
        factory = StaticFactory(values)
        matrix = np.asarray(
            [values[row["pair_uid"]] for row in rows],
            dtype=np.float64,
        )
        artifact = selection.fit_classifier(
            self.policy,
            "rbf_svm",
            {"c": 1.0, "gamma_multiplier": 1.0},
            matrix,
            rows,
            seed=2026072493,
            factory=factory,
            feature_names=["f0", "f1"],
        )
        self.assertEqual(len(factory.calls), 3)
        for fit_rows, hold_rows in factory.calls:
            fit_components = {
                row["component_id"] for row in fit_rows
            }
            hold_components = {
                row["component_id"] for row in hold_rows
            }
            self.assertFalse(fit_components & hold_components)
            self.assertEqual(
                {row["review_label"] for row in fit_rows},
                {"positive", "negative"},
            )
            self.assertEqual(
                {row["review_label"] for row in hold_rows},
                {"positive", "negative"},
            )
        audit = selection.compact_fit_audit(
            artifact["fit_audit"]
        )
        self.assertEqual(audit["calibration_fold_count"], 3)
        self.assertEqual(
            audit["calibration_component_overlap_count"], 0
        )
        self.assertTrue(audit["solver_converged"])

    def test_all_classifier_families_fit_and_joblib_replay(self) -> None:
        rows, values = synthetic_rows()
        matrix = np.asarray(
            [values[row["pair_uid"]] for row in rows],
            dtype=np.float64,
        )
        for classifier_id in selection.CLASSIFIER_IDS:
            with self.subTest(classifier_id=classifier_id):
                factory = StaticFactory(values)
                parameters = selection.hyperparameter_grid(
                    self.policy, classifier_id
                )[0]
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    artifact = selection.fit_classifier(
                        self.policy,
                        classifier_id,
                        parameters,
                        matrix,
                        rows,
                        seed=selection.derived_seed(
                            "unit", classifier_id
                        ),
                        factory=factory,
                        feature_names=["f0", "f1"],
                    )
                    first = selection.apply_classifier(
                        matrix, artifact
                    )
                self.assertEqual(caught, [])
                self.assertEqual(first.shape, (len(rows),))
                self.assertTrue(np.all(np.isfinite(first)))
                self.assertTrue(np.all((first >= 0.0) & (first <= 1.0)))
                payload = selection.serialize_joblib(
                    {"classifier_artifact": artifact}
                )
                replayed = joblib.load(io.BytesIO(payload))
                second = selection.apply_classifier(
                    matrix, replayed["classifier_artifact"]
                )
                np.testing.assert_array_equal(first, second)

    def test_generic_inner_tuning_completes_for_all_families(self) -> None:
        rows, values = synthetic_rows()
        for classifier_id in selection.CLASSIFIER_IDS:
            with self.subTest(classifier_id=classifier_id):
                factory = StaticFactory(values)
                tuned = selection.tune_classifier(
                    self.policy,
                    factory,
                    rows,
                    ["f0", "f1"],
                    classifier_id,
                    fold_count=3,
                    fold_seed=2026072497,
                )
                self.assertEqual(
                    tuned["classifier_id"], classifier_id
                )
                self.assertTrue(
                    np.all(np.isfinite(tuned["oof_scores"]))
                )
                self.assertGreater(
                    tuned["formal_model_fit_count"], 0
                )
                self.assertEqual(
                    len(tuned["fold_diagnostics"]), 3
                )

    def test_valid_labels_are_opened_only_after_both_locks(self) -> None:
        source_text = inspect.getsource(selection.run_selection)
        valid_open = source_text.index(
            'parent_policy, pair_rows, "valid"'
        )
        train_lock_write = source_text.index(
            "parent_common.write_json_immutable(lock_path"
        )
        blind_lock_write = source_text.index(
            "parent_common.write_json_immutable(\n"
            "        blind_lock_path"
        )
        self.assertGreater(valid_open, train_lock_write)
        self.assertGreater(valid_open, blind_lock_write)
        self.assertNotIn(
            'load_evidence_split', source_text[:valid_open]
        )

    def test_no_clone_path_calls_complete_nested_selection_again(
        self,
    ) -> None:
        source_text = inspect.getsource(selection.run_selection)
        self.assertGreaterEqual(
            source_text.count("run_nested_selection("), 2
        )
        self.assertIn(
            'progress_label="no-clone"', source_text
        )

    def test_posthoc_policy_can_never_formally_certify_m0(self) -> None:
        rule = self.policy["selection_rule"]
        self.assertFalse(
            rule["post_hoc_design_can_formally_certify_m0"]
        )
        source_text = inspect.getsource(selection.assess_selection)
        self.assertIn('"formal_m0_certified": False', source_text)
        self.assertIn(
            '"no_transfer_capable_m0"', source_text
        )

    def test_policy_validation_rejects_style_feature_prefix_relaxation(
        self,
    ) -> None:
        modified = copy.deepcopy(self.policy)
        modified["forbidden_feature_name_prefixes"] = ["style_"]
        with self.assertRaisesRegex(
            ValueError, "forbidden-prefix"
        ):
            selection.validate_policy(
                modified, require_frozen=False
            )

    def test_policy_validation_rejects_dependency_drift(self) -> None:
        modified = copy.deepcopy(self.policy)
        modified["dependencies"]["scikit_learn"] = "0.0.0"
        with self.assertRaisesRegex(
            ValueError, "dependency versions"
        ):
            selection.validate_policy(
                modified, require_frozen=False
            )


if __name__ == "__main__":
    unittest.main()
