from __future__ import annotations

import ast
import base64
import copy
import hashlib
import inspect
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common  # noqa: E402
import step28_v13_candidate_sampling as candidate_sampling  # noqa: E402
import step28_v13_compare_independent_dgp_replay as replay_compare_launcher  # noqa: E402
import step28_v13_feature_derangement as feature_derangement  # noqa: E402
import step28_v13_generate_dataset as dataset_generator  # noqa: E402
import step28_v13_history_features as history_features  # noqa: E402
import step28_v13_identity_plan as identity_plan_mod  # noqa: E402
import step28_v13_identity_values as identity_values_mod  # noqa: E402
import step28_v13_independent_dgp_comparator as independent_comparator  # noqa: E402
import step28_v13_independent_private_dgp_replay as independent_replay  # noqa: E402
import step28_v13_integrity_receipts as integrity_receipts  # noqa: E402
import step28_v13_placebo_rewire as placebo_rewire  # noqa: E402
import step28_v13_placebo_support as placebo_support  # noqa: E402
import step28_v13_production_chain as production  # noqa: E402
import step28_v13_producer_dgp_projection as producer_projection  # noqa: E402
import step28_v13_profiles as profiles_mod  # noqa: E402
import step28_v13_run_independent_dgp_replay as replay_launcher  # noqa: E402
import step28_v13_safe_slots as safe_slots  # noqa: E402
import step28_v13_smoke_private_regeneration as smoke_regeneration  # noqa: E402
import step28_v13_structure as structure  # noqa: E402
import step28_v13_text_renderer as text_renderer  # noqa: E402
import step28_v13_world_builder as world_builder  # noqa: E402
import step3_build_seller_profiles as step3  # noqa: E402
import step7_v3_1_source_data as source  # noqa: E402


class Step28V13SyntheticDatasetContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = common.load_policy(
            ROOT / "schema" / "step28_v13_synthetic_chinese_dataset_policy.json",
            mode="development_smoke",
        )
        cls.template = common.load_json(
            ROOT / "schema" / "step28_v13_synthetic_text_templates.json"
        )
        cls.fixture = common.load_json(
            ROOT / "schema" / "step28_v13_parser_template_fixture.json"
        )
        cls.style = common.load_json(
            common.repo_path(
                cls.policy["style_reference_boundary"][
                    "generator_release_inputs"
                ]["profile"]["path"]
            )
        )
        cls.record = structure.build_mode_world_pool(
            cls.policy, mode="development_smoke"
        )[0]
        cls.mode = "development_smoke"
        cls.split = cls.record["split"]
        cls.candidate_policy = (
            candidate_sampling.build_public_candidate_policy(
                cls.policy, mode=cls.mode, split=cls.split
            )
        )
        cls.candidate_key_hex = str(
            cls.policy["randomness"][cls.mode]["candidate_key_hex"]
        )
        cls.candidate_integrity_context = (
            integrity_receipts.build_candidate_integrity_context(
                cls.policy,
                candidate_policy=cls.candidate_policy,
                mode=cls.mode,
                split=cls.split,
            )
        )
        cls.structure_key = common.structure_key_for_split(
            cls.policy,
            mode="development_smoke",
            split=cls.record["split"],
        )
        cls.world = world_builder.build_world(
            policy=cls.policy,
            template=cls.template,
            fixture=cls.fixture,
            style_profile=cls.style,
            mode="development_smoke",
            world_record=cls.record,
            structure_key_hex=cls.structure_key,
        )
        cls.observed_uid_pools = (
            independent_comparator.build_observed_uid_pools(
                world_uid=cls.record["world_uid"],
                sellers=cls.world["public"]["sellers"],
                items=cls.world["public"]["items"],
            )
        )
        cls.independent_expected = independent_replay.replay_typed_dgp(
            cls.policy,
            mode=cls.mode,
            split=cls.split,
            world_uid=cls.record["world_uid"],
            structure_key_hex=cls.structure_key,
            **cls.observed_uid_pools,
        )
        cls.producer_projection = producer_projection.project_world(
            world=cls.world,
            mode=cls.mode,
            split=cls.split,
        )
        cls.independent_replay_audit = (
            independent_comparator.compare_typed_dgp(
                expected_replay=cls.independent_expected,
                producer_projection=cls.producer_projection,
            )
        )
        cls.regeneration_audit = (
            smoke_regeneration.validate_producer_regeneration_match(
                cls.policy,
                mode=cls.mode,
                split=cls.split,
                template=cls.template,
                fixture=cls.fixture,
                style_profile=cls.style,
                world=cls.world,
            )
        )
        cls.registry_profiles = production.registry_profiles_from_sellers(
            cls.policy,
            sellers=cls.world["public"]["sellers"],
        )
        cls.parsed = production.parse_observed_world(
            cls.policy,
            mode="development_smoke",
            split=cls.record["split"],
            sellers=cls.world["public"]["sellers"],
            items=cls.world["public"]["items"],
        )
        cls.processed = production.process_world(
            cls.policy,
            mode=cls.mode,
            split=cls.split,
            template=cls.template,
            world=cls.world,
        )
        cls.profiles, cls.profile_audit = profiles_mod.build_world_profiles(
            cls.policy,
            mode=cls.mode,
            split=cls.split,
            sellers=cls.world["public"]["sellers"],
            items=cls.processed["public"]["profile_safe_items"],
        )

    def _materialize_minimal_complete_release_children(
        self,
        release_root: Path,
        *,
        splits: tuple[str, ...] = dataset_generator.SPLITS,
    ) -> tuple[Path, dict[str, str], dict[str, Path]]:
        release_name = self.policy["development_complete_release"][
            "release_name"
        ]
        parent = release_root / release_name
        parent.mkdir(parents=True, exist_ok=True)
        payload_digests: dict[str, str] = {}
        artifact_paths: dict[str, Path] = {}
        current_hashes = {
            "policy_sha256": common.sha256_file(
                common.DEFAULT_POLICY_PATH
            ),
            "contract_sha256": common.sha256_file(
                common.repo_path(str(self.policy["contract"]["path"]))
            ),
            "template_sha256": common.sha256_file(
                common.repo_path(
                    str(self.policy["template_library"]["path"])
                )
            ),
            "fixture_sha256": common.sha256_file(
                common.repo_path(
                    str(
                        self.policy["identity_design"][
                            "role_template_parser_flag_fixture"
                        ]["path"]
                    )
                )
            ),
        }
        for ordinal, split in enumerate(splits):
            split_root = parent / split
            artifact_path = (
                split_root / "observed" / "redacted_items.jsonl"
            )
            rows = [
                {
                    "split": split,
                    "seller_uid": f"sel_fixture_{ordinal:02d}",
                }
            ]
            common.write_jsonl(artifact_path, rows)
            payload_digest = common.canonical_sha256(
                {"split": split, "rows": rows}
            )
            payload_digests[split] = payload_digest
            artifact_paths[split] = artifact_path
            manifest: dict[str, object] = {
                "version": (
                    "2026-07-29-step28-v13-"
                    "split-dataset-manifest-v5-draft"
                ),
                "status": (
                    "DEVELOPMENT_SMOKE_PASS_NOT_SCIENTIFIC_EVIDENCE"
                ),
                "mode": self.mode,
                "split": split,
                "run_id": self.policy["modes"][self.mode]["run_id"],
                "release_name": release_name,
                "scientific_metrics_produced": False,
                **current_hashes,
                "producer_sha256": dataset_generator._producer_hashes(),
                "split_payload_digest_sha256": payload_digest,
                "world_count": 1,
                "seller_count": 1,
                "item_count": 1,
                "complete_pair_count": 1,
                "candidate_pair_count": 1,
                "files": [
                    common.artifact_record(
                        artifact_path,
                        role="m0_safe_fixture",
                        root=split_root,
                    )
                ],
                "parent_manifests": [],
                "formal_use_forbidden": True,
            }
            manifest["canonical_self_hash"] = common.canonical_sha256(
                manifest
            )
            common.write_json(
                split_root / "split_manifest.json",
                manifest,
            )
        return parent, payload_digests, artifact_paths

    def _assert_independent_support_preflight(
        self,
        result: dict[str, object],
    ) -> None:
        """Recalculate all M1/M2 support gates without production helpers."""

        feature_names = [
            str(value)
            for value in self.policy["history_features"]["feature_names"]
        ]

        def matrix_index(rows):
            return {
                (
                    str(row["world_uid"]),
                    str(row["canonical_pair_uid"]),
                ): np.asarray(
                    [float(row[name]) for name in feature_names],
                    dtype=np.float64,
                )
                for row in rows
            }

        tables = result["tables"]
        m2_index = matrix_index(tables["identity33_all_pairs"])
        primary_keys = sorted(
            {
                (
                    str(row["world_uid"]),
                    str(row["canonical_pair_uid"]),
                )
                for row in tables["candidate_pairs"]
            },
            key=lambda value: (
                value[0].encode("utf-8"),
                value[1].encode("utf-8"),
            ),
        )
        full_keys = sorted(
            m2_index,
            key=lambda value: (
                value[0].encode("utf-8"),
                value[1].encode("utf-8"),
            ),
        )
        self.assertEqual(len(primary_keys), 400)
        self.assertEqual(len(full_keys), 3780)
        m2_primary = np.vstack(
            [m2_index[key] for key in primary_keys]
        )
        scale = np.sqrt(np.mean(np.square(m2_primary), axis=0))
        scale = np.where(scale <= 1e-12, 1.0, scale)
        preflight = result["support_comparability_preflight"]
        self.assertEqual(
            preflight["shared_m2_c40_rms_scale_sha256"],
            common.canonical_sha256(
                [f"{value:.17g}" for value in scale]
            ),
        )

        classifier = self.policy["placebo"]["support_classifier"]
        thresholds = self.policy["placebo"][
            "support_comparability_thresholds"
        ]
        random_seed = int(classifier["random_seed"])
        fold_count = int(classifier["world_grouped_folds"])
        worlds = sorted(
            {key[0] for key in full_keys},
            key=lambda world_uid: (
                hashlib.sha256(
                    str(random_seed).encode("ascii")
                    + b"\x1f"
                    + world_uid.encode("utf-8")
                ).digest(),
                world_uid.encode("utf-8"),
            ),
        )
        fold_by_world = {
            world_uid: index % fold_count
            for index, world_uid in enumerate(worlds)
        }
        fold_hash = common.canonical_sha256(
            [
                {
                    "world_uid": world_uid,
                    "fold": fold_by_world[world_uid],
                }
                for world_uid in sorted(
                    fold_by_world, key=lambda value: value.encode("utf-8")
                )
            ]
        )

        def correlation_difference(left, right):
            left_mean = np.mean(left, axis=0)
            right_mean = np.mean(right, axis=0)
            left_cov = np.cov(left, rowvar=False, ddof=0)
            right_cov = np.cov(right, rowvar=False, ddof=0)
            left_std = np.sqrt(np.diag(left_cov))
            right_std = np.sqrt(np.diag(right_cov))
            left_zero = left_std <= 1e-12
            right_zero = right_std <= 1e-12
            if np.any(left_zero != right_zero):
                return None, False, None
            both_zero = left_zero & right_zero
            if np.any(
                both_zero
                & ~np.isclose(
                    left_mean,
                    right_mean,
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                return None, False, None

            def correlation(covariance, standard_deviation):
                denominator = np.outer(
                    standard_deviation, standard_deviation
                )
                output = np.zeros_like(covariance)
                np.divide(
                    covariance,
                    denominator,
                    out=output,
                    where=denominator > 1e-24,
                )
                np.fill_diagonal(output, 1.0)
                return output

            differences = np.abs(
                correlation(left_cov, left_std)
                - correlation(right_cov, right_std)
            )
            flat_index = int(np.argmax(differences))
            pair = tuple(
                int(value)
                for value in np.unravel_index(
                    flat_index, differences.shape
                )
            )
            return float(differences[pair]), True, pair

        def independent_metrics(keys, m1_index):
            m2 = np.vstack([m2_index[key] for key in keys]) / scale
            m1 = np.vstack([m1_index[key] for key in keys]) / scale
            range_values = np.maximum(
                np.maximum(
                    np.min(m2, axis=0) - np.min(m1, axis=0),
                    0.0,
                ),
                np.max(m1, axis=0) - np.max(m2, axis=0),
            )
            range_index = int(np.argmax(range_values))
            mean_difference = np.mean(m1, axis=0) - np.mean(
                m2, axis=0
            )
            pooled = np.sqrt(
                (
                    np.var(m2, axis=0, ddof=0)
                    + np.var(m1, axis=0, ddof=0)
                )
                / 2.0
            )
            smd_valid = bool(
                np.all(
                    (pooled > 1e-12)
                    | np.isclose(
                        mean_difference,
                        0.0,
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
            )
            smd = np.zeros(33, dtype=np.float64)
            np.divide(
                mean_difference,
                pooled,
                out=smd,
                where=pooled > 1e-12,
            )
            smd_index = int(np.argmax(np.abs(smd)))
            maximum_smd = (
                float(abs(smd[smd_index])) if smd_valid else None
            )
            zero_values = np.abs(
                np.mean(m1 == 0.0, axis=0)
                - np.mean(m2 == 0.0, axis=0)
            )
            zero_index = int(np.argmax(zero_values))
            probabilities = np.asarray(
                thresholds["quantiles"], dtype=np.float64
            )
            quantile_values = np.abs(
                np.quantile(
                    m1, probabilities, axis=0, method="linear"
                )
                - np.quantile(
                    m2, probabilities, axis=0, method="linear"
                )
            )
            quantile_flat = int(np.argmax(quantile_values))
            quantile_index, quantile_feature = (
                int(value)
                for value in np.unravel_index(
                    quantile_flat, quantile_values.shape
                )
            )
            covariance_values = np.abs(
                np.cov(m1, rowvar=False, ddof=0)
                - np.cov(m2, rowvar=False, ddof=0)
            )
            covariance_flat = int(np.argmax(covariance_values))
            covariance_pair = tuple(
                int(value)
                for value in np.unravel_index(
                    covariance_flat, covariance_values.shape
                )
            )
            correlation_value, correlation_valid, correlation_pair = (
                correlation_difference(m2, m1)
            )

            x_rows = []
            labels = []
            folds = []
            for row_index, key in enumerate(keys):
                x_rows.extend((m2[row_index], m1[row_index]))
                labels.extend((0, 1))
                folds.extend(
                    (fold_by_world[key[0]], fold_by_world[key[0]])
                )
            x = np.vstack(x_rows)
            y = np.asarray(labels, dtype=np.int64)
            fold_array = np.asarray(folds, dtype=np.int64)
            scores = np.full(y.shape, np.nan, dtype=np.float64)
            for fold in range(fold_count):
                train = fold_array != fold
                test = fold_array == fold
                model = LogisticRegression(
                    solver=str(classifier["solver"]),
                    penalty=str(classifier["penalty"]),
                    C=float(classifier["C"]),
                    max_iter=int(classifier["max_iter"]),
                    tol=float(classifier["tol"]),
                    class_weight=classifier["class_weight"],
                    fit_intercept=bool(classifier["fit_intercept"]),
                    random_state=random_seed,
                )
                model.fit(x[train], y[train])
                self.assertTrue(
                    np.all(model.n_iter_ < int(classifier["max_iter"]))
                )
                scores[test] = model.decision_function(x[test])
            auc = float(roc_auc_score(y, scores))
            auc_symmetric = max(auc, 1.0 - auc)
            gates = {
                "range_slack_pass": (
                    float(range_values[range_index])
                    <= float(
                        thresholds[
                            "range_slack_in_m2_scale_units"
                        ]
                    )
                ),
                "standardized_mean_difference_pass": (
                    smd_valid
                    and maximum_smd is not None
                    and maximum_smd
                    <= float(
                        thresholds[
                            "maximum_absolute_standardized_mean_difference"
                        ]
                    )
                ),
                "zero_rate_difference_pass": (
                    float(zero_values[zero_index])
                    <= float(
                        thresholds[
                            "maximum_absolute_zero_rate_difference"
                        ]
                    )
                ),
                "quantile_difference_pass": (
                    float(
                        quantile_values[
                            quantile_index, quantile_feature
                        ]
                    )
                    <= float(
                        thresholds[
                            "maximum_absolute_standardized_quantile_difference"
                        ]
                    )
                ),
                "covariance_difference_pass": (
                    float(covariance_values[covariance_pair])
                    <= float(
                        thresholds[
                            "maximum_absolute_covariance_difference"
                        ]
                    )
                ),
                "correlation_difference_pass": (
                    correlation_valid
                    and correlation_value is not None
                    and correlation_value
                    <= float(
                        thresholds[
                            "maximum_absolute_correlation_difference"
                        ]
                    )
                ),
                "two_sample_auc_pass": (
                    auc_symmetric
                    <= float(
                        thresholds[
                            "two_sample_auc_symmetric_maximum"
                        ]
                    )
                ),
            }
            return {
                "maximum_range_excess": float(
                    range_values[range_index]
                ),
                "maximum_range_excess_feature": feature_names[
                    range_index
                ],
                "maximum_absolute_standardized_mean_difference": (
                    maximum_smd
                ),
                "maximum_absolute_standardized_mean_difference_feature": (
                    feature_names[smd_index] if smd_valid else None
                ),
                "standardized_mean_difference_support_valid": smd_valid,
                "maximum_absolute_zero_rate_difference": float(
                    zero_values[zero_index]
                ),
                "maximum_absolute_zero_rate_difference_feature": (
                    feature_names[zero_index]
                ),
                "maximum_absolute_standardized_quantile_difference": (
                    float(
                        quantile_values[
                            quantile_index, quantile_feature
                        ]
                    )
                ),
                "maximum_absolute_standardized_quantile_difference_feature": (
                    feature_names[quantile_feature]
                ),
                "maximum_absolute_standardized_quantile_difference_probability": (
                    float(probabilities[quantile_index])
                ),
                "maximum_absolute_covariance_difference": float(
                    covariance_values[covariance_pair]
                ),
                "maximum_absolute_covariance_difference_features": [
                    feature_names[covariance_pair[0]],
                    feature_names[covariance_pair[1]],
                ],
                "maximum_absolute_correlation_difference": (
                    correlation_value
                ),
                "maximum_absolute_correlation_difference_features": (
                    [
                        feature_names[correlation_pair[0]],
                        feature_names[correlation_pair[1]],
                    ]
                    if correlation_pair is not None
                    else None
                ),
                "correlation_support_valid": correlation_valid,
                "two_sample_auc_symmetric": auc_symmetric,
                "fold_assignment_sha256": fold_hash,
                "gates": gates,
                "all_thresholds_pass": all(gates.values()),
            }

        reported_by_seed = {
            str(row["rewire_seed_id"]): row
            for row in preflight["seed_results"]
        }
        primary_set = set(primary_keys)
        for placebo in result["placebos"]:
            seed_id = str(placebo["rewire_seed_id"])
            m1_index = matrix_index(placebo["identity33_all_pairs"])
            self.assertEqual(set(m1_index), set(m2_index))
            for world_uid in worlds:
                world_full = [
                    key for key in full_keys if key[0] == world_uid
                ]
                for universe_keys in (
                    [
                        key
                        for key in world_full
                        if key in primary_set
                    ],
                    [
                        key
                        for key in world_full
                        if key not in primary_set
                    ],
                ):
                    m2_vectors = Counter(
                        tuple(m2_index[key].tolist())
                        for key in universe_keys
                    )
                    m1_vectors = Counter(
                        tuple(m1_index[key].tolist())
                        for key in universe_keys
                    )
                    self.assertEqual(m1_vectors, m2_vectors)

            reported = reported_by_seed[seed_id]
            for name, keys in (
                ("primary_c40", primary_keys),
                ("secondary_full378", full_keys),
            ):
                expected = independent_metrics(keys, m1_index)
                observed = reported[name]
                for key, value in expected.items():
                    if isinstance(value, float):
                        self.assertTrue(
                            math.isclose(
                                float(observed[key]),
                                value,
                                rel_tol=0.0,
                                abs_tol=1e-12,
                            ),
                            msg=(
                                f"independent support metric drift: "
                                f"{seed_id}.{name}.{key}"
                            ),
                        )
                    else:
                        self.assertEqual(observed[key], value)

    def test_complete_pair_universe_and_production_chain(self) -> None:
        pairs = self.world["public"]["complete_model_pair_endpoints"]
        self.assertEqual(len(pairs), 378)
        self.assertEqual(
            len({row["canonical_pair_uid"] for row in pairs}), 378
        )
        result = production.process_world(
            self.policy,
            mode="development_smoke",
            split=self.record["split"],
            template=self.template,
            world=self.world,
        )
        self.assertEqual(
            len(result["private"]["parsed_identity_occurrences"]),
            len(self.world["private"]["identity_slots_audit"]),
        )
        guards = text_renderer.context_guard_pool(self.template)
        self.assertFalse(
            any(
                any(guard in row["description"] for guard in guards)
                for row in result["public"]["redacted_items"]
            )
        )
        self.assertFalse(
            result["private"]["redaction_structural_audit"][
                "identity_slot_count_observable_in_m0_text"
            ]
        )
        replay_audit = self.regeneration_audit
        self.assertIs(
            replay_audit["producer_regeneration_match_smoke_pass"],
            True,
        )
        self.assertRegex(
            replay_audit["producer_regeneration_match_sha256"],
            r"^[0-9a-f]{64}$",
        )
        self.assertIs(
            replay_audit["producer_regeneration_independent_replay"],
            False,
        )
        self.assertEqual(
            replay_audit["producer_regeneration_evidence_level"],
            "DEVELOPMENT_SMOKE_SAME_IMPLEMENTATION_NOT_FORMAL_SEAL",
        )

    def test_development_smoke_parent_c40_is_label_blind_and_deterministic(
        self,
    ) -> None:
        safe_rows, audit_rows, label_free_audit = (
            candidate_sampling.build_world_c40(
                self.candidate_policy,
                candidate_key_hex=self.candidate_key_hex,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                raw_observed_items=self.world["public"]["items"],
                complete_pair_endpoints=self.world["public"][
                    "complete_model_pair_endpoints"
                ],
            )
        )
        self.assertEqual(len(safe_rows), 40)
        self.assertEqual(len(audit_rows), 378)
        self.assertEqual(
            [list(row) for row in safe_rows],
            [
                self.policy["candidate_design"][
                    "public_safe_projection_columns"
                ]
            ]
            * 40,
        )
        self.assertEqual(
            [list(row) for row in audit_rows],
            [
                self.policy["candidate_design"][
                    "sampling_audit_projection_columns"
                ]
            ]
            * 378,
        )
        self.assertEqual(
            audit_rows,
            sorted(
                audit_rows,
                key=lambda row: (
                    row["world_uid"].encode("utf-8"),
                    row["canonical_pair_uid"].encode("utf-8"),
                ),
            ),
        )
        selected = {
            row["canonical_pair_uid"]: int(row["selected_rank"])
            for row in audit_rows
            if row["selected_bool"] == "true"
        }
        self.assertEqual(sorted(selected.values()), list(range(1, 41)))
        self.assertEqual(
            [row["canonical_pair_uid"] for row in safe_rows],
            [
                pair_uid
                for pair_uid, _rank in sorted(
                    selected.items(), key=lambda item: item[1]
                )
            ],
        )
        key_hex = self.policy["randomness"][self.mode]["candidate_key_hex"]
        world_uid = self.record["world_uid"]
        for row in audit_rows:
            self.assertEqual(
                row["hmac_digest_hex"],
                common.hmac_digest(
                    key_hex,
                    world_uid,
                    row["canonical_pair_uid"],
                ).hex(),
            )
            self.assertRegex(row["lexical_similarity"], r"^\d+\.\d{6}$")
            self.assertRegex(
                row["design_inclusion_probability"],
                r"^(?:\d+\.\d{12})?$",
            )
            self.assertNotIn(
                "structural_support",
                set(filter(None, row["trigger_flags"].split("|"))),
            )
        expected_global_order = sorted(
            selected,
            key=lambda pair_uid: (
                common.hmac_digest(
                    key_hex,
                    world_uid,
                    "selected_global_rank",
                    pair_uid,
                ),
                pair_uid.encode("utf-8"),
            ),
        )
        self.assertEqual(
            expected_global_order,
            [row["canonical_pair_uid"] for row in safe_rows],
        )
        self.assertGreater(
            label_free_audit["primary_layer_sizes"][
                "shared_contact_exact"
            ],
            0,
        )
        self.assertIs(
            label_free_audit["labels_or_oracle_or_model_scores_read"],
            False,
        )
        self.assertIs(
            label_free_audit["ephemeral_step4_raw_evidence_persisted"],
            False,
        )
        forbidden_raw_fields = {
            "shared_contact_values",
            "shared_title_values",
            "shared_description_values",
            "left_preview",
            "right_preview",
            "candidate_rank_score",
            "review_label",
        }
        self.assertTrue(
            all(
                not forbidden_raw_fields.intersection(row)
                for row in (*safe_rows, *audit_rows)
            )
        )

        permuted = candidate_sampling.build_world_c40(
            self.candidate_policy,
            candidate_key_hex=self.candidate_key_hex,
            mode=self.mode,
            split=self.split,
            sellers=list(reversed(self.world["public"]["sellers"])),
            raw_observed_items=list(
                reversed(self.world["public"]["items"])
            ),
            complete_pair_endpoints=list(
                reversed(
                    self.world["public"]["complete_model_pair_endpoints"]
                )
            ),
        )
        self.assertEqual(permuted[0], safe_rows)
        self.assertEqual(permuted[1], audit_rows)

    def test_c40_hamilton_and_input_failures_are_closed(self) -> None:
        self.assertEqual(
            candidate_sampling._hamilton_quotas(
                {
                    "shared_contact_exact": 3,
                    "shared_description_clone": 3,
                    "shared_title_clone": 2,
                    "profile_lexical_neighbor": 0,
                },
                total_slots=5,
                trigger_priority=candidate_sampling.REACHABLE_TRIGGERS,
            ),
            {
                "shared_contact_exact": 2,
                "shared_description_clone": 2,
                "shared_title_clone": 1,
                "profile_lexical_neighbor": 0,
            },
        )
        duplicate_pairs = copy.deepcopy(
            self.world["public"]["complete_model_pair_endpoints"]
        )
        duplicate_pairs[-1] = copy.deepcopy(duplicate_pairs[0])
        with self.assertRaisesRegex(
            common.ContractError,
            "C40 complete-pair lineage drift",
        ):
            candidate_sampling.build_world_c40(
                self.candidate_policy,
                candidate_key_hex=self.candidate_key_hex,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                raw_observed_items=self.world["public"]["items"],
                complete_pair_endpoints=duplicate_pairs,
            )
        with self.assertRaisesRegex(
            common.ContractError,
            "development-smoke only",
        ):
            candidate_sampling.build_world_c40(
                self.candidate_policy,
                candidate_key_hex=self.candidate_key_hex,
                mode="formal",
                split=self.split,
                sellers=self.world["public"]["sellers"],
                raw_observed_items=self.world["public"]["items"],
                complete_pair_endpoints=self.world["public"][
                    "complete_model_pair_endpoints"
                ],
            )

        baseline = candidate_sampling.build_world_c40(
            self.candidate_policy,
            candidate_key_hex=self.candidate_key_hex,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            raw_observed_items=self.world["public"]["items"],
            complete_pair_endpoints=self.world["public"][
                "complete_model_pair_endpoints"
            ],
        )
        changed = candidate_sampling.build_world_c40(
            self.candidate_policy,
            candidate_key_hex="00" * 32,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            raw_observed_items=self.world["public"]["items"],
            complete_pair_endpoints=self.world["public"][
                "complete_model_pair_endpoints"
            ],
        )
        baseline_trigger_projection = [
            (
                row["canonical_pair_uid"],
                row["primary_trigger"],
                row["trigger_flags"],
                row["lexical_similarity"],
                row["structural_support_flag"],
                row["layer_size"],
                row["layer_quota"],
            )
            for row in baseline[1]
        ]
        changed_trigger_projection = [
            (
                row["canonical_pair_uid"],
                row["primary_trigger"],
                row["trigger_flags"],
                row["lexical_similarity"],
                row["structural_support_flag"],
                row["layer_size"],
                row["layer_quota"],
            )
            for row in changed[1]
        ]
        self.assertEqual(
            baseline_trigger_projection,
            changed_trigger_projection,
        )
        self.assertNotEqual(baseline[0], changed[0])

    def test_background_scaffold_matches_registered_rewire_counts(
        self,
    ) -> None:
        background = [
            row
            for row in self.world["private"]["identity_assets"]
            if row["descriptor_kind"] == "background_private"
        ]
        self.assertEqual(len(background), 56)
        by_type_count = Counter(
            (
                row["identity_type"],
                int(
                    row["occurrence_counts"][
                        next(iter(row["occurrence_counts"]))
                    ]
                ),
            )
            for row in background
        )
        self.assertEqual(len(by_type_count), 14)
        self.assertEqual(set(by_type_count.values()), {4})
        per_seller_types: dict[str, set[str]] = defaultdict(set)
        per_seller_counts: dict[str, list[int]] = defaultdict(list)
        for row in background:
            seller_uid = next(iter(row["occurrence_counts"]))
            per_seller_types[seller_uid].add(row["identity_type"])
            per_seller_counts[seller_uid].append(
                int(row["occurrence_counts"][seller_uid])
            )
        self.assertEqual(len(per_seller_types), 28)
        self.assertTrue(
            all(len(identity_types) == 2 for identity_types in per_seller_types.values())
        )
        self.assertEqual(set(per_seller_counts), set(per_seller_types))
        self.assertTrue(
            all(
                sorted(occurrence_counts) == [1, 2]
                for occurrence_counts in per_seller_counts.values()
            )
        )

    def test_primary_m1_matching_is_deterministic_bijective_and_disjoint(
        self,
    ) -> None:
        pairs = self.world["public"]["complete_model_pair_endpoints"]
        candidate_rows, _audit, _generation = (
            candidate_sampling.build_world_c40(
                self.candidate_policy,
                candidate_key_hex=self.candidate_key_hex,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                raw_observed_items=self.world["public"]["items"],
                complete_pair_endpoints=pairs,
            )
        )
        pair_uids = [
            str(row["canonical_pair_uid"]) for row in candidate_rows
        ]
        selected = set(pair_uids)
        endpoints = {
            str(row["canonical_pair_uid"]): (
                str(row["seller_uid_left"]),
                str(row["seller_uid_right"]),
            )
            for row in pairs
            if str(row["canonical_pair_uid"]) in selected
        }
        seed = bytes.fromhex(
            self.policy["randomness"][self.mode][
                "rewire_key_hexes"
            ][0]
        )
        baseline = (
            feature_derangement._perfect_endpoint_disjoint_mapping(
                seed=seed,
                world_uid=str(self.record["world_uid"]),
                universe="primary_c40",
                pair_uids=pair_uids,
                endpoints=endpoints,
            )
        )
        replay = feature_derangement._perfect_endpoint_disjoint_mapping(
            seed=seed,
            world_uid=str(self.record["world_uid"]),
            universe="primary_c40",
            pair_uids=list(reversed(pair_uids)),
            endpoints=endpoints,
        )
        changed = feature_derangement._perfect_endpoint_disjoint_mapping(
            seed=bytes.fromhex(
                self.policy["randomness"][self.mode][
                    "rewire_key_hexes"
                ][1]
            ),
            world_uid=str(self.record["world_uid"]),
            universe="primary_c40",
            pair_uids=pair_uids,
            endpoints=endpoints,
        )
        self.assertEqual(baseline, replay)
        self.assertNotEqual(baseline, changed)
        self.assertEqual(set(baseline), set(pair_uids))
        self.assertEqual(set(baseline.values()), set(pair_uids))
        self.assertTrue(
            all(
                not set(endpoints[destination]).intersection(
                    endpoints[source]
                )
                for destination, source in baseline.items()
            )
        )

    def test_support_comparator_accepts_exact_joint_row_derangement(
        self,
    ) -> None:
        keys = [
            (f"w_{world:064x}", f"pair_{pair:04d}")
            for world in range(10)
            for pair in range(40)
        ]
        m2 = np.asarray(
            [
                [
                    float(
                        ((world + 3) * (pair + 5) * (feature + 7))
                        % 29
                    )
                    / 11.0
                    for feature in range(33)
                ]
                for world in range(10)
                for pair in range(40)
            ],
            dtype=np.float64,
        )
        m1 = np.vstack(
            [
                np.roll(
                    m2[world * 40 : (world + 1) * 40],
                    shift=world + 1,
                    axis=0,
                )
                for world in range(10)
            ]
        )
        result = placebo_support._comparison(
            self.policy,
            keys=keys,
            m2_raw=m2,
            m1_raw=m1,
            scale=placebo_support._shared_scale(m2),
            validity_gate=True,
        )
        self.assertIs(result["all_thresholds_pass"], True)
        self.assertEqual(result["maximum_range_excess"], 0.0)
        self.assertEqual(
            result["maximum_absolute_zero_rate_difference"], 0.0
        )
        self.assertAlmostEqual(
            result["two_sample_auc_symmetric"], 0.5, places=12
        )

    def test_integrity_multiset_hash_is_order_independent_and_keeps_duplicates(
        self,
    ) -> None:
        rows = [
            {"key": "a", "value": 1},
            {"key": "b", "value": 2},
            {"key": "a", "value": 1},
        ]
        baseline = integrity_receipts.canonical_multiset_sha256(rows)
        self.assertEqual(
            baseline,
            integrity_receipts.canonical_multiset_sha256(
                list(reversed(rows))
            ),
        )
        self.assertNotEqual(
            baseline,
            integrity_receipts.canonical_multiset_sha256(rows[:2]),
        )
        protocol_rows = [
            {"键": "值", "n": 1},
            {"键": "值", "n": 1},
            {"键": "更长", "n": 22},
        ]

        def independent_protocol_hash(
            values: list[dict[str, object]],
        ) -> str:
            encoded = sorted(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
                for row in values
            )
            framed = bytearray(
                b"step28-v13-canonical-multiset-v1\x00"
            )
            framed.extend(len(encoded).to_bytes(8, "big"))
            for row in encoded:
                framed.extend(len(row).to_bytes(8, "big"))
                framed.extend(row)
            return hashlib.sha256(framed).hexdigest()

        self.assertEqual(
            integrity_receipts.canonical_multiset_sha256([]),
            "155d7d97fb2b60118dbd7fa12a3ea7c17ff8bbfcf0f6d7ccc14891833469cbdc",
        )
        self.assertEqual(
            integrity_receipts.canonical_multiset_sha256(protocol_rows),
            "c69ecc6a615190f283ce166b6a8385b034d5823573337ba940da17f29b5078d3",
        )
        self.assertEqual(
            integrity_receipts.canonical_multiset_sha256(protocol_rows),
            independent_protocol_hash(protocol_rows),
        )

    def test_integrity_receipt_validator_rejects_schema_and_hash_tampering(
        self,
    ) -> None:
        receipt = integrity_receipts._build_receipt(
            self.policy,
            role="candidate_integrity",
            fixed_boolean_gates={
                "complete_pair_universe_exact": True,
                "candidate_schema_exact": True,
                "candidate_keys_unique": True,
                "sampling_lineage_exact": True,
                (
                    "label_and_oracle_fields_"
                    "absent_from_sealer_arguments"
                ): True,
            },
            fixed_counts={
                "split_world_count": 3,
                "split_complete_pair_count": 1134,
                "split_candidate_pair_count": 120,
                "duplicate_candidate_count": 0,
                "foreign_key_mismatch_count": 0,
            },
            input_parent_hashes={
                "candidate_policy_projection_sha256": "0" * 64,
                "candidate_key_commitment_sha256": "1" * 64,
                "observed_seller_parent_sha256": "2" * 64,
                "raw_observed_item_parent_sha256": "3" * 64,
                "complete_pair_parent_sha256": "4" * 64,
                "candidate_parent_sha256": "5" * 64,
                "sampling_audit_parent_sha256": "6" * 64,
                "policy_parent_sha256": "7" * 64,
                "registered_split_scope_sha256": "8" * 64,
            },
            aggregate_content_hashes={
                "complete_pair_multiset_sha256": "9" * 64,
                "candidate_pair_multiset_sha256": "a" * 64,
                "sampling_lineage_multiset_sha256": "b" * 64,
            },
        )
        tampered_count = copy.deepcopy(receipt)
        tampered_count["fixed_counts"]["split_candidate_pair_count"] += 1
        with self.assertRaisesRegex(
            common.ContractError, "self-hash mismatch$"
        ):
            integrity_receipts.validate_aggregate_receipt(
                self.policy,
                role="candidate_integrity",
                receipt=tampered_count,
            )
        extra_nested_key = copy.deepcopy(receipt)
        extra_nested_key["fixed_counts"]["per_world"] = 40
        extra_nested_key["canonical_self_hash"] = common.canonical_sha256(
            {
                key: value
                for key, value in extra_nested_key.items()
                if key != "canonical_self_hash"
            }
        )
        with self.assertRaisesRegex(
            common.ContractError, "fixed_counts schema drift$"
        ):
            integrity_receipts.validate_aggregate_receipt(
                self.policy,
                role="candidate_integrity",
                receipt=extra_nested_key,
            )

    def test_integrity_sealer_interfaces_do_not_accept_the_full_payload(
        self,
    ) -> None:
        candidate_parameters = set(
            inspect.signature(
                integrity_receipts.build_candidate_integrity_receipt
            ).parameters
        )
        self.assertEqual(
            candidate_parameters,
            {
                "candidate_context",
                "candidate_policy",
                "candidate_key_hex",
                "mode",
                "split",
                "worlds",
                "sellers",
                "raw_observed_items",
                "complete_pair_endpoints",
                "candidate_pairs",
                "candidate_sampling_audit",
            },
        )
        render_parameters = set(
            inspect.signature(
                integrity_receipts.build_render_integrity_receipt
            ).parameters
        )
        self.assertEqual(
            render_parameters,
            {
                "policy",
                "mode",
                "split",
                "template",
                "worlds",
                "sellers",
                "items",
                "redacted_items",
                "parsed_identity_occurrences",
                "identity_slots_audit",
                "noise_slots_audit",
                "render_asts",
                "override_audit",
            },
        )
        dgp_parameters = set(
            inspect.signature(
                integrity_receipts.build_independent_dgp_comparison_receipt
            ).parameters
        )
        self.assertEqual(
            dgp_parameters,
            {
                "policy",
                "mode",
                "split",
                "worlds",
                "per_world_comparison_receipts",
                "producer_typed_dgp_projections",
                "independent_replay_ledgers",
            },
        )

    def test_combined_generator_rejects_an_alternate_policy_path(
        self,
    ) -> None:
        with mock.patch.object(
            sys,
            "argv",
            [
                "step28_v13_generate_dataset.py",
                "--policy",
                str(ROOT / "schema" / "alternate_policy.json"),
                "--validate-config-only",
            ],
        ), self.assertRaisesRegex(
            common.ContractError,
            "exact default policy file",
        ):
            dataset_generator.main()

    def test_candidate_sealer_policy_projection_contains_no_structure_secret(
        self,
    ) -> None:
        projection = self.candidate_policy
        context = self.candidate_integrity_context
        projection_bytes = common.canonical_json_bytes(projection)
        context_bytes = common.canonical_json_bytes(context)
        secret_values = {
            str(self.policy["randomness"][self.mode][name])
            for name in (
                "structure_key_hex",
                "id_namespace_key_hex",
                "id_key_hex",
                "identity_value_key_hex",
                "text_key_hex",
                "query_key_hex",
            )
        }
        self.assertTrue(
            all(
                value.encode("ascii") not in projection_bytes
                and value.encode("ascii") not in context_bytes
                for value in secret_values
            )
        )
        self.assertNotIn("randomness", projection)
        self.assertNotIn("security", projection)
        self.assertNotIn("identity_design", projection)
        self.assertNotIn("policy", set(
            inspect.signature(
                integrity_receipts.build_candidate_integrity_receipt
            ).parameters
        ))
        tampered = copy.deepcopy(projection)
        tampered["randomness"] = {
            "structure_key_hex": self.policy["randomness"][self.mode][
                "structure_key_hex"
            ]
        }
        with self.assertRaisesRegex(
            common.ContractError,
            "projection envelope drift$",
        ):
            candidate_sampling.validate_public_candidate_policy(
                tampered, mode=self.mode, split=self.split
            )
        covert_value = copy.deepcopy(projection)
        covert_value["candidate_design"][
            "history_only_identity_types"
        ][0] = self.policy["randomness"][self.mode][
            "structure_key_hex"
        ]
        with self.assertRaisesRegex(
            common.ContractError,
            "projection schema/secret drift$",
        ):
            candidate_sampling.validate_public_candidate_policy(
                covert_value, mode=self.mode, split=self.split
            )
        with self.assertRaisesRegex(
            common.ContractError,
            "projection schema/secret drift$",
        ):
            integrity_receipts.build_candidate_integrity_context(
                self.policy,
                candidate_policy=covert_value,
                mode=self.mode,
                split=self.split,
            )

    def test_integrity_source_closures_cover_registered_scope_and_candidate_dependencies(
        self,
    ) -> None:
        for role in (
            "render_integrity",
            "candidate_integrity",
            "m1_derangement_integrity",
            "independent_dgp_comparison",
        ):
            members = integrity_receipts._source_closure_members(role)
            self.assertEqual(
                members,
                tuple(
                    sorted(
                        set(members),
                        key=lambda value: value.encode("utf-8"),
                    )
                ),
            )
            self.assertTrue(
                all((ROOT / Path(name)).is_file() for name in members)
            )
            baseline = integrity_receipts._source_closure_sha256(role)
            real_sha256_file = common.sha256_file

            for member in members:
                target = (ROOT / Path(member)).resolve()

                def changed_member(path, *, target=target):
                    path = Path(path).resolve()
                    if path == target:
                        return "f" * 64
                    return real_sha256_file(path)

                with self.subTest(role=role, member=member), mock.patch.object(
                    integrity_receipts.common,
                    "sha256_file",
                    side_effect=changed_member,
                ):
                    self.assertNotEqual(
                        integrity_receipts._source_closure_sha256(
                            role
                        ),
                        baseline,
                    )
        candidate_members = set(
            integrity_receipts._source_closure_members(
                "candidate_integrity"
            )
        )
        self.assertTrue(
            {
                "scripts/step28_v13_structure.py",
                "scripts/step28_v13_candidate_sampling.py",
                "scripts/step28_v13_profiles.py",
                "scripts/step3_build_seller_profiles.py",
                "schema/step3_seller_profile_schema.json",
                "scripts/step4_build_silver_candidates.py",
                "schema/step4_silver_candidate_schema.json",
            }.issubset(candidate_members)
        )
        render_members = set(
            integrity_receipts._source_closure_members(
                "render_integrity"
            )
        )
        self.assertTrue(
            {
                "scripts/step28_v13_text_renderer.py",
                "scripts/step28_history_common.py",
                "scripts/step7_v3_1_source_data.py",
                "scripts/step7_v4_common.py",
            }.issubset(render_members)
        )
        clean_import = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import json,sys;"
                    "from pathlib import Path;"
                    "root=Path.cwd().resolve();"
                    "sys.path.insert(0,str(root/'scripts'));"
                    "import step28_v13_integrity_receipts;"
                    "paths=sorted({"
                    "Path(m.__file__).resolve().relative_to(root).as_posix() "
                    "for m in sys.modules.values() "
                    "if getattr(m,'__file__',None) "
                    "and Path(m.__file__).resolve().is_relative_to(root)"
                    "});"
                    "print(json.dumps(paths))"
                ),
            ],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        clean_loaded = set(json.loads(clean_import.stdout))
        self.assertTrue(clean_loaded.issubset(candidate_members))
        self.assertTrue(
            {
                "scripts/step28_v13_integrity_receipts.py",
                "scripts/step28_v13_candidate_sampling.py",
                "scripts/step28_v13_common.py",
                "scripts/step28_v13_structure.py",
            }.issubset(clean_loaded)
        )
        self.assertTrue(
            {
                "scripts/step28_v13_feature_derangement.py",
                "scripts/step28_v13_placebo_support.py",
                "scripts/step28_v13_independent_dgp_comparator.py",
                "scripts/step28_v13_production_chain.py",
            }.isdisjoint(clean_loaded)
        )
        with mock.patch.object(
            Path,
            "read_text",
            return_value=(
                "import importlib\n"
                "module_name = 'step28_v13_common'\n"
                "importlib.import_module(module_name)\n"
            ),
        ), self.assertRaisesRegex(
            common.ContractError,
            "nonliteral dynamic imports$",
        ):
            integrity_receipts._repo_local_import_paths(
                "scripts/step28_v13_common.py"
            )
        indirect_sources = (
            "import builtins\n",
            (
                "import importlib as il\n"
                "il.import_module(module_name)\n"
            ),
            (
                "from importlib import import_module as im\n"
                "im(module_name)\n"
            ),
            (
                "import importlib\n"
                "getattr(importlib, 'import_module')(module_name)\n"
            ),
            (
                "import importlib\n"
                "loader = importlib.import_module\n"
                "loader(module_name)\n"
            ),
            (
                "import importlib\n"
                "vars(importlib)['import_module'](module_name)\n"
            ),
            "loader.import_module(module_name)\n",
        )
        for source_text in indirect_sources:
            with self.subTest(
                indirect_dynamic_source=source_text
            ), mock.patch.object(
                Path,
                "read_text",
                return_value=source_text,
            ), self.assertRaisesRegex(
                common.ContractError,
                "(aliased|indirect) dynamic imports$",
            ):
                integrity_receipts._repo_local_import_paths(
                    "scripts/step28_v13_common.py"
                )
        reflection_sources = (
            (
                "globals()['__import__']("
                "'step28_v13_common')\n"
            ),
            (
                "__builtins__['__import__']("
                "'step28_v13_common')\n"
            ),
            (
                "compile(\"__import__('step28_v13_common')\", "
                "'<source-closure>', 'exec')\n"
            ),
            "eval(\"__import__('step28_v13_common')\")\n",
            "exec(\"import step28_v13_common\")\n",
            (
                "locals()['__import__']("
                "'step28_v13_common')\n"
            ),
        )
        for source_text in reflection_sources:
            with self.subTest(
                direct_reflection_source=source_text
            ), mock.patch.object(
                Path,
                "read_text",
                return_value=source_text,
            ), self.assertRaisesRegex(
                common.ContractError,
                "direct reflection primitives$",
            ):
                integrity_receipts._repo_local_import_paths(
                    "scripts/step28_v13_common.py"
                )
        with mock.patch.object(
            Path,
            "read_text",
            return_value=(
                "import importlib\n"
                "importlib.import_module("
                "'.hidden', package=__package__)\n"
            ),
        ), self.assertRaisesRegex(
            common.ContractError,
            "relative dynamic imports$",
        ):
            integrity_receipts._repo_local_import_paths(
                "scripts/step28_v13_common.py"
            )
        with mock.patch.object(
            Path,
            "read_text",
            return_value="from . import hidden_dependency\n",
        ), self.assertRaisesRegex(
            common.ContractError,
            "relative imports$",
        ):
            integrity_receipts._repo_local_import_paths(
                "scripts/step28_v13_common.py"
            )

    def test_all_aggregate_receipt_roles_enforce_recursive_scalar_schema(
        self,
    ) -> None:
        deployment = common.load_json(
            ROOT
            / "schema"
            / "step28_v13_dataset_custody_deployment.json"
        )
        schemas = deployment[
            "fixed_nonjoinable_aggregate_receipt_schemas"
        ]
        roles = (
            "render_integrity",
            "candidate_integrity",
            "m1_derangement_integrity",
            "independent_dgp_comparison",
            "structural_audit",
        )

        def rehash(receipt: dict[str, object]) -> None:
            receipt["canonical_self_hash"] = common.canonical_sha256(
                {
                    key: value
                    for key, value in receipt.items()
                    if key != "canonical_self_hash"
                }
            )

        for role_index, role in enumerate(roles):
            with self.subTest(role=role):
                spec = schemas[role]
                receipt = integrity_receipts._build_receipt(
                    self.policy,
                    role=role,
                    fixed_boolean_gates={
                        key: True
                        for key in spec[
                            "fixed_boolean_gates_exact_keys"
                        ]
                    },
                    fixed_counts={
                        key: 0
                        for key in spec["fixed_counts_exact_keys"]
                    },
                    input_parent_hashes={
                        key: f"{role_index + 1:x}" * 64
                        for key in spec[
                            "input_parent_hashes_exact_keys"
                        ]
                    },
                    aggregate_content_hashes={
                        key: f"{role_index + 6:x}" * 64
                        for key in spec[
                            "aggregate_content_hashes_exact_keys"
                        ]
                    },
                )
                wrong_version = copy.deepcopy(receipt)
                wrong_version["version"] += "-tampered"
                rehash(wrong_version)
                with self.assertRaisesRegex(
                    common.ContractError,
                    "version/evidence drift$",
                ):
                    integrity_receipts.validate_aggregate_receipt(
                        self.policy, role=role, receipt=wrong_version
                    )
                wrong_gate_type = copy.deepcopy(receipt)
                gate = next(iter(wrong_gate_type["fixed_boolean_gates"]))
                wrong_gate_type["fixed_boolean_gates"][gate] = 1
                rehash(wrong_gate_type)
                with self.assertRaisesRegex(
                    common.ContractError, "gate type drift$"
                ):
                    integrity_receipts.validate_aggregate_receipt(
                        self.policy, role=role, receipt=wrong_gate_type
                    )
                wrong_count_type = copy.deepcopy(receipt)
                count = next(iter(wrong_count_type["fixed_counts"]))
                wrong_count_type["fixed_counts"][count] = True
                rehash(wrong_count_type)
                with self.assertRaisesRegex(
                    common.ContractError, "count type drift$"
                ):
                    integrity_receipts.validate_aggregate_receipt(
                        self.policy, role=role, receipt=wrong_count_type
                    )
                nested_hash = copy.deepcopy(receipt)
                hash_name = next(
                    iter(nested_hash["aggregate_content_hashes"])
                )
                nested_hash["aggregate_content_hashes"][hash_name] = [
                    "0" * 64
                ]
                rehash(nested_hash)
                with self.assertRaisesRegex(
                    common.ContractError, "contains a nested object$"
                ):
                    integrity_receipts.validate_aggregate_receipt(
                        self.policy, role=role, receipt=nested_hash
                    )
                uppercase_hash = copy.deepcopy(receipt)
                parent_name = next(
                    iter(uppercase_hash["input_parent_hashes"])
                )
                uppercase_hash["input_parent_hashes"][
                    parent_name
                ] = "A" * 64
                rehash(uppercase_hash)
                with self.assertRaisesRegex(
                    common.ContractError,
                    "not a lowercase SHA-256$",
                ):
                    integrity_receipts.validate_aggregate_receipt(
                        self.policy, role=role, receipt=uppercase_hash
                    )

    def test_placebo_rewire_is_exact_deterministic_and_label_free(self) -> None:
        train_record = next(
            row
            for row in structure.build_mode_world_pool(
                self.policy, mode=self.mode
            )
            if row["split"] == "train"
        )
        train_world = world_builder.build_world(
            policy=self.policy,
            template=self.template,
            fixture=self.fixture,
            style_profile=self.style,
            mode=self.mode,
            world_record=train_record,
            structure_key_hex=common.structure_key_for_split(
                self.policy,
                mode=self.mode,
                split="train",
            ),
        )
        train_processed = production.process_world(
            self.policy,
            mode=self.mode,
            split="train",
            template=self.template,
            world=train_world,
        )
        safe_rows, ledger, _audit = safe_slots.project_safe_slots(
            self.policy,
            mode=self.mode,
            split="train",
            sellers=train_world["public"]["sellers"],
            items=train_world["public"]["items"],
            parsed_rows=train_processed["private"][
                "parsed_identity_occurrences"
            ],
            identity_slots_edit=train_world["private"][
                "identity_slots_edit"
            ],
        )
        seed_hex = self.policy["randomness"][self.mode][
            "rewire_key_hexes"
        ][0]
        result = placebo_rewire.build_one_placebo(
            self.policy,
            mode=self.mode,
            split="train",
            seed_hex=seed_hex,
            sellers=train_world["public"]["sellers"],
            items=train_world["public"]["items"],
            safe_slots=safe_rows,
            nuisance_ledger=ledger,
            render_asts=train_world["private"]["render_asts"],
        )
        self.assertEqual(
            result["rewire_seed_id"],
            "rws_" + hashlib.sha256(bytes.fromhex(seed_hex)).hexdigest(),
        )
        self.assertEqual(
            len(result["rewired_slot_assignments"]), len(safe_rows)
        )
        self.assertEqual(
            len(result["rewired_parsed_identity_occurrences"]),
            len(safe_rows),
        )
        self.assertEqual(
            len(result["rewired_history_safe_occurrences"]),
            len(safe_rows),
        )
        self.assertIs(result["label_or_controller_inputs_read"], False)
        self.assertTrue(
            all(
                row["accepted_swap_count"]
                >= 20 * row["original_edge_count"]
                for row in result["rewire_stratum_audit"]
                if row["structurally_fixed_bool"] == "false"
            )
        )
        self.assertTrue(
            all(
                row["original_edge_retention_count"] == 0
                for row in result["rewire_stratum_audit"]
                if row["nuisance_class"] == "direct_or_private"
            )
        )
        original_items = {
            row["item_uid"]: row for row in train_world["public"]["items"]
        }
        rewired_items = {
            row["item_uid"]: row for row in result["rewired_items"]
        }
        self.assertEqual(set(original_items), set(rewired_items))
        self.assertTrue(
            all(
                original_items[item_uid]["title"]
                == rewired_items[item_uid]["title"]
                for item_uid in original_items
            )
        )
        original_identity_degree = Counter(
            row["identity_uid"] for row in safe_rows
        )
        rewired_identity_degree = Counter(
            row["rewired_identity_uid"]
            for row in result["rewired_slot_assignments"]
        )
        self.assertEqual(original_identity_degree, rewired_identity_degree)
        self.assertNotEqual(
            [
                (
                    row["seller_uid"],
                    row["identity_uid"],
                    row["slot_uid"],
                )
                for row in safe_rows
            ],
            [
                (
                    row["seller_uid"],
                    row["rewired_identity_uid"],
                    row["slot_uid"],
                )
                for row in result["rewired_slot_assignments"]
            ],
        )
        replay = placebo_rewire.build_one_placebo(
            self.policy,
            mode=self.mode,
            split="train",
            seed_hex=seed_hex,
            sellers=list(reversed(train_world["public"]["sellers"])),
            items=list(reversed(train_world["public"]["items"])),
            safe_slots=list(reversed(safe_rows)),
            nuisance_ledger=list(reversed(ledger)),
            render_asts=list(reversed(train_world["private"]["render_asts"])),
        )
        self.assertEqual(
            replay["canonical_self_hash"], result["canonical_self_hash"]
        )
        tampered_ledger = copy.deepcopy(ledger)
        tampered_ledger[0]["nuisance_class"] = "risky_product"
        with self.assertRaisesRegex(
            common.ContractError,
            "nuisance class recomputation drift",
        ):
            placebo_rewire.build_one_placebo(
                self.policy,
                mode=self.mode,
                split="train",
                seed_hex=seed_hex,
                sellers=train_world["public"]["sellers"],
                items=train_world["public"]["items"],
                safe_slots=safe_rows,
                nuisance_ledger=tampered_ledger,
                render_asts=train_world["private"]["render_asts"],
            )
        duplicated_ast_slot = copy.deepcopy(
            train_world["private"]["render_asts"]
        )
        source_ast = next(
            row for row in duplicated_ast_slot if row["identity_slot_uids"]
        )
        duplicated_slot_uid = source_ast["identity_slot_uids"][0]
        target_ast = next(
            row
            for row in duplicated_ast_slot
            if row["item_uid"] != source_ast["item_uid"]
            and duplicated_slot_uid not in row["identity_slot_uids"]
        )
        target_ast["identity_slot_uids"] = common.utf8_sort(
            [*target_ast["identity_slot_uids"], duplicated_slot_uid]
        )
        with self.assertRaisesRegex(
            common.ContractError,
            "Rewire AST lineage drift",
        ):
            placebo_rewire.build_one_placebo(
                self.policy,
                mode=self.mode,
                split="train",
                seed_hex=seed_hex,
                sellers=train_world["public"]["sellers"],
                items=train_world["public"]["items"],
                safe_slots=safe_rows,
                nuisance_ledger=ledger,
                render_asts=duplicated_ast_slot,
            )
        with self.assertRaisesRegex(
            common.ContractError,
            "development-smoke train only",
        ):
            placebo_rewire.build_one_placebo(
                self.policy,
                mode="formal",
                split="train",
                seed_hex=seed_hex,
                sellers=train_world["public"]["sellers"],
                items=train_world["public"]["items"],
                safe_slots=safe_rows,
                nuisance_ledger=ledger,
                render_asts=train_world["private"]["render_asts"],
            )

    def test_nuisance_class_uses_identity_level_priority(self) -> None:
        direct = {
            "observed_seller_facing_context": 1,
            "observed_product_data_risk_context": 0,
            "observed_direct_identity_eligible": 1,
            "observed_support_only": 0,
        }
        support = {
            "observed_seller_facing_context": 1,
            "observed_product_data_risk_context": 0,
            "observed_direct_identity_eligible": 0,
            "observed_support_only": 1,
        }
        risky = {
            "observed_seller_facing_context": 0,
            "observed_product_data_risk_context": 1,
            "observed_direct_identity_eligible": 0,
            "observed_support_only": 0,
        }
        self.assertEqual(
            placebo_rewire._aggregate_nuisance(
                [direct, support], seller_degree=2, direct_maximum=3
            ),
            "public_support",
        )
        self.assertEqual(
            placebo_rewire._aggregate_nuisance(
                [direct, support, risky],
                seller_degree=5,
                direct_maximum=3,
            ),
            "risky_product",
        )

    def test_independent_typed_dgp_replay_is_exact_and_import_isolated(
        self,
    ) -> None:
        audit = self.independent_replay_audit
        self.assertIs(audit["independent_typed_dgp_replay_pass"], True)
        self.assertIs(audit["independent_decision_implementation"], True)
        self.assertIs(audit["formal_custody_seal"], False)
        self.assertIs(
            audit["producer_private_input_used_by_replayer"],
            False,
        )
        self.assertEqual(
            audit["producer_typed_projection_sha256"],
            audit["replayer_typed_projection_sha256"],
        )
        self.assertEqual(
            audit["evidence_level"],
            "INDEPENDENT_TYPED_DGP_REPLAY_DEVELOPMENT_INTEGRATION_"
            "NOT_FORMAL_CUSTODY_SEAL",
        )
        replay_path = (
            ROOT
            / "scripts"
            / "step28_v13_independent_private_dgp_replay.py"
        )
        tree = ast.parse(replay_path.read_text(encoding="utf-8"))
        imported_names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_names.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported_names.add(node.module)
        self.assertFalse(
            {
                name
                for name in imported_names
                if name.startswith("step28_v13_")
            }
        )
        launcher_text = (
            ROOT
            / "scripts"
            / "step28_v13_run_independent_dgp_replay.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--key-hex", launcher_text)
        launcher_tree = ast.parse(launcher_text)
        launcher_step28_imports: set[str] = set()
        for node in ast.walk(launcher_tree):
            if isinstance(node, ast.Import):
                launcher_step28_imports.update(
                    alias.name
                    for alias in node.names
                    if alias.name.startswith("step28_v13_")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("step28_v13_")
            ):
                launcher_step28_imports.add(node.module)
        self.assertEqual(
            launcher_step28_imports,
            {"step28_v13_independent_private_dgp_replay"},
        )
        comparator_path = (
            ROOT
            / "scripts"
            / "step28_v13_independent_dgp_comparator.py"
        )
        comparator_tree = ast.parse(
            comparator_path.read_text(encoding="utf-8")
        )
        self.assertFalse(
            {
                alias.name
                for node in ast.walk(comparator_tree)
                if isinstance(node, ast.Import)
                for alias in node.names
                if alias.name.startswith("step28_v13_")
            }
        )
        comparator_launcher_text = (
            ROOT
            / "scripts"
            / "step28_v13_compare_independent_dgp_replay.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--key-hex", comparator_launcher_text)
        self.assertNotIn("--structure-key", comparator_launcher_text)
        key_environment = self.policy["randomness"]["formal"][
            "label_bearing_structure_keys"
        ]["audit_a"]["environment_variable"]
        with mock.patch.dict(
            "os.environ",
            {key_environment: "0" * 64},
            clear=False,
        ):
            with self.assertRaisesRegex(
                replay_compare_launcher.ComparatorLauncherError,
                (
                    "^DGP_COMPARATOR_REGISTERED_"
                    "STRUCTURE_KEY_ENV_PRESENT$"
                ),
            ):
                replay_compare_launcher._reject_registered_key_environment(
                    self.policy
                )
        public_domains = self.policy["independent_replay_public_domains"]
        self.assertEqual(
            self.policy["template_library"]["style_prototype_ids"],
            [
                row["style_id"]
                for row in self.template["style_prototypes"]
            ],
        )
        self.assertEqual(
            public_domains["categories_in_registered_order"],
            self.template["generic_lexicon"]["categories"],
        )
        self.assertEqual(
            public_domains["category_products"],
            self.template["generic_lexicon"]["category_products"],
        )
        self.assertEqual(
            public_domains["attributes"],
            self.template["generic_lexicon"]["attributes"],
        )
        self.assertEqual(
            public_domains["anonymous_category_rank_probability"],
            self.style["anonymous_category_rank_probability"],
        )
        deployment = common.load_json(
            ROOT
            / "schema"
            / "step28_v13_dataset_custody_deployment.json"
        )
        self.assertEqual(deployment["status"], "DRAFT_NOT_DEPLOYABLE")
        self.assertIs(
            deployment["formal_readiness"][
                "ready_for_dataset_release_lock"
            ],
            False,
        )
        self.assertNotIn(
            "parser_redactor_and_structural_audit",
            deployment["capabilities"],
        )
        self.assertTrue(
            {
                "parser_worker",
                "redactor_worker",
                "structural_auditor",
                "split_private_producer_dgp_projection_sealer",
            }.issubset(deployment["capabilities"])
        )
        self.assertIsNone(
            deployment["capabilities"][
                "split_private_producer_dgp_projection_sealer"
            ]["formal_entrypoint"]
        )
        self.assertIs(
            deployment["required_runtime_controls"][
                "self_hash_without_external_custody_parent_is_formal_evidence"
            ],
            False,
        )
        parent_schema = deployment[
            "structural_auditor_parent_seal_projection_schema"
        ]
        self.assertEqual(
            parent_schema["exact_record_count_by_split"],
            {
                "train": 4,
                "development": 3,
                "audit_a": 3,
                "audit_b": 3,
            },
        )
        self.assertNotIn(
            "m1_derangement_integrity",
            parent_schema["role_exact_values_by_split"]["development"],
        )
        self.assertIn(
            "m1_derangement_integrity",
            parent_schema["role_exact_values_by_split"]["train"],
        )
        self.assertIs(self.policy["formal_generation_enabled"], False)
        self.assertEqual(
            self.record["world_uid"],
            "w_003497845547650a980473b05e249937bf825ad0eaefa424baec74f2bd2210f3",
        )
        self.assertEqual(
            self.independent_expected["typed_replay_sha256"],
            "b4d2cd0d369bf49a217942cf8b0a4f965f85c124cbbe39d6c11a876443688710",
        )

    def test_independent_typed_dgp_replay_rejects_decision_tampering(
        self,
    ) -> None:
        cases: list[tuple[str, dict[str, object], str]] = []

        membership = copy.deepcopy(self.world)
        membership_rows = membership["private"]["controller_membership"]
        membership_rows[0]["controller_uid"], membership_rows[-1][
            "controller_uid"
        ] = (
            membership_rows[-1]["controller_uid"],
            membership_rows[0]["controller_uid"],
        )
        cases.append(
            (
                "membership",
                membership,
                "REPLAY_CONTROLLER_MEMBERSHIP_MISMATCH",
            )
        )

        market = copy.deepcopy(self.world)
        market_row = market["public"]["sellers"][0]
        market_row["market"] = next(
            value
            for value in self.policy["world_design"]["markets"]
            if value != market_row["market"]
        )
        cases.append(
            ("market", market, "REPLAY_MARKET_ASSIGNMENT_MISMATCH")
        )

        style = copy.deepcopy(self.world)
        style_rows = style["private"]["controller_style_groups"]
        left_index = next(
            index
            for index, row in enumerate(style_rows)
            if row["style_id"] != style_rows[0]["style_id"]
        )
        style_rows[0]["style_id"], style_rows[left_index]["style_id"] = (
            style_rows[left_index]["style_id"],
            style_rows[0]["style_id"],
        )
        cases.append(
            (
                "style",
                style,
                "REPLAY_CONTROLLER_STYLE_GROUP_MISMATCH",
            )
        )

        mechanism = copy.deepcopy(self.world)
        mechanism["private"]["mechanism_assignments"][0][
            "mechanism_slot_uid"
        ] += "_tampered"
        cases.append(
            (
                "mechanism",
                mechanism,
                "REPLAY_MECHANISM_ASSIGNMENT_MISMATCH",
            )
        )

        target = copy.deepcopy(self.world)
        target["private"]["positive_targets"][0][
            "canonical_pair_uid"
        ] += "_tampered"
        cases.append(
            ("target", target, "REPLAY_POSITIVE_TARGET_MISMATCH")
        )

        override = copy.deepcopy(self.world)
        override_row = next(
            row
            for row in override["private"]["override_audit"]
            if row["override_kind"] == "exact_title_clone"
        )
        override_row["item_uid_left"] = next(
            row["item_uid"]
            for row in override["public"]["items"]
            if row["seller_uid"] == override_row["seller_uid_left"]
            and row["title"]
            and row["item_uid"] != override_row["item_uid_left"]
        )
        cases.append(
            (
                "override_item",
                override,
                "REPLAY_REGISTERED_OVERRIDE_DECISION_MISMATCH",
            )
        )

        identity_type = copy.deepcopy(self.world)
        variable_asset = next(
            row
            for row in identity_type["private"]["identity_assets"]
            if row["fixed_type"] is None
            and len(row["allowed_types"]) > 1
        )
        variable_asset["identity_type"] = next(
            value
            for value in variable_asset["allowed_types"]
            if value != variable_asset["identity_type"]
        )
        cases.append(
            (
                "identity_type",
                identity_type,
                "REPLAY_IDENTITY_ASSET_DECISION_MISMATCH",
            )
        )

        repeat = copy.deepcopy(self.world)
        repeated_asset = next(
            row
            for row in repeat["private"]["identity_assets"]
            if row["repeat_draw_name"] is not None
        )
        repeated_asset["asset_repeat_decision"] = not repeated_asset[
            "asset_repeat_decision"
        ]
        cases.append(
            ("repeat", repeat, "REPLAY_REPEAT_DECISION_MISMATCH")
        )

        proposal = copy.deepcopy(self.world)
        proposal["private"]["solver_audit"][
            "market_proposal_counter"
        ] += 1
        cases.append(
            (
                "proposal_counter",
                proposal,
                "REPLAY_MARKET_PROPOSAL_COUNTER_MISMATCH",
            )
        )

        for label, mutated, error_code in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(
                    independent_comparator.IndependentDgpComparisonError,
                    f"^{error_code}$",
                ):
                    independent_comparator.compare_typed_dgp(
                        expected_replay=self.independent_expected,
                        producer_projection=(
                            producer_projection.project_world(
                                world=mutated,
                                mode=self.mode,
                                split=self.split,
                            )
                        ),
                    )

    def test_replay_envelopes_and_projection_schemas_fail_closed(
        self,
    ) -> None:
        altered_envelope = copy.deepcopy(self.independent_expected)
        altered_envelope["observed_uid_pool_audit"][
            "all_item_count"
        ] += 1
        with self.assertRaisesRegex(
            independent_comparator.IndependentDgpComparisonError,
            "^REPLAY_EXPECTED_LEDGER_SELF_HASH_MISMATCH$",
        ):
            independent_comparator.compare_typed_dgp(
                expected_replay=altered_envelope,
                producer_projection=self.producer_projection,
            )

        altered_identity = copy.deepcopy(self.independent_expected)
        altered_identity["scope"] = "forged_scope"
        altered_identity["canonical_self_hash"] = common.canonical_sha256(
            {
                key: value
                for key, value in altered_identity.items()
                if key != "canonical_self_hash"
            }
        )
        with self.assertRaisesRegex(
            independent_comparator.IndependentDgpComparisonError,
            "^REPLAY_EXPECTED_LEDGER_IDENTITY_INVALID$",
        ):
            independent_comparator.compare_typed_dgp(
                expected_replay=altered_identity,
                producer_projection=self.producer_projection,
            )

        extra_field = copy.deepcopy(self.producer_projection)
        extra_field["tables"]["controller_membership"][0][
            "forbidden_extra"
        ] = "must_fail"
        extra_field["typed_projection_sha256"] = common.canonical_sha256(
            extra_field["tables"]
        )
        extra_field["canonical_self_hash"] = common.canonical_sha256(
            {
                key: value
                for key, value in extra_field.items()
                if key != "canonical_self_hash"
            }
        )
        with self.assertRaisesRegex(
            independent_comparator.IndependentDgpComparisonError,
            (
                "^REPLAY_PRODUCER_CONTROLLER_MEMBERSHIP_"
                "SCHEMA_INVALID$"
            ),
        ):
            independent_comparator.compare_typed_dgp(
                expected_replay=self.independent_expected,
                producer_projection=extra_field,
            )

    def test_json_readers_reject_duplicate_object_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate_json = root / "duplicate.json"
            duplicate_json.write_text(
                '{"version":"a","version":"b"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_launcher.ReplayLauncherError,
                "^REPLAY_LAUNCHER_DUPLICATE_JSON_KEY:version$",
            ):
                replay_launcher._read_json(duplicate_json)
            duplicate_jsonl = root / "duplicate.jsonl"
            duplicate_jsonl.write_text(
                '{"world_uid":"a","world_uid":"b"}\n',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                replay_compare_launcher.ComparatorLauncherError,
                "^DGP_COMPARATOR_DUPLICATE_JSON_KEY:world_uid$",
            ):
                replay_compare_launcher._read_jsonl(duplicate_jsonl)

    def test_independent_replay_rejects_formal_before_release_freeze(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            common.ContractError,
            "development-smoke only",
        ):
            smoke_regeneration.validate_producer_regeneration_match(
                self.policy,
                mode="formal",
                split=self.split,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                world=self.world,
            )
        with self.assertRaisesRegex(
            independent_replay.IndependentReplayError,
            "^REPLAY_FORMAL_RELEASE_NOT_FROZEN$",
        ):
            independent_replay.replay_typed_dgp(
                self.policy,
                mode="formal",
                split=self.split,
                world_uid=self.record["world_uid"],
                structure_key_hex=self.structure_key,
                **self.observed_uid_pools,
            )

    def test_independent_replay_rejects_world_from_another_split(
        self,
    ) -> None:
        foreign_record = next(
            row
            for row in structure.build_mode_world_pool(
                self.policy,
                mode=self.mode,
            )
            if row["split"] != self.split
        )
        with self.assertRaisesRegex(
            independent_replay.IndependentReplayError,
            "^REPLAY_WORLD_UID_NOT_REGISTERED_FOR_SPLIT$",
        ):
            independent_replay.replay_typed_dgp(
                self.policy,
                mode=self.mode,
                split=self.split,
                world_uid=foreign_record["world_uid"],
                structure_key_hex=self.structure_key,
                **self.observed_uid_pools,
            )

    def test_development_entrypoints_reject_formal_before_private_inputs(
        self,
    ) -> None:
        inaccessible = str(ROOT / "definitely_missing_private_input")
        replay_args = SimpleNamespace(
            mode="formal",
            policy=inaccessible,
            split=self.split,
            world_pool=inaccessible,
            seller_pool=inaccessible,
            all_item_pool=inaccessible,
            nonempty_title_pool=inaccessible,
            nonempty_description_pool=inaccessible,
            output_root=inaccessible,
            validate_config_only=False,
        )
        with self.assertRaisesRegex(
            replay_launcher.ReplayLauncherError,
            "^REPLAY_LAUNCHER_FORMAL_CAPABILITY_NOT_IMPLEMENTED$",
        ):
            replay_launcher.run(replay_args)
        comparator_args = SimpleNamespace(
            mode="formal",
            policy=inaccessible,
            split=self.split,
            replay_ledgers=inaccessible,
            replay_receipt=inaccessible,
            producer_projections=inaccessible,
            producer_manifest=inaccessible,
            output_root=inaccessible,
            validate_config_only=False,
        )
        with self.assertRaisesRegex(
            replay_compare_launcher.ComparatorLauncherError,
            "^DGP_COMPARATOR_FORMAL_CAPABILITY_NOT_IMPLEMENTED$",
        ):
            replay_compare_launcher.run(comparator_args)
        with self.assertRaisesRegex(
            common.ContractError,
            "combined generator is development-smoke only",
        ):
            dataset_generator.build_split_payload(
                self.policy,
                mode="formal",
                split=self.split,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
            )
        with self.assertRaisesRegex(
            common.ContractError,
            "combined writer is development-smoke only",
        ):
            dataset_generator.write_split_release(
                self.policy,
                mode="formal",
                split=self.split,
                release_name="must_not_write",
                result={},
            )

    def test_validate_config_only_does_not_open_private_inputs(self) -> None:
        policy_path = (
            ROOT
            / "schema"
            / "step28_v13_synthetic_chinese_dataset_policy.json"
        )
        missing = str(ROOT / "must_not_be_opened_private_input")
        replay_result = replay_launcher.run(
            SimpleNamespace(
                mode=self.mode,
                policy=str(policy_path),
                split=self.split,
                world_pool=missing,
                seller_pool=missing,
                all_item_pool=missing,
                nonempty_title_pool=missing,
                nonempty_description_pool=missing,
                output_root=None,
                validate_config_only=True,
            )
        )
        self.assertIs(replay_result["input_data_opened"], False)
        self.assertIs(replay_result["structure_key_loaded"], False)
        with mock.patch.dict("os.environ", {}, clear=True):
            comparator_result = replay_compare_launcher.run(
                SimpleNamespace(
                    mode=self.mode,
                    policy=str(policy_path),
                    split=self.split,
                    replay_ledgers=missing,
                    replay_receipt=missing,
                    producer_projections=missing,
                    producer_manifest=missing,
                    output_root=None,
                    validate_config_only=True,
                )
            )
        self.assertIs(comparator_result["private_inputs_opened"], False)

    def test_validate_config_only_cli_needs_no_private_path_flags(
        self,
    ) -> None:
        policy_path = (
            ROOT
            / "schema"
            / "step28_v13_synthetic_chinese_dataset_policy.json"
        )
        commands = (
            (
                ROOT
                / "scripts"
                / "step28_v13_run_independent_dgp_replay.py",
                "independent DGP replayer static development "
                "configuration PASS",
            ),
            (
                ROOT
                / "scripts"
                / "step28_v13_compare_independent_dgp_replay.py",
                "DGP comparator static development configuration PASS",
            ),
        )
        for script, expected_output in commands:
            completed = subprocess.run(
                (
                    sys.executable,
                    str(script),
                    "--policy",
                    str(policy_path),
                    "--mode",
                    self.mode,
                    "--split",
                    self.split,
                    "--validate-config-only",
                ),
                cwd=ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                completed.returncode,
                0,
                completed.stderr,
            )
            self.assertIn(expected_output, completed.stdout)

    def test_replay_launcher_rejects_incomplete_split_world_set(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world_path = root / "worlds.csv"
            common.write_csv(
                world_path,
                [{"world_uid": self.record["world_uid"]}],
                ("world_uid",),
            )
            paths: dict[str, Path] = {}
            for name, fields in (
                ("seller", ("world_uid", "seller_uid")),
                (
                    "all_item",
                    ("world_uid", "seller_uid", "item_uid"),
                ),
                (
                    "title",
                    ("world_uid", "seller_uid", "item_uid"),
                ),
                (
                    "description",
                    ("world_uid", "seller_uid", "item_uid"),
                ),
            ):
                path = root / f"{name}.csv"
                common.write_csv(path, [], fields)
                paths[name] = path
            with self.assertRaisesRegex(
                replay_launcher.ReplayLauncherError,
                "^REPLAY_LAUNCHER_COMPLETE_WORLD_SET_MISMATCH$",
            ):
                replay_launcher.run(
                    SimpleNamespace(
                        mode=self.mode,
                        policy=str(
                            ROOT
                            / "schema"
                            / (
                                "step28_v13_synthetic_chinese_"
                                "dataset_policy.json"
                            )
                        ),
                        split=self.split,
                        world_pool=str(world_path),
                        seller_pool=str(paths["seller"]),
                        all_item_pool=str(paths["all_item"]),
                        nonempty_title_pool=str(paths["title"]),
                        nonempty_description_pool=str(
                            paths["description"]
                        ),
                        output_root=str(root / "must_not_exist"),
                        validate_config_only=False,
                    )
                )

    def test_atomic_writer_supports_extended_windows_target_and_short_temp_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            parent = Path(temporary)
            target_name = (
                "independent_replay_registered_seller_uid_pool.csv"
            )
            ordinal = 0
            while len(str(parent / target_name)) < 275:
                parent /= f"d{ordinal:07d}"
                ordinal += 1
            parent.mkdir(parents=True)
            target = parent / target_name
            self.assertGreater(len(str(target)), 260)
            with mock.patch.object(
                common.tempfile,
                "mkstemp",
                wraps=tempfile.mkstemp,
            ) as mkstemp:
                common.write_csv(
                    target,
                    [{"seller_uid": "sel_test"}],
                    ("seller_uid",),
                )
            self.assertEqual(
                Path(common.filesystem_path(target)).read_text(
                    encoding="utf-8"
                ),
                "seller_uid\nsel_test\n",
            )
            common.write_csv(
                target,
                [{"seller_uid": "sel_test"}],
                ("seller_uid",),
            )
            with self.assertRaises(FileExistsError):
                common.write_csv(
                    target,
                    [{"seller_uid": "sel_other"}],
                    ("seller_uid",),
                )
            _args, kwargs = mkstemp.call_args
            self.assertEqual(kwargs["prefix"], ".tmp-")
            self.assertEqual(kwargs["suffix"], ".part")
            self.assertEqual(
                kwargs["dir"],
                common.filesystem_path(target.parent),
            )
            os.unlink(common.filesystem_path(target))

    def test_atomic_no_replace_rejects_check_then_create_race(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            target = root / "race_target.json"
            real_atomic_rename = common.atomic_rename_no_replace

            def create_competitor_then_publish(source, destination):
                Path(destination).write_bytes(b"competitor")
                return real_atomic_rename(source, destination)

            with mock.patch.object(
                common,
                "atomic_rename_no_replace",
                side_effect=create_competitor_then_publish,
            ), self.assertRaises(FileExistsError):
                common.write_json(target, {"producer": True})
            self.assertEqual(target.read_bytes(), b"competitor")
            self.assertEqual(
                list(root.glob(".tmp-*.part")),
                [],
            )

    def test_release_tree_fsync_walk_supports_extended_windows_paths(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary) / "long_release_tree"
            parent = root
            target_name = "registered_long_release_artifact.json"
            ordinal = 0
            while len(str(parent)) < 275:
                parent /= f"n{ordinal:07d}"
                ordinal += 1
            target = parent / target_name
            self.assertGreater(len(str(target.parent)), 260)
            common.write_json(target, {"status": "PASS"})
            opened: list[str] = []
            active_target_descriptors: set[int] = set()
            target_fsync_hit = {"value": False}
            real_open = open

            class TrackedTargetHandle:
                def __init__(self, handle):
                    self.handle = handle

                def __enter__(self):
                    self.handle.__enter__()
                    active_target_descriptors.add(
                        self.handle.fileno()
                    )
                    return self

                def __exit__(self, exc_type, exc_value, traceback):
                    active_target_descriptors.discard(
                        self.handle.fileno()
                    )
                    return self.handle.__exit__(
                        exc_type,
                        exc_value,
                        traceback,
                    )

                def fileno(self):
                    return self.handle.fileno()

            def tracked_open(path, *args, **kwargs):
                opened.append(os.fspath(path))
                handle = real_open(path, *args, **kwargs)
                if os.fspath(path) == common.filesystem_path(target):
                    return TrackedTargetHandle(handle)
                return handle

            real_fsync = os.fsync

            def tracked_fsync(descriptor):
                if descriptor in active_target_descriptors:
                    target_fsync_hit["value"] = True
                return real_fsync(descriptor)

            with mock.patch.object(
                dataset_generator,
                "open",
                side_effect=tracked_open,
                create=True,
            ), mock.patch.object(
                dataset_generator.os,
                "fsync",
                side_effect=tracked_fsync,
            ) as fsync:
                dataset_generator._fsync_release_tree(root)
            self.assertIn(common.filesystem_path(target), opened)
            self.assertGreaterEqual(fsync.call_count, 1)
            self.assertTrue(target_fsync_hit["value"])
            self.assertTrue(
                os.path.isfile(common.filesystem_path(target))
            )
            self.assertEqual(
                common.load_json(target),
                {"status": "PASS"},
            )
            shutil.rmtree(common.filesystem_path(root))

    def test_release_stage_cleanup_removes_extended_length_descendant(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            parent = Path(temporary) / "release_parent"
            parent.mkdir()
            stage = parent / ".staging-train-long-cleanup"
            deep = stage
            ordinal = 0
            while len(str(deep)) < 275:
                deep /= f"c{ordinal:07d}"
                ordinal += 1
            target = deep / "partial.private.json"
            common.write_json(target, {"partial": True})
            self.assertTrue(
                os.path.isfile(common.filesystem_path(target))
            )
            dataset_generator._cleanup_release_stage(
                stage,
                parent=parent,
            )
            self.assertFalse(
                os.path.exists(common.filesystem_path(stage))
            )

    def test_fixture_result_is_required_and_must_remain_qualified(
        self,
    ) -> None:
        fixture_path = (
            ROOT / "schema" / "step28_v13_parser_template_fixture.json"
        )
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            missing_fixture = copy.deepcopy(self.fixture)
            missing_path = root / "missing_fixture_result.json"
            missing_fixture["full_render_context_contract"][
                "output_manifest"
            ]["path"] = missing_path.relative_to(ROOT).as_posix()
            with self.assertRaisesRegex(
                FileNotFoundError,
                "^Missing exhaustive parser/template fixture result:",
            ):
                dataset_generator._validate_fixture_result(
                    missing_fixture,
                    fixture_path=fixture_path,
                )

            tampered_fixture = copy.deepcopy(self.fixture)
            tampered_path = root / "tampered_fixture_result.json"
            tampered_result = common.load_json(
                common.repo_path(
                    str(
                        self.fixture["full_render_context_contract"][
                            "output_manifest"
                        ]["path"]
                    )
                )
            )
            tampered_result["status"] = "PASS_WITHOUT_REPLAY"
            common.write_json(tampered_path, tampered_result)
            tampered_fixture["full_render_context_contract"][
                "output_manifest"
            ]["path"] = tampered_path.relative_to(ROOT).as_posix()
            with self.assertRaisesRegex(
                common.ContractError,
                (
                    "^Exhaustive parser/template fixture result is not "
                    "release-qualified$"
                ),
            ):
                dataset_generator._validate_fixture_result(
                    tampered_fixture,
                    fixture_path=fixture_path,
                )

    def test_release_parent_resolve_escape_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            root = Path(temporary)
            release_root = root / "releases"
            release_root.mkdir()
            release_name = self.policy[
                "development_complete_release"
            ]["release_name"]
            parent = release_root / release_name
            parent.mkdir()
            escaped = root / "escaped"
            escaped.mkdir()
            real_resolve = Path.resolve
            escaped_resolved = real_resolve(escaped, strict=True)

            def resolve_with_parent_escape(path, strict=False):
                candidate = Path(path)
                if candidate == parent:
                    return escaped_resolved
                return real_resolve(candidate, strict=strict)

            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), mock.patch.object(
                Path,
                "resolve",
                autospec=True,
                side_effect=resolve_with_parent_escape,
            ), self.assertRaisesRegex(
                common.ContractError,
                (
                    "^Dataset release parent may not be a symlink, "
                    "junction, or escape$"
                ),
            ):
                dataset_generator._validated_release_parent(
                    self.policy,
                    mode=self.mode,
                    release_name=release_name,
                )

    def test_complete_release_manifest_binds_children_and_fixture(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            release_root = Path(temporary) / "releases"
            parent, digests, artifacts = (
                self._materialize_minimal_complete_release_children(
                    release_root,
                    splits=("train", "development", "audit_a"),
                )
            )
            release_name = self.policy[
                "development_complete_release"
            ]["release_name"]
            missing_digests = {
                split: digests.get(split, "0" * 64)
                for split in dataset_generator.SPLITS
            }
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), self.assertRaisesRegex(
                common.ContractError,
                (
                    "^Complete release parent has missing or "
                    "unexpected entries$"
                ),
            ):
                dataset_generator.write_release_manifest(
                    self.policy,
                    mode=self.mode,
                    release_name=release_name,
                    split_payload_digests=missing_digests,
                )

            _parent, audit_b_digest, audit_b_artifact = (
                self._materialize_minimal_complete_release_children(
                    release_root,
                    splits=("audit_b",),
                )
            )
            digests.update(audit_b_digest)
            artifacts.update(audit_b_artifact)
            ordered_digests = {
                split: digests[split]
                for split in dataset_generator.SPLITS
            }
            orphan = parent / "unregistered.debug.json"
            common.write_json(orphan, {"debug": True})
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), self.assertRaisesRegex(
                common.ContractError,
                (
                    "^Complete release parent has missing or "
                    "unexpected entries$"
                ),
            ):
                dataset_generator.write_release_manifest(
                    self.policy,
                    mode=self.mode,
                    release_name=release_name,
                    split_payload_digests=ordered_digests,
                )
            os.unlink(common.filesystem_path(orphan))

            original_train_bytes = artifacts["train"].read_bytes()
            artifacts["train"].write_bytes(b"tampered")
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), self.assertRaisesRegex(
                common.ContractError,
                "^Published split artifact drift: train/",
            ):
                dataset_generator.write_release_manifest(
                    self.policy,
                    mode=self.mode,
                    release_name=release_name,
                    split_payload_digests=ordered_digests,
                )
            artifacts["train"].write_bytes(original_train_bytes)

            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ):
                manifest_path = (
                    dataset_generator.write_release_manifest(
                        self.policy,
                        mode=self.mode,
                        release_name=release_name,
                        split_payload_digests=ordered_digests,
                    )
                )
            manifest = common.load_json(manifest_path)
            self.assertEqual(
                manifest["canonical_self_hash"],
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in manifest.items()
                        if key != "canonical_self_hash"
                    }
                ),
            )
            self.assertEqual(
                manifest["complete_split_order"],
                list(dataset_generator.SPLITS),
            )
            self.assertEqual(
                [row["split"] for row in manifest["splits"]],
                list(dataset_generator.SPLITS),
            )
            self.assertEqual(
                {row["role"] for row in manifest["parent_manifests"]},
                {
                    "split_train",
                    "split_development",
                    "split_audit_a",
                    "split_audit_b",
                },
            )
            self.assertEqual(
                manifest["m0_exact_mount_allowlist_per_split"],
                [
                    "observed/complete_model_pair_endpoints.csv",
                    "observed/redacted_items.jsonl",
                    "observed/seller_profiles.jsonl",
                ],
            )
            self.assertNotIn(
                "observed/items.jsonl",
                manifest["m0_exact_mount_allowlist_per_split"],
            )
            fixture_result_path = common.repo_path(
                str(
                    self.fixture["full_render_context_contract"][
                        "output_manifest"
                    ]["path"]
                )
            )
            self.assertEqual(
                manifest["preflight_fixture_result"]["file_sha256"],
                common.sha256_file(fixture_result_path),
            )
            self.assertEqual(
                set(entry.name for entry in parent.iterdir()),
                {*dataset_generator.SPLITS, "release_manifest.json"},
            )
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), self.assertRaises(FileExistsError):
                dataset_generator.write_release_manifest(
                    self.policy,
                    mode=self.mode,
                    release_name=release_name,
                    split_payload_digests=ordered_digests,
                )

    def test_complete_release_parent_fsync_failure_is_durability_unknown(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            release_root = Path(temporary) / "releases"
            parent, digests, _artifacts = (
                self._materialize_minimal_complete_release_children(
                    release_root
                )
            )
            release_name = self.policy[
                "development_complete_release"
            ]["release_name"]
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), mock.patch.object(
                dataset_generator,
                "_fsync_directory",
                side_effect=OSError("injected parent fsync failure"),
            ), self.assertRaisesRegex(
                common.ContractError,
                (
                    "^Complete release manifest was published but parent "
                    "directory durability is unknown;"
                ),
            ):
                dataset_generator.write_release_manifest(
                    self.policy,
                    mode=self.mode,
                    release_name=release_name,
                    split_payload_digests=digests,
                )
            manifest_path = parent / "release_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = common.load_json(manifest_path)
            self.assertEqual(
                manifest["canonical_self_hash"],
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in manifest.items()
                        if key != "canonical_self_hash"
                    }
                ),
            )

    def test_replay_uid_pool_order_and_duplicates_fail_closed(
        self,
    ) -> None:
        rows = [
            {"world_uid": "w_b", "seller_uid": "s_2"},
            {"world_uid": "w_a", "seller_uid": "s_1"},
        ]
        with self.assertRaisesRegex(
            replay_launcher.ReplayLauncherError,
            (
                "^REPLAY_LAUNCHER_POOL_ORDER_NONCANONICAL:"
                "seller_uid_pool$"
            ),
        ):
            replay_launcher._require_canonical_unique_rows(
                rows,
                fields=("world_uid", "seller_uid"),
                label="seller_uid_pool",
            )
        duplicate = [
            {"world_uid": "w_a", "seller_uid": "s_1"},
            {"world_uid": "w_a", "seller_uid": "s_1"},
        ]
        with self.assertRaisesRegex(
            replay_launcher.ReplayLauncherError,
            (
                "^REPLAY_LAUNCHER_POOL_ROW_DUPLICATE:"
                "seller_uid_pool$"
            ),
        ):
            replay_launcher._require_canonical_unique_rows(
                duplicate,
                fields=("world_uid", "seller_uid"),
                label="seller_uid_pool",
            )

    def test_independent_replay_rejects_complete_alternative_key_graph(
        self,
    ) -> None:
        alternative_policy = copy.deepcopy(self.policy)
        alternative_key = hashlib.sha256(
            b"step28-v13-independent-replay-alternative-graph-test"
        ).hexdigest()
        self.assertNotEqual(alternative_key, self.structure_key)
        alternative_policy["randomness"]["development_smoke"][
            "structure_key_hex"
        ] = alternative_key
        alternative_world = world_builder.build_world(
            policy=alternative_policy,
            template=self.template,
            fixture=self.fixture,
            style_profile=self.style,
            mode=self.mode,
            world_record=self.record,
            structure_key_hex=alternative_key,
        )
        self.assertNotEqual(
            alternative_world["private"]["controller_membership"],
            self.world["private"]["controller_membership"],
        )

        processed = production.process_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            template=self.template,
            world=alternative_world,
        )
        self.assertIs(
            processed["private"]["parser_structural_audit"][
                "exact_rows_and_flags"
            ],
            True,
        )
        self.assertIs(
            processed["private"]["redaction_structural_audit"][
                "post_redaction_seller_minimums_pass"
            ],
            True,
        )
        projected_slots, _nuisance, slot_audit = safe_slots.project_safe_slots(
            self.policy,
            mode=self.mode,
            split=self.split,
            sellers=alternative_world["public"]["sellers"],
            items=alternative_world["public"]["items"],
            parsed_rows=processed["private"][
                "parsed_identity_occurrences"
            ],
            identity_slots_edit=alternative_world["private"][
                "identity_slots_edit"
            ],
        )
        self.assertEqual(
            len(projected_slots),
            len(alternative_world["private"]["identity_slots_audit"]),
        )
        self.assertIs(
            slot_audit["exact_parser_edit_keyset_equality"],
            True,
        )
        with self.assertRaisesRegex(
            independent_comparator.IndependentDgpComparisonError,
            "^REPLAY_CONTROLLER_MEMBERSHIP_MISMATCH$",
        ):
            independent_comparator.compare_typed_dgp(
                expected_replay=self.independent_expected,
                producer_projection=producer_projection.project_world(
                    world=alternative_world,
                    mode=self.mode,
                    split=self.split,
                ),
            )

    def test_type_infeasible_leaf_advances_to_distinct_typed_topology(
        self,
    ) -> None:
        fixture_policy = copy.deepcopy(self.policy)
        fixture_policy["identity_design"]["slot_feasibility"][
            "type_assignment"
        ]["maximum_membership_complete_assignments"] = 5
        target_seller = (
            "sel_023e83ba8188c1f0bfa64a0b00f833aa3e390733d720c44c84c83"
            "ba74253a051"
        )
        forced_type = "bat"

        producer_positive = identity_plan_mod.build_positive_assets
        producer_negative = identity_plan_mod._iter_hard_negative_plans
        producer_assign_types = identity_plan_mod.assign_asset_types
        producer_topologies: list[str] = []
        producer_type_attempts: list[tuple[bool, int, int]] = []

        def producer_positive_fixture(*args, **kwargs):
            assets, targets = producer_positive(*args, **kwargs)
            narrowed = copy.deepcopy(assets)
            changed = 0
            for row in narrowed:
                if (
                    row["descriptor_kind"]
                    == "single_identity_stable_reuse"
                    and target_seller in row["sellers"]
                ):
                    row["allowed_types"] = [forced_type]
                    changed += 1
            self.assertEqual(changed, 1)
            return narrowed, targets

        def producer_negative_fixture(*args, **kwargs):
            for assets, flags, audit in producer_negative(*args, **kwargs):
                narrowed = copy.deepcopy(assets)
                for row in narrowed:
                    if (
                        row["descriptor_kind"] == "false_rotation"
                        and target_seller in row["sellers"]
                    ):
                        row["allowed_types"] = [forced_type]
                producer_topologies.append(
                    common.canonical_sha256(
                        [
                            {
                                key: row[key]
                                for key in (
                                    "descriptor_kind",
                                    "descriptor_index",
                                    "sellers",
                                    "occurrence_counts",
                                    "allowed_types",
                                    "fixed_type",
                                    "distinct_groups",
                                )
                            }
                            for row in narrowed
                        ]
                    )
                )
                yield narrowed, flags, audit

        def producer_type_fixture(*args, **kwargs):
            typed, nodes = producer_assign_types(*args, **kwargs)
            producer_type_attempts.append(
                (
                    typed is None,
                    nodes,
                    int(kwargs["maximum_nodes_override"]),
                )
            )
            return typed, nodes

        with mock.patch.object(
            identity_plan_mod,
            "build_positive_assets",
            side_effect=producer_positive_fixture,
        ), mock.patch.object(
            identity_plan_mod,
            "_iter_hard_negative_plans",
            side_effect=producer_negative_fixture,
        ), mock.patch.object(
            identity_plan_mod,
            "assign_asset_types",
            side_effect=producer_type_fixture,
        ):
            producer_world = world_builder.build_world(
                policy=fixture_policy,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                mode=self.mode,
                world_record=self.record,
                structure_key_hex=self.structure_key,
            )
        producer_solver = producer_world["private"]["solver_audit"]
        maximum_type_nodes = int(
            fixture_policy["identity_design"]["slot_feasibility"][
                "type_assignment"
            ]["maximum_search_nodes"]
        )
        self.assertEqual(len(producer_topologies), 2)
        self.assertNotEqual(
            producer_topologies[0],
            producer_topologies[1],
        )
        self.assertEqual(
            producer_type_attempts,
            [
                (True, 0, maximum_type_nodes),
                (False, 28, maximum_type_nodes),
            ],
        )
        self.assertEqual(
            producer_solver[
                "selected_membership_complete_assignment_ordinal"
            ],
            1,
        )
        self.assertEqual(
            producer_solver[
                "membership_complete_assignments_type_tested"
            ],
            2,
        )
        self.assertEqual(
            producer_solver["membership_solver_node_count"],
            20,
        )
        self.assertEqual(producer_solver["type_solver_node_count"], 28)

        replay_positive = independent_replay._positive_assets
        replay_negative = independent_replay._iter_hard_negative_leaves
        replay_assign_types = independent_replay._assign_identity_types
        replay_topologies: list[str] = []
        replay_type_attempts: list[tuple[bool, int, int]] = []

        def replay_positive_fixture(*args, **kwargs):
            assets, targets, repeats = replay_positive(*args, **kwargs)
            narrowed = copy.deepcopy(assets)
            changed = 0
            for row in narrowed:
                if (
                    row["descriptor_kind"]
                    == "single_identity_stable_reuse"
                    and target_seller in row["sellers"]
                ):
                    row["allowed_types"] = [forced_type]
                    changed += 1
            self.assertEqual(changed, 1)
            return narrowed, targets, repeats

        def replay_negative_fixture(*args, **kwargs):
            for assets, flags, audit in replay_negative(*args, **kwargs):
                narrowed = copy.deepcopy(assets)
                for row in narrowed:
                    if (
                        row["descriptor_kind"] == "false_rotation"
                        and target_seller in row["sellers"]
                    ):
                        row["allowed_types"] = [forced_type]
                replay_topologies.append(
                    common.canonical_sha256(
                        [
                            {
                                key: row[key]
                                for key in (
                                    "descriptor_kind",
                                    "descriptor_index",
                                    "sellers",
                                    "occurrence_counts",
                                    "allowed_types",
                                    "fixed_type",
                                    "distinct_groups",
                                )
                            }
                            for row in narrowed
                        ]
                    )
                )
                yield narrowed, flags, audit

        def replay_type_fixture(*args, **kwargs):
            typed, nodes = replay_assign_types(*args, **kwargs)
            replay_type_attempts.append(
                (
                    typed is None,
                    nodes,
                    int(kwargs["maximum_nodes"]),
                )
            )
            return typed, nodes

        observed_pools = independent_comparator.build_observed_uid_pools(
            world_uid=self.record["world_uid"],
            sellers=producer_world["public"]["sellers"],
            items=producer_world["public"]["items"],
        )
        with mock.patch.object(
            independent_replay,
            "_positive_assets",
            side_effect=replay_positive_fixture,
        ), mock.patch.object(
            independent_replay,
            "_iter_hard_negative_leaves",
            side_effect=replay_negative_fixture,
        ), mock.patch.object(
            independent_replay,
            "_assign_identity_types",
            side_effect=replay_type_fixture,
        ):
            expected = independent_replay.replay_typed_dgp(
                fixture_policy,
                mode=self.mode,
                split=self.split,
                world_uid=self.record["world_uid"],
                structure_key_hex=self.structure_key,
                **observed_pools,
            )
        replay_solver = expected["tables"]["solver_trace"]
        self.assertEqual(len(replay_topologies), 2)
        self.assertNotEqual(replay_topologies[0], replay_topologies[1])
        self.assertEqual(
            replay_type_attempts,
            [
                (True, 0, maximum_type_nodes),
                (False, 28, maximum_type_nodes),
            ],
        )
        self.assertEqual(
            replay_solver[
                "selected_membership_complete_assignment_ordinal"
            ],
            1,
        )
        self.assertEqual(
            replay_solver[
                "membership_complete_assignments_type_tested"
            ],
            2,
        )
        self.assertEqual(
            replay_solver["membership_solver_node_count"],
            20,
        )
        self.assertEqual(replay_solver["type_solver_node_count"], 28)
        projection = producer_projection.project_world(
            world=producer_world,
            mode=self.mode,
            split=self.split,
        )
        audit = independent_comparator.compare_typed_dgp(
            expected_replay=expected,
            producer_projection=projection,
        )
        self.assertIs(audit["full_typed_projection_exact"], True)

    def test_cross_leaf_type_node_budget_is_cumulative(self) -> None:
        """Accounting test: a failed leaf's nodes reduce the next budget."""

        fixture_policy = copy.deepcopy(self.policy)
        type_contract = fixture_policy["identity_design"][
            "slot_feasibility"
        ]["type_assignment"]
        type_contract["maximum_membership_complete_assignments"] = 2
        type_contract["maximum_search_nodes"] = 100

        producer_assign_types = identity_plan_mod.assign_asset_types
        producer_attempts: list[tuple[int, bool, int]] = []

        def producer_accounting_fixture(*args, **kwargs):
            budget = int(kwargs["maximum_nodes_override"])
            if not producer_attempts:
                result = (None, 7)
            else:
                result = producer_assign_types(*args, **kwargs)
            producer_attempts.append(
                (budget, result[0] is None, int(result[1]))
            )
            return result

        with mock.patch.object(
            identity_plan_mod,
            "assign_asset_types",
            side_effect=producer_accounting_fixture,
        ):
            producer_world = world_builder.build_world(
                policy=fixture_policy,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                mode=self.mode,
                world_record=self.record,
                structure_key_hex=self.structure_key,
            )
        self.assertEqual(
            producer_attempts,
            [(100, True, 7), (93, False, 28)],
        )
        producer_solver = producer_world["private"]["solver_audit"]
        self.assertEqual(
            producer_solver[
                "selected_membership_complete_assignment_ordinal"
            ],
            1,
        )
        self.assertEqual(producer_solver["type_solver_node_count"], 35)

        observed_pools = independent_comparator.build_observed_uid_pools(
            world_uid=self.record["world_uid"],
            sellers=producer_world["public"]["sellers"],
            items=producer_world["public"]["items"],
        )
        replay_assign_types = independent_replay._assign_identity_types
        replay_attempts: list[tuple[int, bool, int]] = []

        def replay_accounting_fixture(*args, **kwargs):
            budget = int(kwargs["maximum_nodes"])
            if not replay_attempts:
                result = (None, 7)
            else:
                result = replay_assign_types(*args, **kwargs)
            replay_attempts.append(
                (budget, result[0] is None, int(result[1]))
            )
            return result

        with mock.patch.object(
            independent_replay,
            "_assign_identity_types",
            side_effect=replay_accounting_fixture,
        ):
            expected = independent_replay.replay_typed_dgp(
                fixture_policy,
                mode=self.mode,
                split=self.split,
                world_uid=self.record["world_uid"],
                structure_key_hex=self.structure_key,
                **observed_pools,
            )
        self.assertEqual(
            replay_attempts,
            [(100, True, 7), (93, False, 28)],
        )
        replay_solver = expected["tables"]["solver_trace"]
        self.assertEqual(
            replay_solver[
                "selected_membership_complete_assignment_ordinal"
            ],
            1,
        )
        self.assertEqual(replay_solver["type_solver_node_count"], 35)
        audit = independent_comparator.compare_typed_dgp(
            expected_replay=expected,
            producer_projection=producer_projection.project_world(
                world=producer_world,
                mode=self.mode,
                split=self.split,
            ),
        )
        self.assertIs(audit["full_typed_projection_exact"], True)

    def test_cross_leaf_solver_node_counts_fail_closed(self) -> None:
        fixture_policy = copy.deepcopy(self.policy)
        type_contract = fixture_policy["identity_design"][
            "slot_feasibility"
        ]["type_assignment"]
        type_contract["maximum_membership_complete_assignments"] = 2
        type_contract["maximum_search_nodes"] = 100

        with mock.patch.object(
            identity_plan_mod,
            "assign_asset_types",
            return_value=(None, 101),
        ), self.assertRaisesRegex(
            common.ContractError,
            "invalid cross-membership node count",
        ):
            world_builder.build_world(
                policy=fixture_policy,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                mode=self.mode,
                world_record=self.record,
                structure_key_hex=self.structure_key,
            )
        with mock.patch.object(
            independent_replay,
            "_assign_identity_types",
            return_value=(None, 101),
        ), self.assertRaisesRegex(
            independent_replay.IndependentReplayError,
            "^REPLAY_TYPE_SOLVER_NODE_COUNT_INVALID$",
        ):
            independent_replay.replay_typed_dgp(
                fixture_policy,
                mode=self.mode,
                split=self.split,
                world_uid=self.record["world_uid"],
                structure_key_hex=self.structure_key,
                **self.observed_uid_pools,
            )

        with mock.patch.object(
            identity_plan_mod,
            "assign_asset_types",
            return_value=(None, 100),
        ), self.assertRaisesRegex(
            common.ContractError,
            "exhausted its cross-membership node budget",
        ):
            world_builder.build_world(
                policy=fixture_policy,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                mode=self.mode,
                world_record=self.record,
                structure_key_hex=self.structure_key,
            )
        with mock.patch.object(
            independent_replay,
            "_assign_identity_types",
            return_value=(None, 100),
        ), self.assertRaisesRegex(
            independent_replay.IndependentReplayError,
            "^REPLAY_TYPE_CROSS_LEAF_BUDGET_EXHAUSTED$",
        ):
            independent_replay.replay_typed_dgp(
                fixture_policy,
                mode=self.mode,
                split=self.split,
                world_uid=self.record["world_uid"],
                structure_key_hex=self.structure_key,
                **self.observed_uid_pools,
            )

    def test_independent_replay_and_no_key_comparator_file_round_trip(
        self,
    ) -> None:
        records = [
            row
            for row in structure.build_mode_world_pool(
                self.policy,
                mode=self.mode,
            )
            if row["split"] == self.split
        ]
        worlds: list[dict[str, object]] = []
        seller_rows: list[dict[str, str]] = []
        all_item_rows: list[dict[str, str]] = []
        title_item_rows: list[dict[str, str]] = []
        description_item_rows: list[dict[str, str]] = []
        projections: list[dict[str, object]] = []
        for record in records:
            world = (
                self.world
                if record["world_uid"] == self.record["world_uid"]
                else world_builder.build_world(
                    policy=self.policy,
                    template=self.template,
                    fixture=self.fixture,
                    style_profile=self.style,
                    mode=self.mode,
                    world_record=record,
                    structure_key_hex=self.structure_key,
                )
            )
            worlds.append(world)
            pools = independent_comparator.build_observed_uid_pools(
                world_uid=str(record["world_uid"]),
                sellers=world["public"]["sellers"],
                items=world["public"]["items"],
            )
            seller_rows.extend(
                {
                    "world_uid": str(record["world_uid"]),
                    "seller_uid": seller_uid,
                }
                for seller_uid in pools["observed_seller_uids"]
            )
            all_item_rows.extend(pools["observed_all_item_uid_rows"])
            title_item_rows.extend(
                pools["observed_nonempty_title_item_uid_rows"]
            )
            description_item_rows.extend(
                pools["observed_nonempty_description_item_uid_rows"]
            )
            projections.append(
                producer_projection.project_world(
                    world=world,
                    mode=self.mode,
                    split=self.split,
                )
            )
        expected_world_uids = independent_replay.registered_world_uids_for_split(
            self.policy,
            mode=self.mode,
            split=self.split,
        )
        self.assertEqual(
            [str(row["world_uid"]) for row in records],
            expected_world_uids,
        )
        for rows, fields in (
            (seller_rows, ("world_uid", "seller_uid")),
            (
                all_item_rows,
                ("world_uid", "seller_uid", "item_uid"),
            ),
            (
                title_item_rows,
                ("world_uid", "seller_uid", "item_uid"),
            ),
            (
                description_item_rows,
                ("world_uid", "seller_uid", "item_uid"),
            ),
        ):
            rows.sort(
                key=lambda row: tuple(
                    str(row[field]).encode("utf-8")
                    for field in fields
                )
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            world_pool = root / "worlds.csv"
            seller_pool = root / "seller_uid_pool.csv"
            all_items = root / "all_item_uid_pool.csv"
            title_items = root / "nonempty_title_item_uid_pool.csv"
            description_items = (
                root / "nonempty_description_item_uid_pool.csv"
            )
            common.write_csv(
                world_pool,
                [
                    {"world_uid": world_uid}
                    for world_uid in expected_world_uids
                ],
                ("world_uid",),
            )
            common.write_csv(
                seller_pool,
                seller_rows,
                ("world_uid", "seller_uid"),
            )
            for path, rows in (
                (all_items, all_item_rows),
                (title_items, title_item_rows),
                (description_items, description_item_rows),
            ):
                common.write_csv(
                    path,
                    rows,
                    ("world_uid", "seller_uid", "item_uid"),
                )
            replay_root = root / "replay"
            replay_args = SimpleNamespace(
                policy=str(
                    ROOT
                    / "schema"
                    / "step28_v13_synthetic_chinese_dataset_policy.json"
                ),
                mode=self.mode,
                split=self.split,
                world_pool=str(world_pool),
                seller_pool=str(seller_pool),
                all_item_pool=str(all_items),
                nonempty_title_pool=str(title_items),
                nonempty_description_pool=str(description_items),
                output_root=str(replay_root),
                validate_config_only=False,
            )
            replay_receipt = replay_launcher.run(replay_args)
            self.assertEqual(
                replay_receipt["world_count"],
                len(expected_world_uids),
            )
            self.assertIs(
                replay_receipt["complete_registered_world_set_exact"],
                True,
            )
            self.assertIs(replay_receipt["structure_key_serialized"], False)

            failed_replay_root = root / "failed_replay"
            failed_replay_args = copy.copy(replay_args)
            failed_replay_args.output_root = str(failed_replay_root)
            replay_write = replay_launcher._write_fsynced
            replay_write_count = 0

            def fail_second_replay_write(path, payload):
                nonlocal replay_write_count
                replay_write_count += 1
                if replay_write_count == 2:
                    raise OSError("injected replay receipt write failure")
                return replay_write(path, payload)

            with mock.patch.object(
                replay_launcher,
                "_write_fsynced",
                side_effect=fail_second_replay_write,
            ), self.assertRaisesRegex(
                OSError,
                "^injected replay receipt write failure$",
            ):
                replay_launcher.run(failed_replay_args)
            self.assertFalse(failed_replay_root.exists())
            self.assertEqual(
                list(
                    root.glob(
                        f".{failed_replay_root.name}.staging-*"
                    )
                ),
                [],
            )
            published_replay_root = root / "published_replay"
            published_replay_args = copy.copy(replay_args)
            published_replay_args.output_root = str(
                published_replay_root
            )

            def fail_replay_parent_fsync(path):
                if Path(path).resolve() == root.resolve():
                    raise OSError("injected replay parent fsync failure")

            with mock.patch.object(
                replay_launcher,
                "_fsync_directory",
                side_effect=fail_replay_parent_fsync,
            ), self.assertRaisesRegex(
                replay_launcher.ReplayLauncherError,
                (
                    "^REPLAY_LAUNCHER_OUTPUT_PUBLISHED_"
                    "PARENT_FSYNC_FAILED:"
                ),
            ):
                replay_launcher.run(published_replay_args)
            self.assertTrue(published_replay_root.is_dir())
            self.assertTrue(
                (
                    published_replay_root
                    / "replay_receipt.private.json"
                ).is_file()
            )

            producer_path = (
                root / "producer_typed_dgp_projection.private.jsonl"
            )
            producer_manifest_path = (
                root
                / "producer_typed_dgp_projection_manifest.private.json"
            )
            common.write_jsonl(producer_path, projections)
            producer_manifest = {
                "version": (
                    "2026-07-28-step28-v13-producer-typed-dgp-"
                    "projection-manifest-v1-draft"
                ),
                "mode": self.mode,
                "split": self.split,
                "evidence_level": (
                    "DEVELOPMENT_PRODUCER_PRIVATE_PROJECTION_"
                    "NOT_FORMAL_CUSTODY_SEAL"
                ),
                "formal_custody_seal": False,
                "policy_sha256": common.sha256_file(
                    ROOT
                    / "schema"
                    / "step28_v13_synthetic_chinese_dataset_policy.json"
                ),
                "world_count": len(expected_world_uids),
                "registered_split_world_count": len(
                    expected_world_uids
                ),
                "complete_registered_world_set_exact": True,
                "registered_world_uids_sha256": (
                    common.canonical_sha256(expected_world_uids)
                ),
                "projection_file": {
                    "role": "private_producer_typed_dgp_projection",
                    "path": (
                        "oracle/"
                        "producer_typed_dgp_projection.private.jsonl"
                    ),
                    "size_bytes": producer_path.stat().st_size,
                    "sha256": common.sha256_file(producer_path),
                },
                "source_record": {
                    "role": "producer_typed_dgp_projector",
                    "path_basename": (
                        "step28_v13_producer_dgp_projection.py"
                    ),
                    "sha256": common.sha256_file(
                        ROOT
                        / "scripts"
                        / "step28_v13_producer_dgp_projection.py"
                    ),
                },
            }
            producer_manifest["canonical_self_hash"] = (
                common.canonical_sha256(producer_manifest)
            )
            common.write_json(
                producer_manifest_path,
                producer_manifest,
            )
            comparison_root = root / "comparison"
            comparison_args = SimpleNamespace(
                policy=str(
                    ROOT
                    / "schema"
                    / "step28_v13_synthetic_chinese_dataset_policy.json"
                ),
                mode=self.mode,
                split=self.split,
                replay_ledgers=str(
                    replay_root / "world_replay_ledgers.private.jsonl"
                ),
                replay_receipt=str(
                    replay_root / "replay_receipt.private.json"
                ),
                producer_projections=str(producer_path),
                producer_manifest=str(producer_manifest_path),
                output_root=str(comparison_root),
                validate_config_only=False,
            )
            comparison_receipt = replay_compare_launcher.run(
                comparison_args
            )
            self.assertEqual(
                comparison_receipt["world_count"],
                len(expected_world_uids),
            )
            self.assertIs(comparison_receipt["all_worlds_exact"], True)
            self.assertEqual(
                comparison_receipt["structure_key_input_count"],
                0,
            )
            self.assertIs(
                comparison_receipt[
                    "registered_key_environment_names_present"
                ],
                False,
            )
            self.assertIn(
                "SELF_HASH_MANIFEST_BOUND",
                comparison_receipt["evidence_level"],
            )
            comparator_sources = {
                row["role"]: row
                for row in comparison_receipt[
                    "comparator_source_records"
                ]
            }
            self.assertEqual(
                set(comparator_sources),
                {
                    "development_comparator_launcher",
                    "development_comparator_implementation",
                },
            )
            self.assertEqual(
                comparator_sources[
                    "development_comparator_launcher"
                ]["sha256"],
                common.sha256_file(
                    ROOT
                    / "scripts"
                    / "step28_v13_compare_independent_dgp_replay.py"
                ),
            )
            self.assertEqual(
                comparator_sources[
                    "development_comparator_implementation"
                ]["sha256"],
                common.sha256_file(
                    ROOT
                    / "scripts"
                    / "step28_v13_independent_dgp_comparator.py"
                ),
            )
            failed_comparison_root = root / "failed_comparison"
            failed_comparison_args = copy.copy(comparison_args)
            failed_comparison_args.output_root = str(
                failed_comparison_root
            )
            with mock.patch.object(
                replay_compare_launcher.os,
                "replace",
                side_effect=OSError(
                    "injected comparator publish failure"
                ),
            ), self.assertRaisesRegex(
                OSError,
                "^injected comparator publish failure$",
            ):
                replay_compare_launcher.run(failed_comparison_args)
            self.assertFalse(failed_comparison_root.exists())
            self.assertEqual(
                list(
                    root.glob(
                        f".{failed_comparison_root.name}.staging-*"
                    )
                ),
                [],
            )
            published_comparison_root = root / "published_comparison"
            published_comparison_args = copy.copy(comparison_args)
            published_comparison_args.output_root = str(
                published_comparison_root
            )

            def fail_comparator_parent_fsync(path):
                if Path(path).resolve() == root.resolve():
                    raise OSError(
                        "injected comparator parent fsync failure"
                    )

            with mock.patch.object(
                replay_compare_launcher,
                "_fsync_directory",
                side_effect=fail_comparator_parent_fsync,
            ), self.assertRaisesRegex(
                replay_compare_launcher.ComparatorLauncherError,
                (
                    "^DGP_COMPARATOR_OUTPUT_PUBLISHED_"
                    "PARENT_FSYNC_FAILED:"
                ),
            ):
                replay_compare_launcher.run(
                    published_comparison_args
                )
            self.assertTrue(published_comparison_root.is_dir())
            self.assertTrue(
                (
                    published_comparison_root
                    / "aggregate_comparison_receipt.json"
                ).is_file()
            )

            secret_forms = (
                self.structure_key.encode("ascii"),
                bytes.fromhex(self.structure_key),
                base64.b64encode(bytes.fromhex(self.structure_key)),
            )
            for output_path in (
                replay_root / "world_replay_ledgers.private.jsonl",
                replay_root / "replay_receipt.private.json",
                comparison_root
                / "world_comparison_receipts.private.jsonl",
                comparison_root / "aggregate_comparison_receipt.json",
            ):
                payload = output_path.read_bytes()
                self.assertFalse(
                    any(secret in payload for secret in secret_forms),
                    output_path.name,
                )

            tampered_replay_receipt = common.load_json(
                replay_root / "replay_receipt.private.json"
            )
            tampered_replay_receipt["world_count"] += 1
            tampered_replay_root = root / "tampered_replay"
            tampered_replay_root.mkdir()
            tampered_replay_ledger_path = (
                tampered_replay_root
                / "world_replay_ledgers.private.jsonl"
            )
            tampered_replay_ledger_path.write_bytes(
                (
                    replay_root
                    / "world_replay_ledgers.private.jsonl"
                ).read_bytes()
            )
            tampered_replay_path = (
                tampered_replay_root / "replay_receipt.private.json"
            )
            common.write_json(
                tampered_replay_path,
                tampered_replay_receipt,
            )
            with self.assertRaisesRegex(
                replay_compare_launcher.ComparatorLauncherError,
                (
                    "^DGP_COMPARATOR_PARENT_SELF_HASH_INVALID:"
                    "replay_receipt$"
                ),
            ):
                replay_compare_launcher.run(
                    SimpleNamespace(
                        policy=str(
                            ROOT
                            / "schema"
                            / (
                                "step28_v13_synthetic_chinese_"
                                "dataset_policy.json"
                            )
                        ),
                        mode=self.mode,
                        split=self.split,
                        replay_ledgers=str(tampered_replay_ledger_path),
                        replay_receipt=str(tampered_replay_path),
                        producer_projections=str(producer_path),
                        producer_manifest=str(producer_manifest_path),
                        output_root=str(root / "tampered_comparison"),
                        validate_config_only=False,
                    )
                )

            stale_source_replay = common.load_json(
                replay_root / "replay_receipt.private.json"
            )
            stale_source_replay["source_records"][0]["sha256"] = "0" * 64
            stale_source_replay["canonical_self_hash"] = (
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in stale_source_replay.items()
                        if key != "canonical_self_hash"
                    }
                )
            )
            stale_source_replay_root = root / "stale_source_replay"
            stale_source_replay_root.mkdir()
            stale_source_replay_ledger = (
                stale_source_replay_root
                / "world_replay_ledgers.private.jsonl"
            )
            stale_source_replay_ledger.write_bytes(
                (
                    replay_root
                    / "world_replay_ledgers.private.jsonl"
                ).read_bytes()
            )
            stale_source_replay_receipt = (
                stale_source_replay_root / "replay_receipt.private.json"
            )
            common.write_json(
                stale_source_replay_receipt,
                stale_source_replay,
            )
            with self.assertRaisesRegex(
                replay_compare_launcher.ComparatorLauncherError,
                "^DGP_COMPARATOR_REPLAY_SOURCE_CLOSURE_MISMATCH$",
            ):
                replay_compare_launcher.run(
                    SimpleNamespace(
                        policy=comparison_args.policy,
                        mode=self.mode,
                        split=self.split,
                        replay_ledgers=str(stale_source_replay_ledger),
                        replay_receipt=str(
                            stale_source_replay_receipt
                        ),
                        producer_projections=str(producer_path),
                        producer_manifest=str(producer_manifest_path),
                        output_root=str(
                            root / "stale_source_replay_comparison"
                        ),
                        validate_config_only=False,
                    )
                )

            swapped_manifest = copy.deepcopy(producer_manifest)
            swapped_manifest["projection_file"]["sha256"] = "0" * 64
            swapped_manifest["canonical_self_hash"] = (
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in swapped_manifest.items()
                        if key != "canonical_self_hash"
                    }
                )
            )
            swapped_producer_root = root / "swapped_producer"
            swapped_producer_root.mkdir()
            swapped_projection_path = (
                swapped_producer_root
                / "producer_typed_dgp_projection.private.jsonl"
            )
            swapped_projection_path.write_bytes(producer_path.read_bytes())
            swapped_manifest_path = (
                swapped_producer_root
                / (
                    "producer_typed_dgp_projection_"
                    "manifest.private.json"
                )
            )
            common.write_json(
                swapped_manifest_path,
                swapped_manifest,
            )
            with self.assertRaisesRegex(
                replay_compare_launcher.ComparatorLauncherError,
                (
                    "^DGP_COMPARATOR_PRODUCER_PROJECTION_"
                    "PARENT_MISMATCH$"
                ),
            ):
                replay_compare_launcher.run(
                    SimpleNamespace(
                        policy=str(
                            ROOT
                            / "schema"
                            / (
                                "step28_v13_synthetic_chinese_"
                                "dataset_policy.json"
                            )
                        ),
                        mode=self.mode,
                        split=self.split,
                        replay_ledgers=str(
                            replay_root
                            / "world_replay_ledgers.private.jsonl"
                        ),
                        replay_receipt=str(
                            replay_root / "replay_receipt.private.json"
                        ),
                        producer_projections=str(swapped_projection_path),
                        producer_manifest=str(swapped_manifest_path),
                        output_root=str(root / "swapped_comparison"),
                        validate_config_only=False,
                    )
                )

            stale_source_producer = copy.deepcopy(producer_manifest)
            stale_source_producer["source_record"]["sha256"] = "0" * 64
            stale_source_producer["canonical_self_hash"] = (
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in stale_source_producer.items()
                        if key != "canonical_self_hash"
                    }
                )
            )
            stale_source_producer_root = (
                root / "stale_source_producer"
            )
            stale_source_producer_root.mkdir()
            stale_source_projection = (
                stale_source_producer_root
                / "producer_typed_dgp_projection.private.jsonl"
            )
            stale_source_projection.write_bytes(producer_path.read_bytes())
            stale_source_manifest = (
                stale_source_producer_root
                / (
                    "producer_typed_dgp_projection_"
                    "manifest.private.json"
                )
            )
            common.write_json(
                stale_source_manifest,
                stale_source_producer,
            )
            with self.assertRaisesRegex(
                replay_compare_launcher.ComparatorLauncherError,
                "^DGP_COMPARATOR_PRODUCER_SOURCE_CLOSURE_MISMATCH$",
            ):
                replay_compare_launcher.run(
                    SimpleNamespace(
                        policy=comparison_args.policy,
                        mode=self.mode,
                        split=self.split,
                        replay_ledgers=comparison_args.replay_ledgers,
                        replay_receipt=comparison_args.replay_receipt,
                        producer_projections=str(
                            stale_source_projection
                        ),
                        producer_manifest=str(stale_source_manifest),
                        output_root=str(
                            root / "stale_source_producer_comparison"
                        ),
                        validate_config_only=False,
                    )
                )

    def test_train_five_seed_m1_receipts_and_persistence_are_exact(
        self,
    ) -> None:
        split = "train"
        result = dataset_generator.build_split_payload(
            self.policy,
            mode=self.mode,
            split=split,
            template=self.template,
            fixture=self.fixture,
            style_profile=self.style,
        )
        self.assertEqual(len(result["placebos"]), 5)
        self.assertIs(
            result["support_comparability_preflight"][
                "all_five_primary_validity_pass"
            ],
            True,
        )
        self._assert_independent_support_preflight(result)
        expected_roles = integrity_receipts.expected_receipt_roles(
            self.policy, split=split
        )
        self.assertEqual(
            expected_roles,
            [
                "candidate_integrity",
                "independent_dgp_comparison",
                "m1_derangement_integrity",
                "render_integrity",
            ],
        )
        m1_receipt = result["aggregate_integrity_receipts"][
            "m1_derangement_integrity"
        ]
        self.assertEqual(
            m1_receipt["fixed_counts"],
            {
                "split_world_count": 10,
                "m1_replicate_count": 5,
                "m2_identity33_row_count": 3780,
                "m1_identity33_row_count": 18900,
                "mapping_row_count": 18900,
                "fixed_point_count": 0,
                "endpoint_overlap_count": 0,
                "multiset_mismatch_count": 0,
                "support_failure_count": 0,
            },
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release_root = root / "releases"
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ):
                producer_root = dataset_generator.write_split_release(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    release_name="train_exact",
                    result=result,
                )
            manifest = common.load_json(
                producer_root / "split_manifest.json"
            )
            self.assertEqual(
                manifest["placebo_replicate_count"], 5
            )
            self.assertEqual(
                manifest["aggregate_integrity_receipt_count"], 4
            )
            self.assertEqual(
                manifest["aggregate_parent_projection_count"], 4
            )
            self.assertIs(
                manifest["m1_support_comparability_pass"], True
            )
            for record in manifest["files"]:
                path = producer_root / str(record["path"])
                self.assertTrue(path.is_file())
                self.assertEqual(path.stat().st_size, record["size_bytes"])
                self.assertEqual(
                    common.sha256_file(path), record["sha256"]
                )
            listed_paths = [str(row["path"]) for row in manifest["files"]]
            self.assertEqual(
                listed_paths,
                sorted(
                    listed_paths, key=lambda value: value.encode("utf-8")
                ),
            )
            self.assertEqual(len(listed_paths), len(set(listed_paths)))
            actual_paths = {
                path.relative_to(producer_root).as_posix()
                for path in producer_root.rglob("*")
                if path.is_file()
                and path.name != "split_manifest.json"
            }
            self.assertEqual(actual_paths, set(listed_paths))
            self.assertEqual(
                manifest["canonical_self_hash"],
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in manifest.items()
                        if key != "canonical_self_hash"
                    }
                ),
            )
            identity33_schema = [
                "canonical_pair_uid",
                "world_uid",
                *self.policy["history_features"]["feature_names"],
            ]
            mapping_schema = self.policy["placebo"][
                "feature_derangement_mapping_schema"
            ]
            for output in result["placebos"]:
                seed_id = str(output["rewire_seed_id"])
                self.assertEqual(
                    (
                        producer_root
                        / "placebo"
                        / seed_id
                        / "identity33_all_pairs.csv"
                    ).read_bytes(),
                    common.csv_bytes(
                        output["identity33_all_pairs"],
                        identity33_schema,
                    ),
                )
                self.assertEqual(
                    (
                        producer_root
                        / "placebo_integrity_private"
                        / seed_id
                        / "feature_derangement_mapping.csv"
                    ).read_bytes(),
                    common.csv_bytes(
                        output["feature_derangement_mapping"],
                        mapping_schema,
                    ),
                )
            structural_path = (
                producer_root
                / "aggregate_integrity"
                / "structural_audit.receipt.json"
            )
            self.assertEqual(
                common.sha256_file(structural_path),
                integrity_receipts.pretty_json_sha256(
                    result["structural_audit_receipt"]
                ),
            )
            projection_path = (
                producer_root
                / "aggregate_integrity"
                / "parent_projections.development.json"
            )
            self.assertEqual(
                common.sha256_file(projection_path),
                integrity_receipts.pretty_json_sha256(
                    result["aggregate_parent_projections"]
                ),
            )
            placebo_dirs = sorted(
                (
                    path.name
                    for path in (producer_root / "placebo").iterdir()
                    if path.is_dir()
                ),
                key=lambda value: value.encode("utf-8"),
            )
            expected_seed_ids = sorted(
                (
                    str(row["rewire_seed_id"])
                    for row in result["placebos"]
                ),
                key=lambda value: value.encode("utf-8"),
            )
            self.assertEqual(placebo_dirs, expected_seed_ids)

            with self.assertRaisesRegex(
                common.ContractError,
                "replicate count drift$",
            ):
                dataset_generator._write_placebo_set(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    stage=root / "missing_seed_stage",
                    placebos=result["placebos"][:-1],
                    support_preflight=result[
                        "support_comparability_preflight"
                    ],
                    m2_identity33_all_pairs=result["tables"][
                        "identity33_all_pairs"
                    ],
                    candidate_pairs=result["tables"][
                        "candidate_pairs"
                    ],
                    complete_pair_endpoints=result["tables"][
                        "complete_model_pair_endpoints"
                    ],
                    m1_integrity_receipt=result[
                        "aggregate_integrity_receipts"
                    ]["m1_derangement_integrity"],
                )

            tampered_placebos = copy.deepcopy(result["placebos"])
            first_mapping = tampered_placebos[0][
                "feature_derangement_mapping"
            ]
            self.assertEqual(
                (
                    first_mapping[0]["world_uid"],
                    first_mapping[0]["universe"],
                ),
                (
                    first_mapping[1]["world_uid"],
                    first_mapping[1]["universe"],
                ),
            )
            first_mapping[0]["source_pair_uid"] = first_mapping[1][
                "source_pair_uid"
            ]
            tampered_placebos[0]["canonical_self_hash"] = (
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in tampered_placebos[0].items()
                        if key != "canonical_self_hash"
                    }
                )
            )
            with self.assertRaisesRegex(
                common.ContractError,
                "derangement mapping drift$",
            ):
                dataset_generator._write_placebo_set(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    stage=root / "duplicate_source_stage",
                    placebos=tampered_placebos,
                    support_preflight=result[
                        "support_comparability_preflight"
                    ],
                    m2_identity33_all_pairs=result["tables"][
                        "identity33_all_pairs"
                    ],
                    candidate_pairs=result["tables"][
                        "candidate_pairs"
                    ],
                    complete_pair_endpoints=result["tables"][
                        "complete_model_pair_endpoints"
                    ],
                    m1_integrity_receipt=result[
                        "aggregate_integrity_receipts"
                    ]["m1_derangement_integrity"],
                )

            forged_support_placebos = copy.deepcopy(
                result["placebos"]
            )
            forged_support_placebos[0][
                "feature_derangement_mapping"
            ][0]["feature_vector_sha256"] = "0" * 64
            forged_support_placebos[0]["canonical_self_hash"] = (
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in forged_support_placebos[
                            0
                        ].items()
                        if key != "canonical_self_hash"
                    }
                )
            )
            with self.assertRaisesRegex(
                common.ContractError,
                "vector lineage drift$",
            ):
                placebo_support.run_support_comparability_preflight(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    m2_identity33_all_pairs=result["tables"][
                        "identity33_all_pairs"
                    ],
                    candidate_pairs=result["tables"][
                        "candidate_pairs"
                    ],
                    complete_pair_endpoints=result["tables"][
                        "complete_model_pair_endpoints"
                    ],
                    placebos=forged_support_placebos,
                )

            forged_support = copy.deepcopy(
                result["support_comparability_preflight"]
            )
            forged_support["mode"] = "formal"
            forged_support["feature_count"] = -999
            forged_support["formal_use_forbidden"] = False
            forged_support["canonical_self_hash"] = common.canonical_sha256(
                {
                    key: value
                    for key, value in forged_support.items()
                    if key != "canonical_self_hash"
                }
            )
            with self.assertRaisesRegex(
                common.ContractError,
                "support preflight replay mismatch$",
            ):
                dataset_generator._write_placebo_set(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    stage=root / "forged_support_stage",
                    placebos=result["placebos"],
                    support_preflight=forged_support,
                    m2_identity33_all_pairs=result["tables"][
                        "identity33_all_pairs"
                    ],
                    candidate_pairs=result["tables"][
                        "candidate_pairs"
                    ],
                    complete_pair_endpoints=result["tables"][
                        "complete_model_pair_endpoints"
                    ],
                    m1_integrity_receipt=result[
                        "aggregate_integrity_receipts"
                    ]["m1_derangement_integrity"],
                )

            coordinated_placebos = copy.deepcopy(result["placebos"])
            target_seed_id = str(
                coordinated_placebos[0]["rewire_seed_id"]
            )
            replacement = copy.deepcopy(coordinated_placebos[1])
            replacement["rewire_seed_id"] = target_seed_id
            for row in replacement["feature_derangement_mapping"]:
                row["rewire_seed_id"] = target_seed_id
            replacement["canonical_self_hash"] = (
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in replacement.items()
                        if key != "canonical_self_hash"
                    }
                )
            )
            coordinated_placebos[0] = replacement
            coordinated_support = (
                placebo_support.run_support_comparability_preflight(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    m2_identity33_all_pairs=result["tables"][
                        "identity33_all_pairs"
                    ],
                    candidate_pairs=result["tables"][
                        "candidate_pairs"
                    ],
                    complete_pair_endpoints=result["tables"][
                        "complete_model_pair_endpoints"
                    ],
                    placebos=coordinated_placebos,
                )
            )
            coordinated_receipt = copy.deepcopy(m1_receipt)
            coordinated_receipt["aggregate_content_hashes"][
                "support_preflight_sha256"
            ] = common.canonical_sha256(coordinated_support)
            coordinated_receipt["canonical_self_hash"] = (
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in coordinated_receipt.items()
                        if key != "canonical_self_hash"
                    }
                )
            )
            with self.assertRaisesRegex(
                common.ContractError,
                "M1 integrity deterministic replay mismatch$",
            ):
                dataset_generator._write_placebo_set(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    stage=root / "coordinated_seed_stage",
                    placebos=coordinated_placebos,
                    support_preflight=coordinated_support,
                    m2_identity33_all_pairs=result["tables"][
                        "identity33_all_pairs"
                    ],
                    candidate_pairs=result["tables"][
                        "candidate_pairs"
                    ],
                    complete_pair_endpoints=result["tables"][
                        "complete_model_pair_endpoints"
                    ],
                    m1_integrity_receipt=coordinated_receipt,
                )
            self.assertFalse(
                (root / "coordinated_seed_stage").exists()
            )

    def test_audit_a_receipts_cannot_be_reused_as_audit_b(
        self,
    ) -> None:
        result = dataset_generator.build_split_payload(
            self.policy,
            mode=self.mode,
            split="audit_a",
            template=self.template,
            fixture=self.fixture,
            style_profile=self.style,
        )
        self.assertEqual(
            {
                int(receipt["fixed_counts"]["split_world_count"])
                for receipt in result[
                    "aggregate_integrity_receipts"
                ].values()
            },
            {5},
        )
        with self.assertRaisesRegex(
            common.ContractError,
            "registered split scope mismatch$",
        ):
            integrity_receipts.build_structural_audit_receipt(
                self.policy,
                mode=self.mode,
                split="audit_b",
                receipts=result["aggregate_integrity_receipts"],
                parent_projections=result[
                    "aggregate_parent_projections"
                ],
            )

    def test_generator_projection_artifacts_feed_separate_replay_chain(
        self,
    ) -> None:
        split = "development"
        result = dataset_generator.build_split_payload(
            self.policy,
            mode=self.mode,
            split=split,
            template=self.template,
            fixture=self.fixture,
            style_profile=self.style,
        )
        split_candidate_policy = (
            candidate_sampling.build_public_candidate_policy(
                self.policy, mode=self.mode, split=split
            )
        )
        split_candidate_context = (
            integrity_receipts.build_candidate_integrity_context(
                self.policy,
                candidate_policy=split_candidate_policy,
                mode=self.mode,
                split=split,
            )
        )
        tampered_projections = copy.deepcopy(
            result["tables"]["producer_typed_dgp_projections"]
        )
        tampered_projection = tampered_projections[0]
        tampered_projection["tables"]["seller_markets"][0][
            "market"
        ] += "_tampered"
        tampered_projection["typed_projection_sha256"] = (
            common.canonical_sha256(tampered_projection["tables"])
        )
        tampered_projection["canonical_self_hash"] = common.canonical_sha256(
            {
                key: value
                for key, value in tampered_projection.items()
                if key != "canonical_self_hash"
            }
        )
        with self.assertRaisesRegex(
            common.ContractError,
            "producer/comparison typed hash drift$",
        ):
            integrity_receipts.build_independent_dgp_comparison_receipt(
                self.policy,
                mode=self.mode,
                split=split,
                worlds=result["tables"]["worlds"],
                per_world_comparison_receipts=[
                    row["independent_typed_dgp_replay_audit"]
                    for row in result["tables"]["world_generation_audit"]
                ],
                producer_typed_dgp_projections=tampered_projections,
                independent_replay_ledgers=result["tables"][
                    "independent_typed_dgp_replay_ledgers"
                ],
            )
        tampered_candidate_audit = copy.deepcopy(
            result["tables"]["candidate_sampling_audit"]
        )
        tampered_candidate_audit[0]["hmac_digest_hex"] = "0" * 64
        with self.assertRaisesRegex(
            common.ContractError,
            "sampling-audit value drift$",
        ):
            integrity_receipts.build_candidate_integrity_receipt(
                split_candidate_context,
                candidate_policy=split_candidate_policy,
                candidate_key_hex=self.candidate_key_hex,
                mode=self.mode,
                split=split,
                worlds=result["tables"]["worlds"],
                sellers=result["tables"]["sellers"],
                raw_observed_items=result["tables"]["items"],
                complete_pair_endpoints=result["tables"][
                    "complete_model_pair_endpoints"
                ],
                candidate_pairs=result["tables"]["candidate_pairs"],
                candidate_sampling_audit=tampered_candidate_audit,
            )
        forged_candidate_audit = copy.deepcopy(
            result["tables"]["candidate_sampling_audit"]
        )
        original_similarity = str(
            forged_candidate_audit[0]["lexical_similarity"]
        )
        forged_candidate_audit[0]["lexical_similarity"] = (
            "1.000000"
            if original_similarity != "1.000000"
            else "-1.000000"
        )
        with self.assertRaisesRegex(
            common.ContractError,
            "independent trigger replay drift$",
        ):
            integrity_receipts.build_candidate_integrity_receipt(
                split_candidate_context,
                candidate_policy=split_candidate_policy,
                candidate_key_hex=self.candidate_key_hex,
                mode=self.mode,
                split=split,
                worlds=result["tables"]["worlds"],
                sellers=result["tables"]["sellers"],
                raw_observed_items=result["tables"]["items"],
                complete_pair_endpoints=result["tables"][
                    "complete_model_pair_endpoints"
                ],
                candidate_pairs=result["tables"]["candidate_pairs"],
                candidate_sampling_audit=forged_candidate_audit,
            )
        real_build_world_c40 = candidate_sampling.build_world_c40
        forged_pair_uid = str(
            forged_candidate_audit[0]["canonical_pair_uid"]
        )
        forged_similarity = str(
            forged_candidate_audit[0]["lexical_similarity"]
        )

        def common_mode_forged_candidate_replay(*args, **kwargs):
            safe_rows, audit_rows, replay_audit = (
                real_build_world_c40(*args, **kwargs)
            )
            audit_rows = copy.deepcopy(audit_rows)
            replay_audit = copy.deepcopy(replay_audit)
            for row in audit_rows:
                if str(row["canonical_pair_uid"]) == forged_pair_uid:
                    row["lexical_similarity"] = forged_similarity
            replay_audit["candidate_sampling_audit_sha256"] = (
                common.canonical_sha256(audit_rows)
            )
            return safe_rows, audit_rows, replay_audit

        with mock.patch.object(
            candidate_sampling,
            "build_world_c40",
            side_effect=common_mode_forged_candidate_replay,
        ), self.assertRaisesRegex(
            common.ContractError,
            "independent trigger replay drift$",
        ):
            integrity_receipts.build_candidate_integrity_receipt(
                split_candidate_context,
                candidate_policy=split_candidate_policy,
                candidate_key_hex=self.candidate_key_hex,
                mode=self.mode,
                split=split,
                worlds=result["tables"]["worlds"],
                sellers=result["tables"]["sellers"],
                raw_observed_items=result["tables"]["items"],
                complete_pair_endpoints=result["tables"][
                    "complete_model_pair_endpoints"
                ],
                candidate_pairs=result["tables"]["candidate_pairs"],
                candidate_sampling_audit=forged_candidate_audit,
            )
        orphan_sellers = copy.deepcopy(result["tables"]["sellers"])
        orphan_seller = dict(orphan_sellers[0])
        orphan_seller["world_uid"] = "wld_orphan_unregistered"
        orphan_sellers.append(orphan_seller)
        with self.assertRaisesRegex(
            common.ContractError,
            "sellers contains an unconsumed row$",
        ):
            integrity_receipts.build_render_integrity_receipt(
                self.policy,
                mode=self.mode,
                split=split,
                template=self.template,
                worlds=result["tables"]["worlds"],
                sellers=orphan_sellers,
                items=result["tables"]["items"],
                redacted_items=result["tables"]["redacted_items"],
                parsed_identity_occurrences=result["tables"][
                    "parsed_identity_occurrences"
                ],
                identity_slots_audit=result["tables"][
                    "identity_slots_audit"
                ],
                noise_slots_audit=result["tables"]["noise_slots_audit"],
                render_asts=result["tables"]["render_asts"],
                override_audit=result["tables"]["override_audit"],
            )
        tampered_replay_ledgers = copy.deepcopy(
            result["tables"]["independent_typed_dgp_replay_ledgers"]
        )
        tampered_ledger = tampered_replay_ledgers[0]
        tampered_ledger["tables"]["seller_markets"][0][
            "market"
        ] += "_tampered"
        tampered_ledger["typed_replay_sha256"] = common.canonical_sha256(
            tampered_ledger["tables"]
        )
        tampered_ledger["canonical_self_hash"] = common.canonical_sha256(
            {
                key: value
                for key, value in tampered_ledger.items()
                if key != "canonical_self_hash"
            }
        )
        with self.assertRaisesRegex(
            common.ContractError,
            "replay/comparison lineage drift$",
        ):
            integrity_receipts.build_independent_dgp_comparison_receipt(
                self.policy,
                mode=self.mode,
                split=split,
                worlds=result["tables"]["worlds"],
                per_world_comparison_receipts=[
                    row["independent_typed_dgp_replay_audit"]
                    for row in result["tables"][
                        "world_generation_audit"
                    ]
                ],
                producer_typed_dgp_projections=result["tables"][
                    "producer_typed_dgp_projections"
                ],
                independent_replay_ledgers=tampered_replay_ledgers,
            )
        with self.assertRaisesRegex(
            common.ContractError,
            "registered split scope mismatch$",
        ):
            integrity_receipts.build_structural_audit_receipt(
                self.policy,
                mode=self.mode,
                split="audit_a",
                receipts=result["aggregate_integrity_receipts"],
                parent_projections=result[
                    "aggregate_parent_projections"
                ],
            )
        tampered_parent_projections = copy.deepcopy(
            result["aggregate_parent_projections"]
        )
        tampered_parent_projections[0][
            "source_closure_sha256"
        ] = "0" * 64
        with self.assertRaisesRegex(
            common.ContractError,
            "parent projection receipt hash mismatch$",
        ):
            integrity_receipts.build_structural_audit_receipt(
                self.policy,
                mode=self.mode,
                split=split,
                receipts=result["aggregate_integrity_receipts"],
                parent_projections=tampered_parent_projections,
            )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            release_root = root / "releases"
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ):
                producer_root = dataset_generator.write_split_release(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    release_name="atomic_success",
                    result=result,
                )
            split_manifest = common.load_json(
                producer_root / "split_manifest.json"
            )
            postseal_orphan = copy.deepcopy(result)
            postseal_orphan["tables"]["sellers"].append(
                {
                    "world_uid": "wld_postseal_orphan",
                    "seller_uid": "sel_postseal_orphan",
                    "market": "market_a",
                }
            )
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), self.assertRaisesRegex(
                common.ContractError,
                "aggregate count gate failed$",
            ):
                dataset_generator.write_split_release(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    release_name="postseal_orphan",
                    result=postseal_orphan,
                )
            postseal_candidate = copy.deepcopy(result)
            candidate_row = postseal_candidate["tables"][
                "candidate_sampling_audit"
            ][0]
            candidate_row["lexical_similarity"] = (
                "1.000000"
                if candidate_row["lexical_similarity"] != "1.000000"
                else "-1.000000"
            )
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), self.assertRaisesRegex(
                common.ContractError,
                "snapshot deterministic regeneration mismatch$",
            ):
                dataset_generator.write_split_release(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    release_name="postseal_candidate",
                    result=postseal_candidate,
                )
            postseal_dgp = copy.deepcopy(result)
            dgp_projection = postseal_dgp["tables"][
                "producer_typed_dgp_projections"
            ][0]
            dgp_projection["tables"]["seller_markets"][0][
                "market"
            ] += "_postseal"
            dgp_projection["typed_projection_sha256"] = (
                common.canonical_sha256(dgp_projection["tables"])
            )
            dgp_projection["canonical_self_hash"] = (
                common.canonical_sha256(
                    {
                        key: value
                        for key, value in dgp_projection.items()
                        if key != "canonical_self_hash"
                    }
                )
            )
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), self.assertRaisesRegex(
                common.ContractError,
                "snapshot deterministic regeneration mismatch$",
            ):
                dataset_generator.write_split_release(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    release_name="postseal_dgp",
                    result=postseal_dgp,
                )
            self.assertEqual(
                split_manifest["version"],
                (
                    "2026-07-29-step28-v13-"
                    "split-dataset-manifest-v5-draft"
                ),
            )
            self.assertEqual(
                split_manifest["split_payload_digest_sha256"],
                common.canonical_sha256(result),
            )
            files = split_manifest["files"]
            self.assertIn(
                "private_producer_typed_dgp_projection_manifest",
                {str(row["role"]) for row in files},
            )
            self.assertEqual(
                split_manifest["aggregate_integrity_receipt_count"],
                3,
            )
            self.assertEqual(
                split_manifest["aggregate_parent_projection_count"],
                3,
            )
            self.assertIsNone(
                split_manifest["m1_support_comparability_pass"]
            )
            expected_roles = integrity_receipts.expected_receipt_roles(
                self.policy, split=split
            )
            self.assertEqual(
                expected_roles,
                [
                    "candidate_integrity",
                    "independent_dgp_comparison",
                    "render_integrity",
                ],
            )
            for role in expected_roles:
                receipt_path = (
                    producer_root
                    / "aggregate_integrity"
                    / f"{role}.receipt.json"
                )
                receipt = common.load_json(receipt_path)
                integrity_receipts.validate_aggregate_receipt(
                    self.policy,
                    role=role,
                    receipt=receipt,
                )
                self.assertEqual(
                    common.sha256_file(receipt_path),
                    integrity_receipts.pretty_json_sha256(receipt),
                )
            structural = common.load_json(
                producer_root
                / "aggregate_integrity"
                / "structural_audit.receipt.json"
            )
            integrity_receipts.validate_aggregate_receipt(
                self.policy,
                role="structural_audit",
                receipt=structural,
            )

            def fail_after_partial_table_write(
                policy,
                *,
                stage,
                payload,
            ):
                del policy, payload
                nested = stage / "observed"
                nested.mkdir()
                (nested / "partial.tmp").write_bytes(b"partial")
                raise OSError("injected dataset table write failure")

            failed_write_parent = release_root / "failed_write"
            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), mock.patch.object(
                dataset_generator,
                "_write_table_set",
                side_effect=fail_after_partial_table_write,
            ), self.assertRaisesRegex(
                OSError,
                "^injected dataset table write failure$",
            ):
                dataset_generator.write_split_release(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    release_name="failed_write",
                    result=result,
                )
            self.assertFalse(
                (failed_write_parent / split).exists()
            )
            self.assertEqual(
                list(failed_write_parent.glob(".staging-*")),
                [],
            )

            failed_replace_parent = release_root / "failed_replace"
            real_atomic_rename = common.atomic_rename_no_replace
            real_fsync_release_tree = (
                dataset_generator._fsync_release_tree
            )
            final_publish_attempt = {"hit": False}
            release_tree_fsync_completed = {"value": False}

            def record_release_tree_fsync(stage):
                real_fsync_release_tree(stage)
                release_tree_fsync_completed["value"] = True

            def fail_only_final_dataset_publish(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                if (
                    source_path.parent == failed_replace_parent
                    and source_path.name.startswith(
                        f".staging-{split}-"
                    )
                    and destination_path
                    == failed_replace_parent / split
                ):
                    final_publish_attempt["hit"] = True
                    self.assertTrue(
                        (source_path / "split_manifest.json").is_file()
                    )
                    materialized_file_count = sum(
                        len(file_names)
                        for _current, _directories, file_names
                        in os.walk(common.filesystem_path(source_path))
                    )
                    self.assertGreater(materialized_file_count, 20)
                    self.assertTrue(
                        release_tree_fsync_completed["value"]
                    )
                    raise OSError(
                        "injected dataset publish failure"
                    )
                return real_atomic_rename(source, destination)

            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), mock.patch.object(
                common,
                "atomic_rename_no_replace",
                side_effect=fail_only_final_dataset_publish,
            ), mock.patch.object(
                dataset_generator,
                "_fsync_release_tree",
                side_effect=record_release_tree_fsync,
            ), self.assertRaisesRegex(
                OSError,
                "^injected dataset publish failure$",
            ):
                dataset_generator.write_split_release(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    release_name="failed_replace",
                    result=result,
                )
            self.assertTrue(final_publish_attempt["hit"])
            self.assertFalse(
                (failed_replace_parent / split).exists()
            )
            self.assertEqual(
                list(failed_replace_parent.glob(".staging-*")),
                [],
            )

            parent_fsync_parent = (
                release_root / "parent_fsync_failure"
            )

            def fail_dataset_parent_fsync(path):
                if Path(path).resolve() == parent_fsync_parent.resolve():
                    raise OSError(
                        "injected dataset parent fsync failure"
                    )

            with mock.patch.object(
                common,
                "mode_output_root",
                return_value=release_root,
            ), mock.patch.object(
                dataset_generator,
                "_fsync_directory",
                side_effect=fail_dataset_parent_fsync,
            ), self.assertRaisesRegex(
                common.ContractError,
                (
                    "^Dataset split output was published but parent "
                    "directory fsync failed:"
                ),
            ):
                dataset_generator.write_split_release(
                    self.policy,
                    mode=self.mode,
                    split=split,
                    release_name="parent_fsync_failure",
                    result=result,
                )
            self.assertTrue(
                (parent_fsync_parent / split).is_dir()
            )

            projection_path = (
                producer_root
                / "oracle"
                / "producer_typed_dgp_projection.private.jsonl"
            )
            manifest_path = (
                producer_root
                / "oracle"
                / (
                    "producer_typed_dgp_projection_"
                    "manifest.private.json"
                )
            )
            projection_rows = replay_compare_launcher._read_jsonl(
                projection_path
            )
            forbidden_projection_keys = {
                "identity_value",
                "identity_uid",
                "global_asset_index",
                "title",
                "description",
                "slot_uid",
            }

            def all_keys(value):
                if isinstance(value, dict):
                    for key, nested in value.items():
                        yield key
                        yield from all_keys(nested)
                elif isinstance(value, list):
                    for nested in value:
                        yield from all_keys(nested)

            self.assertTrue(
                forbidden_projection_keys.isdisjoint(
                    set(all_keys(projection_rows))
                )
            )
            replay_root = root / "replay"
            replay_receipt = replay_launcher.run(
                SimpleNamespace(
                    policy=str(
                        ROOT
                        / "schema"
                        / (
                            "step28_v13_synthetic_chinese_"
                            "dataset_policy.json"
                        )
                    ),
                    mode=self.mode,
                    split=split,
                    world_pool=str(
                        producer_root / "observed" / "worlds.csv"
                    ),
                    seller_pool=str(
                        producer_root
                        / "structural_audit"
                        / "independent_replay_inputs"
                        / "seller_uid_pool.csv"
                    ),
                    all_item_pool=str(
                        producer_root
                        / "structural_audit"
                        / "independent_replay_inputs"
                        / "all_item_uid_pool.csv"
                    ),
                    nonempty_title_pool=str(
                        producer_root
                        / "structural_audit"
                        / "independent_replay_inputs"
                        / "nonempty_title_item_uid_pool.csv"
                    ),
                    nonempty_description_pool=str(
                        producer_root
                        / "structural_audit"
                        / "independent_replay_inputs"
                        / "nonempty_description_item_uid_pool.csv"
                    ),
                    output_root=str(replay_root),
                    validate_config_only=False,
                )
            )
            self.assertEqual(replay_receipt["world_count"], 3)
            persisted_replay_ledgers = (
                replay_compare_launcher._read_jsonl(
                    replay_root
                    / "world_replay_ledgers.private.jsonl"
                )
            )
            self.assertEqual(
                integrity_receipts.canonical_multiset_sha256(
                    persisted_replay_ledgers
                ),
                result["aggregate_integrity_receipts"][
                    "independent_dgp_comparison"
                ]["input_parent_hashes"][
                    "independent_replay_parent_sha256"
                ],
            )
            with mock.patch.dict("os.environ", {}, clear=True):
                comparison = replay_compare_launcher.run(
                    SimpleNamespace(
                        policy=str(
                            ROOT
                            / "schema"
                            / (
                                "step28_v13_synthetic_chinese_"
                                "dataset_policy.json"
                            )
                        ),
                        mode=self.mode,
                        split=split,
                        replay_ledgers=str(
                            replay_root
                            / "world_replay_ledgers.private.jsonl"
                        ),
                        replay_receipt=str(
                            replay_root / "replay_receipt.private.json"
                        ),
                        producer_projections=str(projection_path),
                        producer_manifest=str(manifest_path),
                        output_root=str(root / "comparison"),
                        validate_config_only=False,
                    )
                )
            self.assertEqual(comparison["world_count"], 3)
            self.assertIs(comparison["all_worlds_exact"], True)

    def test_public_identity_boundary_is_slot_count_invariant(self) -> None:
        guards = text_renderer.context_guard_pool(self.template)
        prefix = "基础商品文字 页面序号ZX-1234仅作核对。"
        expected = source.normalize_redacted_text(prefix)
        for count in range(9):
            selected = (
                text_renderer.context_guard_sequence(
                    selector_uid=f"boundary_test_{count}",
                    count=count + 1,
                    template=self.template,
                )
                if count
                else ()
            )
            text = prefix
            if selected:
                text += selected[0]
                for index, guard in enumerate(selected[1:]):
                    text += f" 第{index}个身份残片。" + guard
            clean, audit = production._canonicalize_synthetic_description(
                text,
                guards=guards,
            )
            self.assertEqual(clean, expected)
            self.assertEqual(
                audit["precanonical_context_guard_count"],
                count + 1 if count else 0,
            )

    def test_natural_text_variation_domains_are_complete_and_parser_safe(
        self,
    ) -> None:
        lexicon = self.template["generic_lexicon"]
        modifiers = list(lexicon["title_modifiers"])
        self.assertEqual(len(modifiers), 16)
        self.assertEqual(len(set(modifiers)), 16)
        self.assertGreaterEqual(min(map(len, lexicon["delivery"])), 28)
        self.assertGreaterEqual(min(map(len, lexicon["service"])), 33)
        for split, library in self.template["split_libraries"].items():
            titles = list(library["title_skeletons"])
            self.assertEqual(
                sum("{code}" in value for value in titles),
                4,
                split,
            )
            self.assertEqual(
                sum("{title_modifier}" in value for value in titles),
                4,
                split,
            )
            self.assertTrue(
                all(
                    ("{code}" in value) != ("{title_modifier}" in value)
                    for value in titles
                ),
                split,
            )

        fixture_codes = self.fixture["full_render_context_contract"][
            "fixture_code_values"
        ]
        self.assertEqual(
            {
                text_renderer.title_modifier(code, self.template)
                for code in fixture_codes
            },
            set(modifiers),
        )
        self.assertTrue(text_renderer.english_tag_visible("QAAAAAAAAAA"))
        self.assertFalse(text_renderer.english_tag_visible("QAAAAAAAAAP"))

        neutral_regions = [
            *lexicon["delivery"],
            *lexicon["service"],
            *modifiers,
            *text_renderer.context_guard_pool(self.template),
        ]
        for value in neutral_regions:
            self.assertIsNone(step3.PRODUCT_DATA_RISK_RE.search(value), value)
            self.assertIsNone(step3.SELLER_CONTACT_CUE_RE.search(value), value)
            self.assertIsNone(step3.WALLET_CUE_RE.search(value), value)

    def test_formal_handle_encoding_is_parser_safe(self) -> None:
        encoding = self.policy["identity_design"][
            "identity_value_generation"
        ]["handle_encoding_by_mode"]["formal"]
        self.assertEqual(encoding, "parser_safe_hex_v2")
        key_hex = self.policy["randomness"]["formal"][
            "identity_value_key_hex"
        ]
        for identity_type in ("telegram", "bat", "wechat"):
            values = [
                identity_values_mod.identity_value(
                    key_hex=key_hex,
                    identity_type=identity_type,
                    salt=0,
                    global_asset_index=index,
                    handle_encoding=encoding,
                )
                for index in range(10_000)
            ]
            self.assertEqual(len(values), len(set(values)))
            for value in values:
                self.assertIsNone(
                    step3.PRODUCT_DATA_RISK_RE.search(value),
                    value,
                )

    def test_unknown_columns_fail_at_every_observed_worker_boundary(self) -> None:
        bad_sellers = copy.deepcopy(self.world["public"]["sellers"])
        bad_sellers[0]["label"] = 1
        with self.assertRaises(common.ContractError):
            production.registry_profiles_from_sellers(
                self.policy,
                sellers=bad_sellers,
            )

    def test_item_value_domains_and_persisted_title_are_fail_closed(self) -> None:
        for invalid in (-1, 4, "1", True, float("nan")):
            bad_items = copy.deepcopy(self.world["public"]["items"])
            bad_items[0]["time_bucket"] = invalid
            with self.assertRaises(common.ContractError):
                production.project_history_safe_occurrences(
                    self.policy,
                    mode=self.mode,
                    split=self.split,
                    sellers=self.world["public"]["sellers"],
                    items=bad_items,
                    parsed_rows=self.parsed,
                )

        result = production.process_world(
            self.policy,
            mode="development_smoke",
            split=self.record["split"],
            template=self.template,
            world=self.world,
        )
        bad_redacted = copy.deepcopy(result["public"]["redacted_items"])
        bad_redacted[0]["title"] += " LABEL1"
        with self.assertRaises(common.ContractError):
            production.validate_redaction_against_private_plan(
                self.policy,
                mode=self.mode,
                split=self.record["split"],
                template=self.template,
                sellers=self.world["public"]["sellers"],
                items=self.world["public"]["items"],
                redacted_items=bad_redacted,
                parsed_rows=self.parsed,
                identity_slots_audit=self.world["private"][
                    "identity_slots_audit"
                ],
                noise_slots_audit=self.world["private"]["noise_slots_audit"],
                render_asts=self.world["private"]["render_asts"],
                override_audit=self.world["private"]["override_audit"],
            )

        bad_parser = copy.deepcopy(self.parsed)
        bad_parser[0]["label"] = 1
        with self.assertRaises(common.ContractError):
            production.redact_observed_world(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                sellers=self.world["public"]["sellers"],
                items=self.world["public"]["items"],
                registry_profiles=self.registry_profiles,
                parsed_rows=bad_parser,
            )
        with self.assertRaises(common.ContractError):
            production.project_history_safe_occurrences(
                self.policy,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                items=self.world["public"]["items"],
                parsed_rows=bad_parser,
            )
        with self.assertRaises(common.ContractError):
            safe_slots.project_safe_slots(
                self.policy,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                items=self.world["public"]["items"],
                parsed_rows=bad_parser,
                identity_slots_edit=self.world["private"][
                    "identity_slots_edit"
                ],
            )

        bad_items = copy.deepcopy(self.world["public"]["items"])
        bad_items[0]["controller_uid"] = "forbidden"
        with self.assertRaises(common.ContractError):
            production.project_history_safe_occurrences(
                self.policy,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                items=bad_items,
                parsed_rows=self.parsed,
            )
        with self.assertRaises(common.ContractError):
            safe_slots.project_safe_slots(
                self.policy,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                items=bad_items,
                parsed_rows=self.parsed,
                identity_slots_edit=self.world["private"][
                    "identity_slots_edit"
                ],
            )

    def test_world_builder_rejects_unregistered_record_or_key(self) -> None:
        bad_record = dict(self.record)
        bad_record["mode_global_ordinal"] += 1
        with self.assertRaises(common.ContractError):
            world_builder.build_world(
                policy=self.policy,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                mode="development_smoke",
                world_record=bad_record,
                structure_key_hex=self.structure_key,
            )
        with self.assertRaises(common.ContractError):
            world_builder.build_world(
                policy=self.policy,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                mode="development_smoke",
                world_record=self.record,
                structure_key_hex=self.policy["randomness"][
                    "development_smoke"
                ]["text_key_hex"],
            )
        for invalid_ordinal in (True, float(self.record["mode_global_ordinal"])):
            bad_record = dict(self.record)
            bad_record["mode_global_ordinal"] = invalid_ordinal
            with self.assertRaises(common.ContractError):
                world_builder.build_world(
                    policy=self.policy,
                    template=self.template,
                    fixture=self.fixture,
                    style_profile=self.style,
                    mode=self.mode,
                    world_record=bad_record,
                    structure_key_hex=self.structure_key,
                )

    def test_policy_structure_validation_opens_no_frozen_input(self) -> None:
        with mock.patch.object(
            common,
            "verify_file_pin",
            side_effect=AssertionError("structure validation opened an input"),
        ):
            common.validate_policy(self.policy, mode=self.mode)

    def test_public_prefix_generic_identifier_deletion_fails(self) -> None:
        guards = text_renderer.context_guard_pool(self.template)
        mutated_items = copy.deepcopy(self.world["public"]["items"])
        target = next(
            row
            for row in mutated_items
            if any(
                guard in row["description"]
                for guard in guards
            )
        )
        boundary, _count = production._context_guard_boundary(
            target["description"],
            guards=guards,
        )
        self.assertIsNotNone(boundary)
        target["description"] = (
            target["description"][: int(boundary)]
            + "123456789012"
            + target["description"][int(boundary) :]
        )
        parsed = production.parse_observed_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            items=mutated_items,
        )
        with self.assertRaises(common.ContractError):
            production.redact_observed_world(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                sellers=self.world["public"]["sellers"],
                items=mutated_items,
                registry_profiles=self.registry_profiles,
                parsed_rows=parsed,
            )

    def test_parser_artifact_requires_exact_step3_replay(self) -> None:
        forged = copy.deepcopy(self.parsed)
        row = dict(forged[0])
        row["normalized_value"] = "forgedlabeltoken"
        row["raw_value"] = "FORGED_LABEL_TOKEN"
        signal_uid_raw = "|".join(
            (
                row["source_dataset"],
                row["source_row_number"],
                row["contact_type"],
                row["normalized_value"],
                row["source_field"],
            )
        )
        row["signal_uid"] = hashlib.sha1(
            signal_uid_raw.encode("utf-8")
        ).hexdigest()
        forged.append(row)
        with self.assertRaises(common.ContractError):
            production.validate_parser_artifact(
                self.policy,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                items=self.world["public"]["items"],
                parsed_rows=forged,
            )

    def test_private_audits_reject_empty_or_coordinated_truncation(self) -> None:
        with self.assertRaises(common.ContractError):
            production.project_history_safe_occurrences(
                self.policy,
                mode=self.mode,
                split=self.split,
                sellers=[],
                items=[],
                parsed_rows=[],
            )
        with self.assertRaises(common.ContractError):
            production.validate_parser_against_private_plan(
                self.policy,
                mode=self.mode,
                split=self.split,
                sellers=[],
                items=[],
                parsed_rows=[],
                identity_slots_audit=[],
                noise_slots_audit=[],
                render_asts=[],
            )
        removed_slot = self.world["private"]["identity_slots_audit"][0]
        truncated_plan = [
            row
            for row in self.world["private"]["identity_slots_audit"]
            if row["slot_uid"] != removed_slot["slot_uid"]
        ]
        removed_tuple = production._planned_parser_tuple(removed_slot)
        truncated_parser = [
            row
            for row in self.parsed
            if production._actual_parser_tuple(row) != removed_tuple
        ]
        with self.assertRaises(common.ContractError):
            production.validate_parser_against_private_plan(
                self.policy,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                items=self.world["public"]["items"],
                parsed_rows=truncated_parser,
                identity_slots_audit=truncated_plan,
                noise_slots_audit=self.world["private"]["noise_slots_audit"],
                render_asts=self.world["private"]["render_asts"],
            )

    def test_private_ast_rejects_synchronized_public_text_injection(self) -> None:
        mutated_items = copy.deepcopy(self.world["public"]["items"])
        target = next(row for row in mutated_items if row["title"])
        target["title"] += " 标签一"
        parsed = production.parse_observed_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            items=mutated_items,
        )
        redaction = production.redact_observed_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            template=self.template,
            sellers=self.world["public"]["sellers"],
            items=mutated_items,
            registry_profiles=self.registry_profiles,
            parsed_rows=parsed,
        )
        with self.assertRaises(common.ContractError):
            production.validate_redaction_against_private_plan(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                sellers=self.world["public"]["sellers"],
                items=mutated_items,
                redacted_items=redaction["redacted_items"],
                parsed_rows=parsed,
                identity_slots_audit=self.world["private"][
                    "identity_slots_audit"
                ],
                noise_slots_audit=self.world["private"]["noise_slots_audit"],
                render_asts=self.world["private"]["render_asts"],
                override_audit=self.world["private"]["override_audit"],
            )

    def test_private_ast_binds_noise_and_clean_prefix(self) -> None:
        result = production.process_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            template=self.template,
            world=self.world,
        )
        missing_noise = self.world["private"]["noise_slots_audit"][0]
        truncated_noise = [
            row
            for row in self.world["private"]["noise_slots_audit"]
            if row["noise_slot_uid"] != missing_noise["noise_slot_uid"]
        ]
        tampered_clean = copy.deepcopy(result["public"]["redacted_items"])
        clean = next(
            row for row in tampered_clean if row["item_uid"] == missing_noise["item_uid"]
        )
        clean["description"] = clean["description"].replace(
            missing_noise["raw_surface"],
            "",
            1,
        )
        with self.assertRaises(common.ContractError):
            production.validate_redaction_against_private_plan(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                sellers=self.world["public"]["sellers"],
                items=self.world["public"]["items"],
                redacted_items=tampered_clean,
                parsed_rows=self.parsed,
                identity_slots_audit=self.world["private"][
                    "identity_slots_audit"
                ],
                noise_slots_audit=truncated_noise,
                render_asts=self.world["private"]["render_asts"],
                override_audit=self.world["private"]["override_audit"],
            )

    def test_registered_overrides_are_bound_to_negative_targets(self) -> None:
        bad_override = copy.deepcopy(self.world)
        bad_override["private"]["override_audit"][0]["asset_index"] = -999
        with self.assertRaises(common.ContractError):
            smoke_regeneration.validate_producer_regeneration_match(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                world=bad_override,
            )
        missing_flags = copy.deepcopy(self.world)
        missing_flags["private"]["negative_flags"] = [
            row
            for row in missing_flags["private"]["negative_flags"]
            if row["flag"]
            not in {
                "exact_title_clone_target",
                "high_semantic_similarity_target",
            }
        ]
        with self.assertRaises(common.ContractError):
            smoke_regeneration.validate_producer_regeneration_match(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                world=missing_flags,
            )

    def test_registered_override_hmac_side_and_item_are_replayed(self) -> None:
        swapped_side = copy.deepcopy(self.world)
        clone = next(
            row
            for row in swapped_side["private"]["override_audit"]
            if row["override_kind"] == "exact_title_clone"
        )
        clone["seller_uid_left"], clone["seller_uid_right"] = (
            clone["seller_uid_right"],
            clone["seller_uid_left"],
        )
        clone["item_uid_left"], clone["item_uid_right"] = (
            clone["item_uid_right"],
            clone["item_uid_left"],
        )
        with self.assertRaisesRegex(
            common.ContractError,
            "smoke producer regeneration",
        ):
            smoke_regeneration.validate_producer_regeneration_match(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                world=swapped_side,
            )

        changed_item = copy.deepcopy(self.world)
        target = next(
            row
            for row in changed_item["private"]["override_audit"]
            if row["override_kind"] == "exact_title_clone"
        )
        used_item_uids = {
            str(row[name])
            for row in changed_item["private"]["override_audit"]
            for name in ("item_uid_left", "item_uid_right")
        }
        replacement = next(
            row
            for row in self.world["public"]["items"]
            if row["seller_uid"] == target["seller_uid_left"]
            and row["item_uid"] not in used_item_uids
        )
        target["item_uid_left"] = replacement["item_uid"]
        with self.assertRaisesRegex(
            common.ContractError,
            "smoke producer regeneration",
        ):
            smoke_regeneration.validate_producer_regeneration_match(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                world=changed_item,
            )

        coordinated_items = copy.deepcopy(self.world["public"]["items"])
        coordinated_redacted = copy.deepcopy(
            self.processed["public"]["redacted_items"]
        )
        coordinated_override = copy.deepcopy(
            self.world["private"]["override_audit"]
        )
        coordinated_target = next(
            row
            for row in coordinated_override
            if row["override_kind"] == "exact_title_clone"
        )
        used_item_uids = {
            str(row[name])
            for row in coordinated_override
            for name in ("item_uid_left", "item_uid_right")
        }
        replacement_source = next(
            row
            for row in coordinated_items
            if row["seller_uid"] == coordinated_target["seller_uid_left"]
            and row["item_uid"] not in used_item_uids
            and row["title"]
        )
        replacement_destination = next(
            row
            for row in coordinated_items
            if row["seller_uid"] == coordinated_target["seller_uid_right"]
            and row["item_uid"] not in used_item_uids
            and row["title"]
        )
        ast_index = {
            row["item_uid"]: row
            for row in self.world["private"]["render_asts"]
        }
        styles = {
            row["effective_style_uid"]: row
            for row in production.renderer.reachable_effective_styles(
                self.template
            )
        }

        def replay_base_title(item_uid: str) -> str:
            return production._base_title(
                ast=ast_index[item_uid],
                split=self.split,
                template=self.template,
                styles=styles,
            )

        old_destination_uid = coordinated_target["item_uid_right"]
        new_destination_uid = replacement_destination["item_uid"]
        new_source_uid = replacement_source["item_uid"]
        coordinated_item_index = {
            row["item_uid"]: row for row in coordinated_items
        }
        coordinated_clean_index = {
            row["item_uid"]: row for row in coordinated_redacted
        }
        coordinated_item_index[old_destination_uid]["title"] = (
            replay_base_title(old_destination_uid)
        )
        coordinated_clean_index[old_destination_uid]["title"] = (
            source.normalize_redacted_text(
                coordinated_item_index[old_destination_uid]["title"]
            )
        )
        coordinated_item_index[new_destination_uid]["title"] = (
            replay_base_title(new_source_uid)
        )
        coordinated_clean_index[new_destination_uid]["title"] = (
            source.normalize_redacted_text(
                coordinated_item_index[new_destination_uid]["title"]
            )
        )
        coordinated_target["item_uid_left"] = new_source_uid
        coordinated_target["item_uid_right"] = new_destination_uid
        coordinated_parsed = production.parse_observed_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            items=coordinated_items,
        )
        production.validate_redaction_against_private_plan(
            self.policy,
            mode=self.mode,
            split=self.split,
            template=self.template,
            sellers=self.world["public"]["sellers"],
            items=coordinated_items,
            redacted_items=coordinated_redacted,
            parsed_rows=coordinated_parsed,
            identity_slots_audit=self.world["private"][
                "identity_slots_audit"
            ],
            noise_slots_audit=self.world["private"]["noise_slots_audit"],
            render_asts=self.world["private"]["render_asts"],
            override_audit=coordinated_override,
        )
        coordinated_world = copy.deepcopy(self.world)
        coordinated_world["public"]["items"] = coordinated_items
        coordinated_world["private"]["override_audit"] = coordinated_override
        with self.assertRaisesRegex(
            common.ContractError,
            "smoke producer regeneration",
        ):
            smoke_regeneration.validate_producer_regeneration_match(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                world=coordinated_world,
            )

    def test_registered_override_replay_binds_controller_membership(self) -> None:
        changed_world = copy.deepcopy(self.world)
        changed_membership = changed_world["private"]["controller_membership"]
        left = changed_membership[0]
        right = next(
            row
            for row in changed_membership[1:]
            if row["controller_uid"] != left["controller_uid"]
        )
        left["controller_uid"], right["controller_uid"] = (
            right["controller_uid"],
            left["controller_uid"],
        )
        with self.assertRaisesRegex(
            common.ContractError,
            "smoke producer regeneration",
        ):
            smoke_regeneration.validate_producer_regeneration_match(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                world=changed_world,
            )

    def test_registered_identity_plan_rejects_coordinated_slot_graph_swap(
        self,
    ) -> None:
        assets = self.world["private"]["identity_assets"]
        slots = self.world["private"]["identity_slots_audit"]
        slots_by_identity: dict[str, list[dict[str, object]]] = {}
        for row in slots:
            slots_by_identity.setdefault(str(row["identity_uid"]), []).append(
                row
            )
        selected: tuple[
            dict[str, object],
            dict[str, object],
            dict[str, object],
            dict[str, object],
        ] | None = None
        for one_asset in assets:
            if sum(one_asset["occurrence_counts"].values()) != 1:
                continue
            one_slot = slots_by_identity[str(one_asset["identity_uid"])][0]
            for two_asset in assets:
                if (
                    sum(two_asset["occurrence_counts"].values()) != 2
                    or two_asset["identity_type"] != one_asset["identity_type"]
                    or two_asset["role"] != one_asset["role"]
                ):
                    continue
                two_slot = next(
                    (
                        row
                        for row in slots_by_identity[
                            str(two_asset["identity_uid"])
                        ]
                        if row["seller_uid"] != one_slot["seller_uid"]
                    ),
                    None,
                )
                if two_slot is not None:
                    selected = one_asset, two_asset, one_slot, two_slot
                    break
            if selected is not None:
                break
        self.assertIsNotNone(selected)
        _one_asset, _two_asset, original_one_slot, original_two_slot = selected
        self.assertEqual(
            len(str(original_one_slot["raw_surface"])),
            len(str(original_two_slot["raw_surface"])),
        )
        for name in (
            "identity_uid",
            "downstream_canonical_value",
            "raw_surface",
        ):
            self.assertNotEqual(
                original_one_slot[name],
                original_two_slot[name],
            )

        mutated_items = copy.deepcopy(self.world["public"]["items"])
        mutated_audit = copy.deepcopy(
            self.world["private"]["identity_slots_audit"]
        )
        mutated_edit = copy.deepcopy(
            self.world["private"]["identity_slots_edit"]
        )
        audit_by_slot = {row["slot_uid"]: row for row in mutated_audit}
        edit_by_slot = {row["slot_uid"]: row for row in mutated_edit}
        left = audit_by_slot[original_one_slot["slot_uid"]]
        right = audit_by_slot[original_two_slot["slot_uid"]]
        left_value = {
            name: left[name]
            for name in (
                "identity_uid",
                "downstream_canonical_value",
                "raw_surface",
            )
        }
        right_value = {
            name: right[name]
            for name in (
                "identity_uid",
                "downstream_canonical_value",
                "raw_surface",
            )
        }
        for name in left_value:
            left[name] = right_value[name]
            right[name] = left_value[name]
        for row in (left, right):
            row["bundle_uid"] = safe_slots._bundle_uid(
                self.record["world_uid"],
                row["seller_uid"],
                row["identity_uid"],
            )
        for audit_row, edit_row in (
            (left, edit_by_slot[left["slot_uid"]]),
            (right, edit_by_slot[right["slot_uid"]]),
        ):
            edit_row["downstream_canonical_value"] = audit_row[
                "downstream_canonical_value"
            ]
            edit_row["raw_surface"] = audit_row["raw_surface"]

        item_by_uid = {row["item_uid"]: row for row in mutated_items}
        for original, mutated in (
            (original_one_slot, left),
            (original_two_slot, right),
        ):
            item = item_by_uid[original["item_uid"]]
            start = int(original["start"])
            end = int(original["end"])
            description = str(item["description"])
            self.assertEqual(description[start:end], original["raw_surface"])
            item["description"] = (
                description[:start]
                + str(mutated["raw_surface"])
                + description[end:]
            )

        mutated_parsed = production.parse_observed_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            items=mutated_items,
        )
        mutated_redaction = production.redact_observed_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            template=self.template,
            sellers=self.world["public"]["sellers"],
            items=mutated_items,
            registry_profiles=self.registry_profiles,
            parsed_rows=mutated_parsed,
        )
        parser_audit = production.validate_parser_against_private_plan(
            self.policy,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            items=mutated_items,
            parsed_rows=mutated_parsed,
            identity_slots_audit=mutated_audit,
            noise_slots_audit=self.world["private"]["noise_slots_audit"],
            render_asts=self.world["private"]["render_asts"],
        )
        self.assertTrue(parser_audit["exact_rows_and_flags"])
        self.assertEqual(
            mutated_redaction["redacted_items"],
            self.processed["public"]["redacted_items"],
        )
        projected_slots, _ledger, _audit = safe_slots.project_safe_slots(
            self.policy,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            items=mutated_items,
            parsed_rows=mutated_parsed,
            identity_slots_edit=mutated_edit,
        )
        projected_by_slot = {
            row["slot_uid"]: row for row in projected_slots
        }
        self.assertEqual(
            projected_by_slot[left["slot_uid"]]["downstream_canonical_value"],
            original_two_slot["downstream_canonical_value"],
        )
        self.assertEqual(
            projected_by_slot[right["slot_uid"]][
                "downstream_canonical_value"
            ],
            original_one_slot["downstream_canonical_value"],
        )
        production.validate_redaction_against_private_plan(
            self.policy,
            mode=self.mode,
            split=self.split,
            template=self.template,
            sellers=self.world["public"]["sellers"],
            items=mutated_items,
            redacted_items=mutated_redaction["redacted_items"],
            parsed_rows=mutated_parsed,
            identity_slots_audit=mutated_audit,
            noise_slots_audit=self.world["private"]["noise_slots_audit"],
            render_asts=self.world["private"]["render_asts"],
            override_audit=self.world["private"]["override_audit"],
        )
        mutated_world = copy.deepcopy(self.world)
        mutated_world["public"]["items"] = mutated_items
        mutated_world["private"]["identity_slots_audit"] = mutated_audit
        mutated_world["private"]["identity_slots_edit"] = mutated_edit
        with self.assertRaisesRegex(
            common.ContractError,
            "smoke producer regeneration",
        ):
            smoke_regeneration.validate_producer_regeneration_match(
                self.policy,
                mode=self.mode,
                split=self.split,
                template=self.template,
                fixture=self.fixture,
                style_profile=self.style,
                world=mutated_world,
            )

    def test_observed_workers_are_input_permutation_invariant(self) -> None:
        parsed = production.parse_observed_world(
            self.policy,
            mode="development_smoke",
            split=self.record["split"],
            sellers=list(reversed(self.world["public"]["sellers"])),
            items=list(reversed(self.world["public"]["items"])),
        )
        self.assertEqual(parsed, self.parsed)
        redaction = production.redact_observed_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            template=self.template,
            sellers=list(reversed(self.world["public"]["sellers"])),
            items=list(reversed(self.world["public"]["items"])),
            registry_profiles=list(reversed(self.registry_profiles)),
            parsed_rows=list(reversed(self.parsed)),
        )
        baseline = production.redact_observed_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            template=self.template,
            sellers=self.world["public"]["sellers"],
            items=self.world["public"]["items"],
            registry_profiles=self.registry_profiles,
            parsed_rows=self.parsed,
        )
        self.assertEqual(
            redaction["redacted_items"], baseline["redacted_items"]
        )
        self.assertEqual(
            redaction["registry_hashes"], baseline["registry_hashes"]
        )

    def test_safe_slot_uids_are_independently_reconstructed(self) -> None:
        rows, ledger, audit = safe_slots.project_safe_slots(
            self.policy,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            items=self.world["public"]["items"],
            parsed_rows=self.parsed,
            identity_slots_edit=self.world["private"]["identity_slots_edit"],
        )
        private_by_slot = {
            row["slot_uid"]: row
            for row in self.world["private"]["identity_slots_audit"]
        }
        self.assertEqual(set(private_by_slot), {row["slot_uid"] for row in rows})
        for row in rows:
            private = private_by_slot[row["slot_uid"]]
            self.assertEqual(row["identity_uid"], private["identity_uid"])
            self.assertEqual(row["bundle_uid"], private["bundle_uid"])
        self.assertEqual(audit["safe_slot_count"], len(self.parsed))
        self.assertEqual(
            len({row["identity_uid"] for row in ledger}), len(ledger)
        )

    def test_identity33_is_all_pair_label_free_and_permutation_invariant(
        self,
    ) -> None:
        processed = production.process_world(
            self.policy,
            mode=self.mode,
            split=self.split,
            template=self.template,
            world=self.world,
        )
        occurrences = processed["public"]["history_safe_occurrences"]
        endpoints = self.world["public"]["complete_model_pair_endpoints"]
        item_index = [
            {
                "world_uid": row["world_uid"],
                "seller_uid": row["seller_uid"],
                "item_uid": row["item_uid"],
                "time_bucket": row["time_bucket"],
            }
            for row in self.world["public"]["items"]
        ]
        item_index.sort(
            key=lambda row: (
                row["world_uid"].encode("utf-8"),
                row["seller_uid"].encode("utf-8"),
                row["item_uid"].encode("utf-8"),
            )
        )
        attestation = production.build_history_projection_attestation(
            self.policy,
            mode=self.mode,
            split=self.split,
            world_uid=self.record["world_uid"],
            sellers=self.world["public"]["sellers"],
            items=self.world["public"]["items"],
            history_safe_occurrences=occurrences,
            history_item_index=item_index,
            parsed_rows=processed["private"]["parsed_identity_occurrences"],
            identity_slots_audit=self.world["private"][
                "identity_slots_audit"
            ],
            noise_slots_audit=self.world["private"]["noise_slots_audit"],
            render_asts=self.world["private"]["render_asts"],
        )
        rows, audit = history_features.build_identity33_all_pairs(
            self.policy,
            mode=self.mode,
            split=self.split,
            history_safe_occurrences=occurrences,
            history_item_index=item_index,
            projection_attestations=[attestation],
            complete_model_pair_endpoints=endpoints,
        )
        reversed_rows, reversed_audit = (
            history_features.build_identity33_all_pairs(
                self.policy,
                mode=self.mode,
                split=self.split,
                history_safe_occurrences=list(reversed(occurrences)),
                history_item_index=list(reversed(item_index)),
                projection_attestations=[attestation],
                complete_model_pair_endpoints=list(reversed(endpoints)),
            )
        )
        self.assertEqual(len(rows), 378)
        self.assertEqual(rows, reversed_rows)
        self.assertEqual(audit, reversed_audit)
        self.assertEqual(audit["feature_count"], 33)
        self.assertGreater(audit["zero_feature_pair_count"], 0)

        bad_occurrences = copy.deepcopy(occurrences)
        bad_occurrences[0]["label"] = 1
        with self.assertRaises(common.ContractError):
            history_features.build_identity33_all_pairs(
                self.policy,
                mode=self.mode,
                split=self.split,
                history_safe_occurrences=bad_occurrences,
                history_item_index=item_index,
                projection_attestations=[attestation],
                complete_model_pair_endpoints=endpoints,
            )
        bad_endpoints = copy.deepcopy(endpoints)
        bad_endpoints[0]["classification_bool"] = 1
        with self.assertRaises(common.ContractError):
            history_features.build_identity33_all_pairs(
                self.policy,
                mode=self.mode,
                split=self.split,
                history_safe_occurrences=occurrences,
                history_item_index=item_index,
                projection_attestations=[attestation],
                complete_model_pair_endpoints=bad_endpoints,
            )
        forged_occurrences = copy.deepcopy(occurrences)
        forged = dict(forged_occurrences[0])
        forged_target_seller = next(
            row["seller_uid"]
            for row in self.world["public"]["sellers"]
            if row["seller_uid"] != forged["seller_uid"]
        )
        forged_target_item = next(
            row
            for row in self.world["public"]["items"]
            if row["seller_uid"] == forged_target_seller
        )
        forged["seller_uid"] = forged_target_seller
        forged["item_uid"] = forged_target_item["item_uid"]
        forged["source_row_number"] = forged["item_uid"]
        forged["time_bucket"] = forged_target_item["time_bucket"]
        forged_occurrences.append(forged)
        with self.assertRaises(common.ContractError):
            history_features.build_identity33_all_pairs(
                self.policy,
                mode=self.mode,
                split=self.split,
                history_safe_occurrences=forged_occurrences,
                history_item_index=item_index,
                projection_attestations=[attestation],
                complete_model_pair_endpoints=endpoints,
            )
        with self.assertRaises(common.ContractError):
            production.build_history_projection_attestation(
                self.policy,
                mode=self.mode,
                split=self.split,
                world_uid=self.record["world_uid"],
                sellers=self.world["public"]["sellers"],
                items=self.world["public"]["items"],
                history_safe_occurrences=forged_occurrences,
                history_item_index=item_index,
                parsed_rows=processed["private"][
                    "parsed_identity_occurrences"
                ],
                identity_slots_audit=self.world["private"][
                    "identity_slots_audit"
                ],
                noise_slots_audit=self.world["private"][
                    "noise_slots_audit"
                ],
                render_asts=self.world["private"]["render_asts"],
            )
        with self.assertRaises(common.ContractError):
            production.build_history_projection_attestation(
                self.policy,
                mode="formal",
                split=self.split,
                world_uid=self.record["world_uid"],
                sellers=self.world["public"]["sellers"],
                items=self.world["public"]["items"],
                history_safe_occurrences=occurrences,
                history_item_index=item_index,
                parsed_rows=processed["private"][
                    "parsed_identity_occurrences"
                ],
                identity_slots_audit=self.world["private"][
                    "identity_slots_audit"
                ],
                noise_slots_audit=self.world["private"][
                    "noise_slots_audit"
                ],
                render_asts=self.world["private"]["render_asts"],
            )

    def test_m0_profiles_are_invariant_to_raw_identity_suffix(self) -> None:
        baseline_safe = self.processed["public"]["profile_safe_items"]
        baseline_profiles, _audit = profiles_mod.build_world_profiles(
            self.policy,
            mode=self.mode,
            split=self.split,
            sellers=self.world["public"]["sellers"],
            items=baseline_safe,
        )
        guards = text_renderer.context_guard_pool(self.template)
        mutated_raw = copy.deepcopy(self.world["public"]["items"])
        for row in mutated_raw:
            description = row["description"]
            if description:
                boundary, _count = production._context_guard_boundary(
                    description,
                    guards=guards,
                )
                prefix = (
                    description
                    if boundary is None
                    else description[:boundary]
                )
                selected = text_renderer.context_guard_sequence(
                    selector_uid=f"forged_{row['item_uid']}",
                    count=3,
                    template=self.template,
                )
                row["description"] = (
                    prefix
                    + selected[0]
                    + "伪造身份后缀123456789012"
                    + selected[1]
                    + "第二段伪造身份"
                    + selected[2]
                )
        counterfactual_safe = production.build_profile_safe_items(
            self.policy,
            items=mutated_raw,
            redacted_items=self.processed["public"]["redacted_items"],
        )
        counterfactual_profiles, _counterfactual_audit = (
            profiles_mod.build_world_profiles(
                self.policy,
                mode=self.mode,
                split=self.split,
                sellers=self.world["public"]["sellers"],
                items=counterfactual_safe,
            )
        )
        self.assertEqual(counterfactual_safe, baseline_safe)
        self.assertEqual(counterfactual_profiles, baseline_profiles)


if __name__ == "__main__":
    unittest.main()
