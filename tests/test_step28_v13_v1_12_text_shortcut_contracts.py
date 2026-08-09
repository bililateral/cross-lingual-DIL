from __future__ import annotations

import ast
import hashlib
import inspect
import json
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
from scipy import sparse
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.feature_extraction.text import HashingVectorizer


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_12_preceremony as preceremony
import step28_v13_v1_12_formal_common as formal
import step28_v13_common as common
import step28_v13_production_chain as production
import step28_v13_profiles as profiles
import step28_v13_structure as structure
import step28_v13_world_builder as world_builder
import step28_v13_v1_12_counterfactual_text as counterfactual
import step28_v13_v1_12_assignment_null as assignment_null
import step28_v13_v1_12_style_derangement as subject
import step28_v13_v1_12_text_shortcut_preflight as text_preflight
import step28_v13_v1_12_text_shortcut_runner as text_runner


SELLERS = tuple(f"seller-{index:02d}" for index in range(28))
KNOWN_PAIRS = (
    ("seller-00", "seller-19"),
    ("seller-01", "seller-17"),
    ("seller-02", "seller-20"),
    ("seller-03", "seller-21"),
    ("seller-04", "seller-06"),
    ("seller-05", "seller-09"),
    ("seller-06", "seller-18"),
    ("seller-07", "seller-23"),
    ("seller-08", "seller-03"),
    ("seller-09", "seller-12"),
    ("seller-10", "seller-27"),
    ("seller-11", "seller-05"),
    ("seller-12", "seller-00"),
    ("seller-13", "seller-04"),
    ("seller-14", "seller-01"),
    ("seller-15", "seller-22"),
    ("seller-16", "seller-07"),
    ("seller-17", "seller-14"),
    ("seller-18", "seller-10"),
    ("seller-19", "seller-13"),
    ("seller-20", "seller-26"),
    ("seller-21", "seller-24"),
    ("seller-22", "seller-25"),
    ("seller-23", "seller-15"),
    ("seller-24", "seller-02"),
    ("seller-25", "seller-08"),
    ("seller-26", "seller-16"),
    ("seller-27", "seller-11"),
)


class Step28V13V112TextShortcutContracts(unittest.TestCase):
    @staticmethod
    def _policy() -> dict[str, object]:
        path = (
            ROOT
            / "schema"
            / "step28_v13_v1_12_text_shortcut_audit_policy.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_policy_contract_and_self_hash_close(self) -> None:
        policy_path = (
            ROOT
            / "schema"
            / "step28_v13_v1_12_text_shortcut_audit_policy.json"
        )
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        claimed = policy.pop("canonical_self_hash")
        self.assertEqual(preceremony.canonical_sha256(policy), claimed)
        self.assertEqual(set(policy["authorizations"].values()), {False})
        contract_path = ROOT / policy["contract"]["path"]
        payload = contract_path.read_bytes()
        self.assertEqual(len(payload), policy["contract"]["size_bytes"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), policy["contract"]["sha256"])

    def test_text_policy_fails_closed_on_runtime_drift(self) -> None:
        patches = (
            mock.patch.object(
                text_preflight.platform, "python_version", return_value="0.0.0"
            ),
            mock.patch.object(text_preflight.np, "__version__", "0.0.0"),
            mock.patch.object(text_preflight.scipy, "__version__", "0.0.0"),
            mock.patch.object(text_preflight.sklearn, "__version__", "0.0.0"),
            mock.patch.object(
                text_preflight.unicodedata, "unidata_version", "0.0.0"
            ),
        )
        for index, patcher in enumerate(patches):
            with self.subTest(runtime_component=index), patcher:
                with self.assertRaises(text_preflight.TextShortcutAuditError):
                    text_preflight.load_text_audit_policy()

    def test_derangement_constants_match_machine_policy(self) -> None:
        contract = self._policy()["style_source_derangement"]
        self.assertEqual(subject.DOMAIN.decode("ascii"), contract["domain_ascii"])
        self.assertEqual(subject.FIELD_SEPARATOR.hex(), contract["separator_hex"])
        self.assertEqual(subject.SELLER_COUNT, contract["seller_count"])
        self.assertEqual(
            subject.MAXIMUM_ATTEMPTS,
            contract["attempt_counter"]["maximum_attempts"],
        )

    def test_known_vector_rejects_attempt_zero_and_accepts_attempt_one(self) -> None:
        seller_blob = subject._seller_set_blob(SELLERS)
        attempt_zero = subject._candidate_permutation(
            SELLERS,
            attempt_seed=subject._attempt_seed(
                split="train",
                world_uid="world-known-vector-0001",
                seller_blob=seller_blob,
                attempt=0,
            ),
        )
        self.assertEqual(
            [target for target, source in zip(SELLERS, attempt_zero) if target == source],
            ["seller-11"],
        )

        observed = subject.build_style_source_derangement(
            split="train",
            world_uid="world-known-vector-0001",
            seller_uids=SELLERS,
        )
        self.assertEqual(observed.attempt, 1)
        self.assertEqual(
            observed.seller_set_sha256,
            "af9b2383ec6e4185c96078dbba129e1d7404cc2e39d68dc185a2b8216de81bc1",
        )
        self.assertEqual(
            observed.mapping_sha256,
            "92aaea142d0490b06f73886ead373dcd64b51a843017687772057b7b36a10567",
        )
        self.assertEqual(observed.target_source_pairs, KNOWN_PAIRS)

    def test_input_order_does_not_change_mapping(self) -> None:
        forward = subject.build_style_source_derangement(
            split="development", world_uid="world-order", seller_uids=SELLERS
        )
        reverse = subject.build_style_source_derangement(
            split="development",
            world_uid="world-order",
            seller_uids=tuple(reversed(SELLERS)),
        )
        self.assertEqual(forward, reverse)

    def test_split_and_world_are_domain_separators(self) -> None:
        base = subject.build_style_source_derangement(
            split="train", world_uid="world-a", seller_uids=SELLERS
        )
        changed_split = subject.build_style_source_derangement(
            split="development", world_uid="world-a", seller_uids=SELLERS
        )
        changed_world = subject.build_style_source_derangement(
            split="train", world_uid="world-b", seller_uids=SELLERS
        )
        self.assertNotEqual(base.target_source_pairs, changed_split.target_source_pairs)
        self.assertNotEqual(base.target_source_pairs, changed_world.target_source_pairs)

    def test_mapping_is_a_fixed_point_free_bijection(self) -> None:
        observed = subject.build_style_source_derangement(
            split="audit_a", world_uid="world-bijection", seller_uids=SELLERS
        )
        mapping = observed.as_mapping()
        self.assertEqual(set(mapping), set(SELLERS))
        self.assertEqual(set(mapping.values()), set(SELLERS))
        self.assertTrue(all(target != source for target, source in mapping.items()))

    def test_constructor_surface_cannot_receive_private_attributes(self) -> None:
        signature = inspect.signature(subject.build_style_source_derangement)
        self.assertEqual(
            tuple(signature.parameters), ("split", "world_uid", "seller_uids")
        )
        source = Path(subject.__file__).read_text(encoding="utf-8")
        forbidden_inputs = self._policy()["style_source_derangement"][
            "forbidden_constructor_inputs"
        ]
        for forbidden in forbidden_inputs:
            self.assertNotIn(forbidden, source)
        imports = set()
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imports.add(str(node.module).split(".", 1)[0])
        self.assertEqual(imports, {"__future__", "hashlib", "dataclasses", "typing"})

    def test_randbelow_rejects_out_of_range_block_and_consumes_it(self) -> None:
        stream = subject._Sha256CounterStream(bytes(range(32)))
        with mock.patch.object(
            stream,
            "_next_uint256",
            side_effect=[subject.UINT256_SIZE - 1, 5],
        ) as next_uint256:
            self.assertEqual(stream.randbelow(3), 2)
        self.assertEqual(next_uint256.call_count, 2)

    def test_sha256_stream_uses_successive_uint64_big_endian_counters(self) -> None:
        seed = bytes(range(32))
        stream = subject._Sha256CounterStream(seed)
        expected_zero = int.from_bytes(
            hashlib.sha256(seed + (0).to_bytes(8, "big")).digest(), "big"
        )
        expected_one = int.from_bytes(
            hashlib.sha256(seed + (1).to_bytes(8, "big")).digest(), "big"
        )
        self.assertEqual(stream._next_uint256(), expected_zero)
        self.assertEqual(stream._counter, 1)
        self.assertEqual(stream._next_uint256(), expected_one)
        self.assertEqual(stream._counter, 2)

    def test_invalid_inputs_fail_closed(self) -> None:
        invalid_cases = (
            {"split": "unknown", "world_uid": "world", "seller_uids": SELLERS},
            {"split": "train", "world_uid": "", "seller_uids": SELLERS},
            {"split": "train", "world_uid": "world", "seller_uids": SELLERS[:-1]},
            {
                "split": "train",
                "world_uid": "world",
                "seller_uids": SELLERS[:-1] + (SELLERS[0],),
            },
            {
                "split": "train",
                "world_uid": "world",
                "seller_uids": SELLERS[:-1] + (1,),
            },
            {"split": "train", "world_uid": "world", "seller_uids": "not-a-list"},
        )
        for kwargs in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(subject.StyleDerangementError):
                    subject.build_style_source_derangement(**kwargs)

    def test_word12_tokenizer_and_template_mask_are_frozen(self) -> None:
        self.assertEqual(
            text_preflight.word12_tokens("AbC12，中文-Ｆ９ xY"),
            ["abc12", "中", "文", "xy"],
        )
        self.assertEqual(
            text_preflight.template_mask("Ab中９-!\n"),
            "字字字数-!\n",
        )

    def test_exact_derangement_completion_count_and_relation_extremes(self) -> None:
        self.assertEqual(
            assignment_null.derangement_count(28),
            112162153835443422680893595673,
        )
        all_same = {seller: "same" for seller in SELLERS}
        all_unique = {seller: seller for seller in SELLERS}
        self.assertEqual(
            assignment_null._pair_relation_expectation(
                left_target=SELLERS[0],
                right_target=SELLERS[1],
                attribute_by_seller=all_same,
            ),
            1.0,
        )
        self.assertEqual(
            assignment_null._pair_relation_expectation(
                left_target=SELLERS[0],
                right_target=SELLERS[1],
                attribute_by_seller=all_unique,
            ),
            0.0,
        )

    def test_nontrivial_assignment_null_matches_direct_completion_enumeration(self) -> None:
        attribute = {
            seller: (
                "A"
                if index in {0, 1, 2}
                else "B"
                if index in {3, 4, 5, 6}
                else f"U{index}"
            )
            for index, seller in enumerate(SELLERS)
        }
        left, right = SELLERS[0], SELLERS[1]
        seller_set = set(SELLERS)
        reference_numerator = 0
        reference_denominator = 0
        for source_left in SELLERS:
            for source_right in SELLERS:
                if (
                    source_left == left
                    or source_right == right
                    or source_left == source_right
                ):
                    continue
                available_diagonal_count = len(
                    (seller_set - {left, right})
                    & (seller_set - {source_left, source_right})
                )
                weight = sum(
                    (-1) ** selected
                    * math.comb(available_diagonal_count, selected)
                    * math.factorial(26 - selected)
                    for selected in range(available_diagonal_count + 1)
                )
                reference_denominator += weight
                if attribute[source_left] == attribute[source_right]:
                    reference_numerator += weight
        self.assertEqual(
            reference_denominator, assignment_null.derangement_count(28)
        )
        self.assertEqual(
            assignment_null._pair_relation_expectation(
                left_target=left,
                right_target=right,
                attribute_by_seller=attribute,
            ),
            reference_numerator / reference_denominator,
        )
        self.assertEqual(
            assignment_null._seller_relation_expectation(
                target=left, attribute_by_seller=attribute
            ),
            2.0 / 27.0,
        )

    def test_sparse_dot_and_surface_feature_numeric_goldens(self) -> None:
        left = sparse.csr_matrix(
            np.asarray([[0.0, 0.5, 0.0, 0.25], [0.0, 0.0, 0.0, 0.0]])
        )
        right = sparse.csr_matrix(
            np.asarray([[0.0, 0.2, 0.0, 0.4], [1.0, 0.0, 0.0, 0.0]])
        )
        self.assertTrue(
            np.array_equal(
                text_preflight._rowwise_sorted_sparse_dot(left, right),
                np.asarray([0.2, 0.0], dtype=np.float64),
            )
        )
        surface = text_preflight._surface_pair_features(
            ["\t\n\v\f\r ９²，!", ""],
            field_name="golden",
            left_indices=np.asarray([0], dtype=np.intp),
            right_indices=np.asarray([1], dtype=np.intp),
        )
        expected = {
            "codepoint_length_absdiff__golden": 10.0,
            "codepoint_length_sum__golden": 10.0,
            "newline_count_absdiff__golden": 1.0,
            "newline_count_sum__golden": 1.0,
            "unicode_punctuation_count_absdiff__golden": 2.0,
            "unicode_punctuation_count_sum__golden": 2.0,
            "ascii_whitespace_count_absdiff__golden": 6.0,
            "ascii_whitespace_count_sum__golden": 6.0,
            "unicode_decimal_digit_count_absdiff__golden": 1.0,
            "unicode_decimal_digit_count_sum__golden": 1.0,
            "empty_both__golden": 0.0,
            "empty_xor__golden": 1.0,
        }
        self.assertEqual(set(surface), set(expected))
        for name, value in expected.items():
            self.assertEqual(surface[name].tolist(), [value])

    def test_tree_fold_and_bootstrap_statistics_match_frozen_contract(self) -> None:
        policy = text_preflight.load_text_audit_policy()
        tree = text_preflight._gradient_tree(policy)
        expected_tree = dict(policy["visible_attack"]["gradient_tree"])
        expected_tree.pop("implementation")
        expected_tree.pop("implicit_defaults_allowed")
        self.assertEqual(tree.get_params(deep=False), expected_tree)

        worlds = tuple(f"world-{index:03d}" for index in range(500))
        folds = text_preflight.fold_by_world(
            worlds,
            seed=policy["visible_attack"]["fold_seed"],
            fold_count=5,
        )
        self.assertEqual(Counter(folds.values()), Counter({index: 100 for index in range(5)}))
        self.assertEqual(
            folds,
            text_preflight.fold_by_world(
                tuple(reversed(worlds)),
                seed=policy["visible_attack"]["fold_seed"],
                fold_count=5,
            ),
        )
        self.assertEqual(
            preceremony.canonical_sha256(folds),
            "a58e0b783c39db97078e2bc965b88da9d2f21a995a08e097ec25a2079da5359f",
        )

        metric_drift = json.loads(json.dumps(policy))
        metric_drift["bootstrap"]["point_metrics"]["average_precision"][
            "kwargs"
        ]["pos_label"] = 0
        with self.assertRaises(text_preflight.TextShortcutAuditError):
            text_preflight._score_metrics(
                metric_drift,
                np.asarray([0, 1], dtype=np.int8),
                np.asarray([0.0, 1.0], dtype=np.float64),
            )
        quantile_drift = json.loads(json.dumps(policy))
        quantile_drift["bootstrap"]["quantile_method"] = "linear"
        with self.assertRaises(text_preflight.TextShortcutAuditError):
            text_preflight._bootstrap_upper(
                quantile_drift, np.asarray([0.5, 0.6], dtype=np.float64)
            )

        labels = np.tile(
            np.r_[np.ones(20, dtype=np.int8), np.zeros(352, dtype=np.int8)],
            500,
        )
        row_world = np.repeat(np.arange(500, dtype=np.int16), 372)
        multiplicities = np.ones((2, 500), dtype=np.int16)
        tied_scores = np.zeros(len(labels), dtype=np.float64)
        tied_auc, tied_ap = text_preflight.bootstrap_rank_metrics(
            labels=labels,
            scores=tied_scores,
            multiplicities=multiplicities,
            row_world=row_world,
        )
        self.assertTrue(np.array_equal(tied_auc, np.asarray([0.5, 0.5])))
        self.assertTrue(
            np.array_equal(tied_ap, np.asarray([5.0 / 93.0, 5.0 / 93.0]))
        )

        scores = np.linspace(0.0, 1.0, len(labels), dtype=np.float64)
        observed_auc, observed_ap = text_preflight.bootstrap_rank_metrics(
            labels=labels,
            scores=scores,
            multiplicities=np.ones((1, 500), dtype=np.int16),
            row_world=row_world,
        )
        raw_auc = float(roc_auc_score(labels, scores))
        self.assertAlmostEqual(observed_auc[0], max(raw_auc, 1.0 - raw_auc), places=12)
        self.assertAlmostEqual(
            observed_ap[0], float(average_precision_score(labels, scores)), places=12
        )

    def test_bootstrap_draw_and_model_family_are_deterministic_and_world_grouped(
        self,
    ) -> None:
        policy = text_preflight.load_text_audit_policy()
        test_policy = json.loads(json.dumps(policy))
        test_policy["bootstrap"]["replicates"] = 3
        worlds = tuple(f"world-{index:03d}" for index in range(500))
        first, first_rows, first_hash = text_preflight.draw_world_multiplicities(
            policy=test_policy,
            world_uids=worlds,
            split="development",
        )
        replay, replay_rows, replay_hash = text_preflight.draw_world_multiplicities(
            policy=test_policy,
            world_uids=tuple(reversed(worlds)),
            split="development",
        )
        self.assertTrue(np.array_equal(first, replay))
        self.assertEqual(first_hash, replay_hash)
        self.assertEqual(first.shape, (3, 500))
        self.assertTrue(np.array_equal(np.sum(first, axis=1), np.asarray([500] * 3)))
        self.assertTrue(np.array_equal(first_rows, np.arange(500, dtype=np.int16)))
        self.assertTrue(
            np.array_equal(replay_rows, np.arange(499, -1, -1, dtype=np.int16))
        )

        row_world_uids = tuple(
            world_uid for world_uid in worlds for _row in range(372)
        )
        text_draws, row_world, text_hash = (
            text_preflight.draw_world_multiplicities(
                policy=policy,
                world_uids=row_world_uids,
                split="development",
                seed_field="text_attack_seed",
            )
        )
        assignment_draws, _assignment_rows, assignment_hash = (
            text_preflight.draw_world_multiplicities(
                policy=policy,
                world_uids=worlds,
                split="development",
                seed_field="assignment_seed",
            )
        )
        self.assertEqual(
            text_hash,
            "755f6dfb7a5c1d12842e9702b7a3949ebcd0339fa10f716e3fa0dee7e7a49a51",
        )
        self.assertEqual(
            assignment_hash,
            "dfad1a38ab9ead6befc977af24e545d49e1de8e49cf4f633c12c21f3cfc059dd",
        )
        self.assertEqual(text_draws.shape, (9999, 500))
        self.assertEqual(assignment_draws.shape, (9999, 500))
        weighted_labels = np.tile(
            np.r_[np.ones(20, dtype=np.int8), np.zeros(352, dtype=np.int8)],
            500,
        )
        tied_scores = np.tile(
            np.repeat(np.asarray([0.0, 0.25, 0.5, 0.75]), 93),
            500,
        )
        observed_auc, observed_ap = text_preflight.bootstrap_rank_metrics(
            labels=weighted_labels,
            scores=tied_scores,
            multiplicities=text_draws[:1],
            row_world=row_world,
        )
        sample_weight = text_draws[0][row_world]
        raw_weighted_auc = float(
            roc_auc_score(
                weighted_labels,
                tied_scores,
                sample_weight=sample_weight,
            )
        )
        weighted_ap = float(
            average_precision_score(
                weighted_labels,
                tied_scores,
                sample_weight=sample_weight,
            )
        )
        self.assertAlmostEqual(
            observed_auc[0],
            max(raw_weighted_auc, 1.0 - raw_weighted_auc),
            places=12,
        )
        self.assertAlmostEqual(observed_ap[0], weighted_ap, places=12)

        generator = np.random.Generator(np.random.PCG64DXSM(20260808))
        train_worlds = tuple(
            f"fit-world-{world_index:02d}"
            for world_index in range(10)
            for _row in range(4)
        )
        labels = np.tile(np.asarray([1, 0, 0, 0], dtype=np.int8), 10)
        x_train = generator.normal(size=(40, 2)).astype(np.float64)
        x_development = generator.normal(size=(12, 2)).astype(np.float64)
        oof, development, audit = text_preflight._fit_one_model_family(
            policy=policy,
            x_train=x_train,
            y_train=labels,
            train_worlds=train_worlds,
            x_development=x_development,
        )
        self.assertEqual(set(oof), {"logistic_l2", "gradient_tree"})
        self.assertEqual(set(development), {"logistic_l2", "gradient_tree"})
        self.assertTrue(all(value.shape == (40,) for value in oof.values()))
        self.assertTrue(
            all(value.shape == (12,) for value in development.values())
        )
        self.assertTrue(
            all(np.all(np.isfinite(value)) for value in (*oof.values(), *development.values()))
        )
        self.assertEqual(len(audit["folds"]), 5)
        self.assertTrue(
            all(
                row["fit_world_count"] == 8
                and row["held_out_world_count"] == 2
                and row["fit_row_count"] == 32
                and row["held_out_row_count"] == 8
                for row in audit["folds"]
            )
        )
        replay_oof, replay_development, replay_audit = (
            text_preflight._fit_one_model_family(
                policy=policy,
                x_train=x_train,
                y_train=labels,
                train_worlds=train_worlds,
                x_development=x_development + 17.0,
            )
        )
        self.assertEqual(audit, replay_audit)
        for name in oof:
            self.assertTrue(np.array_equal(oof[name], replay_oof[name]))
        self.assertFalse(
            np.array_equal(
                development["logistic_l2"],
                replay_development["logistic_l2"],
            )
        )

    def test_visible_family_gate_uses_replicatewise_family_maximum(self) -> None:
        policy = text_preflight.load_text_audit_policy()
        test_policy = json.loads(json.dumps(policy))
        test_policy["bootstrap"]["replicates"] = 100
        labels = np.tile(np.asarray([1, 0], dtype=np.int8), 500)

        def split(name: str) -> text_preflight.SplitVisibleAttackMatrices:
            world_uids = tuple(
                f"{name}-world-{world_index:03d}"
                for world_index in range(500)
                for _row in range(2)
            )
            views = {
                view_name: np.arange(1000, dtype=np.float64)[:, None]
                for view_name in test_policy["visible_attack"]["views"]
            }
            return text_preflight.SplitVisibleAttackMatrices(
                split=name,
                world_uids=world_uids,
                pair_uids=tuple(),
                labels=labels.copy(),
                views=views,
                feature_names_by_view={
                    view_name: ("probe",) for view_name in views
                },
            )

        train = split("train")
        development = split("development")
        score = np.linspace(0.0, 1.0, 1000, dtype=np.float64)

        def fake_fit(**_kwargs: object) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, object]]:
            return (
                {"logistic_l2": score, "gradient_tree": score[::-1]},
                {"logistic_l2": score, "gradient_tree": score[::-1]},
                {"frozen": True},
            )

        auc_arrays = [np.full(100, 0.51, dtype=np.float64) for _ in range(6)]
        auc_arrays[0][:4] = 0.90
        auc_arrays[1][4:8] = 0.80
        baseline = 5.0 / 93.0
        ap_arrays = [
            np.full(100, baseline + 0.001, dtype=np.float64) for _ in range(6)
        ]
        ap_arrays[0][:4] = baseline + 0.05
        ap_arrays[1][4:8] = baseline + 0.04
        bootstrap_results = iter(zip(auc_arrays, ap_arrays, strict=True))
        multiplicities = np.ones((100, 500), dtype=np.int16)
        row_world = np.repeat(np.arange(500, dtype=np.int16), 2)
        with mock.patch.object(
            text_preflight, "_fit_one_model_family", side_effect=fake_fit
        ) as fit_mock, mock.patch.object(
            text_preflight,
            "draw_world_multiplicities",
            return_value=(multiplicities, row_world, "draw-hash"),
        ), mock.patch.object(
            text_preflight,
            "bootstrap_rank_metrics",
            side_effect=lambda **_kwargs: next(bootstrap_results),
        ) as bootstrap_mock:
            result = text_preflight.evaluate_visible_attack_family(
                policy=test_policy,
                train=train,
                development=development,
            )
        self.assertEqual(fit_mock.call_count, 3)
        self.assertEqual(bootstrap_mock.call_count, 6)
        self.assertEqual(
            result["bootstrap_95_upper_family_max_symmetric_auc"], 0.80
        )
        self.assertAlmostEqual(
            result["bootstrap_95_upper_family_max_average_precision_uplift"],
            0.04,
            places=15,
        )
        self.assertFalse(all(result["hard_gates"].values()))

    def test_design_split_builder_aligns_visible_and_assignment_rows(self) -> None:
        split = text_preflight.build_design_split_attack_data(
            split="train",
            world_count=3,
            progress_every=0,
        )
        self.assertEqual(split.visible.split, "train")
        self.assertEqual(len(set(split.visible.world_uids)), 3)
        self.assertEqual(len(split.visible.pair_uids), 1116)
        self.assertEqual(int(np.sum(split.visible.labels)), 60)
        self.assertEqual(split.assignment_observed.shape, (1116, 10))
        self.assertEqual(split.assignment_expected.shape, (1116, 10))
        self.assertEqual(split.assignment_world_uids, split.visible.world_uids)
        self.assertEqual(len(split.world_audits), 3)
        self.assertEqual(
            tuple(row["world_uid"] for row in split.world_audits),
            tuple(sorted(set(split.visible.world_uids), key=lambda value: value.encode("utf-8"))),
        )
        self.assertTrue(
            all(
                row["parser_exact_rows_and_flags"] is True
                and row["planned_identity_surface_residue_count"] == 0
                and row["seller_profile_count"] == 28
                for row in split.world_audits
            )
        )
        self.assertTrue(np.all(np.isfinite(split.assignment_observed)))
        self.assertTrue(np.all(np.isfinite(split.assignment_expected)))

    def test_assignment_gate_uses_ten_relation_family_maximum_and_residuals(
        self,
    ) -> None:
        policy = text_preflight.load_text_audit_policy()
        test_policy = json.loads(json.dumps(policy))
        test_policy["bootstrap"]["replicates"] = 100
        relation_names = tuple(
            test_policy["assignment_null_audit"]["pair_gate_relations_in_order"]
        )
        world_uids = tuple(
            f"development-world-{world_index:03d}"
            for world_index in range(500)
            for _row in range(372)
        )
        labels = np.tile(
            np.r_[np.ones(20, dtype=np.int8), np.zeros(352, dtype=np.int8)],
            500,
        )
        visible = text_preflight.SplitVisibleAttackMatrices(
            split="development",
            world_uids=world_uids,
            pair_uids=tuple(),
            labels=labels,
            views={},
            feature_names_by_view={},
        )
        development = text_preflight.DesignSplitAttackData(
            visible=visible,
            assignment_relation_names=relation_names,
            assignment_observed=np.zeros((500 * 372, 10), dtype=np.float64),
            assignment_expected=np.full(
                (500 * 372, 10), 0.25, dtype=np.float64
            ),
            assignment_world_uids=world_uids,
            world_audits=tuple(),
        )
        auc_arrays = [np.full(100, 0.51, dtype=np.float64) for _ in range(10)]
        auc_arrays[0][:4] = 0.90
        auc_arrays[1][4:8] = 0.80
        bootstrap_results = iter(auc_arrays)
        multiplicities = np.ones((100, 500), dtype=np.int16)
        row_world = np.repeat(np.arange(500, dtype=np.int16), 372)
        with mock.patch.object(
            text_preflight,
            "draw_world_multiplicities",
            return_value=(multiplicities, row_world, "assignment-draw-hash"),
        ), mock.patch.object(
            text_preflight,
            "bootstrap_rank_metrics",
            side_effect=lambda **_kwargs: (
                next(bootstrap_results),
                np.zeros(100, dtype=np.float64),
            ),
        ) as bootstrap_mock:
            result = text_preflight.evaluate_assignment_null_gate(
                policy=test_policy,
                development=development,
            )
        self.assertEqual(bootstrap_mock.call_count, 10)
        self.assertEqual(
            result["bootstrap_95_upper_family_max_symmetric_auc"], 0.80
        )
        self.assertFalse(all(result["hard_gates"].values()))
        for metrics in result["relation_metrics"].values():
            self.assertEqual(metrics["residual_mean_positive"], -0.25)
            self.assertEqual(metrics["residual_mean_negative"], -0.25)
        drifted = text_preflight.DesignSplitAttackData(
            visible=visible,
            assignment_relation_names=tuple(reversed(relation_names)),
            assignment_observed=development.assignment_observed,
            assignment_expected=development.assignment_expected,
            assignment_world_uids=world_uids,
            world_audits=tuple(),
        )
        with self.assertRaises(text_preflight.TextShortcutAuditError):
            text_preflight.evaluate_assignment_null_gate(
                policy=test_policy,
                development=drifted,
            )

    def test_text_fast_path_matches_full_identity_remapped_materialization(self) -> None:
        parity = text_preflight.validate_text_fast_full_parity()
        self.assertEqual(parity["status"], "PASS_TEXT_FAST_FULL_REDACTED_PARITY")
        self.assertTrue(parity["design_only"])
        self.assertFalse(parity["formal_seed_or_key_access"])
        self.assertEqual(parity["world_count"], 2)
        self.assertEqual(
            [row["assignment_sha256"] for row in parity["rows"]],
            [
                "4ee7442a7dd3ac56f73c110d91e2d6b3ef65711f20666d53fdc6a64ffb0ef934",
                "646b65ee4ac3e6bf985c88db364cfd76b2579189aa23fa70b7fc10f6dcc02780",
            ],
        )
        self.assertEqual(
            [row["matrix_sha256_by_view"] for row in parity["rows"]],
            [
                {
                    "cf_full": "548489d8ddd9768228d74df51c21dfb8303f3f7be6b3fc1c1f89113b86057fd6",
                    "cf_topic": "5b49c3b0394d15c007aac01404b6f04ac1a34b46e3c41c8614b132e275f804f9",
                    "cf_template_surface": "b60409db44055f52a81a751e7039f63d54a792ca69ab4b242ea6605f9bc6a0d3",
                },
                {
                    "cf_full": "2dc2514d60bc3dd04210f0da85375f8ea53b6fa2cbb2b596c67947ca2d9e85ab",
                    "cf_topic": "abbd7e17d51c5bb5cdc71bc19147340436cf08f6b764fc6f1e4c79d98f22b4b1",
                    "cf_template_surface": "b95e29f56e5e5c8a19a86d3a76ffc884b13d9d5eaa8e50e0e08b88fa27c18e0b",
                },
            ],
        )

    def test_original_projection_matches_full_identity_remapped_materialization(
        self,
    ) -> None:
        parity = text_runner.validate_original_fast_full_parity()
        self.assertEqual(
            parity["status"], "PASS_ORIGINAL_FAST_FULL_REDACTED_PARITY"
        )
        self.assertTrue(parity["design_only"])
        self.assertFalse(parity["formal_seed_or_key_access"])
        self.assertEqual(
            [row["score_matrix_sha256"] for row in parity["rows"]],
            [
                "21bcfde189fd6bad76439e8ae7185badb1f6d78b80ee319de01e1270a602ce53",
                "e564476ab75ff323a4ae0d1357e71c6f46271583ed6c7e75aaa7be1f004d3fd5",
            ],
        )
        self.assertEqual(
            [row["redacted_items_sha256"] for row in parity["rows"]],
            [
                "859625a34a018b751cb51574d62023f579bef447f3225e9e44856fbb952408ce",
                "159974e48945cb7feb060c693bc90a1deb7fa7bd3ef70df78fd9bec7aa16ba0e",
            ],
        )
        self.assertEqual(
            [row["seller_profiles_sha256"] for row in parity["rows"]],
            [
                "1d300c62ee8552d8cf71eefed3088926a0a42ac6ba45fdce3aba5fbd99c6793e",
                "66582c2d070a7bc37d0076c814110dafede8255c58a59816cbac0f451156c688",
            ],
        )

    def test_original_three_world_splits_are_isolated_and_descriptive_only(
        self,
    ) -> None:
        train = text_runner.build_original_design_split(
            split="train", world_count=3, progress_every=0
        )
        development = text_runner.build_original_design_split(
            split="development", world_count=3, progress_every=0
        )
        result = text_runner.evaluate_original_descriptive_attacks(
            policy=text_preflight.load_text_audit_policy(),
            train=train,
            development=development,
        )
        self.assertEqual(
            result["status"],
            "PASS_DESIGN_ORIGINAL_TEXT_ISOLATION_DESCRIPTIVE_ONLY",
        )
        self.assertEqual(len(train.pair_uids), 3 * 378)
        self.assertEqual(len(development.pair_uids), 3 * 378)
        self.assertEqual(int(np.sum(train.labels)), 60)
        self.assertEqual(int(np.sum(development.labels)), 60)
        self.assertEqual(len(train.feature_names), 12)
        self.assertEqual(
            Counter(development.strata),
            Counter(
                {
                    "positive": 60,
                    "exact_title_clone_target": 6,
                    "high_semantic_similarity_target": 12,
                    "other_negative": 1056,
                }
            ),
        )
        self.assertEqual(
            set(result["cross_split_exact_intersection_counts"].values()), {0}
        )
        self.assertEqual(len(train.inventory_sets["item_uid"]), 308)
        self.assertEqual(len(development.inventory_sets["item_uid"]), 282)
        self.assertFalse(
            result["absolute_near_duplicate_threshold_applied"]
        )
        self.assertEqual(
            result["formal_identity_hash_intersection_audit"]["status"],
            "DEFERRED_UNTIL_FORMAL_IDENTITIES_EXIST",
        )

    def test_final_runner_preserves_design_only_authorization_boundary(self) -> None:
        policy = text_preflight.load_text_audit_policy()
        empty_visible = text_preflight.SplitVisibleAttackMatrices(
            split="train",
            world_uids=tuple(),
            pair_uids=tuple(),
            labels=np.asarray([], dtype=np.int8),
            views={},
            feature_names_by_view={},
        )
        empty_counterfactual = text_preflight.DesignSplitAttackData(
            visible=empty_visible,
            assignment_relation_names=tuple(),
            assignment_observed=np.empty((0, 0), dtype=np.float64),
            assignment_expected=np.empty((0, 0), dtype=np.float64),
            assignment_world_uids=tuple(),
            world_audits=tuple(),
        )
        empty_original = text_runner.SplitOriginalDiagnostic(
            split="train",
            world_uids=tuple(),
            pair_uids=tuple(),
            labels=np.asarray([], dtype=np.int8),
            feature_names=tuple(),
            scores=np.empty((0, 0), dtype=np.float64),
            strata=tuple(),
            inventory_sets={},
            visible_leakage_count=0,
            world_audits=tuple(),
        )
        with mock.patch.object(
            text_preflight,
            "validate_text_fast_full_parity",
            return_value={"status": "PASS_TEXT_FAST_FULL_REDACTED_PARITY"},
        ), mock.patch.object(
            text_runner,
            "validate_original_fast_full_parity",
            return_value={"status": "PASS_ORIGINAL_FAST_FULL_REDACTED_PARITY"},
        ), mock.patch.object(
            text_preflight,
            "build_design_split_attack_data",
            side_effect=[empty_counterfactual, empty_counterfactual],
        ), mock.patch.object(
            text_runner,
            "_assignment_split_summary",
            return_value={"status": "DESCRIPTIVE"},
        ), mock.patch.object(
            text_preflight,
            "evaluate_assignment_null_gate",
            return_value={"status": "PASS_ASSIGNMENT_NULL_GATES"},
        ), mock.patch.object(
            text_preflight,
            "evaluate_visible_attack_family",
            return_value={"status": "PASS_VISIBLE_TEXT_SHORTCUT_GATES"},
        ), mock.patch.object(
            text_runner,
            "build_original_design_split",
            side_effect=[empty_original, empty_original],
        ), mock.patch.object(
            text_runner,
            "evaluate_original_descriptive_attacks",
            return_value={
                "status": "PASS_DESIGN_ORIGINAL_TEXT_ISOLATION_DESCRIPTIVE_ONLY"
            },
        ), mock.patch.object(
            text_runner,
            "_rowwise_audit_receipt",
            return_value={
                "status": "PASS_DESIGN_WORLDS_ROW_BY_ROW_RECOMPUTED_IN_MEMORY"
            },
        ):
            receipt = text_runner.run_design_preflight(
                run_id="v13_training_ready_v1_12_cleanroom_20260803",
                progress_every=10,
            )
        self.assertEqual(
            receipt["status"],
            "PASS_DESIGN_TEXT_SHORTCUT_PREFLIGHT_NO_FORMAL_AUTHORIZATION",
        )
        self.assertEqual(
            receipt["formal_authorizations_after_preflight"],
            policy["authorizations"],
        )
        self.assertEqual(set(policy["authorizations"].values()), {False})
        self.assertEqual(receipt["formal_dataset_rows_produced"], 0)
        self.assertEqual(receipt["formal_dataset_rows_audited"], 0)
        self.assertFalse(receipt["model_training_authorized"])

    def test_registered_gate_failure_stops_all_later_stages(self) -> None:
        train_counterfactual = mock.Mock(
            world_audits=({"world_uid": "train-probe"},)
        )
        development_counterfactual = mock.Mock(
            world_audits=({"world_uid": "development-probe"},)
        )
        with mock.patch.object(
            text_preflight,
            "validate_text_fast_full_parity",
            return_value={"status": "PASS_TEXT_FAST_FULL_REDACTED_PARITY"},
        ), mock.patch.object(
            text_runner,
            "validate_original_fast_full_parity",
            return_value={"status": "PASS_ORIGINAL_FAST_FULL_REDACTED_PARITY"},
        ), mock.patch.object(
            text_preflight,
            "build_design_split_attack_data",
            side_effect=[train_counterfactual, development_counterfactual],
        ), mock.patch.object(
            text_runner,
            "_assignment_split_summary",
            return_value={"status": "DESCRIPTIVE"},
        ), mock.patch.object(
            text_preflight,
            "evaluate_assignment_null_gate",
            return_value={"status": "FAIL_ASSIGNMENT_NULL_GATES", "value": 0.9},
        ), mock.patch.object(
            text_preflight, "evaluate_visible_attack_family"
        ) as visible_mock, mock.patch.object(
            text_runner, "build_original_design_split"
        ) as original_mock:
            receipt = text_runner.run_design_preflight(
                run_id="v13_training_ready_v1_12_cleanroom_20260803",
                progress_every=10,
            )
        self.assertEqual(
            receipt["status"],
            "FAIL_DESIGN_TEXT_SHORTCUT_PREFLIGHT_REGISTERED_GATE",
        )
        self.assertEqual(
            receipt["failure_stage"], "assignment_development_hard_gate"
        )
        self.assertEqual(receipt["failure_type"], "registered_gate_failure")
        self.assertEqual(
            set(receipt["counterfactual_world_audit_sha256"]),
            {"train", "development"},
        )
        self.assertTrue(
            all(
                preceremony.HEX_SHA256_RE.fullmatch(value)
                for value in receipt[
                    "counterfactual_world_audit_sha256"
                ].values()
            )
        )
        self.assertEqual(
            receipt["counterfactual_world_audit_sha256"]["train"],
            preceremony.canonical_sha256(train_counterfactual.world_audits),
        )
        self.assertEqual(
            receipt["counterfactual_world_audit_sha256"]["development"],
            preceremony.canonical_sha256(
                development_counterfactual.world_audits
            ),
        )
        self.assertEqual(
            set(receipt["formal_authorizations_after_failure"].values()),
            {False},
        )
        visible_mock.assert_not_called()
        original_mock.assert_not_called()

    def test_rowwise_receipt_requires_all_one_thousand_world_audits(self) -> None:
        def counterfactual(split: str) -> text_preflight.DesignSplitAttackData:
            visible = text_preflight.SplitVisibleAttackMatrices(
                split=split,
                world_uids=tuple(),
                pair_uids=tuple(),
                labels=np.asarray([], dtype=np.int8),
                views={},
                feature_names_by_view={},
            )
            return text_preflight.DesignSplitAttackData(
                visible=visible,
                assignment_relation_names=tuple(),
                assignment_observed=np.empty((0, 0), dtype=np.float64),
                assignment_expected=np.empty((0, 0), dtype=np.float64),
                assignment_world_uids=tuple(),
                world_audits=tuple(
                    {
                        "world_uid": f"{split}-cf-{index:03d}",
                        "source_style_changed_seller_count": (
                            0 if index == 0 else 27
                        ),
                        "raw_title_changed_item_count": 0 if index == 0 else 75,
                        "raw_description_changed_item_count": (
                            0 if index == 0 else 80
                        ),
                        "parser_exact_rows_and_flags": True,
                        "planned_identity_surface_residue_count": 0,
                        "seller_profile_count": 28,
                        "production_audit_sha256": "0" * 64,
                    }
                    for index in range(500)
                ),
            )

        def original(split: str) -> text_runner.SplitOriginalDiagnostic:
            return text_runner.SplitOriginalDiagnostic(
                split=split,
                world_uids=tuple(),
                pair_uids=tuple(),
                labels=np.asarray([], dtype=np.int8),
                feature_names=tuple(),
                scores=np.empty((0, 0), dtype=np.float64),
                strata=tuple(),
                inventory_sets={},
                visible_leakage_count=0,
                world_audits=tuple(
                    {
                        "world_uid": f"{split}-original-{index:03d}",
                        "parser_exact_rows_and_flags": True,
                        "planned_identity_surface_residue_count": 0,
                        "parsed_identity_occurrence_count": 84,
                        "redacted_item_count": 100,
                        "seller_profile_count": 28,
                        "production_audit_sha256": "1" * 64,
                    }
                    for index in range(500)
                ),
            )

        result = text_runner._rowwise_audit_receipt(
            counterfactual_train_world_audits=counterfactual(
                "train"
            ).world_audits,
            counterfactual_development_world_audits=counterfactual(
                "development"
            ).world_audits,
            original_train=original("train"),
            original_development=original("development"),
        )
        self.assertEqual(
            result["status"],
            "PASS_DESIGN_WORLDS_ROW_BY_ROW_RECOMPUTED_IN_MEMORY",
        )
        self.assertEqual(
            result["splits"]["train"]["original_redacted_item_count"],
            50_000,
        )
        self.assertEqual(result["formal_dataset_rows_audited"], 0)
        self.assertEqual(
            result["splits"]["train"][
                "counterfactual_change_counts_descriptive_only"
            ]["source_style_changed_seller_count_minimum"],
            0,
        )
        broken = counterfactual("train")
        broken_rows = list(broken.world_audits)
        broken_rows[0] = {**broken_rows[0], "parser_exact_rows_and_flags": False}
        broken = text_preflight.DesignSplitAttackData(
            visible=broken.visible,
            assignment_relation_names=broken.assignment_relation_names,
            assignment_observed=broken.assignment_observed,
            assignment_expected=broken.assignment_expected,
            assignment_world_uids=broken.assignment_world_uids,
            world_audits=tuple(broken_rows),
        )
        with self.assertRaises(text_runner.TextShortcutRunnerError):
            text_runner._rowwise_audit_receipt(
                counterfactual_train_world_audits=broken.world_audits,
                counterfactual_development_world_audits=counterfactual(
                    "development"
                ).world_audits,
                original_train=original("train"),
                original_development=original("development"),
            )

    def test_failure_receipt_cannot_authorize_or_claim_formal_rows(self) -> None:
        with self.assertRaises(text_runner.TextShortcutStageError) as raised:
            text_runner._run_stage(
                "deliberate_test_stage",
                lambda: (_ for _ in ()).throw(ValueError("deliberate failure")),
            )
        receipt = text_runner._failure_receipt(
            run_id="v13_training_ready_v1_12_cleanroom_20260803",
            error=raised.exception,
        )
        self.assertEqual(
            receipt["status"], "FAIL_DESIGN_TEXT_SHORTCUT_PREFLIGHT_EXCEPTION"
        )
        self.assertEqual(
            set(receipt["formal_authorizations_after_failure"].values()),
            {False},
        )
        self.assertFalse(receipt["formal_seed_or_key_access"])
        self.assertEqual(receipt["formal_dataset_rows_produced"], 0)
        self.assertEqual(receipt["formal_dataset_rows_audited"], 0)
        self.assertFalse(receipt["model_training_authorized"])
        self.assertEqual(receipt["failure_stage"], "deliberate_test_stage")
        self.assertEqual(receipt["cause_type"], "ValueError")
        self.assertEqual(
            receipt["non_reuse_commitment"],
            text_runner.FROZEN_VERSION_NON_REUSE_COMMITMENT,
        )
        claimed = dict(receipt)
        canonical_self_hash = claimed.pop("canonical_self_hash")
        self.assertEqual(
            preceremony.canonical_sha256(claimed), canonical_self_hash
        )

    def test_main_writes_stage_failure_once_and_never_replaces_it(self) -> None:
        reports = ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="step28-v13-v1-12-runner-test-", dir=reports
        ) as directory:
            output = Path(directory) / "failure_receipt.json"
            args = mock.Mock(
                run_design_preflight=True,
                output=str(output),
                progress_every=10,
            )
            error = text_runner.TextShortcutStageError(
                "counterfactual_train_500_worlds", ValueError("deliberate")
            )
            with mock.patch.object(
                text_runner, "parse_args", return_value=args
            ), mock.patch.object(
                text_runner, "run_design_preflight", side_effect=error
            ):
                with self.assertRaises(text_runner.TextShortcutStageError):
                    text_runner.main()
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["status"],
                "FAIL_DESIGN_TEXT_SHORTCUT_PREFLIGHT_EXCEPTION",
            )
            self.assertEqual(
                receipt["failure_stage"], "counterfactual_train_500_worlds"
            )
            original_bytes = output.read_bytes()
            with self.assertRaises(text_runner.TextShortcutRunnerError):
                text_runner._write_receipt_no_replace(output, receipt)
            self.assertEqual(output.read_bytes(), original_bytes)

    def test_source_closure_pins_formal_build_draft_and_self_hash(self) -> None:
        policy = text_preflight.load_text_audit_policy()
        closure = text_runner._source_closure(policy)
        draft = formal.load_and_validate_draft()["draft"]
        self.assertEqual(
            closure["formal_build_draft"]["canonical_self_hash"],
            draft["canonical_self_hash"],
        )
        self.assertTrue(
            preceremony.HEX_SHA256_RE.fullmatch(
                closure["formal_build_draft"]["sha256"]
            )
        )

    def test_earliest_source_closure_failure_still_writes_receipt(self) -> None:
        reports = ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="step28-v13-v1-12-early-failure-test-", dir=reports
        ) as directory:
            output = Path(directory) / "failure_receipt.json"
            args = mock.Mock(
                run_design_preflight=True,
                output=str(output),
                progress_every=10,
            )
            with mock.patch.object(
                text_runner, "parse_args", return_value=args
            ), mock.patch.object(
                text_runner,
                "_source_closure",
                side_effect=text_runner.TextShortcutRunnerError(
                    "deliberately broken closure"
                ),
            ):
                with self.assertRaises(text_runner.TextShortcutStageError):
                    text_runner.main()
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                receipt["status"],
                "FAIL_DESIGN_TEXT_SHORTCUT_PREFLIGHT_EXCEPTION",
            )
            self.assertEqual(receipt["failure_stage"], "source_closure")
            self.assertEqual(
                receipt["source_closure_status"],
                "FAILED_BEFORE_COMPLETE_SOURCE_CLOSURE",
            )
            self.assertEqual(
                set(receipt["formal_authorizations_after_failure"].values()),
                {False},
            )
            self.assertIn("runner", receipt["source_files"])

    def test_unexpected_stage_status_is_stage_aware_failure(self) -> None:
        with self.assertRaises(text_runner.TextShortcutStageError) as raised:
            text_runner._require_stage_status(
                "counterfactual_fast_full_parity",
                {"status": "UNEXPECTED"},
                "PASS_TEXT_FAST_FULL_REDACTED_PARITY",
            )
        self.assertEqual(
            raised.exception.stage, "counterfactual_fast_full_parity"
        )

    def test_non_mapping_gate_results_are_stage_aware_failures(self) -> None:
        for stage, result in (
            ("assignment_development_hard_gate", None),
            ("visible_attack_models_and_hard_gates", []),
        ):
            with self.subTest(stage=stage):
                with self.assertRaises(
                    text_runner.TextShortcutStageError
                ) as raised:
                    text_runner._require_stage_mapping(stage, result)
                self.assertEqual(raised.exception.stage, stage)

    def test_design_run_id_failure_has_exact_stage_and_receipt(self) -> None:
        reports = ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="step28-v13-v1-12-run-id-failure-test-", dir=reports
        ) as directory:
            output = Path(directory) / "failure_receipt.json"
            args = mock.Mock(
                run_design_preflight=True,
                output=str(output),
                progress_every=10,
            )
            with mock.patch.object(
                text_runner, "parse_args", return_value=args
            ), mock.patch.object(
                text_runner,
                "_design_run_id",
                side_effect=text_runner.TextShortcutRunnerError(
                    "deliberately broken draft"
                ),
            ):
                with self.assertRaises(text_runner.TextShortcutStageError):
                    text_runner.main()
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["failure_stage"], "design_run_id")
            self.assertEqual(receipt["run_id"], "unresolved-v1.12-design-preflight")

    def test_policy_failure_has_exact_stage_and_closed_receipt(self) -> None:
        reports = ROOT / "reports"
        reports.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix="step28-v13-v1-12-policy-failure-test-", dir=reports
        ) as directory:
            output = Path(directory) / "failure_receipt.json"
            args = mock.Mock(
                run_design_preflight=True,
                output=str(output),
                progress_every=10,
            )
            with mock.patch.object(
                text_runner, "parse_args", return_value=args
            ), mock.patch.object(
                text_preflight,
                "load_text_audit_policy",
                side_effect=text_runner.TextShortcutRunnerError(
                    "deliberately broken policy"
                ),
            ):
                with self.assertRaises(text_runner.TextShortcutStageError):
                    text_runner.main()
            receipt = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(receipt["failure_stage"], "policy_closure")
            self.assertEqual(
                receipt["policy_validation"]["status"],
                "FAILED_BEFORE_COMPLETE_POLICY_CLOSURE",
            )
            self.assertEqual(
                set(receipt["formal_authorizations_after_failure"].values()),
                {False},
            )


class Step28V13V112CounterfactualTextContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        validated = preceremony.validate_policy()
        cleanroom_policy = validated["policy"]
        base_policy_path = preceremony._repo_path(
            next(
                record["path"]
                for record in cleanroom_policy["design_only_base_inputs"]
                if record["path"].endswith(
                    "synthetic_chinese_dataset_policy.json"
                )
            )
        )
        cls.policy = common.load_policy(
            base_policy_path, mode="development_smoke"
        )
        cls.template, fixture = common.validate_policy_release_documents(
            cls.policy, mode="development_smoke"
        )
        style_profile = common.load_json(
            common.verify_file_pin(
                cls.policy["style_reference_boundary"][
                    "generator_release_inputs"
                ]["profile"],
                label="counterfactual unit style profile",
            )
        )
        record = next(
            row
            for row in structure.build_mode_world_pool(
                cls.policy, mode="development_smoke"
            )
            if row["split"] == "train"
        )
        cls.world = world_builder.build_world(
            policy=cls.policy,
            template=cls.template,
            fixture=fixture,
            style_profile=style_profile,
            mode="development_smoke",
            world_record=record,
            structure_key_hex=common.structure_key_for_split(
                cls.policy, mode="development_smoke", split="train"
            ),
        )

    def _kwargs(self) -> dict[str, object]:
        public = self.world["public"]
        private = self.world["private"]
        return {
            "mode": "development_smoke",
            "split": "train",
            "template": self.template,
            "sellers": public["sellers"],
            "items": public["items"],
            "identity_slots_audit": private["identity_slots_audit"],
            "noise_slots_audit": private["noise_slots_audit"],
            "render_asts": private["render_asts"],
            "override_audit": private["override_audit"],
        }

    def test_identity_mapping_is_byte_equal_to_original_world(self) -> None:
        observed = counterfactual.rerender_identity_mapping_fixture(
            self.policy, **self._kwargs()
        )
        original_processed = production.process_world(
            self.policy,
            mode="development_smoke",
            split="train",
            template=self.template,
            world=self.world,
        )
        expected_profiles, _profile_audit = profiles.build_world_profiles(
            self.policy,
            mode="development_smoke",
            split="train",
            sellers=self.world["public"]["sellers"],
            items=original_processed["public"]["profile_safe_items"],
        )
        self.assertEqual(observed["public"]["raw_items"], self.world["public"]["items"])
        self.assertEqual(
            observed["private"]["render_asts"],
            self.world["private"]["render_asts"],
        )
        self.assertEqual(
            observed["private"]["identity_slots_audit"],
            self.world["private"]["identity_slots_audit"],
        )
        self.assertEqual(
            observed["private"]["noise_slots_audit"],
            self.world["private"]["noise_slots_audit"],
        )
        for name in ("redacted_items", "profile_safe_items"):
            observed_index = {
                row["item_uid"]: row for row in observed["public"][name]
            }
            expected_index = {
                row["item_uid"]: row
                for row in original_processed["public"][name]
            }
            self.assertEqual(len(observed_index), len(observed["public"][name]))
            self.assertEqual(
                len(expected_index), len(original_processed["public"][name])
            )
            self.assertEqual(observed_index, expected_index)
        self.assertEqual(observed["public"]["seller_profiles"], expected_profiles)
        self.assertTrue(observed["audit"]["identity_fixture"])
        self.assertEqual(observed["audit"]["source_style_changed_seller_count"], 0)

    def test_deranged_world_changes_only_registered_text_and_style_fields(self) -> None:
        observed = counterfactual.rerender_counterfactual_world(
            self.policy, **self._kwargs()
        )
        original_items = self.world["public"]["items"]
        updated_items = observed["public"]["raw_items"]
        self.assertEqual(len(updated_items), len(original_items))
        for original, updated in zip(original_items, updated_items, strict=True):
            self.assertEqual(
                {key: value for key, value in original.items() if key not in {"title", "description"}},
                {key: value for key, value in updated.items() if key not in {"title", "description"}},
            )
        for original, updated in zip(
            self.world["private"]["render_asts"],
            observed["private"]["render_asts"],
            strict=True,
        ):
            self.assertEqual(
                {key: value for key, value in original.items() if key != "effective_style_uid"},
                {key: value for key, value in updated.items() if key != "effective_style_uid"},
            )
        for table_name, uid_name in (
            ("identity_slots_audit", "slot_uid"),
            ("noise_slots_audit", "noise_slot_uid"),
        ):
            original_rows = self.world["private"][
                "identity_slots_audit"
                if table_name == "identity_slots_audit"
                else "noise_slots_audit"
            ]
            updated_rows = observed["private"][table_name]
            self.assertEqual(
                [row[uid_name] for row in original_rows],
                [row[uid_name] for row in updated_rows],
            )
            for original, updated in zip(original_rows, updated_rows, strict=True):
                self.assertEqual(
                    {key: value for key, value in original.items() if key not in {"start", "end"}},
                    {key: value for key, value in updated.items() if key not in {"start", "end"}},
                )

        mapping_pairs = observed["audit"]["derangement"]["target_source_pairs"]
        self.assertEqual(len(mapping_pairs), 28)
        self.assertTrue(all(target != source for target, source in mapping_pairs))
        self.assertGreater(observed["audit"]["source_style_changed_seller_count"], 0)
        self.assertEqual(len(observed["public"]["seller_profiles"]), 28)
        self.assertTrue(observed["audit"]["parser"]["exact_rows_and_flags"])
        self.assertEqual(
            observed["audit"]["redaction"]["planned_identity_surface_residue_count"],
            0,
        )

        item_index = {
            row["item_uid"]: row for row in observed["public"]["raw_items"]
        }
        for override in self.world["private"]["override_audit"]:
            if override["override_kind"] == "exact_title_clone":
                self.assertEqual(
                    item_index[override["item_uid_left"]]["title"],
                    item_index[override["item_uid_right"]]["title"],
                )

    def test_counterfactual_visible_attack_matrix_has_exact_neutral_universe(self) -> None:
        observed = counterfactual.rerender_counterfactual_world(
            self.policy, **self._kwargs()
        )
        private = self.world["private"]
        labels = preceremony.validate_full_pair_labels(
            pair_rows=self.world["public"]["complete_model_pair_endpoints"],
            controller_membership=private["controller_membership"],
            expected_world_uid=observed["world_uid"],
        )
        matrices = text_preflight.build_world_visible_attack_matrices(
            policy=text_preflight.load_text_audit_policy(),
            seller_profiles=observed["public"]["seller_profiles"],
            pair_rows=self.world["public"]["complete_model_pair_endpoints"],
            label_rows=labels,
            negative_flags=private["negative_flags"],
            override_audit=private["override_audit"],
        )
        self.assertEqual(len(matrices.pair_uids), 372)
        self.assertEqual(len(matrices.excluded_pair_uids), 6)
        self.assertEqual(int(np.sum(matrices.labels)), 20)
        self.assertEqual(
            {name: value.shape for name, value in matrices.views.items()},
            {
                "cf_full": (372, 75),
                "cf_topic": (372, 14),
                "cf_template_surface": (372, 56),
            },
        )
        self.assertTrue(
            all(np.all(np.isfinite(value)) for value in matrices.views.values())
        )
        audit_policy = text_preflight.load_text_audit_policy()
        for view_name, matrix in matrices.views.items():
            self.assertEqual(
                matrices.feature_names_by_view[view_name],
                tuple(
                    audit_policy["visible_attack"]["views"][view_name][
                        "feature_names_in_order"
                    ]
                ),
            )
            self.assertEqual(
                matrix.shape[1], len(matrices.feature_names_by_view[view_name])
            )

        pair_index = {
            row["canonical_pair_uid"]: row
            for row in self.world["public"]["complete_model_pair_endpoints"]
        }
        first_pair = pair_index[matrices.pair_uids[0]]
        profile_index = {
            row["seller_uid"]: row for row in observed["public"]["seller_profiles"]
        }
        fields = audit_policy["visible_attack"]["m0_fields_in_order"]
        separator = bytes.fromhex(
            audit_policy["visible_attack"]["combined_field_separator_utf8_hex"]
        ).decode("utf-8")
        combined_documents = [
            separator.join(profile_index[first_pair[endpoint]][field] for field in fields)
            for endpoint in ("seller_uid_left", "seller_uid_right")
        ]
        kwargs = dict(
            audit_policy["visible_attack"]["vectorizers"]["char3"][
                "constructor_kwargs"
            ]
        )
        kwargs["ngram_range"] = tuple(kwargs["ngram_range"])
        kwargs["dtype"] = np.float64
        vectors = HashingVectorizer(**kwargs).transform(combined_documents).tocsr()
        product = vectors[0].multiply(vectors[1]).tocsr()
        product.sort_indices()
        expected_combined = 0.0
        for value in product.data:
            expected_combined += float(value)
        full_names = matrices.feature_names_by_view["cf_full"]
        combined_index = full_names.index("char3_cosine__all_fields")
        self.assertEqual(
            matrices.views["cf_full"][0, combined_index], expected_combined
        )

        summary_specs = (
            (
                "cf_full",
                audit_policy["visible_attack"]["similarity_summaries"][
                    "cf_full_sources_in_order"
                ],
                (
                    "similarity_max__field_char_word",
                    "similarity_mean__field_char_word",
                    "similarity_top2_mean__field_char_word",
                ),
            ),
            (
                "cf_template_surface",
                audit_policy["visible_attack"]["similarity_summaries"][
                    "cf_template_sources_in_order"
                ],
                (
                    "masked_similarity_max__text_fields",
                    "masked_similarity_mean__text_fields",
                    "masked_similarity_top2_mean__text_fields",
                ),
            ),
        )
        for view_name, source_names, summary_names in summary_specs:
            names = matrices.feature_names_by_view[view_name]
            source = np.asarray(
                [matrices.views[view_name][0, names.index(name)] for name in source_names],
                dtype=np.float64,
            )
            ordered = np.sort(source)
            expected_summary = (
                ordered[-1],
                np.mean(source, dtype=np.float64),
                np.mean(ordered[-2:], dtype=np.float64),
            )
            self.assertEqual(
                tuple(matrices.views[view_name][0, names.index(name)] for name in summary_names),
                expected_summary,
            )
        replay = text_preflight.build_world_visible_attack_matrices(
            policy=text_preflight.load_text_audit_policy(),
            seller_profiles=list(reversed(observed["public"]["seller_profiles"])),
            pair_rows=list(reversed(self.world["public"]["complete_model_pair_endpoints"])),
            label_rows=list(reversed(labels)),
            negative_flags=list(reversed(private["negative_flags"])),
            override_audit=list(reversed(private["override_audit"])),
        )
        self.assertEqual(matrices.pair_uids, replay.pair_uids)
        self.assertTrue(np.array_equal(matrices.labels, replay.labels))
        for name in matrices.views:
            self.assertTrue(np.array_equal(matrices.views[name], replay.views[name]))

    def test_original_description_rejects_flag_override_lineage_drift(self) -> None:
        projection = text_runner._project_original_visible_world(
            execution_policy=self.policy,
            template=self.template,
            split="train",
            world=self.world,
        )
        labels = preceremony.validate_full_pair_labels(
            pair_rows=self.world["public"]["complete_model_pair_endpoints"],
            controller_membership=self.world["private"]["controller_membership"],
            expected_world_uid=self.world["public"]["world"]["world_uid"],
        )
        drifted = json.loads(json.dumps(self.world))
        drifted["private"]["override_audit"][0]["asset_index"] += 1000
        with self.assertRaises(text_runner.TextShortcutRunnerError):
            text_runner._build_original_world_diagnostic(
                policy=text_preflight.load_text_audit_policy(),
                world=drifted,
                projection=projection,
                label_rows=labels,
            )

    def test_assignment_null_uses_actual_world_conditioning_without_classifier(self) -> None:
        observed = counterfactual.rerender_counterfactual_world(
            self.policy, **self._kwargs()
        )
        private = self.world["private"]
        label_rows = preceremony.validate_full_pair_labels(
            pair_rows=self.world["public"]["complete_model_pair_endpoints"],
            controller_membership=private["controller_membership"],
            expected_world_uid=observed["world_uid"],
        )
        matrices = text_preflight.build_world_visible_attack_matrices(
            policy=text_preflight.load_text_audit_policy(),
            seller_profiles=observed["public"]["seller_profiles"],
            pair_rows=self.world["public"]["complete_model_pair_endpoints"],
            label_rows=label_rows,
            negative_flags=private["negative_flags"],
            override_audit=private["override_audit"],
        )
        pair_index = {
            row["canonical_pair_uid"]: row
            for row in self.world["public"]["complete_model_pair_endpoints"]
        }
        eligible_pairs = [pair_index[pair_uid] for pair_uid in matrices.pair_uids]
        audit = assignment_null.build_assignment_null_rows(
            policy=text_preflight.load_text_audit_policy(),
            template=self.template,
            sellers=self.world["public"]["sellers"],
            render_asts=private["render_asts"],
            controller_membership=private["controller_membership"],
            controller_style_groups=private["controller_style_groups"],
            target_source_pairs=observed["audit"]["derangement"][
                "target_source_pairs"
            ],
            eligible_pair_rows=eligible_pairs,
            labels=matrices.labels.tolist(),
        )
        self.assertFalse(audit["classifier_fitted"])
        self.assertEqual(len(audit["seller_rows"]), 28)
        self.assertEqual(len(audit["pair_rows"]), 372)
        self.assertEqual(sum(row["label"] for row in audit["pair_rows"]), 20)
        positive_controller_expectation = [
            row["expected__same_source_controller"]
            for row in audit["pair_rows"]
            if row["label"] == 1
        ]
        negative_controller_expectation = [
            row["expected__same_source_controller"]
            for row in audit["pair_rows"]
            if row["label"] == 0
        ]
        self.assertEqual(
            float(np.mean(positive_controller_expectation)),
            0.05375579975579976,
        )
        self.assertEqual(
            float(np.mean(negative_controller_expectation)),
            0.05286268708143707,
        )
        source_by_target = dict(
            observed["audit"]["derangement"]["target_source_pairs"]
        )
        controller_by_seller = {
            row["seller_uid"]: row["controller_uid"]
            for row in private["controller_membership"]
        }
        for row in audit["seller_rows"]:
            self.assertEqual(
                row["source_controller_equals_target_controller"],
                int(
                    controller_by_seller[row["source_seller_uid"]]
                    == controller_by_seller[row["target_seller_uid"]]
                ),
            )
        for row in audit["pair_rows"]:
            pair = pair_index[row["canonical_pair_uid"]]
            source_left = source_by_target[pair["seller_uid_left"]]
            source_right = source_by_target[pair["seller_uid_right"]]
            self.assertEqual(
                row["same_source_controller"],
                float(
                    controller_by_seller[source_left]
                    == controller_by_seller[source_right]
                ),
            )
            for name in text_preflight.load_text_audit_policy()[
                "assignment_null_audit"
            ]["pair_gate_relations_in_order"]:
                self.assertIn(row[name], (0.0, 1.0) if name != "same_source_factor_proportion" else tuple(index / 6 for index in range(7)))
                self.assertGreaterEqual(row[f"expected__{name}"], 0.0)
                self.assertLessEqual(row[f"expected__{name}"], 1.0)


if __name__ == "__main__":
    unittest.main()
