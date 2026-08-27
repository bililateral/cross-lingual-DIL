#!/usr/bin/env python3
"""Contracts for the Step28-v13 v1.13 V9.3 abstract preflights."""

from __future__ import annotations

import copy
from collections import defaultdict
import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_v1_13_balanced_schedule_v9_3 as schedule
import step28_v13_v1_13_build_joint_noise_signatures_v9_3 as signatures
import step28_v13_v1_13_construct_registered_negative_plan_v9_3 as negative_constructor
import step28_v13_v1_13_registered_negative_plan_v9_3 as negative_plan
import step28_v13_v1_13_replay_joint_noise_signatures_v9_3 as signature_replay


PREFLIGHT = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "design_preflight_v2_20260825"
)
JOINT_SIGNATURE = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "joint_noise_signature_preflight_v2_20260826.json"
)
NEGATIVE_PREFLIGHT = (
    ROOT
    / "reports"
    / "step28_v13_v1_13_balanced_schedule_v9_3"
    / "__successor_registered_negative_plan_unassigned__"
)


def read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"Expected a JSON object: {path}")
    return value


class BalancedScheduleContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = read_json(PREFLIGHT / "train_balanced_schedule.json")
        cls.development = read_json(
            PREFLIGHT / "development_balanced_schedule.json"
        )

    def test_published_schedules_have_exact_balance_and_frozen_distance(self) -> None:
        train_audit = schedule.validate_schedule(self.train)
        development_audit = schedule.validate_schedule(self.development)
        self.assertEqual(train_audit["seller_pair_count_histogram"], {"26": 206, "27": 172})
        self.assertEqual(development_audit["noise_pair_count_histogram"], {"26": 206, "27": 172})

        pair_audit = schedule.validate_train_development_pair(
            self.train, self.development
        )
        self.assertEqual(
            pair_audit["seller_pair_indicator_distance"],
            {
                "vector_length": 378,
                "high_value": 27,
                "high_count_per_split": 172,
                "high_set_intersection": 83,
                "indicator_hamming_distance": 178,
            },
        )
        self.assertEqual(
            pair_audit["noise_pair_indicator_distance"]["indicator_hamming_distance"],
            184,
        )
        self.assertEqual(
            pair_audit["seller_triad_indicator_distance"]["indicator_hamming_distance"],
            12,
        )
        self.assertEqual(
            pair_audit["noise_triad_indicator_distance"]["indicator_hamming_distance"],
            12,
        )
        self.assertEqual(
            pair_audit["seller_global_relabel_audit"]["shared_exact_pattern_count"],
            0,
        )
        self.assertEqual(
            pair_audit["noise_global_relabel_audit"]["shared_exact_pattern_count"],
            0,
        )

    def test_self_hash_and_exact_schema_fail_closed(self) -> None:
        for mutation in ("hash", "schema"):
            with self.subTest(mutation=mutation):
                tampered = copy.deepcopy(self.train)
                if mutation == "hash":
                    tampered["canonical_self_sha256"] = "0" * 64
                else:
                    tampered["unexpected"] = 1
                    tampered["canonical_self_sha256"] = schedule.canonical_self_sha256(
                        tampered
                    )
                with self.assertRaises(schedule.BalancedScheduleError):
                    schedule.validate_schedule(tampered)

    def test_noise_mapping_and_train_development_identity_fail_closed(self) -> None:
        duplicate_noise = copy.deepcopy(self.train)
        noise = duplicate_noise["worlds"][0]["noise_slot_by_seller_slot"]
        noise[0] = noise[1]
        duplicate_noise["canonical_self_sha256"] = schedule.canonical_self_sha256(
            duplicate_noise
        )
        with self.assertRaises(schedule.BalancedScheduleError):
            schedule.validate_schedule(duplicate_noise)

        relabelled_copy = copy.deepcopy(self.train)
        relabelled_copy["split"] = "development"
        relabelled_copy["canonical_self_sha256"] = schedule.canonical_self_sha256(
            relabelled_copy
        )
        with self.assertRaises(schedule.BalancedScheduleError):
            schedule.validate_train_development_pair(self.train, relabelled_copy)

    def test_fixed_global_relabel_isomorphism_fails_closed(self) -> None:
        mapping = tuple((slot + 5) % 28 for slot in range(28))
        relabelled = copy.deepcopy(self.train)
        relabelled["split"] = "development"
        for world in relabelled["worlds"]:
            old_noise = world["noise_slot_by_seller_slot"]
            world["controller_groups"] = [
                sorted(mapping[slot] for slot in group)
                for group in world["controller_groups"]
            ]
            world["controller_groups"] = [
                *sorted(group for group in world["controller_groups"] if len(group) == 3),
                *sorted(group for group in world["controller_groups"] if len(group) == 2),
            ]
            new_noise = [0] * 28
            for seller in range(28):
                new_noise[mapping[seller]] = mapping[old_noise[seller]]
            world["noise_slot_by_seller_slot"] = new_noise
        relabelled["canonical_self_sha256"] = schedule.canonical_self_sha256(
            relabelled
        )
        with self.assertRaisesRegex(
            schedule.BalancedScheduleError, "fixed global slot relabel"
        ):
            schedule.validate_train_development_pair(self.train, relabelled)


class JointNoiseSignatureContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.payload = read_json(JOINT_SIGNATURE)

    def test_aggregate_preflight_replays_and_has_expected_role_capacity(self) -> None:
        audit = signatures.validate_payload(self.payload)
        self.assertEqual(audit["seller_count"], 648)
        self.assertEqual(audit["selected_item_row_count"], 3354)
        self.assertEqual(audit["observed_signature_count"], 33)
        self.assertEqual(audit["noise_slot_count"], 28)
        self.assertEqual(audit["title_eligible_noise_slot_count"], 28)
        self.assertEqual(
            audit["title_and_description_eligible_noise_slot_count"], 28
        )
        self.assertEqual(self.payload["source_seller_count"], 676)
        self.assertEqual(self.payload["excluded_seller_count"], 28)
        self.assertEqual(self.payload["source_selected_item_row_count"], 3439)
        self.assertEqual(self.payload["excluded_selected_item_row_count"], 85)

    def test_integerization_and_forbidden_output_flags_fail_closed(self) -> None:
        for mutation in ("allocation", "forbidden"):
            with self.subTest(mutation=mutation):
                tampered = copy.deepcopy(self.payload)
                if mutation == "allocation":
                    tampered["signature_frequency_and_integerization"][0][
                        "allocated_slot_count"
                    ] += 1
                else:
                    tampered["contains_seller_uid"] = True
                tampered["canonical_self_sha256"] = signatures.canonical_self_sha256(
                    tampered
                )
                with self.assertRaises(signatures.JointNoiseSignatureError):
                    signatures.validate_payload(tampered)

    def test_inconsistent_joint_empty_mask_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.payload)
        tampered["noise_slot_multiset"][0]["signature"]["joint_empty_mask"] = "11"
        tampered["canonical_self_sha256"] = signatures.canonical_self_sha256(
            tampered
        )
        with self.assertRaises(signatures.JointNoiseSignatureError):
            signatures.validate_payload(tampered)

    def test_signature_clipping_and_largest_remainder_ties_are_deterministic(self) -> None:
        singleton = signatures._signature([(7, True, False)])
        self.assertEqual(singleton["item_count"], 2)
        self.assertEqual(singleton["title_present_mask"], "11")
        clipped = signatures._signature(
            [(index, True, index % 2 == 0) for index in range(2, 12)]
        )
        self.assertEqual(clipped["item_count"], 8)

        rows = [
            {"item_count": 2, "title_present_mask": bits, "description_present_mask": "11", "joint_empty_mask": "00"}
            for bits in ("01", "10", "11")
        ]
        encoded = {signatures._signature_key(row): row for row in rows}
        ordered = sorted(encoded)
        counts = {
            ordered[0]: 216,
            ordered[1]: 216,
            ordered[2]: 216,
        }
        integerized, slots = signatures._largest_remainder(
            signatures.Counter(counts), encoded
        )
        allocations = {
            signatures._signature_key(row["signature"]): row["allocated_slot_count"]
            for row in integerized
        }
        self.assertEqual(allocations[ordered[0]], 10)
        self.assertEqual(allocations[ordered[1]], 9)
        self.assertEqual(len(slots), 28)

    def test_nested_provenance_and_status_tampering_fail_closed(self) -> None:
        mutations = (
            ("status", lambda value: value.__setitem__("status", "PASS")),
            (
                "source_pin",
                lambda value: value["source_pins"]["item_manifest"].__setitem__(
                    "sha256", "0" * 64
                ),
            ),
            (
                "nested_sensitive_key",
                lambda value: value["signature_definition"].__setitem__(
                    "seller_uid", "forbidden"
                ),
            ),
        )
        for name, mutate in mutations:
            with self.subTest(name=name):
                tampered = copy.deepcopy(self.payload)
                mutate(tampered)
                tampered["canonical_self_sha256"] = signatures.canonical_self_sha256(
                    tampered
                )
                with self.assertRaises(signatures.JointNoiseSignatureError):
                    signatures.validate_payload(tampered)

    def test_independent_raw_source_replay_closes_derived_tables(self) -> None:
        audit = signature_replay.replay(self.payload)
        self.assertEqual(audit["source_seller_count"], 676)
        self.assertEqual(audit["seller_count"], 648)
        self.assertEqual(audit["excluded_seller_count"], 28)
        self.assertEqual(audit["source_selected_item_row_count"], 3439)
        self.assertEqual(audit["selected_item_row_count"], 3354)
        self.assertEqual(audit["excluded_selected_item_row_count"], 85)
        self.assertEqual(
            audit["status"],
            "PASS_INDEPENDENT_RAW_SOURCE_REPLAY_ONLY_NOT_METHOD_OR_TRAINING_QUALIFIED",
        )


class RegisteredNegativeConstructorPrimitiveContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train = read_json(PREFLIGHT / "train_balanced_schedule.json")
        cls.development = read_json(
            PREFLIGHT / "development_balanced_schedule.json"
        )
        cls.joint_signatures = read_json(JOINT_SIGNATURE)

    def test_incremental_objective_matches_full_recomputation(self) -> None:
        search = negative_constructor.JointSearch(
            self.train, self.joint_signatures, split="train"
        )
        self.assertEqual(search.objective, search._full_objective())
        accepted = 0
        for _attempt in range(1000):
            proposal = search._change_delta(search._proposal())
            if proposal is None:
                continue
            objective_delta, changes, new_rows = proposal
            before_full = search._full_objective()
            for (family, index), count_delta in changes.items():
                search.arrays[family][index] += count_delta
            for world, row in new_rows.items():
                search.assignments[world] = row
            search.objective += objective_delta
            self.assertEqual(
                search.objective,
                search._full_objective(),
                msg=(
                    f"incremental drift at attempt={_attempt} "
                    f"declared_delta={objective_delta} "
                    f"actual_delta={search._full_objective() - before_full} "
                    f"changes={changes}"
                ),
            )
            accepted += 1
        self.assertGreater(accepted, 100)

    def test_terminal_v17_formal_entry_is_unassigned_and_fails_closed(self) -> None:
        expected_output = ROOT / negative_constructor.FORMAL_OUTPUT_RELATIVE
        self.assertEqual(
            expected_output.name,
            "__successor_registered_negative_plan_unassigned__",
        )
        supplied = {
            "train_schedule": PREFLIGHT / "train_balanced_schedule.json",
            "development_schedule": (
                PREFLIGHT / "development_balanced_schedule.json"
            ),
            "joint_signatures": JOINT_SIGNATURE,
        }
        for name, path in supplied.items():
            expected_relative, expected_sha256 = (
                negative_constructor.FORMAL_INPUT_PINS[name]
            )
            self.assertEqual(path.resolve(), (ROOT / expected_relative).resolve())
            self.assertEqual(
                negative_constructor.hashlib.sha256(path.read_bytes()).hexdigest(),
                expected_sha256,
            )
        with self.assertRaisesRegex(
            negative_constructor.RegisteredNegativeConstructionError,
            "successor is unassigned",
        ):
            negative_constructor.validate_formal_invocation(
                output_directory=expected_output,
                train_schedule_path=supplied["train_schedule"],
                development_schedule_path=supplied["development_schedule"],
                joint_signature_path=supplied["joint_signatures"],
            )

    def test_development_initialization_is_independent_and_valid(self) -> None:
        train_search = negative_constructor.JointSearch(
            self.train, self.joint_signatures, split="train"
        )
        development_search = negative_constructor.JointSearch(
            self.development, self.joint_signatures, split="development"
        )
        controllers, _triads, _noise = negative_constructor._world_arrays(
            self.development
        )
        self.assertNotEqual(
            train_search.assignments.tolist(), development_search.assignments.tolist()
        )
        for world, row in enumerate(development_search.assignments):
            self.assertEqual(len(set(map(int, row))), 12)
            for left, right in negative_constructor.PAIR_POSITIONS:
                self.assertNotEqual(
                    controllers[world, int(row[left])],
                    controllers[world, int(row[right])],
                )

    def test_targeted_row_replacement_preserves_uniqueness(self) -> None:
        search = negative_constructor.JointSearch(
            self.train, self.joint_signatures, split="train"
        )
        row = search.assignments[0].copy()
        replacements = search._row_replacements_for_targets(
            0,
            row,
            {0: int(row[1]), 1: int(row[0])},
        )
        self.assertTrue(replacements)
        proposal = search._change_delta(replacements)
        self.assertIsNotNone(proposal)
        _delta, _changes, new_rows = proposal
        self.assertEqual(len(set(map(int, new_rows[0]))), 12)
        self.assertEqual(int(new_rows[0][0]), int(row[1]))
        self.assertEqual(int(new_rows[0][1]), int(row[0]))

    def test_targeted_generator_can_help_a_violated_cell(self) -> None:
        search = negative_constructor.JointSearch(
            self.train, self.joint_signatures, split="train"
        )
        violated = [
            cell
            for cell in search._constraint_cells()
            if not cell[2] <= int(search.arrays[cell[0]][cell[1]]) <= cell[3]
        ]
        source_cell = violated[0]
        found_helpful = False
        for source_key, _kind, replacements in search._targeted_repair_proposals(
            [source_cell]
        ):
            proposal = search._change_delta(replacements)
            if proposal is None:
                continue
            _delta, changes, _new_rows = proposal
            family, index, lower, upper = source_cell
            current = int(search.arrays[family][index])
            change = changes.get(source_key, 0)
            if (current < lower and change > 0) or (current > upper and change < 0):
                found_helpful = True
                break
        self.assertTrue(found_helpful)

    def test_targeted_strict_improver_matches_independent_rebuild(self) -> None:
        search = negative_constructor.JointSearch(
            self.train, self.joint_signatures, split="train"
        )
        violated = [
            cell
            for cell in search._constraint_cells()
            if not cell[2] <= int(search.arrays[cell[0]][cell[1]]) <= cell[3]
        ]
        selected = None
        for _source_key, _kind, replacements in search._targeted_repair_proposals(
            violated[:8]
        ):
            proposal = search._change_delta(replacements)
            if proposal is not None and proposal[0] < 0:
                selected = proposal
                break
        self.assertIsNotNone(selected)
        declared_delta, _changes, new_rows = selected
        before = search._full_objective()
        for world, row in new_rows.items():
            search.assignments[world] = row
        for array in search.arrays.values():
            array.fill(0)
        for world, row in enumerate(search.assignments):
            search._accumulate_world(world, row, 1)
        after = search._full_objective()
        self.assertLess(after, before)
        self.assertEqual(after - before, declared_delta)

    def test_same_position_cross_world_swap_preserves_seller_marginals(self) -> None:
        search = negative_constructor.JointSearch(
            self.train, self.joint_signatures, split="train"
        )
        proposal = None
        for position in range(12):
            for first_world in range(40):
                for second_world in range(first_world + 1, 40):
                    first = int(search.assignments[first_world, position])
                    second = int(search.assignments[second_world, position])
                    if first == second:
                        continue
                    proposal = search._change_delta(
                        [
                            (first_world, position, second),
                            (second_world, position, first),
                        ]
                    )
                    if proposal is not None:
                        break
                if proposal is not None:
                    break
            if proposal is not None:
                break
        self.assertIsNotNone(proposal)
        _objective_delta, changes, _new_rows = proposal
        self.assertFalse(
            any(
                family in {"role_seller", "endpoint_seller"}
                for family, _index in changes
            )
        )

    def test_same_role_within_world_swap_preserves_role_endpoint_marginals(self) -> None:
        search = negative_constructor.JointSearch(
            self.train, self.joint_signatures, split="train"
        )
        proposal = None
        for world in range(50):
            row = search.assignments[world]
            for first_position in range(12):
                for second_position in range(first_position + 1, 12):
                    if (
                        negative_constructor.ROLE_BY_POSITION[first_position]
                        != negative_constructor.ROLE_BY_POSITION[second_position]
                    ):
                        continue
                    proposal = search._change_delta(
                        [
                            (world, first_position, int(row[second_position])),
                            (world, second_position, int(row[first_position])),
                        ]
                    )
                    if proposal is not None:
                        break
                if proposal is not None:
                    break
            if proposal is not None:
                break
        self.assertIsNotNone(proposal)
        _objective_delta, changes, _new_rows = proposal
        self.assertFalse(
            any(
                family
                in {
                    "role_seller",
                    "role_noise",
                    "endpoint_seller",
                    "endpoint_noise",
                    "role_triad",
                    "role_size_seller",
                    "role_size_noise",
                }
                for family, _index in changes
            )
        )

    def test_semantic_domain_is_replayed_from_the_pinned_policy(self) -> None:
        audit = negative_plan.replay_semantic_public_domain()
        self.assertEqual(
            audit["domain_sha256"], negative_plan.SEMANTIC_DOMAIN_SHA256
        )
        self.assertEqual(audit["category_product_counts"], [2, 1, 2, 2, 1, 2, 1, 1])
        self.assertEqual(audit["attribute_count"], 10)
        expected_counts = negative_plan._largest_remainder_counts(
            2000, negative_plan.SEMANTIC_CATEGORY_WEIGHTS
        )
        self.assertEqual(expected_counts, (559, 383, 254, 117, 82, 81, 76, 448))
        train_sequence = negative_plan._semantic_asset_sequence("train")
        development_sequence = negative_plan._semantic_asset_sequence("development")
        self.assertEqual(len(train_sequence), 2000)
        self.assertNotEqual(train_sequence, development_sequence)
        self.assertEqual(
            tuple(
                sum(int(row[0] == category) for row in train_sequence)
                for category in range(8)
            ),
            expected_counts,
        )
        self.assertEqual(
            negative_constructor._semantic_asset_sequence("train"), train_sequence
        )

    def test_global_relabel_audit_detects_bijection_and_conflict(self) -> None:
        left = list(range(28)) * 3
        shifted = [int((value + 7) % 28) for value in left]
        audit = negative_plan._global_relabel_audit(
            left, shifted, cardinality=28
        )
        self.assertTrue(audit["complete_global_relabel"])
        shifted[-1] = shifted[-2]
        audit = negative_plan._global_relabel_audit(
            left, shifted, cardinality=28
        )
        self.assertFalse(audit["complete_global_relabel"])
        self.assertGreater(audit["mapping_conflict_count"], 0)

    def test_option_incidence_exactly_replays_all_global_count_arrays(self) -> None:
        search = negative_constructor.JointSearch(
            self.train, self.joint_signatures, split="train"
        )
        replayed: defaultdict[tuple[str, tuple[int, ...]], int] = defaultdict(int)
        for world, row in enumerate(search.assignments):
            for pair_index, (left_position, right_position) in enumerate(
                negative_constructor.PAIR_POSITIONS
            ):
                for key, count in search._option_contributions(
                    world,
                    pair_index,
                    int(row[left_position]),
                    int(row[right_position]),
                ).items():
                    replayed[key] += count
        for family, index, _lower, _upper in search._constraint_cells():
            self.assertEqual(
                replayed[(family, index)],
                int(search.arrays[family][index]),
                msg=f"incidence drift for {family}{index}",
            )

    def test_coarse_milp_candidate_solver_closes_toy_bounds(self) -> None:
        result = negative_constructor._solve_sparse_candidate_milp(
            current_bounds=[(0, 1, 1), (2, 1, 1)],
            candidate_cell_changes=[{0: 1}, {1: -1}],
            candidate_worlds=[(0,), (1,)],
            candidate_costs=[1.0, 1.0],
            time_limit_seconds=10,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["selected_candidate_indices"], [0, 1])
        self.assertEqual(result["total_slack"], 0)

    def test_candidate_solver_slack_objective_is_lexicographic(self) -> None:
        result = negative_constructor._solve_sparse_candidate_milp(
            current_bounds=[(0, 1, 1)],
            candidate_cell_changes=[{0: 1}, {}],
            candidate_worlds=[(0,), (1,)],
            candidate_costs=[100_000.0, 1.0],
            time_limit_seconds=10,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["total_slack"], 0)
        self.assertEqual(result["selected_candidate_indices"], [0])
    def test_targeted_column_collection_is_full_delta_deduplicated(self) -> None:
        search = negative_constructor.JointSearch(
            self.train, self.joint_signatures, split="train"
        )
        cells = search._constraint_cells()
        cell_index = {
            (family, index): ordinal
            for ordinal, (family, index, _lower, _upper) in enumerate(cells)
        }
        violated = [
            cell
            for cell in cells
            if negative_constructor.JointSearch._cell_violation(
                int(search.arrays[cell[0]][cell[1]]), cell[2], cell[3]
            )
        ]
        candidates, audit = search._collect_targeted_candidate_columns(
            target_cells=violated[:2],
            cells=cells,
            cell_index=cell_index,
        )
        self.assertGreater(audit["enumerated_proposal_count"], 0)
        self.assertGreater(len(candidates), 0)
        self.assertEqual(audit["omitted_by_search_budget_count"], 0)
        equivalence_keys = [
            search._candidate_equivalence_key(candidate, cell_index)
            for candidate in candidates
        ]
        self.assertEqual(len(equivalence_keys), len(set(equivalence_keys)))
        for changes, _new_rows, objective_delta, _kind in candidates:
            self.assertTrue(set(changes).issubset(cell_index))
            self.assertEqual(
                objective_delta,
                search._objective_delta_for_changes(changes),
            )

@unittest.skipUnless(
    NEGATIVE_PREFLIGHT.is_dir(),
    "V9.3 registered-negative preflight has not been published yet",
)
class RegisteredNegativePlanContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_schedule = read_json(PREFLIGHT / "train_balanced_schedule.json")
        cls.development_schedule = read_json(
            PREFLIGHT / "development_balanced_schedule.json"
        )
        cls.joint_signatures = read_json(JOINT_SIGNATURE)
        cls.train_plan = read_json(
            NEGATIVE_PREFLIGHT / "train_registered_negative_plan.json"
        )
        cls.development_plan = read_json(
            NEGATIVE_PREFLIGHT / "development_registered_negative_plan.json"
        )
        cls.receipt = read_json(NEGATIVE_PREFLIGHT / "construction_receipt.json")

    def test_published_plans_pass_independent_full_validation(self) -> None:
        train_audit = negative_plan.validate_plan(
            self.train_plan, self.train_schedule, self.joint_signatures
        )
        development_audit = negative_plan.validate_plan(
            self.development_plan,
            self.development_schedule,
            self.joint_signatures,
        )
        self.assertEqual(
            train_audit["plan_canonical_self_sha256"],
            self.train_plan["canonical_self_sha256"],
        )
        self.assertEqual(
            development_audit["plan_canonical_self_sha256"],
            self.development_plan["canonical_self_sha256"],
        )
        self.assertNotEqual(
            train_audit["plan_canonical_self_sha256"],
            development_audit["plan_canonical_self_sha256"],
        )
        self.assertEqual(train_audit["registered_pair_count_per_world"], 6)
        self.assertEqual(train_audit["registered_endpoint_count_per_world"], 12)
        self.assertEqual(train_audit["semantic_asset_count"], 2000)
        self.assertEqual(
            train_audit["directed_pair_histograms"],
            {
                "exact_title_clone": {"1": 512, "2": 244},
                "high_semantic_similarity": {"2": 268, "3": 488},
            },
        )
        self.assertEqual(
            train_audit["role_triad_totals"],
            {
                "exact_title_clone:source": 429,
                "exact_title_clone:target": 429,
                "high_semantic_similarity:left": 857,
                "high_semantic_similarity:right": 857,
            },
        )

    def test_construction_receipt_pins_the_zero_objective_solution(self) -> None:
        for split in ("train", "development"):
            search = self.receipt[f"{split}_search"]
            self.assertGreater(search["initial_objective"], 0)
            self.assertGreater(search["accepted_moves"], 0)
            if search["solution_stage"] == "annealing_or_cold_polishing":
                self.assertGreater(search["solved_iteration"], 0)
            else:
                self.assertIsNone(search["solved_iteration"])
                self.assertEqual(
                    search["exact_local_repair"]["final_objective"], 0
                )
        self.assertEqual(
            self.receipt["canonical_self_sha256"],
            negative_plan.canonical_self_sha256(self.receipt),
        )

    def test_duplicate_endpoint_and_item_ordinal_tampering_fail_closed(self) -> None:
        for mutation in ("duplicate_endpoint", "logical_item"):
            with self.subTest(mutation=mutation):
                tampered = copy.deepcopy(self.train_plan)
                first = tampered["worlds"][0]["assignments"][0]["endpoints"]
                if mutation == "duplicate_endpoint":
                    first[1]["seller_slot"] = first[0]["seller_slot"]
                else:
                    first[0]["logical_item_ordinal"] = 99
                tampered["canonical_self_sha256"] = negative_plan.canonical_self_sha256(
                    tampered
                )
                with self.assertRaises(negative_plan.RegisteredNegativePlanError):
                    negative_plan.validate_plan(
                        tampered, self.train_schedule, self.joint_signatures
                    )

    def test_semantic_asset_tampering_fails_closed(self) -> None:
        tampered = copy.deepcopy(self.train_plan)
        semantic = tampered["worlds"][0]["assignments"][2]["semantic_asset"]
        semantic["right_title_skeleton_ordinal"] = semantic[
            "left_title_skeleton_ordinal"
        ]
        tampered["canonical_self_sha256"] = negative_plan.canonical_self_sha256(
            tampered
        )
        with self.assertRaises(negative_plan.RegisteredNegativePlanError):
            negative_plan.validate_plan(
                tampered, self.train_schedule, self.joint_signatures
            )

    def test_plan_has_no_truth_or_private_identity_fields(self) -> None:
        forbidden = {
            "label",
            "pair_label",
            "controller_uid",
            "controller_id",
            "truth",
            "identity_history",
            "seller_uid",
            "item_uid",
        }

        def walk(value: object) -> None:
            if isinstance(value, dict):
                self.assertFalse(forbidden & set(value))
                for child in value.values():
                    walk(child)
            elif isinstance(value, list):
                for child in value:
                    walk(child)

        walk(self.train_plan)
        walk(self.development_plan)


if __name__ == "__main__":
    unittest.main()
