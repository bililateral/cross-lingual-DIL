from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from dataclasses import fields, replace
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step28_v13_common as common  # noqa: E402
import step28_v13_v1_13_candidate_selection as subject  # noqa: E402
import step28_v13_v1_13_natural_variation as natural  # noqa: E402


def digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


class FakeSession:
    def __init__(self, template: natural.AssembledDevelopmentCandidate) -> None:
        self.template = template
        self.indices: list[int] = []

    def render(self, candidate_index: int) -> natural.AssembledDevelopmentCandidate:
        self.indices.append(candidate_index)
        return replace(self.template, candidate_index=candidate_index)


class Step28V13V113CandidateSelectionContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = subject.load_policy()
        cls.selector = subject.DevelopmentSmokeCandidateSelector()
        cls.accepted = cls.selector.select()
        cls.context = cls.selector._context
        cls.material = cls.selector._material
        cls.candidate = natural.AssembledDevelopmentCandidate(
            candidate_index=cls.accepted.candidate_index,
            world_bytes=cls.accepted.world_bytes,
            world_sha256=cls.accepted.world_sha256,
            profiles_bytes=cls.accepted.profiles_bytes,
            profiles_sha256=cls.accepted.profiles_sha256,
            profile_provenance_bytes=cls.accepted.profile_provenance_bytes,
            profile_provenance_sha256=cls.accepted.profile_provenance_sha256,
            identity33_bytes=cls.accepted.identity33_bytes,
            identity33_sha256=cls.accepted.identity33_sha256,
            natural_output_sha256=cls.accepted.natural_output_sha256,
            candidate_invariant_sha256=cls.accepted.candidate_invariant_sha256,
            identity_parent_sha256=cls.accepted.identity_parent_sha256,
        )
        cls.observation = subject._FinalDocumentObservation(
            redacted_items_bytes=cls.accepted.redacted_items_bytes,
            redacted_items_sha256=cls.accepted.redacted_items_sha256,
            profiles_bytes=cls.accepted.profiles_bytes,
            profiles_sha256=cls.accepted.profiles_sha256,
            item_hash_rows_bytes=cls.accepted.item_hash_rows_bytes,
            item_hash_rows_sha256=cls.accepted.item_hash_rows_sha256,
            seller_hash_rows_bytes=cls.accepted.seller_hash_rows_bytes,
            seller_hash_rows_sha256=cls.accepted.seller_hash_rows_sha256,
        )

    def test_policy_is_self_hashed_exactly_pinned_and_fully_unauthorized(self) -> None:
        policy = self.policy
        unsigned = dict(policy)
        claimed = unsigned.pop("canonical_self_hash")
        self.assertEqual(common.canonical_sha256(unsigned), claimed)
        self.assertTrue(all(value is False for value in policy["formal_authorizations"].values()))
        self.assertEqual(set(policy["frozen_inputs"]), set(subject.FROZEN_INPUT_KEYS))
        for spec in policy["frozen_inputs"].values():
            path = ROOT / spec["path"]
            self.assertEqual(path.stat().st_size, spec["size_bytes"])
            self.assertEqual(common.sha256_file(path), spec["sha256"])

    def test_frozen_input_key_cannot_be_rebound_to_another_real_file(self) -> None:
        tampered = copy.deepcopy(self.policy)
        alternate = tampered["frozen_inputs"]["candidate_selection_tests"]
        tampered["frozen_inputs"]["candidate_selection"] = copy.deepcopy(alternate)
        unsigned = dict(tampered)
        unsigned.pop("canonical_self_hash")
        tampered["canonical_self_hash"] = common.canonical_sha256(unsigned)
        with self.assertRaises(subject.CandidateSelectionError):
            subject._validate_policy(tampered)

    def test_public_selector_exposes_no_screening_or_fault_authority(self) -> None:
        self.assertEqual(list(inspect.signature(subject.DevelopmentSmokeCandidateSelector).parameters), [])
        self.assertEqual(list(inspect.signature(subject.DevelopmentSmokeCandidateSelector.select).parameters), ["self"])
        self.assertEqual(
            list(
                inspect.signature(
                    subject.DevelopmentSmokeCandidateSelector.validate_completed_candidate
                ).parameters
            ),
            ["self", "value"],
        )
        with self.assertRaises(TypeError):
            subject.DevelopmentSmokeCandidateSelector(history={digest("x")})  # type: ignore[call-arg]

    def test_smoke_context_binds_history_and_explicit_empty_states(self) -> None:
        value = subject._selection_context_value(self.context)
        expected = self.policy["smoke_collision_context"]
        self.assertEqual(value["historical_item"]["count"], expected["historical_item_count"])
        self.assertEqual(value["historical_item"]["hashes_sha256"], expected["historical_item_hashes_sha256"])
        self.assertEqual(value["historical_seller"]["count"], expected["historical_seller_count"])
        self.assertEqual(value["historical_seller"]["hashes_sha256"], expected["historical_seller_hashes_sha256"])
        for name in ("current_item", "current_seller", "predecessor_item", "predecessor_seller"):
            self.assertEqual(value[name]["count"], 0)
        self.assertEqual(value["predecessor_seal_pins"], [])
        self.assertEqual(
            hashlib.sha256(common.canonical_json_bytes(value)).hexdigest(),
            self.accepted.selection_context_sha256,
        )

    def test_title_clone_qualification_is_pre_candidate_complete(self) -> None:
        value = json.loads(self.material.exact_title_clone_qualification_bytes)
        self.assertEqual(value["row_count"], 2)
        self.assertEqual(
            hashlib.sha256(self.material.exact_title_clone_qualification_bytes).hexdigest(),
            self.material.exact_title_clone_qualification_sha256,
        )
        for row in value["rows"]:
            self.assertEqual(row["override_kind"], "exact_title_clone")
            self.assertIs(row["source_title_nonempty"], True)
            self.assertIs(row["target_title_nonempty"], True)
            self.assertIs(row["target_description_nonempty"], True)
            self.assertIs(row["source_unused_before_registration"], True)
            self.assertIs(row["target_unused_before_registration"], True)
            self.assertEqual(
                row["candidate_parent_full_state_sha256"],
                self.material.candidate_parent_full_state_sha256,
            )

    def test_final_hashes_use_production_redacted_rows_and_verified_profiles(self) -> None:
        redacted = json.loads(self.accepted.redacted_items_bytes)
        item_rows = json.loads(self.accepted.item_hash_rows_bytes)
        self.assertEqual(len(redacted), 105)
        self.assertEqual(len(item_rows), 105)
        for item, row in zip(redacted, item_rows, strict=True):
            self.assertEqual(item["item_uid"], row["item_uid"])
            self.assertEqual(
                subject.collision.item_document_hash(
                    title=item["title"], description=item["description"]
                ),
                row["document_sha256"],
            )
        profiles = json.loads(self.accepted.profiles_bytes)
        seller_rows = json.loads(self.accepted.seller_hash_rows_bytes)
        self.assertEqual(len(profiles), 28)
        for profile, row in zip(profiles, seller_rows, strict=True):
            self.assertEqual(profile["seller_uid"], row["seller_uid"])
            self.assertEqual(
                subject.collision.seller_document_hash(profile),
                row["document_sha256"],
            )

    def test_redacted_observation_does_not_expose_raw_identity_values(self) -> None:
        world = self.candidate.thaw_world()
        raw_values = {
            str(row["identity_value"])
            for row in world["private"]["identity_assets"]
        }
        visible = self.accepted.redacted_items_bytes.decode("utf-8")
        self.assertTrue(raw_values)
        self.assertFalse(any(value and value in visible for value in raw_values))

    def test_accepted_carries_full_sorted_registry_and_identity_deltas(self) -> None:
        accepted = self.accepted
        for values, count, claimed in (
            (
                accepted.item_registry_delta,
                accepted.item_registry_delta_count,
                accepted.item_registry_delta_sha256,
            ),
            (
                accepted.seller_registry_delta,
                accepted.seller_registry_delta_count,
                accepted.seller_registry_delta_sha256,
            ),
            (
                accepted.allocation_delta,
                accepted.allocation_delta_count,
                accepted.allocation_delta_sha256,
            ),
        ):
            self.assertEqual(values, tuple(sorted(values)))
            self.assertEqual(len(values), len(set(values)))
            self.assertEqual(len(values), count)
            self.assertEqual(common.canonical_sha256(list(values)), claimed)
        self.assertEqual(accepted.item_registry_delta_count, 105)
        self.assertEqual(accepted.seller_registry_delta_count, 28)
        self.assertGreater(accepted.allocation_delta_count, 0)

    def test_accepted_projection_covers_every_field_except_its_root(self) -> None:
        projected = subject._accepted_state_projection(self.accepted)
        dataclass_fields = {field.name for field in fields(self.accepted)}
        self.assertEqual(set(projected), dataclass_fields - {"accepted_state_sha256"})
        self.assertEqual(
            common.canonical_sha256(projected), self.accepted.accepted_state_sha256
        )

    def test_smoke_selection_matches_frozen_golden(self) -> None:
        expected = self.policy["expected_smoke_selection"]
        observed = {
            "world_uid": self.accepted.world_uid,
            "accepted_candidate_index": self.accepted.candidate_index,
            "candidates_examined": self.accepted.candidates_examined,
            "rejected_candidate_count": self.accepted.rejected_candidate_count,
            "item_count": self.accepted.item_registry_delta_count,
            "seller_count": self.accepted.seller_registry_delta_count,
            "selection_context_sha256": self.accepted.selection_context_sha256,
            "exact_title_clone_qualification_sha256": self.accepted.exact_title_clone_qualification_sha256,
            "redacted_items_sha256": self.accepted.redacted_items_sha256,
            "profiles_sha256": self.accepted.profiles_sha256,
            "item_registry_delta_sha256": self.accepted.item_registry_delta_sha256,
            "seller_registry_delta_sha256": self.accepted.seller_registry_delta_sha256,
            "allocation_delta_count": self.accepted.allocation_delta_count,
            "allocation_delta_sha256": self.accepted.allocation_delta_sha256,
            "accepted_state_sha256": self.accepted.accepted_state_sha256,
        }
        self.assertEqual(observed, expected)

    def _synthetic_context(
        self,
        *,
        historical_item: tuple[str, ...] = (),
        historical_seller: tuple[str, ...] = (),
        current_item: tuple[str, ...] = (),
        current_seller: tuple[str, ...] = (),
        predecessor_item: tuple[str, ...] = (),
        predecessor_seller: tuple[str, ...] = (),
    ) -> subject.FrozenCollisionContext:
        return subject.FrozenCollisionContext(
            mode=subject.ALLOWED_MODE,
            split=subject.ALLOWED_SPLIT,
            world_ordinal=0,
            historical_item_hashes=historical_item,
            historical_seller_hashes=historical_seller,
            current_item_hashes=current_item,
            current_seller_hashes=current_seller,
            predecessor_item_hashes=predecessor_item,
            predecessor_seller_hashes=predecessor_seller,
            previous_world_marker_sha256=digest("no previous marker"),
            predecessor_seal_pins=(),
        )

    @staticmethod
    def _synthetic_rows(
        *, item_hashes: list[str], seller_hashes: list[str]
    ) -> tuple[bytes, bytes]:
        items = [
            {"row_ordinal": index, "item_uid": f"item-{index}", "document_sha256": value}
            for index, value in enumerate(item_hashes)
        ]
        sellers = [
            {"row_ordinal": index, "seller_uid": f"seller-{index}", "document_sha256": value}
            for index, value in enumerate(seller_hashes)
        ]
        return common.canonical_json_bytes(items), common.canonical_json_bytes(sellers)

    def test_pure_classifier_covers_all_eight_categories(self) -> None:
        item_a, item_b = digest("item-a"), digest("item-b")
        seller_a, seller_b = digest("seller-a"), digest("seller-b")
        cases = {
            "same_world_item_document": ([item_a, item_a], [seller_a, seller_b], self._synthetic_context()),
            "same_world_seller_document": ([item_a, item_b], [seller_a, seller_a], self._synthetic_context()),
            "historical_item_document": ([item_a, item_b], [seller_a, seller_b], self._synthetic_context(historical_item=(item_a,))),
            "historical_seller_document": ([item_a, item_b], [seller_a, seller_b], self._synthetic_context(historical_seller=(seller_a,))),
            "current_split_item_document": ([item_a, item_b], [seller_a, seller_b], self._synthetic_context(current_item=(item_a,))),
            "current_split_seller_document": ([item_a, item_b], [seller_a, seller_b], self._synthetic_context(current_seller=(seller_a,))),
            "predecessor_item_document": ([item_a, item_b], [seller_a, seller_b], self._synthetic_context(predecessor_item=(item_a,))),
            "predecessor_seller_document": ([item_a, item_b], [seller_a, seller_b], self._synthetic_context(predecessor_seller=(seller_a,))),
        }
        for expected, (items, sellers, context) in cases.items():
            with self.subTest(category=expected):
                item_bytes, seller_bytes = self._synthetic_rows(item_hashes=items, seller_hashes=sellers)
                result = subject._classify_document_collisions(
                    item_hash_rows_bytes=item_bytes,
                    seller_hash_rows_bytes=seller_bytes,
                    context=context,
                )
                self.assertEqual(result.categories, (expected,))

    def test_pure_classifier_reports_multiple_categories_for_one_candidate(self) -> None:
        item = digest("same item")
        seller = digest("same seller")
        item_bytes, seller_bytes = self._synthetic_rows(
            item_hashes=[item, item], seller_hashes=[seller, seller]
        )
        context = self._synthetic_context(
            historical_item=(item,), current_seller=(seller,)
        )
        result = subject._classify_document_collisions(
            item_hash_rows_bytes=item_bytes,
            seller_hash_rows_bytes=seller_bytes,
            context=context,
        )
        self.assertEqual(
            result.categories,
            (
                "same_world_item_document",
                "same_world_seller_document",
                "historical_item_document",
                "current_split_seller_document",
            ),
        )

    def test_classifier_result_must_bind_categories_to_all_hit_counts(self) -> None:
        forged = subject.CollisionClassification(
            categories=("historical_item_document",),
            hit_counts=tuple((name, 0) for name in subject.COLLISION_CATEGORIES),
        )
        with self.assertRaises(subject.CandidateSelectionError):
            subject._validate_collision_classification(forged)

    def test_malformed_classifier_result_poison_selector_before_next_candidate(self) -> None:
        selector = self._state_selector()
        forged = subject.CollisionClassification(
            categories=("historical_item_document",),
            hit_counts=tuple((name, 0) for name in subject.COLLISION_CATEGORIES),
        )
        with (
            mock.patch.object(
                subject,
                "_replay_final_document_observation",
                return_value=self.observation,
            ),
            mock.patch.object(
                subject,
                "_classify_document_collisions",
                return_value=forged,
            ),
        ):
            with self.assertRaises(subject.CandidateSelectionError):
                selector.select()
        self.assertEqual(selector._session.indices, [0])
        self.assertTrue(selector._failed)

    def _state_selector(self) -> subject.DevelopmentSmokeCandidateSelector:
        selector = subject.DevelopmentSmokeCandidateSelector.__new__(
            subject.DevelopmentSmokeCandidateSelector
        )
        selector._policy = self.policy
        selector._context = self.context
        selector._material = self.material
        selector._session = FakeSession(self.candidate)
        selector._failed = False
        selector._completed = False
        selector._trusted_accepted_candidate = None
        selector._trusted_accepted_state_sha256 = None
        return selector

    @staticmethod
    def _classification(*categories: str) -> subject.CollisionClassification:
        return subject.CollisionClassification(
            categories=tuple(categories),
            hit_counts=tuple(
                (name, int(name in categories)) for name in subject.COLLISION_CATEGORIES
            ),
        )

    @staticmethod
    def _recommit(
        value: subject.AcceptedDevelopmentCandidate,
    ) -> subject.AcceptedDevelopmentCandidate:
        provisional = replace(value, accepted_state_sha256="0" * 64)
        return replace(
            provisional,
            accepted_state_sha256=common.canonical_sha256(
                subject._accepted_state_projection(provisional)
            ),
        )

    def test_selector_retries_collision_zero_and_accepts_candidate_one(self) -> None:
        selector = self._state_selector()
        with (
            mock.patch.object(
                subject,
                "_replay_final_document_observation",
                return_value=self.observation,
            ),
            mock.patch.object(
                subject,
                "_classify_document_collisions",
                side_effect=[
                    self._classification("historical_item_document"),
                    self._classification(),
                    self._classification(),
                ],
            ),
        ):
            accepted = selector.select()
        self.assertEqual(accepted.candidate_index, 1)
        self.assertEqual(accepted.candidates_examined, 2)
        self.assertEqual(selector._session.indices, [0, 1])
        counts = json.loads(accepted.rejection_counts_bytes)
        self.assertEqual(counts["historical_item_document"], 1)

    def test_selector_exhaustion_is_permanent_and_visits_exactly_32(self) -> None:
        selector = self._state_selector()
        with (
            mock.patch.object(
                subject,
                "_replay_final_document_observation",
                return_value=self.observation,
            ),
            mock.patch.object(
                subject,
                "_classify_document_collisions",
                return_value=self._classification("same_world_item_document"),
            ),
        ):
            with self.assertRaises(subject.CandidateSelectionError):
                selector.select()
        self.assertEqual(selector._session.indices, list(range(32)))
        self.assertTrue(selector._failed)
        with self.assertRaises(subject.CandidateSelectionError):
            selector.select()

    def test_fatal_observation_error_wins_before_collision_classifier(self) -> None:
        selector = self._state_selector()
        classifier = mock.Mock(return_value=self._classification("historical_item_document"))
        with (
            mock.patch.object(
                subject,
                "_replay_final_document_observation",
                side_effect=RuntimeError("fatal observation"),
            ),
            mock.patch.object(subject, "_classify_document_collisions", classifier),
        ):
            with self.assertRaises(RuntimeError):
                selector.select()
        classifier.assert_not_called()
        self.assertEqual(selector._session.indices, [0])
        self.assertTrue(selector._failed)

    def test_selector_is_poisoned_by_keyboard_interrupt_and_system_exit(self) -> None:
        for failure in (KeyboardInterrupt(), SystemExit(7)):
            selector = self._state_selector()
            with mock.patch.object(
                subject,
                "_replay_final_document_observation",
                side_effect=failure,
            ):
                with self.assertRaises(type(failure)):
                    selector.select()
            self.assertTrue(selector._failed)
            with self.assertRaises(subject.CandidateSelectionError):
                selector.select()

    def test_successful_selector_cannot_be_reentered(self) -> None:
        selector = self._state_selector()
        with (
            mock.patch.object(
                subject,
                "_replay_final_document_observation",
                return_value=self.observation,
            ),
            mock.patch.object(
                subject,
                "_classify_document_collisions",
                return_value=self._classification(),
            ),
        ):
            selector.select()
        self.assertTrue(selector._completed)
        with self.assertRaises(subject.CandidateSelectionError):
            selector.select()

    def test_completed_selector_revalidates_against_retained_authority(self) -> None:
        selector = self._state_selector()
        selector._completed = True
        selector._trusted_accepted_candidate = self.candidate
        selector._trusted_accepted_state_sha256 = self.accepted.accepted_state_sha256
        with (
            mock.patch.object(
                subject,
                "_replay_final_document_observation",
                return_value=self.observation,
            ),
            mock.patch.object(
                subject,
                "_classify_document_collisions",
                return_value=self._classification(),
            ),
        ):
            selector.validate_completed_candidate(self.accepted)
        self.assertFalse(selector._failed)

    def test_completed_selector_rejects_private_world_forgery_and_is_poisoned(self) -> None:
        selector = self._state_selector()
        selector._completed = True
        selector._trusted_accepted_candidate = self.candidate
        selector._trusted_accepted_state_sha256 = self.accepted.accepted_state_sha256
        world = json.loads(self.accepted.world_bytes)
        world["private"]["negative_flags"][0]["flag"] += "_forged"
        world_bytes = common.canonical_json_bytes(world)
        forged = self._recommit(
            replace(
                self.accepted,
                world_bytes=world_bytes,
                world_sha256=hashlib.sha256(world_bytes).hexdigest(),
            )
        )
        with self.assertRaises(subject.CandidateSelectionError):
            selector.validate_completed_candidate(forged)
        self.assertTrue(selector._failed)
        with self.assertRaises(subject.CandidateSelectionError):
            selector.validate_completed_candidate(self.accepted)

    def test_completed_selector_rejects_recomputed_rejection_history_root(self) -> None:
        selector = self._state_selector()
        selector._completed = True
        selector._trusted_accepted_candidate = self.candidate
        selector._trusted_accepted_state_sha256 = self.accepted.accepted_state_sha256
        rejection_counts = json.loads(self.accepted.rejection_counts_bytes)
        rejection_counts["historical_item_document"] = 1
        rejection_bytes = common.canonical_json_bytes(rejection_counts)
        forged = self._recommit(
            replace(
                self.accepted,
                rejection_counts_bytes=rejection_bytes,
                rejection_counts_sha256=hashlib.sha256(rejection_bytes).hexdigest(),
            )
        )
        self.assertNotEqual(
            forged.accepted_state_sha256,
            selector._trusted_accepted_state_sha256,
        )
        with self.assertRaises(subject.CandidateSelectionError):
            selector.validate_completed_candidate(forged)
        self.assertTrue(selector._failed)

    def test_accepted_context_allocation_and_payload_forgery_fail(self) -> None:
        for forged in (
            replace(self.accepted, allocation_delta=self.accepted.allocation_delta[:-1]),
            replace(self.accepted, selection_context_bytes=common.canonical_json_bytes({})),
            replace(self.accepted, redacted_items_bytes=self.accepted.redacted_items_bytes + b"\n"),
            replace(self.accepted, accepted_state_sha256="0" * 64),
        ):
            with self.subTest(field_drift=forged.accepted_state_sha256):
                with self.assertRaises(subject.CandidateSelectionError):
                    subject._validate_accepted_candidate(
                        forged,
                        context=self.context,
                        material=self.material,
                        trusted_candidate=self.candidate,
                    )

        self_consistent_wrong_root = replace(
            self.accepted,
            candidate_parent_full_state_sha256="0" * 64,
            accepted_state_sha256="0" * 64,
        )
        self_consistent_wrong_root = replace(
            self_consistent_wrong_root,
            accepted_state_sha256=common.canonical_sha256(
                subject._accepted_state_projection(self_consistent_wrong_root)
            ),
        )
        with self.assertRaises(subject.CandidateSelectionError):
            subject._validate_accepted_candidate(
                self_consistent_wrong_root,
                context=self.context,
                material=self.material,
                trusted_candidate=self.candidate,
            )

    def test_self_consistent_historical_collision_and_decoupled_rows_fail(self) -> None:
        item_rows = json.loads(self.accepted.item_hash_rows_bytes)
        item_rows[0]["document_sha256"] = self.context.historical_item_hashes[0]
        item_rows_bytes = common.canonical_json_bytes(item_rows)
        item_delta = tuple(sorted(row["document_sha256"] for row in item_rows))
        forged = replace(
            self.accepted,
            item_hash_rows_bytes=item_rows_bytes,
            item_hash_rows_sha256=hashlib.sha256(item_rows_bytes).hexdigest(),
            item_registry_delta=item_delta,
            item_registry_delta_count=len(item_delta),
            item_registry_delta_sha256=common.canonical_sha256(list(item_delta)),
        )
        forged = self._recommit(forged)
        with self.assertRaises(subject.CandidateSelectionError):
            subject._validate_accepted_candidate(
                forged,
                context=self.context,
                material=self.material,
                trusted_candidate=self.candidate,
            )

    def test_self_consistent_short_row_sets_fail(self) -> None:
        item_rows = json.loads(self.accepted.item_hash_rows_bytes)[:-1]
        seller_rows = json.loads(self.accepted.seller_hash_rows_bytes)[:-1]
        item_rows_bytes = common.canonical_json_bytes(item_rows)
        seller_rows_bytes = common.canonical_json_bytes(seller_rows)
        item_delta = tuple(sorted(row["document_sha256"] for row in item_rows))
        seller_delta = tuple(sorted(row["document_sha256"] for row in seller_rows))
        forged = replace(
            self.accepted,
            item_hash_rows_bytes=item_rows_bytes,
            item_hash_rows_sha256=hashlib.sha256(item_rows_bytes).hexdigest(),
            seller_hash_rows_bytes=seller_rows_bytes,
            seller_hash_rows_sha256=hashlib.sha256(seller_rows_bytes).hexdigest(),
            item_registry_delta=item_delta,
            item_registry_delta_count=len(item_delta),
            item_registry_delta_sha256=common.canonical_sha256(list(item_delta)),
            seller_registry_delta=seller_delta,
            seller_registry_delta_count=len(seller_delta),
            seller_registry_delta_sha256=common.canonical_sha256(list(seller_delta)),
        )
        forged = self._recommit(forged)
        with self.assertRaises(subject.CandidateSelectionError):
            subject._validate_accepted_candidate(
                forged,
                context=self.context,
                material=self.material,
                trusted_candidate=self.candidate,
            )

    def test_self_consistent_invalid_natural_output_hash_fails(self) -> None:
        forged = self._recommit(
            replace(self.accepted, natural_output_sha256="not-a-sha")
        )
        with self.assertRaises(subject.CandidateSelectionError):
            subject._validate_accepted_candidate(
                forged,
                context=self.context,
                material=self.material,
                trusted_candidate=self.candidate,
            )

    def test_bool_row_ordinal_is_not_an_integer(self) -> None:
        rows = json.loads(self.accepted.item_hash_rows_bytes)
        rows[0]["row_ordinal"] = False
        with self.assertRaises(subject.CandidateSelectionError):
            subject._classify_document_collisions(
                item_hash_rows_bytes=common.canonical_json_bytes(rows),
                seller_hash_rows_bytes=self.accepted.seller_hash_rows_bytes,
                context=self.context,
            )

    def test_same_world_duplicate_is_classified_before_multiplicity_gate(self) -> None:
        rows = json.loads(self.accepted.item_hash_rows_bytes)
        rows[1]["document_sha256"] = rows[0]["document_sha256"]
        classification = subject._classify_document_collisions(
            item_hash_rows_bytes=common.canonical_json_bytes(rows),
            seller_hash_rows_bytes=self.accepted.seller_hash_rows_bytes,
            context=self.context,
        )
        self.assertIn("same_world_item_document", classification.categories)

    def test_independent_profile_replay_rejects_forged_profile_bytes(self) -> None:
        forged = replace(
            self.candidate,
            profiles_bytes=common.canonical_json_bytes([]),
            profiles_sha256=common.canonical_sha256([]),
        )
        with self.assertRaises(subject.CandidateSelectionError):
            subject._replay_final_document_observation(forged)

    def test_source_has_no_filesystem_mutation_or_formal_generation_calls(self) -> None:
        path = SCRIPTS / "step28_v13_v1_13_candidate_selection.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        forbidden = {"write_text", "write_bytes", "mkdir", "unlink", "replace", "rename", "rmdir"}
        calls = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(calls & forbidden)

    def test_main_prints_only_noncommittable_zero_formal_summary(self) -> None:
        stream = io.StringIO()
        with mock.patch.object(
            subject.DevelopmentSmokeCandidateSelector,
            "select",
            return_value=self.accepted,
        ), redirect_stdout(stream):
            subject.main()
        receipt = json.loads(stream.getvalue())
        self.assertIs(receipt["design_smoke_only"], True)
        self.assertIs(receipt["committable"], False)
        self.assertEqual(receipt["formal_seeds_generated"], 0)
        self.assertEqual(receipt["formal_rows_generated"], 0)
        self.assertEqual(receipt["formal_transactions_written"], 0)
        self.assertEqual(receipt["formal_models_trained"], 0)
        self.assertNotIn("allocation_delta", receipt)


if __name__ == "__main__":
    unittest.main()
