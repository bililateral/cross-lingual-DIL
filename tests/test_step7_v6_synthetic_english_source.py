import csv
import hashlib
import json
import sys
import unittest
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import step7_v5_build_english_source_dataset as v5
import step7_v6_build_synthetic_english_source as v6


def read_csv(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Step7V6ConstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy = v6.load_policy()
        strong = read_csv(ROOT / "suspected_sockpuppet_strong.csv")
        registry, _ = v5.parse_vendor_registry()
        fingerprint_vendor_ids, vendor_fingerprints, _, _ = v5.parse_auxiliary_pgp()
        weak = read_csv(ROOT / "suspected_sockpuppet_weak.csv")
        global_identity = v6.build_global_identity_components(
            registry, vendor_fingerprints, weak, strong
        )
        holdout, _, holdout_tokens, _, holdout_components = (
            v6.recover_v5_holdout_fingerprints(
            cls.policy["construction"],
            registry,
            global_identity,
            )
        )
        cls.groups, cls.topology_audit = v6.build_valid_topology_groups(
            cls.policy,
            strong,
            registry,
            fingerprint_vendor_ids,
            vendor_fingerprints,
            global_identity,
            holdout,
            holdout_tokens,
            holdout_components,
        )

    def test_topology_count_is_the_recomputed_clean_count(self):
        self.assertEqual(len(self.groups), 460)
        self.assertEqual(self.topology_audit["selected_component_reuse"], 0)
        self.assertEqual(self.topology_audit["selected_holdout_evidence_overlaps"], 0)
        self.assertEqual(sum(map(len, self.groups.values())), 1015)
        self.assertEqual(
            sum(len(value) * (len(value) - 1) // 2 for value in self.groups.values()),
            650,
        )

    def test_partition_has_exact_controller_capacities(self):
        sizes = {key: len(value) for key, value in self.groups.items()}
        split = v6.balanced_partition(
            sizes,
            self.policy["construction"]["split_controller_counts"],
            self.policy["construction"]["split_seed"],
        )
        self.assertEqual(
            Counter(split.values()),
            Counter(self.policy["construction"]["split_controller_counts"]),
        )

    def test_global_identity_graph_keeps_ineligible_bridge_nodes(self):
        fingerprint_one = "1" * 40
        fingerprint_two = "2" * 40
        registry = {
            ("1", "1"): {"user_name": "left", "imposter": 0},
            ("2", "2"): {"user_name": "bridge", "imposter": 0},
            ("3", "3"): {"user_name": "right", "imposter": 0},
        }
        global_identity = v6.build_global_identity_components(
            registry,
            {
                "1": {fingerprint_one},
                "2": {fingerprint_one, fingerprint_two},
                "3": {fingerprint_two},
            },
            [],
            [],
        )
        components = global_identity["component_by_key"]
        self.assertEqual(components[("1", "1")], components[("2", "2")])
        self.assertEqual(components[("2", "2")], components[("3", "3")])

    def test_donor_topology_disjointness_uses_components_and_all_evidence(self):
        topology = [
            {
                "conflict_component_uid": "topology-component",
                "conflict_evidence_tokens": {("weak_pgp", "shared")},
            }
        ]
        base_donor = {
            "component_uid": "donor-component",
            "identity_fingerprints": {"different-main-fingerprint"},
            "identity_evidence_tokens": {("auxiliary_pgp", "donor-only")},
        }
        self.assertTrue(
            v6.donor_is_disjoint_from_topology(base_donor, topology, "topology-main")
        )
        same_component = {**base_donor, "component_uid": "topology-component"}
        self.assertFalse(
            v6.donor_is_disjoint_from_topology(
                same_component, topology, "topology-main"
            )
        )
        shared_weak_evidence = {
            **base_donor,
            "identity_evidence_tokens": {("weak_pgp", "shared")},
        }
        self.assertFalse(
            v6.donor_is_disjoint_from_topology(
                shared_weak_evidence, topology, "topology-main"
            )
        )

    def test_item_allocator_is_disjoint_and_balanced(self):
        items = []
        for index in range(30):
            items.append(
                {
                    "source_row_number": index + 2,
                    "title_clean": f"Title words {index}",
                    "description_clean": "description words " * (index + 1),
                }
            )
        buckets = v6.allocate_items(items, 3, 8, "unit-test", minimum_tokens=250)
        self.assertEqual([len(value) for value in buckets], [8, 8, 8])
        self.assertTrue(
            all(sum(v6.item_token_count(row) for row in bucket) >= 250 for bucket in buckets)
        )
        observed = [row["source_row_number"] for value in buckets for row in value]
        self.assertEqual(len(observed), len(set(observed)))

    def test_short_near_copy_is_not_silently_treated_as_zero(self):
        self.assertTrue(
            v6.fields_are_near_duplicates(
                "premiumcocaine1g",
                "premiumcocaine2g",
                long_width=10,
                long_threshold=0.90,
            )
        )
        self.assertFalse(
            v6.fields_are_near_duplicates(
                "premiumcocaine1g",
                "freshmushroomkit",
                long_width=10,
                long_threshold=0.90,
            )
        )

    def test_short_source_alias_requires_a_real_token_boundary(self):
        self.assertEqual(v6.source_alias_residuals("someone", {"one"}, 3), set())
        self.assertEqual(
            v6.source_alias_residuals("contact o.n.e now", {"one"}, 3),
            {"one"},
        )

    def test_candidate_pool_preserves_real_within_account_templates(self):
        rows = [
            {
                "source_row_number": 2,
                "title_clean": "premium cocaine 1g",
                "description_clean": "first independent description with many details",
                "category_clean": "one",
                "title_key": "premium cocaine 1g",
                "description_key": "first independent description with many details",
            },
            {
                "source_row_number": 3,
                "title_clean": "premium cocaine 2g",
                "description_clean": "second unrelated narrative written for another product",
                "category_clean": "two",
                "title_key": "premium cocaine 2g",
                "description_key": "second unrelated narrative written for another product",
            },
        ]
        selected = v6.select_copy_distinct_items(rows, 2, "field-only-test")
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(row["title_clean"] for row in selected))
        self.assertTrue(all(row["description_clean"] for row in selected))

    def test_near_template_cleanup_is_cross_account_not_within_account(self):
        buckets = [
            [
                {"title_style": "W15 W17 N11", "description_style": "W12 W17 N11"},
                {"title_style": "W15 W17 N12", "description_style": "W12 W17 N12"},
            ],
            [
                {"title_style": "W15 W17 N13", "description_style": "N2 / W6"},
                {"title_style": "W9 W2", "description_style": "W10 W5"},
            ],
        ]
        audit = v6.clear_cross_bucket_near_fields(
            buckets, ("title_style", "description_style")
        )
        self.assertEqual(audit["rows_cleared_by_field"], {"title_style": 3})
        self.assertEqual(
            [row["title_style"] for row in buckets[0]], ["", ""]
        )
        self.assertEqual(
            [row["title_style"] for row in buckets[1]], ["", "W9 W2"]
        )
        self.assertEqual(
            [row["description_style"] for row in buckets[0]],
            ["W12 W17 N11", "W12 W17 N12"],
        )

    def test_near_template_cleanup_includes_cross_field_matches(self):
        shared = "W6 W3 W6 W7: W6 W5 W6 W4: W8 W7 W5 W11"
        buckets = [
            [{"title_style": "W4 W5", "description_style": shared}],
            [{"title_style": shared, "description_style": "W9 W2"}],
        ]
        audit = v6.clear_cross_bucket_near_fields(
            buckets, ("title_style", "description_style")
        )
        self.assertEqual(buckets[0][0]["description_style"], "")
        self.assertEqual(buckets[1][0]["title_style"], "")
        self.assertGreater(
            audit["matches_by_field"]["description_style->title_style"], 0
        )

    def test_local_copy_cleanup_covers_final_stream_length_scale(self):
        shared = " ".join(["W10"] * 14)
        left = shared + " " + " ".join(["W1"] * 30)
        right = shared + " " + " ".join(["N1"] * 30)
        self.assertTrue(v6.fields_share_local_copy(left, right, width=40))
        self.assertFalse(
            v6.fields_are_near_duplicates(
                v6.normalized_copy_text(left),
                v6.normalized_copy_text(right),
                long_width=10,
                long_threshold=0.90,
            )
        )
        buckets = [
            [{"title_style": left, "description_style": "W7 W8"}],
            [{"title_style": "W2", "description_style": right}],
        ]
        audit = v6.clear_cross_bucket_near_fields(
            buckets,
            ("title_style", "description_style"),
            local_copy_width=40,
        )
        self.assertEqual(buckets[0][0]["title_style"], "")
        self.assertEqual(buckets[1][0]["description_style"], "")
        self.assertGreater(
            audit["local_copy_matches_by_field"][
                "title_style->description_style"
            ],
            0,
        )

    def test_local_copy_donor_is_rejected_before_pair_generation(self):
        shared = " ".join(["W10"] * 14)
        accounts = {
            "a": {
                "donor_uid": "donor-a",
                "style_stream": shared + " " + " ".join(["W1"] * 30),
            },
            "b": {
                "donor_uid": "donor-a",
                "style_stream": shared + " " + " ".join(["N1"] * 30),
            },
            "c": {
                "donor_uid": "donor-b",
                "style_stream": "W3 W4 W5 W6",
            },
            "d": {
                "donor_uid": "donor-b",
                "style_stream": "W7 W8 W9 W10",
            },
        }
        self.assertEqual(
            v6.find_local_copy_donor_uids(
                accounts,
                {"controller-a": ["a", "b"], "controller-b": ["c", "d"]},
            ),
            ["donor-a"],
        )

    def test_transfer_style_projection_removes_lexical_and_script_identity(self):
        english = v6.transferable_style_projection("Premium apples, shipped in 2days!")
        chinese = v6.transferable_style_projection("优质苹果，两天内发货！")
        multilingual = v6.transferable_style_projection("café Ελληνικά русский １２３")
        self.assertNotIn("Premium", english)
        self.assertNotIn("苹果", chinese)
        self.assertNotIn("H", chinese)
        self.assertNotIn("days", english)
        self.assertRegex(english, r"W\d+")
        self.assertRegex(chinese, r"W\d+")
        self.assertEqual(v6.style_projection_residual_count(multilingual), 0)

    def test_transfer_style_projection_rejects_unicode_side_channels(self):
        source = "cafe\u0301 س\u0650\u200d ™ ₿ 中文，１２３"
        projected = v6.transferable_style_projection(source)
        self.assertEqual(projected, "W4 W1 * $ W2.N3")
        self.assertEqual(v6.style_projection_residual_count(projected), 0)
        self.assertNotIn("\u0301", projected)
        self.assertNotIn("\u0650", projected)
        self.assertNotIn("\u200d", projected)
        self.assertNotIn("™", projected)
        self.assertNotIn("₿", projected)
        self.assertNotIn("中文", projected)

    def test_compatibility_symbols_do_not_expand_into_word_shapes(self):
        projected = v6.transferable_style_projection("™ ℡ Ⅷ ㏂ ½ ㍑")
        self.assertEqual(projected, "* * N1 * N1 *")

    def test_exact_style_deduplication_is_cross_account_and_cross_field(self):
        rows = [
            {
                "account_uid": "a",
                "title_style": "W4 W4",
                "description_style": "W9 W3",
            },
            {
                "account_uid": "b",
                "title_style": "W8 W2",
                "description_style": "W4 W4",
            },
        ]
        audit = v6.clear_cross_account_exact_style_fields(rows)
        self.assertEqual(audit["shared_style_values_removed"], 1)
        self.assertEqual([row["title_style"] for row in rows], ["", "W8 W2"])
        self.assertEqual(
            [row["description_style"] for row in rows], ["W9 W3", ""]
        )

    def test_unified_style_stream_drops_boundaries_and_uses_ordinary_space(self):
        rows = [
            {
                "item_uid": "b",
                "title_style": "",
                "description_style": "W7.",
            },
            {
                "item_uid": "a",
                "title_style": "W4 W2",
                "description_style": "",
            },
            {
                "item_uid": "c",
                "title_style": "W5!",
                "description_style": "W8",
            },
        ]
        stream = v6.unified_account_style_stream(rows)
        self.assertEqual(stream, "W4 W2 W7. W5! W8")
        self.assertEqual(
            len(v6.ADJACENT_WORD_PLACEHOLDER_RE.findall(stream)),
            0,
        )
        self.assertEqual(
            len(v6.ADJACENT_NUMBER_PLACEHOLDER_RE.findall(stream)),
            0,
        )

    def test_style_stream_budget_is_exact_deterministic_and_nonoverlapping(self):
        source = "prefix " + " ".join(
            f"W{index % 9 + 1}!" if index % 2 else f"N{index % 9 + 1},"
            for index in range(1, 121)
        )
        observed, audit = v6.budget_style_stream(source, 23)
        repeated, repeated_audit = v6.budget_style_stream(source, 23)
        self.assertEqual(observed, repeated)
        self.assertEqual(audit, repeated_audit)
        self.assertEqual(
            len(v6.STYLE_PLACEHOLDER_RE.findall(observed)), 23
        )
        self.assertEqual(audit["source_placeholder_count"], 120)
        self.assertEqual(audit["selected_placeholder_count"], 23)
        self.assertEqual(
            [end - start for start, end in audit["source_ranges"]],
            [8, 7, 8],
        )
        first, middle, last = audit["source_ranges"]
        self.assertEqual(first[0], 0)
        self.assertLessEqual(first[1], middle[0])
        self.assertLessEqual(middle[1], last[0])
        self.assertEqual(last[1], 120)
        self.assertNotIn("SLOT", observed)
        with self.assertRaises(v6.SyntheticSourceError):
            v6.budget_style_stream("W1 W2 W3", 4)

    def test_unified_style_budget_matches_frozen_total(self):
        budget = self.policy["construction"]["style_stream_budget"]
        items = [
            {
                "item_uid": "b",
                "title_style": "W2 W3",
                "description_style": "N1 N2",
            },
            {
                "item_uid": "a",
                "title_style": "W4",
                "description_style": "W5",
            },
        ]
        self.assertEqual(
            v6.unified_account_style_stream(items),
            "W4 W5 W2 W3 N1 N2",
        )
        raw = " ".join(["W4"] * 140)
        observed, _ = v6.budget_style_stream(
            raw, budget["total_placeholders"]
        )
        summary = v6.style_account_summary(
            [{"title_style": observed, "description_style": ""}]
        )
        self.assertEqual(summary["token_count"], 100)

    def test_fixed_budget_makes_all_shortcut_structure_pair_invariant(self):
        def summary(word_length: int, punctuation: str):
            return v6.style_account_summary(
                [
                    {
                        "title_style": " ".join(
                            f"W{word_length}{punctuation}" for _ in range(100)
                        ),
                        "description_style": "",
                    }
                ]
            )

        first = summary(1, ".")
        second = summary(7, "!")
        third = summary(12, ";")
        seam_features = set(
            v6.symmetric_pair_feature_names(v6.STYLE_SEAM_ACCOUNT_FIELDS)
        )
        structural_features = set(
            self.policy["construction"]["negative_matching_feature_weights"]
        ) - seam_features
        first_pair = v6.style_pair_covariates(first, second)
        second_pair = v6.style_pair_covariates(second, third)
        self.assertEqual(
            {name: first_pair[name] for name in structural_features},
            {name: second_pair[name] for name in structural_features},
        )

    def test_item_slot_features_do_not_enter_stream_view_matching(self):
        left = v6.style_account_summary(
            [
                {"title_style": "W4", "description_style": "W5 W6"},
                {"title_style": "", "description_style": "W7"},
            ]
        )
        right = v6.style_account_summary(
            [
                {"title_style": "W3 W2", "description_style": ""},
                {"title_style": "W8", "description_style": "W9"},
            ]
        )
        covariates = v6.style_pair_covariates(left, right)
        expected = set()
        for field in v6.STYLE_DISTRIBUTION_ACCOUNT_FIELDS:
            expected.update(
                {
                    f"minimum_{field}",
                    f"maximum_{field}",
                    f"absolute_{field}_difference",
                }
            )
        self.assertTrue(expected <= set(covariates))
        base_structural = set(v5.pair_covariates(left, right)) - {
            "category_jaccard",
            "token_jaccard",
        }
        stream_presence = {
            feature
            for field in (
                "empty_title_item_count",
                "empty_description_item_count",
            )
            for feature in (
                f"minimum_{field}",
                f"maximum_{field}",
                f"absolute_{field}_difference",
            )
        }
        seam_features = {
            feature
            for field in v6.STYLE_SEAM_ACCOUNT_FIELDS
            for feature in (
                f"minimum_{field}",
                f"maximum_{field}",
                f"absolute_{field}_difference",
            )
        }
        self.assertEqual(
            set(self.policy["construction"]["negative_matching_feature_weights"]),
            base_structural | stream_presence | seam_features,
        )
        self.assertEqual(len(base_structural | stream_presence), 14)
        self.assertEqual(len(seam_features), 12)
        self.assertFalse((base_structural | stream_presence) & seam_features)
        self.assertEqual(
            expected
            & set(self.policy["construction"]["negative_matching_feature_weights"]),
            stream_presence | seam_features,
        )
        self.assertEqual(covariates["maximum_empty_title_item_count"], 1.0)
        self.assertEqual(
            covariates["absolute_empty_description_item_count_difference"],
            1.0,
        )
        self.assertEqual(covariates["maximum_both_empty_item_count"], 0.0)

    def test_negative_derangement_balances_every_account_degree(self):
        accounts = {}
        accounts_by_controller = defaultdict(list)
        for controller_index in range(4):
            controller_uid = f"controller-{controller_index}"
            for role in range(2):
                uid = f"account-{controller_index}-{role}"
                token = hashlib.sha256(uid.encode("utf-8")).hexdigest()
                items = [
                    {
                        "title_clean": token,
                        "description_clean": token[::-1],
                        "category_clean": f"category-{controller_index}",
                    }
                ]
                style_index = controller_index * 2 + role
                style_items = [
                    {
                        "title_style": (
                            f"W3 W4 W5 W6 W7 W8 W9 W{10 + style_index}"
                        ),
                        "description_style": (
                            f"N{1 + style_index} / W{20 + style_index * 7}"
                        ),
                    }
                ]
                accounts[uid] = {
                    "account_uid": uid,
                    "controller_uid": controller_uid,
                    "role": role,
                    "split": "train",
                    "items": items,
                    "summary": v5.account_summary(items),
                    "style_items": style_items,
                }
                accounts[uid]["style_summary"] = v6.style_account_summary(
                    accounts[uid]["style_items"]
                )
                accounts_by_controller[controller_uid].append(uid)
        positives = v6.positive_pairs(self.policy, accounts, accounts_by_controller)
        negatives, diagnostic = v6.select_negative_pairs(
            self.policy, accounts, accounts_by_controller, positives
        )
        repeated, repeated_diagnostic = v6.select_negative_pairs(
            self.policy, accounts, accounts_by_controller, positives
        )
        self.assertEqual(len(negatives), 8)
        self.assertEqual(negatives, repeated)
        self.assertEqual(diagnostic, repeated_diagnostic)
        self.assertEqual(diagnostic["cell_count"], 1)
        self.assertEqual(len(diagnostic["distance_feature_names"]), 26)
        self.assertEqual(diagnostic["cells"][0]["selected_edges"], 8)
        self.assertEqual(diagnostic["cells"][0]["solve_rounds"], 1)
        self.assertEqual(
            len({tuple(sorted((row["left_uid"], row["right_uid"]))) for row in negatives}),
            8,
        )
        degrees = Counter(
            uid for row in negatives for uid in (row["left_uid"], row["right_uid"])
        )
        self.assertEqual(set(degrees.values()), {2})
        self.assertTrue(
            all(
                v6.account_pair_has_style_near_duplicate(
                    accounts[row["left_uid"]], accounts[row["right_uid"]]
                )
                for row in negatives
            )
        )

    def test_joint_edge_cost_compares_features_in_standardized_units(self):
        target = {"small_unit": 0.0, "large_unit": 0.0}
        weights = {"small_unit": 1.0, "large_unit": 1.0}
        scales = {"small_unit": 1.0, "large_unit": 100.0}
        small_unit_cost = v6.standardized_pair_cost(
            target,
            {"small_unit": 1.0, "large_unit": 0.0},
            scales,
            weights,
        )
        large_unit_cost = v6.standardized_pair_cost(
            target,
            {"small_unit": 0.0, "large_unit": 100.0},
            scales,
            weights,
        )
        self.assertEqual(small_unit_cost, 1.0)
        self.assertEqual(large_unit_cost, 1.0)


class Step7V6PublishedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = ROOT / "reports" / "step7_v6_synthetic_english_source" / "v6_20260903"
        if not cls.root.is_dir():
            raise AssertionError("Build Step7 V6 before running artifact tests")
        cls.manifest = json.loads((cls.root / "manifest.json").read_text(encoding="utf-8"))
        cls.audit = json.loads((cls.root / "quality_audit.json").read_text(encoding="utf-8"))
        cls.pairs = read_csv(cls.root / "public_pairs.csv")
        cls.labels = {row["pair_uid"]: row for row in read_csv(cls.root / "labels.csv")}

    def test_manifest_and_all_frozen_gates_pass(self):
        self.assertEqual(
            self.manifest["status"],
            "PASSED_STYLE_ONLY_TRAINING_AUGMENTATION_QUALIFIED",
        )
        self.assertEqual(self.audit["status"], "PASSED")
        self.assertTrue(all(self.audit["gate_results"].values()))
        self.assertEqual(
            self.audit["near_style_indicator"]["positive_with_indicator"], 0
        )
        self.assertEqual(
            self.audit["local_style_copy_indicator"]["positive_with_indicator"],
            0,
        )
        self.assertLessEqual(
            self.audit["local_style_copy_indicator"][
                "maximum_heldout_bidirectional_roc_auc"
            ],
            self.policy_gate(
                "maximum_local_style_copy_indicator_bidirectional_roc_auc"
            ),
        )
        self.assertLessEqual(
            self.audit["local_style_copy_indicator"][
                "maximum_heldout_average_precision_lift_over_prevalence"
            ],
            self.policy_gate(
                "maximum_local_style_copy_indicator_ap_lift_over_prevalence"
            ),
        )
        self.assertLessEqual(
            self.audit["near_style_indicator"][
                "maximum_heldout_bidirectional_roc_auc"
            ],
            self.policy_gate("maximum_near_style_indicator_bidirectional_roc_auc"),
        )
        self.assertEqual(
            self.audit["synthesis_audit"]["donor_topology_component_collisions"],
            0,
        )
        self.assertEqual(
            self.audit["synthesis_audit"]["donor_topology_evidence_overlaps"], 0
        )
        self.assertEqual(
            set(self.audit["structural_proxy"]["feature_names"]),
            set(v6.load_policy()["construction"]["negative_matching_feature_weights"])
            - set(v6.symmetric_pair_feature_names(v6.STYLE_SEAM_ACCOUNT_FIELDS)),
        )
        self.assertEqual(
            set(self.audit["seam_proxy"]["feature_names"]),
            set(v6.symmetric_pair_feature_names(v6.STYLE_SEAM_ACCOUNT_FIELDS)),
        )
        matching = self.audit["synthesis_audit"][
            "negative_matching_diagnostic"
        ]
        self.assertEqual(
            set(matching["distance_feature_names"]),
            set(v6.load_policy()["construction"]["negative_matching_feature_weights"]),
        )
        self.assertGreater(matching["cell_count"], 0)
        self.assertEqual(
            {cell["split"] for cell in matching["cells"]},
            {"train", "development", "synthetic_audit"},
        )
        self.assertLessEqual(
            self.audit["seam_proxy"]["maximum_heldout_bidirectional_roc_auc"],
            self.policy_gate("maximum_style_seam_proxy_bidirectional_roc_auc"),
        )
        self.assertLessEqual(
            self.audit["seam_proxy"][
                "maximum_heldout_average_precision_lift_over_prevalence"
            ],
            self.policy_gate("maximum_style_seam_proxy_ap_lift_over_prevalence"),
        )
        self.assertEqual(
            self.audit["model_visibility"]["training_authorized_granularity"],
            "field_neutral_account_style_stream",
        )
        self.assertFalse(
            self.audit["model_visibility"]["item_slot_boundaries_published"]
        )
        manifest_without_self_hash = dict(self.manifest)
        observed_self_hash = manifest_without_self_hash.pop("manifest_self_sha256")
        self.assertEqual(v6.canonical_sha256(manifest_without_self_hash), observed_self_hash)
        self.assertEqual(
            sha256(ROOT / self.manifest["builder_path"]),
            self.manifest["builder_sha256"],
        )
        self.assertEqual(
            sha256(ROOT / self.manifest["policy_path"]),
            self.manifest["policy_sha256"],
        )
        for record in self.manifest["files"]:
            path = self.root / record["path"]
            self.assertEqual(path.stat().st_size, record["size_bytes"])
            self.assertEqual(sha256(path), record["sha256"])
        expected_files = {record["path"] for record in self.manifest["files"]}
        expected_files.add("manifest.json")
        self.assertEqual(
            {path.name for path in self.root.iterdir() if path.is_file()},
            expected_files,
        )

    def policy_gate(self, name):
        return v6.load_policy()["quality_gates"][name]

    def test_exact_pair_and_label_counts(self):
        self.assertEqual(len(self.pairs), 1950)
        self.assertEqual(set(row["pair_uid"] for row in self.pairs), set(self.labels))
        counts = Counter(int(row["label"]) for row in self.labels.values())
        self.assertEqual(counts, Counter({0: 1300, 1: 650}))
        positive_weight = sum(
            float(row["sample_weight"])
            for row in self.labels.values()
            if row["label"] == "1"
        )
        negative_weight = sum(
            float(row["sample_weight"])
            for row in self.labels.values()
            if row["label"] == "0"
        )
        self.assertAlmostEqual(positive_weight, 460.0, places=9)
        self.assertAlmostEqual(negative_weight, 460.0, places=9)

    def test_every_account_has_one_slot_free_style_stream(self):
        self.assertFalse((self.root / "public_items_full_clean.jsonl").exists())
        self.assertFalse((self.root / "public_items_style_projection.jsonl").exists())
        rows = [
            json.loads(line)
            for line in (self.root / "public_accounts_style_projection.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertEqual(len(rows), 1015)
        self.assertEqual(len({row["account_uid"] for row in rows}), 1015)
        for row in rows:
            self.assertEqual(
                set(row),
                {"account_uid", "split", "style_stream"},
            )
            self.assertTrue(row["style_stream"])
            self.assertEqual(
                v6.style_projection_residual_count(row["style_stream"]), 0
            )
            self.assertEqual(
                len(v6.STYLE_PLACEHOLDER_RE.findall(row["style_stream"])),
                100,
            )
        self.assertEqual(self.audit["counts"]["source_items_used"], 8120)
        self.assertEqual(self.audit["counts"]["public_accounts_per_view"], 1015)
        self.assertEqual(
            self.audit["model_visibility"]["cross_account_exact_style_values"], 0
        )
        self.assertEqual(
            self.audit["model_visibility"][
                "cross_account_exact_style_stream_values"
            ],
            0,
        )
        self.assertEqual(
            self.audit["synthesis_audit"]["style_stream_budget_mismatches"],
            0,
        )

    def test_every_account_has_balanced_positive_and_negative_incidence(self):
        positive_degrees = Counter()
        negative_degrees = Counter()
        positive_weights = defaultdict(float)
        negative_weights = defaultdict(float)
        for pair in self.pairs:
            label_row = self.labels[pair["pair_uid"]]
            label = int(label_row["label"])
            weight = float(label_row["sample_weight"])
            degree_target = positive_degrees if label else negative_degrees
            weight_target = positive_weights if label else negative_weights
            for uid in (pair["account_left_uid"], pair["account_right_uid"]):
                degree_target[uid] += 1
                weight_target[uid] += weight
        self.assertEqual(set(positive_degrees), set(negative_degrees))
        for uid in positive_degrees:
            self.assertEqual(negative_degrees[uid], 2 * positive_degrees[uid])
            self.assertAlmostEqual(negative_weights[uid], positive_weights[uid], places=12)

    def test_pair_endpoints_stay_in_one_split(self):
        account_split = {}
        with (self.root / "public_accounts_style_projection.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                account_split[row["account_uid"]] = row["split"]
        for pair in self.pairs:
            self.assertEqual(account_split[pair["account_left_uid"]], pair["split"])
            self.assertEqual(account_split[pair["account_right_uid"]], pair["split"])

    def test_retrieval_qrels_equal_positive_graph(self):
        expected = set()
        for pair in self.pairs:
            if self.labels[pair["pair_uid"]]["label"] != "1":
                continue
            left = pair["account_left_uid"]
            right = pair["account_right_uid"]
            expected.add((left, right))
            expected.add((right, left))
        qrel_rows = read_csv(self.root / "retrieval_qrels.csv")
        observed = {
            (row["query_account_uid"], row["relevant_account_uid"])
            for row in qrel_rows
        }
        self.assertEqual(len(qrel_rows), len(observed))
        self.assertEqual(observed, expected)
        query_rows = read_csv(self.root / "retrieval_queries.csv")
        query_uids = [row["query_account_uid"] for row in query_rows]
        self.assertEqual(len(query_uids), len(set(query_uids)))
        public_accounts = {
            json.loads(line)["account_uid"]
            for line in (self.root / "public_accounts_style_projection.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        }
        self.assertEqual(set(query_uids), public_accounts)
        split_by_query = {
            row["query_account_uid"]: row["split"] for row in query_rows
        }
        self.assertTrue(all(left != right for left, right in observed))
        self.assertTrue(
            all(split_by_query[left] == split_by_query[right] for left, right in observed)
        )
        protocol = v6.load_policy()["retrieval_protocol"]
        self.assertTrue(protocol["exclude_query_account"])
        self.assertEqual(
            protocol["candidate_universe"],
            "all other synthetic accounts in the query split",
        )


if __name__ == "__main__":
    unittest.main()
