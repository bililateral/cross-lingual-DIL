"""Handmade ER strength contracts; file fixtures are explicitly not native weights."""
from __future__ import annotations

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import step28_er_weight as method
import step28_er_weight_run as runner
import step28_er_weight_evaluate as evaluation
import test_step28_bge_continual_contracts as fixtures


def prepared(count: int = 288) -> tuple:
    model, c = fixtures.tiny_model(), fixtures.config()
    old, current = fixtures.handmade_group("old", 1), fixtures.handmade_group("new", 2)
    optimizer = method.core.make_optimizer(model, c)
    fixtures.toy_prior_step(model, optimizer, old, c, count)
    return model, optimizer, c, current, old


def clone(model: object, optimizer: object, c: dict) -> tuple:
    other = copy.deepcopy(model)
    opt = method.core.make_optimizer(other, c)
    opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    return other, opt


def independent_update(model: object, optimizer: object, current: object, old: object,
                       c: dict, weight: float) -> dict:
    """Separate unweighted derivatives, then explicit linear combination before clipping."""
    model.train()
    parameters = list(model.parameters())
    components, losses = {}, {}
    for name, group, seed in (("current", current, 401), ("history", old, 402)):
        torch.manual_seed(seed)
        scores = method.base.logits(model, group, c, "split_rank")
        truth = torch.tensor(group.labels, dtype=torch.float32)
        terms = method.parent.ranking.objectives(scores, truth, .5)
        # Scalar traversal independently verifies each unweighted objective.
        scalar = fixtures.scalar_losses(scores.detach().numpy(), group.labels)
        for key, value in terms.items():
            np.testing.assert_allclose(float(value.detach()), scalar[key], rtol=2e-6, atol=2e-6)
        components[name] = torch.autograd.grad(terms["total"], parameters)
        losses[name] = float(terms["total"].detach())
    optimizer.zero_grad(set_to_none=True)
    for p, current_grad, old_grad in zip(parameters, components["current"], components["history"], strict=True):
        p.grad = current_grad + weight * old_grad
    for group, lr in zip(optimizer.param_groups, (method.parent.stage_lr(1), .001), strict=True):
        group["lr"] = lr
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, c["optimizer"]["clip_norm"]))
    optimizer.step()
    return {"current_total": losses["current"], "history_total": losses["history"], "gradient_norm": norm}


def gate_fixture(home: Path) -> tuple:
    """Small file-identity fixtures; not proof of native save/reload or training."""
    old_job = home / "original"
    old_root = old_job / "run"
    old_root.mkdir(parents=True)
    old_manifest, _ = fixtures.gate_fixture(old_root)
    partition = method.data.read_json(old_root / "partition.json")
    for order in method.ORDERS:
        for stage in (2, 3):
            key = f"{order}_er_stage{stage}"
            path = old_root / old_manifest["training"][key]["path"]
            log = method.data.read_json(path)
            log["current_dropout_stream"] = method.data.seed_for(20260918, order, stage, "current")
            method.data.write_json(path, log)
            old_manifest["training"][key] = method.data.record(path, old_root)
    reference = {"job": old_job, "manifest": old_manifest, "partition": partition}
    root = home / "new"
    for directory in ("scores", "maps", "points", "updates", "models", "memory"):
        (root / directory).mkdir(parents=True)
    method.data.write_json(root / "partition.json", partition)
    manifest = {"status": runner.COMPLETE, "source_files": method.sources(),
                "policy_sha256": method.POLICY_SHA256, "physical_updates": 3456,
                "gradient_group_presentations": 6912, "points": {}, "training": {},
                "restored_starts": {}, "memories": {}, "partition": method.data.record(root / "partition.json", root)}
    for order in method.ORDERS:
        shared = runner.old_point(reference, order + "_shared")
        for arm, weight in method.ARMS.items():
            manifest["restored_starts"][order + "_" + arm] = {
                "full_checkpoint": shared["full_checkpoint"], "adam_step": 288,
                "first_scores_replayed_exactly": True, "model_state_sha256": shared["model_state_sha256"],
                "first_map": shared["first_map_parameters"],
                "memory_source": old_manifest["memories"][order + "_er_after1"]["file"],
                "memory_summary": old_manifest["memories"][order + "_er_after1"]}
            retained = copy.deepcopy(old_manifest["memories"][f"{order}_er_stage2"])
            memory_path = root / "memory" / (method.point_name(order, arm, 2) + ".json")
            memory_path.write_bytes((old_root / retained["file"]["path"]).read_bytes())
            retained["file"] = method.data.record(memory_path, root)
            manifest["memories"][method.point_name(order, arm, 2)] = retained
            for stage in (2, 3):
                name = method.point_name(order, arm, stage)
                old_point = runner.old_point(reference, f"{order}_er_stage{stage}")
                point = copy.deepcopy(old_point)
                point.update(name=name, history_weight=weight, policy_sha256=method.POLICY_SHA256)
                for role, rec in old_point["scores"].items():
                    values = np.load(old_root / rec["path"], allow_pickle=False)
                    point["scores"][role] = runner.save_array(root / "scores" / f"{name}_{role}.npy", values, root)
                mapping = method.data.read_json(old_root / old_point["mapping"]["path"])
                mapping["score_source"] = point["scores"]["calibration"]
                map_path = root / "maps" / (name + ".json")
                method.data.write_json(map_path, mapping)
                point["mapping"] = method.data.record(map_path, root)
                model_path = root / "models" / (name + ".pt")
                model_path.write_bytes(b"handmade identity fixture; not a native model")
                point["model"] = method.data.record(model_path, root)
                point_path = root / "points" / (name + ".json")
                method.data.write_json(point_path, point)
                manifest["points"][name] = method.data.record(point_path, root)
                log = copy.deepcopy(runner.old_training(reference, order, stage))
                log.update(history_weight=weight, update_columns=list(method.STEP_COLUMNS),
                           observations={str(s): {module: {
                               "finite_nonzero_combined_gradient": True,
                               "parameters_changed": module == "head" or s != 288}
                               for module in ("encoder", "head")} for s in (1, 29, 30, 288)})
                values = np.zeros((288, len(method.STEP_COLUMNS)), dtype=np.float64)
                fields = {key: values[:, i] for i, key in enumerate(method.STEP_COLUMNS)}
                for role in ("current", "history"):
                    for term in ("bce", "rank", "hard"):
                        fields[role + "_" + term][:] = 1.
                    fields[role + "_total"][:] = 2.5
                fields["weighted_history_total"][:] = 2.5 * weight
                fields["total"][:] = 2.5 + 2.5 * weight
                fields["history_weight"][:], fields["head_lr"][:], fields["gradient_norm"][:] = weight, .001, 1.
                fields["encoder_lr"][:] = [method.parent.stage_lr(i) for i in range(1, 289)]
                log["update_file"] = runner.save_array(root / "updates" / (name + ".npy"), values, root)
                log_path = root / "updates" / (name + ".json")
                method.data.write_json(log_path, log)
                manifest["training"][name] = method.data.record(log_path, root)
    return root, manifest, reference


class WeightContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_confirmed_scope(self) -> None:
        policy = method.contract()
        self.assertEqual(policy["physical_updates"], 2 * 3 * 2 * 288)
        self.assertEqual(len(method.expected_points()), 12)
        self.assertEqual(policy["metric_count_sets"], 3 * len(method.expected_points()))
        self.assertFalse(policy["supervision"]["test_access"])

    def test_original_er_exact_update_parity(self) -> None:
        model, opt, c, current, old = prepared()
        original, original_opt = clone(model, opt, c)
        a = method.update(model, opt, current, old, c, 1., 2, 1, 401, 402)
        b = method.parent.update(original, original_opt, current, old, None, c, "er", 2, 1, 401, 402)
        self.assertEqual(method.core.state_digest(model.state_dict()), method.core.state_digest(original.state_dict()))
        self.assertEqual(method.core.state_digest(opt.state_dict()), method.core.state_digest(original_opt.state_dict()))
        for key in ("current_total", "history_total", "gradient_norm", "total"):
            self.assertEqual(a[key], b[key])

    def test_weights_match_independent_derivatives_and_adam(self) -> None:
        for weight in (.5, .25):
            with self.subTest(weight=weight):
                model, opt, c, current, old = prepared()
                reference, reference_opt = clone(model, opt, c)
                expected = independent_update(reference, reference_opt, current, old, c, weight)
                observed = method.update(model, opt, current, old, c, weight, 2, 1, 401, 402, observe=True)
                for key, value in expected.items():
                    self.assertAlmostEqual(observed[key], value, places=6)
                for actual, ref in zip(model.parameters(), reference.parameters(), strict=True):
                    torch.testing.assert_close(actual, ref, rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(actual.grad, ref.grad, rtol=1e-6, atol=1e-7)
                    for key in ("exp_avg", "exp_avg_sq", "step"):
                        torch.testing.assert_close(opt.state[actual][key], reference_opt.state[ref][key], rtol=1e-6, atol=1e-8)
                self.assertEqual(method.parent.adam_step(opt), 289)
                self.assertEqual(observed["weighted_history_total"], weight * observed["history_total"])

    def test_only_history_weight_changes_forwards(self) -> None:
        model, opt, c, current, old = prepared()
        scores = []
        for weight in (.5, .25):
            other, other_opt = clone(model, opt, c)
            captured = []
            hook = other.head.register_forward_hook(lambda _, __, result: captured.append(result.detach().clone()))
            try:
                method.update(other, other_opt, current, old, c, weight, 2, 1, 401, 402)
            finally:
                hook.remove()
            self.assertEqual(len(captured), 2)
            scores.append(captured)
        for a, b in zip(*scores, strict=True):
            self.assertTrue(torch.equal(a, b))

    def test_unknown_coefficients_and_missing_history_rejected(self) -> None:
        model, opt, c, current, old = prepared()
        for weight in (True, 0., -.5, .75, float("nan")):
            with self.assertRaises(ValueError):
                method.update(model, opt, current, old, c, weight, 2, 1, 401, 402)
        with self.assertRaises(ValueError):
            method.update(model, opt, current, current, c, .5, 2, 1, 401, 402)
        with self.assertRaises(ValueError):
            method.update(model, opt, current, None, c, .5, 2, 1, 401, 402)

    def test_stage_last_step_preserves_encoder_but_updates_head(self) -> None:
        model, opt, c, current, old = prepared(575)
        row = method.update(model, opt, current, old, c, .25, 2, 288, 401, 402, observe=True)
        self.assertFalse(row["modules"]["encoder"]["parameters_changed"])
        self.assertTrue(row["modules"]["head"]["parameters_changed"])
        self.assertEqual(row["adam_step"], 576)

    def test_tiny_full_restore_preserves_next_weighted_update(self) -> None:
        model, opt, c, current, old = prepared()
        restored, restored_opt = clone(model, opt, c)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tiny.pt"
            record = method.core.save_state(path, model, opt, {"fixture": "handmade"})
            method.core.restore_state(path, restored, restored_opt, record["state_sha256"])
            method.update(model, opt, current, old, c, .5, 2, 1, 401, 402)
            method.update(restored, restored_opt, current, old, c, .5, 2, 1, 401, 402)
            self.assertEqual(method.core.state_digest(model.state_dict()), method.core.state_digest(restored.state_dict()))
            self.assertEqual(method.core.state_digest(opt.state_dict()), method.core.state_digest(restored_opt.state_dict()))

    def test_resume_supply_cannot_revisit_first_or_skip(self) -> None:
        groups, partition = {}, {}
        for role, count in (("fit", 48), ("calibration", 12)):
            groups[role] = [fixtures.handmade_group(f"{role}_{d}_{i}") for d in "ABC" for i in range(count)]
            partition[role] = [{"group_uid": g.uid, "domain": g.uid.split("_")[1]} for g in groups[role]]
        supply = method.ContinuationSupply(groups, partition)
        shared = {"order": "ABC", "stage": 1, "completed_updates": 288,
                  "full_model_adam_and_rng_restore_verified": True, "model_state_sha256": "handmade fixture"}
        supply.resume("half", "ABC", shared)
        for stage in (1, 3):
            with self.assertRaises(ValueError):
                supply.current("half", "ABC", stage)
        fit, _ = supply.current("half", "ABC", 2)
        self.assertTrue(all(g.uid.startswith("fit_B_") for g in fit))
        with self.assertRaises(ValueError):
            supply.resume("half", "ABC", shared)
        with self.assertRaises(ValueError):
            supply.resume("bad", "BCA", shared)

    def test_real_tiny_checkpoint_replay_and_weight_metadata(self) -> None:
        model, opt, c, _, _ = prepared(576)
        cal = [fixtures.handmade_group(f"cal_{i}") for i in range(12)]
        valid = [fixtures.handmade_group(f"valid_{i}") for i in range(60)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("work", "models", "scores", "maps", "points"):
                (root / folder).mkdir()
            point = runner.checkpoint(root, "ABC_half_stage2", model, opt, c, "ABC", 2,
                                      cal, valid, {"a": 1., "b": 0.}, fixtures.HandmadeBudget(), .5)
            self.assertEqual(point["policy_sha256"], method.POLICY_SHA256)
            self.assertEqual(point["history_weight"], .5)
            self.assertFalse((root / point["full_checkpoint"]["path"]).exists())
            self.assertTrue((root / point["model"]["path"]).exists())
            metadata = method.core.restore_state(root / point["model"]["path"], model, None, point["model"]["state_sha256"])
            self.assertEqual(metadata["history_weight"], .5)
            self.assertEqual(metadata["policy_sha256"], method.POLICY_SHA256)

    def test_gate_rejects_changed_start_and_calibration_domain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, reference = gate_fixture(Path(directory))
            wrong = copy.deepcopy(manifest)
            wrong["restored_starts"]["ABC_half"]["model_state_sha256"] = "wrong"
            with self.assertRaises(ValueError):
                runner.blind_gate(root, wrong, reference)
            name = "ABC_half_stage2"
            point_path = root / manifest["points"][name]["path"]
            point = method.data.read_json(point_path)
            map_path = root / point["mapping"]["path"]
            mapping = method.data.read_json(map_path)
            mapping["actual_domain"] = "A"
            method.data.write_json(map_path, mapping)
            point["mapping"] = method.data.record(map_path, root)
            method.data.write_json(point_path, point)
            manifest["points"][name] = method.data.record(point_path, root)
            with self.assertRaises(ValueError):
                runner.blind_gate(root, manifest, reference)

    def test_complete_blind_gate_and_missing_endpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, reference = gate_fixture(Path(directory))
            self.assertEqual(len(runner.blind_gate(root, manifest, reference)), 12)
            del manifest["points"]["ABC_half_stage2"]
            with self.assertRaises(ValueError):
                runner.blind_gate(root, manifest, reference)

    def test_gate_rejects_wrong_weight_and_unpaired_history(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, reference = gate_fixture(Path(directory))
            name = "ABC_half_stage2"
            path = root / manifest["training"][name]["path"]
            original = method.data.read_json(path)
            for field, value in (("history_weight", .25), ("history_ids", list(reversed(original["history_ids"])))):
                changed = copy.deepcopy(original)
                changed[field] = value
                method.data.write_json(path, changed)
                manifest["training"][name] = method.data.record(path, root)
                with self.assertRaises(ValueError):
                    runner.blind_gate(root, manifest, reference)

    def test_gate_rejects_wrong_accumulated_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, reference = gate_fixture(Path(directory))
            name = "ABC_half_stage2"
            log_path = root / manifest["training"][name]["path"]
            log = method.data.read_json(log_path)
            array_path = root / log["update_file"]["path"]
            values = np.load(array_path, allow_pickle=False)
            values[:, method.STEP_COLUMNS.index("weighted_history_total")] *= 2
            np.save(array_path, values, allow_pickle=False)
            log["update_file"] = method.data.record(array_path, root)
            method.data.write_json(log_path, log)
            manifest["training"][name] = method.data.record(log_path, root)
            with self.assertRaises(ValueError):
                runner.blind_gate(root, manifest, reference)

    def test_collect_requires_every_point_and_preserves_36_sets(self) -> None:
        groups = [fixtures.handmade_group(f"valid_{d}_{i}") for d in "ABC" for i in range(20)]
        partition = {"development": [{"group_uid": g.uid, "domain": g.uid.split("_")[1]} for g in groups]}
        raw = np.tile(np.linspace(-2, 1, 378, dtype=np.float32), (60, 1))
        scores = {name: {"raw": raw, "stage-cal": raw.astype(np.float64), "first-cal": raw.astype(np.float64)}
                  for name in method.expected_points()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "evaluation"
            result = runner.collect(root, scores, groups, partition, method.sources())
            self.assertEqual(result["status"], "ALL_36_ER_WEIGHT_MATRICES_SAVED_BEFORE_COMPARISONS")
            self.assertEqual(len(list(root.glob("*.npy"))), 36)
            self.assertEqual(len(list(root.glob("*_counts.json"))), 36)
            self.assertFalse((root / "evaluation.json").exists())
            del scores[method.expected_points()[0]]
            with self.assertRaises(ValueError):
                runner.collect(Path(directory) / "missing", scores, groups, partition, method.sources())

    def test_original_23_guards_drive_selection_and_fallback(self) -> None:
        endpoints = {arm: {"primary": {"O": {"map": {"mean": .4}, "recall_at_5": {"mean": .5}}}}
                     for arm in method.ARMS}
        comparisons = {arm + "_minus_er": {"interpretation": {"pilot_observed_checks_pass": False}}
                       for arm in method.ARMS}
        self.assertEqual(evaluation.select_configuration(endpoints, comparisons)["selected"], "er")
        for row in comparisons.values():
            row["interpretation"]["pilot_observed_checks_pass"] = True
        self.assertEqual(evaluation.select_configuration(endpoints, comparisons)["selected"], "half")
        endpoints["quarter"]["primary"]["O"]["recall_at_5"]["mean"] = .51
        self.assertEqual(evaluation.select_configuration(endpoints, comparisons)["selected"], "quarter")
        endpoints["half"]["primary"]["O"]["map"]["mean"] = .41
        self.assertEqual(evaluation.select_configuration(endpoints, comparisons)["selected"], "half")
        comparisons["half_minus_er"]["interpretation"]["pilot_observed_checks_pass"] = False
        self.assertEqual(evaluation.select_configuration(endpoints, comparisons)["selected"], "quarter")

    def test_candidate_views_and_paired_endpoint_arithmetic(self) -> None:
        old, domains = fixtures.synthetic_matrices()
        for roles in old.values():
            for values in roles.values():
                values[:] = .3
        new = {}
        for order in method.ORDERS:
            for arm, gain in (("half", .1), ("quarter", .05)):
                for stage in (2, 3):
                    values = np.full((60, 22), .3 + gain, dtype=np.float64)
                    for key in ("brier", "log_loss"):
                        values[:, method.metrics.COLUMNS.index(key)] = .3 - gain
                    new[method.point_name(order, arm, stage)] = {role: values.copy() for role in method.ROLES}
        result = evaluation.evaluate_matrices(new, old, domains)
        self.assertAlmostEqual(result["endpoints"]["half"]["primary"]["O"]["map"]["mean"], .4)
        self.assertAlmostEqual(result["endpoints"]["half"]["primary"]["F_first"]["map"]["mean"], -.1)
        self.assertEqual(result["selection"]["selected"], "half")
        for comparison in result["comparisons"].values():
            self.assertEqual(len(comparison["interpretation"]["checks"]), 23)
            self.assertTrue(comparison["interpretation"]["pilot_observed_checks_pass"])


if __name__ == "__main__":
    unittest.main()
