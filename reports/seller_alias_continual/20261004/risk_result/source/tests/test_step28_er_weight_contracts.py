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
                       c: dict, weight: float, reference: np.ndarray | None = None,
                       step: int = 1) -> dict:
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
        needs_mse = name == "history" and reference is not None
        components[name] = torch.autograd.grad(terms["total"], parameters, retain_graph=needs_mse)
        losses[name] = float(terms["total"].detach())
        if needs_mse:
            difference = scores - torch.tensor(reference, dtype=scores.dtype)
            mse = torch.dot(difference, difference) / len(reference)
            components["mse"] = torch.autograd.grad(mse, parameters)
            losses["mse"] = float(mse.detach())
    optimizer.zero_grad(set_to_none=True)
    for index, (p, current_grad, old_grad) in enumerate(zip(parameters, components["current"], components["history"], strict=True)):
        p.grad = current_grad + weight * old_grad
        if reference is not None:
            p.grad = p.grad + .5 * components["mse"][index]
    for group, lr in zip(optimizer.param_groups, (method.parent.stage_lr(step), .001), strict=True):
        group["lr"] = lr
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, c["optimizer"]["clip_norm"]))
    optimizer.step()
    result = {"current_total": losses["current"], "history_total": losses["history"], "gradient_norm": norm}
    if reference is not None:
        result["logit_mse"] = losses["mse"]
    return result


def gate_fixture(home: Path, p: dict | None = None) -> tuple:
    """Small file-identity fixtures; not proof of native save/reload or training."""
    p = method.contract() if p is None else p
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
    manifest = {"status": runner.complete_status(p), "source_files": method.sources(p),
                "policy_sha256": method.policy_sha256(p), "physical_updates": p["physical_updates"],
                "gradient_group_presentations": p["gradient_group_presentations"], "points": {}, "training": {},
                "restored_starts": {}, "memories": {}, "partition": method.data.record(root / "partition.json", root)}
    for order in method.ORDERS:
        shared = runner.old_point(reference, order + "_shared")
        memory_arm = p.get("memory_arm", "er")
        first_memory = old_manifest["memories"][order + "_" + memory_arm + "_after1"]
        for arm, weight in p["arms"].items():
            manifest["restored_starts"][order + "_" + arm] = {
                "full_checkpoint": shared["full_checkpoint"], "adam_step": 288,
                "first_scores_replayed_exactly": True, "model_state_sha256": shared["model_state_sha256"],
                "first_map": shared["first_map_parameters"],
                "memory_source": first_memory["file"], "memory_summary": first_memory}
            retained = copy.deepcopy(old_manifest["memories"][f"{order}_{memory_arm}_stage2"])
            memory_path = root / "memory" / (method.point_name(order, arm, 2, p) + ".json")
            memory_path.write_bytes((old_root / retained["file"]["path"]).read_bytes())
            retained["file"] = method.data.record(memory_path, root)
            if method.with_logits(p):
                retained["reference_update"] = {
                    "model_state_sha256": "handmade_fixture_not_weights", "mode": "eval",
                    "old_survivors_unchanged": True,
                    "new_target_ids": sorted(set(retained["references"]) - set(first_memory["references"]))}
            manifest["memories"][method.point_name(order, arm, 2, p)] = retained
            for stage in (2, 3):
                name = method.point_name(order, arm, stage, p)
                old_point = runner.old_point(reference, f"{order}_er_stage{stage}")
                point = copy.deepcopy(old_point)
                point.update(name=name, history_weight=weight, policy_sha256=method.policy_sha256(p))
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
                log.update(history_weight=weight, update_columns=list(method.step_columns(p)),
                           observations={str(s): {module: {
                               "finite_nonzero_combined_gradient": True,
                               "parameters_changed": module == "head" or s != 288}
                               for module in ("encoder", "head")} for s in (1, 29, 30, 288)})
                if method.with_logits(p):
                    source_log = old_manifest["training"][f"{order}_logit_stage{stage}"]
                    log["memory_after_training"] = method.data.read_json(old_root / source_log["path"])["memory_after_training"]
                values = np.zeros((288, len(method.step_columns(p))), dtype=np.float64)
                fields = {key: values[:, i] for i, key in enumerate(method.step_columns(p))}
                for role in ("current", "history"):
                    for term in ("bce", "rank", "hard"):
                        fields[role + "_" + term][:] = 1.
                    fields[role + "_total"][:] = 2.5
                fields["weighted_history_total"][:] = 2.5 * weight
                fields["total"][:] = 2.5 + 2.5 * weight
                if method.with_logits(p):
                    fields["logit_mse"][:], fields["logit_weight"][:] = 2., .5
                    fields["total"][:] += 1.
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
        for weight in (.5, .25, .1):
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
        for weight in (.5, .25, .1):
            other, other_opt = clone(model, opt, c)
            captured = []
            hook = other.head.register_forward_hook(lambda _, __, result: captured.append(result.detach().clone()))
            try:
                method.update(other, other_opt, current, old, c, weight, 2, 1, 401, 402)
            finally:
                hook.remove()
            self.assertEqual(len(captured), 2)
            scores.append(captured)
        for captured in scores[1:]:
            for a, b in zip(scores[0], captured, strict=True):
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

    def test_low_scope_and_six_point_blind_gate(self) -> None:
        p = method.contract("low")
        self.assertEqual(p["arms"], {"tenth": .1})
        self.assertEqual(p["physical_updates"], 1728)
        self.assertEqual(p["runtime"]["maximum_gpu_stage_seconds"], 43200)
        self.assertEqual(p["runtime"]["maximum_output_bytes"], 16 * 1024 ** 3)
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, reference = gate_fixture(Path(directory), p)
            self.assertEqual(len(runner.blind_gate(root, manifest, reference, p)), 6)
            with self.assertRaises(ValueError):
                runner.blind_gate(root, manifest, reference)  # Cannot route low through the old scope.
            del manifest["points"]["CAB_tenth_stage3"]
            with self.assertRaises(ValueError):
                runner.blind_gate(root, manifest, reference, p)

    def test_low_checkpoint_uses_new_policy_and_weight(self) -> None:
        p = method.contract("low")
        model, opt, c, _, _ = prepared(576)
        cal = [fixtures.handmade_group(f"cal_{i}") for i in range(12)]
        valid = [fixtures.handmade_group(f"valid_{i}") for i in range(60)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("work", "models", "scores", "maps", "points"):
                (root / folder).mkdir()
            point = runner.checkpoint(root, "ABC_tenth_stage2", model, opt, c, "ABC", 2,
                                      cal, valid, {"a": 1., "b": 0.}, fixtures.HandmadeBudget(), .1, p)
            self.assertEqual(point["history_weight"], .1)
            self.assertEqual(point["policy_sha256"], method.LOW_POLICY_SHA256)
            self.assertFalse((root / point["full_checkpoint"]["path"]).exists())

    def test_low_real_tiny_stage_uses_paired_schedule_and_tenth_updates(self) -> None:
        p = method.contract("low")
        model, opt, c, _, _ = prepared()
        current = [fixtures.handmade_group(f"fit_B_{i}") for i in range(48)]
        history = [fixtures.handmade_group(f"fit_A_{i}", 1) for i in range(48)]
        memory = method.parent.Memory("ABC", method.parent.contract()["memory_seed"], False, {"a": 1., "b": 0.})
        memory.retain(history, 1, None)
        oracle = method.parent.Memory.from_bytes(memory.to_bytes())
        oracle.begin_stage(2)
        sequence, _ = method.parent.schedule(current, method.parent.contract(), "ABC", 2)
        reference_log = {"current_ids": [g.uid for g in sequence],
                         "history_ids": [oracle.draw()[0].uid for _ in range(288)],
                         "memory_after_training": oracle.summary()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "updates").mkdir()
            budget = fixtures.HandmadeBudget()
            budget.state = lambda: {"handmade_fixture": True}
            row = runner.train_stage(model, opt, current, memory, c, "ABC", 2, "tenth", root,
                                     reference_log, budget, p)
            values = np.load(root / row["update_file"]["path"], allow_pickle=False)
            self.assertTrue(np.all(values[:, method.STEP_COLUMNS.index("history_weight")] == .1))
            self.assertEqual(row["history_ids"], reference_log["history_ids"])
            self.assertEqual(method.parent.adam_step(opt), 576)

    def test_low_endpoint_differences_use_quarter_and_seq(self) -> None:
        p = method.contract("low")
        old, domains = fixtures.synthetic_matrices()
        for roles in old.values():
            for values in roles.values():
                values[:] = .3
        new = {}
        for order in method.ORDERS:
            for arm, gain in (("half", .04), ("quarter", .08), ("tenth", .1)):
                for stage in (2, 3):
                    values = np.full((60, 22), .3 + gain, dtype=np.float64)
                    for key in ("brier", "log_loss"):
                        values[:, method.metrics.COLUMNS.index(key)] = .3 - gain
                    (new if arm == "tenth" else old)[f"{order}_{arm}_stage{stage}"] = {
                        role: values.copy() for role in method.ROLES}
        result = evaluation.evaluate_matrices(new, old, domains, p)
        self.assertAlmostEqual(result["comparisons"]["tenth_minus_quarter"]["primary"]["O"]["map"]["mean"], .02)
        self.assertAlmostEqual(result["comparisons"]["tenth_minus_seq"]["primary"]["O"]["map"]["mean"], .1)
        self.assertEqual(result["selection"]["selected"], "tenth")

    def test_low_selection_uses_quarter_independently_of_seq(self) -> None:
        p = method.contract("low")
        endpoints = {"tenth": {"primary": {"O": {"map": {"mean": .4}, "recall_at_5": {"mean": .5}}}}}
        comparisons = {"tenth_minus_" + ref: {"interpretation": {"pilot_observed_checks_pass": False}}
                       for ref in ("quarter", "seq")}
        for versus_seq in (False, True):
            comparisons["tenth_minus_seq"]["interpretation"]["pilot_observed_checks_pass"] = versus_seq
            self.assertEqual(evaluation.select_configuration(endpoints, comparisons, p)["selected"], "quarter")
        comparisons["tenth_minus_seq"]["interpretation"]["pilot_observed_checks_pass"] = False
        comparisons["tenth_minus_quarter"]["interpretation"]["pilot_observed_checks_pass"] = True
        self.assertEqual(evaluation.select_configuration(endpoints, comparisons, p)["selected"], "tenth")

    def test_low_full_collection_and_saved_reference_recovery(self) -> None:
        p = method.contract("low")
        def available_job(spec: dict) -> Path:
            local = method.data.ROOT / spec["local_small_job"]
            return local if local.is_dir() else method.data.ROOT / spec["linux_job"]

        old_job, weight_job = available_job(p["baseline"]), available_job(p["weight_reference"])
        reference = method.baseline(p, old_job, weight_job)
        # Identifiers only from the saved result; all new truth/text/scores here are handmade.
        groups = [fixtures.handmade_group(uid) for uid in reference["collected"]["group_ids"]]
        raw = np.tile(np.linspace(-2, 1, 378, dtype=np.float32), (60, 1))
        scores = {name: {role: raw if role == "raw" else raw.astype(np.float64) for role in method.ROLES}
                  for name in method.expected_points(p)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "collection"
            runner.collect(root, scores, groups, reference["partition"], method.sources(p), p)
            self.assertEqual(len(list(root.glob("*.npy"))), 18)
            with mock.patch.object(method.base, "load_model", side_effect=AssertionError("Saved-only recovery")):
                result = evaluation.finalize(root, old_job / "evaluation", p, weight_job / "evaluation")
            self.assertEqual(result["new_metric_count_sets"], 18)
            self.assertEqual(result["reused_metric_count_sets"], 81)
            self.assertEqual(set(result["comparisons"]), {"tenth_minus_quarter", "tenth_minus_seq"})
            for comparison in result["comparisons"].values():
                self.assertEqual(len(comparison["interpretation"]["checks"]), 23)
            saved = method.data.read_json(weight_job / "evaluation/evaluation.json")
            for arm in ("quarter", "half", "seq", "er"):
                for role, endpoints in result["endpoints"][arm].items():
                    for endpoint, rows in endpoints.items():
                        for metric, row in rows.items():
                            expected = saved["endpoints"][arm][role][endpoint][metric]
                            np.testing.assert_allclose(row["mean"], expected["mean"], rtol=0, atol=1e-12)
                            np.testing.assert_allclose(row["conditional_95pct_interval"],
                                                       expected["conditional_95pct_interval"], rtol=0, atol=1e-12)
                            for order in method.ORDERS:
                                self.assertAlmostEqual(row["per_order"][order], expected["per_order"][order], places=12)
            before = (root / "evaluation.json").read_bytes()
            evaluation.finalize(root, old_job / "evaluation", p, weight_job / "evaluation")
            self.assertEqual((root / "evaluation.json").read_bytes(), before)


class LogitWeightContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_matched_scope_and_gate_rejects_wrong_mse_and_target_origin(self) -> None:
        p = method.contract("logit")
        self.assertEqual(p["arms"], {"logit_quarter": .25})
        self.assertEqual(p["loss"]["logit_mse"], .5)
        self.assertEqual(p["physical_updates"], 1728)
        self.assertEqual(set(method.all_weights(p)), {"seq", "quarter", "logit_quarter"})
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, reference = gate_fixture(Path(directory), p)
            self.assertEqual(len(runner.blind_gate(root, manifest, reference, p)), 6)
            name = "ABC_logit_quarter_stage2"
            wrong = copy.deepcopy(manifest)
            wrong["memories"][name]["reference_update"]["model_state_sha256"] = "old_LOGIT1_model"
            with self.assertRaisesRegex(ValueError, "origin-stage model"):
                runner.blind_gate(root, wrong, reference, p)
            wrong = copy.deepcopy(manifest)
            targets = wrong["restored_starts"]["ABC_logit_quarter"]["memory_summary"]["references"]
            targets[next(iter(targets))] = "refreshed_old_target"
            with self.assertRaisesRegex(ValueError, "targets"):
                runner.blind_gate(root, wrong, reference, p)
            log_path = root / manifest["training"][name]["path"]
            log = method.data.read_json(log_path)
            values = np.load(root / log["update_file"]["path"], allow_pickle=False)
            values[:, method.step_columns(p).index("logit_weight")] = .125
            log["update_file"] = runner.save_array(root / "updates" / (name + "_wrong_mse.npy"), values, root)
            method.data.write_json(log_path, log)
            wrong = copy.deepcopy(manifest)
            wrong["training"][name] = method.data.record(log_path, root)
            with self.assertRaisesRegex(ValueError, "MSE weight"):
                runner.blind_gate(root, wrong, reference, p)

    def test_original_logit_exact_parity_and_two_forwards_one_clip(self) -> None:
        model, opt, c, current, old = prepared()
        other, other_opt = clone(model, opt, c)
        target = np.linspace(-.6, .4, 378, dtype=np.float32)
        with mock.patch.object(method.base, "logits", wraps=method.base.logits) as forwards, \
             mock.patch.object(torch.nn.utils, "clip_grad_norm_", wraps=torch.nn.utils.clip_grad_norm_) as clips:
            actual = method.update(model, opt, current, old, c, 1., 2, 1, 401, 402,
                                   reference=target, logit_weight=.5)
        self.assertEqual((forwards.call_count, clips.call_count), (2, 1))
        expected = method.parent.update(other, other_opt, current, old, target, c, "logit", 2, 1, 401, 402)
        self.assertEqual(method.core.state_digest(model.state_dict()), method.core.state_digest(other.state_dict()))
        self.assertEqual(method.core.state_digest(opt.state_dict()), method.core.state_digest(other_opt.state_dict()))
        self.assertEqual(actual["logit_mse"], expected["logit_mse"])

    def test_matched_supervision_and_half_mse_independent_gradients_adam(self) -> None:
        target = np.linspace(-1.5, 1.2, 378, dtype=np.float32)
        for weight, clip, step in ((w, c, s) for w in (.25, .1) for c, s in ((.01, 1), (1e6, 1), (1., 288))):
            with self.subTest(weight=weight, clip=clip, step=step):
                model, opt, c, current, old = prepared(288 + step - 1)
                c["optimizer"]["clip_norm"] = clip
                other, other_opt = clone(model, opt, c)
                expected = independent_update(other, other_opt, current, old, c, weight, target, step)
                actual = method.update(model, opt, current, old, c, weight, 2, step, 401, 402,
                                       reference=target, logit_weight=.5, observe=True)
                for key, value in expected.items():
                    self.assertAlmostEqual(actual[key], value, places=6)
                self.assertEqual(actual["total"], actual["current_total"] + weight * actual["history_total"] + .5 * actual["logit_mse"])
                for parameter, reference in zip(model.parameters(), other.parameters(), strict=True):
                    torch.testing.assert_close(parameter, reference, rtol=1e-6, atol=1e-7)
                    torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-6, atol=1e-7)
                    for key in ("exp_avg", "exp_avg_sq", "step"):
                        torch.testing.assert_close(opt.state[parameter][key], other_opt.state[reference][key], rtol=1e-6, atol=1e-8)
                self.assertEqual(actual["modules"]["encoder"]["parameters_changed"], step != 288)
                self.assertEqual(method.parent.adam_step(opt), 288 + step)

    def test_invalid_logit_reference_rejected_before_update(self) -> None:
        for target, coefficient in ((None, .5), (np.zeros(378, np.float32), .125),
                                    (np.zeros(378, np.float64), .5), (np.zeros(377, np.float32), .5),
                                    (np.full(378, np.nan, np.float32), .5)):
            model, opt, c, current, old = prepared()
            before = method.core.state_digest(model.state_dict())
            with self.assertRaises(ValueError):
                method.update(model, opt, current, old, c, .25, 2, 1, 401, 402,
                              reference=target, logit_weight=coefficient)
            self.assertEqual(method.core.state_digest(model.state_dict()), before)
            self.assertEqual(method.parent.adam_step(opt), 288)

    def test_logit_real_continuation_checkpoint_and_unrefreshing_memory(self) -> None:
        p = method.contract("logit_low")
        model, opt, c, _, _ = prepared()
        history = [fixtures.handmade_group(f"fit_A_{i}", 1) for i in range(48)]
        current = [fixtures.handmade_group(f"fit_B_{i}", 2) for i in range(48)]
        memory = method.parent.Memory("ABC", method.parent.contract()["memory_seed"], True, {"a": 1., "b": 0.})
        memory.retain(history, 1, lambda rows: method.parent.ranking.score(model, rows, c))
        initial_targets = {uid: row.copy() for uid, row in memory.references.items()}
        oracle = method.parent.Memory.from_bytes(memory.to_bytes())
        oracle.begin_stage(2)
        sequence, _ = method.parent.schedule(current, method.parent.contract(), "ABC", 2)
        reference_log = {"current_ids": [g.uid for g in sequence], "history_ids": [oracle.draw()[0].uid for _ in range(288)],
                         "memory_after_training": oracle.summary()}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for folder in ("updates", "work", "models", "scores", "maps", "points"):
                (root / folder).mkdir()
            budget = fixtures.HandmadeBudget()
            budget.state = lambda: {"handmade_fixture": True}
            row = runner.train_stage(model, opt, current, memory, c, "ABC", 2, "logit_tenth", root, reference_log, budget, p)
            self.assertEqual(row["history_ids"], reference_log["history_ids"])
            values = np.load(root / row["update_file"]["path"], allow_pickle=False)
            self.assertTrue(np.all(values[:, method.step_columns(p).index("logit_weight")] == .5))
            self.assertTrue(np.all(values[:, method.step_columns(p).index("history_weight")] == .1))
            cal = [fixtures.handmade_group(f"cal_{i}") for i in range(12)]
            valid = [fixtures.handmade_group(f"valid_{i}") for i in range(60)]
            point = runner.checkpoint(root, "ABC_logit_tenth_stage2", model, opt, c, "ABC", 2,
                                      cal, valid, memory.first_map, budget, .1, p)
            self.assertEqual(point["learner_auxiliary"]["checkpoint_metadata"]["logit_weight"], .5)
            self.assertTrue(point["full_model_adam_and_rng_restore_verified"])
            memory.auxiliary = point["learner_auxiliary"]
            scored = {}
            def score(rows: list) -> np.ndarray:
                result = method.parent.ranking.score(model, rows, c)
                scored.update({g.uid: value.copy() for g, value in zip(rows, result, strict=True)})
                return result
            memory.retain(current, 2, score)
            self.assertEqual(set(scored), set(memory.references) - set(initial_targets))
            for uid, value in memory.references.items():
                np.testing.assert_array_equal(value, initial_targets[uid] if uid in initial_targets else scored[uid])
            self.assertEqual(method.parent.Memory.from_bytes(memory.to_bytes()).to_bytes(), memory.to_bytes())

    def test_logit_complete_saved_comparisons_reuse_45_sets(self) -> None:
        p = method.contract("logit")
        def available(spec: dict) -> Path:
            local = method.data.ROOT / spec["local_small_job"]
            return local if local.is_dir() else method.data.ROOT / spec["linux_job"]
        old_job, weight_job = available(p["baseline"]), available(p["weight_reference"])
        reference = method.baseline(p, old_job, weight_job)
        groups = [fixtures.handmade_group(uid) for uid in reference["collected"]["group_ids"]]
        raw = np.tile(np.linspace(-2, 1, 378, dtype=np.float32), (60, 1))
        scores = {name: {role: raw if role == "raw" else raw.astype(np.float64) for role in method.ROLES}
                  for name in method.expected_points(p)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "collection"
            runner.collect(root, scores, groups, reference["partition"], method.sources(p), p)
            with mock.patch.object(evaluation.previous, "bootstrap_draws", side_effect=RuntimeError("saved-first injection")):
                with self.assertRaisesRegex(RuntimeError, "saved-first"):
                    evaluation.finalize(root, old_job / "evaluation", p, weight_job / "evaluation")
            with mock.patch.object(method.base, "load_model", side_effect=AssertionError("Saved-only recovery")), \
                 mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No extra parse")):
                result = evaluation.finalize(root, old_job / "evaluation", p, weight_job / "evaluation")
            self.assertEqual((result["new_metric_count_sets"], result["reused_metric_count_sets"]), (18, 45))
            self.assertEqual(set(result["endpoints"]), {"seq", "quarter", "logit_quarter"})
            self.assertEqual(set(result["comparisons"]), {"logit_quarter_minus_quarter", "quarter_minus_seq", "logit_quarter_minus_seq"})
            saved = method.data.read_json(weight_job / "evaluation/evaluation.json")
            self.assertEqual(result["comparisons"]["quarter_minus_seq"]["interpretation"], saved["comparisons"]["quarter_minus_seq"]["interpretation"])
            self.assertNotIn("selection", result)
            self.assertEqual(result["method_checks"]["increment_against_matched_er_passes"], result["comparisons"]["logit_quarter_minus_quarter"]["interpretation"]["pilot_observed_checks_pass"])
            before = (root / "evaluation.json").read_bytes()
            evaluation.finalize(root, old_job / "evaluation", p, weight_job / "evaluation")
            self.assertEqual(before, (root / "evaluation.json").read_bytes())


class LogitLowContracts(unittest.TestCase):
    @staticmethod
    def references(p: dict) -> tuple:
        def available(spec: dict) -> Path:
            local = method.data.ROOT / spec["local_small_job"]
            return local if local.is_dir() else method.data.ROOT / spec["linux_job"]
        return available(p["baseline"]), {
            arm: available(spec) for arm, spec in p["continuation_references"].items()}

    def test_low_logit_scope_and_wrong_supervision_rejected(self) -> None:
        p = method.contract("logit_low")
        self.assertEqual(p["arms"], {"logit_tenth": .1})
        self.assertEqual(p["evaluation"]["primary_comparison"], "logit_tenth_minus_tenth")
        self.assertEqual(p["loss"]["logit_mse"], .5)
        self.assertEqual(p["physical_updates"], 1728)
        with tempfile.TemporaryDirectory() as directory:
            root, manifest, reference = gate_fixture(Path(directory), p)
            self.assertEqual(len(runner.blind_gate(root, manifest, reference, p)), 6)
            name = "ABC_logit_tenth_stage2"
            log_path = root / manifest["training"][name]["path"]
            log = method.data.read_json(log_path)
            values = np.load(root / log["update_file"]["path"], allow_pickle=False)
            columns = {key: values[:, i] for i, key in enumerate(method.step_columns(p))}
            columns["weighted_history_total"][:] = .25 * columns["history_total"]
            columns["total"][:] = columns["current_total"] + columns["weighted_history_total"] + .5 * columns["logit_mse"]
            log["update_file"] = runner.save_array(root / "updates/wrong_weight.npy", values, root)
            method.data.write_json(log_path, log)
            manifest["training"][name] = method.data.record(log_path, root)
            with self.assertRaisesRegex(ValueError, "Historical weight"):
                runner.blind_gate(root, manifest, reference, p)

    def test_two_reference_sources_and_changed_start_rejected(self) -> None:
        p = method.contract("logit_low")
        original, jobs = self.references(p)
        reference = method.baseline(p, original, continuation_jobs=jobs)
        self.assertEqual(set(reference["continuation_references"]), {"tenth", "logit_quarter"})
        with tempfile.TemporaryDirectory() as directory:
            changed = copy.deepcopy(p)
            altered = Path(directory)
            spec = changed["continuation_references"]["tenth"]
            for name in spec["records"]:
                target = altered / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((jobs["tenth"] / name).read_bytes())
            target = altered / "run/manifest.json"
            manifest = method.data.read_json(target)
            manifest["restored_starts"]["ABC_tenth"]["model_state_sha256"] = "different_start"
            method.data.write_json(target, manifest)
            spec["records"]["run/manifest.json"] = method.data.record(target, altered)
            with self.assertRaisesRegex(ValueError, "different shared start"):
                method.baseline(changed, original, continuation_jobs={**jobs, "tenth": altered})

    def test_saved_three_comparisons_use_63_reference_sets(self) -> None:
        p = method.contract("logit_low")
        original, jobs = self.references(p)
        reference = method.baseline(p, original, continuation_jobs=jobs)
        groups = [fixtures.handmade_group(uid) for uid in reference["collected"]["group_ids"]]
        raw = np.tile(np.linspace(-2, 1, 378, dtype=np.float32), (60, 1))
        scores = {name: {role: raw if role == "raw" else raw.astype(np.float64) for role in method.ROLES}
                  for name in method.expected_points(p)}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "collection"
            runner.collect(root, scores, groups, reference["partition"], method.sources(p), p)
            with mock.patch.object(method.base, "load_model", side_effect=AssertionError("Saved only")), \
                 mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No labels")):
                result = evaluation.finalize(root, original / "evaluation", p, continuation_jobs=jobs)
            self.assertEqual((result["new_metric_count_sets"], result["reused_metric_count_sets"]), (18, 63))
            self.assertEqual(set(result["endpoints"]), {"seq", "tenth", "logit_quarter", "logit_tenth"})
            self.assertEqual(set(result["comparisons"]), {
                "logit_tenth_minus_tenth", "logit_tenth_minus_logit_quarter", "logit_tenth_minus_seq"})
            for arm, job in jobs.items():
                saved = method.data.read_json(job / "evaluation/evaluation.json")
                self.assertEqual(result["endpoints"][arm], saved["endpoints"][arm])
                for endpoint in ("O", "N", "Z"):
                    for metric in ("map", "recall_at_5"):
                        delta = result["comparisons"]["logit_tenth_minus_" + arm]["primary"][endpoint][metric]["mean"]
                        expected = (result["endpoints"]["logit_tenth"]["primary"][endpoint][metric]["mean"]
                                    - saved["endpoints"][arm]["primary"][endpoint][metric]["mean"])
                        self.assertAlmostEqual(delta, expected, places=14)
            self.assertNotIn("selection", result)
            self.assertEqual(result["method_checks"]["increment_against_matched_er_passes"],
                             result["comparisons"]["logit_tenth_minus_tenth"]["interpretation"]["pilot_observed_checks_pass"])
            before = (root / "evaluation.json").read_bytes()
            evaluation.finalize(root, original / "evaluation", p, continuation_jobs=jobs)
            self.assertEqual(before, (root / "evaluation.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
