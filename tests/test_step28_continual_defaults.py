"""Only the original-method adapter delta; handwritten inputs, Linux py310 only."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import step28_continual_defaults as m
import step28_continual_defaults_run as run
from test_step28_record_replay import handmade_group, TinyEncoder
from test_step28_positive_ranking import independent_pool, independent_loss, independent_rank, LOOKUP


def old_loss(scores, group):
    terms = []
    for q in range(28):
        ids = [LOOKUP[tuple(sorted((q, j)))] for j in range(28) if j != q]
        positives = [k for k in ids if group.labels[k]]
        terms.append(torch.logsumexp(scores[ids], 0)-scores[positives].mean())
    return independent_loss(scores, group)-independent_rank(scores, group.labels)+torch.stack(terms).mean()


def reference(table, group, seed, arm, target=None, *, positive=False, mean_only=False, no_regroup=False):
    pooling = "H_pos" if arm == "G_half" and not mean_only else "S"
    slots = [m.five.record.assignment(group), m.five.record.assignment(group, seed)]
    values = [independent_pool(table, a, pooling) for a in slots]
    loss = independent_loss if positive else old_loss
    supervised = loss(values[0], group) if no_regroup else sum(loss(v, group) for v in values)/2
    if target is None:
        return supervised
    teacher = independent_pool(torch.as_tensor(target, dtype=table.dtype), slots[0], pooling)
    return .1*supervised+.5*(values[0]-teacher).square().mean()


def fixture(name, step=624):
    spec = m.Spec(name, "ABC", 4)
    method, c = m.Method(spec), m.config(spec)
    torch.manual_seed(41)
    model = m.core.build_model(TinyEncoder(), 8, 5)
    m.five.checkpoint_encoder(model.encoder)
    method.bind(model, method.metadata())
    optimizer = m.core.make_optimizer(model, c)
    if step:
        # Handwritten optimizer state; not evidence of prior formal training.
        for p in model.parameters():
            optimizer.state[p] = dict(step=torch.tensor(float(step)), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
    return spec, method, c, model, optimizer


class DefaultsTests(unittest.TestCase):
    def evidence(self, name, row):
        m.data.write_json(Path(os.environ["DEFAULTS_TEST_ROOT"])/(name+".json"), row)

    def test_cli_rejection_preserves_paths_and_original_error(self):
        with tempfile.TemporaryDirectory(dir=os.environ["DEFAULTS_TEST_ROOT"]) as temporary:
            root = Path(temporary)
            job, gate_path, spec_path = root/"job", root/"gate.json", root/"spec.json"
            spec = m.specifications()[0]
            m.data.write_json(spec_path, spec.identity())
            gate = dict(scope="original_s_h_three_stage", train_data=True, campaign=str(root.resolve()))
            command = ["defaults", "--job", str(job), "--spec", str(spec_path),
                       "--deadline", "99999", "--gate", str(gate_path)]
            with mock.patch.object(sys, "argv", command):
                m.data.write_json(gate_path, {**gate, "train_data": False})
                with self.assertRaisesRegex(ValueError, "Fresh independently authorized"):
                    run.main()
                self.assertFalse(job.exists())
                m.data.write_json(gate_path, gate)
                error = RuntimeError("before_job_creation")
                with mock.patch.object(run.inherited, "execute", side_effect=error):
                    with self.assertRaises(RuntimeError) as caught:
                        run.main()
                    self.assertIs(caught.exception, error)
                self.assertFalse(job.exists())
                job.mkdir()
                original = b"prior failure must remain unchanged\n"
                (job/"failure.json").write_bytes(original)
                with mock.patch.object(run.inherited, "execute") as inherited:
                    with self.assertRaisesRegex(ValueError, "Fresh independently authorized"):
                        run.main()
                    inherited.assert_not_called()
                self.assertEqual(list(job.iterdir()), [job/"failure.json"])
                self.assertEqual((job/"failure.json").read_bytes(), original)
                (job/"failure.json").unlink()
                job.rmdir()
                def accepted_failure(*args, **kwargs):
                    job.mkdir()
                    raise RuntimeError("after_owned_job_creation")
                with mock.patch.object(run.inherited, "execute", side_effect=accepted_failure):
                    with self.assertRaisesRegex(RuntimeError, "after_owned_job_creation"):
                        run.main()
                self.assertEqual(m.data.read_json(job/"failure.json")["status"], "STOP_NO_RETRY")
        self.evidence("cli_rejection", dict(early_error_preserved=True, absent_job_not_created=True,
            existing_job_unchanged=True, accepted_job_failure_recorded=True,
            downstream_run_simulated=True, actual_native_updates=0))

    def test_original_objectives_and_missing_operations(self):
        group = handmade_group("current")
        target = (np.sin(np.arange(1540)/7)*3).astype(np.float32)
        evidence = {}
        for name in m.METHODS:
            spec = m.Spec(name, "ABC", 4)
            method = m.Method(spec)
            table = torch.tensor(np.sin(np.arange(1540)*.3), dtype=torch.float64, requires_grad=True)
            differences = {}
            for teacher in (None, target):
                actual, _ = method.tuned.objective(table, group, 17, spec.arm, teacher)
                expected = reference(table, group, 17, spec.arm, teacher)
                ga, = torch.autograd.grad(actual, table, retain_graph=True)
                ge, = torch.autograd.grad(expected, table, retain_graph=True)
                torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
                torch.testing.assert_close(ga, ge, atol=1e-12, rtol=1e-12)
                mutants = dict(positive_rank=reference(table, group, 17, spec.arm, teacher, positive=True),
                               missing_regroup=reference(table, group, 17, spec.arm, teacher, no_regroup=True))
                if spec.arm == "G_half":
                    mutants["mean_instead_of_half"] = reference(table, group, 17, spec.arm, teacher, mean_only=True)
                if teacher is not None:
                    mutants["missing_teacher"] = .1*reference(table, group, 17, spec.arm)
                for key, value in mutants.items():
                    gm, = torch.autograd.grad(value, table, retain_graph=True)
                    difference = float((ga-gm).abs().max())
                    self.assertGreater(difference, 1e-7)
                    differences[("current_" if teacher is None else "history_")+key] = difference
            evidence[name] = differences
        self.evidence("objectives", evidence)

    def test_actual_updates_scoring_and_restore(self):
        current, history = handmade_group("current"), handmade_group("history")
        target = (np.sin(np.arange(1540)/7)*3).astype(np.float32)
        evidence = {}
        for name in m.METHODS:
            spec, method, c, model, optimizer = fixture(name)
            direct = copy.deepcopy(model)
            direct.encoder._five_outer_checkpoint = False
            other = m.core.make_optimizer(direct, c)
            other.load_state_dict(copy.deepcopy(optimizer.state_dict()))
            before = {k: v.clone() for k, v in model.state_dict().items()}
            metadata = {**method.metadata(), "rng": m.diag.rng_state(), "modes": [v.training for v in model.modules()]}
            with tempfile.TemporaryDirectory(dir=os.environ["DEFAULTS_TEST_ROOT"]) as temporary:
                path = Path(temporary)/"full.pt"
                receipt = m.core.save_state(path, model, optimizer, metadata)
                actual = method.update(model, optimizer, current, history, target, c, 3, 49)
                direct.train()
                other.zero_grad(set_to_none=True)
                for role, group, teacher in (("current", current, None), ("history", history, target)):
                    torch.manual_seed(m.data.seed_for(spec.roots["schedule"], spec.order, 3, 49, role, "dropout"))
                    z = m.five.record.record_vectors(direct, group, c)
                    i, j = torch.triu_indices(len(z), len(z), 1)
                    table = direct.head(torch.cat(((z[i]-z[j]).abs(), z[i]*z[j]), 1)).flatten()
                    seed = m.data.seed_for(spec.roots["schedule"], spec.order, 3, 49, role, group.uid, "regroup")
                    reference(table, group, seed, spec.arm, teacher).backward()
                torch.nn.utils.clip_grad_norm_(direct.parameters(), 1.)
                for group, rate in zip(other.param_groups, spec.rates(49), strict=True):
                    group["lr"] = rate
                other.step()
                errors = []
                changes = {}
                for (key, p), q in zip(model.named_parameters(), direct.parameters(), strict=True):
                    torch.testing.assert_close(p, q, atol=2e-6, rtol=2e-5)
                    for field in ("step", "exp_avg", "exp_avg_sq"):
                        torch.testing.assert_close(optimizer.state[p][field], other.state[q][field], atol=2e-6, rtol=2e-5)
                    errors.append(float((p-q).detach().abs().max()))
                    changes[key] = float((p.detach()-before[key]).abs().max())
                self.assertTrue(any(v > 0 for k, v in changes.items() if k.startswith("encoder.")))
                self.assertTrue(any(v > 0 for k, v in changes.items() if k.startswith("head.")))
                self.assertEqual((actual["encoder_lr"], actual["head_lr"]), spec.rates(49))
                self.assertEqual(m.parent.adam_step(optimizer), 625)
                digest = m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict()))
                method.restore(path, receipt, model, optimizer)
                repeated = method.update(model, optimizer, current, history, target, c, 3, 49)
                self.assertEqual(actual, repeated)
                self.assertEqual(digest, m.core.state_digest(dict(model=model.state_dict(), optimizer=optimizer.state_dict())))
                with m.diag.neutral(model):
                    actual_scores = method.score(model, [current], c)
                    z = m.five.record.record_vectors(model, current, c)
                    table = m.five.record.pair_table(model, z)
                    expected_scores = independent_pool(table, m.five.record.assignment(current),
                        "H_pos" if spec.arm == "G_half" else "S").float().numpy()
                    np.testing.assert_allclose(actual_scores[0], expected_scores, atol=2e-6, rtol=2e-5)
                inference = m.core.save_state(Path(temporary)/"infer.pt", model, None, metadata)
                method.restore(Path(temporary)/"infer.pt", inference, model, None)
                with m.diag.neutral(model):
                    np.testing.assert_array_equal(actual_scores, method.score(model, [current], c))
            # No-history stage and unit learning-rate multiplier also reach the real updater.
            first_spec, first_method, first_c, first_model, first_optimizer = fixture(name, step=0)
            first = first_method.update(first_model, first_optimizer, current, None, None, first_c, 1, 1)
            self.assertEqual(m.parent.adam_step(first_optimizer), 1)
            self.assertNotIn("history", first)
            self.assertEqual((first["encoder_lr"], first["head_lr"]), first_spec.rates(1))
            memory = method.new_memory(dict(a=1., b=0.))
            payload = memory.to_bytes()
            self.assertEqual(method.memory_from(payload).to_bytes(), payload)
            wrong_name = "H_default" if name == "S_default" else "S_default"
            with self.assertRaises(ValueError):
                m.Method(m.Spec(wrong_name, "ABC", 4)).memory_from(payload)
            evidence[name] = dict(max_parameter_error=max(errors), parameter_changes=changes,
                actual_adam_step=625, first_stage_adam_step=1, next_update_bitwise_equal=True,
                inference_bitwise_equal=True, independent_scores_match=True, empty_memory_identity_checked=True)
        self.evidence("updates_restore", evidence)

    def test_configuration_and_actual_runner_wiring(self):
        specs = m.specifications()
        self.assertEqual(len(specs), 18)
        self.assertEqual(len({s.run_id for s in specs}), 18)
        self.assertEqual(len(m.five.specifications()), 45)
        for spec in specs:
            self.assertEqual(m.Spec.from_dict(spec.identity()), spec)
            self.assertEqual((spec.lambda_rank, spec.lr_multiplier), (1., 1.))
            self.assertEqual(spec.roots, m.five.Spec("S_tuned", spec.order, spec.seed).roots)
            self.assertEqual(m.config(spec)["continual_spec"], spec.identity())
        spec = specs[0]
        with tempfile.TemporaryDirectory(dir=os.environ["DEFAULTS_TEST_ROOT"]) as temporary:
            root = Path(temporary)
            job, gate_path = root/spec.run_id, root/"gate.json"
            gate = dict(scope="original_s_h_three_stage", train_data=True, campaign=str(root.resolve()),
                        status="APPROVED_IMPLEMENTATION_AND_RESOURCES", sources=run.sources(),
                        runs=[s.identity() for s in specs])
            m.data.write_json(gate_path, gate)
            # Stop at config construction, before CUDA/model/corpus entry. This
            # tests the real wrapper->runner selection, not a simulated training pass.
            with mock.patch.object(torch, "set_num_threads"), mock.patch.object(torch, "set_num_interop_threads"), \
                    mock.patch.object(run.inherited, "Budget"), \
                    mock.patch.object(m, "config", side_effect=RuntimeError("selected_original_config")) as config:
                with self.assertRaisesRegex(RuntimeError, "selected_original_config"):
                    run.execute(job, spec, 99999., gate_path)
                config.assert_called_once_with(spec)
                gate["runs"] = [s.identity() for s in m.five.specifications()]
                m.data.write_json(gate_path, gate)
                with self.assertRaisesRegex(ValueError, "run-list gate"):
                    run.execute(job, spec, 99999., gate_path)
                self.assertEqual(config.call_count, 1)
        self.evidence("wiring", dict(original_runs=18, old_five_runs=45, old_gate_rejected=True,
            actual_runner_selected_original_config=True, gpu_or_formal_data_access=0))


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    unittest.main(verbosity=2)
