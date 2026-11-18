"""Targeted Linux CPU evidence for the new six-way entry. Synthetic groups only."""
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import step28_history_benchmark as b
import step28_history_benchmark_evaluate as ev
from test_step28_continual_comparison import Encoder, group

EVIDENCE = {}
DEVICE = "cpu"


class BenchmarkTest(unittest.TestCase):
    def test_six_native_training_routes_and_real_update(self):
        rows = b.plan()
        self.assertEqual(len({r["id"] for r in rows}), 54)
        self.assertEqual(len({json.dumps(r["native"], sort_keys=True) for r in rows}), 54)
        self.assertEqual([r["id"] for r in rows if b.gradient_stopping(r)], ["CAB_s6_L2R_text_fixed"])
        limits = dict(allocator_bytes=1024, rss_bytes=2**32, disk_bytes=2**30, small_bytes=2**30,
                      free_reserve_bytes=1)
        evidence = {}
        current = [group(i) for i in range(48)]
        for row in rows[:6]:
            name = row["name"]
            spec = b.native(name, "ABC", 4)
            # Observe the actual production dispatcher at the native runner boundary.
            with (patch.object(b.five_run, "train_trajectory") as five_runner,
                  patch.object(b.comparison_run, "train_trajectory") as comparison_runner,
                  patch.object(b.forever_run, "execute") as forever_runner):
                b.train(Path("/unused"), row, time.monotonic()+120, dict(campaign="/unused", limits=limits))
            self.assertEqual(sum(m.call_count for m in (five_runner, comparison_runner, forever_runner)), 1)
            torch.manual_seed(23)
            if name == b.forever.NAME:
                self.assertEqual(forever_runner.call_args.args[1], spec)
                c = b.forever.config(spec)
                module = b.five.core.build_model(Encoder(), 8, 5).to(DEVICE)
                b.five.checkpoint_encoder(module.encoder)
                optimizer = b.five.core.make_optimizer(module, c)
                before = b.five.core.state_digest(module.state_dict())
                learner = b.forever.Learner(spec, module, optimizer, c)
                learner.update(current[0], "current")
                self.assertEqual(learner.counts["current"], 1)
                scores = b.forever.record.score(module, [current[0]], c)
            elif name in b.comparison.NEW_METHODS:
                self.assertEqual(comparison_runner.call_args.args[1], spec)
                self.assertFalse(comparison_runner.call_args.kwargs["calibration_stop_on_gradient"])
                method, c = b.comparison.Method(spec), b.comparison.config(spec)
                module = b.five.core.build_model(Encoder(), 8, 5).to(DEVICE)
                if name == "L2R_text_fixed":
                    del module.head
                    b.five.checkpoint_encoder(module.encoder[0])
                    optimizer = torch.optim.AdamW(module.encoder.parameters(), lr=1e-5, foreach=False)
                else:
                    b.five.checkpoint_encoder(module.encoder)
                    optimizer = b.five.core.make_optimizer(module, c)
                method.bind(module, method.metadata())
                before = b.five.core.state_digest(module.state_dict())
                stage = method.stage(1, current, method.memory(), module)
                method.update(module, optimizer, stage, current[0], c, 1)
                scores = method.score(module, [current[0]], c)
            else:
                self.assertEqual(five_runner.call_args.args[1], spec)
                self.assertIs(five_runner.call_args.kwargs["train_inputs"], b.comparison_run.training_inputs)
                runtime = five_runner.call_args.args[3]
                method, c = runtime.Method(spec), runtime.config(spec)
                module = b.five.core.build_model(Encoder(), 8 if name == b.H else 16, 5).to(DEVICE)
                b.five.checkpoint_encoder(module.encoder)
                method.bind(module, method.metadata())
                optimizer = b.five.core.make_optimizer(module, c)
                # A declared synthetic second-stage boundary, followed by one actual update.
                for parameter in module.parameters():
                    optimizer.state[parameter] = dict(step=torch.tensor(288.), exp_avg=torch.zeros_like(parameter),
                                                       exp_avg_sq=torch.zeros_like(parameter))
                before = b.five.core.state_digest(module.state_dict())
                target = None
                if name == b.H:
                    target = b.five.record.reference(module, current[1], c)+np.float32(.5)
                elif name == "LOGIT0.1_ref":
                    target = method.score(module, [current[1]], c)[0]+np.float32(.5)
                log = method.update(module, optimizer, current[0], current[1], target, c, 2, 1)
                if name == b.H:
                    hist = log["history"]
                    self.assertAlmostEqual(hist["teacher_contribution"], b.BETA*hist["distillation"], places=5)
                    self.assertAlmostEqual(hist["supervised_contribution"], b.ALPHA*hist["supervised"], places=5)
                    self.assertGreater(hist["teacher_contribution"], 0)
                scores = method.score(module, [current[0]], c)
            self.assertNotEqual(before, b.five.core.state_digest(module.state_dict()))
            self.assertEqual(next(module.parameters()).device.type, DEVICE)
            self.assertEqual(scores.shape, (1, 378))
            self.assertTrue(np.isfinite(scores).all())
            evidence[name] = dict(adam_step=b.five.parent.adam_step(optimizer), finite_scores=scores.size)
        EVIDENCE[f"six_real_{DEVICE}_updates"] = evidence

    def test_endpoint_with_only_current_calibration(self):
        spec = b.native("ER0.1_ref", "ABC", 4)
        method, c = b.five.Method(spec), b.five.config(spec)
        torch.manual_seed(91)
        model = b.five.core.build_model(Encoder(), 16, 5).to(DEVICE)
        method.bind(model, method.metadata())
        optimizer = b.five.core.make_optimizer(model, c)
        for p in model.parameters():
            optimizer.state[p] = dict(step=torch.tensor(288.), exp_avg=torch.zeros_like(p), exp_avg_sq=torch.zeros_like(p))
        current, calibration = [group(i) for i in range(48)], [group(i) for i in range(48, 60)]
        limits = dict(allocator_bytes=1024, rss_bytes=2**32, disk_bytes=2**30, small_bytes=2**30, free_reserve_bytes=1)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for sub in ("maps", "memory", "models", "work", "scores", "points"):
                (root/sub).mkdir()
            budget = b.Budget(root, time.monotonic()+240, limits)
            point, memory, _ = b.five_run.checkpoint(root, method, model, optimizer, None, current, calibration,
                                                     None, c, 1, None, budget)
            self.assertEqual(point["development_ids"], [])
            self.assertEqual(point["calibration_ids"], [g.uid for g in calibration])
            self.assertEqual(set(point["scores"]), {"calibration", "full-calibration", "inference-calibration"})
            self.assertTrue(point["full_restore_verified"] and point["inference_restore_verified"])
            self.assertLessEqual(len(memory.to_bytes()), 1048576)
            EVIDENCE["train_only_endpoint"] = dict(current_calibration_groups=12, old_dev_scores=0,
                    restored_full=True, restored_inference=True, memory_bytes=len(memory.to_bytes()))

    def test_missing_receipt_and_misaligned_blind_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipts = [dict(id=r["id"], success=True, bucket="training") for r in b.plan()[:-1]]
            b.data.write_json(root/"training_complete.json", dict(sources=[], completed=receipts,
                             status="ALL_TRAINING_COMPLETE_EVALUATION_CLOSED"))
            with self.assertRaisesRegex(ValueError, "54 successful"):
                b.completed(root, {"sources": []})
            self.assertFalse((root/"evaluation_access.json").exists())
        ids = [f"{d}-{i}" for d in "ABC" for i in range(40)]
        saved = dict(groups=ids, domains=[key[0] for key in ids], runs={}, forwards=12960)
        for row in b.plan():
            cells = {}
            for stage, pos in ev.CELLS:
                key, domain = f"{stage}_{pos}", row["order"][pos-1]
                cells[key] = dict(domain=domain, ids=[i for i in ids if i[0] == domain],
                                  scores={role: {"path": f'{row["id"]}_{key}_{role}.npy'} for role in ev.ROLES})
            saved["runs"][row["id"]] = dict(native=row["native"], cells=cells)
        ev.validate_cells(saved)
        cells = next(iter(saved["runs"].values()))["cells"]
        last = cells.pop("3_3")
        with self.assertRaises(ValueError):
            ev.validate_cells(saved)
        cells["3_3"] = last
        last["ids"] = list(reversed(last["ids"]))
        with self.assertRaises(ValueError):
            ev.validate_cells(saved)
        EVIDENCE["blind_gate"] = dict(complete_cells=324, missing_trajectory="rejected_before_pairs",
                missing_cell="rejected", wrong_group_alignment="rejected")

    def test_blind_loader_all_six_native_endpoint_formats(self):
        from types import SimpleNamespace
        groups = {}
        for d, domain in enumerate("ABC"):
            labelled = [group(d*40+i) for i in range(40)]
            groups[domain] = [b.data.Group(g.uid, g.sellers, g.items) for g in labelled]
        checked = {}
        with tempfile.TemporaryDirectory() as directory:
            for row in b.plan()[:6]:
                name = row["name"]
                spec = b.native(name, "ABC", 4)
                root = Path(directory)/name
                root.mkdir()
                def tiny():
                    model = b.five.core.build_model(Encoder(), 16 if name in ("ER0.1_ref", "LOGIT0.1_ref") else 8, 5).to(DEVICE)
                    if name == "L2R_text_fixed":
                        del model.head
                    return model
                model = tiny()
                is_forever = name == b.forever.NAME
                runtime = b.history if name == b.H else b.comparison if name in b.comparison.NEW_METHODS else b.five
                method = None if is_forever else runtime.Method(spec)
                points = {}
                for stage in (1, 2, 3):
                    with torch.no_grad():
                        next(model.parameters()).add_(.03*stage)
                    maps = dict(first={"a": 1.0, "b": 0.0}, stage={"a": 1.0, "b": .1*stage})
                    counts = dict(current=288*stage, replay=0, fixture=0)
                    metadata = (dict(spec=spec.identity(), counts=counts) if is_forever else
                                {**method.metadata(), "modes": [m.training for m in model.modules()],
                                 "rng": b.five.diag.rng_state()})
                    path = root/f"stage{stage}.pt"
                    state = b.five.core.save_state(path, model, None, dict(stage=stage, maps=maps, **metadata))
                    points[str(stage)] = dict(model={**state, **b.data.record(path, root)},
                                state_sha256=state["state_sha256"], counts=counts, maps=maps)
                factory = b.forever if is_forever else runtime.Method
                with patch.object(factory, "fresh", side_effect=lambda *args: (tiny(), None)):
                    saved = ev.blind_trajectory({**row, "root": root, "record": {"points": points}},
                                                groups, root, SimpleNamespace(check=lambda: None))
                self.assertEqual(len(saved["cells"]), 6)
                for key, cell in saved["cells"].items():
                    raw = ev.load_blind(root, cell["scores"]["raw"])
                    roles, _ = ev.shared.calibrated_roles(raw, points[key[0]]["maps"])
                    for role in ev.ROLES:
                        np.testing.assert_array_equal(roles[role], ev.load_blind(root, cell["scores"][role]))
                checked[name] = dict(restored_endpoints=3, blind_cells=6, scored_groups=240)
        EVIDENCE["native_blind_restore"] = checked

    def test_reservation_and_historical_gate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            gate = root/"gate.json"
            b.data.write_json(gate, {"status": "APPROVED_COMPARISON_IMPLEMENTATION_AND_RESOURCES"})
            with self.assertRaisesRegex(ValueError, "separate reviewed"):
                b.read_gate(gate)
            end = time.monotonic()+80
            row = dict(pid=os.getpid(), cwd=str(b.SCRIPTS.parent), command=[
                sys.executable, "-B", str(Path(b.__file__).resolve()), "worker", "--gate", str(gate),
                "--id", root.name, "--deadline", str(end)])
            b.data.write_json(root/"process.json", row)
            b.check_dispatch(root, gate, "worker", end, 120)
            b.data.write_json(root/"process.json", {**row, "pid": os.getpid()+1})
            with self.assertRaisesRegex(ValueError, "reservation"):
                b.check_dispatch(root, gate, "worker", end, 120)
        EVIDENCE["dispatch"] = dict(own_claim="accepted", foreign_pid="rejected", historical_gate="rejected")

    def test_table_and_single_logical_opening(self):
        cells = {"1_1": .8, "2_1": .7, "2_2": .6, "3_1": .5, "3_2": .4, "3_3": .9}
        values = ev.field_means(cells)
        for key, expected in dict(O=.45, F=.25, A_acq=.7, N=.75, Z=.9, final_mean=.6).items():
            self.assertAlmostEqual(values[key], expected)
        per = {r["id"]: dict(values) for r in b.plan()}
        per["CAB_s6_"+b.H]["O"] += .09
        pairs = ev.paired_table(per)["h_minus_baseline"]
        self.assertEqual(len(pairs), 5)
        self.assertTrue(all(abs(p["mean"]["O"]-.01)<1e-12 for p in pairs.values()))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root/"generation"/"pairs.jsonl"
            path.parent.mkdir()
            g = group(5)
            owners = [{"seller_uid": seller, "controller_uid": str(i//2 if i<16 else 8+(i-16)//3)}
                      for i, seller in enumerate(g.sellers)]
            # Generate only one handwritten label fixture, never the prospective cohort.
            rows = list(ev.shared.generator.pair_rows([dict(r, group_uid=g.uid) for r in owners]))
            path.write_text(json.dumps(rows)+"\n", encoding="utf-8")
            rec = {"files": {"pairs.jsonl": b.data.record(path, path.parent)}}
            self.assertEqual(tuple(ev.open_pairs(root, rec, [g])[g.uid]), g.labels)
            with self.assertRaisesRegex(ValueError, "already attempted"):
                ev.open_pairs(root, rec, [g])
        EVIDENCE["paired_report"] = dict(contrasts=5, matched_per_contrast=9, hand_computed_delta=.01,
                                        second_pair_opening="rejected")


if __name__ == "__main__":
    if sys.platform != "linux":
        raise SystemExit("Use the existing Linux py310")
    if sys.argv[1:] == ["--cuda"]:
        DEVICE = "cuda"
        torch.cuda.set_per_process_memory_fraction((512*2**20)/torch.cuda.get_device_properties(0).total_memory)
        b.five.parent.configure_numerics()
        cases = ("test_six_native_training_routes_and_real_update", "test_endpoint_with_only_current_calibration",
                 "test_blind_loader_all_six_native_endpoint_formats")
    elif not sys.argv[1:] and not torch.cuda.is_available():
        cases = unittest.defaultTestLoader.getTestCaseNames(BenchmarkTest)
    else:
        raise SystemExit("CPU checks require invisible GPUs; the bounded CUDA probe takes --cuda")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.monotonic()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(BenchmarkTest(name) for name in cases))
    if DEVICE == "cuda":
        torch.cuda.synchronize()
        EVIDENCE["device"] = dict(name=torch.cuda.get_device_name(0), allocated_peak=torch.cuda.max_memory_allocated(),
                                  reserved_peak=torch.cuda.max_memory_reserved())
    print(json.dumps(dict(passed=result.wasSuccessful(), cases=result.testsRun, seconds=time.monotonic()-started,
          cpu_affinity=sorted(os.sched_getaffinity(0)), python=sys.executable, formal_data_reads=0,
          pretrained_loads=0, gpu_used=DEVICE == "cuda", evidence=EVIDENCE), sort_keys=True), flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)
