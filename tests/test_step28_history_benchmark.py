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
            winners = {r["id"]: {} for r in b.plan()[:-1]}
            b.data.write_json(root/"training_complete.json", dict(sources=[], winners=winners,
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
            task = b.plan()[0]["id"]
            process = {**row, **b.dispatch.kernel_identity(os.getpid()), "attempt":2, "task_id":task,
                       "command":[sys.executable, "-B", str(Path(b.__file__).resolve()), "worker",
                                  "--gate", str(gate), "--id", task, "--deadline", str(end), "--attempt", "2"]}
            b.data.write_json(root/"process.json", process)
            b.check_dispatch(root, gate, "worker", end, 120, task_id=task, attempt=2)
            b.data.write_json(root/"process.json", {**process, "attempt":1})
            with self.assertRaisesRegex(ValueError, "kernel process identity"):
                b.check_dispatch(root, gate, "worker", end, 120, task_id=task, attempt=2)
        EVIDENCE["dispatch"] = dict(own_claim="accepted", foreign_pid="rejected", historical_gate="rejected",
                                    same_kernel_second_attempt="accepted", wrong_attempt="rejected")

    def test_generation_rechecks_162_endpoint_files(self):
        from types import SimpleNamespace
        rows = b.plan()
        partition = {"calibration": [dict(domain=d, group_uid=f"{d}-{i}") for d in "ABC" for i in range(12)]}
        access = dict(train=1, valid=0, heldout=0, owners=0, worker_train_reads=1)
        mapping = {"a": 1.0, "b": 0.0}
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)
            winners = {}
            for row in rows:
                retry = row == rows[-1]
                attempt = 2 if retry else 1
                task = row["id"]
                job = campaign/"training"/task/f"attempt-{attempt}"
                root = job/"run"
                root.mkdir(parents=True)
                b.data.write_json(job/"dispatch_receipt.json", dict(task_id=task, attempt=attempt, slot=0, success=True, released=True))
                is_forever = row["name"] == b.forever.NAME
                status = (b.forever_run.COMPLETE if is_forever else b.comparison_run.COMPLETE
                          if row["name"] in b.comparison.NEW_METHODS else b.TRAIN_CAL_COMPLETE)
                points = {}
                for stage in (1, 2, 3):
                    path = root/f"model-{stage}.pt"
                    path.write_bytes(f"{row['id']}-{stage}".encode())
                    map_path = root/f"map-{stage}.json"
                    ftol = 0. if b.gradient_stopping(row) else b.five.parent.calibration.OPTIONS["ftol"]
                    b.data.write_json(map_path, {**mapping, "status":"PASS_CALIBRATION_FIT", "options":{"ftol":ftol}})
                    points[str(stage)] = dict(stage=stage, adam_step=288*stage, counts={"current":288*stage},
                        model=b.data.record(path, root), full_restore_verified=True, inference_restore_verified=True,
                        maps=dict(stage=mapping, first=mapping), calibration_ids=[f"{row['order'][stage-1]}-{i}" for i in range(12)])
                    points[str(stage)]["calibration" if is_forever else "mapping"] = b.data.record(map_path, root)
                b.data.write_json(root/"completion.json", dict(spec=row["native"], source_files=[], points=points,
                    updates=864, status=status, partition=partition, access=access,
                    counts=dict(current=864, fixture=0, replay=0)))
                winners[task] = dict(attempt=attempt, receipt=b.queue.record(job/"dispatch_receipt.json"),
                                     completion=b.queue.record(root/"completion.json"))
                if retry:
                    first = job.parent/"attempt-1"/"dispatch_receipt.json"
                    first.parent.mkdir()
                    b.data.write_json(first, dict(task_id=task, attempt=1, slot=0, success=False, released=True, failure_kind="CUDA_OOM"))
                    winners[task]["first_oom_receipt"] = b.queue.record(first)
            b.data.write_json(campaign/"training_complete.json", dict(status="ALL_TRAINING_COMPLETE_EVALUATION_CLOSED",
                              sources=[], winners=winners))
            budget = SimpleNamespace(check=lambda: None)
            with patch.object(ev.shared, "generate") as generate:
                ev.run("generation", campaign, campaign/"generation", {"sources":[]}, budget)
                self.assertEqual(generate.call_count, 1)
                generate.reset_mock()
                for index in (0, len(rows)-1):
                    task = rows[index]["id"]
                    path = campaign/"training"/task/f"attempt-{winners[task]['attempt']}"/"run"/"model-3.pt"
                    original = path.read_bytes()
                    path.unlink()
                    with self.assertRaises(FileNotFoundError):
                        ev.run("generation", campaign, campaign/"generation", {"sources":[]}, budget)
                    path.write_bytes(b"x"*len(original))
                    with self.assertRaisesRegex(ValueError, "File differs"):
                        ev.run("generation", campaign, campaign/"generation", {"sources":[]}, budget)
                    path.write_bytes(original)
                last = rows[-1]["id"]
                winners[last]["first_oom_receipt"] = winners[last]["receipt"]
                b.data.write_json(campaign/"training_complete.json",
                                  dict(status="ALL_TRAINING_COMPLETE_EVALUATION_CLOSED", sources=[], winners=winners))
                with self.assertRaisesRegex(ValueError, "different training attempt"):
                    ev.run("generation", campaign, campaign/"generation", {"sources":[]}, budget)
                generate.assert_not_called()
            self.assertFalse((campaign/"evaluation_access.json").exists())
        EVIDENCE["before_generation"] = dict(handwritten_endpoints=162, intact="passed",
                missing_H="blocked", changed_H="blocked", missing_FOREVER="blocked", changed_FOREVER="blocked",
                substituted_first_oom_receipt="blocked", official_groups_generated=0, official_pairs_opened=0)

    def test_interrupted_worker_validation_keeps_receipt_and_fee(self):
        with tempfile.TemporaryDirectory() as directory:
            campaign = Path(directory)/"campaign"
            limits = dict(rss_bytes=4*2**30, disk_bytes=2**30, free_reserve_bytes=1)
            scope = dict(campaign=str(campaign), workspace=directory, limits=limits, workers=2,
                         prior_used_seconds={"training":7., "control":0.}, budget_seconds={"training":240., "control":60.})
            tasks = [dict(id=name, seconds=100., gpu=False) for name in ("interrupted_fixture", "peer_fixture")]
            def command(row, job, deadline):
                delay = ".08" if row["id"] == "interrupted_fixture" else "20"
                return [sys.executable, "-B", "-c", f"import time; time.sleep({delay})"]
            def validate(row, job):
                raise KeyboardInterrupt("synthetic interruption during endpoint verification")
            core = min(os.sched_getaffinity(0))
            # Two real children share the one CPU allowed to this synthetic probe.
            with (patch.object(b.queue, "GpuMemory") as monitor,
                  patch.object(b.queue.psutil.Process, "cpu_affinity", return_value=[core, core])):
                monitor.return_value.snapshot.return_value = {"free": 2**30, "processes": {}}
                with self.assertRaises(KeyboardInterrupt):
                    b.queue.dispatch(scope, tasks, command, validate, poll_seconds=.02)
            receipt, peer = [b.data.read_json(campaign/t["id"]/"dispatch_receipt.json") for t in tasks]
            failure = b.data.read_json(campaign/"failure.json")
            self.assertEqual(failure["completed"], [receipt, peer])
            self.assertEqual(failure["active"], {})
            self.assertFalse(receipt["success"])
            self.assertIn("KeyboardInterrupt", receipt["error"])
            self.assertGreater(receipt["actual_seconds"], 0)
            self.assertFalse(peer["success"])
            self.assertGreater(peer["actual_seconds"], 0)
            peer_pid = b.data.read_json(campaign/"peer_fixture"/"process.json")["pid"]
            self.assertFalse(b.queue.psutil.pid_exists(peer_pid))
            self.assertAlmostEqual(failure["used"]["training"], 7.+receipt["actual_seconds"]+peer["actual_seconds"])
            self.assertFalse((campaign/"training_complete.json").exists())
            EVIDENCE["interrupted_validation"] = dict(real_cpu_child=True, exit_status="interrupted",
                    receipts_per_child=1, inherited_seconds=7., charged_seconds=receipt["actual_seconds"],
                    peer_charged_seconds=peer["actual_seconds"], peer_drained=True, downstream_closed=True)

    def test_gate_binds_authorized_sources_and_stop_reserve(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scripts = root/"workspace"/"scripts"
            scripts.mkdir(parents=True)
            source = scripts/"handwritten.py"
            source.write_text("# synthetic source identity\n")
            analysis = root/"analysis.txt"
            analysis.write_text("handwritten test fixture")
            files = [b.queue.record(source)]
            scope = dict(experiment=b.EXPERIMENT, runs=b.plan(), cohort=b.cohort(), actions=["training", *b.POST],
                         sources=files, workers=1, worker_failure_policy=b.dispatch.POLICY, analysis_plan=b.queue.record(analysis),
                         cpu_test_seconds=0, other_gpu_commitments=[], allow_observed_foreign_gpu_processes=False,
                         seconds_per_method={name:90 for name in b.NAMES}, post_seconds={name:90 for name in b.POST},
                         prior_used_seconds={k:0 for k in ("control", "training", *b.POST)},
                         budget_seconds={k:6000 for k in ("control", "training", *b.POST)},
                         limits={k:1024 for k in ("rss_bytes", "disk_bytes", "small_bytes", "free_reserve_bytes",
                                                  "free_gpu_bytes", "allocator_bytes", "process_gpu_bytes",
                                                  "worker_rss_bytes", "free_ram_bytes", "save_reserve_bytes")})
            def fixture_gate(authorization_sources):
                auth = root/"synthetic-authorization.json"
                b.data.write_json(auth, {**scope, "sources":authorization_sources, "authorize_execution":True})
                path = root/"synthetic-gate.json"
                b.data.write_json(path, {**scope, "approved":True, "review_closed":True, "workspace":str(scripts.parent),
                                         "campaign":str(root/"campaign"), "authorization":b.queue.record(auth)})
                return path
            with patch.object(b, "SCRIPTS", scripts), patch.object(b, "source_paths", return_value=[source]):
                self.assertEqual(b.read_gate(fixture_gate(files))["sources"], files)
                with self.assertRaisesRegex(ValueError, "differ from the decision"):
                    b.read_gate(fixture_gate([]))
                for key, name in (("seconds_per_method", b.H), ("post_seconds", "generation")):
                    scope[key][name] = 60
                    with self.assertRaisesRegex(ValueError, "budget"):
                        b.read_gate(fixture_gate(files))
                    scope[key][name] = 90
            self.assertFalse((root/"campaign").exists())
        EVIDENCE["formal_gate_conditions"] = dict(exact_synthetic_source="accepted",
                    different_authorized_source="blocked", training_at_stop_reserve="blocked",
                    post_at_stop_reserve="blocked", formal_campaign_created=False)

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
    elif sys.argv[1:] == ["--review-fixes"] and not torch.cuda.is_available():
        cases = ("test_generation_rechecks_162_endpoint_files", "test_interrupted_worker_validation_keeps_receipt_and_fee",
                 "test_gate_binds_authorized_sources_and_stop_reserve", "test_reservation_and_historical_gate_rejection")
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
