"""Affected shared-state lifecycle, checkpoint and endpoint checks on handwritten inputs."""
from __future__ import annotations

import ast
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

import step28_function_memory_run as run
import test_step28_function_memory as fixtures


class IntegrationTests(unittest.TestCase):
    def test_watchdog_stops_even_when_failure_record_cannot_be_written(self):
        # Execute the actual nested function without a live thread or process exit.
        path=Path(run.__file__).with_name("step28_function_memory_verify.py")
        tree=ast.parse(path.read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        watch=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=="watchdog")
        for write_fails in (False,True):
            fake_exit=mock.Mock(side_effect=SystemExit(2))
            env={"stopped":mock.Mock(wait=mock.Mock(return_value=False)),
                 "check":mock.Mock(side_effect=RuntimeError("handwritten budget stop")),
                 "write":mock.Mock(side_effect=OSError("handwritten disk failure") if write_fails else None),
                 "report":{},"os":mock.Mock(_exit=fake_exit)}
            exec(compile(ast.Module(body=[watch],type_ignores=[]),str(path),"exec"),env)
            with self.assertRaises(SystemExit) as error:
                env["watchdog"]()
            self.assertEqual(error.exception.code,2)
            fake_exit.assert_called_once_with(2)
            self.assertEqual(env["report"]["status"],"BUDGET_STOP")

    def test_formal_gate_distinguishes_cpu_and_native_evidence(self):
        # In-memory qualification fixture only: no gate file, execute or model load.
        source=run.sources(); p=run.policy()
        job=run.data.ROOT/"reports/handwritten_gate_never_executed"
        gate_path=run.data.ROOT/"handwritten_gate_not_written.json"
        cpu_path=run.data.ROOT/"handwritten_cpu.json"
        gpu_path=run.data.ROOT/"handwritten_gpu.json"
        cpu={"status":"PASS_HANDWRITTEN_ONLY","mode":"cpu","source_files":source}
        gpu={**cpu,"mode":"gpu","native":{"kind":"native_handwritten_first_optimizer_step"}}
        gate={"status":"APPROVED_FUNCTION_MEMORY_PILOT","source_files":source,
              "job":job.relative_to(run.data.ROOT).as_posix(),"runtime":p["runtime"],
              "supervision":p["supervision"],"review_disposition":"NO_OPEN_BLOCKERS",
              "integration_cpu":{"path":cpu_path.name},"native":{"path":gpu_path.name}}
        cases=[(cpu,gpu,True),(cpu,cpu,False),(gpu,gpu,False),
               (cpu,{**cpu,"mode":"gpu"},False),
               (cpu,{**gpu,"native":{"kind":"wrong"}},False)]
        for cpu_record,gpu_record,accepted in cases:
            records={run.POLICY:p,gate_path:gate,cpu_path:cpu_record,gpu_path:gpu_record}
            with mock.patch.object(run.data,"read_json",side_effect=lambda path: records[path]), \
                    mock.patch.object(run.data,"verify",side_effect=lambda path,record:path):
                if accepted:
                    self.assertEqual(run.validate_gate(job,gate_path),p)
                else:
                    with self.assertRaises(ValueError): run.validate_gate(job,gate_path)

    def test_shared_restore_legal_supply_actual_checkpoint_and_next_update(self):
        c=run.method.config()
        model=fixtures.tiny_model(); optimizer=run.method.make_optimizer(model,c)
        current=fixtures.handmade_group("seed_current"); old=fixtures.handmade_group("seed_history")
        x,memory=fixtures.fixture_memory(model,old,c)
        run.method.optimization_step(model,optimizer,current,old,x,memory,c,1e-5,current_seed=1,history_seed=2)
        # Synthetic shared-state fixture: slot counters are set explicitly, not claimed as 288 computed updates.
        for state in optimizer.state.values(): state["step"].fill_(288)
        selected={"fit":[],"calibration":[]}; partition={"fit":[],"calibration":[]}
        for role,count in (("fit",48),("calibration",12)):
            for domain in "ABC":
                for i in range(count):
                    g=fixtures.handmade_group(f"{role}_{domain}_{i:02}")
                    selected[role].append(g); partition[role].append({"group_uid":g.uid,"domain":domain})
        valid=[fixtures.handmade_group(f"valid_{d}_{i:02}") for d in "ABC" for i in range(20)]
        supply=run.parent.Supply(selected,partition)
        with self.assertRaises(ValueError): supply.current("ABC","ABC",2)
        current,cal=supply.current("ABC","ABC",1)
        with tempfile.TemporaryDirectory(prefix="shared_",dir=os.environ["FUNCTION_MEMORY_TEST_ROOT"]) as folder:
            root=Path(folder)
            for sub in ("models","memory","work","points","scores","maps"): (root/sub).mkdir()
            rng=run.previous.rng_state()
            state_digest=run.core.state_digest(model.state_dict())
            record=run.core.save_state(root/"shared.pt",model,optimizer,{"completed_updates":288,"rng":rng})
            shared={"full_checkpoint":{**record,"path":"shared.pt"},"model_state_sha256":state_digest}
            with mock.patch.object(run.base,"load_model",side_effect=lambda *a:fixtures.tiny_model()):
                restored,opt=run.previous.restore_branch(root,shared,c)
            self.assertEqual(run.previous.rng_state(),rng)
            self.assertEqual(run.core.state_digest(opt.state_dict()),run.core.state_digest(optimizer.state_dict()))
            save=run.core.save_state
            def disturb(path,model,optimizer,metadata):
                result=save(path,model,optimizer,metadata)
                with torch.no_grad():
                    next(model.parameters()).add_(.3)
                    if optimizer is not None:
                        next(iter(optimizer.state.values()))["exp_avg"].add_(.4)
                torch.rand(7)
                return result
            budget=type("Budget",(),{"check":lambda self,*a:None})()
            memory=run.method.Memory("ABC")
            with mock.patch.object(run.core,"save_state",side_effect=disturb):
                point,memory=run.checkpoint(root,run.point_name("ABC",1),restored,opt,memory,current,cal,valid,c,1,budget)
            self.assertEqual(run.previous.rng_state(),rng)
            self.assertEqual(run.core.state_digest(restored.state_dict()),state_digest)
            self.assertEqual(run.core.state_digest(opt.state_dict()),run.core.state_digest(optimizer.state_dict()))
            self.assertEqual(memory.count,48)
            self.assertLess(point["memory_bytes"],1048576)
            self.assertEqual(point["model_state_sha256"],state_digest)
            self.assertFalse(list((root/"work").iterdir()))
            # Full raw scores, maps and complete memory were actually written/restored by the production checkpoint.
            run.previous.array(root,point["scores"]["raw"],(60,378),np.float32)
            next_current,_=supply.current("ABC","ABC",2)
            with self.assertRaises(ValueError): supply.current("ABC","ABC",1)
            memory.begin_stage(2)
            sequence,_=run.parent.schedule(next_current,run.parent.contract(),"ABC",2)
            row=run.method.update(restored,opt,sequence[0],memory,c,2,1)
            self.assertEqual(row["adam_step"],289); self.assertEqual(memory.draw_count,1)
            self.assertAlmostEqual(row["total"],row["current_total"]+.1*row["history_total"]+.5*row["function"],places=12)

    def test_second_stage_new_is_direct_paired_field_and_five_fixed_criteria(self):
        domains=[d for d in "ABC" for _ in range(20)]
        arrays={}
        for i,order in enumerate(run.parent.ORDERS):
            for stage in (1,2,3):
                values=np.arange(60*22,dtype=float).reshape(60,22)/1000+i+stage
                arrays[run.evaluation.point_name(order,"er",stage)]={r:values.copy() for r in run.ROLES}
        a2=run.second_new_fields(arrays,domains,"primary")
        n=run.evaluation.endpoint_fields(arrays,domains,"er","primary","N")
        z=run.evaluation.endpoint_fields(arrays,domains,"er","primary","Z")
        np.testing.assert_allclose(a2,2*n-z,atol=1e-14,rtol=0)
        for i,order in enumerate(run.parent.ORDERS):
            d="ABC".index(order[1])
            np.testing.assert_array_equal(a2[i,d],arrays[run.evaluation.point_name(order,"er",2)]["raw"][20*d:20*(d+1)])
        result=run.comparisons(arrays,arrays,domains)
        self.assertEqual(len(result["continuation_checks"]),5)
        self.assertFalse(result["development_criteria_pass"])
        self.assertTrue(result["continuation_checks"]["A2_map_lower_ge_minus_point01"])

    def test_production_source_binding_and_no_recovery_entry(self):
        self.assertIs(run.train.__globals__["method"],run.method)
        self.assertIs(run.checkpoint.__globals__["method"],run.method)
        self.assertIs(run.method.update.__globals__["optimization_step"],run.method.optimization_step)
        tree=ast.parse(Path(run.__file__).read_text())
        train=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="train")
        calls=[ast.unparse(n.func) for n in ast.walk(train) if isinstance(n,ast.Call)]
        for name in ("previous.restore_branch","supply.current","method.update","checkpoint"):
            self.assertIn(name,calls)
        self.assertNotIn("method.load_model",calls)
        self.assertFalse(hasattr(run,"recover_statistics"))
        self.assertEqual(run.policy()["physical_updates"],1728)
        self.assertEqual(len(run.expected_points()),9)
        self.assertTrue(all((run.data.ROOT/r["path"]).is_file() for r in run.sources()))
