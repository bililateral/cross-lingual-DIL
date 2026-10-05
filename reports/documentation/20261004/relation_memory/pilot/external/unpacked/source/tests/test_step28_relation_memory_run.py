"""Handwritten integration, real three-stage updates/checkpoints/calibration/metrics."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
import step28_relation_memory_run as run
import test_step28_relation_memory as fixtures


class IntegrationTests(unittest.TestCase):
    def test_full_handwritten_dispatch_checkpoint_blind_gate_and_collection(self):
        labelled={}; groups={"train":[],"development":[]}; metadata=[]
        for split,count in (("train",60),("development",20)):
            for domain in "ABC":
                for i in range(count):
                    g=fixtures.handmade_group(f"{split}_{domain}_{i:02}")
                    labelled[g.uid]=g
                    groups[split].append(run.data.Group(g.uid,g.sellers,g.items,None))
                    metadata.append({"group_uid":g.uid,"domain":domain,"split":split,"group_index":str(i)})
        c=run.method.config(); p=run.policy()
        def load_labels(rows,config,split):
            return [labelled[g.uid] for g in rows]
        original_parse=run.previous.parse_once
        def parse(job,rows,config,split):
            return original_parse(job,rows,config,split,loader=load_labels)
        with tempfile.TemporaryDirectory(prefix="relation_handwritten_") as temp:
            job=Path(temp)
            run.data.write_json(job/"access.json",dict(train=0,valid=0,heldout=0,owners=0))
            budget=run.base.persistence.Budget(job,{"runtime":{"maximum_gpu_stage_seconds":1200,"maximum_output_bytes":128*2**20}})
            with mock.patch.object(run.base.public,"public_inputs",return_value=(groups,metadata,{"handwritten":True})), \
                 mock.patch.object(run.core,"model_files",return_value=c["models"]["split_rank"]), \
                 mock.patch.object(run.method,"load_model",side_effect=lambda config:fixtures.tiny_model()), \
                 mock.patch.object(run.previous,"parse_once",side_effect=parse):
                manifest,partition,valid=run.train(job,p,budget)
            self.assertEqual(manifest["physical_updates"],2592)
            self.assertEqual(run.data.read_json(job/"access.json")["valid"],0)
            bad=copy.deepcopy(manifest); bad["physical_updates"]-=1
            with self.assertRaisesRegex(ValueError,"Incomplete blind"):
                run.blind_gate(job/"run",bad)
            scores,checked=run.blind_gate(job/"run",manifest)
            self.assertEqual(checked,partition)
            truth=parse(job,valid,c,"development")
            with self.assertRaisesRegex(ValueError,"already been attempted"):
                parse(job,valid,c,"development")
            collection=run.collect(job/"evaluation",scores,truth,partition)
            self.assertEqual(sum(map(len,collection["points"].values())),28)
            self.assertEqual(run.data.read_json(job/"access.json"),dict(train=1,valid=1,heldout=0,owners=0))
            # Saved-matrix pipeline uses a hand-built comparator, no real baseline access.
            baseline=job/"baseline"; (baseline/"reference").mkdir(parents=True)
            cols=[{k:collection[k] for k in ("group_ids","domains","metric_columns")} for _ in range(2)]
            for col in cols: col["points"]={}
            for order in run.parent.ORDERS:
                for stage in (1,2,3):
                    index=1 if stage==1 else 0
                    name=order+"_shared" if stage==1 else f"{order}_logit_tenth_stage{stage}"
                    rec=collection["points"][run.point_name(order,stage)]
                    cols[index]["points"][name]=rec
                    dest=baseline/"reference" if index else baseline
                    for role in rec.values():
                        for file in role.values():
                            shutil.copyfile(job/"evaluation"/file["path"],dest/file["path"])
            refs=[]
            for folder,col in zip((baseline,baseline/"reference"),cols):
                run.data.write_json(folder/"collected.json",col)
                refs.append(run.data.record(folder/"collected.json",baseline))
            pp=copy.deepcopy(p); pp["reference"]={"linux_root":str(baseline),"collections":refs}
            with mock.patch.object(run,"policy",return_value=pp):
                result=run.finalize(job/"evaluation")
            self.assertFalse(result["worth_matched_replay"])
            self.assertEqual(result["delta"]["O"]["map"]["mean"],0.)
            self.assertEqual(len(result["absolute_stage_results"]),10)
            self.assertEqual(len(list((job/"run/models").glob("*.pt"))),9)
            self.assertFalse(list((job/"run/work").glob("*.pt")))

    def test_candidate_first_endpoint_and_observed_continuation(self):
        ref={}; cand={}; domains=[d for d in "ABC" for _ in range(20)]
        col=run.parent.metrics.COLUMNS.index("map")
        for order in run.parent.ORDERS:
            for stage in (1,2,3):
                name=run.evaluation.point_name(order,"er",stage)
                ref[name]={role:np.full((60,22),.5,np.float64) for role in run.ROLES}
                cand[name]=copy.deepcopy(ref[name])
                for role in run.ROLES:
                    if stage==1:
                        cand[name][role][np.asarray(domains)==order[0],col]=.9
                    if stage==3:
                        cand[name][role][np.asarray(domains)!=order[2],col]=.6
        r=run.comparisons(cand,ref,domains)
        self.assertTrue(r["worth_matched_replay"])
        self.assertAlmostEqual(r["delta"]["O"]["map"]["mean"],.1)
        self.assertAlmostEqual(r["endpoints"]["relation"]["raw"]["F_first"]["map"]["mean"],.3)
        cand["ABC_er_stage3"]["raw"][:,col]=.1
        r=run.comparisons(cand,ref,domains)
        self.assertFalse(r["worth_matched_replay"])


if __name__=="__main__":
    if os.environ.get("CUDA_VISIBLE_DEVICES")!="": raise RuntimeError("CPU only")
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    out=Path(sys.argv[1]); out.mkdir(parents=True,exist_ok=False)
    tick=time.monotonic()
    with (out/"unittest.txt").open("w",encoding="utf-8") as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationTests))
    report={"status":"PASS_HANDWRITTEN_ONLY" if result.wasSuccessful() else "FAIL",
            "tests_run":result.testsRun,"failures":len(result.failures),"errors":len(result.errors),"skipped":len(result.skipped),
            "elapsed_seconds":time.monotonic()-tick,"source_files":run.sources(),"formal_data":False,
            "full_three_order_small_model_updates":2592}
    run.data.write_json(out/"result.json",report)
    raise SystemExit(0 if result.wasSuccessful() else 1)
