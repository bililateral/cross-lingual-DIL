"""Independently check completed relation-pilot evidence, using saved arrays only.

Run with the existing Linux py310, one CPU and no GPU. Does not import training
code, read text/labels, load a model, or modify the original job.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np


ORDERS = ("ABC", "BCA", "CAB")
ROLES = ("raw", "stage-cal", "first-cal", "primary")
ENDPOINTS = ("O", "N", "Z", "F_first", "F", "G", "final_all")


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path, record: dict) -> Path:
    path = root / record["path"]
    payload = path.read_bytes()
    assert len(payload) == record["bytes"]
    assert hashlib.sha256(payload).hexdigest() == record["sha256"], str(path)
    return path


def seed(value: int, *parts: object) -> int:
    payload = (json.dumps([value, *parts], separators=(",", ":")) + "\n").encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63-1)


def main(job: Path, workspace: Path, output: Path) -> None:
    started = time.monotonic()
    assert not output.exists(), "New output required"
    completion = read(job / "completion.json")
    assert completion["status"] == "COMPLETE_RELATION_FIXED_POINT"
    assert not (job / "failure.json").exists()
    assert completion["access"] == read(job / "access.json") == dict(train=1, valid=1, heldout=0, owners=0)
    before = read(job / "before_valid.json")
    assert before["status"] == "PASS_COMPLETE_BLIND_GATE" and before["access"]["valid"] == 0
    verify(job, before["manifest"])
    saved = read(verify(job, completion["evaluation"]))
    collected = read(verify(job / "evaluation", saved["collected"]))
    assert collected["status"] == "ALL_28_RELATION_METRIC_COUNT_SETS_SAVED"
    assert sum(len(v) for v in collected["points"].values()) == 28
    assert completion["physical_updates"] == 2592
    budget = completion["budget"]
    for key, limit in (("elapsed_seconds",86400),("peak_observed_bytes",24*2**30),
                       ("peak_cuda_reserved_bytes",28*2**30),("peak_rss_bytes",64*2**30)):
        assert budget[key] <= limit
    execution = read(job / "execution.json")
    manifest = read(job / "run/manifest.json")
    assert execution["sources"] == manifest["source_files"] == collected["source_files"] == saved["source_files"]
    for record in execution["sources"]:
        verify(workspace, record)
    verify(workspace, execution["gate"])
    assert manifest["physical_updates"] == 2592 and manifest["gradient_group_presentations"] == 4320
    partition = read(verify(job / "run", manifest["partition"]))
    uid_domain = {r["group_uid"]:r["domain"] for r in partition["fit"]}
    points = {}
    for order in ORDERS:
        members, seen = [], 0
        retention_rng = random.Random(seed(20260930, order, "retention"))
        for stage in (1,2,3):
            name = f"{order}_relation_stage{stage}"
            point = read(verify(job / "run", manifest["points"][name]))
            tr = read(verify(job / "run", manifest["training"][name]))
            fit = sorted(uid for uid,d in uid_domain.items() if d == order[stage-1])
            rng = random.Random(seed(20260918,order,stage,"current"))
            schedule = []
            for _ in range(6):
                shuffled = fit.copy(); rng.shuffle(shuffled); schedule.extend(shuffled)
            assert tr["current_ids"] == schedule and len(tr["updates"]) == 288
            draws_rng = random.Random(seed(20260930,order,stage,"history_draws"))
            history = [members[draws_rng.randrange(6)] for _ in range(288)] if stage > 1 else [None]*288
            assert tr["history_ids"] == history
            assert point["full_restore_verified"] and tr["adam_step"] == point["adam_step"] == stage*288
            for i, update in enumerate(tr["updates"],1):
                assert update["step"] == i and update["adam_step"] == (stage-1)*288+i
                assert all(np.isfinite(v) for v in update.values())
                assert np.isclose(update["total"],update["current"]+update["history"],rtol=1e-12,atol=1e-12)
                lr = 1e-5*(i/29 if i <= 29 else (288-i)/259)
                assert np.isclose(update["encoder_lr"],lr,rtol=1e-12,atol=1e-16) and update["head_lr"] == .001
            if stage < 3:
                for uid in fit:
                    seen += 1
                    index = len(members) if len(members)<6 else retention_rng.randrange(seen)
                    if index < 6:
                        if index == len(members): members.append(uid)
                        else: members[index] = uid
            assert point["memory_summary"] == dict(members=members,count=seen,seen=seen,stage=min(stage,2))
            assert point["memory_bytes"] == point["memory"]["bytes"] <= 2**20
            verify(job / "run",point["memory"])
            mapping = read(verify(job / "run",point["mapping"]))
            assert mapping["status"] == "PASS_CALIBRATION_FIT" and mapping["a"] > 0
            assert point["calibration_ids"] == [r["group_uid"] for r in partition["calibration"] if r["domain"]==order[stage-1]]
            scores = {role:np.load(verify(job / "run",record),allow_pickle=False) for role,record in point["scores"].items()}
            for role,key in (("stage-cal","stage"),("first-cal","first")):
                expected = scores["raw"].astype(np.float64)*point["maps"][key]["a"]+point["maps"][key]["b"]
                np.testing.assert_array_equal(scores[role],expected)
            points[name] = {"memory_bytes":point["memory_bytes"],"members_by_domain":dict(Counter(uid_domain[u] for u in members)),
                "calibration":{k:mapping[k] for k in ("a","b","final_nll","raw_brier","calibrated_brier")},
                "mean_current_loss":float(np.mean([u["current"] for u in tr["updates"]])),
                "mean_history_loss":float(np.mean([u["history"] for u in tr["updates"]])),
                "min_gradient_norm":min(u["gradient_norm"] for u in tr["updates"]),
                "max_gradient_norm":max(u["gradient_norm"] for u in tr["updates"]),
                "retention":point["retention"]}
    baseline = workspace / saved["reference"]["linux_root"]
    basecols = [read(verify(baseline,r)) for r in saved["reference"]["collections"]]
    for col in basecols:
        for key in ("group_ids","domains","metric_columns"):
            assert col[key] == collected[key]
    columns = collected["metric_columns"]
    assert len(set(collected["group_ids"])) == 60 and len(columns) == 22
    rows = {d:np.flatnonzero(np.array(collected["domains"])==d) for d in "ABC"}
    assert all(len(r)==20 for r in rows.values())
    rank = [i for i,k in enumerate(columns) if k not in ("brier","log_loss","precision","recall","f1","specificity","balanced_accuracy","mcc")]
    losses = [columns.index(k) for k in ("brier","log_loss")]
    arrays, counts = {}, {}
    for arm in ("relation","logit0.1"):
        arrays[arm],counts[arm] = {},{}
        for order in ORDERS:
            for stage in (1,2,3):
                if arm == "relation": root,col,name = job / "evaluation",collected,f"{order}_relation_stage{stage}"
                elif stage == 1: root,col,name = baseline / "reference",basecols[1],order+"_shared"
                else: root,col,name = baseline,basecols[0],f"{order}_logit_tenth_stage{stage}"
                a,c = {},{}
                for role in ROLES[:3]:
                    rec = col["points"][name][role]
                    a[role] = np.load(verify(root,rec["matrix"]),allow_pickle=False)
                    assert a[role].shape == (60,22) and a[role].dtype == np.float64 and np.isfinite(a[role]).all()
                    c[role] = read(verify(root,rec["counts"]))
                    assert len(c[role]) == 60
                    for row in c[role]:
                        assert all(type(row[k]) is int and row[k]>=0 for k in ("tp","fp","fn","tn"))
                        assert row["tp"]+row["fn"] == 20 and row["fp"]+row["tn"] == 358
                for role in ROLES[1:3]: np.testing.assert_array_equal(a[role][:,rank],a["raw"][:,rank])
                a["primary"] = a["stage-cal"].copy(); a["primary"][:,rank] = a["raw"][:,rank]
                arrays[arm][order,stage],counts[arm][order,stage] = a,c
    # Each endpoint is formed explicitly for each shared actual-domain group.
    def field(arm: str, role: str, ep: str) -> np.ndarray:
        out = np.zeros((3,60,22))
        for oi,order in enumerate(ORDERS):
            def cell(stage: int, arrival: int) -> np.ndarray:
                result = np.zeros((60,22)); ids = rows[order[arrival-1]]
                result[ids] = arrays[arm][order,stage][role][ids]
                return result
            r11,r12,r22,r23,r31,r32,r33 = (cell(s,a) for s,a in ((1,1),(1,2),(2,2),(2,3),(3,1),(3,2),(3,3)))
            value = {"O":(r31+r32)/2,"N":(r22+r33)/2,"Z":r33,"F_first":r11-r31,
                     "F":(r11-r31+r22-r32)/2,"G":(r22-r12+r33-r23)/2,"final_all":(r31+r32+r33)/3}[ep]
            if ep in ("F_first","F","G"): value[:,losses] *= -1
            out[oi] = value
        return out
    draws = np.random.Generator(np.random.PCG64(20260930)).integers(0,20,(5000,3,20))
    def check_stats(f: np.ndarray, expected: dict) -> None:
        per_order = f.sum(1)/20
        means = per_order.mean(0)
        averaged = f.mean(0)
        boot = sum(averaged[rows[d]][draws[:,di]].mean(1) for di,d in enumerate("ABC"))
        ci = np.quantile(boot,[.025,.975],axis=0,method="linear")
        for k,name in enumerate(columns):
            np.testing.assert_allclose(means[k],expected[name]["mean"],rtol=0,atol=1e-12)
            np.testing.assert_allclose(ci[:,k],expected[name]["conditional_95pct_interval"],rtol=0,atol=1e-12)
            np.testing.assert_allclose(per_order[:,k],[expected[name]["per_order"][o] for o in ORDERS],rtol=0,atol=1e-12)
    for arm in arrays:
        for role in ROLES:
            for ep in ENDPOINTS: check_stats(field(arm,role,ep),saved["endpoints"][arm][role][ep])
    for ep in ENDPOINTS: check_stats(field("relation","primary",ep)-field("logit0.1","primary",ep),saved["delta"][ep])
    checks = {"O_map_positive":saved["delta"]["O"]["map"]["mean"]>0}
    checks.update({f"{e}_map_non_decrease":saved["delta"][e]["map"]["mean"]>=0 for e in ("N","Z")})
    checks.update({f"{e}_AP_non_decrease":saved["delta"][e]["average_precision"]["mean"]>=0 for e in ("O","N","Z")})
    assert checks == saved["continuation_checks"]
    assert all(checks.values()) == saved["observed_continuation_checks_pass"] == completion["worth_matched_replay"]
    classification, paths = {},{}
    for arm in arrays:
        classification[arm],paths[arm] = {},{}
        for ep,pairs in {"O":((3,1),(3,2)),"N":((2,2),(3,3)),"Z":((3,3),)}.items():
            total = dict.fromkeys(("tp","fp","fn","tn"),0)
            for order in ORDERS:
                for s,a in pairs:
                    for i in rows[order[a-1]]:
                        for k in total: total[k] += counts[arm][order,s]["stage-cal"][i][k]
            tp,fp,fn,tn = (total[k] for k in ("tp","fp","fn","tn"))
            classification[arm][ep] = {**total,"precision":tp/(tp+fp) if tp+fp else 0,"recall":tp/(tp+fn),"fpr":fp/(fp+tn),"f1":2*tp/(2*tp+fp+fn)}
        for order in ORDERS:
            paths[arm][order] = {str(stage):{d:{metric:float(arrays[arm][order,stage]["primary"][rows[d],columns.index(metric)].mean()) for metric in ("map","average_precision","brier","log_loss")} for d in "ABC"} for stage in (1,2,3)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps({"status":"PASS_SAVED_RESULT_INDEPENDENT_VERIFICATION","seconds":time.monotonic()-started,
        "scope":"No labels/text/model loading; independent saved-matrix endpoints, bootstrap and evidence checks",
        "source_files":len(execution["sources"]),"metric_count_sets":28,"verified_statistic_triplets":63*22,
        "continuation_checks":checks,"training_points":points,"fixed_half":classification,"stage_metrics":paths,
        "script_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"PASS","output":str(output),"seconds":time.monotonic()-started}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job",type=Path,required=True)
    parser.add_argument("--workspace",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    main(args.job,args.workspace,args.output)
