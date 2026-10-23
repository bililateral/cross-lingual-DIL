#!/usr/bin/env python3
"""只读取送审 ZIP 的文档/JSON并做身份核对及普通算术。

不导入项目模块，不训练/推理/重采样，不生成科研数据，不连接网络。
用法: python verify_mechanical.py INPUT.zip OUTPUT_DIRECTORY
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

EXPECTED_BYTES = 233582
EXPECTED_SHA256 = "17d1bfd3214d6b9f164274527925ef2d14774967edd0ae2d22947e620097f487"
P = "reports/documentation/20261023/continual_mainline/"
NATIVE = "reports/documentation/20261018/optuna_implementation/repair/native_completed/qualification/native/result.json"
SELECTED = "reports/seller_alias_continual/20261023/optuna_result/job/selected.json"
GIB = 2**30


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_zip", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    out = args.output_directory
    out.mkdir(parents=True, exist_ok=True)
    archive_bytes = args.input_zip.read_bytes()
    sha = hashlib.sha256(archive_bytes).hexdigest()
    assert len(archive_bytes) == EXPECTED_BYTES and sha == EXPECTED_SHA256
    with zipfile.ZipFile(args.input_zip) as z:
        names = z.namelist()
        assert len(names) == len(set(names)) == 35
        assert all(not PurePosixPath(n).is_absolute() and ".." not in PurePosixPath(n).parts for n in names)
        manifest = json.loads(z.read("input_manifest.json"))
        rows = []
        for e in manifest["files"]:
            payload = z.read(e["path"])
            actual_sha = hashlib.sha256(payload).hexdigest()
            rows.append({"path": e["path"], "expected_bytes": e["size_bytes"],
                         "actual_bytes": len(payload), "expected_sha256": e["sha256"],
                         "actual_sha256": actual_sha,
                         "matches": len(payload) == e["size_bytes"] and actual_sha == e["sha256"]})
        expected_names = {e["path"] for e in manifest["files"]} | {"input_manifest.json"}
        assert set(names) == expected_names and len(rows) == 34 and all(e["matches"] for e in rows)
        budget = json.loads(z.read(P + "budget.json"))
        native = json.loads(z.read(NATIVE))
        selected = json.loads(z.read(SELECTED))
    identity = {"input_name": args.input_zip.name, "bytes": len(archive_bytes), "sha256": sha,
                "zip_members": len(names), "manifest_payloads": len(rows), "all_payloads_match": True,
                "missing_members": [], "extra_members": [], "duplicate_members": [], "files": rows}
    write_json(out / "input_identity_check.json", identity)
    projections = {}
    for arm in ("S", "S_pos", "H_pos"):
        m = native["arms"][arm]["measurements"]
        parts = {"training_seconds_proxy": 288 * (m["t_A"] + 2*m["t_B"]),
                 "group_forward_seconds_proxy": 2808*m["v"],
                 "teacher_projection_seconds_proxy": 72*m["h"],
                 "load_seconds_proxy": m["L"],
                 "checkpoint_io_seconds_proxy": 3*(m["s_F"]+m["r_F"]+m["s_I"]+m["r_I"]),
                 "unmeasured_allowance_seconds": 2700}
        seconds = 1.25*sum(parts.values())
        expected = budget["old_native_projection"][arm]
        assert abs(seconds-expected) < 1e-9
        projections[arm] = {"source_path": NATIVE, "source_pointer": "/arms/"+arm+"/measurements",
                            "stored_measurements": m, "components": parts, "factor": 1.25,
                            "computed_seconds": seconds, "budget_seconds": expected,
                            "computed_hours": seconds/3600,
                            "status": "OLD_MEASUREMENT_MECHANICAL_PROJECTION_NOT_NEW_QUALIFICATION"}
    v = max(native["arms"][a]["measurements"]["v"] for a in native["arms"])
    counts = {"HPO_updates": 36*576, "full_updates": 54*864,
              "total_updates": 36*576+54*864,
              "HPO_gradient_groups": 12*576+24*864,
              "full_gradient_groups": 9*864+45*1440,
              "total_gradient_groups": 12*576+24*864+9*864+45*1440,
              "HPO_endpoints": 36*2, "full_endpoints": 54*3,
              "HPO_epoch_captures": 36*12, "full_epoch_captures": 54*18,
              "initial_scoring_states": 6*3,
              "confirmation_scoring_states": 54*3+6*3,
              "confirmation_full_matrix_group_forwards": (54*3+6*3)*120,
              "confirmation_core_5cell_group_forwards": 54*5*40}
    assert counts["total_updates"] == budget["optimizer_updates"]["total"]
    assert counts["total_gradient_groups"] == budget["gradient_group_presentations"]["total"]
    assert counts["confirmation_full_matrix_group_forwards"] == budget["endpoints"]["confirmation_group_forwards"]
    buckets = budget["budget_seconds"]
    assert sum(buckets.values()) == budget["total_seconds"] == 2820600
    original_nontraining_hours = (sum(buckets.values())-buckets["baseline_search"]-buckets["full_trajectories"])/3600
    designs = []
    definitions = [
        ("D1_single_seed_pair", "S_tuned,H_pos_tuned", 2, 1, 0, 0, 0),
        ("D2_pair", "S_tuned,H_pos_tuned", 2, 3, 0, 0, 0),
        ("D3_mainline_recommended", "SEQ_ref,S_tuned,H_pos_tuned", 3, 3, 1, 0, 0),
        ("D3L_strong_reference", "LOGIT0.1_tuned,S_tuned,H_pos_tuned", 3, 3, 0, 1, 0),
        ("D4_add_LOGIT_ref", "SEQ_ref,LOGIT0.1_ref,S_tuned,H_pos_tuned", 4, 3, 1, 0, 0),
        ("D4_add_LOGIT_tuned", "SEQ_ref,LOGIT0.1_tuned,S_tuned,H_pos_tuned", 4, 3, 1, 1, 0),
        ("ORIGINAL", "SEQ_tuned,ER0.1_tuned,LOGIT0.1_tuned,S_tuned,S_pos_tuned,H_pos_tuned", 6, 3, 1, 3, 1)
    ]
    for name, arms, a, seeds, seq_arms, tuned_baselines, tuned_seq_arms in definitions:
        orders = 3
        paths = a*orders*seeds
        trials = 12*tuned_baselines
        seq_trials = 12*tuned_seq_arms
        gradients = (seq_arms*orders*seeds*864 + (a-seq_arms)*orders*seeds*1440
                     + seq_trials*576+(trials-seq_trials)*864)
        hard_training = 10*paths+6*trials
        core_forwards = 200*paths
        extra_seq_G = 80*orders*seeds*seq_arms
        row = {"id": name, "arms": arms, "orders": orders, "new_seeds": seeds,
               "new_HPO_trials": trials, "full_paths": paths,
               "successful_updates": 864*paths+576*trials,
               "gradient_group_presentations": gradients,
               "stage_inference_weights": 3*paths,
               "stage_inference_weight_GiB_proxy": 3*paths*1310000000/GIB,
               "core_5cell_group_forwards": core_forwards,
               "extra_SEQ_G_forwards": extra_seq_G,
               "core_plus_SEQ_G_forwards": core_forwards+extra_seq_G,
               "all_methods_core_plus_R21_forwards": 240*paths,
               "all_methods_core_plus_G_forwards": 280*paths,
               "training_hard_hours_using_original_6h_10h_limits": hard_training,
               "illustrative_total_keep_original_all_27_5_nontraining_hours": hard_training+original_nontraining_hours,
               "core_forward_old_proxy_hours_only": core_forwards*v*1.25/3600,
               "core_plus_SEQ_G_old_proxy_hours_only": (core_forwards+extra_seq_G)*v*1.25/3600}
        designs.append(row)
    checks = {
        "scope": "文档与已开放JSON字段的机械算术；无项目代码执行，无科研训练/推理/重采样/数据生成。",
        "source_paths": {"proposal": P+"proposal.zh.txt", "budget": P+"budget.json", "native": NATIVE, "selected": SELECTED},
        "source_lines": {"original_budget_counts": [130,153], "endpoint_definitions": [103,117], "P3_formula": [146,150]},
        "budget_buckets_seconds": buckets, "total_seconds": sum(buckets.values()),
        "total_hours": sum(buckets.values())/3600, "original_nontraining_hours": original_nontraining_hours,
        "time_semantics": {
            "budget_hours": "累计任务外层elapsed及互斥职责桶的拟上限；不是CPU user/system时间、并行墙钟或GPU设备小时。",
            "overlap": "不同run重叠仍逐run累计；同run的launcher等待与child执行不重复计费；纯资源等待另列。",
            "old_220_300h": "原单卡串行未验证外推；不转换成并行工期承诺。",
            "parallel_wall_clock_estimate_hours": None,
            "current_server_resources_verified": False,
            "parallel_budget_reservation": "已结算U+活跃run预留之和+未覆盖的必要尾部预留≤桶上限；退出后以实际替换预留。",
            "gpu_device_hours": "各物理GPU实际占用区间另计；同卡多worker按该卡时间并集，不等于任务小时之和。",
            "same_gpu_condition": "适用已有或未来获批的最小吞吐/资源证据；未执行测量。"
        },
        "counts": counts,
        "P3_group_forward_decomposition": {"epoch_stage1": 6*(48+60), "epoch_stages2_3": 2*6*(48+6+60+6),
           "three_endpoints_three_score_passes": 3*3*(12+60), "birth_teacher_upper": 12,
           "initial_old_dev": 60, "sum": 6*(48+60)+2*6*(48+6+60+6)+3*3*(12+60)+12+60},
        "P3_teacher_projection_count": {"count": 2*6*6, "unit_note": "h是单次历史教师投影秒数，不是小时。"},
        "old_measurement_projections": projections,
        "old_maximum_v_seconds": v,
        "original_confirmation_forward_proxy_hours_only": 21600*v*1.25/3600,
        "original_180_inference_bytes_proxy": 180*1310000000,
        "original_180_inference_GiB_proxy": 180*1310000000/GIB,
        "original_256GiB_slack_before_other_files": 256-180*1310000000/GIB,
        "remove_18_initial_weights_GiB_proxy": 18*1310000000/GIB,
        "selected_parameter_fields": {a:{"trial":selected[a]["number"],"lambda_rank":selected[a]["spec"]["lambda_rank"],
             "lr_multiplier":selected[a]["spec"]["lr_multiplier"]} for a in ("S","S_pos","H_pos")},
        "generic_formulas": {"p": "a*q*s; a方法数,q顺序数,s固定seed数,b需新增12次HPO的基线数",
             "new_HPO_trials": "12*b", "successful_updates": "864*p+6912*b",
             "stage_weights": "3*p", "core_confirmation_forwards": "200*p",
             "training_hard_hours": "10*p+72*b",
             "interpretation": "保留原单run拟硬限作累计任务范围比较，不是并行墙钟、测得的新限额或实施批准。"},
        "designs": designs,
        "not_verified": ["新代码实现", "新CPU/GPU资格", "新实际耗时", "未来同时存活文件峰值", "服务器当前资源", "训练seed在包外历史中从未使用"],
        "budget_wording_ambiguity": {"proposal_lines": [139,150], "detail": "bootstrap同时出现于A桶和确认桶说明；需选定唯一归属，未证明实际重复收费。"}
    }
    write_json(out / "mechanical_checks.json", checks)
    with (out / "design_comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(designs[0]))
        writer.writeheader()
        writer.writerows(designs)
    print(json.dumps({"identity_ok": True, "members": 35, "payloads": 34,
                      "original_budget_hours": checks["total_hours"], "P3_seconds": {a:v["computed_seconds"] for a,v in projections.items()},
                      "outputs": ["input_identity_check.json", "mechanical_checks.json", "design_comparison.csv"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
