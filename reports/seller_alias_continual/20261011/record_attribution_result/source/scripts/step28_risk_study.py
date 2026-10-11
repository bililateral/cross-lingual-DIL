"""Frozen single-point risk replay policy and incremental-evidence bindings."""
from __future__ import annotations

import copy

import step28_er_weight as paired

data = paired.data
POLICY = data.ROOT / "schema/step28_risk_policy.json"
POLICY_SHA256 = "5706629736dfe9c598f1d59cc61f04bb65e9fd83b4275e391dfc3d2ebf2e5548"


def contract():
    if data.sha256(POLICY) != POLICY_SHA256:
        raise ValueError("Risk policy changed")
    p = data.read_json(POLICY)
    if (p["arms"] != {"risk": .1} or p["loss"]["risk_retention"] != .5
            or p["physical_updates"] != 1728 or p["orders"] != list(paired.ORDERS)
            or p["parent_policy_sha256"] != paired.parent.POLICY_SHA256
            or p["supervision"]["test_access"] or p["supervision"]["owners_access"]):
        raise ValueError("Risk intervention differs")
    return p


def sources():
    paths = ["scripts/step28_er_weight.py", "scripts/step28_er_weight_run.py",
             "scripts/step28_er_weight_evaluate.py", "scripts/step28_risk_replay.py",
             "scripts/step28_risk_execution.py", "scripts/step28_risk_study.py",
             "scripts/step28_risk_execution_check.py", "tests/test_step28_risk_replay.py",
             "scripts/step28_risk_replay_check.py", "tests/test_step28_er_weight_contracts.py",
             "tests/test_step28_risk_execution.py", "schema/step28_risk_policy.json",
             "docs/SELLER_ALIAS_RISK_PILOT.zh.md",
             "reports/documentation/20261003/risk_pilot/decision.json"]
    return sorted(paired.prior.sources() + [data.record(data.ROOT / name, data.ROOT) for name in paths],
                  key=lambda r: r["path"])


def verify_audit(audit, path, current_sources):
    if (audit.get("status") != "PASS_RISK_REPLAY_HANDMADE_CPU" or audit.get("source_files") != current_sources
            or audit.get("contracts", {}).get("failed") != 0 or audit.get("contracts", {}).get("skipped") != 0
            or audit.get("formal_inputs") is not False or audit.get("formal_labels") is not False
            or not audit.get("native_risk_gradient_increment_verified")):
        raise ValueError("Risk native and incremental CPU evidence required")
    for record in audit["native"].values():
        native = data.read_json(data.verify(path.parent / record["path"], record))
        if not all(native["retention_gradient_decomposition"][name]["different_actual_parameter_update"]
                   for name in ("encoder_query", "head_hidden")):
            raise ValueError("Native risk update evidence is incomplete")


def resolve_comparator(p, output):
    """Bind the preselected running LOGIT.1 job after its complete result exists.

    No choice by effect: exact job, authorization, policy and start are preselected.
    This reads saved results only; missing completion leaves the complete new
    collection available for later finalize without any new label parse.
    """
    result = copy.deepcopy(p)
    spec = result.pop("pending_logit_reference")
    job = data.ROOT / spec["linux_job"]
    data.verify(job / "execution.json", spec["execution"])
    if data.read_json(job / "completion.json")["status"] != "COMPLETE_LOGIT_WEIGHT_DEVELOPMENT_COMPARISON":
        raise ValueError("Preselected LOGIT.1 has no complete result")
    records = {name: data.record(job / name, job) for name in
               ("execution.json", "run/manifest.json", "run/partition.json",
                "evaluation/collected.json", "evaluation/evaluation.json")}
    result["continuation_references"]["logit_tenth"] = {
        "linux_job": spec["linux_job"], "policy_sha256": spec["policy_sha256"],
        "memory_arm": "logit", "evidence_tag": "LOGIT_WEIGHT", "records": records}
    binding = {"preselected_job": spec, "completed_records": records,
               "selection_by_results": False}
    path = output / "logit_reference_binding.json"
    if path.exists() and data.read_json(path) != binding:
        raise ValueError("Comparator changed after binding")
    data.write_json(path, binding)
    return result
