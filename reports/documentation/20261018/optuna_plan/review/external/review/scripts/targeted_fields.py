"""Read saved fields and compare proposal display / historical budget figures only."""
from pathlib import Path
from decimal import Decimal
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"
OUT = ROOT / "mechanical"

def main():
    summary_rel = "reports/seller_alias_continual/20261018/positive_ranking_result/analysis/summary.json"
    summary = json.loads((INPUT/summary_rel).read_text(encoding="utf-8"))
    proposal = (INPUT/"proposal.zh.txt").read_text(encoding="utf-8")
    found = re.search(r"固定历史参考S0：A=([0-9.]+)，B=([0-9.]+)", proposal)
    displayed = dict(zip("AB",found.groups()))
    references = []
    for domain in "AB":
        value = summary["absolute"]["B_S"]["raw"][domain]["map"]
        references.append({"json_path": ["absolute","B_S","raw",domain,"map"], "saved_value": value,
            "proposal_display": displayed[domain], "display_matches_15_decimal_places": f"{value:.15f}"==displayed[domain]})
    policy = json.loads((INPUT/"schema/step28_bge_continual_policy.json").read_text(encoding="utf-8"))
    base = json.loads((INPUT/"schema/step28_chinese_base_policy.json").read_text(encoding="utf-8"))
    review_rel = "reports/seller_alias_continual/20261018/positive_ranking_result/review/external/S_pos_H_pos_external_review_20261018.zh.txt"
    old_review = (INPUT/review_rel).read_text(encoding="utf-8")
    match = re.search(r"正式资源投影独立重算为([0-9.]+)秒", old_review)
    prior = Decimal(match.group(1))
    result = {"s0_saved_summary_source": summary_rel,
        "s0_saved_summary_sha256": hashlib.sha256((INPUT/summary_rel).read_bytes()).hexdigest(),
        "s0_reference_display_check": references,
        "s0_reference_limit": "These are preserved independent-analysis fields, not a new reduction of the 52 baseline payloads.",
        "inherited_training_seed_roots": {k:policy[k] for k in ("initialization_seed","schedule_seed","memory_seed")},
        "inherited_partition_seed": base["partition"]["seed"],
        "historical_projection_source": review_rel,
        "historical_two_arm_projection_seconds": str(prior),
        "proposed_two_run_caps_seconds": 2*9000,
        "old_projection_exceeds_two_new_run_caps": prior>2*9000,
        "historical_projection_per_arm_mean_seconds": str(prior/2),
        "projection_limit": "Historical report field and arithmetic only; not current qualification and not evidence of inevitable runtime overrun.",
        "summary_sections": {key: list(summary[key]) if isinstance(summary[key],dict) else type(summary[key]).__name__ for key in ("paired_map_groups","diagnostic_summary","overfit_interpretations","success")}}
    (OUT/"targeted_fields.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
