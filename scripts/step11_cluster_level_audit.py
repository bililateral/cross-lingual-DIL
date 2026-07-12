from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
REPORTS = ROOT / "reports"
AUDIT_VERSION = "step11_cluster_level_audit_v2_hash_closed"
PUBLICATION_MANIFEST_VERSION = "step11_explicit_summary_manifest_v2_hash_closed"
DEFAULT_OUTPUT_CSV = REPORTS / "step11_cluster_level_audit.current_manifest.csv"
DEFAULT_OUTPUT_SUMMARY = REPORTS / "step11_cluster_level_audit.current_manifest.json"
DEFAULT_STEP5_ZH_LABELS = REPORTS / "step5_zh_target_strict_frozen_silver_labels.csv"
PROOF_CONTACT_PREFIXES = (
    "telegram:",
    "wickr:",
    "jabber:",
    "qq:",
    "wechat:",
    "bat:",
    "wallet:",
    "phone:",
)
EMAIL_PROOF_POSITIVE_TERMS = (
    "seller-facing",
    "seller-specific",
    "seller contact",
    "direct shared-contact",
    "direct shared contact",
)
EMAIL_PROOF_EXCLUSION_TERMS = (
    "victim",
    "product",
    "sample",
    "customer",
    "leaked",
    "external url",
    "data content",
)
DECISION_ORDER = [
    "same_controller_high_confidence",
    "same_controller_core_with_possible_expansion",
    "partial_anchor",
    "template_clone_not_controller",
    "semantic_topic_not_controller",
    "uncertain",
]

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import step11_cluster_chinese_graph as step11  # noqa: E402
from immutable_artifact_io import csv_bytes, json_bytes, write_immutable_bundle  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a cluster-level audit over an explicit current Step 11 summary allow-list. "
            "The audit deduplicates exact seller sets across primary graph views and allows "
            "same-controller claims only for identifier/contact-anchored cores."
        )
    )
    parser.add_argument(
        "--summary",
        dest="summary_paths",
        action="append",
        type=Path,
        default=[],
        help=(
            "Explicit Step 11 clustering summary path. Repeat this option to lock the audit "
            "to a reviewed current summary set. Reports-dir globbing is intentionally disabled."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help=(
            "Hash-verified explicit Step11 summary manifest. This is the preferred v6 mode; "
            "do not combine it with --summary."
        ),
    )
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-summary", type=Path, default=DEFAULT_OUTPUT_SUMMARY)
    parser.add_argument(
        "--step5-label-csv",
        type=Path,
        default=DEFAULT_STEP5_ZH_LABELS,
        help=(
            "Frozen Step 5 zh_target_strict label CSV used only for strict proof-edge auditing. "
            "Step 11 predictions are still treated as candidate discovery surfaces."
        ),
    )
    parser.add_argument(
        "--publication-v6",
        action="store_true",
        help="Require a publication-v6 explicit manifest; --summary inputs are rejected.",
    )
    return parser.parse_args()


def validate_input_mode(args: argparse.Namespace) -> None:
    if args.publication_v6 and (args.summary_paths or not args.manifest):
        raise SystemExit("--publication-v6 accepts --manifest only; --summary is prohibited")
    if bool(args.manifest) == bool(args.summary_paths):
        raise SystemExit("Pass exactly one of --manifest or one-or-more --summary paths")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_paths_from_manifest(
    manifest_path: Path,
    *,
    require_publication_v6: bool = False,
    required_graph_mode: str | None = None,
) -> tuple[list[Path], dict]:
    manifest = load_json(manifest_path)
    expected_manifest_sha = str(manifest.get("manifest_sha256", ""))
    core = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    observed_manifest_sha = hashlib.sha256(
        json.dumps(core, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    if not expected_manifest_sha or observed_manifest_sha != expected_manifest_sha:
        raise ValueError(f"Step11 manifest self-hash mismatch: {manifest_path}")
    if manifest.get("selection_mode") != "explicit_allowlist_only":
        raise ValueError(f"Step11 manifest is not explicit allow-list mode: {manifest_path}")
    if require_publication_v6 and not bool(manifest.get("publication_v6", False)):
        raise ValueError(f"Step11 manifest is not publication-v6 mode: {manifest_path}")
    if require_publication_v6 and manifest.get("manifest_version") != PUBLICATION_MANIFEST_VERSION:
        raise ValueError(f"Step11 publication manifest format is stale: {manifest_path}")
    manifest_csv_value = str(manifest.get("manifest_csv_path", "") or "").strip()
    manifest_csv_sha = str(manifest.get("manifest_csv_sha256", "") or "").strip()
    if require_publication_v6 and (not manifest_csv_value or not manifest_csv_sha):
        raise ValueError(f"Step11 publication manifest does not bind its CSV: {manifest_path}")
    if manifest_csv_value:
        manifest_csv_path = resolve_input_path(Path(manifest_csv_value)).resolve()
        if not manifest_csv_path.exists() or sha256_file(manifest_csv_path) != manifest_csv_sha:
            raise ValueError(f"Step11 manifest CSV hash mismatch: {manifest_csv_path}")
        manifest_csv_rows = load_csv(manifest_csv_path)
        if int(manifest.get("manifest_csv_row_count", -1)) != len(manifest_csv_rows):
            raise ValueError(f"Step11 manifest CSV row count mismatch: {manifest_csv_path}")
    graph_mode = str(manifest.get("graph_validation_mode", "") or "")
    if required_graph_mode and graph_mode != required_graph_mode:
        raise ValueError(
            f"Step11 manifest graph mode mismatch: expected={required_graph_mode!r} actual={graph_mode!r}"
        )
    records = manifest.get("summaries") or []
    if int(manifest.get("summary_count", -1)) != len(records) or not records:
        raise ValueError(f"Step11 manifest summary count mismatch: {manifest_path}")
    paths: list[Path] = []
    scorer_tokens: list[str] = []
    for record in records:
        path = resolve_input_path(Path(str(record["summary_path"]))).resolve()
        if not path.exists() or sha256_file(path) != record.get("summary_sha256"):
            raise ValueError(f"Step11 manifest summary hash mismatch: {path}")
        runtime_policy_path = resolve_input_path(Path(str(record["runtime_policy_path"]))).resolve()
        if not runtime_policy_path.exists() or sha256_file(runtime_policy_path) != record.get(
            "runtime_policy_sha256"
        ):
            raise ValueError(f"Step11 manifest runtime-policy hash mismatch: {runtime_policy_path}")
        for output_record in json.loads(record.get("output_file_records_json", "[]")):
            output_path = resolve_input_path(Path(str(output_record["path"]))).resolve()
            if not output_path.exists() or sha256_file(output_path) != output_record.get("sha256"):
                raise ValueError(f"Step11 manifest output hash mismatch: {output_path}")
        if str(record.get("graph_validation_mode", "")) != graph_mode:
            raise ValueError(f"Step11 manifest record graph mode mismatch: {path}")
        summary = load_json(path)
        scorer_token = str((summary.get("selected_scorer", {}) or {}).get("scorer_token", ""))
        if not scorer_token or scorer_token != str(record.get("scorer_token", "")):
            raise ValueError(f"Step11 manifest scorer-token mismatch: {path}")
        scorer_tokens.append(scorer_token)
        paths.append(path)
    if len(paths) != len(set(paths)):
        raise ValueError(f"Step11 manifest contains duplicate summary paths: {manifest_path}")
    if len(scorer_tokens) != len(set(scorer_tokens)):
        raise ValueError(f"Step11 manifest contains duplicate scorer tokens: {manifest_path}")
    expected_tokens = [str(value) for value in manifest.get("expected_scorer_tokens", []) or []]
    if require_publication_v6 and (
        not expected_tokens or set(expected_tokens) != set(scorer_tokens)
    ):
        raise ValueError(f"Step11 publication manifest scorer roster is incomplete: {manifest_path}")
    return paths, manifest


def load_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def resolve_input_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ROOT / path


def truthy(value: object) -> bool:
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y"}


def numeric_gt_zero(value: object) -> bool:
    try:
        return float(str(value or "0").strip()) > 0.0
    except ValueError:
        return False


def round_float(value: object) -> float:
    return round(float(value or 0.0), 6)


def split_preview(value: object) -> list[str]:
    text = str(value or "").strip()
    if not text:
        return []
    parts: list[str] = []
    for chunk in text.replace("||", "|").replace(";", "|").split("|"):
        stripped = chunk.strip()
        if stripped:
            parts.append(stripped)
    return parts


def edge_has_contact(edge: dict) -> bool:
    return truthy(edge.get("has_shared_contact_exact")) or numeric_gt_zero(edge.get("shared_contact_count_capped"))


def edge_has_pgp(edge: dict) -> bool:
    return truthy(edge.get("has_shared_pgp_fingerprint")) or numeric_gt_zero(edge.get("shared_pgp_fingerprint_count_capped"))


def edge_has_identifier(edge: dict) -> bool:
    stratum = str(edge.get("review_stratum", "")).strip()
    return edge_has_contact(edge) or edge_has_pgp(edge) or stratum in {"identifier_plus_text", "identifier_primary"}


def edge_has_title_clone(edge: dict) -> bool:
    return truthy(edge.get("has_shared_title_clone")) or numeric_gt_zero(edge.get("shared_title_count_capped"))


def edge_has_description_clone(edge: dict) -> bool:
    return truthy(edge.get("has_shared_description_clone")) or numeric_gt_zero(edge.get("shared_description_count_capped"))


def load_step5_label_index(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(f"Step 5 label CSV not found for strict cluster audit: {path}")
    return {row["pair_uid"]: row for row in load_csv(path)}


def contact_values_text(label_row: dict) -> str:
    return str(label_row.get("shared_contact_values", "") or "").strip().lower()


def label_review_notes_text(label_row: dict) -> str:
    return str(label_row.get("review_notes", "") or "").strip().lower()


def label_has_proof_contact_type(label_row: dict) -> bool:
    values = contact_values_text(label_row)
    if any(prefix in values for prefix in PROOF_CONTACT_PREFIXES):
        return True
    if "email:" not in values:
        return False
    notes = label_review_notes_text(label_row)
    has_positive_context = any(term in notes for term in EMAIL_PROOF_POSITIVE_TERMS)
    has_exclusion_context = any(term in notes for term in EMAIL_PROOF_EXCLUSION_TERMS)
    return has_positive_context and not has_exclusion_context


def label_is_proof_positive(edge: dict, label_index: dict[str, dict]) -> bool:
    label_row = label_index.get(edge.get("pair_uid", ""))
    if not label_row:
        return False
    if str(label_row.get("review_label", "")).strip() != "positive":
        return False
    if not truthy(label_row.get("usable_for_core_transfer")):
        return False
    if numeric_gt_zero(label_row.get("shared_pgp_fingerprint_count")):
        return True
    return label_has_proof_contact_type(label_row)


def label_class(edge: dict, label_index: dict[str, dict]) -> str:
    label_row = label_index.get(edge.get("pair_uid", ""))
    if not label_row:
        return "missing"
    review_label = str(label_row.get("review_label", "") or "").strip() or "blank"
    if label_is_proof_positive(edge, label_index):
        return "proof_positive"
    if review_label == "positive":
        return "nonproof_positive"
    if review_label == "negative":
        return "negative"
    if review_label == "uncertain":
        return "uncertain"
    return review_label


def primary_threshold_token(summary: dict) -> tuple[float, str]:
    threshold = float((summary.get("selected_scorer", {}) or {}).get("graph_primary_threshold"))
    token = step11.threshold_token(threshold)
    return threshold, token


def find_primary_cluster_path(summary: dict, threshold_token: str) -> Path:
    clusters = ((summary.get("output_paths", {}) or {}).get("clusters_by_threshold", {}) or {})
    cluster_path_value = clusters.get(threshold_token)
    if not cluster_path_value:
        available = ", ".join(sorted(clusters))
        raise SystemExit(
            f"Primary threshold token {threshold_token} is not in {summary['output_paths']['summary']}. "
            f"Available tokens: {available}"
        )
    return resolve_path(cluster_path_value)


def read_primary_cluster_members(cluster_path: Path) -> dict[str, dict]:
    rows = load_csv(cluster_path)
    clusters: dict[str, dict] = {}
    for row in rows:
        cluster_id = row["cluster_id"]
        payload = clusters.setdefault(
            cluster_id,
            {
                "cluster_rows": [],
                "members": set(),
                "seller_raw_members": [],
                "top_categories": [],
                "contact_member_preview_count": 0,
            },
        )
        seller_uid = row["seller_uid"]
        payload["cluster_rows"].append(row)
        payload["members"].add(seller_uid)
        seller_raw = str(row.get("source_seller_raw", "")).strip()
        if seller_raw:
            payload["seller_raw_members"].append(seller_raw)
        payload["top_categories"].extend(split_preview(row.get("top_category_preview")))
        if str(row.get("contact_preview", "")).strip():
            payload["contact_member_preview_count"] += 1
    return clusters


def read_retained_edges(summary: dict, threshold: float, threshold_token: str, policy: dict) -> list[dict]:
    scored_path = resolve_path(summary["output_paths"]["scored_pairs"])
    scored_rows = load_csv(scored_path)
    edge_flag = f"edge_at_threshold_{threshold_token}"
    threshold_edges = []
    for row in scored_rows:
        if edge_flag in row:
            keep = truthy(row.get(edge_flag))
        else:
            keep = float(row.get("prob_positive", 0.0) or 0.0) >= threshold
        if keep:
            threshold_edges.append(row)
    pair_score_lookup = {row["pair_uid"]: float(row["prob_positive"]) for row in scored_rows}
    scorer_token = (summary.get("selected_scorer", {}) or {}).get("scorer_token", "")
    kept_edges, _diagnostics = step11.apply_graph_edge_filters(
        threshold_edges,
        pair_score_lookup,
        scorer_token,
        policy,
    )
    return kept_edges


def seller_set_id(members: set[str]) -> str:
    payload = "\n".join(sorted(members))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def summarize_edges(edges: list[dict], label_index: dict[str, dict]) -> dict:
    stratum_counts = Counter(str(edge.get("review_stratum", "") or "__blank__") for edge in edges)
    label_counts = Counter(label_class(edge, label_index) for edge in edges)
    contact_edges = sum(1 for edge in edges if edge_has_contact(edge))
    pgp_edges = sum(1 for edge in edges if edge_has_pgp(edge))
    identifier_edges = sum(1 for edge in edges if edge_has_identifier(edge))
    title_clone_edges = sum(1 for edge in edges if edge_has_title_clone(edge))
    description_clone_edges = sum(1 for edge in edges if edge_has_description_clone(edge))
    score_values = [float(edge.get("prob_positive", 0.0) or 0.0) for edge in edges]
    return {
        "edge_count": len(edges),
        "contact_edges": contact_edges,
        "pgp_edges": pgp_edges,
        "identifier_edges": identifier_edges,
        "proof_positive_edges": label_counts.get("proof_positive", 0),
        "nonproof_positive_edges": label_counts.get("nonproof_positive", 0),
        "negative_edges": label_counts.get("negative", 0),
        "uncertain_edges": label_counts.get("uncertain", 0),
        "missing_label_edges": label_counts.get("missing", 0),
        "label_counts": label_counts,
        "title_clone_edges": title_clone_edges,
        "description_clone_edges": description_clone_edges,
        "stratum_counts": stratum_counts,
        "score_mean": round_float(sum(score_values) / len(score_values)) if score_values else 0.0,
    }


def edge_stratum_counts_text(counter: Counter) -> str:
    return " | ".join(f"{key}:{counter[key]}" for key in sorted(counter))


def label_counts_text(counter: Counter) -> str:
    return " | ".join(f"{key}:{counter[key]}" for key in sorted(counter))


def choose_best_appearance(appearances: list[dict]) -> dict:
    family_priority = {"step15": 0, "step9": 1, "step9_calibration": 2, "step7": 3}
    return sorted(
        appearances,
        key=lambda item: (
            -item["proof_positive_edges"],
            -item["nonproof_positive_edges"],
            -item["identifier_edges"],
            -item["contact_edges"],
            -item["edge_count"],
            family_priority.get(item["family"], 99),
            item["cluster_rank"],
            item["scorer_token"],
        ),
    )[0]


def decide_cluster(aggregate: dict) -> tuple[str, str, str, str]:
    edge_count = int(aggregate["max_edge_count"])
    proof_positive_edges = int(aggregate["max_proof_positive_edges"])
    identifier_edges = int(aggregate["max_identifier_edges"])
    contact_edges = int(aggregate["max_contact_edges"])
    pgp_edges = int(aggregate["max_pgp_edges"])
    title_clone_edges = int(aggregate["max_title_clone_edges"])
    description_clone_edges = int(aggregate["max_description_clone_edges"])
    stratum_counts: Counter = aggregate["edge_stratum_counts"]

    if edge_count > 0 and proof_positive_edges == edge_count:
        return (
            "same_controller_high_confidence",
            "high",
            "can_support_same_controller_core_claim",
            "All retained edges in the strongest observed cluster are frozen proof-positive seller-facing identity edges.",
        )

    if proof_positive_edges >= 2:
        return (
            "same_controller_core_with_possible_expansion",
            "high",
            "use_only_identifier_core_for_claim; review_expansion_edges_separately",
            "Multiple frozen proof-positive seller-facing identity edges support a same-controller core; non-proof expansion must be kept separate.",
        )

    if proof_positive_edges == 1:
        return (
            "partial_anchor",
            "medium",
            "retain_anchor_pair/core; do_not_claim_full_cluster_without_pair_level_rereview",
            "One frozen proof-positive seller-facing identity edge exists, but the component also contains non-proof expansion.",
        )

    if identifier_edges > 0 or contact_edges + pgp_edges > 0:
        return (
            "uncertain",
            "low",
            "manual_review_before_claim",
            "Identifier-like features are present, but no retained edge is a frozen proof-positive seller-facing identity edge.",
        )

    clone_edges = max(title_clone_edges, description_clone_edges, stratum_counts.get("text_clone_primary", 0))
    semantic_edges = stratum_counts.get("semantic_structural", 0) + stratum_counts.get("semantic_only", 0)

    if edge_count > 0 and clone_edges / edge_count >= 0.5:
        return (
            "template_clone_not_controller",
            "low",
            "use_for_review_discovery_only",
            "The cluster is dominated by clone/template evidence without a seller-specific identity anchor.",
        )

    if edge_count > 0 and semantic_edges / edge_count >= 0.5:
        return (
            "semantic_topic_not_controller",
            "low",
            "use_for_review_discovery_only",
            "The cluster is dominated by semantic or topic evidence without a seller-specific identity anchor.",
        )

    return (
        "uncertain",
        "low",
        "manual_review_before_claim",
        "The cluster has mixed weak evidence and lacks a direct seller-specific identity anchor.",
    )


def validate_runtime_posthoc_label_binding(
    summary: dict,
    policy: dict,
    label_path: Path,
) -> None:
    gate = policy.get("step15_v6_validation_gate", {}) or {}
    if not gate:
        return
    configured = str((policy.get("input_paths", {}) or {}).get("step5_frozen_labels", ""))
    if not configured:
        raise ValueError("Step11-v6 runtime policy does not record its Step5 posthoc label path")
    configured_path = resolve_input_path(Path(configured)).resolve()
    if configured_path != label_path.resolve():
        raise ValueError(
            f"Cluster audit label path differs from Step11 runtime binding: {configured_path}"
        )
    expected_hash = str(
        (gate.get("frozen_input_file_sha256", {}) or {}).get(Path(configured).as_posix(), "")
    )
    observed_hash = sha256_file(label_path)
    if not expected_hash or observed_hash != expected_hash:
        raise ValueError("Cluster audit Step5 label hash differs from the frozen runtime binding")
    posthoc = summary.get("posthoc_frozen_label_audit", {}) or {}
    if str(posthoc.get("sha256", "")) != observed_hash:
        raise ValueError("Step11 summary posthoc label hash differs from cluster-audit labels")
    if bool(posthoc.get("used_by_model_features", True)) or bool(
        posthoc.get("used_by_graph_filter_decisions", True)
    ):
        raise ValueError("Step11 summary violates posthoc-only frozen-label use")


def audit_summary(
    summary_path: Path,
    label_index: dict[str, dict],
    label_path: Path,
) -> list[dict]:
    summary = load_json(summary_path)
    runtime_policy_value = str(summary.get("policy_path", "")).strip()
    if not runtime_policy_value:
        raise ValueError(f"Step11 summary does not record policy_path: {summary_path}")
    runtime_policy_path = resolve_input_path(Path(runtime_policy_value)).resolve()
    if not runtime_policy_path.exists():
        raise FileNotFoundError(
            f"Step11 summary runtime policy is missing: {runtime_policy_path}"
        )
    policy = load_json(runtime_policy_path)
    validate_runtime_posthoc_label_binding(summary, policy, label_path)
    threshold, threshold_token = primary_threshold_token(summary)
    cluster_path = find_primary_cluster_path(summary, threshold_token)
    clusters = read_primary_cluster_members(cluster_path)
    retained_edges = read_retained_edges(summary, threshold, threshold_token, policy)
    edges_by_cluster: dict[str, list[dict]] = {}
    scorer = summary.get("selected_scorer", {}) or {}

    for cluster_id, payload in clusters.items():
        members = payload["members"]
        edges_by_cluster[cluster_id] = [
            edge
            for edge in retained_edges
            if edge["seller_uid_left"] in members and edge["seller_uid_right"] in members
        ]

    rows = []
    for cluster_id, payload in clusters.items():
        members = payload["members"]
        first_row = payload["cluster_rows"][0]
        edge_summary = summarize_edges(edges_by_cluster.get(cluster_id, []), label_index)
        rows.append(
            {
                "seller_set_id": seller_set_id(members),
                "members": members,
                "seller_raw_members": payload["seller_raw_members"],
                "top_categories": payload["top_categories"],
                "contact_member_preview_count": payload["contact_member_preview_count"],
                "summary_path": rel(summary_path),
                "runtime_policy_path": rel(runtime_policy_path),
                "cluster_path": rel(cluster_path),
                "scorer_token": scorer.get("scorer_token", ""),
                "family": scorer.get("scorer_family", ""),
                "cluster_id": cluster_id,
                "cluster_rank": int(first_row.get("cluster_rank", 0) or 0),
                "graph_threshold": threshold,
                "cluster_score_mean": round_float(first_row.get("cluster_score_mean", 0.0)),
                **edge_summary,
            }
        )
    return rows


def aggregate_audit_rows(cluster_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in cluster_rows:
        grouped[(row["scorer_token"], row["seller_set_id"])].append(row)

    audit_rows = []
    for (scorer_token, seller_set_key), appearances in sorted(grouped.items()):
        best = choose_best_appearance(appearances)
        aggregate_strata = Counter()
        aggregate_labels = Counter()
        top_categories = Counter()
        seller_raw_members = sorted(set(best["seller_raw_members"]))
        for appearance in appearances:
            aggregate_strata.update(appearance["stratum_counts"])
            aggregate_labels.update(appearance["label_counts"])
            top_categories.update(appearance["top_categories"])

        aggregate = {
            "max_edge_count": max(item["edge_count"] for item in appearances),
            "max_contact_edges": max(item["contact_edges"] for item in appearances),
            "max_pgp_edges": max(item["pgp_edges"] for item in appearances),
            "max_identifier_edges": max(item["identifier_edges"] for item in appearances),
            "max_proof_positive_edges": max(item["proof_positive_edges"] for item in appearances),
            "max_nonproof_positive_edges": max(item["nonproof_positive_edges"] for item in appearances),
            "max_negative_edges": max(item["negative_edges"] for item in appearances),
            "max_uncertain_edges": max(item["uncertain_edges"] for item in appearances),
            "max_missing_label_edges": max(item["missing_label_edges"] for item in appearances),
            "max_title_clone_edges": max(item["title_clone_edges"] for item in appearances),
            "max_description_clone_edges": max(item["description_clone_edges"] for item in appearances),
            "edge_stratum_counts": aggregate_strata,
        }
        decision, confidence, recommended_action, rationale = decide_cluster(aggregate)
        appearance_text = " || ".join(
            f"{item['scorer_token']}:rank{item['cluster_rank']}:thr{step11.threshold_token(item['graph_threshold'])}"
            for item in sorted(appearances, key=lambda value: (value["scorer_token"], value["cluster_rank"]))
        )
        top_category_text = " | ".join(name for name, _count in top_categories.most_common(8))

        audit_rows.append(
            {
                "audit_version": AUDIT_VERSION,
                "comparison_scope": "per_scorer_no_cross_model_max",
                "audit_cluster_set_id": f"{scorer_token}::{seller_set_key}",
                "seller_set_id": seller_set_key,
                "scorer_token": scorer_token,
                "family": best["family"],
                "decision": decision,
                "confidence": confidence,
                "recommended_action": recommended_action,
                "rationale": rationale,
                "unique_seller_count": len(best["members"]),
                "scorer_appearance_count": len(appearances),
                "best_scorer_token": best["scorer_token"],
                "best_family": best["family"],
                "best_source_summary": best["summary_path"],
                "best_cluster_id": best["cluster_id"],
                "best_cluster_rank": best["cluster_rank"],
                "best_graph_threshold": best["graph_threshold"],
                "max_cluster_score_mean": max(item["cluster_score_mean"] for item in appearances),
                "max_edge_count": aggregate["max_edge_count"],
                "max_contact_edges": aggregate["max_contact_edges"],
                "max_pgp_edges": aggregate["max_pgp_edges"],
                "max_identifier_edges": aggregate["max_identifier_edges"],
                "max_proof_positive_edges": aggregate["max_proof_positive_edges"],
                "max_nonproof_positive_edges": aggregate["max_nonproof_positive_edges"],
                "max_negative_edges": aggregate["max_negative_edges"],
                "max_uncertain_edges": aggregate["max_uncertain_edges"],
                "max_missing_label_edges": aggregate["max_missing_label_edges"],
                "max_title_clone_edges": aggregate["max_title_clone_edges"],
                "max_description_clone_edges": aggregate["max_description_clone_edges"],
                "edge_stratum_counts": edge_stratum_counts_text(aggregate_strata),
                "edge_label_counts": label_counts_text(aggregate_labels),
                "top_categories": top_category_text,
                "contact_member_preview_count": max(item["contact_member_preview_count"] for item in appearances),
                "seller_raw_members": " || ".join(seller_raw_members),
                "appearances": appearance_text,
            }
        )

    return sorted(
        audit_rows,
        key=lambda row: (
            {decision: index for index, decision in enumerate(DECISION_ORDER)}.get(row["decision"], 99),
            -int(row["max_edge_count"]),
            row["audit_cluster_set_id"],
        ),
    )


def main() -> None:
    args = parse_args()
    validate_input_mode(args)
    label_path = resolve_input_path(args.step5_label_csv)
    label_index = load_step5_label_index(label_path)
    manifest_path = resolve_input_path(args.manifest).resolve() if args.manifest else None
    manifest = None
    if manifest_path is not None:
        summary_paths, manifest = summary_paths_from_manifest(
            manifest_path,
            require_publication_v6=args.publication_v6,
        )
    else:
        summary_paths = [resolve_input_path(path).resolve() for path in args.summary_paths]

    cluster_rows = []
    for summary_path in summary_paths:
        cluster_rows.extend(audit_summary(summary_path, label_index, label_path))

    audit_rows = aggregate_audit_rows(cluster_rows)
    fieldnames = [
        "audit_version",
        "comparison_scope",
        "audit_cluster_set_id",
        "seller_set_id",
        "scorer_token",
        "family",
        "decision",
        "confidence",
        "recommended_action",
        "rationale",
        "unique_seller_count",
        "scorer_appearance_count",
        "best_scorer_token",
        "best_family",
        "best_source_summary",
        "best_cluster_id",
        "best_cluster_rank",
        "best_graph_threshold",
        "max_cluster_score_mean",
        "max_edge_count",
        "max_contact_edges",
        "max_pgp_edges",
        "max_identifier_edges",
        "max_proof_positive_edges",
        "max_nonproof_positive_edges",
        "max_negative_edges",
        "max_uncertain_edges",
        "max_missing_label_edges",
        "max_title_clone_edges",
        "max_description_clone_edges",
        "edge_stratum_counts",
        "edge_label_counts",
        "top_categories",
        "contact_member_preview_count",
        "seller_raw_members",
        "appearances",
    ]
    decision_counts = Counter(row["decision"] for row in audit_rows)
    confidence_counts = Counter(row["confidence"] for row in audit_rows)
    per_scorer_decision_counts: dict[str, dict[str, int]] = {}
    per_scorer_cluster_counts: Counter = Counter()
    for row in audit_rows:
        token = str(row["scorer_token"])
        per_scorer_cluster_counts[token] += 1
        per_scorer_decision_counts.setdefault(token, {decision: 0 for decision in DECISION_ORDER})
        per_scorer_decision_counts[token][str(row["decision"])] += 1
    unique_seller_sets = {str(row["seller_set_id"]) for row in audit_rows}
    output_csv_path = resolve_input_path(args.output_csv)
    output_summary_path = resolve_input_path(args.output_summary)
    audit_csv_payload = csv_bytes(audit_rows, fieldnames, encoding="utf-8-sig")
    summary_payload = {
        "audit_version": AUDIT_VERSION,
        "generated_at": datetime.now().date().isoformat(),
        "input_summary_count": len(summary_paths),
        "input_summaries": [rel(path) for path in summary_paths],
        "strict_step5_label_csv": rel(label_path),
        "summary_selection_mode": "explicit_manifest" if manifest_path else "explicit_paths",
        "publication_v6": bool(args.publication_v6),
        "input_manifest": rel(manifest_path) if manifest_path else None,
        "input_manifest_sha256": manifest.get("manifest_sha256") if manifest else None,
        "input_manifest_file_sha256": sha256_file(manifest_path) if manifest_path else None,
        "graph_validation_mode": manifest.get("graph_validation_mode") if manifest else None,
        "primary_cluster_count_total": len(cluster_rows),
        "audited_per_scorer_cluster_count": len(audit_rows),
        "unique_cluster_set_count": len(unique_seller_sets),
        "decision_count_scope": "per_scorer_cluster_appearances_no_cross_model_max",
        "decision_counts": {decision: decision_counts.get(decision, 0) for decision in DECISION_ORDER},
        "per_scorer_cluster_counts": dict(sorted(per_scorer_cluster_counts.items())),
        "per_scorer_decision_counts": dict(sorted(per_scorer_decision_counts.items())),
        "confidence_counts": dict(sorted(confidence_counts.items())),
        "output_csv": rel(output_csv_path),
        "output_csv_sha256": hashlib.sha256(audit_csv_payload).hexdigest(),
        "output_csv_row_count": len(audit_rows),
        "method_notes": [
            "Only an explicit hash-verified --manifest or explicit --summary inputs are audited; reports/ globbing is disabled.",
            "Only each summary's primary graph threshold and output_paths-referenced files are read.",
            "Retained edges are reconstructed with scripts/step11_cluster_chinese_graph.py graph filters.",
            "Exact seller_uid sets are deduplicated across all current primary Step 11 graph views.",
            "Decisions are computed independently per scorer. Cross-model maximum appearances are never used to assign a decision or hide scorer differences.",
            "Identifier-like Step 11 features are not sufficient for a same-controller claim.",
            "Same-controller claims are allowed only for retained edges that join to Step 5 frozen positive usable_for_core_transfer labels and expose proof-level seller-facing contact/PGP evidence.",
            "External URL, product/victim-data email, parser-noise contact, negative, uncertain, and unlabeled edges remain review-discovery surfaces.",
        ],
    }
    canonical = json.dumps(
        summary_payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    summary_payload["audit_sha256"] = hashlib.sha256(canonical).hexdigest()
    write_immutable_bundle(
        [
            (output_csv_path, audit_csv_payload),
            (
                output_summary_path,
                json_bytes(summary_payload, ensure_ascii=False, indent=2),
            ),
        ]
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
