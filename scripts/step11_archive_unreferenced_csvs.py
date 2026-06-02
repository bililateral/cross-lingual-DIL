from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "reports"
DEFAULT_ARCHIVE_DIR = REPORTS / "step11_unreferenced_csv_archive_20260421"
DEFAULT_ARCHIVE_MANIFEST_CSV = REPORTS / "step11_unreferenced_csv_archive_manifest_20260421.csv"
DEFAULT_ARCHIVE_MANIFEST_JSON = REPORTS / "step11_unreferenced_csv_archive_manifest_20260421.json"
DEFAULT_CURRENT_MANIFEST_CSV = REPORTS / "step11_current_output_paths_manifest_20260421.csv"
DEFAULT_CURRENT_MANIFEST_JSON = REPORTS / "step11_current_output_paths_manifest_20260421.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Archive Step 11 CSV files that are not referenced by the current "
            "top-level *_clustering_summary.json output_paths sections."
        )
    )
    parser.add_argument("--reports-dir", type=Path, default=REPORTS)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--archive-manifest-csv", type=Path, default=DEFAULT_ARCHIVE_MANIFEST_CSV)
    parser.add_argument("--archive-manifest-json", type=Path, default=DEFAULT_ARCHIVE_MANIFEST_JSON)
    parser.add_argument("--current-manifest-csv", type=Path, default=DEFAULT_CURRENT_MANIFEST_CSV)
    parser.add_argument("--current-manifest-json", type=Path, default=DEFAULT_CURRENT_MANIFEST_JSON)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute manifests without moving files.",
    )
    return parser.parse_args()


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_archive_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(1, 10000):
        candidate = path.with_name(f"{stem}.duplicate_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise SystemExit(f"Could not find a unique archive target for {path}")


def resolve_report_path(path_value: str, reports_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def collect_current_output_paths(reports_dir: Path) -> tuple[list[Path], list[dict], set[str]]:
    summary_paths = sorted(reports_dir.glob("step11_*_clustering_summary.json"))
    current_rows: list[dict] = []
    referenced_csv_names: set[str] = set()

    for summary_path in summary_paths:
        summary = load_json(summary_path)
        output_paths = summary.get("output_paths", {}) or {}
        scorer = ((summary.get("selected_scorer", {}) or {}).get("scorer_token")) or summary_path.stem

        summary_output = output_paths.get("summary")
        if summary_output:
            current_rows.append(
                {
                    "summary_file": summary_path.name,
                    "scorer_token": scorer,
                    "output_type": "summary",
                    "threshold_token": "",
                    "output_path": rel(resolve_report_path(summary_output, reports_dir)),
                    "authority": "summary.output_paths",
                }
            )

        scored_pairs = output_paths.get("scored_pairs")
        if scored_pairs:
            scored_path = resolve_report_path(scored_pairs, reports_dir)
            referenced_csv_names.add(scored_path.name)
            current_rows.append(
                {
                    "summary_file": summary_path.name,
                    "scorer_token": scorer,
                    "output_type": "scored_pairs",
                    "threshold_token": "",
                    "output_path": rel(scored_path),
                    "authority": "summary.output_paths",
                }
            )

        clusters = output_paths.get("clusters_by_threshold", {}) or {}
        for threshold_token, cluster_path_value in sorted(clusters.items()):
            cluster_path = resolve_report_path(cluster_path_value, reports_dir)
            referenced_csv_names.add(cluster_path.name)
            current_rows.append(
                {
                    "summary_file": summary_path.name,
                    "scorer_token": scorer,
                    "output_type": "clusters_by_threshold",
                    "threshold_token": str(threshold_token),
                    "output_path": rel(cluster_path),
                    "authority": "summary.output_paths",
                }
            )

    return summary_paths, current_rows, referenced_csv_names


def collect_step11_csvs(reports_dir: Path) -> list[Path]:
    csv_paths = []
    for path in reports_dir.glob("step11_*_zh_target_strict_*.csv"):
        name = path.name
        if "_clusters.threshold_" in name or name.endswith("_scored_pairs.csv"):
            csv_paths.append(path)
    return sorted(csv_paths)


def main() -> None:
    args = parse_args()
    reports_dir = args.reports_dir.resolve()
    archive_dir = args.archive_dir.resolve()
    generated_at = datetime.now().isoformat(timespec="seconds")

    summary_paths, current_rows, referenced_csv_names = collect_current_output_paths(reports_dir)
    csv_paths = collect_step11_csvs(reports_dir)
    unreferenced_paths = [path for path in csv_paths if path.name not in referenced_csv_names]

    if not args.dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

    archive_rows: list[dict] = []
    for source_path in unreferenced_paths:
        destination_path = archive_dir / source_path.name
        if not args.dry_run:
            destination_path = unique_archive_path(destination_path)
        row = {
            "generated_at": generated_at,
            "file_name": source_path.name,
            "original_relative_path": rel(source_path),
            "archive_relative_path": rel(destination_path),
            "size_bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "referenced_by_current_summary_output_paths": "false",
            "reason": "not referenced by any current top-level Step 11 clustering summary output_paths entry",
            "action": "dry_run" if args.dry_run else "moved_to_archive",
        }
        archive_rows.append(row)
        if not args.dry_run:
            source_path.replace(destination_path)

    archive_fieldnames = [
        "generated_at",
        "file_name",
        "original_relative_path",
        "archive_relative_path",
        "size_bytes",
        "sha256",
        "referenced_by_current_summary_output_paths",
        "reason",
        "action",
    ]
    write_csv(args.archive_manifest_csv, archive_rows, archive_fieldnames)

    current_fieldnames = [
        "summary_file",
        "scorer_token",
        "output_type",
        "threshold_token",
        "output_path",
        "authority",
    ]
    write_csv(args.current_manifest_csv, current_rows, current_fieldnames)

    summary_payload = {
        "generated_at": generated_at,
        "dry_run": bool(args.dry_run),
        "authority_rule": "Treat each current Step 11 *_clustering_summary.json output_paths section as authoritative.",
        "summary_count": len(summary_paths),
        "current_output_path_manifest_csv": rel(args.current_manifest_csv),
        "current_output_path_count_including_summaries": len(current_rows),
        "current_referenced_csv_count": len(referenced_csv_names),
        "step11_csv_count_before_archive": len(csv_paths),
        "unreferenced_csv_count": len(unreferenced_paths),
        "archive_dir": rel(archive_dir),
        "archive_manifest_csv": rel(args.archive_manifest_csv),
        "archived_files": archive_rows,
    }
    write_json(args.archive_manifest_json, summary_payload)
    write_json(
        args.current_manifest_json,
        {
            "generated_at": generated_at,
            "authority_rule": "Treat each current Step 11 *_clustering_summary.json output_paths section as authoritative.",
            "summary_count": len(summary_paths),
            "output_path_count_including_summaries": len(current_rows),
            "referenced_csv_count": len(referenced_csv_names),
            "outputs": current_rows,
        },
    )

    print(
        json.dumps(
            {
                "summary_count": len(summary_paths),
                "current_referenced_csv_count": len(referenced_csv_names),
                "step11_csv_count_before_archive": len(csv_paths),
                "unreferenced_csv_count": len(unreferenced_paths),
                "archive_dir": rel(archive_dir),
                "dry_run": bool(args.dry_run),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
