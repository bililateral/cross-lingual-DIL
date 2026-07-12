#!/usr/bin/env python3
"""Build an explicit, hash-verified manifest for one Step15 run summary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument(
        "--extra-summary",
        action="append",
        default=[],
        help="Additional versioned summary whose referenced outputs must be frozen in the same manifest.",
    )
    parser.add_argument("--step9-summary")
    parser.add_argument("--step9-experiment", action="append", default=[])
    parser.add_argument("--step9-ratio-token", default="100pct")
    parser.add_argument("--step9-seed", action="append", type=int, default=[])
    parser.add_argument("--extra-file", action="append", default=[])
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--evaluation-role",
        default="fixed_internal_development_test_not_prospective_final_holdout",
    )
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def referenced_outputs(summary: dict) -> list[Path]:
    paths: set[Path] = set()
    for run in summary.get("runs", []):
        for value in (run.get("output_paths") or {}).values():
            if value:
                paths.add(resolve(str(value)))
    return sorted(paths, key=lambda path: str(path))


def referenced_step9_outputs(
    summary: dict,
    experiments: list[str],
    ratio_token: str,
    seeds: list[int],
) -> list[Path]:
    paths: set[Path] = set()
    for experiment_name in experiments:
        experiment = (summary.get("experiments") or {}).get(experiment_name)
        if not experiment:
            raise ValueError(f"Step15 active manifest cannot find Step9 experiment {experiment_name}")
        for seed in seeds:
            run_key = f"{ratio_token}_seed_{seed}"
            run = (experiment.get("runs") or {}).get(run_key)
            if not run:
                raise ValueError(
                    f"Step15 active manifest cannot find Step9 run {experiment_name}/{run_key}"
                )
            for value in (run.get("artifacts") or {}).values():
                if value:
                    paths.add(resolve(str(value)))
    return sorted(paths, key=lambda path: str(path))


def main() -> None:
    args = parse_args()
    summary_path = resolve(args.summary)
    extra_summary_paths = [resolve(value) for value in args.extra_summary]
    step9_summary_path = resolve(args.step9_summary) if args.step9_summary else None
    extra_file_paths = [resolve(value) for value in args.extra_file]
    policy_path = resolve(args.policy)
    output_json = resolve(args.output_json)
    output_csv = resolve(args.output_csv)
    if bool(step9_summary_path) != bool(args.step9_experiment and args.step9_seed):
        raise ValueError(
            "--step9-summary requires at least one --step9-experiment and --step9-seed, and vice versa"
        )
    for path in (
        summary_path,
        policy_path,
        *extra_summary_paths,
        *extra_file_paths,
        *([step9_summary_path] if step9_summary_path else []),
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    summary = load_json(summary_path)
    extra_summaries = [(path, load_json(path)) for path in extra_summary_paths]
    step9_summary = load_json(step9_summary_path) if step9_summary_path else None
    step9_output_paths = referenced_step9_outputs(
        step9_summary,
        [str(value) for value in args.step9_experiment],
        str(args.step9_ratio_token),
        [int(value) for value in args.step9_seed],
    ) if step9_summary is not None else []
    output_paths = sorted(
        {
            *referenced_outputs(summary),
            *(path for _, payload in extra_summaries for path in referenced_outputs(payload)),
            *step9_output_paths,
            *extra_file_paths,
        },
        key=lambda path: str(path),
    )
    missing = [path for path in output_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Step15 manifest refuses an incomplete run: {len(missing)} outputs missing; first={missing[0]}"
        )
    records = []
    for role, path in [
        ("summary", summary_path),
        *[("extra_summary", path) for path, _ in extra_summaries],
        *([("step9_summary", step9_summary_path)] if step9_summary_path else []),
        ("policy", policy_path),
        *[("run_output", path) for path in output_paths],
    ]:
        records.append(
            {
                "run_id": args.run_id,
                "role": role,
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest_core = {
        "run_id": args.run_id,
        "step": summary.get("step"),
        "policy_version": summary.get("policy_version"),
        "evaluation_role": args.evaluation_role,
        "manifest_generation_git_commit": git_commit(),
        "experiments": summary.get("experiments", []),
        "phases": summary.get("phases", []),
        "seeds": summary.get("seeds", []),
        "run_count": len(summary.get("runs", [])),
        "extra_summaries": [str(path.relative_to(ROOT)) for path, _ in extra_summaries],
        "extra_files": [str(path.relative_to(ROOT)) for path in extra_file_paths],
        "step9_selection": {
            "summary": str(step9_summary_path.relative_to(ROOT)) if step9_summary_path else None,
            "experiments": [str(value) for value in args.step9_experiment],
            "ratio_token": str(args.step9_ratio_token) if step9_summary_path else None,
            "seeds": [int(value) for value in args.step9_seed],
        },
        "files": records,
    }
    canonical = json.dumps(manifest_core, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    manifest_core["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if output_json.exists():
        previous = load_json(output_json)
        if previous.get("manifest_sha256") != manifest_core["manifest_sha256"]:
            raise ValueError(
                "Refusing to overwrite an existing active-run manifest with different content; "
                "use a new versioned output path"
            )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    temporary_json = output_json.with_name(f".{output_json.name}.tmp")
    with temporary_json.open("w", encoding="utf-8") as handle:
        json.dump(manifest_core, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_json.replace(output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary_csv = output_csv.with_name(f".{output_csv.name}.tmp")
    with temporary_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    temporary_csv.replace(output_csv)
    print(
        json.dumps(
            {
                "manifest": str(output_json.relative_to(ROOT)),
                "file_count": len(records),
                "manifest_sha256": manifest_core["manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
