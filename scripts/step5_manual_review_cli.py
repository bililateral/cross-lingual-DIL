from __future__ import annotations

import argparse
import csv
import shutil
import sys
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
QUEUE_PATHS = {
    "en_content_train_pool": ROOT / "reports" / "step5_en_balanced_review_queue.csv",
    "zh_target_strict": ROOT / "reports" / "step5_zh_target_strict_balanced_review_queue.csv",
    "zh_target_aux": ROOT / "reports" / "step5_zh_target_aux_balanced_review_queue.csv",
}
LABEL_COMMANDS = {
    "p": "positive",
    "n": "negative",
    "u": "uncertain",
}
ACTION_HELP = "p=positive n=negative u=uncertain s=skip c=clear m=more q=quit"


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")


def load_csv(path: Path) -> tuple[list[dict], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        value.encode("utf-8", errors="replace").decode("utf-8")
                        if isinstance(value, str)
                        else value
                    )
                    for key, value in row.items()
                }
            )


def normalize_text(value: str) -> str:
    return str(value or "").strip()


def normalize_label(value: str) -> str:
    return normalize_text(value).lower()


def normalize_status(value: str) -> str:
    return normalize_text(value).lower()


def review_state(row: dict) -> str:
    label = normalize_label(row.get("review_label", ""))
    status = normalize_status(row.get("review_status", ""))
    if label:
        return f"{status or 'reviewed'}:{label}"
    return status or "pending"


def sort_key(row: dict) -> tuple[int, str]:
    raw_rank = normalize_text(row.get("balanced_review_rank", ""))
    rank = int(raw_rank) if raw_rank.isdigit() else 10**9
    return rank, row.get("pair_uid", "")


def terminal_width(default: int = 108) -> int:
    return shutil.get_terminal_size((default, 30)).columns


def wrap_block(text: str, indent: str = "  ", width: int | None = None) -> str:
    width = width or terminal_width()
    content = normalize_text(text)
    if not content:
        return f"{indent}<empty>"
    lines = content.splitlines() or [content]
    wrapped = []
    for line in lines:
        wrapped.extend(textwrap.wrap(line, width=max(40, width - len(indent))) or [""])
    return "\n".join(f"{indent}{line}" for line in wrapped)


def print_summary(pool_rows: dict[str, list[dict]]) -> None:
    print("Step 5 Manual Review Summary")
    print("=" * min(terminal_width(), 80))
    for pool, rows in pool_rows.items():
        status_counts = Counter()
        stratum_counts = Counter()
        priority_counts = Counter()
        for row in rows:
            status_counts[review_state(row)] += 1
            stratum_counts[row["review_stratum"]] += 1
            priority_counts[row["review_priority"]] += 1

        print(f"[{pool}] total={len(rows)} pending={status_counts.get('pending', 0)} reviewed={len(rows) - status_counts.get('pending', 0)}")
        print(f"  priorities: {dict(priority_counts)}")
        print(f"  strata: {dict(stratum_counts)}")
        print(f"  states: {dict(status_counts)}")
        if pool == "zh_target_strict":
            print_targeted_focus_summary(rows)
        print("")


def print_targeted_focus_summary(rows: list[dict]) -> None:
    target_strata = ("semantic_only", "semantic_structural")
    print("  targeted_expansion:")
    for stratum in target_strata:
        reviewed_label_counts = Counter()
        pending_priority_counts = Counter()
        pending_start_rank_by_priority: dict[str, str] = {}
        for row in sorted(rows, key=sort_key):
            if row.get("review_stratum") != stratum:
                continue
            label = normalize_label(row.get("review_label", ""))
            if label:
                reviewed_label_counts[label] += 1
                continue
            priority = normalize_text(row.get("review_priority", "")) or "unknown"
            pending_priority_counts[priority] += 1
            pending_start_rank_by_priority.setdefault(priority, row.get("balanced_review_rank", ""))
        reviewed_total = sum(reviewed_label_counts.values())
        pending_total = sum(pending_priority_counts.values())
        print(
            f"    {stratum}: reviewed={reviewed_total} "
            f"labels={dict(reviewed_label_counts)} pending={pending_total}"
        )
        if pending_priority_counts:
            print(
                f"      pending_by_priority={dict(pending_priority_counts)} "
                f"start_rank_by_priority={pending_start_rank_by_priority}"
            )


def build_backup(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = path.with_name(f"{path.stem}.bak.{timestamp}{path.suffix}")
    shutil.copy2(path, backup_path)
    return backup_path


def filter_rows(
    rows: list[dict],
    start_rank: int,
    include_reviewed: bool,
    priorities: set[str],
    strata: set[str],
    limit: int | None,
) -> list[dict]:
    selected = []
    for row in sorted(rows, key=sort_key):
        if int(row["balanced_review_rank"]) < start_rank:
            continue
        if not include_reviewed and normalize_label(row.get("review_label", "")):
            continue
        if priorities and row["review_priority"] not in priorities:
            continue
        if strata and row["review_stratum"] not in strata:
            continue
        selected.append(row)
        if limit is not None and len(selected) >= limit:
            break
    return selected


def print_row(row: dict, ordinal: int, total: int, show_full: bool = False) -> None:
    width = terminal_width()
    print("")
    print("=" * min(width, 100))
    print(
        f"[{ordinal}/{total}] rank={row['balanced_review_rank']} "
        f"state={review_state(row)} stratum={row['review_stratum']} priority={row['review_priority']}"
    )
    print(f"scope={row['candidate_scope']} same_market={row['same_market_raw']} pair_uid={row['pair_uid']}")
    print(f"left={row['source_market_raw_left']} | {row['source_seller_raw_left']}")
    print(f"right={row['source_market_raw_right']} | {row['source_seller_raw_right']}")
    print(
        "scores="
        f"rank:{row['candidate_rank_score']} "
        f"lexical:{row['lexical_similarity']} "
        f"structural:{row['structural_support_score']}"
    )
    print(f"rule_hits={row['candidate_rule_hits']}")

    key_fields = [
        ("shared_contact_values", row.get("shared_contact_values", "")),
        ("shared_title_values", row.get("shared_title_values", "")),
        ("shared_description_values", row.get("shared_description_values", "")),
        ("shared_category_values", row.get("shared_category_values", "")),
        ("shared_pgp_fingerprint_values", row.get("shared_pgp_fingerprint_values", "")),
        ("left_preview", row.get("left_preview", "")),
        ("right_preview", row.get("right_preview", "")),
    ]

    for label, value in key_fields:
        content = normalize_text(value)
        if not content:
            continue
        print(f"{label}:")
        if show_full:
            print(wrap_block(content, width=width))
        else:
            preview = textwrap.shorten(content.replace("\n", " "), width=max(80, width - 4), placeholder=" ...")
            print(wrap_block(preview, width=width))

    existing_notes = normalize_text(row.get("review_notes", ""))
    if existing_notes:
        print("review_notes:")
        print(wrap_block(existing_notes, width=width))

    existing_reviewer = normalize_text(row.get("reviewer_id", ""))
    if existing_reviewer:
        print(f"reviewer_id: {existing_reviewer}")

    print(f"actions: {ACTION_HELP}")


def prompt_reviewer_id(initial_value: str | None) -> str:
    reviewer_id = normalize_text(initial_value or "")
    while not reviewer_id:
        reviewer_id = normalize_text(input("reviewer_id: "))
    return reviewer_id


def prompt_notes(existing: str) -> str:
    hint = "review_notes (empty allowed, enter '.' to reuse existing): "
    raw = input(hint)
    if raw.strip() == ".":
        return existing
    return raw.strip()


def clear_review(row: dict) -> None:
    row["review_status"] = "pending"
    row["review_label"] = ""
    row["reviewer_id"] = ""
    row["review_notes"] = ""


def run_review_session(
    pool: str,
    queue_path: Path,
    rows: list[dict],
    fieldnames: list[str],
    selected_rows: list[dict],
    reviewer_id: str,
) -> None:
    if not selected_rows:
        print("没有符合条件的待审核记录。")
        return

    backup_path = build_backup(queue_path)
    print(f"已创建备份: {backup_path}")
    print(f"开始审核 pool={pool}，共 {len(selected_rows)} 条。")

    index = 0
    changed_count = 0
    while index < len(selected_rows):
        row = selected_rows[index]
        print_row(row, index + 1, len(selected_rows), show_full=False)
        command = normalize_text(input("action> ")).lower()

        if command in LABEL_COMMANDS:
            label = LABEL_COMMANDS[command]
            notes = prompt_notes(normalize_text(row.get("review_notes", "")))
            row["review_status"] = "reviewed"
            row["review_label"] = label
            row["reviewer_id"] = reviewer_id
            row["review_notes"] = notes
            write_csv(queue_path, rows, fieldnames)
            changed_count += 1
            print(f"已保存: rank={row['balanced_review_rank']} label={label}")
            index += 1
            continue

        if command == "s":
            index += 1
            continue

        if command == "c":
            clear_review(row)
            write_csv(queue_path, rows, fieldnames)
            changed_count += 1
            print(f"已清空: rank={row['balanced_review_rank']}")
            index += 1
            continue

        if command == "m":
            print_row(row, index + 1, len(selected_rows), show_full=True)
            continue

        if command == "q":
            print(f"结束审核，本次变更 {changed_count} 条。")
            return

        print("无效命令。")

    print(f"审核完成，本次变更 {changed_count} 条。")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 5 manual review CLI for balanced sockpuppet review queues."
    )
    parser.add_argument("--pool", choices=sorted(QUEUE_PATHS), help="要审核的队列池")
    parser.add_argument("--queue-path", type=Path, help="可选：直接审核指定 CSV 队列，而不是默认的 pool 队列")
    parser.add_argument("--queue-label", help="可选：配合 --queue-path 使用的人类可读标签")
    parser.add_argument("--reviewer-id", help="审核人 ID，会写入 reviewer_id 列")
    parser.add_argument("--start-rank", type=int, default=1, help="从指定 balanced_review_rank 开始")
    parser.add_argument("--priority", action="append", default=[], help="只看指定 priority，可重复")
    parser.add_argument("--stratum", action="append", default=[], help="只看指定 review_stratum，可重复")
    parser.add_argument("--limit", type=int, help="本次最多处理多少条")
    parser.add_argument("--include-reviewed", action="store_true", help="默认只看未打 label 的记录；加上后可重审已标记录")
    parser.add_argument("--summary", action="store_true", help="只显示摘要，不进入交互")
    args = parser.parse_args()

    if args.queue_path and args.pool:
        parser.error("--queue-path 和 --pool 不能同时使用")
    if not args.summary and not (args.pool or args.queue_path):
        parser.error("交互审核时必须提供 --pool 或 --queue-path")
    return args


def main() -> None:
    configure_stdout()
    args = parse_args()

    if args.queue_path:
        queue_path = args.queue_path if args.queue_path.is_absolute() else ROOT / args.queue_path
        rows, fieldnames = load_csv(queue_path)
        queue_label = args.queue_label or queue_path.stem
        pool_rows = {queue_label: rows}
    else:
        pool_rows = {}
        for pool, path in QUEUE_PATHS.items():
            rows, _fieldnames = load_csv(path)
            pool_rows[pool] = rows

    if args.summary:
        if args.queue_path:
            print_summary(pool_rows)
        elif args.pool:
            print_summary({args.pool: pool_rows[args.pool]})
        else:
            print_summary(pool_rows)
        return

    if args.queue_path:
        queue_path = args.queue_path if args.queue_path.is_absolute() else ROOT / args.queue_path
        rows, fieldnames = load_csv(queue_path)
        queue_label = args.queue_label or queue_path.stem
    else:
        queue_path = QUEUE_PATHS[args.pool]
        rows, fieldnames = load_csv(queue_path)
        queue_label = args.pool
    selected_rows = filter_rows(
        rows,
        start_rank=args.start_rank,
        include_reviewed=args.include_reviewed,
        priorities={item.strip() for item in args.priority if item.strip()},
        strata={item.strip() for item in args.stratum if item.strip()},
        limit=args.limit,
    )
    reviewer_id = prompt_reviewer_id(args.reviewer_id)
    run_review_session(queue_label, queue_path, rows, fieldnames, selected_rows, reviewer_id)


if __name__ == "__main__":
    main()
