"""Deterministic, fail-closed writers for publication-facing artifacts."""

from __future__ import annotations

import csv
import io
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def json_bytes(
    payload: Any,
    *,
    ensure_ascii: bool = False,
    indent: int | None = 2,
    trailing_newline: bool = True,
) -> bytes:
    text = json.dumps(payload, ensure_ascii=ensure_ascii, indent=indent)
    if trailing_newline:
        text += "\n"
    return text.encode("utf-8")


def csv_bytes(
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
    *,
    encoding: str = "utf-8-sig",
) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fieldnames))
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode(encoding)


def text_bytes(text: str, *, encoding: str = "utf-8") -> bytes:
    return text.encode(encoding)


def validate_immutable_target(path: Path, payload: bytes) -> str:
    """Allow an absent target or an exact replay; reject same-name content drift."""

    if not path.exists():
        return "new"
    if not path.is_file():
        raise FileExistsError(f"Immutable artifact target is not a file: {path}")
    if path.read_bytes() == payload:
        return "identical_replay"
    raise FileExistsError(
        "Refusing to overwrite a publication artifact with different content: "
        f"{path}. Use a new run-id or output path."
    )


def _atomic_write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # A hard-link commit is atomic and, unlike replace(), cannot clobber a
            # target created by another process between preflight and commit.
            os.link(temporary_path, path)
        except FileExistsError:
            validate_immutable_target(path, payload)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_immutable_bundle(items: Iterable[tuple[Path, bytes]]) -> dict[str, str]:
    """Preflight a bundle, then atomically create only its missing members."""

    materialized = [(Path(path), payload) for path, payload in items]
    paths = [path.resolve() for path, _ in materialized]
    if len(paths) != len(set(paths)):
        raise ValueError("Immutable artifact bundle contains duplicate target paths")

    statuses = {
        str(path): validate_immutable_target(path, payload)
        for path, payload in materialized
    }
    for path, payload in materialized:
        if statuses[str(path)] == "new":
            _atomic_write_new(path, payload)
    return statuses


def write_immutable_bytes(path: Path, payload: bytes) -> str:
    return write_immutable_bundle([(path, payload)])[str(path)]
