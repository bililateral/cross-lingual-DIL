#!/usr/bin/env python3
"""Train-only character TF-IDF on the existing public Chinese style stream.

Exploratory development ranking diagnostic, not a transfer experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import scipy
import sklearn
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

import step28_saved_model_audit as audit
import step28_style_transfer_common as style

ROOT = audit.ROOT
OUTPUT = ROOT / "reports/seller_alias_character_baseline/20260906"
PRIOR = "reports/saved_model_audit/20260906/diagnostic.json"
PINS = {
    "scripts/step28_style_transfer_common.py": "b3cfa6de654e4cc2cab7877e666f8f8582f7d4ed5143ae4357a8359d3f33e59b",
    "scripts/step7_build_synthetic_english_source.py": "2b27c900a387a92f9b0e09ce8dbab00a33505763813d9e3c4c475db88fd26241",
    "scripts/step7_build_english_source_dataset.py": "b337277d24338e77997448320d0d7eb458eeffdffd318008e9383853c9ea2151",
    "scripts/step28_saved_model_audit.py": "08a8c4fc6ebfd16ea247cf0f71b2c27982be53045818aa280b727530635c24c6",
    "schema/step28_style_transfer_policy.json": "17d4389fe903a29cb22563510ca96b03a03655c66a66347535318c05c10b621b",
    "reports/saved_model_audit/20260906/diagnostic.json": "42774c28500632b21c667b839f5c5652b8523cb3a6120e2379e5cbaf357045f7",
    "reports/step28_synthetic_chinese_dataset/v9_4_1_formal_500x4_attempt1_20260829/root_manifest.json": "9190ecdead719ace3bdf74f635c1c23e0a0d19482cb4a9d4d2f6fc1f31dbbbe3"
}
VECTORIZER = {
    "analyzer": "char", "ngram_range": (3, 5), "min_df": 2,
    "max_features": 30000, "sublinear_tf": True, "norm": "l2",
    "use_idf": True, "smooth_idf": True, "lowercase": True,
}


def vocabulary_digest(vectorizer: TfidfVectorizer) -> str:
    # scikit-learn can use NumPy integers for vocabulary indices.
    entries = [(term, int(index)) for term, index in sorted(vectorizer.vocabulary_.items())]
    encoded = json.dumps(entries, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fit_vectors(train: list[str], development: list[str]) -> tuple[Any, sparse.csr_matrix]:
    if not train or not development or any(not isinstance(x, str) or not x for x in train + development):
        raise ValueError("Expected nonempty public account streams")
    vectorizer = TfidfVectorizer(**VECTORIZER, dtype=np.float64)
    vectorizer.fit(train)
    matrix = vectorizer.transform(development).tocsr()
    if not np.isfinite(matrix.data).all() or not np.isfinite(vectorizer.idf_).all():
        raise ValueError("Nonfinite TF-IDF")
    return vectorizer, matrix


def score_pairs(matrix: sparse.csr_matrix, left: np.ndarray, right: np.ndarray,
                batch_size: int = 512) -> np.ndarray:
    """Dot products of L2-normalized TF-IDF rows; a zero row scores zero."""
    left, right = np.asarray(left), np.asarray(right)
    if (left.ndim != 1 or right.shape != left.shape or left.dtype.kind not in "iu"
            or right.dtype.kind not in "iu" or batch_size <= 0):
        raise ValueError("Invalid pair index shape, type or batch size")
    if len(left) and (left.min() < 0 or right.min() < 0
                     or left.max() >= matrix.shape[0] or right.max() >= matrix.shape[0]):
        raise ValueError("Account index outside matrix")
    scores = np.empty(len(left), dtype="<f8")
    for start in range(0, len(left), batch_size):
        stop = start + batch_size
        scores[start:stop] = np.asarray(
            matrix[left[start:stop]].multiply(matrix[right[start:stop]]).sum(axis=1)
        ).ravel()
    if not np.isfinite(scores).all() or np.any((scores < -1e-12) | (scores > 1 + 1e-12)):
        raise ValueError("Invalid normalized character cosine")
    return scores


def run() -> tuple[dict[str, Any], np.ndarray]:
    records: list[dict[str, Any]] = []
    for path, digest in PINS.items():
        audit.checked_file(ROOT, {"path": path, "sha256": digest}, path, records)
    prior = audit.read_json(ROOT / PRIOR)
    registered = {row["path"]: row for row in prior["inputs"]}

    def verified(relative: str) -> Path:
        return audit.checked_file(ROOT, registered[relative], relative, records)

    policy = audit.read_json(ROOT / "schema/step28_style_transfer_policy.json")
    corpus_root = ROOT / policy["roots"]["chinese_public"]
    manifest = audit.read_json(corpus_root / "root_manifest.json")
    public_records = {row["path"]: row for row in manifest["public_files"]}
    corpora = {}
    for split in ("train", "development"):
        relative = f"{split}/observed/redacted_items.jsonl"
        audit.checked_file(corpus_root, public_records[relative], relative, records)
        worlds = style.load_chinese_style_streams(policy, split)
        corpora[split] = {(world, uid): stream for world, accounts in worlds.items()
                          for uid, stream in accounts.items()}
    train_keys, dev_keys = sorted(corpora["train"]), sorted(corpora["development"])
    if {uid for _, uid in train_keys} & {uid for _, uid in dev_keys}:
        raise ValueError("Train/development account overlap")
    train_docs = [corpora["train"][key] for key in train_keys]
    dev_docs = [corpora["development"][key] for key in dev_keys]
    vectorizer, matrix = fit_vectors(train_docs, dev_docs)
    dev_index = {key: i for i, key in enumerate(dev_keys)}
    row_relative = (audit.PROJECTION / "base_v1/development/row_keys.csv").relative_to(ROOT).as_posix()
    identity_rows = (audit.PROJECTION / "identity_v1/development/row_keys.csv").relative_to(ROOT).as_posix()
    row_path = verified(row_relative)
    verified(identity_rows)
    if registered[row_relative]["sha256"] != registered[identity_rows]["sha256"]:
        raise ValueError("Base/identity row alignment differs")
    with row_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    left, right = [], []
    for row in rows:
        if row["split"] != "development":
            raise ValueError("Non-development scoring row")
        left.append(dev_index[(row["world_uid"], row["seller_uid_left"])])
        right.append(dev_index[(row["world_uid"], row["seller_uid_right"])])
    scores = score_pairs(matrix, np.asarray(left, dtype=np.int64), np.asarray(right, dtype=np.int64))

    # No supervised input is opened until vocabulary, IDF and all scores are fixed.
    execution = audit.read_json(verified("schema/step28_train_development_execution_policy.json"))
    spec = execution["authorized_private_inputs"]["development_labels"]
    if spec["path"] != "development/pair_labels.csv":
        raise ValueError("Unexpected supervision role")
    label_relative = (Path(execution["private_supervision_root"]) / spec["path"]).as_posix()
    labels, world_ids = audit.aligned_development_labels(row_path, verified(label_relative))
    features = audit.read_json(verified("schema/step28_model_experiment_policy.json"))["feature_contract"]
    identity_path = verified((audit.PROJECTION / "identity_v1/development/identity33.npy").relative_to(ROOT).as_posix())
    identity = np.load(identity_path, allow_pickle=False)
    if identity.shape != (len(scores), 33) or len(labels) != len(scores) or not np.isfinite(identity).all():
        raise ValueError("Identity/label/score dimensions or values differ")
    strong = (identity[:, features["identity33"].index("verified_direct_token_count_log1p")] > 0) | (
        identity[:, features["identity33"].index("strong_rotation_path_count_log1p")] > 0)
    groups = [np.flatnonzero(world_ids == world) for world in sorted(set(world_ids))]
    masks = {"all": np.ones(len(labels), dtype=bool), "no_strong": ~strong,
             "zero_identity": np.all(identity == 0, axis=1)}
    summaries = {name: audit.summarize_subset(labels, scores, groups, mask) for name, mask in masks.items()}
    for name, summary in summaries.items():
        old = prior["models"]["m3_base"][name]
        for key in ("rows", "positives", "negatives", "eligible_worlds", "total_worlds"):
            if summary[key] != old[key]:
                raise ValueError("Comparison population differs from prior audit")

    def digest_json(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()

    return {
        "status": "EXPLORATORY_CHARACTER_STYLE_BASELINE_COMPLETE",
        "claim": "Unsupervised train-only TF-IDF diagnostic on the existing Chinese style input; no new method or transfer claim.",
        "config": VECTORIZER, "config_origin": "Unchanged character 3-5 gram settings from the English V6 style positive control; no English data is read.",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__, "scikit_learn": sklearn.__version__},
        "script": audit.file_record(Path(__file__).resolve()), "inputs": records,
        "fit": {"train_accounts": len(train_docs), "development_accounts": len(dev_docs),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "vocabulary_sha256": vocabulary_digest(vectorizer),
                "idf_sha256": hashlib.sha256(np.asarray(vectorizer.idf_, dtype="<f8").tobytes()).hexdigest(),
                "train_streams_sha256": digest_json([[*key, corpora["train"][key]] for key in train_keys]),
                "development_streams_sha256": digest_json([[*key, corpora["development"][key]] for key in dev_keys]),
                "development_zero_vectors": int(np.sum(np.diff(matrix.indptr) == 0)),
                "development_nnz": int(matrix.nnz)},
        "character_cosine": summaries,
        "saved_comparators": {model: {name: prior["models"][model][name] for name in masks}
                              for model in ("m0", "m2", "m3_base", "m3_joint")},
        "truth_reads": {"development_labels": 1, "train_labels": 0, "qrels": 0, "audit_a": 0, "audit_b": 0},
        "limitations": [
            "Post hoc reused-development diagnosis; no confidence intervals, significance, confirmatory or real-Chinese effectiveness claim.",
            "Only the existing lexical-free style stream is used. No claim that it exhausts public text information or identifies human authors.",
            "TF-IDF sees all 500 training worlds without labels; this is not a paired low-label-budget training comparison.",
            "Default character analysis lowercases placeholders and collapses repeated whitespace; no improvement to the input representation is asserted.",
            "Saved M3 comparators use different richer feature representations; differences do not identify a causal contribution.",
            "Complete old negative transfer results remain closed; no threshold selection, neural optimization or Linux use.",
            "Vocabulary/IDF are deterministically reconstructible from pinned train inputs, code and environment; only scores and small receipt are retained.",
        ],
    }, scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"])
    parser.parse_args()
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite an existing baseline run")
    result, scores = run()
    OUTPUT.mkdir(parents=True)
    path = OUTPUT / "scores.npy"
    with path.open("xb") as handle:
        np.save(handle, scores, allow_pickle=False)
    if not np.array_equal(np.load(path, allow_pickle=False), scores):
        raise ValueError("Score disk replay mismatch")
    result["score_file"] = audit.file_record(path)
    with (OUTPUT / "diagnostic.json").open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "scores": len(scores),
                      "no_strong": result["character_cosine"]["no_strong"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
