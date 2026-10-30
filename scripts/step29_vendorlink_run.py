"""One sequential trajectory, actual method updates and full-state recovery.

No campaign dispatch or formal authorization is fabricated here. The CLI needs
an explicit, source-bound operation permit; handwritten qualification calls the
same execute() path with tiny randomly initialized models and fixture loaders.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from step29_vendorlink_data import Group, seed_for
from step29_vendorlink_methods import Memory, Model, Settings, make_optimizer, update_h, update_vendor


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state_all(state["cuda"])


def save_state(path, model, optimizer, memory, cursor, identity):
    value = {"identity": identity, "settings": asdict(model.settings), "model": model.state_dict(),
        "classes": model.classes, "optimizer": optimizer.state_dict(), "memory": memory.state(),
        "cursor": cursor, "rng": rng_state(), "modes": [m.training for m in model.modules()]}
    temporary = path.with_suffix(".partial")
    torch.save(value, temporary)
    temporary.replace(path)


def load_state(path, model, memory, identity):
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value["identity"] != identity or value["settings"] != asdict(model.settings):
        raise ValueError("Recovery must retain method/data/config identity")
    if model.settings.method == "VendorLink_SEQ":
        model.start_vendor_stage(value["classes"], 0)
    model.load_state_dict(value["model"])
    optimizer = make_optimizer(model)
    optimizer.load_state_dict(value["optimizer"])
    memory.restore(value["memory"])
    for module, mode in zip(model.modules(), value["modes"], strict=True):
        module.training = mode
    restore_rng(value["rng"])
    return optimizer, value["cursor"]


def set_rates(optimizer, settings, update, total):
    if not 1 <= update <= total:
        raise ValueError("Actual update outside stage exposure schedule")
    if settings.method == "VendorLink_SEQ":
        # Author's zero-warmup linear scheduler, explicitly reset per new market.
        optimizer.param_groups[0]["lr"] = settings.vendor_lr*(total-update+1)/total
    else:
        warmup = min(total, math.ceil(.1*total))
        factor = update/warmup if update <= warmup else (total-update)/(total-warmup)
        optimizer.param_groups[0]["lr"] = settings.encoder_lr*settings.lr_multiplier*factor
        optimizer.param_groups[1]["lr"] = settings.head_lr*settings.lr_multiplier


def execute(model, load_fit, job, *, order, seed, epochs, identity, check=lambda: None, resume=None,
            inference_endpoints=False):
    """Only requests the current domain. History comes exclusively from Memory.

    H retains its Adam state across domains, as in the existing mainline.
    VendorLink resets its optimizer with the new current-domain class head.
    Save each epoch and each stage; failed attempts retain their own directory.
    """
    if len(order) != 3 or len(set(order)) != 3 or epochs < 1 or job.exists():
        raise ValueError("Three distinct domains, positive exposure and fresh job required")
    if inference_endpoints and resume is not None:
        raise ValueError("Compact formal endpoints do not authorize a recovery")
    job.mkdir(parents=True)
    settings = model.settings
    memory = Memory(seed_for(seed, order, "reservoir"), settings)
    cursor = {"stage": 0, "next_epoch": 0, "stage_updates": 0, "updates": 0, "complete": False}
    optimizer = None
    if resume is not None:
        optimizer, cursor = load_state(resume, model, memory, identity)
    if cursor["complete"]:
        raise ValueError("Completed trajectory must not be resumed")
    # Exposure counts, not FLOPs: H encodes title and description separately.
    # Old recovery payloads without this field cannot establish earlier work.
    fields = ("current_groups", "current_ads", "history_groups", "history_ads", "teacher_groups", "teacher_ads")
    workload = cursor.setdefault("workload", {
        "complete_from_start": resume is None,
        "coverage": "completed_optimizer_updates_and_completed_teacher_calls",
        "totals": dict.fromkeys(fields, 0),
        "stages": [dict(stage=i+1, domain=domain, **dict.fromkeys(fields, 0))
                   for i, domain in enumerate(order)]})

    def ad_count(group):
        return sum(len(view.records) for view in group.views)

    def record_work(stage, kind, groups, ads):
        for suffix, count in (("groups", groups), ("ads", ads)):
            key = f"{kind}_{suffix}"
            workload["totals"][key] += count
            workload["stages"][stage][key] += count

    inherited = []
    try:
        # A fresh recovery directory must retain every already learned endpoint.
        # Copy existing states; never retrain earlier domains to fill this list.
        if resume is not None:
            for index in range(1, cursor["stage"]+1):
                check()
                source = Path(resume).resolve().parent/f"stage{index}.pt"
                target = job/source.name
                digest = file_sha256(source)  # Missing ancestors fail before further training.
                shutil.copyfile(source, target)
                if file_sha256(target) != digest:
                    raise ValueError("Inherited endpoint copy differs")
                inherited.append({"path": target.name, "source": str(source), "sha256": digest})
        for stage in range(cursor["stage"], 3):
            check()
            groups = load_fit(order[stage])
            if not groups or any(g.domain != order[stage] or g.split != "fit" for g in groups):
                raise ValueError("Current-stage fit input only")
            for group in groups:
                group.validate(training=True)
            classes = sorted({v.identity for g in groups for v in g.views})
            if cursor["next_epoch"] == 0:
                if settings.method == "VendorLink_SEQ":
                    model.start_vendor_stage(classes, seed_for(seed, order, stage, "head"))
                    optimizer = make_optimizer(model)
                elif optimizer is None:
                    optimizer = make_optimizer(model)
            elif settings.method == "VendorLink_SEQ" and tuple(classes) != model.classes:
                raise ValueError("Recovered classifier no longer matches current fit identities")
            if settings.method == "VendorLink_SEQ":
                count = sum(sum(len(v.records) for v in g.views) for g in groups)
                per_epoch = math.ceil(count/settings.vendor_batch)
            else:
                per_epoch = len(groups)
            total = epochs*per_epoch
            for epoch in range(cursor["next_epoch"], epochs):
                stream = random.Random(seed_for(seed, order, stage, epoch, "current"))
                if settings.method == "VendorLink_SEQ":
                    examples = [(r, v.identity) for g in groups for v in g.views for r in v.records]
                    stream.shuffle(examples)
                    batches = [examples[k:k+settings.vendor_batch] for k in range(0, len(examples), settings.vendor_batch)]
                else:
                    batches = list(groups)
                    stream.shuffle(batches)
                for batch in batches:
                    check()
                    update = cursor["stage_updates"]+1
                    step_seed = seed_for(seed, order, stage, update, "update")
                    set_rates(optimizer, settings, update, total)
                    if settings.method == "VendorLink_SEQ":
                        records, labels = zip(*batch)
                        log = update_vendor(model, optimizer, records, labels, step_seed, check)
                        work = dict.fromkeys(fields, 0)
                        work["current_ads"] = len(records)
                    else:
                        history, teacher = (None, None)
                        if stage and settings.method != "H_SEQ":
                            history, teacher = memory.draw(seed_for(seed, order, stage, update, "history"))
                        log = update_h(model, optimizer, batch, history, teacher, step_seed, check)
                        work = dict.fromkeys(fields, 0)
                        work.update(current_groups=1, current_ads=ad_count(batch))
                        if history is not None:
                            work.update(history_groups=1, history_ads=ad_count(history))
                    for kind in ("current", "history"):
                        record_work(stage, kind, work[f"{kind}_groups"], work[f"{kind}_ads"])
                    cursor["stage_updates"], cursor["updates"] = update, cursor["updates"]+1
                    with (job/"updates.jsonl").open("a", encoding="utf-8") as output:
                        output.write(json.dumps({"stage": stage+1, "epoch": epoch+1, "update": update,
                                                 "total_updates": cursor["updates"], "workload": work, **log})+"\n")
                cursor["next_epoch"] = epoch+1
                save_state(job/"latest.pt", model, optimizer, memory, cursor.copy(), identity)
                write_json(job/"workload.json", workload)
            if stage < 2 and settings.method in ("H_ER", "H_full"):
                def record_teacher(group):
                    record_work(stage, "teacher", 1, ad_count(group))
                    # Preserve every completed teacher even if later retention fails.
                    write_json(job/"workload.json", workload)
                memory.retain(groups, stage+1, model, check, record_teacher=record_teacher)
            # Each endpoint also includes the post-retention state required for continuation.
            cursor = {"stage": stage+1, "next_epoch": 0, "stage_updates": 0,
                      "updates": cursor["updates"], "complete": stage == 2, "workload": workload}
            if inference_endpoints:
                endpoint = job/f"stage{stage+1}.pt"
                torch.save(dict(identity=identity, settings=asdict(settings), classes=model.classes,
                                model=model.state_dict(), stage=stage+1, updates=cursor['updates']), endpoint)
            else:
                save_state(job/f"stage{stage+1}.pt", model, optimizer, memory, cursor.copy(), identity)
            save_state(job/"latest.pt", model, optimizer, memory, cursor.copy(), identity)
            write_json(job/"workload.json", workload)
            del groups
        write_json(job/"completion.json", {"status": "THREE_STAGES_TRAINED_NOT_EVALUATED",
            "identity": identity, "updates": cursor["updates"], "memory_bytes": len(memory.bytes()),
            "stage_endpoints": [f"stage{i}.pt" for i in (1, 2, 3)], "inherited_endpoints": inherited,
            "inference_endpoints": inference_endpoints, "workload": workload})
        return cursor
    except BaseException as error:
        write_json(job/"failure.json", {"status": "STOP_NO_AUTOMATIC_RETRY", "error": repr(error), "cursor": cursor})
        raise


def evaluate(model, groups, check=lambda: None):
    """Blind scoring only; no qrels/identity labels or model selection."""
    result = {}
    for group in groups:
        if any(v.identity is not None for v in group.views) or group.split == "fit":
            raise ValueError("Evaluation entry accepts blind evaluation inputs only")
        result[group.uid] = {"view_ids": [v.uid for v in group.views], "scores": model.score(group, check).tolist()}
    return result


def query_ap(scores, labels, view_ids):
    """Per-query AP with fixed opaque-ID ties; independent of input row order.

    Scores and private qrels must share the exported view_ids ordering.
    """
    n = len(view_ids)
    if n < 2 or len(set(view_ids)) != n or not all(isinstance(v, str) and v for v in view_ids):
        raise ValueError("Distinct opaque view IDs required")
    matrix, truth = np.zeros((n, n)), np.zeros((n, n), dtype=bool)
    indices = np.triu_indices(n, 1)
    if len(scores) != len(indices[0]) or len(labels) != len(scores):
        raise ValueError("Incomplete pair table")
    if not np.isfinite(scores).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Finite scores and binary aligned qrels required")
    matrix[indices], truth[indices] = scores, labels
    matrix += matrix.T
    truth |= truth.T
    result = []
    for q in range(n):
        candidates = [i for i in range(n) if i != q]
        ordered = sorted(candidates, key=lambda i: (-matrix[q, i], view_ids[i]))
        hits = truth[q, ordered]
        if not hits.any():
            raise ValueError("Query has no known positive")
        result.append(float(((hits.cumsum()/np.arange(1, n))*hits).sum()/hits.sum()))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    args = parser.parse_args()
    raw = args.config.read_bytes()
    config, permit = json.loads(raw), json.loads(args.authorization.read_text(encoding="utf-8"))
    digest = hashlib.sha256(raw).hexdigest()
    if (permit.get("operation") != "vendorlink_single_trajectory_training"
            or permit.get("approved") is not True or permit.get("config_sha256") != digest
            or permit.get("job") != str(args.job.resolve()) or permit.get("seconds", 0) <= 0
            or permit.get("native_and_external_review_closed") is not True):
        raise ValueError("Formal model/data/compute permission and closed prerequisites required")
    # No permit is produced by this implementation or by handwritten qualification.
    if args.job.exists():
        raise ValueError("Fresh job required; failed work is never silently overwritten")
    if config.get("resume"):
        raise ValueError("Formal retry needs a separately reviewed recovery contract; API recovery is qualified only")
    from transformers import BertModel, BertTokenizer
    started = time.monotonic()
    deadline = started+permit["seconds"]
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Single-trajectory authorized wall time exhausted")
    settings = Settings(**config["settings"])
    torch.manual_seed(seed_for(config["seed"], "initialization"))
    bert = BertModel.from_pretrained(config["model_path"], local_files_only=True)
    tokenizer = BertTokenizer.from_pretrained(config["model_path"], local_files_only=True, do_lower_case=True)
    model = Model(bert, tokenizer, settings).to(config["device"])
    def load_fit(domain):
        source = config["fit_sources"][domain]
        content = Path(source["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != source["sha256"]:
            raise ValueError("Current fit source differs")
        return [Group.from_dict(g) for g in json.loads(content)]
    identity = {"config_sha256": digest, "task": "observed_account_views_not_verified_aliases"}
    try:
        execute(model, load_fit, args.job, order=config["order"], seed=config["seed"], epochs=config["epochs"],
                identity=identity, check=check)
    finally:
        if args.job.exists():
            write_json(args.job/"usage.json", {"wall_seconds": time.monotonic()-started,
                "budget_seconds": permit["seconds"], "campaign_budget": "not_implemented_or_authorized"})


if __name__ == "__main__":
    main()
