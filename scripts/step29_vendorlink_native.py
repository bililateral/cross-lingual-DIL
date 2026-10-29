"""Finite native qualification of production operations, not a formal campaign."""
from dataclasses import asdict, replace
import argparse
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from step29_vendorlink_data import Ad, View, Group, seed_for
from step29_vendorlink_methods import METHODS, Memory, Model, Settings, make_optimizer, update_h, update_vendor
from step29_vendorlink_run import (file_sha256, write_json, save_state, load_state, set_rates, rng_state, evaluate)
from step29_vendorlink_native_control import consume_launch, write as atomic_write


def long_group(group):
    """Handwritten stress text with original fit structure; never an effect sample."""
    text = "blue item text sample "*300
    return replace(group, views=tuple(replace(v, records=tuple(
        Ad(r.uid, text, text) for r in v.records)) for v in group.views))


def blind_shape(domain="shape"):
    views = tuple(View(f"shape-{i}-{j}", tuple(Ad(f"shape-{i}-{j}-{k}",
        "blue item text sample "*300, "blue item text sample "*300) for k in range(2)))
        for i in range(11) for j in range(2))
    return Group("handwritten_44_ad_blind", domain, "dev", views).blind()


def assert_state_equal(a, b):
    """Predeclared GPU tolerance; identities, RNG and nonfloats remain exact."""
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=1e-5, atol=1e-6,
                                   equal_nan=False, check_device=False)
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise AssertionError("NumPy RNG differs")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise AssertionError("State keys differ")
        for k in a:
            assert_state_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError("State lengths differ")
        for x, y in zip(a, b):
            assert_state_equal(x, y)
    elif a != b:
        raise AssertionError("Non-tensor state differs")


def sync(model):
    if next(model.parameters()).is_cuda:
        torch.cuda.synchronize()


def peak(model):
    if not next(model.parameters()).is_cuda:
        return dict(allocated=0, reserved=0)
    return dict(allocated=torch.cuda.max_memory_allocated(), reserved=torch.cuda.max_memory_reserved())


def parameter_probe(model):
    # Complete selected tensors, not a single parameter element that might stay zero.
    return {"encoder": model.encoder.embeddings.word_embeddings.weight.detach().cpu().clone(),
            "head": (model.classifier.weight if model.settings.method == "VendorLink_SEQ"
                     else model.head[0].weight).detach().cpu().clone()}


def step(model, optimizer, group, memory, stage, number, total, seed, check, output, label):
    before = parameter_probe(model)
    set_rates(optimizer, model.settings, number, total)
    rates = [g["lr"] for g in optimizer.param_groups]
    sync(model)
    started = time.monotonic()
    if model.settings.method == "VendorLink_SEQ":
        records = [r for v in group.views for r in v.records]
        identities = [v.identity for v in group.views for _ in v.records]
        log = update_vendor(model, optimizer, records, identities, seed, check)
    else:
        history, teacher = (memory.draw(seed_for(seed, "history")) if stage > 1
                            and model.settings.method != "H_SEQ" else (None, None))
        log = update_h(model, optimizer, group, history, teacher, seed, check)
    sync(model)
    ended = time.monotonic()
    after = parameter_probe(model)
    changes = {k: float((after[k]-before[k]).abs().max()) for k in before}
    if changes["head"] <= 0 or (rates[0] > 0 and changes["encoder"] <= 0):
        raise AssertionError("Actual encoder/head update missing")
    row = dict(label=label, stage=stage, update=number, started=started, ended=ended,
               seconds=ended-started, rates=rates, changes=changes, peak=peak(model), losses=log)
    with output.open("a") as stream:
        stream.write(json.dumps(row)+"\n")
    return row


def gradient_kd(model, history, teacher, check):
    saved = model.settings
    gradients, logs = [], []
    try:
        for method in ("H_full", "H_ER"):
            model.settings = replace(saved, method=method)
            model.train()
            model.zero_grad(set_to_none=True)
            logs.append(model.backward_h(history, 813, teacher, True, check))
            # Compare all affected gradients; do not infer a difference from scalar KD.
            gradients.append({n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
                              if p.grad is not None})
        maxima = {}
        for prefix in ("encoder.", "head."):
            maxima[prefix] = max(float((gradients[0][n]-gradients[1][n]).abs().max())
                                for n in gradients[0] if n.startswith(prefix))
        if any(not np.isfinite(x) or x <= 0 for x in maxima.values()) or logs[0]["distillation"] <= 0:
            raise AssertionError("Natural teacher KD does not affect actual gradients")
        return dict(gradient_max_difference=maxima, full=logs[0], replay_only=logs[1],
                    optimizer_step_comparison=False)
    finally:
        model.settings = saved
        model.zero_grad(set_to_none=True)


def verify_restore(path, model, memory, identity, expected):
    optimizer, cursor = load_state(path, model, memory, identity)
    actual = dict(model=model.state_dict(), classes=model.classes, optimizer=optimizer.state_dict(),
                  memory=memory.state(), cursor=cursor, rng=rng_state(), modes=[m.training for m in model.modules()])
    for key in actual:
        assert_state_equal(actual[key], expected[key])
    if not optimizer.state:
        raise AssertionError("Restore did not contain nonempty Adam")
    return optimizer


def qualify_g1(factory, load_stage, job, check=lambda: None):
    job.mkdir(parents=True, exist_ok=False)
    evidence = {}
    for method in METHODS:
        check()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        atomic_write(job/"active.json", dict(method=method))
        model = factory(method)
        folder = job/method
        folder.mkdir()
        identity = dict(scope="vendorlink_native_only", method=method)
        memory = Memory(411, model.settings)
        optimizer = None
        updates, rows, retention = 0, [], []
        result = dict(classes=[], recovery=None, kd=None)
        previous_teacher = {}
        for stage, domain in enumerate(("AlphaBay", "Dream", "Agora"), 1):
            check()
            groups, classes = load_stage(domain, folder/"access.jsonl")
            result["classes"].append(len(classes))
            if method == "VendorLink_SEQ":
                model.start_vendor_stage(classes, seed_for(stage, "head"))
                optimizer = make_optimizer(model)
            elif optimizer is None:
                optimizer = make_optimizer(model)
            current = long_group(groups[0]) if stage == 3 else groups[0]
            total = 2 if method == "H_full" and stage == 2 else 1
            seed = seed_for(method, stage, 1)
            rows.append(step(model, optimizer, current, memory, stage, 1, total, seed, check,
                             folder/"updates.jsonl", "long_stress" if stage == 3 else "real_fit"))
            updates += 1
            if method == "VendorLink_SEQ" and stage == 2:
                # Reconstruct stage boundary from the genuine preceding endpoint.
                expected_path = folder/"expected.pt"
                save_state(expected_path, model, optimizer, memory, {"stage": 2}, identity)
                expected = torch.load(expected_path, map_location="cpu", weights_only=False)
                before = torch.load(folder/"stage1.pt", map_location="cpu", weights_only=False)
                optimizer = verify_restore(folder/"stage1.pt", model, memory, identity, before)
                model.start_vendor_stage(classes, seed_for(stage, "head"))
                optimizer = make_optimizer(model)
                rows.append(step(model, optimizer, current, memory, stage, 1, total, seed, check,
                                 folder/"updates.jsonl", "restored_stage_boundary"))
                updates += 1
                save_state(folder/"repeated.pt", model, optimizer, memory, {"stage": 2}, identity)
                assert_state_equal(torch.load(folder/"repeated.pt", map_location="cpu", weights_only=False), expected)
                result["recovery"] = "nonempty_state_roundtrip_and_stage_boundary_next_step"
                del before, expected
            if method == "H_full" and stage == 2:
                history, teacher = memory.draw(813)
                result["kd"] = gradient_kd(model, history, teacher, check)
                before_path = folder/"before_extra.pt"
                save_state(before_path, model, optimizer, memory, {"stage": 2, "next": 2}, identity)
                before = torch.load(before_path, map_location="cpu", weights_only=False)
                extra_seed = seed_for(method, stage, 2)
                rows.append(step(model, optimizer, current, memory, stage, 2, 2, extra_seed, check,
                                 folder/"updates.jsonl", "extra_for_history_drift"))
                updates += 1
                save_state(folder/"expected.pt", model, optimizer, memory, {"stage": 2, "next": 3}, identity)
                expected = torch.load(folder/"expected.pt", map_location="cpu", weights_only=False)
                optimizer = verify_restore(before_path, model, memory, identity, before)
                rows.append(step(model, optimizer, current, memory, stage, 2, 2, extra_seed, check,
                                 folder/"updates.jsonl", "restored_nonempty_adam_next_step"))
                updates += 1
                save_state(folder/"repeated.pt", model, optimizer, memory, {"stage": 2, "next": 3}, identity)
                assert_state_equal(torch.load(folder/"repeated.pt", map_location="cpu", weights_only=False), expected)
                result["recovery"] = "nonempty_adam_memory_rng_next_step"
                del before, expected
            if stage < 3 and method in ("H_ER", "H_full"):
                memory.retain(groups, stage, model, check)
                for row in memory.entries:
                    uid = row["group"]["uid"]
                    if uid in previous_teacher:
                        torch.testing.assert_close(row["teacher"], previous_teacher[uid], atol=0, rtol=0)
                    previous_teacher[uid] = row["teacher"].clone()
                retention.append(dict(stage=stage, groups=len(memory.entries), bytes=len(memory.bytes()),
                                      origins=[r["origin"] for r in memory.entries]))
                if len(memory.entries) != stage*3:
                    raise AssertionError("Complete six-group Memory not covered")
            save_state(folder/f"stage{stage}.pt", model, optimizer, memory, {"stage": stage}, identity)
        # Real tokenizer shape checks, then actual blind inference at the largest evaluation shape.
        texts = [r.title for v in current.views for r in v.records]
        vendor = method == "VendorLink_SEQ"
        features = model.tokenizer(texts, padding=True, truncation=True,
            max_length=model.settings.vendor_tokens if vendor else model.settings.channel_tokens, return_tensors="pt")
        expected_tokens = model.settings.vendor_tokens if vendor else model.settings.channel_tokens
        if features["input_ids"].shape != (32, expected_tokens):
            raise AssertionError("Maximum tokenized training shape not reached")
        scores = evaluate(model, [blind_shape()], check)
        if len(scores["handwritten_44_ad_blind"]["scores"]) != 231:
            raise AssertionError("Maximum blind score table incomplete")
        if not np.isfinite(scores["handwritten_44_ad_blind"]["scores"]).all():
            raise AssertionError("Blind scores are nonfinite")
        result.update(updates=updates, retention=retention, peak=peak(model), step_rows=rows,
                      tokenized_shape=list(features["input_ids"].shape), blind_view_pairs=231,
                      checkpoint=str(folder/"stage3.pt"), settings=asdict(model.settings))
        write_json(folder/"result.json", result)
        evidence[method] = result
        del model, optimizer, memory, current, groups, scores, features
        gc.collect()
    if sum(r["updates"] for r in evidence.values()) != 15:
        raise AssertionError("Qualification update ledger differs")
    write_json(job/"result.json", dict(status="G1_PASSED", methods=evidence, updates=15))
    return evidence


def stage_loader(samples):
    next_stage = 0
    order = ("AlphaBay", "Dream", "Agora")
    def load(domain, log):
        nonlocal next_stage
        if domain != order[next_stage % 3]:
            raise ValueError("Future/repeated source access outside the stage sequence")
        source = samples[domain]
        path = Path(source["path"])
        if path.stat().st_size != source["bytes"] or file_sha256(path) != source["sha256"]:
            raise ValueError("Qualification source differs")
        value = json.loads(path.read_text())
        groups = [Group.from_dict(g) for g in value["groups"]]
        classes = value["classes"]
        if (value["domain"] != domain or len(groups) != 3 or len(classes) != source["classes"]
                or len(classes) != len(set(classes)) or any(g.domain != domain or g.split != "fit" for g in groups)
                or any(len(g.views) != 16 or any(len(v.records) != 2 for v in g.views) for g in groups)
                or not {v.identity for g in groups for v in g.views} <= set(classes)):
            raise ValueError("Invalid current-domain small source or full class list")
        with log.open("a") as out:
            out.write(json.dumps(dict(domain=domain, path=str(path), sha256=source["sha256"],
                                      groups=3, classes=len(classes)))+"\n")
        next_stage += 1
        return groups, classes
    return load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--permit", type=Path, required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--mode", choices=("g1", "g2"), required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--gate", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    permit = json.loads(args.permit.read_text())
    if (sys.platform != "linux" or permit.get("operation") != "vendorlink_native_qualification"
            or not permit.get("approved") or not permit.get("implementation_review_closed")
            or permit.get("config_sha256") != file_sha256(args.config)):
        raise ValueError("Approved fixed config and closed implementation review required")
    deadline = consume_launch(args.config, permit, config)-20
    # The inherited launch also binds argv, including job/checkpoint/gate/method.
    manifest = json.loads(Path(config["export_manifest"]).read_text())
    samples = manifest["qualification_samples"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(
        config["limits"]["gpu_allocator_bytes"]/torch.cuda.get_device_properties(0).total_memory)
    def check():
        if time.monotonic() >= deadline:
            raise TimeoutError("Native worker budget tail reserved")
    from transformers import BertModel, BertTokenizer
    def factory(method):
        torch.manual_seed(41)
        random.seed(41)
        np.random.seed(41)
        return Model(BertModel.from_pretrained(config["model_path"], local_files_only=True),
                     BertTokenizer.from_pretrained(config["model_path"], local_files_only=True, do_lower_case=True),
                     Settings(method)).to("cuda")
    if args.mode == "g1":
        qualify_g1(factory, stage_loader(samples), args.job, check)
    else:
        args.job.mkdir(parents=True, exist_ok=False)
        model = factory(args.method)
        memory = Memory(411, model.settings)
        identity = dict(scope="vendorlink_native_only", method=args.method)
        optimizer, _ = load_state(args.checkpoint, model, memory, identity)
        # G2 resumes the final current market; no earlier source or global cache is read.
        source = samples["Agora"]
        if file_sha256(Path(source["path"])) != source["sha256"]:
            raise ValueError("G2 current source differs")
        value = json.loads(Path(source["path"]).read_text())
        current = long_group(Group.from_dict(value["groups"][0]))
        write_json(args.job/"access.json", dict(current="Agora", sha256=source["sha256"], history="checkpoint_memory"))
        sync(model)
        write_json(args.job/"ready.json", dict(pid=__import__("os").getpid(), ready=time.monotonic()))
        while not args.gate.exists():
            check()
            time.sleep(.05)
        rows = [step(model, optimizer, current, memory, 3, k, 3, seed_for("g2", k), check,
                     args.job/"updates.jsonl", "three_way_long_stress") for k in (1, 2)]
        write_json(args.job/"result.json", dict(status="G2_WORKER_PASSED", updates=2, rows=rows, peak=peak(model)))


if __name__ == "__main__":
    main()
