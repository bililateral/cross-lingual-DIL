"""CPU-only native-model audit with one handmade group; no research data access."""
from __future__ import annotations

import argparse
import gc
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import time
from unittest import mock

import numpy as np

import step28_base_model as base


def tensor_hash(value) -> str:
    return hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest()


def run(arm: str, destination: Path) -> None:
    import torch

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "" or destination.exists():
        raise ValueError("Use CPU and a new evidence file")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = time.monotonic()
    c = base.contract()
    actual_files = base.core.model_files(base.model_config(c, arm))
    for key in ("file_count", "total_size_bytes", "content_sha256"):
        if actual_files[key] != c["models"][arm][key]:
            raise ValueError("Native archive mismatch")
    # Explicit controller sizes, with labels derived independently of production.
    controllers = [k for k, size in enumerate([3] * 4 + [2] * 8) for _ in range(size)]
    labels = tuple(int(controllers[i] == controllers[j]) for i, j in itertools.combinations(range(28), 2))
    items = tuple(tuple((f"item_{i:02}_{j}", f"商品{i % 7}测试{j}",
                         f"手工核验内容{controllers[i]}，账号描述{i}。") for j in range(2)) for i in range(28))
    group = base.data.Group("handmade_native_audit", tuple(f"seller_{i:02}" for i in range(28)), items, labels)
    group.validate()
    if len(labels) != 378 or sum(labels) != 20:
        raise ValueError("Handmade class structure differs")
    with mock.patch.object(base.public, "public_inputs", side_effect=AssertionError("No formal inputs")), \
         mock.patch.object(base.public, "attach_labels", side_effect=AssertionError("No formal labels")):
        model = base.load_model(c, arm, device="cpu")
        optimizer = base.core.make_optimizer(model, c)
        optimized = [p for g in optimizer.param_groups for p in g["params"]]
        if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in optimized} != {id(p) for p in model.parameters()}:
            raise ValueError("Optimizer misses or duplicates model parameters")
        if not all(p.requires_grad for p in model.parameters()):
            raise ValueError("Unexpected frozen parameters")
        model.eval()
        texts = [c["models"][arm]["prefix"] + title + "\n" + description for _, title, description in items[0]]
        features = model.encoder.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        with torch.inference_mode():
            sentence = model.encoder({k: v.clone() for k, v in features.items()})["sentence_embedding"]
            tokens = model.encoder[0]({k: v.clone() for k, v in features.items()})["token_embeddings"]
            if arm == "multilingual_e5_large":
                mask = features["attention_mask"].unsqueeze(-1)
                independent = (tokens * mask).sum(1) / mask.sum(1)
            else:
                independent = tokens[:, 0]
            if arm == "labse":
                dense = model.encoder[2]
                independent = torch.tanh(torch.nn.functional.linear(independent, dense.linear.weight, dense.linear.bias))
            independent = torch.nn.functional.normalize(independent, dim=1)
            pooling_error = float((sentence - independent).abs().max())
            torch.testing.assert_close(sentence, independent, rtol=1e-5, atol=1e-6)
        probes = {name: p for name, p in model.named_parameters()
                  if ".attention.self.query.weight" in name or ".embeddings.word_embeddings.weight" in name
                  or name.startswith("head.") or name.startswith("encoder.2.linear.")}
        expected_layers = model.encoder[0].auto_model.config.num_hidden_layers
        if sum(".attention.self.query.weight" in n for n in probes) != expected_layers:
            raise ValueError("Backbone layer probes incomplete")
        before = {n: tensor_hash(p) for n, p in probes.items()}
        scores_before = base.score(model, [group], c, arm)
        captured = {}

        def head_output(module, inputs, output):
            captured["features"] = inputs[0].detach().numpy().copy()
            captured["logits"] = output.detach().flatten().numpy().copy()

        hook = model.head.register_forward_hook(head_output)
        update = base.update(model, optimizer, group, c, arm, seed=20260918, observe=True)
        hook.remove()
        z = captured["logits"].astype(np.float64)
        reference_bce = float(np.mean(np.logaddexp(0., z) - np.asarray(labels) * z))
        if abs(update["bce"] - reference_bce) > 1e-6:
            raise ValueError("Full-group BCE differs from independent scalar formula")
        parameters = []
        for name, p in probes.items():
            norm = float(p.grad.norm()) if p.grad is not None else 0.
            changed = tensor_hash(p) != before[name]
            if not np.isfinite(norm) or norm <= 0 or not changed:
                raise ValueError("Backbone/embedding/head probe did not learn: " + name)
            parameters.append({"name": name, "gradient_norm_after_clip": norm, "changed": changed,
                               "adam_step": float(optimizer.state[p]["step"])})
        no_grad = [n for n, p in model.named_parameters() if p.grad is None]
        if any(".pooler." not in n for n in no_grad):
            raise ValueError("Unexpected disconnected trainable parameter: " + repr(no_grad))
        if any(float(state["step"]) != 1 for state in optimizer.state.values()):
            raise ValueError("Wrong optimizer step")
        scores_after = base.score(model, [group], c, arm)
        if np.array_equal(scores_before, scores_after):
            raise ValueError("Actual model scores did not change")
        result = {"status": "NATIVE_CPU_HANDMADE_UPDATE_VERIFIED", "arm": arm,
                  "native_class": type(model.encoder[0].auto_model).__name__,
                  "embedding_dimension": model.encoder.get_sentence_embedding_dimension(),
                  "backbone_layers": expected_layers, "parameter_count": sum(p.numel() for p in model.parameters()),
                  "trainable_parameter_count": sum(p.numel() for p in model.parameters() if p.requires_grad),
                  "optimized_parameter_tensors": len(optimized), "actual_adam_state_tensors": len(optimizer.state),
                  "native_pooling_max_error": pooling_error, "gradient_checkpointing_enabled": model.encoder[0].auto_model.is_gradient_checkpointing,
                  "layer_embedding_head_probes": parameters, "unused_pooler_parameters": no_grad,
                  "update": update, "independent_bce": reference_bce,
                  "loss_absolute_error": abs(update["bce"] - reference_bce),
                  "pair_count": len(z), "positive_pairs": sum(labels), "items": 56,
                  "logit_change_max": float(np.max(np.abs(scores_before - scores_after))),
                  "formal_label_reads": 0, "formal_training_updates": 0, "handmade_updates": 1,
                  "saved_model_payloads": 0, "source_files": base.sources(), "model_archive": actual_files,
                  "device": "cpu", "torch": torch.__version__, "seconds": time.monotonic() - started,
                  "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "limitations": "One handmade CPU float32 step; no CUDA/bf16, formal-data learning, checkpoint or empirical target claim."}
        destination.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: result[k] for k in ("status", "arm", "seconds", "backbone_layers", "loss_absolute_error", "max_rss_kib")}))
        del optimizer, model
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=base.ARMS, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    run(args.arm, args.out)
