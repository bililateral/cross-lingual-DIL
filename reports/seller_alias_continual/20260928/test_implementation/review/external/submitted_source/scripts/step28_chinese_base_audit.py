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

import step28_chinese_base as base


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
        texts = base.record_texts(group, c["interventions"][arm]["representation"])[:2]
        features = model.encoder.tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
        with torch.inference_mode():
            sentence = model.encoder({k: v.clone() for k, v in features.items()})["sentence_embedding"]
            tokens = model.encoder[0]({k: v.clone() for k, v in features.items()})["token_embeddings"]
            independent = tokens[:, 0]
            independent = torch.nn.functional.normalize(independent, dim=1)
            pooling_error = float((sentence - independent).abs().max())
            torch.testing.assert_close(sentence, independent, rtol=1e-5, atol=1e-6)
        # Trace each objective separately through actual BERT and the production account path.
        model.train()
        torch.manual_seed(20260924)
        probe_logits = base.logits(model, group, c, arm)
        terms = base.objectives(probe_logits, torch.tensor(labels, dtype=torch.float32),
                                c["interventions"][arm]["rank_weight"])
        term_probes = [model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,
                       model.head[0].weight]
        term_gradients = {}
        for name in ("bce", "rank"):
            gradients = torch.autograd.grad(terms[name], term_probes, retain_graph=True)
            norms = [float(g.norm()) for g in gradients]
            if not all(np.isfinite(n) and n > 0 for n in norms):
                raise ValueError("Objective is disconnected from actual encoder/head: " + name)
            term_gradients[name] = norms
        del probe_logits, terms, gradients
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
        pairs = list(itertools.combinations(range(28), 2))
        rank_terms = []
        for query in range(28):
            indices = [index for index, pair in enumerate(pairs) if query in pair]
            values = z[indices]
            positives = np.asarray(labels)[indices].astype(bool)
            maximum = float(values.max())
            rank_terms.append(maximum + np.log(np.exp(values-maximum).sum()) - values[positives].mean())
        reference_rank = float(np.mean(rank_terms))
        if (abs(update["rank"] - reference_rank) > 1e-6
                or abs(update["total"] - reference_bce -
                       c["interventions"][arm]["rank_weight"] * reference_rank) > 2e-6):
            raise ValueError("Actual training objective differs from independent scalar rank/BCE")
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
                  "independent_rank": reference_rank, "objective_probe_gradient_norms": term_gradients,
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
