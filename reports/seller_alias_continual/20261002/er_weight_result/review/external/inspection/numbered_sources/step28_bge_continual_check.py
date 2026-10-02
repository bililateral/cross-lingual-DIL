    1  """Linux CPU verification using handmade text only; never formal supervision.
    2  
    3  Native checks perform two actual updates per method. The Adam counter is openly
    4  rebased after the first update to isolate stage-two code. Full 288-step continuity
    5  and checkpoint branches are separately exercised with the tiny handmade model.
    6  """
    7  from __future__ import annotations
    8  
    9  import argparse
   10  import gc
   11  import os
   12  from pathlib import Path
   13  import platform
   14  import resource
   15  import sys
   16  import time
   17  from typing import Callable
   18  import unittest
   19  from unittest import mock
   20  
   21  import numpy as np
   22  
   23  import step28_bge_continual as method
   24  import step28_bge_continual_run as runner
   25  
   26  
   27  def native(arm: str, fixtures: object, policy: dict, check: Callable[[], None]) -> dict:
   28      import torch
   29  
   30      started = time.monotonic()
   31      c = method.config(policy)
   32      model = method.base.load_model(c, "split_rank", "cpu")
   33      optimizer = method.core.make_optimizer(model, c)
   34      old, current = fixtures.handmade_group("native_A", 1), fixtures.handmade_group("native_B", 2)
   35      initial = method.core.state_digest(model.state_dict())
   36      optimized = [p for group in optimizer.param_groups for p in group["params"]]
   37      if len({id(p) for p in optimized}) != len(optimized) or {id(p) for p in optimized} != {id(p) for p in model.parameters()}:
   38          raise ValueError("Native optimizer parameter coverage differs")
   39      warm = method.update(model, optimizer, old, None, None, c, "seq", 1, 1, 81, 82,
   40                           observe=True, check=check)
   41      after_warm = method.core.state_digest(model.state_dict())
   42      reference = method.ranking.score(model, [old], c, check)[0]
   43      # This is a declared test fixture. No claim that 288 native updates occurred.
   44      for state in optimizer.state.values():
   45          state["step"].fill_(288)
   46      prior_optimizer = method.core.state_digest(optimizer.state_dict())
   47      component = {}
   48      if arm == "logit":
   49          model.train()
   50          torch.manual_seed(108)
   51          prediction = method.base.logits(model, old, c, "split_rank", check)
   52          target = torch.tensor(reference, dtype=prediction.dtype)
   53          penalty = (prediction - target).square().mean()
   54          probes = [model.encoder[0].auto_model.encoder.layer[0].attention.self.query.weight,
   55                    model.head[0].weight]
   56          gradients = torch.autograd.grad(penalty, probes)
   57          norms = [float(g.norm()) for g in gradients]
   58          if target.requires_grad or not all(np.isfinite(v) and v > 0 for v in norms):
   59              raise ValueError("Historical MSE alone does not reach the real encoder and head")
   60          component = {"encoder_first_query_weight": norms[0], "head_hidden_weight": norms[1],
   61                       "reference_is_detached": True, "reference_mode": "origin_model_eval",
   62                       "student_mode": "train"}
   63          del prediction, target, penalty, probes, gradients
   64      probes = {name: parameter for name, parameter in model.named_parameters()
   65                if ".attention.self.query.weight" in name or ".embeddings.word_embeddings.weight" in name
   66                or name.startswith("head.")}
   67      layers = model.encoder[0].auto_model.config.num_hidden_layers
   68      if sum(".attention.self.query.weight" in name for name in probes) != layers:
   69          raise ValueError("Native layer probes are incomplete")
   70      before = {name: method.core.state_digest(parameter) for name, parameter in probes.items()}
   71      captured = []
   72      hook = model.head.register_forward_hook(lambda _, __, out: captured.append(out.detach().flatten().numpy().copy()))
   73      try:
   74          update = method.update(model, optimizer, current, old if arm != "seq" else None,
   75                                 reference if arm == "logit" else None, c, arm, 2, 1, 107, 108,
   76                                 observe=True, check=check)
   77      finally:
   78          hook.remove()
   79      if len(captured) != (1 if arm == "seq" else 2):
   80          raise ValueError("Unexpected current/history forwards")
   81      independent = {"current": fixtures.scalar_losses(captured[0], current.labels)}
   82      if arm != "seq":
   83          independent["history"] = fixtures.scalar_losses(captured[1], old.labels)
   84      for role, losses in independent.items():
   85          for name, value in losses.items():
   86              if abs(update[role + "_" + name] - value) > (4e-6 if name == "total" else 2e-6):
   87                  raise ValueError("Native loss differs from scalar reference: " + role + "/" + name)
   88      expected_mse = float(np.mean((captured[1].astype(np.float64) - reference) ** 2)) if arm == "logit" else 0.
   89      if abs(update["logit_mse"] - expected_mse) > 1e-6:
   90          raise ValueError("Native logit MSE differs from scalar reference")
   91      rows = []
   92      for name, parameter in probes.items():
   93          norm = float(parameter.grad.norm()) if parameter.grad is not None else 0.
   94          changed = before[name] != method.core.state_digest(parameter)
   95          if not np.isfinite(norm) or norm <= 0 or not changed:
   96              raise ValueError("Native parameter has no actual update: " + name)
   97          rows.append({"name": name, "gradient_norm_after_clip": norm, "parameters_changed": changed,
   98                       "adam_step": float(optimizer.state[parameter]["step"])})
   99      unused = [name for name, parameter in model.named_parameters() if parameter.grad is None]
  100      if any(".pooler." not in name for name in unused) or method.adam_step(optimizer) != 289:
  101          raise ValueError("Unexpected unused parameters or Adam counter")
  102      result = {"method": arm, "native_updates_actually_executed": 2,
  103                "handmade_counter_fixture": "One actual warm update, then step counters rebased 1 -> 288; native continuity over 288 updates is not claimed",
  104                "initial_model_state_sha256": initial, "after_warm_model_state_sha256": after_warm,
  105                "before_stage2_optimizer_sha256": prior_optimizer,
  106                "after_model_state_sha256": method.core.state_digest(model.state_dict()),
  107                "after_optimizer_state_sha256": method.core.state_digest(optimizer.state_dict()),
  108                "warm_update": warm, "stage2_update": update,
  109                "captured_current_logits": captured[0].tolist(),
  110                "captured_history_logits": captured[1].tolist() if arm != "seq" else None,
  111                "origin_eval_reference": reference.tolist(), "independent_loss": independent,
  112                "independent_logit_mse": expected_mse, "logit_term_gradient": component,
  113                "parameter_probes": rows, "unused_parameters": unused, "backbone_layers": layers,
  114                "parameter_count": sum(p.numel() for p in model.parameters()),
  115                "seconds": time.monotonic() - started}
  116      del model, optimizer, optimized, probes
  117      gc.collect()
  118      return result
  119  
  120  
  121  def run(destination: Path) -> dict:
  122      import torch
  123  
  124      if platform.system() != "Linux" or os.environ.get("CUDA_VISIBLE_DEVICES") != "" or destination.exists():
  125          raise ValueError("Use Linux CPU and a new evidence file; CUDA_VISIBLE_DEVICES must be empty")
  126      policy, source_files = method.contract(), runner.sources()
  127      torch.set_num_threads(1)
  128      os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
  129      torch.use_deterministic_algorithms(True)
  130      started = time.monotonic()
  131      destination.parent.mkdir(parents=True, exist_ok=True)
  132  
  133      def check():
  134          if time.monotonic() - started >= policy["cpu_verification"]["maximum_seconds"]:
  135              raise RuntimeError("CPU verification time limit reached")
  136          if sum(p.stat().st_size for p in destination.parent.rglob("*") if p.is_file()) > policy["cpu_verification"]["maximum_evidence_bytes"]:
  137              raise RuntimeError("CPU evidence budget exceeded")
  138  
  139      sys.path.insert(0, str(method.data.ROOT / "tests"))
  140      import test_step28_bge_continual_contracts as fixtures
  141      try:
  142          with mock.patch.object(method.base.public, "public_inputs", side_effect=AssertionError("No formal text")), \
  143               mock.patch.object(method.base.public, "attach_labels", side_effect=AssertionError("No formal CSV labels")), \
  144               mock.patch.object(method.data, "Archive", side_effect=AssertionError("No data archive")):
  145              suite = unittest.defaultTestLoader.loadTestsFromModule(fixtures)
  146              tested = unittest.TextTestRunner(verbosity=2).run(suite)
  147              contracts = {"passed": tested.testsRun - len(tested.failures) - len(tested.errors) - len(tested.skipped),
  148                           "failed": len(tested.failures) + len(tested.errors), "skipped": len(tested.skipped)}
  149              method.data.write_json(destination.parent / "contracts.json", contracts)
  150              if not tested.wasSuccessful() or tested.skipped:
  151                  raise RuntimeError("Applicable handmade tests must pass without skips")
  152              check()
  153              c = method.config(policy)
  154              archive = method.core.model_files(method.base.model_config(c, "split_rank"))
  155              if any(archive[k] != c["models"]["split_rank"][k]
  156                     for k in ("file_count", "total_size_bytes", "content_sha256")):
  157                  raise ValueError("Native pretrained archive differs")
  158              reports = {}
  159              for arm in method.UPDATED:
  160                  print(method.data.json_bytes({"event": "native_start", "method": arm}).decode(), flush=True)
  161                  reports[arm] = native(arm, fixtures, policy, check)
  162                  method.data.write_json(destination.parent / (arm + ".json"), reports[arm])
  163                  print(method.data.json_bytes({"event": "native_end", "method": arm,
  164                                                 "seconds": reports[arm]["seconds"]}).decode(), flush=True)
  165              for arm in ("er", "logit"):
  166                  for field in ("initial_model_state_sha256", "after_warm_model_state_sha256",
  167                                "before_stage2_optimizer_sha256", "captured_current_logits", "origin_eval_reference"):
  168                      if reports[arm][field] != reports["seq"][field]:
  169                          raise ValueError("Native shared-state/current pairing differs: " + field)
  170              if reports["er"]["captured_history_logits"] != reports["logit"]["captured_history_logits"]:
  171                  raise ValueError("Native history forwards are not paired")
  172              if source_files != runner.sources():
  173                  raise ValueError("Sources changed during verification")
  174              result = {"status": "PASS_BGE_CONTINUAL_HANDMADE_CPU", "source_files": source_files,
  175                        "contracts": contracts, "native": {arm: method.data.record(destination.parent / (arm + ".json"), destination.parent)
  176                                                           for arm in method.UPDATED},
  177                        "native_updates_actually_executed": 6, "formal_inputs": False, "formal_labels": False,
  178                        "formal_updates": 0, "gpu": False, "retained_native_weights": 0, "archive": archive,
  179                        "seconds": time.monotonic() - started,
  180                        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
  181                        "environment": {"python": platform.python_version(), "torch": torch.__version__,
  182                                        "numpy": np.__version__, "cpu_affinity": sorted(os.sched_getaffinity(0))},
  183                        "limitations": "Tiny full-stage/reload plus native two-update fixtures; no formal data/performance, native full checkpoint, CUDA/BF16 or 288-step native continuity claim"}
  184              method.data.write_json(destination, result)
  185              check()
  186              return result
  187      except Exception as error:
  188          method.data.write_json(destination.parent / "failure.json", {
  189              "status": "CPU_VERIFICATION_FAILED", "error": str(error), "seconds": time.monotonic() - started,
  190              "formal_inputs": False, "formal_labels": False})
  191          raise
  192  
  193  
  194  if __name__ == "__main__":
  195      parser = argparse.ArgumentParser(description=__doc__)
  196      parser.add_argument("--out", type=Path, required=True)
  197      args = parser.parse_args()
  198      result = run(args.out.resolve())
  199      print(method.data.json_bytes({k: result[k] for k in ("status", "native_updates_actually_executed", "seconds")}).decode())
