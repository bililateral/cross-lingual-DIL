"""One analytic 3x3 witness, using the unchanged failed-run solver on Linux CPU.

This is not a replay of the unavailable formal K blocks or a repaired solver.
For K=log(9*P_star), the objective at any feasible Q is -KL(Q||P_star).
Thus P_star and the optimal value zero are independent exact references.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
import time

started = time.time()
workspace = Path(sys.argv[1]).resolve()
output = Path(sys.argv[2]).resolve()
sys.path.insert(0, str(workspace / "scripts"))
import torch
import step28_balanced_pooling as method

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
source = workspace / "scripts/step28_balanced_pooling.py"
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
assert source_hash == "21973808be64245e330cef08d28d841acea3716ba8b07043403a0c22b1958aac"
assert not torch.cuda.is_initialized()
zero_value, zero_plan, zero_info = method.solve(torch.zeros((1, 3, 3), dtype=torch.float64))
assert zero_value.abs().max() < 1e-12
assert (zero_plan - 1 / 9).abs().max() < 1e-12

epsilon = float(sys.argv[3]) if len(sys.argv) > 3 else 1e-8
p_star = torch.tensor([
    [1 / 3 - .05 - epsilon, .05, epsilon],
    [.05, 1 / 3 - .05 - epsilon, epsilon],
    [epsilon, epsilon, 1 / 3 - 2 * epsilon],
], dtype=torch.float64)
k = torch.log(9 * p_star)
assert (p_star.sum(0) - 1 / 3).abs().max() < 1e-15
assert (p_star.sum(1) - 1 / 3).abs().max() < 1e-15
try:
    value, plan, info = method.solve(k[None])
    status = "ORIGINAL_ACCEPTED"
    q = plan[0]
    details = info
    actual_value = float(value[0])
except method.SolverFailure as exc:
    status = "ORIGINAL_REJECTED"
    details = exc.details
    # Read locals from this handwritten exception only; no instrumentation or
    # change to the production solver's calculation, stopping, or return path.
    trace = exc.__traceback__
    while trace and trace.tb_frame.f_code.co_name != "_solve":
        trace = trace.tb_next
    assert trace is not None
    local = trace.tb_frame.f_locals
    q = local["q"][0].clone()
    actual_value = float(local["mu"][0] + local["primal"][0])

q_feasible = float(torch.maximum((q.sum(0) - 1 / 3).abs().max(), (q.sum(1) - 1 / 3).abs().max()))
plan_l1 = float((q - p_star).abs().sum())
reference_value = 0.0
independent_kl = float(torch.special.xlogy(q, q / p_star).sum())
result = dict(
    purpose="HANDWRITTEN_MECHANISM_WITNESS_NOT_FORMAL_REPLAY",
    source_sha256=source_hash, solver=method.SOLVER, torch=torch.__version__,
    gpu_initialized=torch.cuda.is_initialized(), cpu_threads=torch.get_num_threads(),
    epsilon=epsilon, input=k.tolist(), independent_optimal_plan=p_star.tolist(),
    independent_optimal_value=reference_value, original_status=status,
    original_diagnostics=details, final_rounded_plan=q.tolist(),
    actual_value=actual_value, value_error=abs(actual_value - reference_value),
    rounded_feasibility=q_feasible, plan_l1_error=plan_l1,
    independent_kl=independent_kl, float32_plan_l1_error=float((q.float().double()-p_star).abs().sum()),
    local_slow_mode_factor=(1 - 9 * epsilon)**2,
    zero_control=zero_info,
    formal_inputs=False, formal_labels=False, models_loaded=False,
    elapsed_seconds=time.time()-started,
)
output.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: result[key] for key in (
    "original_status", "value_error", "rounded_feasibility", "plan_l1_error", "elapsed_seconds")}))
