from __future__ import annotations

def identity(arm: str = "G_half") -> dict:
    if arm != "G_half":
        raise ValueError("Only the fixed G_half candidate is authorized")
    return {"arm": arm, "mean_weight": .5, "logavgexp_weight": .5, "tau": 1.,
            "record_order": "canonical_offdiagonal_upper_v1", "output_dtype": "float32",
            "numerics": {"pooling": "float64_then_one_float32_cast", "encoder_forward": "bfloat16",
                         "head_parameters_gradients_adam": "float32", "tf32": False,
                         "deterministic_algorithms": True}}

def block_scores(blocks, arm: str = "G_half", stats: dict | None = None):
    identity(arm)
    if (blocks.ndim != 3 or not 2 <= blocks.shape[1] <= 8 or not 2 <= blocks.shape[2] <= 8
            or blocks.dtype not in (torch.float32, torch.float64) or not torch.isfinite(blocks).all()):
        raise ValueError("Finite complete 2..8 record blocks required")
    values = blocks.double().flatten(1)
    result = .5*values.mean(1)+.5*(torch.logsumexp(values, 1)-math.log(values.shape[1]))
    result = result.to(blocks.dtype)
    if not torch.isfinite(result).all():
        raise ValueError("Nonfinite G_half output")
    return result

def account_logits(table, slots, arm: str, stats: dict | None = None, context=None):
    identity(arm)
    n = len(slots)
    if table.shape != (n*(n-1)//2,) or not np.array_equal(np.unique(slots), np.arange(28)):
        raise ValueError("Complete table and all canonical account slots required")
    members = [np.flatnonzero(slots == i) for i in range(28)]
    if any(not 2 <= len(x) <= 8 for x in members):
        raise ValueError("Account records outside contract")
    i, j = torch.triu_indices(n, n, 1, device=table.device)
    matrix = table.new_zeros((n, n))
    matrix[i, j], matrix[j, i] = table, table
    buckets = {}
    for index, (a, b) in enumerate(zip(*np.triu_indices(28, 1))):
        buckets.setdefault((len(members[a]), len(members[b])), []).append((index, members[a], members[b]))
    scores = table.new_zeros(378)
    for entries in buckets.values():
        ids = torch.tensor([e[0] for e in entries], device=table.device)
        rows = torch.as_tensor(np.stack([e[1] for e in entries]), device=table.device)
        cols = torch.as_tensor(np.stack([e[2] for e in entries]), device=table.device)
        blocks = matrix[rows[:, :, None], cols[:, None, :]]
        scores = scores.index_copy(0, ids, block_scores(blocks, arm, stats))
    return scores
