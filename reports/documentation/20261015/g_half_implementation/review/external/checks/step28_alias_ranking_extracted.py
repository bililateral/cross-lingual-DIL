from __future__ import annotations

def hard_candidates(predicted: Any, truth: Any, count: int = 5) -> tuple[Any, Any, Any]:
    """Known-negative indices only, deterministic exact-tie order, live score values."""
    import torch

    if predicted.shape != (378,) or truth.shape != (378,) or count != 5:
        raise ValueError("Expected 28-account scores and five hard negatives")
    # Independent shape/label/connectivity validation remains in base.objectives.
    if not torch.isfinite(predicted).all() or not torch.isin(truth, truth.new_tensor([0, 1])).all():
        raise ValueError("Nonfinite scores or nonbinary labels")
    left, right = torch.triu_indices(28, 28, 1, device=predicted.device)
    scores = predicted.new_zeros((28, 28))
    scores[left, right], scores[right, left] = predicted, predicted
    positive = torch.zeros((28, 28), dtype=torch.bool, device=predicted.device)
    positive[left, right], positive[right, left] = truth.bool(), truth.bool()
    negative = ~positive & ~torch.eye(28, dtype=torch.bool, device=predicted.device)
    if (positive.sum(1) < 1).any() or (negative.sum(1) < count).any():
        raise ValueError("Each query needs positive and known negative candidates")
    selected = torch.argsort(scores.detach().masked_fill(~negative, -torch.inf),
                             dim=1, descending=True, stable=True)[:, :count]
    if not negative.gather(1, selected).all():
        raise ValueError("Self or positive entered hard negatives")
    return scores, positive, selected
