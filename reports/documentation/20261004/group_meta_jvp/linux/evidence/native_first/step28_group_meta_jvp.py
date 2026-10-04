"""Exact forward-over-reverse support HVP, reusing the staged outer update."""
from typing import Callable

import torch

import step28_group_meta_staged as staged


def hvp(loss: Callable, parameters: dict, vector: tuple) -> tuple:
    """Full Hessian times fixed Av; no surrounding autograd history retained.

    The scalar loss must use the same RNG and local selection branch as the
    support first-order pass. Unused zeros returned by torch.func must not
    change the ordinary first-order participation mask in staged_gradient.
    """
    values = {name: p.detach() for name, p in parameters.items()}
    direction = {name: v.detach() for name, v in zip(parameters, vector, strict=True)}
    product = torch.func.jvp(torch.func.grad(loss), (values,), (direction,))[1]
    return tuple(product[name].detach() for name in parameters)


def staged_gradient(parameters: dict, support: tuple, query: tuple, rates: dict,
                    beta: float, progress: Callable = lambda phase: None) -> tuple:
    return staged.staged_gradient(parameters, support, query, rates, beta, progress,
                                  lambda i, p, v: hvp(support[i], p, v))


def gradient(*args, **kwargs) -> tuple:
    return staged.gradient(*args, **kwargs, hessian_product=hvp)


def update(*args, **kwargs) -> dict:
    return staged.update(*args, **kwargs, hessian_product=hvp)
