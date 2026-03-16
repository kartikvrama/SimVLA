"""Shared helpers for interview-practice notebooks.

These utilities keep notebook tests deterministic and lightweight.
"""

from __future__ import annotations

import math
import random
from typing import Callable, Iterable, Sequence

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    """Set random seeds across common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assert_shape(t: torch.Tensor, expected: Sequence[int], name: str = "tensor") -> None:
    if tuple(t.shape) != tuple(expected):
        raise AssertionError(f"{name} shape mismatch: got {tuple(t.shape)}, expected {tuple(expected)}")


def assert_close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-6, rtol: float = 1e-5, msg: str = "") -> None:
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        max_diff = (a - b).abs().max().item()
        raise AssertionError(f"Tensors are not close (max_diff={max_diff}). {msg}")


def expect_raises(fn: Callable[[], None], exc_type: type[BaseException], msg_contains: str | None = None) -> None:
    try:
        fn()
    except exc_type as err:  # pragma: no cover - exercised in notebook runtime
        if msg_contains and msg_contains not in str(err):
            raise AssertionError(
                f"Exception message does not contain expected text: {msg_contains!r}. Got: {str(err)!r}"
            ) from err
        return
    except Exception as err:  # pragma: no cover
        raise AssertionError(f"Expected {exc_type.__name__}, got {type(err).__name__}: {err}") from err
    raise AssertionError(f"Expected exception {exc_type.__name__} but none was raised.")


def tiny_token_batch(batch_size: int = 2, seq_len: int = 5, vocab_size: int = 31) -> torch.Tensor:
    return torch.randint(0, vocab_size, (batch_size, seq_len))


def tiny_image_batch(batch_size: int = 2, views: int = 3, channels: int = 3, height: int = 32, width: int = 32) -> torch.Tensor:
    return torch.rand(batch_size, views, channels, height, width)


def tiny_proprio_batch(batch_size: int = 2, dim: int = 8) -> torch.Tensor:
    return torch.randn(batch_size, dim)


def tiny_action_batch(batch_size: int = 2, horizon: int = 6, dim: int = 7) -> torch.Tensor:
    return torch.randn(batch_size, horizon, dim)


def print_test_banner(name: str) -> None:
    print(f"[PASS] {name}")


def cosine_decay(step: int, total_steps: int, min_ratio: float = 0.1) -> float:
    """Return scalar multiplier in [min_ratio, 1]."""
    if total_steps <= 0:
        return 1.0
    progress = min(max(step / total_steps, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_ratio + (1.0 - min_ratio) * cosine
