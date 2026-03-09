from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class SamplingBudgetConfig:
    """Generic compute budget for sampler-side token usage.

    `max_sampling_tokens` matches the repository's existing `sampling_tokens`
    metric: every sampled token counts once, including final emitted tokens and
    internal proposal/rollout tokens.
    """

    max_sampling_tokens: Optional[int] = None

    def normalized(self) -> "SamplingBudgetConfig":
        if self.max_sampling_tokens is None:
            return SamplingBudgetConfig(max_sampling_tokens=None)
        return SamplingBudgetConfig(max_sampling_tokens=max(int(self.max_sampling_tokens), 0))


class SamplingBudgetTracker:
    """Stateful tracker for sampler compute spend."""

    def __init__(self, config: Optional[SamplingBudgetConfig] = None):
        normalized = (config or SamplingBudgetConfig()).normalized()
        self.max_sampling_tokens = normalized.max_sampling_tokens
        self.sampling_tokens_used = 0

    def remaining_sampling_tokens(self) -> Optional[int]:
        if self.max_sampling_tokens is None:
            return None
        return max(int(self.max_sampling_tokens) - int(self.sampling_tokens_used), 0)

    def clamp_sampling_tokens(self, requested_tokens: int) -> int:
        requested = max(int(requested_tokens), 0)
        remaining = self.remaining_sampling_tokens()
        if remaining is None:
            return requested
        return min(requested, remaining)

    def can_spend(self, requested_tokens: int = 1) -> bool:
        needed = max(int(requested_tokens), 0)
        remaining = self.remaining_sampling_tokens()
        if remaining is None:
            return True
        return remaining >= needed

    def spend(self, observed_tokens: int) -> int:
        spent = max(int(observed_tokens), 0)
        remaining = self.remaining_sampling_tokens()
        if remaining is not None and spent > remaining:
            raise ValueError(
                f"Sampling budget exceeded: tried to spend {spent} tokens with only {remaining} remaining."
            )
        self.sampling_tokens_used += spent
        return spent

    def exhausted(self) -> bool:
        remaining = self.remaining_sampling_tokens()
        return remaining is not None and remaining == 0

    def metadata(self) -> Dict[str, Any]:
        return {
            "budget_active": self.max_sampling_tokens is not None,
            "budget_max_sampling_tokens": self.max_sampling_tokens,
            "budget_used_sampling_tokens": int(self.sampling_tokens_used),
            "budget_remaining_sampling_tokens": self.remaining_sampling_tokens(),
            "budget_exhausted": self.exhausted(),
        }


def fit_batched_sampling_plan(
    *,
    num_prefixes: int,
    samples_per_prefix: int,
    max_tokens_per_sample: int,
    remaining_sampling_tokens: Optional[int],
) -> Tuple[int, int]:
    """Fit a batched sampling request under a hard token budget.

    Returns `(samples_per_prefix, max_tokens_per_sample)` such that the worst
    case token usage

      num_prefixes * samples_per_prefix * max_tokens_per_sample

    does not exceed `remaining_sampling_tokens`. We preserve the requested
    per-sample token horizon first and reduce the number of samples before
    shortening individual samples.
    """

    prefixes = max(int(num_prefixes), 0)
    samples = max(int(samples_per_prefix), 0)
    tokens = max(int(max_tokens_per_sample), 0)
    if prefixes <= 0 or samples <= 0 or tokens <= 0:
        return 0, 0

    if remaining_sampling_tokens is None:
        return samples, tokens

    remaining = max(int(remaining_sampling_tokens), 0)
    if remaining <= 0:
        return 0, 0

    max_full_samples = remaining // (prefixes * tokens)
    if max_full_samples >= samples:
        return samples, tokens
    if max_full_samples >= 1:
        return max_full_samples, tokens

    max_shortened_tokens = remaining // prefixes
    if max_shortened_tokens >= 1:
        return 1, min(tokens, max_shortened_tokens)

    return 0, 0
