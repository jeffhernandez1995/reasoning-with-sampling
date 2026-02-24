"""Approximate (scalable) power sampling.

Implements the approximate power-sampling procedure from arXiv:2601.21590v1.
Core equations:
  - Monte Carlo scaling factor estimate (Eq. 3)
  - Power-distribution approximation (Eq. 4)
  - Jackknife correction (Eq. 5 + Eq. 6)

This module is backend-agnostic via `PowerApproxScorer`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch

DEFAULT_APPROX_LOOKAHEAD_TOKENS = 192


@dataclass
class PowerSamplerApproxConfig:
    # We follow repository convention alpha = 1 / temp.
    temp: float = 0.25
    # Candidate set size Top-K_t.
    top_k: int = 8
    # For block sampling (B > 1): sample L candidates then keep Top-K.
    candidate_pool_size: int = 32
    # Rollouts per candidate (M_t).
    rollouts_per_candidate: int = 8
    # Truncated rollout horizon H_t. None falls back to DEFAULT_APPROX_LOOKAHEAD_TOKENS.
    lookahead_tokens: Optional[int] = DEFAULT_APPROX_LOOKAHEAD_TOKENS
    # 1 => token-level Algorithm 1. >1 => Appendix-B block variant.
    block_size: int = DEFAULT_APPROX_LOOKAHEAD_TOKENS
    # Use Eq. 5 jackknife correction when M_t > 1.
    use_jackknife: bool = True


class PowerApproxScorer(Protocol):
    @property
    def max_seq_len(self) -> Optional[int]:
        ...

    def topk_next_tokens(self, prefix: List[int], k: int) -> Tuple[List[int], List[float]]:
        """Return Top-k token ids and log p(token|prefix) under base model p."""
        ...

    def sample_continuations(
        self,
        prefixes: List[List[int]],
        *,
        max_new_tokens: int,
        n: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[List[List[int]]], List[List[float]]]:
        """Sample n continuations per prefix from base model p.

        Returns:
          continuations[prefix_i][sample_j] -> generated token ids
          logp_sums[prefix_i][sample_j] -> sum log p(tokens | prefix)
        """
        ...


def _clamp_new_tokens(max_new_tokens: int, prompt_len: int, max_seq_len: Optional[int]) -> int:
    if max_seq_len is None:
        return max(0, int(max_new_tokens))
    allowed = int(max_seq_len) - int(prompt_len)
    return max(0, min(int(max_new_tokens), allowed))


def _softmax_from_logits(logits_1d: torch.Tensor) -> torch.Tensor:
    if logits_1d.numel() == 0:
        return logits_1d
    log_z = torch.logsumexp(logits_1d, dim=0)
    probs = torch.exp(logits_1d - log_z)
    if not torch.all(torch.isfinite(probs)):
        return torch.full_like(logits_1d, 1.0 / float(max(int(logits_1d.numel()), 1)))
    return probs


def _mean_logp(logp_sums: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Convert cumulative log-probabilities to per-token means."""
    lengths = torch.clamp(lengths.to(dtype=logp_sums.dtype), min=1.0)
    return logp_sums / lengths


def _compute_power_probs(
    *,
    logp_items: torch.Tensor,  # (K,)
    item_lengths: torch.Tensor,  # (K,)
    rollout_logp_sums: torch.Tensor,  # (K, M)
    rollout_lengths: torch.Tensor,  # (K, M)
    alpha: float,
    use_jackknife: bool,
) -> torch.Tensor:
    """Compute the approximate power distribution over candidates.

    Base estimator (Eq. 4):
      p_hat(i) ∝ p(i)^alpha * z_hat(i),
      z_hat(i) = (1/M) * sum_r exp((alpha - 1) * log p_rollout_{i,r}).

    We use per-token mean log-probabilities for both candidate and rollout
    weights to reduce variable-length bias.

    Jackknife (Eq. 5 + Eq. 6):
      p_jk(i) = M * p_hat(i) - ((M - 1) / M) * sum_s p_hat_{-s}(i).
    """
    k = int(logp_items.numel())
    if k == 0:
        return torch.empty((0,), dtype=torch.float32, device=logp_items.device)

    if int(item_lengths.numel()) != k:
        item_lengths = torch.ones((k,), dtype=torch.float32, device=logp_items.device)
    item_logp_mean = _mean_logp(logp_items, item_lengths)

    if alpha <= 0:
        return _softmax_from_logits(item_logp_mean)

    if rollout_logp_sums.numel() == 0:
        return _softmax_from_logits(alpha * item_logp_mean)

    m = int(rollout_logp_sums.shape[1])
    if m <= 0:
        return _softmax_from_logits(alpha * item_logp_mean)

    if tuple(rollout_lengths.shape) != tuple(rollout_logp_sums.shape):
        rollout_lengths = torch.ones_like(rollout_logp_sums, dtype=torch.float32)
    rollout_logp_mean = _mean_logp(rollout_logp_sums, rollout_lengths)

    log_w = (alpha - 1.0) * rollout_logp_mean
    log_w = torch.where(torch.isfinite(log_w), log_w, torch.full_like(log_w, -1e9))

    # log z_hat(i) = logmeanexp(log_w_i)
    log_z = torch.logsumexp(log_w, dim=1) - math.log(float(m))
    log_num = alpha * item_logp_mean + log_z
    p_hat = _softmax_from_logits(log_num)

    if not use_jackknife or m <= 1:
        return p_hat

    # Stable leave-one-out estimates in scaled exp-space.
    m_i = torch.max(log_w, dim=1, keepdim=True).values
    w_scaled = torch.exp(log_w - m_i)
    sum_scaled = torch.sum(w_scaled, dim=1, keepdim=True)
    sum_minus = torch.clamp(sum_scaled - w_scaled, min=1e-30)
    log_z_loo = m_i + torch.log(sum_minus / float(m - 1))

    log_num_loo = alpha * item_logp_mean.view(k, 1) + log_z_loo
    # For each leave-one-out sample s, normalize across candidates i.
    p_hat_loo = torch.softmax(log_num_loo.transpose(0, 1), dim=1)  # (M, K)
    sum_p_hat_loo = torch.sum(p_hat_loo, dim=0)  # (K,)

    p_jk = float(m) * p_hat - (float(m - 1) / float(m)) * sum_p_hat_loo
    p_jk = torch.clamp(p_jk, min=0.0)
    total = torch.sum(p_jk)
    if not torch.isfinite(total) or float(total.item()) <= 0.0:
        return p_hat
    return p_jk / total


def _sample_index(probs: torch.Tensor, rng: np.random.Generator) -> int:
    p_np = probs.detach().cpu().numpy().astype(np.float64)
    p_np = np.clip(p_np, 0.0, 1.0)
    total = p_np.sum()
    if not np.isfinite(total) or total <= 0.0:
        p_np = np.ones_like(p_np) / max(len(p_np), 1)
    else:
        p_np /= total
    return int(rng.choice(len(p_np), p=p_np))


def _resolve_lookahead(cfg: PowerSamplerApproxConfig, remaining: int) -> int:
    """Bound rollout horizon to avoid pathological full-horizon rollouts."""
    cap = cfg.lookahead_tokens
    if cap is None:
        cap = DEFAULT_APPROX_LOOKAHEAD_TOKENS
    cap_int = max(0, int(cap))
    return min(max(0, int(remaining)), cap_int)


def approx_power_sample(
    scorer: PowerApproxScorer,
    context: List[int],
    *,
    cfg: PowerSamplerApproxConfig,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """Sample a continuation from the approximate power distribution.

    Returns:
      - full sequence including context
      - diagnostics
    """
    if rng is None:
        rng = np.random.default_rng()

    if cfg.temp <= 0:
        raise ValueError("cfg.temp must be > 0 (alpha = 1/cfg.temp).")
    if eos_token_id is None:
        raise ValueError("eos_token_id must be provided for approximate power sampling.")

    alpha = 1.0 / float(cfg.temp)
    eos_id = int(eos_token_id)
    block_size = max(1, int(cfg.block_size))

    max_new_tokens = _clamp_new_tokens(max_new_tokens, len(context), scorer.max_seq_len)
    if max_new_tokens <= 0:
        return context.copy(), {
            "steps": 0.0,
            "rollouts": 0.0,
            "rollout_tokens": 0.0,
            "candidate_tokens": 0.0,
            "output_tokens": 0.0,
            "sampling_tokens": 0.0,
            "internal_sampling_tokens": 0.0,
        }

    seq = context.copy()
    steps = 0
    total_rollouts = 0
    total_rollout_tokens = 0
    total_candidate_tokens = 0
    total_sampling_tokens = 0

    if block_size == 1:
        while (len(seq) - len(context)) < max_new_tokens:
            remaining = max_new_tokens - (len(seq) - len(context))
            if remaining <= 0:
                break

            k = max(1, int(cfg.top_k))
            cand_tokens, cand_logps = scorer.topk_next_tokens(seq, k)
            k = len(cand_tokens)
            if k == 0:
                break

            logp_items = torch.tensor(cand_logps, dtype=torch.float32)
            item_lengths = torch.ones((k,), dtype=torch.float32)
            lookahead = _resolve_lookahead(cfg, remaining - 1)
            m = max(0, int(cfg.rollouts_per_candidate))

            if lookahead <= 0 or m <= 0:
                probs = _softmax_from_logits(alpha * _mean_logp(logp_items, item_lengths))
            else:
                rollout_logp = torch.zeros((k, m), dtype=torch.float32)
                rollout_lengths = torch.ones((k, m), dtype=torch.float32)
                non_terminal = [i for i, tok in enumerate(cand_tokens) if int(tok) != eos_id]
                if non_terminal:
                    prefixes = [seq + [int(cand_tokens[i])] for i in non_terminal]
                    rollout_len = _clamp_new_tokens(lookahead, len(prefixes[0]), scorer.max_seq_len)
                    if rollout_len > 0:
                        conts, logp_sums = scorer.sample_continuations(
                            prefixes,
                            max_new_tokens=rollout_len,
                            n=m,
                            temperature=1.0,
                            top_p=1.0,
                            top_k=None,
                            eos_token_id=eos_id,
                        )
                        logp_sums_t = torch.as_tensor(logp_sums, dtype=torch.float32)
                        rollout_lengths_t = torch.as_tensor(
                            [[float(max(len(tokens), 1)) for tokens in per_prefix] for per_prefix in conts],
                            dtype=torch.float32,
                        )
                        for local_idx, i in enumerate(non_terminal):
                            rollout_logp[i, :] = logp_sums_t[local_idx]
                            rollout_lengths[i, :] = rollout_lengths_t[local_idx]
                        total_rollouts += len(non_terminal) * m
                        rollout_token_count = int(sum(len(tokens) for per_prefix in conts for tokens in per_prefix))
                        total_rollout_tokens += rollout_token_count
                        total_sampling_tokens += rollout_token_count

                probs = _compute_power_probs(
                    logp_items=logp_items,
                    item_lengths=item_lengths,
                    rollout_logp_sums=rollout_logp,
                    rollout_lengths=rollout_lengths,
                    alpha=alpha,
                    use_jackknife=bool(cfg.use_jackknife),
                )

            choice = _sample_index(probs, rng)
            token = int(cand_tokens[choice])
            seq.append(token)
            steps += 1
            total_sampling_tokens += 1
            if token == eos_id:
                break

        internal_sampling_tokens = max(total_sampling_tokens - steps, 0)
        return seq, {
            "steps": float(steps),
            "rollouts": float(total_rollouts),
            "rollout_tokens": float(total_rollout_tokens),
            "candidate_tokens": float(total_candidate_tokens),
            "output_tokens": float(steps),
            "sampling_tokens": float(total_sampling_tokens),
            "internal_sampling_tokens": float(internal_sampling_tokens),
        }

    # Appendix-B style block version.
    while (len(seq) - len(context)) < max_new_tokens:
        remaining = max_new_tokens - (len(seq) - len(context))
        if remaining <= 0:
            break

        # Final chunk uses low-temp block distribution (Eq. 16 analogue).
        if remaining <= block_size:
            l = max(1, int(cfg.candidate_pool_size))
            k = max(1, min(int(cfg.top_k), l))
            chunk_len = _clamp_new_tokens(remaining, len(seq), scorer.max_seq_len)
            if chunk_len <= 0:
                break

            conts, logp_sums = scorer.sample_continuations(
                [seq],
                max_new_tokens=chunk_len,
                n=l,
                temperature=float(cfg.temp),
                top_p=1.0,
                top_k=None,
                eos_token_id=eos_id,
            )
            candidate_blocks = conts[0]
            candidate_logp = np.asarray(logp_sums[0], dtype=np.float64)
            candidate_token_count = int(sum(len(tokens) for tokens in candidate_blocks))
            total_candidate_tokens += candidate_token_count
            total_sampling_tokens += candidate_token_count

            order = np.argsort(candidate_logp)[::-1][:k]
            top_blocks = [candidate_blocks[i] for i in order]
            top_logp = torch.tensor([float(candidate_logp[i]) for i in order], dtype=torch.float32)
            top_lengths = torch.tensor([float(max(len(block), 1)) for block in top_blocks], dtype=torch.float32)
            probs = _softmax_from_logits(alpha * _mean_logp(top_logp, top_lengths))

            chosen = top_blocks[_sample_index(probs, rng)]
            for tok in chosen:
                if (len(seq) - len(context)) >= max_new_tokens:
                    break
                seq.append(int(tok))
                steps += 1
                if int(tok) == eos_id:
                    break
            break

        l = max(1, int(cfg.candidate_pool_size))
        k = max(1, min(int(cfg.top_k), l))
        block_len = _clamp_new_tokens(block_size, len(seq), scorer.max_seq_len)
        if block_len <= 0:
            break

        conts, logp_sums = scorer.sample_continuations(
            [seq],
            max_new_tokens=block_len,
            n=l,
            temperature=float(cfg.temp),
            top_p=1.0,
            top_k=None,
            eos_token_id=eos_id,
        )
        candidate_blocks = conts[0]
        candidate_logp = np.asarray(logp_sums[0], dtype=np.float64)
        candidate_token_count = int(sum(len(tokens) for tokens in candidate_blocks))
        total_candidate_tokens += candidate_token_count
        total_sampling_tokens += candidate_token_count

        order = np.argsort(candidate_logp)[::-1][:k]
        top_blocks = [candidate_blocks[i] for i in order]
        top_logp = torch.tensor([float(candidate_logp[i]) for i in order], dtype=torch.float32)
        top_lengths = torch.tensor([float(max(len(block), 1)) for block in top_blocks], dtype=torch.float32)

        lookahead = _resolve_lookahead(cfg, remaining - block_len)
        m = max(0, int(cfg.rollouts_per_candidate))

        if lookahead <= 0 or m <= 0:
            probs = _softmax_from_logits(alpha * _mean_logp(top_logp, top_lengths))
        else:
            terminal = [eos_id in block for block in top_blocks]
            rollout_logp = torch.zeros((k, m), dtype=torch.float32)
            rollout_lengths = torch.ones((k, m), dtype=torch.float32)
            non_terminal = [i for i, term in enumerate(terminal) if not term]

            if non_terminal:
                prefixes = [seq + [int(t) for t in top_blocks[i]] for i in non_terminal]
                rollout_len = _clamp_new_tokens(lookahead, len(prefixes[0]), scorer.max_seq_len)
                if rollout_len > 0:
                    conts2, logp_sums2 = scorer.sample_continuations(
                        prefixes,
                        max_new_tokens=rollout_len,
                        n=m,
                        temperature=1.0,
                        top_p=1.0,
                        top_k=None,
                        eos_token_id=eos_id,
                    )
                    logp_sums2_t = torch.as_tensor(logp_sums2, dtype=torch.float32)
                    rollout_lengths2_t = torch.as_tensor(
                        [[float(max(len(tokens), 1)) for tokens in per_prefix] for per_prefix in conts2],
                        dtype=torch.float32,
                    )
                    for local_idx, i in enumerate(non_terminal):
                        rollout_logp[i, :] = logp_sums2_t[local_idx]
                        rollout_lengths[i, :] = rollout_lengths2_t[local_idx]
                    total_rollouts += len(non_terminal) * m
                    rollout_token_count = int(sum(len(tokens) for per_prefix in conts2 for tokens in per_prefix))
                    total_rollout_tokens += rollout_token_count
                    total_sampling_tokens += rollout_token_count

            probs = _compute_power_probs(
                logp_items=top_logp,
                item_lengths=top_lengths,
                rollout_logp_sums=rollout_logp,
                rollout_lengths=rollout_lengths,
                alpha=alpha,
                use_jackknife=bool(cfg.use_jackknife),
            )

        chosen = top_blocks[_sample_index(probs, rng)]
        for tok in chosen:
            if (len(seq) - len(context)) >= max_new_tokens:
                break
            seq.append(int(tok))
            steps += 1
            if int(tok) == eos_id:
                break

        if seq and seq[-1] == eos_id:
            break

    internal_sampling_tokens = max(total_sampling_tokens - steps, 0)
    return seq, {
        "steps": float(steps),
        "rollouts": float(total_rollouts),
        "rollout_tokens": float(total_rollout_tokens),
        "candidate_tokens": float(total_candidate_tokens),
        "output_tokens": float(steps),
        "sampling_tokens": float(total_sampling_tokens),
        "internal_sampling_tokens": float(internal_sampling_tokens),
    }
