"""Cumulant / entropy-rate approximation for power-distribution sampling.

This module implements a *training-free* approximation to sampling from the
global power distribution p^alpha (Karan & Du, 2025) without Monte Carlo
lookahead rollouts of sequence likelihoods (Ji et al., 2026).

Recall the token-level decomposition (Ji et al., 2026, Thm 3.1):

  p_pow_alpha(x_t | prefix) ∝ p(x_t | prefix)^alpha · ζ_t(x_t)
  ζ_t(x_t) = \sum_{x_{t+1:T}} p(x_{t+1:T} | prefix, x_t)^alpha

The scalable sampler in arXiv:2601.21590 estimates ζ_t(x_t) by Monte Carlo
sampling of future trajectories and evaluating their log-likelihood under the
base model. Here we instead approximate log ζ_t(x_t) via a *cumulant expansion*
of the random future log-likelihood under the base model.

Let S = log p(X_{t+1:t+H} | prefix, x_t), where X_{t+1:t+H} ~ p(·|prefix,x_t)
is a (possibly early-terminated) rollout of length H. Then

  ζ_t(x_t) = E[ exp((alpha-1) · S) ].

The log of this quantity is the cumulant generating function (CGF) of S at
lambda = alpha-1:

  log ζ_t(x_t) = K_S(lambda),  K_S(lambda) = log E[exp(lambda S)].

We approximate K_S using the first (and optionally second) cumulants:

  K_S(lambda) ≈ lambda·E[S] + (lambda^2/2)·Var[S].

E[S] is the negative entropy-rate along the rollout, and Var[S] is the
varentropy-rate (sum of per-step variances of log p(next_token)).

This approximation replaces M rollouts-per-candidate (Ji et al., 2026) with a
small number R of *moment rollouts* (often R=1) per candidate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch

DEFAULT_LOOKAHEAD_TOKENS = 192


@dataclass
class PowerSamplerCumulantConfig:
    # Power exponent parameter. We follow convention alpha = 1 / temp.
    temp: float = 0.25
    # Candidate set size Top-K_t for token-level sampling.
    top_k: int = 8
    # For block sampling (B > 1): sample L candidates then keep Top-K.
    candidate_pool_size: int = 32
    # Truncated rollout horizon H_t.
    lookahead_tokens: Optional[int] = DEFAULT_LOOKAHEAD_TOKENS
    # 1 => token-level; >1 => block-level (Appendix-B style).
    block_size: int = DEFAULT_LOOKAHEAD_TOKENS
    # Number of *moment rollouts* per candidate prefix.
    moment_rollouts: int = 1
    # 1 => entropy-rate only; 2 => entropy-rate + varentropy-rate.
    cumulant_order: int = 2
    # Rollout sampling controls (default: true base-model rollouts).
    rollout_temperature: float = 1.0
    rollout_top_p: float = 1.0
    rollout_top_k: Optional[int] = None


class PowerCumulantScorer(Protocol):
    """Backend interface for cumulant power sampling."""

    @property
    def max_seq_len(self) -> Optional[int]:
        ...

    def topk_next_tokens(self, prefix: List[int], k: int) -> Tuple[List[int], List[float]]:
        """Return Top-k token ids and log p(token|prefix) under base model p."""

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

    def sample_continuations_with_moments(
        self,
        prefixes: List[List[int]],
        *,
        max_new_tokens: int,
        n: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[List[List[int]]], List[List[float]], List[List[float]], List[List[float]]]:
        """Sample n continuations per prefix, and return moment sums.

        Returns:
          continuations[prefix_i][sample_j] -> generated token ids
          logp_sums[prefix_i][sample_j] -> sum log p(tokens | prefix)
          mean_logp_sums[prefix_i][sample_j] -> sum_s E_{X~p_s}[log p_s(X)]
          var_logp_sums[prefix_i][sample_j] -> sum_s Var_{X~p_s}[log p_s(X)]
        """


def _clamp_new_tokens(max_new_tokens: int, prompt_len: int, max_seq_len: Optional[int]) -> int:
    if max_seq_len is None:
        return max(0, int(max_new_tokens))
    allowed = int(max_seq_len) - int(prompt_len)
    return max(0, min(int(max_new_tokens), allowed))


def _resolve_lookahead(cfg: PowerSamplerCumulantConfig, remaining: int) -> int:
    cap = cfg.lookahead_tokens
    if cap is None:
        cap = DEFAULT_LOOKAHEAD_TOKENS
    cap_int = max(0, int(cap))
    return min(max(0, int(remaining)), cap_int)


def _softmax_from_logits(logits_1d: torch.Tensor) -> torch.Tensor:
    if logits_1d.numel() == 0:
        return logits_1d
    log_z = torch.logsumexp(logits_1d, dim=0)
    probs = torch.exp(logits_1d - log_z)
    if not torch.all(torch.isfinite(probs)):
        return torch.full_like(logits_1d, 1.0 / float(max(int(logits_1d.numel()), 1)))
    return probs


def _sample_index(probs: torch.Tensor, rng: np.random.Generator) -> int:
    p_np = probs.detach().cpu().numpy().astype(np.float64)
    p_np = np.clip(p_np, 0.0, 1.0)
    total = p_np.sum()
    if not np.isfinite(total) or total <= 0.0:
        p_np = np.ones_like(p_np) / max(len(p_np), 1)
    else:
        p_np /= total
    return int(rng.choice(len(p_np), p=p_np))


def _logmeanexp(x: torch.Tensor, dim: int) -> torch.Tensor:
    """log(mean(exp(x))) along dim."""
    if x.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=x.device)
    m = torch.max(x, dim=dim, keepdim=True).values
    return (m + torch.log(torch.mean(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)


def _compute_log_zeta_from_moments(
    mean_logp_sums: torch.Tensor,  # (K, R)
    var_logp_sums: torch.Tensor,  # (K, R)
    *,
    alpha: float,
    cumulant_order: int,
) -> torch.Tensor:
    """Compute log ζ per candidate from moment rollouts."""
    lam = float(alpha - 1.0)
    log_z = lam * mean_logp_sums
    if int(cumulant_order) >= 2:
        log_z = log_z + 0.5 * (lam * lam) * var_logp_sums
    return _logmeanexp(log_z, dim=1)


def cumulant_power_sample(
    scorer: PowerCumulantScorer,
    context: List[int],
    *,
    cfg: PowerSamplerCumulantConfig,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """Sample a continuation using cumulant / entropy-rate power approximation."""

    if rng is None:
        rng = np.random.default_rng()

    if cfg.temp <= 0:
        raise ValueError("cfg.temp must be > 0 (alpha = 1/cfg.temp).")
    if eos_token_id is None:
        raise ValueError("eos_token_id must be provided.")

    alpha = 1.0 / float(cfg.temp)
    eos_id = int(eos_token_id)
    block_size = max(1, int(cfg.block_size))
    cumulant_order = max(1, int(cfg.cumulant_order))
    r = max(0, int(cfg.moment_rollouts))

    max_new_tokens = _clamp_new_tokens(max_new_tokens, len(context), scorer.max_seq_len)
    if max_new_tokens <= 0:
        return context.copy(), {
            "steps": 0.0,
            "moment_rollouts": 0.0,
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

    # ------------------------------------------------------------------
    # Token-level variant (Algorithm-1 analogue).
    # ------------------------------------------------------------------
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
            lookahead = _resolve_lookahead(cfg, remaining - 1)

            if lookahead <= 0 or r <= 0:
                probs = _softmax_from_logits(alpha * logp_items)
            else:
                log_zeta = torch.zeros((k,), dtype=torch.float32)
                non_terminal = [i for i, tok in enumerate(cand_tokens) if int(tok) != eos_id]

                if non_terminal:
                    prefixes = [seq + [int(cand_tokens[i])] for i in non_terminal]
                    rollout_len = _clamp_new_tokens(lookahead, len(prefixes[0]), scorer.max_seq_len)
                    if rollout_len > 0:
                        conts, _logp_sums, mean_sums, var_sums = scorer.sample_continuations_with_moments(
                            prefixes,
                            max_new_tokens=rollout_len,
                            n=r,
                            temperature=float(cfg.rollout_temperature),
                            top_p=float(cfg.rollout_top_p),
                            top_k=cfg.rollout_top_k,
                            eos_token_id=eos_id,
                        )
                        mean_t = torch.as_tensor(mean_sums, dtype=torch.float32)
                        var_t = torch.as_tensor(var_sums, dtype=torch.float32)
                        log_z_local = _compute_log_zeta_from_moments(
                            mean_t,
                            var_t,
                            alpha=alpha,
                            cumulant_order=cumulant_order,
                        )
                        for local_idx, i in enumerate(non_terminal):
                            log_zeta[i] = log_z_local[local_idx]

                        total_rollouts += len(non_terminal) * r
                        rollout_token_count = int(sum(len(tokens) for per_prefix in conts for tokens in per_prefix))
                        total_rollout_tokens += rollout_token_count
                        total_sampling_tokens += rollout_token_count

                log_num = alpha * logp_items + log_zeta
                probs = _softmax_from_logits(log_num)

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
            "moment_rollouts": float(total_rollouts),
            "rollout_tokens": float(total_rollout_tokens),
            "candidate_tokens": float(total_candidate_tokens),
            "output_tokens": float(steps),
            "sampling_tokens": float(total_sampling_tokens),
            "internal_sampling_tokens": float(internal_sampling_tokens),
        }

    # ------------------------------------------------------------------
    # Block-level variant (Appendix-B analogue).
    # ------------------------------------------------------------------
    while (len(seq) - len(context)) < max_new_tokens:
        remaining = max_new_tokens - (len(seq) - len(context))
        if remaining <= 0:
            break

        # Final chunk: fall back to low-temperature block distribution.
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
                temperature=1.0,
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
            probs = _softmax_from_logits(alpha * top_logp)

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

        # Draw L candidate blocks from the base model.
        conts, logp_sums = scorer.sample_continuations(
            [seq],
            max_new_tokens=block_len,
            n=l,
            temperature=1.0,
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

        lookahead = _resolve_lookahead(cfg, remaining - block_len)
        if lookahead <= 0 or r <= 0:
            probs = _softmax_from_logits(alpha * top_logp)
        else:
            log_zeta = torch.zeros((k,), dtype=torch.float32)
            terminal = [eos_id in block for block in top_blocks]
            non_terminal = [i for i, term in enumerate(terminal) if not term]
            if non_terminal:
                prefixes = [seq + [int(t) for t in top_blocks[i]] for i in non_terminal]
                rollout_len = _clamp_new_tokens(lookahead, len(prefixes[0]), scorer.max_seq_len)
                if rollout_len > 0:
                    conts2, _logp_sums2, mean_sums2, var_sums2 = scorer.sample_continuations_with_moments(
                        prefixes,
                        max_new_tokens=rollout_len,
                        n=r,
                        temperature=float(cfg.rollout_temperature),
                        top_p=float(cfg.rollout_top_p),
                        top_k=cfg.rollout_top_k,
                        eos_token_id=eos_id,
                    )
                    mean_t = torch.as_tensor(mean_sums2, dtype=torch.float32)
                    var_t = torch.as_tensor(var_sums2, dtype=torch.float32)
                    log_z_local = _compute_log_zeta_from_moments(
                        mean_t,
                        var_t,
                        alpha=alpha,
                        cumulant_order=cumulant_order,
                    )
                    for local_idx, i in enumerate(non_terminal):
                        log_zeta[i] = log_z_local[local_idx]
                    total_rollouts += len(non_terminal) * r
                    rollout_token_count = int(sum(len(tokens) for per_prefix in conts2 for tokens in per_prefix))
                    total_rollout_tokens += rollout_token_count
                    total_sampling_tokens += rollout_token_count
            log_num = alpha * top_logp + log_zeta
            probs = _softmax_from_logits(log_num)

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
        "moment_rollouts": float(total_rollouts),
        "rollout_tokens": float(total_rollout_tokens),
        "candidate_tokens": float(total_candidate_tokens),
        "output_tokens": float(steps),
        "sampling_tokens": float(total_sampling_tokens),
        "internal_sampling_tokens": float(internal_sampling_tokens),
    }
