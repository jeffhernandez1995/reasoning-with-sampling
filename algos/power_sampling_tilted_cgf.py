"""Tilted-CGF / saddlepoint-inspired approximation for power-distribution sampling.

This module implements an *importance-sampling* estimator of the power-distribution
future factor ζ_t(x_t) using per-step cumulant generating functions (CGFs).

Background
----------

For an autoregressive model p, the (token-level) power distribution with exponent
alpha (> 1) has the decomposition (Ji et al., 2026, Thm. 3.1):

  p_pow_alpha(x_t | prefix) ∝ p(x_t | prefix)^alpha · ζ_t(x_t)
  ζ_t(x_t) = \sum_{x_{t+1:T}} p(x_{t+1:T} | prefix, x_t)^alpha

The scalable sampler in arXiv:2601.21590 estimates ζ_t(x_t) via Monte Carlo rollouts
from the *base* model p and weights exp((alpha-1) * log p(rollout)).

Here we use a standard rare-event / exponential-tilting trick to reduce estimator
variance and expose a family of proposal distributions indexed by an exponent β.

Let target exponent be α and proposal exponent be β (β >= 1). Define the *per-step
tilted* proposal q_β by sampling each next token from:

  q_β(v | h) = p(v | h)^β / Z_β(h),
  Z_β(h) = \sum_u p(u | h)^β.

Then for a rollout y = (y_1, ..., y_L) (terminated early by EOS or by horizon H),
the importance weight for estimating ζ is:

  w(y) = p(y | h)^α / q_β(y | h)
       = \prod_{j=1}^L Z_β(h_j) · p(y_j | h_j)^{α-β}.

Taking logs:

  log w(y) = \sum_{j=1}^L [ log Z_β(h_j) + (α-β) log p(y_j | h_j) ].

Special cases:
  - β = 1 recovers the vanilla Monte Carlo estimator from rollouts under p.
  - β = α corresponds to sampling rollouts under low-temperature decoding (p^α)
    and the weight simplifies to \prod_j Z_α(h_j), i.e. it depends only on per-step
    CGFs log Z_α(h).

We estimate ζ with R proposal rollouts per candidate prefix:

  ζ_hat = (1/R) \sum_{r=1}^R exp(log w_r),
  log ζ_hat = logmeanexp(log w_1, ..., log w_R).

This provides a *tilted CGF* approximation that is often more numerically stable
than exp((α-1) * log p(rollout)) and can work with small R.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch


DEFAULT_LOOKAHEAD_TOKENS = 192


@dataclass
class PowerSamplerTiltedCGFConfig:
    # Target power exponent parameter, encoded as temp=1/alpha.
    temp: float = 0.25
    # Candidate set size Top-K_t.
    top_k: int = 8
    # For block sampling (B > 1): sample L candidates then keep Top-K.
    candidate_pool_size: int = 32
    # Truncated rollout horizon H_t.
    lookahead_tokens: Optional[int] = DEFAULT_LOOKAHEAD_TOKENS
    # 1 => token-level; >1 => block-level (Appendix-B style).
    block_size: int = DEFAULT_LOOKAHEAD_TOKENS
    # Proposal rollouts per candidate (R).
    rollouts_per_candidate: int = 1

    # Proposal temperature. If None, uses `temp` (i.e., β = α). If set to 1.0,
    # β=1 and the method degenerates to vanilla MC weighting.
    proposal_temperature: Optional[float] = None
    proposal_top_p: float = 1.0
    proposal_top_k: Optional[int] = None

    # Optional damping on log ζ when forming token weights.
    # zeta_weight=0.0 => low-temperature sampling.
    zeta_weight: float = 1.0

    # Optional length normalization for block scoring (useful if EOS truncates blocks).
    length_normalize_logp: bool = False
    length_penalty: float = 1.0

    # Whether proposal rollouts should stop on EOS. If False, EOS is treated as a
    # regular token for the purposes of rollouts (fixed-length lookahead).
    rollout_stop_on_eos: bool = True


class PowerTiltedCGFScorer(Protocol):
    @property
    def max_seq_len(self) -> Optional[int]:
        ...

    def topk_next_tokens(self, prefix: List[int], k: int) -> Tuple[List[int], List[float]]:
        ...

    def sample_continuations_with_cgf(
        self,
        prefixes: List[List[int]],
        *,
        max_new_tokens: int,
        n: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> Tuple[List[List[List[int]]], List[List[float]], List[List[float]]]:
        """Sample n continuations per prefix, returning base logp sums and CGF sums.

        Returns:
          continuations[prefix_i][sample_j] -> generated token ids
          logp_sums[prefix_i][sample_j] -> sum log p_base(tokens | prefix)
          logz_sums[prefix_i][sample_j] -> sum log Z_β(history_step), where β=1/temperature
        """


def _clamp_new_tokens(max_new_tokens: int, prompt_len: int, max_seq_len: Optional[int]) -> int:
    if max_seq_len is None:
        return max(0, int(max_new_tokens))
    allowed = int(max_seq_len) - int(prompt_len)
    return max(0, min(int(max_new_tokens), allowed))


def _resolve_lookahead(cfg: PowerSamplerTiltedCGFConfig, remaining: int) -> int:
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
    if x.numel() == 0:
        return torch.empty((0,), dtype=torch.float32, device=x.device)
    m = torch.max(x, dim=dim, keepdim=True).values
    return (m + torch.log(torch.mean(torch.exp(x - m), dim=dim, keepdim=True))).squeeze(dim)


def _compute_log_zeta_from_tilted_rollouts(
    *,
    logp_sums: torch.Tensor,  # (K, R)
    logz_sums: torch.Tensor,  # (K, R)
    alpha: float,
    beta: float,
) -> torch.Tensor:
    """Compute log ζ per candidate from proposal rollouts.

    Each rollout returns:
      - logp_sums: \sum log p_base(token | history)
      - logz_sums: \sum log Z_β(history), Z_β = \sum_v p_base(v|h)^β

    Importance weight:
      log w = logz_sums + (α-β) * logp_sums
    ζ_hat = mean(exp(log w))
    log ζ_hat = logmeanexp(log w).
    """

    logw = logz_sums + float(alpha - beta) * logp_sums
    return _logmeanexp(logw, dim=1)


def tilted_cgf_power_sample(
    scorer: PowerTiltedCGFScorer,
    context: List[int],
    *,
    cfg: PowerSamplerTiltedCGFConfig,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], Dict[str, float]]:
    """Sample a continuation using the tilted-CGF estimator of ζ."""

    if rng is None:
        rng = np.random.default_rng()
    if cfg.temp <= 0:
        raise ValueError("cfg.temp must be > 0 (alpha = 1/cfg.temp).")
    if eos_token_id is None:
        raise ValueError("eos_token_id must be provided.")

    alpha = 1.0 / float(cfg.temp)
    proposal_temp = float(cfg.proposal_temperature) if cfg.proposal_temperature is not None else float(cfg.temp)
    if proposal_temp <= 0:
        raise ValueError("proposal_temperature must be > 0.")
    beta = 1.0 / proposal_temp

    eos_id = int(eos_token_id)
    block_size = max(1, int(cfg.block_size))
    r = max(0, int(cfg.rollouts_per_candidate))
    zeta_weight = float(cfg.zeta_weight)
    length_normalize_logp = bool(cfg.length_normalize_logp)
    length_penalty = float(cfg.length_penalty)
    rollout_stop_on_eos = bool(cfg.rollout_stop_on_eos)

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

    # -------------------------------
    # Token-level variant.
    # -------------------------------
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
                        conts, logp_sums, logz_sums = scorer.sample_continuations_with_cgf(
                            prefixes,
                            max_new_tokens=rollout_len,
                            n=r,
                            temperature=proposal_temp,
                            top_p=float(cfg.proposal_top_p),
                            top_k=cfg.proposal_top_k,
                            eos_token_id=eos_id if rollout_stop_on_eos else None,
                        )
                        logp_t = torch.as_tensor(logp_sums, dtype=torch.float32)
                        logz_t = torch.as_tensor(logz_sums, dtype=torch.float32)
                        log_z_local = _compute_log_zeta_from_tilted_rollouts(
                            logp_sums=logp_t,
                            logz_sums=logz_t,
                            alpha=alpha,
                            beta=beta,
                        )
                        for local_idx, i in enumerate(non_terminal):
                            log_zeta[i] = log_z_local[local_idx]
                        total_rollouts += len(non_terminal) * r
                        rollout_token_count = int(sum(len(tokens) for per_prefix in conts for tokens in per_prefix))
                        total_rollout_tokens += rollout_token_count
                        total_sampling_tokens += rollout_token_count

                log_num = alpha * logp_items + zeta_weight * log_zeta
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
            "rollouts": float(total_rollouts),
            "rollout_tokens": float(total_rollout_tokens),
            "candidate_tokens": float(total_candidate_tokens),
            "output_tokens": float(steps),
            "sampling_tokens": float(total_sampling_tokens),
            "internal_sampling_tokens": float(internal_sampling_tokens),
        }

    # -------------------------------
    # Block-level variant.
    # -------------------------------
    while (len(seq) - len(context)) < max_new_tokens:
        remaining = max_new_tokens - (len(seq) - len(context))
        if remaining <= 0:
            break

        # Final chunk: fall back to low-temperature distribution over sampled blocks.
        if remaining <= block_size:
            l = max(1, int(cfg.candidate_pool_size))
            k = max(1, min(int(cfg.top_k), l))
            chunk_len = _clamp_new_tokens(remaining, len(seq), scorer.max_seq_len)
            if chunk_len <= 0:
                break

            conts, logp_sums, _logz = scorer.sample_continuations_with_cgf(
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

            if length_normalize_logp:
                lengths = np.asarray([max(len(b), 1) for b in candidate_blocks], dtype=np.float64)
                scores = candidate_logp / np.power(lengths, max(length_penalty, 1e-6))
                order = np.argsort(scores)[::-1][:k]
                top_scores = [float(scores[i]) for i in order]
            else:
                order = np.argsort(candidate_logp)[::-1][:k]
                top_scores = [float(candidate_logp[i]) for i in order]

            top_blocks = [candidate_blocks[i] for i in order]
            top_logp = torch.tensor(top_scores, dtype=torch.float32)
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

        # Candidate blocks from base model.
        conts, logp_sums, _logz = scorer.sample_continuations_with_cgf(
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

        if length_normalize_logp:
            lengths = np.asarray([max(len(b), 1) for b in candidate_blocks], dtype=np.float64)
            scores = candidate_logp / np.power(lengths, max(length_penalty, 1e-6))
            order = np.argsort(scores)[::-1][:k]
            top_scores = [float(scores[i]) for i in order]
        else:
            order = np.argsort(candidate_logp)[::-1][:k]
            top_scores = [float(candidate_logp[i]) for i in order]

        top_blocks = [candidate_blocks[i] for i in order]
        top_logp = torch.tensor(top_scores, dtype=torch.float32)

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
                    conts2, logp_sums2, logz_sums2 = scorer.sample_continuations_with_cgf(
                        prefixes,
                        max_new_tokens=rollout_len,
                        n=r,
                        temperature=proposal_temp,
                        top_p=float(cfg.proposal_top_p),
                        top_k=cfg.proposal_top_k,
                        eos_token_id=eos_id if rollout_stop_on_eos else None,
                    )
                    logp_t = torch.as_tensor(logp_sums2, dtype=torch.float32)
                    logz_t = torch.as_tensor(logz_sums2, dtype=torch.float32)
                    log_z_local = _compute_log_zeta_from_tilted_rollouts(
                        logp_sums=logp_t,
                        logz_sums=logz_t,
                        alpha=alpha,
                        beta=beta,
                    )
                    for local_idx, i in enumerate(non_terminal):
                        log_zeta[i] = log_z_local[local_idx]
                    total_rollouts += len(non_terminal) * r
                    rollout_token_count = int(sum(len(tokens) for per_prefix in conts2 for tokens in per_prefix))
                    total_rollout_tokens += rollout_token_count
                    total_sampling_tokens += rollout_token_count

            log_num = alpha * top_logp + zeta_weight * log_zeta
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

    return seq, {
        "steps": float(steps),
        "rollouts": float(total_rollouts),
        "rollout_tokens": float(total_rollout_tokens),
        "candidate_tokens": float(total_candidate_tokens),
        "output_tokens": float(steps),
        "sampling_tokens": float(total_sampling_tokens),
        "internal_sampling_tokens": float(max(total_sampling_tokens - steps, 0)),
    }
