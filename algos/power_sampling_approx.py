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
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import numpy as np
import torch

from sampling_budget import SamplingBudgetConfig, SamplingBudgetTracker, fit_batched_sampling_plan

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


@dataclass
class ApproxTrajectoryCandidate:
    token_ids: List[int]
    target_log_score: float
    completion_text: str
    answer_key: str
    completed: bool
    trajectory_index: int
    stop_reason: str


@dataclass
class ApproxSingleTrajectoryResult:
    token_ids: List[int]
    target_log_score: float
    steps: int
    rollouts: int
    rollout_tokens: int
    candidate_tokens: int
    stop_reason: str
    candidate: ApproxTrajectoryCandidate


def _normalize_stop_mode(stop_mode: str) -> str:
    normalized = str(stop_mode).strip().lower()
    if normalized in {"eos", "original", "legacy"}:
        return "eos"
    if normalized in {"budget", "budget_driven", "budget-driven"}:
        return "budget"
    raise ValueError(f"Unsupported approximate power stop_mode: {stop_mode}")


def _normalize_budget_strategy(budget_strategy: str) -> str:
    normalized = str(budget_strategy).strip().lower()
    if normalized in {"restart", "restarts", "search"}:
        return "restart"
    raise ValueError(f"Unsupported approximate power budget_strategy: {budget_strategy}")


def _normalize_selection_mode(selection_mode: str) -> str:
    normalized = str(selection_mode).strip().lower()
    if normalized in {"last", "best_logp", "vote", "weighted_vote"}:
        return normalized
    raise ValueError(f"Unsupported approximate power selection_mode: {selection_mode}")


def _canonicalize_answer_key(text: str) -> str:
    return " ".join(str(text).strip().split())


def _extract_answer_key(
    completion_text: str,
    *,
    answer_extractor: Optional[Callable[[str], Any]],
) -> str:
    if answer_extractor is None:
        return _canonicalize_answer_key(completion_text)
    try:
        extracted = answer_extractor(completion_text)
    except Exception:
        extracted = None
    if extracted is None:
        return _canonicalize_answer_key(completion_text)
    extracted_text = _canonicalize_answer_key(str(extracted))
    if extracted_text:
        return extracted_text
    return _canonicalize_answer_key(completion_text)


def _logsumexp(values: List[float]) -> float:
    if not values:
        return -float("inf")
    max_val = max(values)
    if not math.isfinite(max_val):
        return max_val
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


def _decode_completion_text(
    token_ids: List[int],
    *,
    context_len: int,
    decode_tokens: Optional[Callable[[List[int]], str]],
) -> str:
    generated_token_ids = token_ids[context_len:]
    if decode_tokens is None:
        return " ".join(str(int(tok)) for tok in generated_token_ids)
    try:
        return str(decode_tokens(generated_token_ids))
    except Exception:
        return " ".join(str(int(tok)) for tok in generated_token_ids)


def _build_candidate(
    *,
    token_ids: List[int],
    context_len: int,
    trajectory_index: int,
    stop_reason: str,
    target_log_score: float,
    decode_tokens: Optional[Callable[[List[int]], str]],
    answer_extractor: Optional[Callable[[str], Any]],
) -> ApproxTrajectoryCandidate:
    completion_text = _decode_completion_text(
        token_ids,
        context_len=context_len,
        decode_tokens=decode_tokens,
    )
    answer_key = _extract_answer_key(completion_text, answer_extractor=answer_extractor)
    return ApproxTrajectoryCandidate(
        token_ids=list(token_ids),
        target_log_score=float(target_log_score),
        completion_text=completion_text,
        answer_key=answer_key,
        completed=stop_reason == "eos",
        trajectory_index=int(trajectory_index),
        stop_reason=stop_reason,
    )


def _select_candidate(
    *,
    completed_candidates: List[ApproxTrajectoryCandidate],
    fallback_candidates: List[ApproxTrajectoryCandidate],
    selection_mode: str,
) -> Tuple[Optional[ApproxTrajectoryCandidate], str]:
    completed_pool = list(completed_candidates)
    fallback_pool = list(fallback_candidates)
    pool = completed_pool if completed_pool else fallback_pool
    if not pool:
        return None, "none"

    mode = _normalize_selection_mode(selection_mode)
    if mode == "last":
        return pool[-1], "last"

    if mode == "best_logp":
        best = max(pool, key=lambda cand: (cand.target_log_score, int(cand.completed), -cand.trajectory_index))
        return best, "best_logp"

    grouped: Dict[str, List[ApproxTrajectoryCandidate]] = {}
    for cand in pool:
        grouped.setdefault(cand.answer_key, []).append(cand)

    best_group_key = None
    best_group_tuple = None
    for group_key, group in grouped.items():
        group_scores = [cand.target_log_score for cand in group]
        group_best = max(group_scores)
        if mode == "vote":
            group_tuple = (len(group), group_best, -min(cand.trajectory_index for cand in group), group_key)
        else:
            group_tuple = (
                _logsumexp(group_scores),
                len(group),
                group_best,
                -min(cand.trajectory_index for cand in group),
                group_key,
            )
        if best_group_tuple is None or group_tuple > best_group_tuple:
            best_group_tuple = group_tuple
            best_group_key = group_key

    assert best_group_key is not None
    winning_group = grouped[best_group_key]
    best = max(winning_group, key=lambda cand: (cand.target_log_score, int(cand.completed), -cand.trajectory_index))
    return best, mode


def _build_diagnostics(
    *,
    selected_token_ids: List[int],
    context_len: int,
    total_rollouts: int,
    total_rollout_tokens: int,
    total_candidate_tokens: int,
    budget_tracker: SamplingBudgetTracker,
    stop_reason: str,
    stop_mode: str,
    budget_strategy: str,
    selection_mode: str,
    chains_started: int,
    completed_candidates: int,
    selected_candidate_source: str,
) -> Dict[str, Any]:
    output_tokens = max(len(selected_token_ids) - context_len, 0)
    sampling_tokens = int(budget_tracker.sampling_tokens_used)
    diagnostics: Dict[str, Any] = {
        "steps": float(output_tokens),
        "rollouts": float(total_rollouts),
        "rollout_tokens": float(total_rollout_tokens),
        "candidate_tokens": float(total_candidate_tokens),
        "output_tokens": float(output_tokens),
        "sampling_tokens": float(sampling_tokens),
        "internal_sampling_tokens": float(max(sampling_tokens - output_tokens, 0)),
        "budget_stop_reason": stop_reason,
        "approx_stop_mode": stop_mode,
        "approx_budget_strategy": budget_strategy,
        "approx_selection_mode": selection_mode,
        "approx_chains_started": int(chains_started),
        "approx_completed_candidates": int(completed_candidates),
        "approx_selected_candidate_source": selected_candidate_source,
    }
    diagnostics.update(budget_tracker.metadata())
    return diagnostics


def _run_single_approx_trajectory(
    scorer: PowerApproxScorer,
    context: List[int],
    *,
    cfg: PowerSamplerApproxConfig,
    max_new_tokens: int,
    eos_token_id: int,
    rng: np.random.Generator,
    budget_tracker: SamplingBudgetTracker,
    trajectory_index: int,
    decode_tokens: Optional[Callable[[List[int]], str]],
    answer_extractor: Optional[Callable[[str], Any]],
) -> ApproxSingleTrajectoryResult:
    alpha = 1.0 / float(cfg.temp)
    eos_id = int(eos_token_id)
    block_size = max(1, int(cfg.block_size))
    context_len = len(context)
    seq = context.copy()
    steps = 0
    total_rollouts = 0
    total_rollout_tokens = 0
    total_candidate_tokens = 0
    selected_logp_sum = 0.0
    stop_reason = "max_new_tokens"

    max_new_tokens = _clamp_new_tokens(max_new_tokens, len(context), scorer.max_seq_len)
    if max_new_tokens <= 0:
        candidate = _build_candidate(
            token_ids=seq,
            context_len=context_len,
            trajectory_index=trajectory_index,
            stop_reason=stop_reason,
            target_log_score=0.0,
            decode_tokens=decode_tokens,
            answer_extractor=answer_extractor,
        )
        return ApproxSingleTrajectoryResult(
            token_ids=seq,
            target_log_score=0.0,
            steps=steps,
            rollouts=total_rollouts,
            rollout_tokens=total_rollout_tokens,
            candidate_tokens=total_candidate_tokens,
            stop_reason=stop_reason,
            candidate=candidate,
        )

    if block_size == 1:
        while (len(seq) - context_len) < max_new_tokens:
            if budget_tracker.exhausted():
                stop_reason = "sampling_budget_exhausted"
                break

            remaining = max_new_tokens - (len(seq) - context_len)
            if remaining <= 0:
                stop_reason = "max_new_tokens"
                break

            k = max(1, int(cfg.top_k))
            cand_tokens, cand_logps = scorer.topk_next_tokens(seq, k)
            k = len(cand_tokens)
            if k == 0:
                stop_reason = "no_candidates"
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
                    remaining_for_rollouts = budget_tracker.remaining_sampling_tokens()
                    if remaining_for_rollouts is not None:
                        remaining_for_rollouts = max(int(remaining_for_rollouts) - 1, 0)
                    rollout_m, rollout_len = fit_batched_sampling_plan(
                        num_prefixes=len(non_terminal),
                        samples_per_prefix=m,
                        max_tokens_per_sample=rollout_len,
                        remaining_sampling_tokens=remaining_for_rollouts,
                    )
                    if rollout_m > 0 and rollout_len > 0:
                        rollout_logp = torch.zeros((k, rollout_m), dtype=torch.float32)
                        rollout_lengths = torch.ones((k, rollout_m), dtype=torch.float32)
                        conts, logp_sums = scorer.sample_continuations(
                            prefixes,
                            max_new_tokens=rollout_len,
                            n=rollout_m,
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
                        for local_idx, candidate_idx in enumerate(non_terminal):
                            rollout_logp[candidate_idx, :] = logp_sums_t[local_idx]
                            rollout_lengths[candidate_idx, :] = rollout_lengths_t[local_idx]
                        total_rollouts += len(non_terminal) * rollout_m
                        rollout_token_count = int(sum(len(tokens) for per_prefix in conts for tokens in per_prefix))
                        total_rollout_tokens += rollout_token_count
                        budget_tracker.spend(rollout_token_count)

                probs = _compute_power_probs(
                    logp_items=logp_items,
                    item_lengths=item_lengths,
                    rollout_logp_sums=rollout_logp,
                    rollout_lengths=rollout_lengths,
                    alpha=alpha,
                    use_jackknife=bool(cfg.use_jackknife),
                )

            if not budget_tracker.can_spend(1):
                stop_reason = "sampling_budget_exhausted"
                break

            choice = _sample_index(probs, rng)
            token = int(cand_tokens[choice])
            seq.append(token)
            steps += 1
            selected_logp_sum += float(cand_logps[choice])
            budget_tracker.spend(1)
            if token == eos_id:
                stop_reason = "eos"
                break
    else:
        while (len(seq) - context_len) < max_new_tokens:
            if budget_tracker.exhausted():
                stop_reason = "sampling_budget_exhausted"
                break

            remaining = max_new_tokens - (len(seq) - context_len)
            if remaining <= 0:
                stop_reason = "max_new_tokens"
                break

            final_chunk = remaining <= block_size
            requested_pool_size = max(1, int(cfg.candidate_pool_size))
            requested_k = max(1, min(int(cfg.top_k), requested_pool_size))
            requested_chunk_len = _clamp_new_tokens(
                remaining if final_chunk else block_size,
                len(seq),
                scorer.max_seq_len,
            )
            if requested_chunk_len <= 0:
                stop_reason = "max_new_tokens"
                break

            candidate_pool_size, chunk_len = fit_batched_sampling_plan(
                num_prefixes=1,
                samples_per_prefix=requested_pool_size,
                max_tokens_per_sample=requested_chunk_len,
                remaining_sampling_tokens=budget_tracker.remaining_sampling_tokens(),
            )
            if candidate_pool_size <= 0 or chunk_len <= 0:
                stop_reason = "sampling_budget_exhausted"
                break

            conts, logp_sums = scorer.sample_continuations(
                [seq],
                max_new_tokens=chunk_len,
                n=candidate_pool_size,
                temperature=float(cfg.temp),
                top_p=1.0,
                top_k=None,
                eos_token_id=eos_id,
            )
            candidate_blocks = conts[0]
            candidate_logp = np.asarray(logp_sums[0], dtype=np.float64)
            if len(candidate_blocks) == 0:
                stop_reason = "empty_candidate_pool"
                break

            candidate_token_count = int(sum(len(tokens) for tokens in candidate_blocks))
            total_candidate_tokens += candidate_token_count
            budget_tracker.spend(candidate_token_count)

            k = max(1, min(requested_k, len(candidate_blocks)))
            order = np.argsort(candidate_logp)[::-1][:k]
            top_blocks = [candidate_blocks[i] for i in order]
            top_logp = torch.tensor([float(candidate_logp[i]) for i in order], dtype=torch.float32)
            top_lengths = torch.tensor([float(max(len(block), 1)) for block in top_blocks], dtype=torch.float32)

            lookahead = _resolve_lookahead(cfg, remaining - chunk_len)
            m = max(0, int(cfg.rollouts_per_candidate))

            if lookahead <= 0 or m <= 0:
                probs = _softmax_from_logits(alpha * _mean_logp(top_logp, top_lengths))
            else:
                terminal = [eos_id in block for block in top_blocks]
                rollout_logp = torch.zeros((k, m), dtype=torch.float32)
                rollout_lengths = torch.ones((k, m), dtype=torch.float32)
                non_terminal = [i for i, is_terminal in enumerate(terminal) if not is_terminal]

                if non_terminal:
                    prefixes = [seq + [int(t) for t in top_blocks[i]] for i in non_terminal]
                    rollout_len = _clamp_new_tokens(lookahead, len(prefixes[0]), scorer.max_seq_len)
                    rollout_m, rollout_len = fit_batched_sampling_plan(
                        num_prefixes=len(non_terminal),
                        samples_per_prefix=m,
                        max_tokens_per_sample=rollout_len,
                        remaining_sampling_tokens=budget_tracker.remaining_sampling_tokens(),
                    )
                    if rollout_m > 0 and rollout_len > 0:
                        rollout_logp = torch.zeros((k, rollout_m), dtype=torch.float32)
                        rollout_lengths = torch.ones((k, rollout_m), dtype=torch.float32)
                        conts2, logp_sums2 = scorer.sample_continuations(
                            prefixes,
                            max_new_tokens=rollout_len,
                            n=rollout_m,
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
                        for local_idx, candidate_idx in enumerate(non_terminal):
                            rollout_logp[candidate_idx, :] = logp_sums2_t[local_idx]
                            rollout_lengths[candidate_idx, :] = rollout_lengths2_t[local_idx]
                        total_rollouts += len(non_terminal) * rollout_m
                        rollout_token_count = int(sum(len(tokens) for per_prefix in conts2 for tokens in per_prefix))
                        total_rollout_tokens += rollout_token_count
                        budget_tracker.spend(rollout_token_count)

                probs = _compute_power_probs(
                    logp_items=top_logp,
                    item_lengths=top_lengths,
                    rollout_logp_sums=rollout_logp,
                    rollout_lengths=rollout_lengths,
                    alpha=alpha,
                    use_jackknife=bool(cfg.use_jackknife),
                )

            choice = _sample_index(probs, rng)
            chosen = top_blocks[choice]
            selected_logp_sum += float(top_logp[choice].item())
            saw_eos = False
            for tok in chosen:
                if (len(seq) - context_len) >= max_new_tokens:
                    break
                seq.append(int(tok))
                steps += 1
                if int(tok) == eos_id:
                    saw_eos = True
                    break

            if saw_eos:
                stop_reason = "eos"
                break

    candidate = _build_candidate(
        token_ids=seq,
        context_len=context_len,
        trajectory_index=trajectory_index,
        stop_reason=stop_reason,
        target_log_score=alpha * selected_logp_sum,
        decode_tokens=decode_tokens,
        answer_extractor=answer_extractor,
    )
    return ApproxSingleTrajectoryResult(
        token_ids=seq,
        target_log_score=alpha * selected_logp_sum,
        steps=steps,
        rollouts=total_rollouts,
        rollout_tokens=total_rollout_tokens,
        candidate_tokens=total_candidate_tokens,
        stop_reason=stop_reason,
        candidate=candidate,
    )


def _run_budget_restart_search(
    scorer: PowerApproxScorer,
    context: List[int],
    *,
    cfg: PowerSamplerApproxConfig,
    max_new_tokens: int,
    eos_token_id: int,
    rng: np.random.Generator,
    budget_tracker: SamplingBudgetTracker,
    selection_mode: str,
    decode_tokens: Optional[Callable[[List[int]], str]],
    answer_extractor: Optional[Callable[[str], Any]],
) -> Tuple[List[int], Dict[str, Any]]:
    context_len = len(context)
    chains_started = 0
    total_rollouts = 0
    total_rollout_tokens = 0
    total_candidate_tokens = 0
    completed_candidates: List[ApproxTrajectoryCandidate] = []
    fallback_candidates: List[ApproxTrajectoryCandidate] = []
    stop_reason = "sampling_budget_exhausted"

    while not budget_tracker.exhausted():
        chains_started += 1
        before = int(budget_tracker.sampling_tokens_used)
        trajectory = _run_single_approx_trajectory(
            scorer,
            context,
            cfg=cfg,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            rng=rng,
            budget_tracker=budget_tracker,
            trajectory_index=chains_started,
            decode_tokens=decode_tokens,
            answer_extractor=answer_extractor,
        )
        total_rollouts += int(trajectory.rollouts)
        total_rollout_tokens += int(trajectory.rollout_tokens)
        total_candidate_tokens += int(trajectory.candidate_tokens)
        fallback_candidates.append(trajectory.candidate)
        if trajectory.candidate.completed:
            completed_candidates.append(trajectory.candidate)

        spent = int(budget_tracker.sampling_tokens_used) - before
        stop_reason = str(trajectory.stop_reason)
        if spent <= 0:
            break
        if budget_tracker.exhausted():
            stop_reason = "sampling_budget_exhausted"
            break

    selected_candidate, selected_source = _select_candidate(
        completed_candidates=completed_candidates,
        fallback_candidates=fallback_candidates,
        selection_mode=selection_mode,
    )
    if selected_candidate is None:
        selected_candidate = _build_candidate(
            token_ids=context.copy(),
            context_len=context_len,
            trajectory_index=0,
            stop_reason="no_progress",
            target_log_score=0.0,
            decode_tokens=decode_tokens,
            answer_extractor=answer_extractor,
        )
        selected_source = "prompt_fallback"

    diagnostics = _build_diagnostics(
        selected_token_ids=selected_candidate.token_ids,
        context_len=context_len,
        total_rollouts=total_rollouts,
        total_rollout_tokens=total_rollout_tokens,
        total_candidate_tokens=total_candidate_tokens,
        budget_tracker=budget_tracker,
        stop_reason=stop_reason,
        stop_mode="budget",
        budget_strategy="restart",
        selection_mode=selection_mode,
        chains_started=chains_started,
        completed_candidates=len(completed_candidates),
        selected_candidate_source=selected_source,
    )
    return selected_candidate.token_ids, diagnostics


def approx_power_sample(
    scorer: PowerApproxScorer,
    context: List[int],
    *,
    cfg: PowerSamplerApproxConfig,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    stop_mode: str = "eos",
    budget_strategy: str = "restart",
    selection_mode: str = "best_logp",
    answer_extractor: Optional[Callable[[str], Any]] = None,
    decode_tokens: Optional[Callable[[List[int]], str]] = None,
    budget_config: Optional[SamplingBudgetConfig] = None,
) -> Tuple[List[int], Dict[str, Any]]:
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
    stop_mode = _normalize_stop_mode(stop_mode)
    budget_strategy = _normalize_budget_strategy(budget_strategy)
    selection_mode = _normalize_selection_mode(selection_mode)
    budget_tracker = SamplingBudgetTracker(budget_config)

    if stop_mode == "budget" and budget_tracker.max_sampling_tokens is None:
        raise ValueError("Approximate power stop_mode='budget' requires max_sampling_tokens to be set.")

    eos_id = int(eos_token_id)
    max_new_tokens = _clamp_new_tokens(max_new_tokens, len(context), scorer.max_seq_len)
    if stop_mode == "budget" and budget_strategy == "restart":
        return _run_budget_restart_search(
            scorer,
            context,
            cfg=cfg,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            rng=rng,
            budget_tracker=budget_tracker,
            selection_mode=selection_mode,
            decode_tokens=decode_tokens,
            answer_extractor=answer_extractor,
        )

    trajectory = _run_single_approx_trajectory(
        scorer,
        context,
        cfg=cfg,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_id,
        rng=rng,
        budget_tracker=budget_tracker,
        trajectory_index=1,
        decode_tokens=decode_tokens,
        answer_extractor=answer_extractor,
    )
    diagnostics = _build_diagnostics(
        selected_token_ids=trajectory.token_ids,
        context_len=len(context),
        total_rollouts=int(trajectory.rollouts),
        total_rollout_tokens=int(trajectory.rollout_tokens),
        total_candidate_tokens=int(trajectory.candidate_tokens),
        budget_tracker=budget_tracker,
        stop_reason=str(trajectory.stop_reason),
        stop_mode=stop_mode,
        budget_strategy="none" if stop_mode == "eos" else budget_strategy,
        selection_mode="last",
        chains_started=1,
        completed_candidates=1 if trajectory.candidate.completed else 0,
        selected_candidate_source="last",
    )
    return trajectory.token_ids, diagnostics
