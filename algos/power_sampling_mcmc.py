"""Regular (MCMC) power sampling algorithm.

This is the MCMC trajectory sampler used for approximate sampling from p^alpha
as implemented in the original project pipeline. It mirrors the previous
implementation in `power_samp_utils.py` so existing behavior is preserved.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from sampling_budget import SamplingBudgetConfig, SamplingBudgetTracker


class AutoregressiveSampler:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.block_size = self.model.config.max_position_embeddings

    @torch.no_grad()
    def next_token(self, prefix):
        torch_prefix = torch.tensor([prefix], dtype=torch.long, device=self.device)
        prefix_cond = torch_prefix if torch_prefix.size(1) <= self.block_size else torch_prefix[:, -self.block_size :]
        logits = self.model(prefix_cond).logits[0, -1, :]
        probs = F.softmax(logits, dim=-1)
        return torch.log(probs)


@dataclass
class MCMCCandidate:
    token_ids: List[int]
    log_probs_norm: List[float]
    log_probs_unnorm: List[float]
    target_log_score: float
    completion_text: str
    answer_key: str
    completed: bool
    chain_index: int
    chain_stop_reason: str


@dataclass
class SingleChainMCMCResult:
    generated: List[int]
    log_probs_norm: List[float]
    log_probs_unnorm: List[float]
    attempts: int
    acceptances: int
    growth_rounds: int
    stop_reason: str
    candidate: Optional[MCMCCandidate]


def naive_temp(
    p: AutoregressiveSampler,
    context: List[int],
    temp: float,
    seq_len: int,
) -> Tuple[List[int], List[float], List[float]]:
    """Low-temperature proposal distribution sampler."""
    c = len(context)
    input_ids = torch.tensor([context], dtype=torch.long, device=p.device)
    output = p.model.generate(
        input_ids=input_ids,
        max_new_tokens=seq_len - c,
        do_sample=True,
        temperature=temp,
        eos_token_id=p.tokenizer.eos_token_id,
        pad_token_id=p.tokenizer.eos_token_id,
        return_dict_in_generate=True,
        output_scores=True,
        output_logits=True,
    )
    unscaled_logits = torch.stack(output.logits, dim=0)
    scaled_logits = torch.stack(output.scores, dim=0)
    tokens = output.sequences[0][c:]
    proposal = output.sequences[0].tolist()

    assert len(tokens) == unscaled_logits.shape[0] == scaled_logits.shape[0]

    idx = tokens.view(unscaled_logits.shape[0], 1, 1)
    log_probs_unnorm = (
        (1 / temp) * torch.gather(F.log_softmax(unscaled_logits, dim=-1), -1, idx)
    ).view(-1).tolist()
    log_probs_norm = torch.gather(F.log_softmax(scaled_logits, dim=-1), -1, idx).view(-1).tolist()

    assert len(tokens) == len(log_probs_unnorm) == len(log_probs_norm)
    return proposal, log_probs_norm, log_probs_unnorm


def _log_sampler_setup(method_name, temp, max_new_tokens, block_num, jump_size, mcmc_steps, context_len):
    alpha = float("inf") if temp == 0 else 1 / temp
    print(
        f"[{method_name}] setup: temp={temp} alpha={alpha} "
        f"max_new_tokens={max_new_tokens} block_num={block_num} "
        f"jump_size={jump_size} mcmc_steps={mcmc_steps} context_len={context_len}",
        flush=True,
    )


def _log_block_summary(method_name, block_idx, block_num, attempts, acceptances, context_len, seq_len, max_new_tokens):
    acceptance_ratio = acceptances / max(attempts, 1)
    generated_tokens = max(seq_len - context_len, 0)
    print(
        f"[{method_name}] block={block_idx}/{block_num} "
        f"generated={generated_tokens}/{max_new_tokens} "
        f"attempts={attempts} accepts={acceptances} acceptance_ratio={acceptance_ratio:.3f}",
        flush=True,
    )


def _normalize_stop_mode(stop_mode: str) -> str:
    normalized = str(stop_mode).strip().lower()
    if normalized in {"eos", "original", "legacy"}:
        return "eos"
    if normalized in {"budget", "budget_driven", "budget-driven"}:
        return "budget"
    raise ValueError(f"Unsupported MCMC stop_mode: {stop_mode}")


def _normalize_budget_strategy(budget_strategy: str) -> str:
    normalized = str(budget_strategy).strip().lower()
    if normalized in {"refine", "refinement"}:
        return "refine"
    if normalized in {"restart", "restarts", "search"}:
        return "restart"
    raise ValueError(f"Unsupported MCMC budget_strategy: {budget_strategy}")


def _normalize_selection_mode(selection_mode: str) -> str:
    normalized = str(selection_mode).strip().lower()
    if normalized in {"last", "best_logp", "vote", "weighted_vote"}:
        return normalized
    raise ValueError(f"Unsupported MCMC selection_mode: {selection_mode}")


def _truncate_at_first_eos(
    generated: List[int],
    log_probs_norm: List[float],
    log_probs_unnorm: List[float],
    *,
    context_len: int,
    eos_token_id: Optional[int],
) -> Tuple[List[int], List[float], List[float]]:
    if eos_token_id is None or eos_token_id not in generated:
        return generated, log_probs_norm, log_probs_unnorm
    eos_idx = generated.index(int(eos_token_id))
    tail_len = max(eos_idx + 1 - int(context_len), 0)
    return generated[: eos_idx + 1], log_probs_norm[:tail_len], log_probs_unnorm[:tail_len]


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


def _build_candidate(
    *,
    tokenizer,
    generated: List[int],
    log_probs_norm: List[float],
    log_probs_unnorm: List[float],
    context_len: int,
    chain_index: int,
    chain_stop_reason: str,
    answer_extractor: Optional[Callable[[str], Any]],
) -> MCMCCandidate:
    generated_token_ids = generated[context_len:]
    completion_text = tokenizer.decode(generated_token_ids, skip_special_tokens=True)
    answer_key = _extract_answer_key(completion_text, answer_extractor=answer_extractor)
    return MCMCCandidate(
        token_ids=list(generated),
        log_probs_norm=list(log_probs_norm),
        log_probs_unnorm=list(log_probs_unnorm),
        target_log_score=float(sum(log_probs_unnorm)),
        completion_text=completion_text,
        answer_key=answer_key,
        completed=chain_stop_reason == "eos",
        chain_index=int(chain_index),
        chain_stop_reason=chain_stop_reason,
    )


def _select_candidate(
    *,
    completed_candidates: List[MCMCCandidate],
    fallback_candidates: List[MCMCCandidate],
    selection_mode: str,
) -> Tuple[Optional[MCMCCandidate], str]:
    completed_pool = list(completed_candidates)
    fallback_pool = list(fallback_candidates)
    pool = completed_pool if completed_pool else fallback_pool
    if not pool:
        return None, "none"

    mode = _normalize_selection_mode(selection_mode)
    if mode == "last":
        return pool[-1], "last"

    if mode == "best_logp":
        best = max(pool, key=lambda cand: (cand.target_log_score, int(cand.completed), -cand.chain_index))
        return best, "best_logp"

    grouped: Dict[str, List[MCMCCandidate]] = {}
    for cand in pool:
        grouped.setdefault(cand.answer_key, []).append(cand)

    best_group_key = None
    best_group_tuple = None
    for group_key, group in grouped.items():
        group_scores = [cand.target_log_score for cand in group]
        group_best = max(group_scores)
        if mode == "vote":
            group_tuple = (len(group), group_best, -min(cand.chain_index for cand in group), group_key)
        else:
            group_tuple = (_logsumexp(group_scores), len(group), group_best, -min(cand.chain_index for cand in group), group_key)
        if best_group_tuple is None or group_tuple > best_group_tuple:
            best_group_tuple = group_tuple
            best_group_key = group_key

    assert best_group_key is not None
    winning_group = grouped[best_group_key]
    best = max(winning_group, key=lambda cand: (cand.target_log_score, int(cand.completed), -cand.chain_index))
    return best, mode


def _run_mcmc_proposal_step(
    *,
    p: AutoregressiveSampler,
    generated: List[int],
    log_probs_norm: List[float],
    log_probs_unnorm: List[float],
    context_len: int,
    temp: float,
    budget_tracker: SamplingBudgetTracker,
) -> Tuple[List[int], List[float], List[float], bool, bool, Optional[str]]:
    if budget_tracker.exhausted():
        return generated, log_probs_norm, log_probs_unnorm, False, False, "sampling_budget_exhausted"

    if len(generated) <= context_len:
        return generated, log_probs_norm, log_probs_unnorm, False, False, "no_generated_tokens"

    t = len(generated)
    idx_min = context_len
    remaining_budget = budget_tracker.remaining_sampling_tokens()
    if remaining_budget is not None:
        idx_min = max(context_len, t - int(remaining_budget))
    if idx_min > t - 1:
        return generated, log_probs_norm, log_probs_unnorm, False, False, "sampling_budget_exhausted"

    idx = random.randint(idx_min, t - 1)
    proposal, log_prob_prop, target_log_prob_prop = naive_temp(p, generated[:idx], temp=temp, seq_len=t)
    if len(log_prob_prop) <= 0:
        return generated, log_probs_norm, log_probs_unnorm, False, False, "empty_proposal"

    budget_tracker.spend(len(log_prob_prop))
    s = len(proposal)
    assert len(log_prob_prop) == s - idx
    assert len(target_log_prob_prop) == s - idx

    log_prob_cur = log_probs_norm.copy()[idx - context_len : s - context_len]
    target_log_prob_cur = log_probs_unnorm.copy()[idx - context_len : s - context_len]
    log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

    accepted = False
    if log_r >= 0.0 or random.random() < math.exp(log_r):
        accepted = True
        generated = proposal.copy()
        log_probs_norm[idx - context_len :] = log_prob_prop.copy()
        log_probs_unnorm[idx - context_len :] = target_log_prob_prop.copy()

    return generated, log_probs_norm, log_probs_unnorm, True, accepted, None


def _build_diagnostics(
    *,
    generated: List[int],
    context_len: int,
    acceptance_ratio: float,
    budget_tracker: SamplingBudgetTracker,
    stop_reason: Optional[str],
    block_num: int,
    mcmc_steps: int,
    stop_mode: str,
    proposal_attempts: int,
    proposal_acceptances: int,
    growth_rounds: int,
    refinement_rounds: int,
    budget_strategy: str = "none",
    selection_mode: str = "last",
    chains_started: int = 1,
    completed_candidates: int = 0,
    selected_candidate_source: str = "last",
) -> Dict[str, Any]:
    output_tokens = max(len(generated) - context_len, 0)
    sampling_tokens = int(budget_tracker.sampling_tokens_used)
    diagnostics: Dict[str, Any] = {
        "output_tokens": float(output_tokens),
        "sampling_tokens": float(sampling_tokens),
        "internal_sampling_tokens": float(max(sampling_tokens - output_tokens, 0)),
        "acceptance_ratio": float(acceptance_ratio),
        "budget_stop_reason": stop_reason,
        "mcmc_block_num": int(block_num),
        "mcmc_steps": int(mcmc_steps),
        "mcmc_stop_mode": stop_mode,
        "mcmc_budget_strategy": budget_strategy,
        "mcmc_selection_mode": selection_mode,
        "mcmc_proposal_attempts": int(proposal_attempts),
        "mcmc_proposal_acceptances": int(proposal_acceptances),
        "mcmc_growth_rounds": int(growth_rounds),
        "mcmc_refinement_rounds": int(refinement_rounds),
        "mcmc_chains_started": int(chains_started),
        "mcmc_completed_candidates": int(completed_candidates),
        "mcmc_selected_candidate_source": selected_candidate_source,
    }
    diagnostics.update(budget_tracker.metadata())
    return diagnostics


def _run_single_chain_to_terminal(
    *,
    p: AutoregressiveSampler,
    context: List[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int,
    budget_tracker: SamplingBudgetTracker,
    chain_index: int,
    answer_extractor: Optional[Callable[[str], Any]],
) -> SingleChainMCMCResult:
    c = len(context)
    generated = context.copy()
    log_probs_norm: List[float] = []
    log_probs_unnorm: List[float] = []
    attempts = 0
    acceptances = 0
    growth_rounds = 0
    stop_reason = "block_num_completed"
    jump_size = 0 if max_new_tokens == 0 else max(1, int(math.ceil(max_new_tokens / block_num)))
    eos_token_id = p.tokenizer.eos_token_id

    if max_new_tokens == 0:
        stop_reason = "max_new_tokens"
        candidate = _build_candidate(
            tokenizer=p.tokenizer,
            generated=generated,
            log_probs_norm=log_probs_norm,
            log_probs_unnorm=log_probs_unnorm,
            context_len=c,
            chain_index=chain_index,
            chain_stop_reason=stop_reason,
            answer_extractor=answer_extractor,
        )
        return SingleChainMCMCResult(
            generated=generated,
            log_probs_norm=log_probs_norm,
            log_probs_unnorm=log_probs_unnorm,
            attempts=attempts,
            acceptances=acceptances,
            growth_rounds=growth_rounds,
            stop_reason=stop_reason,
            candidate=candidate,
        )

    for block_idx in tqdm(range(block_num), desc=f"power_samp chain {chain_index} blocks", leave=False):
        current_output_tokens = max(len(generated) - c, 0)
        remaining_output_tokens = max(max_new_tokens - current_output_tokens, 0)
        if remaining_output_tokens <= 0:
            stop_reason = "max_new_tokens"
            break

        extension_tokens = min(jump_size, remaining_output_tokens)
        extension_tokens = budget_tracker.clamp_sampling_tokens(extension_tokens)
        if extension_tokens <= 0:
            stop_reason = "sampling_budget_exhausted"
            break

        generated, lp_norm, lp_unnorm = naive_temp(
            p,
            generated,
            temp=temp,
            seq_len=len(generated) + extension_tokens,
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)
        budget_tracker.spend(len(lp_norm))
        growth_rounds += 1

        for _ in tqdm(range(mcmc_steps), desc=f"power_samp chain {chain_index} block {block_idx + 1}/{block_num}", leave=False):
            generated, log_probs_norm, log_probs_unnorm, attempted, accepted, proposal_stop_reason = _run_mcmc_proposal_step(
                p=p,
                generated=generated,
                log_probs_norm=log_probs_norm,
                log_probs_unnorm=log_probs_unnorm,
                context_len=c,
                temp=temp,
                budget_tracker=budget_tracker,
            )
            if proposal_stop_reason is not None:
                stop_reason = proposal_stop_reason
                break
            if attempted:
                attempts += 1
            if accepted:
                acceptances += 1

        _log_block_summary("power_samp", block_idx + 1, block_num, attempts, acceptances, c, len(generated), max_new_tokens)

        if eos_token_id is not None and int(eos_token_id) in generated:
            generated, log_probs_norm, log_probs_unnorm = _truncate_at_first_eos(
                generated,
                log_probs_norm,
                log_probs_unnorm,
                context_len=c,
                eos_token_id=eos_token_id,
            )
            stop_reason = "eos"
            break

        if stop_reason not in {"block_num_completed"}:
            break

        if budget_tracker.exhausted():
            stop_reason = "sampling_budget_exhausted"
            break
    else:
        stop_reason = "block_num_completed"

    candidate = _build_candidate(
        tokenizer=p.tokenizer,
        generated=generated,
        log_probs_norm=log_probs_norm,
        log_probs_unnorm=log_probs_unnorm,
        context_len=c,
        chain_index=chain_index,
        chain_stop_reason=stop_reason,
        answer_extractor=answer_extractor,
    )
    return SingleChainMCMCResult(
        generated=generated,
        log_probs_norm=log_probs_norm,
        log_probs_unnorm=log_probs_unnorm,
        attempts=attempts,
        acceptances=acceptances,
        growth_rounds=growth_rounds,
        stop_reason=stop_reason,
        candidate=candidate,
    )


def _run_budget_restart_search(
    *,
    p: AutoregressiveSampler,
    context: List[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int,
    budget_tracker: SamplingBudgetTracker,
    selection_mode: str,
    answer_extractor: Optional[Callable[[str], Any]],
) -> Tuple[List[int], List[float], List[float], float, Dict[str, Any]]:
    c = len(context)
    attempts = 0
    acceptances = 0
    growth_rounds = 0
    refinement_rounds = 0
    chains_started = 0
    completed_candidates: List[MCMCCandidate] = []
    fallback_candidates: List[MCMCCandidate] = []
    stop_reason = "sampling_budget_exhausted"

    while not budget_tracker.exhausted():
        chains_started += 1
        before = int(budget_tracker.sampling_tokens_used)
        chain_result = _run_single_chain_to_terminal(
            p=p,
            context=context,
            temp=temp,
            mcmc_steps=mcmc_steps,
            max_new_tokens=max_new_tokens,
            block_num=block_num,
            budget_tracker=budget_tracker,
            chain_index=chains_started,
            answer_extractor=answer_extractor,
        )
        attempts += int(chain_result.attempts)
        acceptances += int(chain_result.acceptances)
        growth_rounds += int(chain_result.growth_rounds)

        if chain_result.candidate is not None:
            fallback_candidates.append(chain_result.candidate)
            if chain_result.candidate.completed:
                completed_candidates.append(chain_result.candidate)

        spent = int(budget_tracker.sampling_tokens_used) - before
        stop_reason = str(chain_result.stop_reason)
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
            tokenizer=p.tokenizer,
            generated=context.copy(),
            log_probs_norm=[],
            log_probs_unnorm=[],
            context_len=c,
            chain_index=0,
            chain_stop_reason="no_progress",
            answer_extractor=answer_extractor,
        )
        selected_source = "prompt_fallback"

    acceptance_ratio = acceptances / max(attempts, 1)
    diagnostics = _build_diagnostics(
        generated=selected_candidate.token_ids,
        context_len=c,
        acceptance_ratio=acceptance_ratio,
        budget_tracker=budget_tracker,
        stop_reason=stop_reason,
        block_num=block_num,
        mcmc_steps=mcmc_steps,
        stop_mode="budget",
        proposal_attempts=attempts,
        proposal_acceptances=acceptances,
        growth_rounds=growth_rounds,
        refinement_rounds=refinement_rounds,
        budget_strategy="restart",
        selection_mode=selection_mode,
        chains_started=chains_started,
        completed_candidates=len(completed_candidates),
        selected_candidate_source=selected_source,
    )
    return (
        selected_candidate.token_ids,
        selected_candidate.log_probs_norm,
        selected_candidate.log_probs_unnorm,
        acceptance_ratio,
        diagnostics,
    )


def mcmc_power_samp(
    p: AutoregressiveSampler,
    context: List[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 16,
    stop_mode: str = "eos",
    budget_strategy: str = "restart",
    selection_mode: str = "best_logp",
    answer_extractor: Optional[Callable[[str], Any]] = None,
    budget_config: Optional[SamplingBudgetConfig] = None,
    return_diagnostics: bool = False,
):
    """Power sampling with autoregressive MCMC.

    Args:
      return_diagnostics: If True, appends a diagnostics dictionary to the
        return tuple.
    """
    c = len(context)
    generated = context.copy() if context is not None else []
    log_probs_norm: List[float] = []
    log_probs_unnorm: List[float] = []
    stop_mode = _normalize_stop_mode(stop_mode)
    budget_strategy = _normalize_budget_strategy(budget_strategy)
    selection_mode = _normalize_selection_mode(selection_mode)
    budget_tracker = SamplingBudgetTracker(budget_config)

    max_new_tokens = max(0, int(max_new_tokens))
    block_num = max(1, int(block_num))
    jump_size = 0 if max_new_tokens == 0 else max(1, int(math.ceil(max_new_tokens / block_num)))
    if stop_mode == "budget" and budget_tracker.max_sampling_tokens is None:
        raise ValueError("MCMC stop_mode='budget' requires max_sampling_tokens to be set.")
    _log_sampler_setup(
        method_name="power_samp",
        temp=temp,
        max_new_tokens=max_new_tokens,
        block_num=block_num,
        jump_size=jump_size,
        mcmc_steps=mcmc_steps,
        context_len=c,
    )
    if stop_mode == "budget" and budget_strategy == "restart":
        generated, log_probs_norm, log_probs_unnorm, acceptance_ratio, diagnostics = _run_budget_restart_search(
            p=p,
            context=context,
            temp=temp,
            mcmc_steps=mcmc_steps,
            max_new_tokens=max_new_tokens,
            block_num=block_num,
            budget_tracker=budget_tracker,
            selection_mode=selection_mode,
            answer_extractor=answer_extractor,
        )
        if return_diagnostics:
            return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio, diagnostics
        return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio

    attempts = 0
    acceptances = 0
    growth_rounds = 0
    refinement_rounds = 0
    stop_reason: Optional[str] = None
    growth_complete_reason: Optional[str] = None
    eos_token_id = p.tokenizer.eos_token_id

    if max_new_tokens == 0:
        diagnostics = _build_diagnostics(
            generated=generated,
            context_len=c,
            acceptance_ratio=0.0,
            budget_tracker=budget_tracker,
            stop_reason="max_new_tokens",
            block_num=block_num,
            mcmc_steps=mcmc_steps,
            stop_mode=stop_mode,
            proposal_attempts=attempts,
            proposal_acceptances=acceptances,
            growth_rounds=growth_rounds,
            refinement_rounds=refinement_rounds,
            budget_strategy="none" if stop_mode == "eos" else budget_strategy,
            selection_mode="last",
        )
        if return_diagnostics:
            return generated, log_probs_norm, log_probs_unnorm, 0.0, diagnostics
        return generated, log_probs_norm, log_probs_unnorm, 0.0

    for block_idx in tqdm(range(block_num), desc="power_samp blocks", leave=False):
        if stop_mode == "budget" and eos_token_id is not None and int(eos_token_id) in generated:
            growth_complete_reason = "eos"
            break

        current_output_tokens = max(len(generated) - c, 0)
        remaining_output_tokens = max(max_new_tokens - current_output_tokens, 0)
        if remaining_output_tokens <= 0:
            growth_complete_reason = "max_new_tokens"
            break

        extension_tokens = min(jump_size, remaining_output_tokens)
        extension_tokens = budget_tracker.clamp_sampling_tokens(extension_tokens)
        if extension_tokens <= 0:
            growth_complete_reason = "sampling_budget_exhausted"
            break

        generated, lp_norm, lp_unnorm = naive_temp(
            p,
            generated,
            temp=temp,
            seq_len=len(generated) + extension_tokens,
        )
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)
        budget_tracker.spend(len(lp_norm))
        growth_rounds += 1

        for _ in tqdm(range(mcmc_steps), desc=f"power_samp block {block_idx + 1}/{block_num}", leave=False):
            generated, log_probs_norm, log_probs_unnorm, attempted, accepted, proposal_stop_reason = _run_mcmc_proposal_step(
                p=p,
                generated=generated,
                log_probs_norm=log_probs_norm,
                log_probs_unnorm=log_probs_unnorm,
                context_len=c,
                temp=temp,
                budget_tracker=budget_tracker,
            )
            if proposal_stop_reason is not None:
                growth_complete_reason = proposal_stop_reason
                break
            if attempted:
                attempts += 1
            if accepted:
                acceptances += 1

        _log_block_summary("power_samp", block_idx + 1, block_num, attempts, acceptances, c, len(generated), max_new_tokens)

        if eos_token_id is not None and int(eos_token_id) in generated:
            generated, log_probs_norm, log_probs_unnorm = _truncate_at_first_eos(
                generated,
                log_probs_norm,
                log_probs_unnorm,
                context_len=c,
                eos_token_id=eos_token_id,
            )
            if stop_mode == "eos":
                print(f"[power_samp] stopped early due to EOS at block {block_idx + 1}/{block_num}", flush=True)
                stop_reason = "eos"
                acceptance_ratio = acceptances / max(attempts, 1)
                diagnostics = _build_diagnostics(
                    generated=generated,
                    context_len=c,
                    acceptance_ratio=acceptance_ratio,
                    budget_tracker=budget_tracker,
                    stop_reason=stop_reason,
                    block_num=block_num,
                    mcmc_steps=mcmc_steps,
                    stop_mode=stop_mode,
                    proposal_attempts=attempts,
                    proposal_acceptances=acceptances,
                    growth_rounds=growth_rounds,
                    refinement_rounds=refinement_rounds,
                    budget_strategy="none" if stop_mode == "eos" else budget_strategy,
                    selection_mode="last",
                )
                if return_diagnostics:
                    return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio, diagnostics
                return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio
            growth_complete_reason = "eos"
            break

        if growth_complete_reason is not None:
            break

        if budget_tracker.exhausted():
            growth_complete_reason = "sampling_budget_exhausted"
            break

        if max(len(generated) - c, 0) >= max_new_tokens:
            growth_complete_reason = "max_new_tokens"
            break
    else:
        if growth_complete_reason is None:
            growth_complete_reason = "block_num_completed"

    if stop_mode == "budget" and not budget_tracker.exhausted() and len(generated) > c:
        while not budget_tracker.exhausted():
            round_attempts = 0
            round_acceptances = 0
            refinement_rounds += 1

            for _ in range(mcmc_steps):
                generated, log_probs_norm, log_probs_unnorm, attempted, accepted, proposal_stop_reason = _run_mcmc_proposal_step(
                    p=p,
                    generated=generated,
                    log_probs_norm=log_probs_norm,
                    log_probs_unnorm=log_probs_unnorm,
                    context_len=c,
                    temp=temp,
                    budget_tracker=budget_tracker,
                )
                if proposal_stop_reason is not None:
                    stop_reason = proposal_stop_reason
                    break
                if attempted:
                    attempts += 1
                    round_attempts += 1
                if accepted:
                    acceptances += 1
                    round_acceptances += 1

            if round_attempts == 0:
                if stop_reason is None:
                    stop_reason = growth_complete_reason or "no_valid_proposals"
                break

            if budget_tracker.exhausted():
                stop_reason = "sampling_budget_exhausted"
                break

            if stop_reason is not None:
                break

        if stop_reason is None:
            stop_reason = growth_complete_reason

    if stop_reason is None:
        stop_reason = growth_complete_reason

    if stop_mode == "budget" and eos_token_id is not None and int(eos_token_id) in generated:
        generated, log_probs_norm, log_probs_unnorm = _truncate_at_first_eos(
            generated,
            log_probs_norm,
            log_probs_unnorm,
            context_len=c,
            eos_token_id=eos_token_id,
        )

    acceptance_ratio = acceptances / max(attempts, 1)
    diagnostics = _build_diagnostics(
        generated=generated,
        context_len=c,
        acceptance_ratio=acceptance_ratio,
        budget_tracker=budget_tracker,
        stop_reason=stop_reason,
        block_num=block_num,
        mcmc_steps=mcmc_steps,
        stop_mode=stop_mode,
        proposal_attempts=attempts,
        proposal_acceptances=acceptances,
        growth_rounds=growth_rounds,
        refinement_rounds=refinement_rounds,
        budget_strategy="none" if stop_mode == "eos" else budget_strategy,
        selection_mode="last",
    )
    if return_diagnostics:
        return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio, diagnostics
    return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio
