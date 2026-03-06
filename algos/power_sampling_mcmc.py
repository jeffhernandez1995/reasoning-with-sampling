"""Regular (MCMC) power sampling algorithm.

This is the MCMC trajectory sampler used for approximate sampling from p^alpha
as implemented in the original project pipeline. It mirrors the previous
implementation in `power_samp_utils.py` so existing behavior is preserved.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Optional, Tuple

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
        "mcmc_proposal_attempts": int(proposal_attempts),
        "mcmc_proposal_acceptances": int(proposal_acceptances),
        "mcmc_growth_rounds": int(growth_rounds),
        "mcmc_refinement_rounds": int(refinement_rounds),
    }
    diagnostics.update(budget_tracker.metadata())
    return diagnostics


def mcmc_power_samp(
    p: AutoregressiveSampler,
    context: List[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 16,
    stop_mode: str = "eos",
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
    )
    if return_diagnostics:
        return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio, diagnostics
    return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio
