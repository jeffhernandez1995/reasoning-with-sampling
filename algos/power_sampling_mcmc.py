"""Regular (MCMC) power sampling algorithm.

This is the MCMC trajectory sampler used for approximate sampling from p^alpha
as implemented in the original project pipeline. It mirrors the previous
implementation in `power_samp_utils.py` so existing behavior is preserved.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


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


def mcmc_power_samp(
    p: AutoregressiveSampler,
    context: List[int],
    temp: float,
    mcmc_steps: int,
    max_new_tokens: int,
    block_num: int = 16,
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
    total_sampling_tokens = 0

    assert max_new_tokens % block_num == 0
    jump_size = int(max_new_tokens // block_num)
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

    for block_idx in tqdm(range(block_num), desc="power_samp blocks", leave=False):
        generated, lp_norm, lp_unnorm = naive_temp(p, generated, temp=temp, seq_len=jump_size + len(generated))
        log_probs_norm.extend(lp_norm)
        log_probs_unnorm.extend(lp_unnorm)
        total_sampling_tokens += len(lp_norm)

        for _ in tqdm(range(mcmc_steps), desc=f"power_samp block {block_idx + 1}/{block_num}", leave=False):
            attempts += 1
            t = len(generated)
            idx = random.randint(c, t - 1)

            proposal, log_prob_prop, target_log_prob_prop = naive_temp(p, generated[:idx], temp=temp, seq_len=t)
            total_sampling_tokens += len(log_prob_prop)
            s = len(proposal)
            assert len(log_prob_prop) == s - idx
            assert len(target_log_prob_prop) == s - idx

            log_prob_cur = log_probs_norm.copy()[idx - c : s - c]
            target_log_prob_cur = log_probs_unnorm.copy()[idx - c : s - c]
            log_r = sum(target_log_prob_prop) + sum(log_prob_cur) - sum(target_log_prob_cur) - sum(log_prob_prop)

            if np.random.rand() < np.exp(log_r):
                acceptances += 1
                generated = proposal.copy()
                log_probs_norm[idx - c :] = log_prob_prop.copy()
                log_probs_unnorm[idx - c :] = target_log_prob_prop.copy()

        _log_block_summary("power_samp", block_idx + 1, block_num, attempts, acceptances, c, len(generated), max_new_tokens)

        if p.tokenizer.eos_token_id in generated:
            eos_idx = generated.index(p.tokenizer.eos_token_id)
            generated = generated[: eos_idx + 1]
            log_probs_norm = log_probs_norm[: eos_idx + 1]
            log_probs_unnorm = log_probs_unnorm[: eos_idx + 1]
            print(f"[power_samp] stopped early due to EOS at block {block_idx + 1}/{block_num}", flush=True)
            acceptance_ratio = acceptances / max(attempts, 1)
            diagnostics: Dict[str, Any] = {
                "output_tokens": float(max(len(generated) - c, 0)),
                "sampling_tokens": float(total_sampling_tokens),
                "internal_sampling_tokens": float(max(total_sampling_tokens - max(len(generated) - c, 0), 0)),
                "acceptance_ratio": float(acceptance_ratio),
            }
            if return_diagnostics:
                return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio, diagnostics
            return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio

    acceptance_ratio = acceptances / max(attempts, 1)
    diagnostics = {
        "output_tokens": float(max(len(generated) - c, 0)),
        "sampling_tokens": float(total_sampling_tokens),
        "internal_sampling_tokens": float(max(total_sampling_tokens - max(len(generated) - c, 0), 0)),
        "acceptance_ratio": float(acceptance_ratio),
    }
    if return_diagnostics:
        return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio, diagnostics
    return generated, log_probs_norm, log_probs_unnorm, acceptance_ratio
