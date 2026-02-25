"""Deterministic truncated-partition recursion for power-distribution sampling.

This module implements a *deterministic* approximation to the token-wise scaling
factor \zeta_t(x_t) (a.k.a. the future partition term) from Theorem 3.1 of
"Approximate Power Distribution Sampling for LLM Reasoning" (arXiv:2601.21590).

Background
----------
The power distribution on full sequences reweights the base model p by an
exponent \alpha > 1. At the token level, Theorem 3.1 shows that we can write the
conditional next-token distribution as

  p^{(pow)}_\alpha(x_t | h) \propto p(x_t | h)^\alpha \; \zeta_t(x_t),

with

  \zeta_t(x_t) = \sum_{x_{t+1:T}} p(x_{t+1:T} | h, x_t)^\alpha.

The Monte Carlo algorithm in arXiv:2601.21590 estimates \zeta_t(x_t) via rollouts.
Here we instead estimate it *deterministically* via a truncated partition
recursion, i.e. by expanding a bounded lookahead tree and summing p^\alpha mass
over the retained paths.

Core approximation
------------------
We introduce a truncated horizon H and a local branching factor K_b.
Define a truncated partition:

  Z_H(h) := \sum_{x_{1:H}} \prod_{i=1}^H p(x_i | h, x_{<i})^\alpha.

Then we approximate \zeta_t(x_t) \approx Z_H(h, x_t).
Computing Z_H exactly is still exponential in H, so we additionally prune the
lookahead tree with a per-candidate beam width B. The recursion is deterministic
(given p and hyperparameters): at each depth we keep the top-B partial paths by
current log-weight.

Implementation notes
--------------------
This code is Hugging Face / PyTorch oriented and uses KV caching.
We perform batched single-token decoding for all active beams at a given depth.

The sampler is designed to integrate cleanly with the existing codebase:
  - dataclass config
  - returns (token_ids, diagnostics)

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch


@dataclass
class PowerSamplerDTRConfig:
    """Configuration for deterministic truncated recursion (DTR).

    Notation:
      - \alpha = 1 / temp
      - K = top_k (candidate set for the *actual* next token)
      - H = lookahead_tokens (truncation horizon for \zeta)
      - K_b = branch_factor (local branching factor inside the lookahead tree)
      - B = beam_width (max active partial paths per candidate during recursion)
    """

    # We follow repo convention alpha = 1 / temp.
    temp: float = 0.25

    # Candidate set size for next-token sampling (Top-K).
    top_k: int = 8

    # Truncated partition horizon H (number of tokens *after* the candidate token).
    lookahead_tokens: int = 16

    # Local branching factor K_b for the deterministic recursion (Top-K_b at each node).
    branch_factor: int = 8

    # Beam width B used to prune partial paths *per candidate* at each depth.
    beam_width: int = 16

    # If True, force EOS into every Top-K_b set (by replacing the last entry when absent).
    include_eos_in_branch: bool = True

    # Optional additional pruning: drop children whose log-weight is more than this
    # margin below the best child for that candidate at that depth.
    # None disables margin pruning.
    prune_logw_margin: Optional[float] = None

    # Chunk model forward passes along the batch dimension to avoid GPU OOM.
    # None disables chunking.
    max_forward_batch_size: Optional[int] = None


def _clamp_new_tokens(max_new_tokens: int, prompt_len: int, max_seq_len: Optional[int]) -> int:
    if max_seq_len is None:
        return max(0, int(max_new_tokens))
    allowed = int(max_seq_len) - int(prompt_len)
    return max(0, min(int(max_new_tokens), allowed))


def _past_seq_len(past_key_values) -> Optional[int]:
    """Best-effort extraction of cached sequence length from HF past_key_values."""
    if past_key_values is None:
        return None
    try:
        layer0 = past_key_values[0]
        if isinstance(layer0, (tuple, list)) and torch.is_tensor(layer0[0]):
            # (batch, n_heads, seq_len, head_dim)
            return int(layer0[0].shape[2])
    except Exception:
        return None
    return None


def _expand_past(past_key_values, batch_size: int):
    """Expand a past_key_values batch of size 1 to batch_size by repeating."""
    if past_key_values is None:
        return None
    batch_size = int(batch_size)
    if batch_size <= 1:
        return past_key_values
    new_past = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)):
            raise TypeError(f"Unsupported past_key_values layer type: {type(layer)}")
        tensors = []
        for t in layer:
            if torch.is_tensor(t):
                if t.shape[0] != 1:
                    raise ValueError("_expand_past expects batch size 1 in past_key_values")
                tensors.append(t.expand(batch_size, *t.shape[1:]).contiguous())
            else:
                tensors.append(t)
        new_past.append(tuple(tensors))
    return tuple(new_past)


def _index_past(past_key_values, indices: torch.Tensor):
    """Select a subset of beams from past_key_values along batch dim."""
    if past_key_values is None:
        return None
    if indices.numel() == 0:
        return None
    indices = indices.to(dtype=torch.long)
    new_past = []
    for layer in past_key_values:
        if not isinstance(layer, (tuple, list)):
            raise TypeError(f"Unsupported past_key_values layer type: {type(layer)}")
        tensors = []
        for t in layer:
            if torch.is_tensor(t):
                tensors.append(torch.index_select(t, 0, indices))
            else:
                tensors.append(t)
        new_past.append(tuple(tensors))
    return tuple(new_past)


def _topk_logp_from_logits(
    logits: torch.Tensor,
    *,
    k: int,
    eos_token_id: Optional[int] = None,
    include_eos: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (topk_ids, topk_logp) from unnormalized logits.

    Computes log p(token|prefix) for Top-K tokens in a numerically stable way:
      logp = logit - logsumexp(logits).

    Args:
      logits: (batch, vocab)
      k: desired top-k (clamped to vocab)
      eos_token_id: optional EOS token to force into the set
      include_eos: if True, force EOS into the returned Top-K ids

    Returns:
      topk_ids: (batch, k)
      topk_logp: (batch, k)
    """
    if logits.dim() != 2:
        raise ValueError(f"Expected logits with shape (batch, vocab), got {tuple(logits.shape)}")
    batch, vocab = int(logits.shape[0]), int(logits.shape[1])
    k = max(1, min(int(k), vocab))

    logits_f = logits.float()
    topk_logits, topk_ids = torch.topk(logits_f, k=k, dim=-1)
    log_z = torch.logsumexp(logits_f, dim=-1, keepdim=True)
    topk_logp = topk_logits - log_z

    if include_eos and eos_token_id is not None:
        eos_id = int(eos_token_id)
        if 0 <= eos_id < vocab:
            eos_in = topk_ids.eq(eos_id).any(dim=-1)
            if not bool(torch.all(eos_in).item()):
                eos_logp = logits_f[:, eos_id] - log_z.squeeze(-1)
                missing = ~eos_in
                if bool(missing.any().item()):
                    topk_ids[missing, -1] = eos_id
                    topk_logp[missing, -1] = eos_logp[missing]

    return topk_ids, topk_logp


def _logaddexp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.logaddexp(a, b)


@torch.inference_mode()
def _forward_one_token(
    model,
    *,
    input_ids: torch.Tensor,  # (batch, 1)
    past_key_values,
    attention_mask: Optional[torch.Tensor],
    max_forward_batch_size: Optional[int] = None,
):
    """Forward the model for one decoding step with optional chunking.

    Chunking is rarely needed for typical settings (K<=8, B<=16), but it can be
    helpful when experimenting with larger beam widths or longer prompts.
    """
    if max_forward_batch_size is None or input_ids.shape[0] <= int(max_forward_batch_size):
        return model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )

    # Chunk along batch dimension.
    bs = int(input_ids.shape[0])
    chunk = int(max_forward_batch_size)

    logits_chunks: List[torch.Tensor] = []
    past_chunks = []

    # We need to index-select the corresponding past cache for each chunk.
    all_indices = torch.arange(bs, device=input_ids.device)

    for start in range(0, bs, chunk):
        end = min(bs, start + chunk)
        ids_chunk = input_ids[start:end]
        mask_chunk = None
        if attention_mask is not None:
            mask_chunk = attention_mask[start:end]
        past_chunk = _index_past(past_key_values, all_indices[start:end])
        out = model(
            input_ids=ids_chunk,
            attention_mask=mask_chunk,
            past_key_values=past_chunk,
            use_cache=True,
        )
        logits_chunks.append(out.logits)
        past_chunks.append(out.past_key_values)

    logits = torch.cat(logits_chunks, dim=0)

    # Concatenate past_key_values along batch dim.
    merged_past = []
    num_layers = len(past_chunks[0])
    for layer_idx in range(num_layers):
        layer_parts = [p[layer_idx] for p in past_chunks]
        merged_layer = []
        for tensor_idx in range(len(layer_parts[0])):
            if torch.is_tensor(layer_parts[0][tensor_idx]):
                merged_layer.append(torch.cat([lp[tensor_idx] for lp in layer_parts], dim=0))
            else:
                merged_layer.append(layer_parts[0][tensor_idx])
        merged_past.append(tuple(merged_layer))

    # Return a lightweight object mimicking HF outputs.
    return type("Out", (), {"logits": logits, "past_key_values": tuple(merged_past)})


@torch.inference_mode()
def truncated_partition_logz(
    model,
    *,
    start_past_key_values,
    start_next_logits: torch.Tensor,  # (S, vocab)
    alpha: float,
    lookahead_tokens: int,
    branch_factor: int,
    beam_width: int,
    eos_token_id: int,
    include_eos_in_branch: bool,
    prune_logw_margin: Optional[float] = None,
    max_forward_batch_size: Optional[int] = None,
    return_stats: bool = False,
) -> Any:
    """Estimate log Z for each start state via deterministic truncated recursion.

    Args:
      start_past_key_values: past cache for each start prefix (batch S).
      start_next_logits: logits predicting next token after each start prefix.
      alpha: power exponent (alpha = 1/temp).
      lookahead_tokens: horizon H.
      branch_factor: local Top-K_b.
      beam_width: beam width B per start.

    Returns:
      If return_stats=False:
        logZ: (S,) tensor with log partition estimates.
      If return_stats=True:
        (logZ, stats) where stats tracks lookahead decoding workload.
    """
    device = start_next_logits.device
    s = int(start_next_logits.shape[0])
    h = max(0, int(lookahead_tokens))
    branch_factor = max(1, int(branch_factor))
    beam_width = max(1, int(beam_width))
    if max_forward_batch_size is not None:
        max_forward_batch_size = max(1, int(max_forward_batch_size))
    if h <= 0 or s == 0:
        zeros = torch.zeros((s,), dtype=torch.float32, device=device)
        if not return_stats:
            return zeros
        return zeros, {
            "lookahead_forward_tokens": 0.0,
            "lookahead_active_beams_sum": 0.0,
            "lookahead_active_beam_steps": 0.0,
        }

    beam_group = torch.arange(s, device=device, dtype=torch.long)
    beam_logw = torch.zeros((s,), dtype=torch.float32, device=device)
    beam_past = start_past_key_values
    beam_top_ids, beam_top_logp = _topk_logp_from_logits(
        start_next_logits,
        k=branch_factor,
        eos_token_id=eos_token_id,
        include_eos=include_eos_in_branch,
    )

    finished_logz = torch.full((s,), -float("inf"), dtype=torch.float32, device=device)
    lookahead_forward_tokens = 0
    lookahead_active_beams_sum = 0
    lookahead_active_beam_steps = 0

    for _depth in range(h):
        b = int(beam_group.shape[0])
        if b == 0:
            break
        lookahead_active_beams_sum += b
        lookahead_active_beam_steps += 1

        k_b = int(beam_top_ids.shape[1])
        child_ids = beam_top_ids.reshape(-1)
        child_logp = beam_top_logp.reshape(-1)
        parent_idx = torch.arange(b, device=device, dtype=torch.long).repeat_interleave(k_b)
        child_group = beam_group.repeat_interleave(k_b)
        child_logw = beam_logw.repeat_interleave(k_b) + float(alpha) * child_logp

        selected_positions: List[int] = []
        for g in range(s):
            mask = child_group.eq(g)
            if not bool(mask.any().item()):
                continue
            idxs = torch.nonzero(mask, as_tuple=False).view(-1)
            weights = child_logw.index_select(0, idxs)
            top_n = min(int(beam_width), int(weights.numel()))
            if top_n <= 0:
                continue
            top_local = torch.topk(weights, k=top_n, dim=0).indices
            chosen = idxs.index_select(0, top_local)

            if prune_logw_margin is not None and chosen.numel() > 0:
                best = float(child_logw.index_select(0, chosen).max().item())
                margin = float(prune_logw_margin)
                keep = child_logw.index_select(0, chosen) >= (best - margin)
                chosen = chosen.index_select(0, torch.nonzero(keep, as_tuple=False).view(-1))

            selected_positions.extend(chosen.detach().cpu().tolist())

        if not selected_positions:
            break

        sel = torch.tensor(selected_positions, device=device, dtype=torch.long)
        sel_child_ids = child_ids.index_select(0, sel)
        sel_child_group = child_group.index_select(0, sel)
        sel_child_logw = child_logw.index_select(0, sel)
        sel_parent_idx = parent_idx.index_select(0, sel)

        is_eos = sel_child_ids.eq(int(eos_token_id))
        if bool(is_eos.any().item()):
            eos_groups = sel_child_group[is_eos]
            eos_logw = sel_child_logw[is_eos]
            for g, w in zip(eos_groups.tolist(), eos_logw):
                finished_logz[g] = _logaddexp(finished_logz[g], w)

        keep = ~is_eos
        if not bool(keep.any().item()):
            # Everything terminated.
            beam_group = torch.empty((0,), device=device, dtype=torch.long)
            beam_logw = torch.empty((0,), device=device, dtype=torch.float32)
            beam_past = None
            beam_top_ids = torch.empty((0, k_b), device=device, dtype=torch.long)
            beam_top_logp = torch.empty((0, k_b), device=device, dtype=torch.float32)
            break

        next_group = sel_child_group[keep]
        next_logw = sel_child_logw[keep]
        next_tokens = sel_child_ids[keep]
        next_parent = sel_parent_idx[keep]
        lookahead_forward_tokens += int(next_tokens.shape[0])

        parent_past = _index_past(beam_past, next_parent)
        seq_len = _past_seq_len(parent_past)
        attn_mask = None
        if seq_len is not None:
            attn_mask = torch.ones((int(next_tokens.shape[0]), int(seq_len) + 1), device=device, dtype=torch.long)

        out = _forward_one_token(
            model,
            input_ids=next_tokens.unsqueeze(1),
            past_key_values=parent_past,
            attention_mask=attn_mask,
            max_forward_batch_size=max_forward_batch_size,
        )
        next_logits = out.logits[:, -1, :]
        next_past = out.past_key_values
        next_top_ids, next_top_logp = _topk_logp_from_logits(
            next_logits,
            k=branch_factor,
            eos_token_id=eos_token_id,
            include_eos=include_eos_in_branch,
        )

        beam_group = next_group
        beam_logw = next_logw
        beam_past = next_past
        beam_top_ids = next_top_ids
        beam_top_logp = next_top_logp

    # Sum remaining active beams at depth H.
    active_logz = torch.full((s,), -float("inf"), dtype=torch.float32, device=device)
    if beam_group.numel() > 0:
        for g in range(s):
            mask = beam_group.eq(g)
            if not bool(mask.any().item()):
                continue
            w = beam_logw[mask]
            active_logz[g] = torch.logsumexp(w, dim=0)

    total = _logaddexp(finished_logz, active_logz)
    # If a group has no mass (should be rare), default to log(1)=0.
    total = torch.where(torch.isfinite(total), total, torch.zeros_like(total))
    if not return_stats:
        return total
    return total, {
        "lookahead_forward_tokens": float(lookahead_forward_tokens),
        "lookahead_active_beams_sum": float(lookahead_active_beams_sum),
        "lookahead_active_beam_steps": float(lookahead_active_beam_steps),
    }


def _sample_index_from_logits(logits_1d: torch.Tensor, rng: np.random.Generator) -> int:
    probs = torch.softmax(logits_1d, dim=-1)
    p_np = probs.detach().cpu().numpy().astype(np.float64)
    p_np = np.clip(p_np, 0.0, 1.0)
    total = float(p_np.sum())
    if not np.isfinite(total) or total <= 0.0:
        p_np = np.ones_like(p_np) / max(len(p_np), 1)
    else:
        p_np /= total
    return int(rng.choice(len(p_np), p=p_np))


@torch.inference_mode()
def dtr_power_sample(
    model,
    tokenizer,
    device: str,
    context: List[int],
    *,
    cfg: PowerSamplerDTRConfig,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[List[int], Dict[str, Any]]:
    """Autoregressively sample from a DTR-approximated power distribution.

    Returns:
      - full sequence including context
      - diagnostics
    """
    if rng is None:
        rng = np.random.default_rng()

    if cfg.temp <= 0:
        raise ValueError("cfg.temp must be > 0 (alpha = 1/cfg.temp).")
    alpha = 1.0 / float(cfg.temp)

    eos_id = eos_token_id
    if eos_id is None:
        eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("eos_token_id must be provided or tokenizer.eos_token_id must be set.")
    eos_id = int(eos_id)

    max_seq_len = getattr(model.config, "max_position_embeddings", None)
    if max_seq_len is None:
        max_seq_len = getattr(model.config, "n_positions", None)
    max_seq_len = None if max_seq_len is None else int(max_seq_len)

    prompt = context.copy()
    prompt_len = len(prompt)
    max_new_tokens = _clamp_new_tokens(max_new_tokens, prompt_len, max_seq_len)
    if max_new_tokens <= 0:
        return prompt, {
            "steps": 0.0,
            "output_tokens": 0.0,
            "sampling_tokens": 0.0,
            "internal_sampling_tokens": 0.0,
            "candidate_tokens": 0.0,
            "lookahead_tokens": 0.0,
            "lookahead_calls": 0.0,
            "avg_beams": 0.0,
        }

    # Prefill once.
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    past = out.past_key_values
    next_logits = out.logits[:, -1, :]
    past_len = _past_seq_len(past)

    seq = prompt.copy()
    steps = 0
    lookahead_calls = 0
    total_candidate_tokens = 0
    total_lookahead_tokens = 0

    # Diagnostics for average active beams in lookahead recursion.
    total_active_beams_sum = 0
    total_active_beam_steps = 0

    for _ in range(max_new_tokens):
        # Candidate set for the next token.
        top_k = max(1, int(cfg.top_k))
        vocab = int(next_logits.shape[-1])
        top_k = min(top_k, vocab)
        logits_f = next_logits[0].float()
        cand_logits, cand_ids = torch.topk(logits_f, k=top_k, dim=-1)
        log_z = torch.logsumexp(logits_f, dim=-1)
        cand_logp = cand_logits - log_z
        total_candidate_tokens += int(top_k)

        # Advance each candidate token once to get its KV cache and next logits.
        past_rep = _expand_past(past, batch_size=top_k)
        attn_mask = None
        if past_len is not None:
            attn_mask = torch.ones((top_k, int(past_len) + 1), device=device, dtype=torch.long)
        out_cand = model(
            input_ids=cand_ids.unsqueeze(1),
            attention_mask=attn_mask,
            past_key_values=past_rep,
            use_cache=True,
        )
        cand_past = out_cand.past_key_values
        cand_next_logits = out_cand.logits[:, -1, :]

        # Determine allowable lookahead given context window.
        lookahead = max(0, int(cfg.lookahead_tokens))
        if max_seq_len is not None:
            current_len = past_len if past_len is not None else len(seq)
            lookahead = min(lookahead, max(0, int(max_seq_len) - int(current_len) - 1))

        branch_factor = max(1, int(cfg.branch_factor))
        beam_width = max(1, int(cfg.beam_width))

        logz_future = torch.zeros((top_k,), dtype=torch.float32, device=device)
        non_eos = ~cand_ids.eq(eos_id)
        if lookahead > 0 and bool(non_eos.any().item()):
            idx = torch.nonzero(non_eos, as_tuple=False).view(-1)
            sub_past = _index_past(cand_past, idx)
            sub_logits = cand_next_logits.index_select(0, idx)
            lookahead_calls += 1
            sub_logz, sub_stats = truncated_partition_logz(
                model,
                start_past_key_values=sub_past,
                start_next_logits=sub_logits,
                alpha=float(alpha),
                lookahead_tokens=int(lookahead),
                branch_factor=int(branch_factor),
                beam_width=int(beam_width),
                eos_token_id=int(eos_id),
                include_eos_in_branch=bool(cfg.include_eos_in_branch),
                prune_logw_margin=cfg.prune_logw_margin,
                max_forward_batch_size=cfg.max_forward_batch_size,
                return_stats=True,
            )
            logz_future[idx] = sub_logz

            total_lookahead_tokens += int(sub_stats.get("lookahead_forward_tokens", 0.0))
            total_active_beams_sum += int(sub_stats.get("lookahead_active_beams_sum", 0.0))
            total_active_beam_steps += int(sub_stats.get("lookahead_active_beam_steps", 0.0))

        # Sample token from the approximate power distribution over the Top-K set.
        log_num = float(alpha) * cand_logp + logz_future
        choice = _sample_index_from_logits(log_num, rng)
        token = int(cand_ids[choice].item())
        seq.append(token)
        steps += 1
        if token == eos_id:
            break

        # Advance the main chain by one token.
        past_len = _past_seq_len(past)
        attn_mask_main = None
        if past_len is not None:
            attn_mask_main = torch.ones((1, int(past_len) + 1), device=device, dtype=torch.long)
        out = model(
            input_ids=torch.tensor([[token]], device=device, dtype=torch.long),
            attention_mask=attn_mask_main,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_logits = out.logits[:, -1, :]
        past_len = _past_seq_len(past)

    avg_beams = float(total_active_beams_sum) / max(float(total_active_beam_steps), 1.0)
    sampling_tokens = int(steps) + int(total_candidate_tokens) + int(total_lookahead_tokens)
    return seq, {
        "steps": float(steps),
        "output_tokens": float(steps),
        "sampling_tokens": float(sampling_tokens),
        "internal_sampling_tokens": float(max(sampling_tokens - steps, 0)),
        "candidate_tokens": float(total_candidate_tokens),
        "lookahead_tokens": float(total_lookahead_tokens),
        "lookahead_calls": float(lookahead_calls),
        "avg_beams": float(avg_beams),
    }
