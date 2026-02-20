"""Sequential Monte Carlo (SMC) / Feynman--Kac power sampling.

This module implements a particle-filter style sampler for the tempered
("power") distribution over continuations:

  pi(x_{1:T} | c) propto p(x_{1:T} | c)^alpha,

where p is the base autoregressive model distribution and alpha = 1 / temp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F


@dataclass
class PowerSamplerSMCConfig:
    """Hyperparameters for SMC power sampling."""

    # Repository convention: alpha = 1 / temp.
    temp: float = 0.25

    # Number of particles.
    num_particles: int = 8

    # Resample when ESS / N < threshold.
    ess_threshold: float = 0.5

    # Only consider resampling every K steps (>= 1).
    resample_interval: int = 8

    # Resampling scheme: {"systematic", "stratified", "multinomial"}.
    resample_method: str = "systematic"

    # Proposal distribution temperature. 1.0 => proposal == base model.
    proposal_temperature: float = 1.0

    # Optional proposal truncation (approximate).
    proposal_top_k: Optional[int] = None
    proposal_top_p: float = 1.0

    # Numerical safety: clamp per-step log-weight increments when > 0.
    max_logw_step: float = 50.0

    # Stop early if all particles emitted EOS.
    stop_on_all_eos: bool = True

    # Optional RNG seed for sampling and resampling.
    seed: Optional[int] = None


def _apply_top_k(logits: torch.Tensor, top_k: Optional[int]) -> torch.Tensor:
    if top_k is None:
        return logits
    k = int(top_k)
    if k <= 0:
        return logits
    vocab_size = int(logits.shape[-1])
    if k >= vocab_size:
        return logits
    top_values, _ = torch.topk(logits, k=k, dim=-1)
    kth = top_values[..., -1].unsqueeze(-1)
    return torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)


def _apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    top_p_val = float(top_p)
    if top_p_val >= 1.0:
        return logits
    if top_p_val <= 0.0:
        argmax_ids = torch.argmax(logits, dim=-1, keepdim=True)
        return torch.full_like(logits, -float("inf")).scatter(1, argmax_ids, logits.gather(1, argmax_ids))

    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_mask = cumulative_probs > top_p_val
    sorted_mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
    return logits.scatter(1, sorted_indices, sorted_logits)


def _normalized_weights(logw: torch.Tensor) -> torch.Tensor:
    # logw: (N,)
    if logw.numel() == 0:
        return logw

    safe_logw = torch.where(torch.isfinite(logw), logw, torch.full_like(logw, -1e9))
    safe_logw = safe_logw - torch.logsumexp(safe_logw, dim=0)
    weights = torch.exp(safe_logw)

    if not torch.all(torch.isfinite(weights)):
        n = max(int(weights.numel()), 1)
        return torch.full_like(weights, 1.0 / float(n))
    return weights


def _ess(weights: torch.Tensor) -> float:
    if weights.numel() == 0:
        return 0.0
    denom = torch.sum(weights * weights)
    if not torch.isfinite(denom) or float(denom.item()) <= 0.0:
        return 0.0
    return float((1.0 / denom).item())


def _systematic_resample(weights: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    n = int(weights.numel())
    if n <= 0:
        return torch.empty((0,), dtype=torch.long, device=weights.device)

    u0 = torch.rand((1,), device=weights.device, generator=generator) / float(n)
    positions = u0 + torch.arange(n, device=weights.device, dtype=torch.float32) / float(n)

    cdf = torch.cumsum(weights, dim=0)
    indices = torch.searchsorted(cdf, positions, right=False)
    return torch.clamp(indices, 0, n - 1).to(torch.long)


def _stratified_resample(weights: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    n = int(weights.numel())
    if n <= 0:
        return torch.empty((0,), dtype=torch.long, device=weights.device)

    u = torch.rand((n,), device=weights.device, generator=generator)
    positions = (u + torch.arange(n, device=weights.device, dtype=torch.float32)) / float(n)

    cdf = torch.cumsum(weights, dim=0)
    indices = torch.searchsorted(cdf, positions, right=False)
    return torch.clamp(indices, 0, n - 1).to(torch.long)


def _multinomial_resample(weights: torch.Tensor, *, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    n = int(weights.numel())
    if n <= 0:
        return torch.empty((0,), dtype=torch.long, device=weights.device)
    return torch.multinomial(weights, num_samples=n, replacement=True, generator=generator)


def _resample_indices(
    weights: torch.Tensor,
    *,
    method: str,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    method_l = str(method).strip().lower()
    if method_l == "multinomial":
        return _multinomial_resample(weights, generator=generator)
    if method_l == "stratified":
        return _stratified_resample(weights, generator=generator)
    if method_l == "systematic":
        return _systematic_resample(weights, generator=generator)
    raise ValueError(f"Unknown resample_method='{method}'. Expected one of: systematic, stratified, multinomial")


def _reorder_past_key_values(model: Any, past_key_values: Any, indices: torch.Tensor) -> Any:
    """Reorder a HuggingFace-style KV cache along batch dimension."""

    if past_key_values is None:
        return None

    # Most HF models expose this for beam search.
    if hasattr(model, "_reorder_cache"):
        try:
            return model._reorder_cache(past_key_values, indices)
        except Exception:
            pass

    # Fallback for tuple(layer)->(k,v) caches.
    try:
        reordered = []
        for layer_past in past_key_values:
            if layer_past is None:
                reordered.append(None)
                continue
            if isinstance(layer_past, (tuple, list)) and len(layer_past) == 2:
                k, v = layer_past
                if torch.is_tensor(k):
                    k = k.index_select(0, indices.to(device=k.device))
                if torch.is_tensor(v):
                    v = v.index_select(0, indices.to(device=v.device))
                reordered.append((k, v))
            else:
                reordered.append(layer_past)
        return tuple(reordered)
    except Exception:
        return past_key_values


def _sanitize_log_q(log_q: torch.Tensor, fallback_log_q: torch.Tensor) -> torch.Tensor:
    """Replace invalid proposal rows with a valid fallback distribution."""

    if torch.all(torch.isfinite(log_q)):
        return log_q
    good_rows = torch.all(torch.isfinite(log_q), dim=-1)
    if torch.all(good_rows):
        return log_q
    fixed = log_q.clone()
    fixed[~good_rows] = fallback_log_q[~good_rows]
    return fixed


@torch.inference_mode()
def smc_power_sample(
    model: Any,
    tokenizer: Any,
    device: str,
    context: list[int],
    *,
    cfg: PowerSamplerSMCConfig,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
) -> Tuple[list[int], Dict[str, Any]]:
    """Sample from the power distribution using an SMC/Feynman--Kac filter.

    Returns:
      - token ids for full sequence (prompt + generated, truncated at first EOS)
      - diagnostics dictionary
    """

    if cfg.temp <= 0:
        raise ValueError("cfg.temp must be > 0 (alpha = 1/cfg.temp).")

    alpha = 1.0 / float(cfg.temp)
    max_new_tokens = max(0, int(max_new_tokens))
    if max_new_tokens == 0:
        return context.copy(), {"steps": 0.0, "num_resamples": 0.0, "avg_ess": 0.0, "final_ess": 0.0}

    num_particles = max(1, int(cfg.num_particles))
    prompt = list(context)
    if not prompt:
        raise ValueError("context must contain at least one token.")

    max_seq_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if max_seq_len is None:
        max_seq_len = getattr(getattr(model, "config", None), "n_positions", None)
    if max_seq_len is not None:
        allowed_new_tokens = int(max_seq_len) - len(prompt)
        max_new_tokens = max(0, min(max_new_tokens, allowed_new_tokens))
        if max_new_tokens == 0:
            return context.copy(), {"steps": 0.0, "num_resamples": 0.0, "avg_ess": 0.0, "final_ess": 0.0}

    eos_id = eos_token_id
    if eos_id is None:
        eos_id = getattr(tokenizer, "eos_token_id", None)
    eos_id_int = None if eos_id is None else int(eos_id)

    pad_id = pad_token_id
    if pad_id is None:
        pad_id = getattr(tokenizer, "pad_token_id", None)
    if pad_id is None:
        pad_id = eos_id_int
    if pad_id is None:
        pad_id = 0
    pad_id_int = int(pad_id)

    prompt_ids = torch.tensor(prompt, dtype=torch.long)
    batch_input_ids = prompt_ids.unsqueeze(0).repeat(num_particles, 1).to(device)
    prompt_len = int(batch_input_ids.shape[1])
    total_len = prompt_len + max_new_tokens

    attention_mask = torch.ones((num_particles, total_len), dtype=torch.long, device=batch_input_ids.device)

    generator: Optional[torch.Generator] = None
    if cfg.seed is not None:
        try:
            generator = torch.Generator(device=batch_input_ids.device)
        except Exception:
            generator = torch.Generator()
        generator.manual_seed(int(cfg.seed))

    outputs = model(
        input_ids=batch_input_ids,
        attention_mask=attention_mask[:, :prompt_len],
        use_cache=True,
    )
    next_logits = outputs.logits[:, -1, :].float()
    past_key_values = outputs.past_key_values

    generated = torch.full((num_particles, max_new_tokens), pad_id_int, dtype=torch.long, device=batch_input_ids.device)
    finished = torch.zeros((num_particles,), dtype=torch.bool, device=batch_input_ids.device)
    logw = torch.zeros((num_particles,), dtype=torch.float32, device=batch_input_ids.device)

    num_resamples = 0
    ess_sum = 0.0
    ess_count = 0
    steps_done = 0

    proposal_temperature = float(cfg.proposal_temperature)
    if proposal_temperature <= 0:
        raise ValueError("cfg.proposal_temperature must be > 0")

    resample_interval = max(1, int(cfg.resample_interval))
    ess_threshold = float(cfg.ess_threshold)
    max_logw_step = float(cfg.max_logw_step)

    eos_tensor: Optional[torch.Tensor] = None
    if eos_id_int is not None:
        eos_tensor = torch.tensor(eos_id_int, device=batch_input_ids.device, dtype=torch.long)

    for step_idx in range(max_new_tokens):
        already_finished = finished

        # Base model probabilities p(x_t | x_<t).
        log_p = F.log_softmax(next_logits, dim=-1)

        # Proposal q(x_t | x_<t).
        proposal_logits = next_logits
        if proposal_temperature != 1.0:
            proposal_logits = proposal_logits / proposal_temperature
        proposal_logits = _apply_top_k(proposal_logits, cfg.proposal_top_k)
        proposal_logits = _apply_top_p(proposal_logits, cfg.proposal_top_p)
        log_q = F.log_softmax(proposal_logits, dim=-1)
        log_q = _sanitize_log_q(log_q, log_p)

        # Sample from q.
        sampled_ids = torch.multinomial(torch.exp(log_q), num_samples=1, generator=generator).squeeze(1)

        if eos_tensor is not None:
            # Absorb once EOS has been sampled.
            sampled_ids = torch.where(already_finished, eos_tensor, sampled_ids)

        # Incremental importance weight: alpha * log p - log q.
        log_p_tok = log_p.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
        log_q_tok = log_q.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
        increment = alpha * log_p_tok - log_q_tok
        increment = torch.where(already_finished, torch.zeros_like(increment), increment)
        if max_logw_step > 0:
            increment = torch.clamp(increment, min=-max_logw_step, max=max_logw_step)
        logw = logw + increment

        generated[:, step_idx] = sampled_ids

        if eos_id_int is not None:
            finished = already_finished | sampled_ids.eq(eos_id_int)

        steps_done = step_idx + 1

        should_resample = False
        normalized = None

        if num_particles > 1 and ((step_idx + 1) % resample_interval == 0):
            normalized = _normalized_weights(logw)
            ess_val = _ess(normalized)
            ess_sum += ess_val
            ess_count += 1
            if ess_threshold > 0.0 and (ess_val / float(num_particles)) < ess_threshold:
                should_resample = True

        if should_resample and normalized is not None:
            indices = _resample_indices(normalized, method=cfg.resample_method, generator=generator)
            generated = generated.index_select(0, indices)
            finished = finished.index_select(0, indices)
            past_key_values = _reorder_past_key_values(model, past_key_values, indices)
            sampled_ids = generated[:, step_idx]
            logw = torch.zeros_like(logw)
            num_resamples += 1

        if bool(cfg.stop_on_all_eos) and eos_id_int is not None and bool(torch.all(finished).item()):
            break

        current_len = prompt_len + step_idx + 1
        outputs = model(
            input_ids=sampled_ids.unsqueeze(1),
            attention_mask=attention_mask[:, :current_len],
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_logits = outputs.logits[:, -1, :].float()
        past_key_values = outputs.past_key_values

    final_weights = _normalized_weights(logw)
    final_ess = _ess(final_weights)
    final_idx = int(torch.multinomial(final_weights, num_samples=1, generator=generator).item())
    chosen = generated[final_idx, :steps_done].detach().cpu().tolist()

    if eos_id_int is not None and eos_id_int in chosen:
        eos_pos = chosen.index(eos_id_int)
        chosen = chosen[: eos_pos + 1]

    out = prompt + [int(tok) for tok in chosen]
    avg_ess = float(ess_sum / max(ess_count, 1))

    diagnostics: Dict[str, Any] = {
        "steps": float(steps_done),
        "num_particles": float(num_particles),
        "num_resamples": float(num_resamples),
        "avg_ess": avg_ess,
        "final_ess": float(final_ess),
        "ess_threshold": float(cfg.ess_threshold),
        "resample_interval": float(resample_interval),
        "resample_method": str(cfg.resample_method),
        "proposal_temperature": float(cfg.proposal_temperature),
        "proposal_top_k": None if cfg.proposal_top_k is None else float(int(cfg.proposal_top_k)),
        "proposal_top_p": float(cfg.proposal_top_p),
        "temperature": float(cfg.temp),
        "alpha": float(alpha),
    }
    return out, diagnostics
