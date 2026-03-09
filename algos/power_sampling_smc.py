"""Sequential Monte Carlo (SMC) / Feynman--Kac power sampling.

This module implements a particle-filter style sampler for the "power"
(tempered / sharpened) distribution over continuations:

  pi(x_{1:T} | c) propto p(x_{1:T} | c)^alpha,

where p is the base autoregressive model distribution and alpha = 1 / temp.

It also supports an auxiliary particle filter (APF) variant with lookahead
resampling based on the one-step partition function.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
import torch.nn.functional as F

from sampling_budget import SamplingBudgetConfig, SamplingBudgetTracker


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

    # --- Auxiliary particle filter / lookahead resampling ---
    # If True, perform lookahead pre-resampling using m_t computed from
    # current next-token logits before sampling x_t.
    use_auxiliary: bool = False

    # If True, resample at every APF opportunity (every resample_interval
    # steps) regardless of ESS. Otherwise use ESS(w_{t-1} * m_t).
    auxiliary_resample_always: bool = False

    # Temperature used for auxiliary lookahead mass m_t. If None, defaults to
    # cfg.temp (i.e., alpha_aux == alpha).
    auxiliary_temperature: Optional[float] = None


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


@dataclass
class SMCCandidate:
    token_ids: list[int]
    target_log_score: float
    completion_text: str
    answer_key: str
    completed: bool
    run_index: int
    stop_reason: str


@dataclass
class SMCSingleRunResult:
    token_ids: list[int]
    target_log_score: float
    steps_done: int
    output_tokens: int
    num_particles: int
    num_resamples: int
    ess_sum: float
    ess_count: int
    final_ess: float
    stop_reason: str
    candidate: SMCCandidate


def _normalize_stop_mode(stop_mode: str) -> str:
    normalized = str(stop_mode).strip().lower()
    if normalized in {"eos", "original", "legacy"}:
        return "eos"
    if normalized in {"budget", "budget_driven", "budget-driven"}:
        return "budget"
    raise ValueError(f"Unsupported SMC stop_mode: {stop_mode}")


def _normalize_budget_strategy(budget_strategy: str) -> str:
    normalized = str(budget_strategy).strip().lower()
    if normalized in {"restart", "restarts", "search"}:
        return "restart"
    raise ValueError(f"Unsupported SMC budget_strategy: {budget_strategy}")


def _normalize_selection_mode(selection_mode: str) -> str:
    normalized = str(selection_mode).strip().lower()
    if normalized in {"last", "best_logp", "vote", "weighted_vote"}:
        return normalized
    raise ValueError(f"Unsupported SMC selection_mode: {selection_mode}")


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


def _logsumexp(values: list[float]) -> float:
    if not values:
        return -float("inf")
    max_val = max(values)
    if not math.isfinite(max_val):
        return max_val
    return max_val + math.log(sum(math.exp(v - max_val) for v in values))


def _decode_completion_text(
    token_ids: list[int],
    *,
    context_len: int,
    decode_tokens: Optional[Callable[[list[int]], str]],
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
    token_ids: list[int],
    context_len: int,
    run_index: int,
    stop_reason: str,
    target_log_score: float,
    completed: bool,
    decode_tokens: Optional[Callable[[list[int]], str]],
    answer_extractor: Optional[Callable[[str], Any]],
) -> SMCCandidate:
    completion_text = _decode_completion_text(
        token_ids,
        context_len=context_len,
        decode_tokens=decode_tokens,
    )
    answer_key = _extract_answer_key(completion_text, answer_extractor=answer_extractor)
    return SMCCandidate(
        token_ids=list(token_ids),
        target_log_score=float(target_log_score),
        completion_text=completion_text,
        answer_key=answer_key,
        completed=bool(completed),
        run_index=int(run_index),
        stop_reason=stop_reason,
    )


def _select_candidate(
    *,
    completed_candidates: list[SMCCandidate],
    fallback_candidates: list[SMCCandidate],
    selection_mode: str,
) -> Tuple[Optional[SMCCandidate], str]:
    completed_pool = list(completed_candidates)
    fallback_pool = list(fallback_candidates)
    pool = completed_pool if completed_pool else fallback_pool
    if not pool:
        return None, "none"

    mode = _normalize_selection_mode(selection_mode)
    if mode == "last":
        return pool[-1], "last"

    if mode == "best_logp":
        best = max(pool, key=lambda cand: (cand.target_log_score, int(cand.completed), -cand.run_index))
        return best, "best_logp"

    grouped: Dict[str, list[SMCCandidate]] = {}
    for cand in pool:
        grouped.setdefault(cand.answer_key, []).append(cand)

    best_group_key = None
    best_group_tuple = None
    for group_key, group in grouped.items():
        group_scores = [cand.target_log_score for cand in group]
        group_best = max(group_scores)
        if mode == "vote":
            group_tuple = (len(group), group_best, -min(cand.run_index for cand in group), group_key)
        else:
            group_tuple = (
                _logsumexp(group_scores),
                len(group),
                group_best,
                -min(cand.run_index for cand in group),
                group_key,
            )
        if best_group_tuple is None or group_tuple > best_group_tuple:
            best_group_tuple = group_tuple
            best_group_key = group_key

    assert best_group_key is not None
    winning_group = grouped[best_group_key]
    best = max(winning_group, key=lambda cand: (cand.target_log_score, int(cand.completed), -cand.run_index))
    return best, mode


def _build_diagnostics(
    *,
    selected_result: Optional[SMCSingleRunResult],
    context_len: int,
    budget_tracker: SamplingBudgetTracker,
    total_steps: int,
    total_num_resamples: int,
    total_ess_sum: float,
    total_ess_count: int,
    stop_reason: str,
    stop_mode: str,
    budget_strategy: str,
    selection_mode: str,
    runs_started: int,
    completed_candidates: int,
    selected_candidate_source: str,
) -> Dict[str, Any]:
    selected_token_ids = [] if selected_result is None else selected_result.token_ids
    output_tokens = max(len(selected_token_ids) - context_len, 0)
    sampling_tokens = int(budget_tracker.sampling_tokens_used)
    avg_ess = float(total_ess_sum / max(total_ess_count, 1))
    diagnostics: Dict[str, Any] = {
        "steps": float(total_steps),
        "output_tokens": float(output_tokens),
        "sampling_tokens": float(sampling_tokens),
        "internal_sampling_tokens": float(max(sampling_tokens - output_tokens, 0)),
        "num_particles": float(0 if selected_result is None else selected_result.num_particles),
        "num_resamples": float(total_num_resamples),
        "avg_ess": avg_ess,
        "final_ess": float(0.0 if selected_result is None else selected_result.final_ess),
        "smc_stop_mode": stop_mode,
        "smc_budget_strategy": budget_strategy,
        "smc_selection_mode": selection_mode,
        "smc_runs_started": int(runs_started),
        "smc_completed_candidates": int(completed_candidates),
        "smc_selected_candidate_source": selected_candidate_source,
        "budget_stop_reason": stop_reason,
        "ess_threshold": 0.0,
        "resample_interval": 0.0,
        "resample_method": "",
        "proposal_temperature": 0.0,
        "proposal_top_k": None,
        "proposal_top_p": 0.0,
        "use_auxiliary": False,
        "auxiliary_resample_always": False,
        "auxiliary_temperature": None,
        "auxiliary_alpha": 0.0,
        "temperature": 0.0,
        "alpha": 0.0,
    }
    diagnostics.update(budget_tracker.metadata())
    return diagnostics


@torch.inference_mode()
def _run_single_smc_filter(
    model: Any,
    tokenizer: Any,
    device: str,
    context: list[int],
    *,
    cfg: PowerSamplerSMCConfig,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    budget_tracker: SamplingBudgetTracker,
    run_index: int,
    decode_tokens: Optional[Callable[[list[int]], str]],
    answer_extractor: Optional[Callable[[str], Any]],
) -> SMCSingleRunResult:
    alpha = 1.0 / float(cfg.temp)
    prompt = list(context)
    context_len = len(prompt)
    if not prompt:
        raise ValueError("context must contain at least one token.")

    max_new_tokens = max(0, int(max_new_tokens))
    if max_new_tokens == 0:
        candidate = _build_candidate(
            token_ids=prompt.copy(),
            context_len=context_len,
            run_index=run_index,
            stop_reason="max_new_tokens",
            target_log_score=0.0,
            completed=False,
            decode_tokens=decode_tokens,
            answer_extractor=answer_extractor,
        )
        return SMCSingleRunResult(
            token_ids=prompt.copy(),
            target_log_score=0.0,
            steps_done=0,
            output_tokens=0,
            num_particles=0,
            num_resamples=0,
            ess_sum=0.0,
            ess_count=0,
            final_ess=0.0,
            stop_reason="max_new_tokens",
            candidate=candidate,
        )

    max_seq_len = getattr(getattr(model, "config", None), "max_position_embeddings", None)
    if max_seq_len is None:
        max_seq_len = getattr(getattr(model, "config", None), "n_positions", None)
    if max_seq_len is not None:
        allowed_new_tokens = int(max_seq_len) - len(prompt)
        max_new_tokens = max(0, min(max_new_tokens, allowed_new_tokens))
        if max_new_tokens == 0:
            candidate = _build_candidate(
                token_ids=prompt.copy(),
                context_len=context_len,
                run_index=run_index,
                stop_reason="max_new_tokens",
                target_log_score=0.0,
                completed=False,
                decode_tokens=decode_tokens,
                answer_extractor=answer_extractor,
            )
            return SMCSingleRunResult(
                token_ids=prompt.copy(),
                target_log_score=0.0,
                steps_done=0,
                output_tokens=0,
                num_particles=0,
                num_resamples=0,
                ess_sum=0.0,
                ess_count=0,
                final_ess=0.0,
                stop_reason="max_new_tokens",
                candidate=candidate,
            )

    configured_particles = max(1, int(cfg.num_particles))
    remaining_budget = budget_tracker.remaining_sampling_tokens()
    if remaining_budget is not None:
        if remaining_budget <= 0:
            candidate = _build_candidate(
                token_ids=prompt.copy(),
                context_len=context_len,
                run_index=run_index,
                stop_reason="sampling_budget_exhausted",
                target_log_score=0.0,
                completed=False,
                decode_tokens=decode_tokens,
                answer_extractor=answer_extractor,
            )
            return SMCSingleRunResult(
                token_ids=prompt.copy(),
                target_log_score=0.0,
                steps_done=0,
                output_tokens=0,
                num_particles=0,
                num_resamples=0,
                ess_sum=0.0,
                ess_count=0,
                final_ess=0.0,
                stop_reason="sampling_budget_exhausted",
                candidate=candidate,
            )
        configured_particles = min(configured_particles, int(remaining_budget))

    eos_id = eos_token_id if eos_token_id is not None else getattr(tokenizer, "eos_token_id", None)
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
    batch_input_ids = prompt_ids.unsqueeze(0).repeat(configured_particles, 1).to(device)
    prompt_len = int(batch_input_ids.shape[1])
    total_len = prompt_len + max_new_tokens
    attention_mask = torch.ones((configured_particles, total_len), dtype=torch.long, device=batch_input_ids.device)

    generator: Optional[torch.Generator] = None
    if cfg.seed is not None:
        try:
            generator = torch.Generator(device=batch_input_ids.device)
        except Exception:
            generator = torch.Generator()
        generator.manual_seed(int(cfg.seed) + int(run_index) - 1)

    outputs = model(
        input_ids=batch_input_ids,
        attention_mask=attention_mask[:, :prompt_len],
        use_cache=True,
    )
    next_logits = outputs.logits[:, -1, :].float()
    past_key_values = outputs.past_key_values

    generated = torch.full(
        (configured_particles, max_new_tokens),
        pad_id_int,
        dtype=torch.long,
        device=batch_input_ids.device,
    )
    finished = torch.zeros((configured_particles,), dtype=torch.bool, device=batch_input_ids.device)
    logw = torch.zeros((configured_particles,), dtype=torch.float32, device=batch_input_ids.device)
    path_logp = torch.zeros((configured_particles,), dtype=torch.float32, device=batch_input_ids.device)

    num_resamples = 0
    ess_sum = 0.0
    ess_count = 0
    steps_done = 0
    stop_reason = "max_new_tokens"

    proposal_temperature = float(cfg.proposal_temperature)
    if proposal_temperature <= 0:
        raise ValueError("cfg.proposal_temperature must be > 0")

    resample_interval = max(1, int(cfg.resample_interval))
    ess_threshold = float(cfg.ess_threshold)
    max_logw_step = float(cfg.max_logw_step)
    alpha_aux = alpha
    if cfg.auxiliary_temperature is not None:
        aux_temp = float(cfg.auxiliary_temperature)
        if aux_temp <= 0:
            raise ValueError("cfg.auxiliary_temperature must be > 0")
        alpha_aux = 1.0 / aux_temp

    eos_tensor: Optional[torch.Tensor] = None
    if eos_id_int is not None:
        eos_tensor = torch.tensor(eos_id_int, device=batch_input_ids.device, dtype=torch.long)

    for step_idx in range(max_new_tokens):
        if budget_tracker.exhausted():
            stop_reason = "sampling_budget_exhausted"
            break

        log_m_selected: Optional[torch.Tensor] = None
        if bool(cfg.use_auxiliary) and configured_particles > 1 and (step_idx % resample_interval == 0):
            log_z = torch.logsumexp(next_logits, dim=-1)
            log_z_aux = torch.logsumexp(next_logits * float(alpha_aux), dim=-1)
            log_m = log_z_aux - float(alpha_aux) * log_z
            log_m = torch.where(torch.isfinite(log_m), log_m, torch.zeros_like(log_m))
            log_m = torch.where(finished, torch.zeros_like(log_m), log_m)

            aux_weights = _normalized_weights(logw + log_m)
            ess_val = _ess(aux_weights)
            ess_sum += ess_val
            ess_count += 1

            should_resample = bool(cfg.auxiliary_resample_always)
            if not should_resample and ess_threshold > 0.0:
                should_resample = (ess_val / float(configured_particles)) < ess_threshold

            if should_resample:
                indices = _resample_indices(aux_weights, method=cfg.resample_method, generator=generator)
                generated = generated.index_select(0, indices)
                finished = finished.index_select(0, indices)
                path_logp = path_logp.index_select(0, indices)
                past_key_values = _reorder_past_key_values(model, past_key_values, indices)
                next_logits = next_logits.index_select(0, indices)
                log_m_selected = log_m.index_select(0, indices)
                logw = torch.zeros_like(logw)
                num_resamples += 1

        already_finished = finished
        alive_count = int((~already_finished).sum().item())
        if alive_count <= 0:
            stop_reason = "all_particles_eos"
            break

        remaining_budget = budget_tracker.remaining_sampling_tokens()
        if remaining_budget is not None and alive_count > int(remaining_budget):
            stop_reason = "sampling_budget_exhausted"
            break

        log_p = F.log_softmax(next_logits, dim=-1)
        proposal_logits = next_logits
        if proposal_temperature != 1.0:
            proposal_logits = proposal_logits / proposal_temperature
        proposal_logits = _apply_top_k(proposal_logits, cfg.proposal_top_k)
        proposal_logits = _apply_top_p(proposal_logits, cfg.proposal_top_p)
        log_q = F.log_softmax(proposal_logits, dim=-1)
        log_q = _sanitize_log_q(log_q, log_p)

        sampled_ids = torch.multinomial(torch.exp(log_q), num_samples=1, generator=generator).squeeze(1)
        budget_tracker.spend(alive_count)

        if eos_tensor is not None:
            sampled_ids = torch.where(already_finished, eos_tensor, sampled_ids)

        log_p_tok = log_p.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
        log_q_tok = log_q.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)
        path_logp = path_logp + torch.where(already_finished, torch.zeros_like(log_p_tok), log_p_tok)

        increment = alpha * log_p_tok - log_q_tok
        if log_m_selected is not None:
            increment = increment - log_m_selected
        increment = torch.where(already_finished, torch.zeros_like(increment), increment)
        if max_logw_step > 0:
            increment = torch.clamp(increment, min=-max_logw_step, max=max_logw_step)
        logw = logw + increment

        generated[:, step_idx] = sampled_ids
        if eos_id_int is not None:
            finished = already_finished | sampled_ids.eq(eos_id_int)

        steps_done = step_idx + 1

        if not bool(cfg.use_auxiliary):
            should_resample = False
            normalized = None

            if configured_particles > 1 and ((step_idx + 1) % resample_interval == 0):
                normalized = _normalized_weights(logw)
                ess_val = _ess(normalized)
                ess_sum += ess_val
                ess_count += 1
                if ess_threshold > 0.0 and (ess_val / float(configured_particles)) < ess_threshold:
                    should_resample = True

            if should_resample and normalized is not None:
                indices = _resample_indices(normalized, method=cfg.resample_method, generator=generator)
                generated = generated.index_select(0, indices)
                finished = finished.index_select(0, indices)
                path_logp = path_logp.index_select(0, indices)
                past_key_values = _reorder_past_key_values(model, past_key_values, indices)
                sampled_ids = generated[:, step_idx]
                logw = torch.zeros_like(logw)
                num_resamples += 1

        if bool(cfg.stop_on_all_eos) and eos_id_int is not None and bool(torch.all(finished).item()):
            stop_reason = "all_particles_eos"
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
    completed = bool(eos_id_int is not None and eos_id_int in chosen)
    if completed and eos_id_int is not None:
        eos_pos = chosen.index(eos_id_int)
        chosen = chosen[: eos_pos + 1]

    out = prompt + [int(tok) for tok in chosen]
    output_tokens = max(len(out) - len(prompt), 0)
    target_log_score = float(alpha * float(path_logp[final_idx].item()))

    candidate = _build_candidate(
        token_ids=out,
        context_len=context_len,
        run_index=run_index,
        stop_reason=stop_reason,
        target_log_score=target_log_score,
        completed=completed,
        decode_tokens=decode_tokens,
        answer_extractor=answer_extractor,
    )
    return SMCSingleRunResult(
        token_ids=out,
        target_log_score=target_log_score,
        steps_done=steps_done,
        output_tokens=output_tokens,
        num_particles=configured_particles,
        num_resamples=num_resamples,
        ess_sum=ess_sum,
        ess_count=ess_count,
        final_ess=float(final_ess),
        stop_reason=stop_reason,
        candidate=candidate,
    )


def _run_budget_restart_search(
    model: Any,
    tokenizer: Any,
    device: str,
    context: list[int],
    *,
    cfg: PowerSamplerSMCConfig,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    pad_token_id: Optional[int],
    budget_tracker: SamplingBudgetTracker,
    selection_mode: str,
    decode_tokens: Optional[Callable[[list[int]], str]],
    answer_extractor: Optional[Callable[[str], Any]],
) -> Tuple[list[int], Dict[str, Any]]:
    context_len = len(context)
    runs_started = 0
    total_steps = 0
    total_num_resamples = 0
    total_ess_sum = 0.0
    total_ess_count = 0
    completed_candidates: list[SMCCandidate] = []
    fallback_candidates: list[SMCCandidate] = []
    results_by_run: Dict[int, SMCSingleRunResult] = {}
    stop_reason = "sampling_budget_exhausted"

    while not budget_tracker.exhausted():
        runs_started += 1
        before = int(budget_tracker.sampling_tokens_used)
        result = _run_single_smc_filter(
            model,
            tokenizer,
            device,
            context,
            cfg=cfg,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            budget_tracker=budget_tracker,
            run_index=runs_started,
            decode_tokens=decode_tokens,
            answer_extractor=answer_extractor,
        )
        results_by_run[runs_started] = result
        total_steps += int(result.steps_done)
        total_num_resamples += int(result.num_resamples)
        total_ess_sum += float(result.ess_sum)
        total_ess_count += int(result.ess_count)
        fallback_candidates.append(result.candidate)
        if result.candidate.completed:
            completed_candidates.append(result.candidate)

        spent = int(budget_tracker.sampling_tokens_used) - before
        stop_reason = str(result.stop_reason)
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
    selected_result: Optional[SMCSingleRunResult] = None
    if selected_candidate is None:
        selected_candidate = _build_candidate(
            token_ids=context.copy(),
            context_len=context_len,
            run_index=0,
            stop_reason="no_progress",
            target_log_score=0.0,
            completed=False,
            decode_tokens=decode_tokens,
            answer_extractor=answer_extractor,
        )
        selected_source = "prompt_fallback"
    else:
        selected_result = results_by_run.get(int(selected_candidate.run_index))

    if selected_result is None and int(selected_candidate.run_index) == 0:
        selected_result = SMCSingleRunResult(
            token_ids=selected_candidate.token_ids,
            target_log_score=selected_candidate.target_log_score,
            steps_done=0,
            output_tokens=max(len(selected_candidate.token_ids) - context_len, 0),
            num_particles=0,
            num_resamples=0,
            ess_sum=0.0,
            ess_count=0,
            final_ess=0.0,
            stop_reason="no_progress",
            candidate=selected_candidate,
        )

    diagnostics = _build_diagnostics(
        selected_result=selected_result,
        context_len=context_len,
        budget_tracker=budget_tracker,
        total_steps=total_steps,
        total_num_resamples=total_num_resamples,
        total_ess_sum=total_ess_sum,
        total_ess_count=total_ess_count,
        stop_reason=stop_reason,
        stop_mode="budget",
        budget_strategy="restart",
        selection_mode=selection_mode,
        runs_started=runs_started,
        completed_candidates=len(completed_candidates),
        selected_candidate_source=selected_source,
    )
    diagnostics.update(
        {
            "ess_threshold": float(cfg.ess_threshold),
            "resample_interval": float(max(1, int(cfg.resample_interval))),
            "resample_method": str(cfg.resample_method),
            "proposal_temperature": float(cfg.proposal_temperature),
            "proposal_top_k": None if cfg.proposal_top_k is None else float(int(cfg.proposal_top_k)),
            "proposal_top_p": float(cfg.proposal_top_p),
            "use_auxiliary": bool(cfg.use_auxiliary),
            "auxiliary_resample_always": bool(cfg.auxiliary_resample_always),
            "auxiliary_temperature": None
            if cfg.auxiliary_temperature is None
            else float(cfg.auxiliary_temperature),
            "auxiliary_alpha": float(1.0 / float(cfg.auxiliary_temperature))
            if cfg.auxiliary_temperature is not None
            else float(1.0 / float(cfg.temp)),
            "temperature": float(cfg.temp),
            "alpha": float(1.0 / float(cfg.temp)),
        }
    )
    if selected_result is not None:
        diagnostics.update(
            {
                "num_particles": float(selected_result.num_particles),
                "final_ess": float(selected_result.final_ess),
            }
        )
    return selected_candidate.token_ids, diagnostics


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
    stop_mode: str = "eos",
    budget_strategy: str = "restart",
    selection_mode: str = "best_logp",
    answer_extractor: Optional[Callable[[str], Any]] = None,
    decode_tokens: Optional[Callable[[list[int]], str]] = None,
    budget_config: Optional[SamplingBudgetConfig] = None,
) -> Tuple[list[int], Dict[str, Any]]:
    """Sample from the power distribution using an SMC/Feynman--Kac filter.

    Returns:
      - token ids for full sequence (prompt + generated, truncated at first EOS)
      - diagnostics dictionary
    """

    if cfg.temp <= 0:
        raise ValueError("cfg.temp must be > 0 (alpha = 1/cfg.temp).")
    stop_mode = _normalize_stop_mode(stop_mode)
    budget_strategy = _normalize_budget_strategy(budget_strategy)
    selection_mode = _normalize_selection_mode(selection_mode)
    budget_tracker = SamplingBudgetTracker(budget_config)

    if stop_mode == "budget" and budget_tracker.max_sampling_tokens is None:
        raise ValueError("SMC stop_mode='budget' requires max_sampling_tokens to be set.")

    if stop_mode == "budget" and budget_strategy == "restart":
        return _run_budget_restart_search(
            model,
            tokenizer,
            device,
            context,
            cfg=cfg,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            budget_tracker=budget_tracker,
            selection_mode=selection_mode,
            decode_tokens=decode_tokens,
            answer_extractor=answer_extractor,
        )

    single = _run_single_smc_filter(
        model,
        tokenizer,
        device,
        context,
        cfg=cfg,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        budget_tracker=budget_tracker,
        run_index=1,
        decode_tokens=decode_tokens,
        answer_extractor=answer_extractor,
    )
    diagnostics = _build_diagnostics(
        selected_result=single,
        context_len=len(context),
        budget_tracker=budget_tracker,
        total_steps=int(single.steps_done),
        total_num_resamples=int(single.num_resamples),
        total_ess_sum=float(single.ess_sum),
        total_ess_count=int(single.ess_count),
        stop_reason=str(single.stop_reason),
        stop_mode=stop_mode,
        budget_strategy="none" if stop_mode == "eos" else budget_strategy,
        selection_mode="last",
        runs_started=1,
        completed_candidates=1 if single.candidate.completed else 0,
        selected_candidate_source="last",
    )
    diagnostics.update(
        {
            "num_particles": float(single.num_particles),
            "final_ess": float(single.final_ess),
            "ess_threshold": float(cfg.ess_threshold),
            "resample_interval": float(max(1, int(cfg.resample_interval))),
            "resample_method": str(cfg.resample_method),
            "proposal_temperature": float(cfg.proposal_temperature),
            "proposal_top_k": None if cfg.proposal_top_k is None else float(int(cfg.proposal_top_k)),
            "proposal_top_p": float(cfg.proposal_top_p),
            "use_auxiliary": bool(cfg.use_auxiliary),
            "auxiliary_resample_always": bool(cfg.auxiliary_resample_always),
            "auxiliary_temperature": None
            if cfg.auxiliary_temperature is None
            else float(cfg.auxiliary_temperature),
            "auxiliary_alpha": float(1.0 / float(cfg.auxiliary_temperature))
            if cfg.auxiliary_temperature is not None
            else float(1.0 / float(cfg.temp)),
            "temperature": float(cfg.temp),
            "alpha": float(1.0 / float(cfg.temp)),
        }
    )
    return single.token_ids, diagnostics
