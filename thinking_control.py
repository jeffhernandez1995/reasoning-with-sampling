from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence

import torch
import transformers


DEFAULT_EARLY_STOPPING_TEXT = (
    "\n\nConsidering the limited time by the user, I have to give the solution "
    "based on the thinking directly now.\n</think>\n\n"
)


@dataclass
class ThinkingControlConfig:
    mode: str = "none"
    max_thinking_tokens: Optional[int] = None
    min_thinking_tokens: int = 0
    ignore_eot_attempts: int = 0
    eot_trigger_topk: int = 1
    wait_text: str = "\nWait\n"
    early_stopping_text: str = DEFAULT_EARLY_STOPPING_TEXT
    extra_generation_tokens: int = 0


@dataclass
class ControlledGenerationOutput:
    token_ids: List[int]
    latency_seconds: float
    metadata: Dict[str, Any]


def _encode_text(tokenizer, text: str) -> List[int]:
    if not text:
        return []
    return [int(tok) for tok in tokenizer.encode(text, add_special_tokens=False)]


def _single_token_id(tokenizer, text: str) -> Optional[int]:
    ids = _encode_text(tokenizer, text)
    if len(ids) != 1:
        return None
    return int(ids[0])


def _safe_generate(model, **kwargs):
    if "min_p" in kwargs and kwargs["min_p"] is None:
        kwargs.pop("min_p", None)
    try:
        return model.generate(**kwargs)
    except TypeError as exc:
        if "min_p" in str(exc) and "min_p" in kwargs:
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("min_p", None)
            return model.generate(**retry_kwargs)
        raise
    except ValueError as exc:
        msg = str(exc)
        if "min_p" in msg and "min_p" in kwargs:
            retry_kwargs = dict(kwargs)
            retry_kwargs.pop("min_p", None)
            return model.generate(**retry_kwargs)
        raise


def _to_long_tensor(token_ids: Sequence[int], device) -> torch.LongTensor:
    return torch.tensor([list(token_ids)], dtype=torch.long, device=device)


def _last_index(token_ids: Sequence[int], target_id: int) -> int:
    for idx in range(len(token_ids) - 1, -1, -1):
        if int(token_ids[idx]) == int(target_id):
            return idx
    return -1


class Qwen3BudgetForcingLogitsProcessor(transformers.LogitsProcessor):
    MODE_NONE = 0
    MODE_EARLY = 1
    MODE_WAIT = 2

    def __init__(
        self,
        tokenizer,
        prompt_len: int,
        think_end_id: int,
        *,
        max_thinking_tokens: Optional[int],
        min_thinking_tokens: int,
        ignore_eot_attempts: int,
        eot_trigger_topk: int,
        wait_text: str,
        early_stopping_text: str,
    ):
        self.prompt_len = int(prompt_len)
        self.think_end_id = int(think_end_id)
        self.max_thinking_tokens = None if max_thinking_tokens is None else int(max_thinking_tokens)
        self.min_thinking_tokens = max(0, int(min_thinking_tokens))
        self.ignore_eot_attempts = max(0, int(ignore_eot_attempts))
        self.eot_trigger_topk = max(1, int(eot_trigger_topk))
        self.think_start_id = _single_token_id(tokenizer, "<think>")
        self.wait_ids = _encode_text(tokenizer, wait_text)
        self.early_ids = _encode_text(tokenizer, early_stopping_text)

        self._initialized = False
        self._in_think = None
        self._think_start_pos = None
        self._ignore_left = None
        self._force_mode = None
        self._force_idx = None

        self.ignored_eot_count = 0
        self.forced_wait_count = 0
        self.forced_early_stop_count = 0

    def _init_state(self, input_ids: torch.LongTensor):
        bs, _ = input_ids.shape
        device = input_ids.device
        self._in_think = torch.zeros(bs, dtype=torch.bool, device=device)
        self._think_start_pos = torch.full((bs,), -1, dtype=torch.long, device=device)
        self._ignore_left = torch.full((bs,), self.ignore_eot_attempts, dtype=torch.long, device=device)
        self._force_mode = torch.full((bs,), self.MODE_NONE, dtype=torch.long, device=device)
        self._force_idx = torch.zeros(bs, dtype=torch.long, device=device)

        prompt_ids = input_ids[:, : self.prompt_len]
        for row_idx in range(bs):
            row = [int(tok) for tok in prompt_ids[row_idx].tolist()]
            last_end = _last_index(row, self.think_end_id)
            if self.think_start_id is None:
                self._in_think[row_idx] = last_end < 0
                self._think_start_pos[row_idx] = max(self.prompt_len - 1, 0)
                continue
            last_start = _last_index(row, int(self.think_start_id))
            self._in_think[row_idx] = last_start > last_end
            self._think_start_pos[row_idx] = last_start

        self._initialized = True

    def _force_sequence_token(self, scores: torch.FloatTensor, row_idx: int, seq_ids: List[int], neg_inf: float):
        if not seq_ids:
            self._force_mode[row_idx] = self.MODE_NONE
            self._force_idx[row_idx] = 0
            return

        token_idx = int(self._force_idx[row_idx].item())
        if token_idx >= len(seq_ids):
            self._force_mode[row_idx] = self.MODE_NONE
            self._force_idx[row_idx] = 0
            return

        token_id = int(seq_ids[token_idx])
        scores[row_idx, :] = neg_inf
        scores[row_idx, token_id] = 0.0
        self._force_idx[row_idx] = token_idx + 1
        if int(self._force_idx[row_idx].item()) >= len(seq_ids):
            self._force_mode[row_idx] = self.MODE_NONE
            self._force_idx[row_idx] = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if (not self._initialized) or (self._in_think is None) or (self._in_think.numel() != input_ids.shape[0]):
            self._init_state(input_ids)

        bs, seq_len = input_ids.shape
        device = input_ids.device
        neg_inf = torch.finfo(scores.dtype).min
        last_tok = input_ids[:, -1]

        if self.think_start_id is not None:
            started_now = (~self._in_think) & last_tok.eq(int(self.think_start_id))
            if started_now.any():
                self._in_think[started_now] = True
                self._think_start_pos[started_now] = int(seq_len - 1)

        ended_now = self._in_think & last_tok.eq(self.think_end_id)
        if ended_now.any():
            self._in_think[ended_now] = False

        forcing = self._force_mode != self.MODE_NONE
        if forcing.any():
            rows = forcing.nonzero(as_tuple=False).squeeze(-1).tolist()
            for row_idx in rows:
                mode = int(self._force_mode[row_idx].item())
                seq_ids = self.early_ids if mode == self.MODE_EARLY else self.wait_ids
                self._force_sequence_token(scores, row_idx, seq_ids, neg_inf)
            return scores

        active = self._in_think
        if not active.any():
            return scores

        think_len = torch.clamp((seq_len - 1) - self._think_start_pos, min=0)

        if self.max_thinking_tokens is not None:
            over = active & (think_len >= int(self.max_thinking_tokens))
            if over.any():
                rows = over.nonzero(as_tuple=False).squeeze(-1).tolist()
                for row_idx in rows:
                    self._force_mode[row_idx] = self.MODE_EARLY
                    self._force_idx[row_idx] = 0
                    self.forced_early_stop_count += 1
                    self._force_sequence_token(scores, row_idx, self.early_ids, neg_inf)
                return scores

        if self.min_thinking_tokens > 0:
            too_short = active & (think_len < int(self.min_thinking_tokens))
            if too_short.any():
                scores[too_short, self.think_end_id] = neg_inf

        can_ignore = active & (self._ignore_left > 0)
        if not can_ignore.any():
            return scores

        wants_end = torch.zeros(bs, dtype=torch.bool, device=device)
        if self.eot_trigger_topk <= 1:
            top1 = scores.argmax(dim=-1)
            wants_end = can_ignore & top1.eq(self.think_end_id)
        else:
            k = min(self.eot_trigger_topk, int(scores.shape[-1]))
            rows = can_ignore.nonzero(as_tuple=False).squeeze(-1)
            topk_ids = scores[rows].topk(k=k, dim=-1).indices
            in_topk = topk_ids.eq(self.think_end_id).any(dim=-1)
            wants_end[rows] = in_topk

        if wants_end.any():
            rows = wants_end.nonzero(as_tuple=False).squeeze(-1).tolist()
            for row_idx in rows:
                self._ignore_left[row_idx] = max(int(self._ignore_left[row_idx].item()) - 1, 0)
                self._force_mode[row_idx] = self.MODE_WAIT
                self._force_idx[row_idx] = 0
                self.ignored_eot_count += 1
                self.forced_wait_count += 1
                self._force_sequence_token(scores, row_idx, self.wait_ids, neg_inf)
            return scores

        return scores


def _generate_single_pass(
    *,
    model,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    generation_kwargs: Dict[str, Any],
) -> List[int]:
    prompt_len = int(input_ids.shape[1])
    call_kwargs = dict(generation_kwargs)
    call_kwargs["max_new_tokens"] = int(max_new_tokens)
    output = _safe_generate(model, input_ids=input_ids, **call_kwargs)
    return output[0][prompt_len:].detach().cpu().tolist()


def _generate_multi_pass(
    *,
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    generation_kwargs: Dict[str, Any],
    think_end_id: int,
    control: ThinkingControlConfig,
) -> ControlledGenerationOutput:
    device = input_ids.device
    prompt_token_ids = [int(tok) for tok in input_ids[0].detach().cpu().tolist()]
    eos_id = tokenizer.eos_token_id

    max_new_tokens = max(0, int(max_new_tokens))
    extra_tokens = max(0, int(control.extra_generation_tokens))
    generation_limit = max_new_tokens + extra_tokens

    think_start_id = _single_token_id(tokenizer, "<think>")
    in_think = True
    if think_start_id is not None:
        last_start = _last_index(prompt_token_ids, int(think_start_id))
        last_end = _last_index(prompt_token_ids, int(think_end_id))
        in_think = last_start > last_end

    wait_ids = _encode_text(tokenizer, control.wait_text)
    early_ids = _encode_text(tokenizer, control.early_stopping_text)
    if not early_ids:
        early_ids = [int(think_end_id)]
    elif int(think_end_id) not in early_ids:
        early_ids = early_ids + [int(think_end_id)]

    ignore_left = max(0, int(control.ignore_eot_attempts))
    continuation: List[int] = []
    ignored_eot_count = 0
    forced_wait_count = 0
    forced_early_stop_count = 0

    def thinking_len() -> int:
        if int(think_end_id) in continuation:
            end_idx = continuation.index(int(think_end_id))
            return end_idx + 1
        return len(continuation)

    start = perf_counter()
    max_loops = max(32, generation_limit + 8)
    loop_count = 0
    while len(continuation) < generation_limit and loop_count < max_loops:
        loop_count += 1
        remaining_total = generation_limit - len(continuation)
        if remaining_total <= 0:
            break

        if in_think:
            cur_think_len = thinking_len()
            if control.max_thinking_tokens is not None and cur_think_len >= int(control.max_thinking_tokens):
                inject = early_ids[:remaining_total]
                if inject:
                    continuation.extend(inject)
                    forced_early_stop_count += 1
                    if int(think_end_id) in inject:
                        in_think = False
                        continue
                break

            phase_budget = remaining_total
            if control.max_thinking_tokens is not None:
                remaining_think = int(control.max_thinking_tokens) - cur_think_len
                phase_budget = min(phase_budget, max(remaining_think, 0))
            if phase_budget <= 0:
                break

            eos_ids = [int(think_end_id)]
            if eos_id is not None and int(eos_id) != int(think_end_id):
                eos_ids.append(int(eos_id))

            phase_input = _to_long_tensor(prompt_token_ids + continuation, device=device)
            phase_kwargs = dict(generation_kwargs)
            phase_kwargs["eos_token_id"] = eos_ids
            new_tokens = _generate_single_pass(
                model=model,
                input_ids=phase_input,
                max_new_tokens=phase_budget,
                generation_kwargs=phase_kwargs,
            )
            if not new_tokens:
                break

            continuation.extend([int(tok) for tok in new_tokens])
            saw_end_think = int(think_end_id) in new_tokens
            hit_eos = eos_id is not None and int(eos_id) in new_tokens

            if saw_end_think:
                need_min = thinking_len() < int(control.min_thinking_tokens)
                can_ignore = ignore_left > 0
                if need_min or can_ignore:
                    end_idx = _last_index(continuation, int(think_end_id))
                    if end_idx >= 0:
                        del continuation[end_idx]
                    if can_ignore:
                        ignore_left -= 1
                    ignored_eot_count += 1

                    remaining_after_pop = generation_limit - len(continuation)
                    inject_wait = wait_ids[:remaining_after_pop]
                    if inject_wait:
                        continuation.extend(inject_wait)
                        forced_wait_count += 1
                    in_think = True
                    continue

                in_think = False
                continue

            if hit_eos:
                break
            continue

        phase_input = _to_long_tensor(prompt_token_ids + continuation, device=device)
        phase_kwargs = dict(generation_kwargs)
        if eos_id is not None:
            phase_kwargs["eos_token_id"] = int(eos_id)
        new_tokens = _generate_single_pass(
            model=model,
            input_ids=phase_input,
            max_new_tokens=remaining_total,
            generation_kwargs=phase_kwargs,
        )
        if not new_tokens:
            break
        continuation.extend([int(tok) for tok in new_tokens])
        if eos_id is not None and int(eos_id) in new_tokens:
            break

    latency_seconds = perf_counter() - start
    metadata = {
        "thinking_control_mode": "multi_pass",
        "thinking_control_active": True,
        "thinking_budget_max_tokens": control.max_thinking_tokens,
        "thinking_budget_min_tokens": int(control.min_thinking_tokens),
        "thinking_ignore_eot_attempts": int(control.ignore_eot_attempts),
        "thinking_ignore_eot_remaining": int(ignore_left),
        "thinking_ignored_eot_count": int(ignored_eot_count),
        "thinking_forced_wait_count": int(forced_wait_count),
        "thinking_forced_early_stop_count": int(forced_early_stop_count),
        "thinking_generation_limit_tokens": int(generation_limit),
        "thinking_generated_tokens": int(len(continuation)),
        "thinking_reached_generation_limit": bool(len(continuation) >= generation_limit),
    }
    return ControlledGenerationOutput(
        token_ids=continuation,
        latency_seconds=latency_seconds,
        metadata=metadata,
    )


def generate_with_thinking_control(
    *,
    model,
    tokenizer,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    generation_kwargs: Optional[Dict[str, Any]],
    think_end_id: Optional[int],
    control: ThinkingControlConfig,
) -> ControlledGenerationOutput:
    generation_kwargs = dict(generation_kwargs or {})
    mode = str(control.mode).strip().lower()
    control_active = mode != "none" and think_end_id is not None

    if not control_active:
        start = perf_counter()
        token_ids = _generate_single_pass(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
        )
        latency_seconds = perf_counter() - start
        metadata = {
            "thinking_control_mode": "none",
            "thinking_control_active": False,
            "thinking_generated_tokens": int(len(token_ids)),
        }
        return ControlledGenerationOutput(
            token_ids=token_ids,
            latency_seconds=latency_seconds,
            metadata=metadata,
        )

    if mode == "multi_pass":
        return _generate_multi_pass(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            generation_kwargs=generation_kwargs,
            think_end_id=int(think_end_id),
            control=control,
        )

    if mode != "logits":
        raise ValueError(f"Unsupported thinking control mode: {control.mode}")

    prompt_len = int(input_ids.shape[1])
    budget_processor = Qwen3BudgetForcingLogitsProcessor(
        tokenizer=tokenizer,
        prompt_len=prompt_len,
        think_end_id=int(think_end_id),
        max_thinking_tokens=control.max_thinking_tokens,
        min_thinking_tokens=control.min_thinking_tokens,
        ignore_eot_attempts=control.ignore_eot_attempts,
        eot_trigger_topk=control.eot_trigger_topk,
        wait_text=control.wait_text,
        early_stopping_text=control.early_stopping_text,
    )

    call_kwargs = dict(generation_kwargs)
    existing_processors = call_kwargs.pop("logits_processor", None)
    if existing_processors is None:
        processors = transformers.LogitsProcessorList()
    elif isinstance(existing_processors, transformers.LogitsProcessorList):
        processors = existing_processors
    else:
        processors = transformers.LogitsProcessorList(list(existing_processors))
    processors.append(budget_processor)
    call_kwargs["logits_processor"] = processors

    generation_limit = max(0, int(max_new_tokens)) + max(0, int(control.extra_generation_tokens))
    start = perf_counter()
    token_ids = _generate_single_pass(
        model=model,
        input_ids=input_ids,
        max_new_tokens=generation_limit,
        generation_kwargs=call_kwargs,
    )
    latency_seconds = perf_counter() - start
    metadata = {
        "thinking_control_mode": "logits",
        "thinking_control_active": True,
        "thinking_budget_max_tokens": control.max_thinking_tokens,
        "thinking_budget_min_tokens": int(control.min_thinking_tokens),
        "thinking_ignore_eot_attempts": int(control.ignore_eot_attempts),
        "thinking_ignored_eot_count": int(budget_processor.ignored_eot_count),
        "thinking_forced_wait_count": int(budget_processor.forced_wait_count),
        "thinking_forced_early_stop_count": int(budget_processor.forced_early_stop_count),
        "thinking_generation_limit_tokens": int(generation_limit),
        "thinking_generated_tokens": int(len(token_ids)),
        "thinking_reached_generation_limit": bool(len(token_ids) >= generation_limit),
    }
    return ControlledGenerationOutput(
        token_ids=token_ids,
        latency_seconds=latency_seconds,
        metadata=metadata,
    )
