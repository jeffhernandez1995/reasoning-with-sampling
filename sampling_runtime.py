import json
import math
import os
from dataclasses import asdict, dataclass, field
from numbers import Number
from time import perf_counter
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import transformers

from algos.power_sampling_approx import PowerSamplerApproxConfig, approx_power_sample
from algos.power_sampling_mcmc import AutoregressiveSampler, mcmc_power_samp


MODEL_NAME_BY_ALIAS = {
    "qwen": "Qwen/Qwen2.5-7B",
    "qwen_math": "Qwen/Qwen2.5-Math-7B",
    "qwen_math_grpo": "stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
    "phi": "microsoft/Phi-3.5-mini-instruct",
    "phi_grpo": "microsoft/Phi-3.5-mini-instruct",
    "tulu": "allenai/Llama-3.1-Tulu-3-8B-DPO",
    # Kept for backwards compatibility with historical scripts.
    "qwen_grpo": "Qwen/Qwen2.5-7B",
}


@dataclass
class SamplingOutput:
    method: str
    completion: str
    token_ids: List[int]
    latency_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    full_completion: Optional[str] = None
    full_token_ids: Optional[List[int]] = None


class _ApproxPowerHFScorer:
    """Hugging Face adapter used by approximate power sampling."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self._max_seq_len = getattr(model.config, "max_position_embeddings", None)
        if self._max_seq_len is None:
            self._max_seq_len = getattr(model.config, "n_positions", None)
        self._pad_token_id = tokenizer.pad_token_id
        if self._pad_token_id is None:
            self._pad_token_id = tokenizer.eos_token_id
        if self._pad_token_id is None:
            self._pad_token_id = 0

    @property
    def max_seq_len(self) -> Optional[int]:
        if self._max_seq_len is None:
            return None
        return int(self._max_seq_len)

    @torch.no_grad()
    def topk_next_tokens(self, prefix: List[int], k: int):
        if not prefix:
            raise ValueError("Prefix must contain at least one token.")

        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        if self.max_seq_len is not None and input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len :]

        logits = self.model(input_ids).logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        k = max(1, min(int(k), int(log_probs.shape[0])))
        top_values, top_indices = torch.topk(log_probs, k=k, dim=-1)
        return (
            top_indices.detach().cpu().tolist(),
            top_values.detach().cpu().tolist(),
        )

    @torch.no_grad()
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
    ):
        if not prefixes:
            return [], []

        n = max(0, int(n))
        max_new_tokens = max(0, int(max_new_tokens))
        if n == 0:
            return [[[] for _ in range(0)] for _ in prefixes], [[0.0 for _ in range(0)] for _ in prefixes]
        if max_new_tokens == 0:
            return [[[] for _ in range(n)] for _ in prefixes], [[0.0 for _ in range(n)] for _ in prefixes]

        eos_id = eos_token_id if eos_token_id is not None else self.tokenizer.eos_token_id
        generation_top_k = 0 if top_k is None else int(top_k)

        continuation_rows: List[List[List[int]]] = [[[] for _ in range(n)] for _ in prefixes]
        logp_rows: List[List[float]] = [[0.0 for _ in range(n)] for _ in prefixes]

        # Group by prefix length so sequence slicing after generation is unambiguous.
        groups: Dict[int, List[int]] = {}
        for i, prefix in enumerate(prefixes):
            groups.setdefault(len(prefix), []).append(i)

        for prefix_len, group_indices in groups.items():
            flat_prefixes = []
            flat_meta = []
            for idx in group_indices:
                for sample_idx in range(n):
                    flat_prefixes.append(prefixes[idx])
                    flat_meta.append((idx, sample_idx))

            batch_input_ids = torch.tensor(flat_prefixes, dtype=torch.long, device=self.device)
            output = self.model.generate(
                batch_input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=float(temperature),
                top_p=float(top_p),
                top_k=generation_top_k,
                eos_token_id=eos_id,
                pad_token_id=self._pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )

            generated = output.sequences[:, prefix_len:]  # (batch, generated_steps)
            steps = len(output.scores)
            gathered_logps: Optional[torch.Tensor]
            if steps > 0:
                per_step_logps = []
                for step_idx in range(steps):
                    step_scores = output.scores[step_idx].float()
                    step_token_ids = generated[:, step_idx]
                    step_log_probs = F.log_softmax(step_scores, dim=-1)
                    step_logp = step_log_probs.gather(1, step_token_ids.unsqueeze(1)).squeeze(1)
                    per_step_logps.append(step_logp)
                gathered_logps = torch.stack(per_step_logps, dim=1)
                gathered_logps_list = gathered_logps.detach().cpu().tolist()
            else:
                gathered_logps_list = [[] for _ in range(generated.shape[0])]

            generated_list = generated.detach().cpu().tolist()

            for row_idx, (orig_idx, sample_idx) in enumerate(flat_meta):
                row_tokens = generated_list[row_idx]
                row_logps = gathered_logps_list[row_idx]

                out_tokens: List[int] = []
                out_logp_sum = 0.0
                for tok, tok_logp in zip(row_tokens, row_logps):
                    tok = int(tok)
                    out_tokens.append(tok)
                    out_logp_sum += float(tok_logp)
                    if eos_id is not None and tok == int(eos_id):
                        break

                continuation_rows[orig_idx][sample_idx] = out_tokens
                logp_rows[orig_idx][sample_idx] = out_logp_sum

        return continuation_rows, logp_rows


class GenericSampler:
    """Shared sampling engine that cleanly separates sampling from task-specific evaluation."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.autoreg_sampler = AutoregressiveSampler(model, tokenizer, device)
        self.approx_power_scorer = _ApproxPowerHFScorer(model, tokenizer, device)
        self._method_registry = {
            "power": self._sample_power,
            "power_approx": self._sample_power_approx,
        }

    @property
    def available_methods(self) -> List[str]:
        return sorted(self._method_registry.keys())

    def sample_standard(self, input_ids, max_new_tokens: int = 3072) -> SamplingOutput:
        prompt_len = input_ids.shape[1]
        start = perf_counter()
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
        )
        latency_seconds = perf_counter() - start
        generated_ids = output.sequences[0][prompt_len:].detach().cpu().tolist()
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return SamplingOutput(
            method="standard",
            completion=completion,
            token_ids=generated_ids,
            latency_seconds=latency_seconds,
        )

    def sample_temperature(self, input_ids, temperature: float, max_new_tokens: int = 3072) -> SamplingOutput:
        prompt_len = input_ids.shape[1]
        start = perf_counter()
        output = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True,
            do_sample=True,
            temperature=temperature,
        )
        latency_seconds = perf_counter() - start
        generated_ids = output.sequences[0][prompt_len:].detach().cpu().tolist()
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return SamplingOutput(
            method="temperature",
            completion=completion,
            token_ids=generated_ids,
            latency_seconds=latency_seconds,
            metadata={"temperature": temperature},
        )

    def sample_method(
        self,
        input_ids,
        method: str,
        temperature: float,
        mcmc_steps: int,
        max_new_tokens: int = 3072,
        method_config: Optional[Dict[str, Any]] = None,
    ) -> SamplingOutput:
        if method not in self._method_registry:
            supported = ", ".join(self.available_methods)
            raise ValueError(f"Unsupported sampling method '{method}'. Available methods: {supported}")
        return self._method_registry[method](
            input_ids=input_ids,
            temperature=temperature,
            mcmc_steps=mcmc_steps,
            max_new_tokens=max_new_tokens,
            method_config=method_config or {},
        )

    def _sample_power(
        self,
        input_ids,
        temperature: float,
        mcmc_steps: int,
        max_new_tokens: int,
        method_config: Optional[Dict[str, Any]] = None,
    ) -> SamplingOutput:
        prompt_token_ids = input_ids[0].detach().cpu().tolist()
        start = perf_counter()
        sampled_token_ids, _, _, acceptance_ratio = mcmc_power_samp(
            self.autoreg_sampler,
            prompt_token_ids,
            temperature,
            mcmc_steps,
            max_new_tokens=max_new_tokens,
        )
        latency_seconds = perf_counter() - start

        generated_token_ids = sampled_token_ids
        if sampled_token_ids[: len(prompt_token_ids)] == prompt_token_ids:
            generated_token_ids = sampled_token_ids[len(prompt_token_ids) :]

        completion = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        full_completion = self.tokenizer.decode(sampled_token_ids, skip_special_tokens=True)

        return SamplingOutput(
            method="power",
            completion=completion,
            token_ids=generated_token_ids,
            latency_seconds=latency_seconds,
            metadata={
                "acceptance_ratio": acceptance_ratio,
                "temperature": temperature,
                "mcmc_steps": mcmc_steps,
            },
            full_completion=full_completion,
            full_token_ids=sampled_token_ids,
        )

    def _sample_power_approx(
        self,
        input_ids,
        temperature: float,
        mcmc_steps: int,
        max_new_tokens: int,
        method_config: Optional[Dict[str, Any]] = None,
    ) -> SamplingOutput:
        del mcmc_steps  # Unused for the approximate method.
        method_config = method_config or {}

        def _as_int(name: str, default: int) -> int:
            value = method_config.get(name, default)
            return int(value)

        def _as_opt_int(name: str, default: Optional[int]) -> Optional[int]:
            value = method_config.get(name, default)
            if value is None:
                return None
            return int(value)

        def _as_bool(name: str, default: bool) -> bool:
            value = method_config.get(name, default)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

        cfg = PowerSamplerApproxConfig(
            temp=float(temperature),
            top_k=_as_int("top_k", 8),
            candidate_pool_size=_as_int("candidate_pool_size", 32),
            rollouts_per_candidate=_as_int("rollouts_per_candidate", 8),
            lookahead_tokens=_as_opt_int("lookahead_tokens", None),
            block_size=_as_int("block_size", 1),
            use_jackknife=_as_bool("use_jackknife", True),
        )

        prompt_token_ids = input_ids[0].detach().cpu().tolist()
        eos_id = method_config.get("eos_token_id", self.tokenizer.eos_token_id)
        seed_value = method_config.get("seed")
        rng = np.random.default_rng(seed=seed_value)

        start = perf_counter()
        sampled_token_ids, diag = approx_power_sample(
            self.approx_power_scorer,
            prompt_token_ids,
            cfg=cfg,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            rng=rng,
        )
        latency_seconds = perf_counter() - start

        generated_token_ids = sampled_token_ids
        if sampled_token_ids[: len(prompt_token_ids)] == prompt_token_ids:
            generated_token_ids = sampled_token_ids[len(prompt_token_ids) :]

        completion = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        full_completion = self.tokenizer.decode(sampled_token_ids, skip_special_tokens=True)

        metadata: Dict[str, Any] = {
            "temperature": temperature,
            "approx_config": asdict(cfg),
        }
        metadata.update(diag)

        return SamplingOutput(
            method="power_approx",
            completion=completion,
            token_ids=generated_token_ids,
            latency_seconds=latency_seconds,
            metadata=metadata,
            full_completion=full_completion,
            full_token_ids=sampled_token_ids,
        )


def resolve_model_name(model_alias: str) -> str:
    if model_alias not in MODEL_NAME_BY_ALIAS:
        supported = ", ".join(sorted(MODEL_NAME_BY_ALIAS.keys()))
        raise ValueError(f"Unknown model alias '{model_alias}'. Supported aliases: {supported}")
    return MODEL_NAME_BY_ALIAS[model_alias]


def load_model_and_tokenizer(
    model_alias: str,
    device: str,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
):
    model_name = resolve_model_name(model_alias)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    ).to(device)
    return model, tokenizer


class WandbSampleLogger:
    """Light wrapper so scripts can log samples/metrics without hard dependency on wandb."""

    def __init__(
        self,
        project: Optional[str],
        entity: Optional[str],
        run_name: Optional[str],
        config: Dict[str, Any],
        sample_log_limit: int = 20,
    ):
        self.sample_log_limit = max(sample_log_limit, 0)
        self.sample_rows: List[Dict[str, Any]] = []
        self._wandb = None
        self._run = None

        project = project or None
        entity = entity or None
        run_name = run_name or None

        if not project:
            return

        try:
            import wandb  # type: ignore

            self._wandb = wandb
            self._run = wandb.init(project=project, entity=entity, name=run_name, config=config)
        except Exception as exc:
            print(f"W&B disabled: {exc}")

    @property
    def enabled(self) -> bool:
        return self._run is not None and self._wandb is not None

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if not self.enabled:
            return
        clean_metrics = {}
        for key, value in metrics.items():
            if self._is_finite_number(value):
                clean_metrics[key] = float(value)
        if clean_metrics:
            self._wandb.log(clean_metrics, step=step)

    def log_file(self, file_path: str):
        if not self.enabled or not os.path.exists(file_path):
            return
        try:
            self._wandb.save(file_path, policy="now")
        except Exception as exc:
            print(f"W&B file upload skipped for {file_path}: {exc}")

    def log_sample(self, sample: Dict[str, Any]):
        if not self.enabled or len(self.sample_rows) >= self.sample_log_limit:
            return
        row = {}
        for key, value in sample.items():
            if value is None:
                continue
            if isinstance(value, (dict, list, tuple)):
                row[key] = json.dumps(value)
            else:
                row[key] = value
        self.sample_rows.append(row)

    def finish(self, summary: Optional[Dict[str, Any]] = None):
        if not self.enabled:
            return

        if self.sample_rows:
            column_names = sorted({key for row in self.sample_rows for key in row.keys()})
            sample_table = self._wandb.Table(columns=column_names)
            for row in self.sample_rows:
                sample_table.add_data(*[row.get(name) for name in column_names])
            self._wandb.log({"sample_outputs": sample_table})

        if summary:
            for key, value in summary.items():
                if self._is_finite_number(value):
                    self._run.summary[key] = float(value)

        self._wandb.finish()

    @staticmethod
    def _is_finite_number(value: Any) -> bool:
        return isinstance(value, Number) and math.isfinite(float(value))
