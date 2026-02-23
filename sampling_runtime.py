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
from algos.power_sampling_smc import PowerSamplerSMCConfig, smc_power_sample


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

    @torch.inference_mode()
    def topk_next_tokens(self, prefix: List[int], k: int):
        if not prefix:
            raise ValueError("Prefix must contain at least one token.")

        input_ids = torch.tensor([prefix], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        if self.max_seq_len is not None and input_ids.shape[1] > self.max_seq_len:
            input_ids = input_ids[:, -self.max_seq_len :]
            attention_mask = attention_mask[:, -self.max_seq_len :]

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits[0, -1, :]
        log_probs = F.log_softmax(logits, dim=-1)
        k = max(1, min(int(k), int(log_probs.shape[0])))
        top_values, top_indices = torch.topk(log_probs, k=k, dim=-1)
        return (
            top_indices.detach().cpu().tolist(),
            top_values.detach().cpu().tolist(),
        )

    @torch.inference_mode()
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
        eos_id_int = None if eos_id is None else int(eos_id)
        top_k_val = None if top_k is None else int(top_k)
        top_p_val = float(top_p)
        temperature_val = float(temperature)
        if temperature_val <= 0:
            raise ValueError("temperature must be > 0.")

        continuation_rows: List[List[List[int]]] = [[[] for _ in range(n)] for _ in prefixes]
        logp_rows: List[List[float]] = [[0.0 for _ in range(n)] for _ in prefixes]

        clipped_prefixes: List[List[int]] = prefixes
        if self.max_seq_len is not None:
            max_len = int(self.max_seq_len)
            clipped_prefixes = [prefix[-max_len:] if len(prefix) > max_len else prefix for prefix in prefixes]

        def _apply_top_k(logits: torch.Tensor) -> torch.Tensor:
            if top_k_val is None or top_k_val <= 0:
                return logits
            k = min(int(top_k_val), int(logits.shape[-1]))
            if k >= int(logits.shape[-1]):
                return logits
            top_values, _ = torch.topk(logits, k=k, dim=-1)
            kth = top_values[:, -1].unsqueeze(-1)
            return torch.where(logits < kth, torch.full_like(logits, -float("inf")), logits)

        def _apply_top_p(logits: torch.Tensor) -> torch.Tensor:
            if top_p_val >= 1.0:
                return logits
            if top_p_val <= 0.0:
                argmax_ids = torch.argmax(logits, dim=-1, keepdim=True)
                return torch.full_like(logits, -float("inf")).scatter(1, argmax_ids, logits.gather(1, argmax_ids))
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            sorted_probs = F.softmax(sorted_logits, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            sorted_mask = cumulative > top_p_val
            sorted_mask[:, 0] = False
            sorted_logits = sorted_logits.masked_fill(sorted_mask, -float("inf"))
            return logits.scatter(1, sorted_indices, sorted_logits)

        # Group by prefix length so prefill batching is unambiguous.
        groups: Dict[int, List[int]] = {}
        for i, prefix in enumerate(clipped_prefixes):
            groups.setdefault(len(prefix), []).append(i)

        pad_token = torch.tensor(self._pad_token_id, device=self.device, dtype=torch.long)

        for _, group_indices in groups.items():
            flat_prefixes = []
            flat_meta = []
            for idx in group_indices:
                for sample_idx in range(n):
                    flat_prefixes.append(clipped_prefixes[idx])
                    flat_meta.append((idx, sample_idx))

            batch_input_ids = torch.tensor(flat_prefixes, dtype=torch.long, device=self.device)
            batch_size, prefix_len = batch_input_ids.shape
            full_attention_mask = torch.ones(
                (batch_size, prefix_len + max_new_tokens),
                dtype=batch_input_ids.dtype,
                device=self.device,
            )
            outputs = self.model(
                input_ids=batch_input_ids,
                attention_mask=full_attention_mask[:, :prefix_len],
                use_cache=True,
            )
            next_logits = outputs.logits[:, -1, :].float()
            past_key_values = outputs.past_key_values

            generated = torch.full(
                (batch_size, max_new_tokens),
                self._pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            logp_sums = torch.zeros((batch_size,), dtype=torch.float32, device=self.device)
            finished = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
            generated_steps = 0
            current_len = int(prefix_len)

            for step_idx in range(max_new_tokens):
                step_logits = next_logits
                if temperature_val != 1.0:
                    step_logits = step_logits / temperature_val
                step_logits = _apply_top_k(step_logits)
                step_logits = _apply_top_p(step_logits)

                step_log_probs = F.log_softmax(step_logits, dim=-1)
                sampled_ids = torch.multinomial(torch.exp(step_log_probs), num_samples=1).squeeze(1)
                sampled_logp = step_log_probs.gather(1, sampled_ids.unsqueeze(1)).squeeze(1)

                sampled_ids = torch.where(finished, pad_token, sampled_ids)
                sampled_logp = torch.where(finished, torch.zeros_like(sampled_logp), sampled_logp)

                generated[:, step_idx] = sampled_ids
                logp_sums += sampled_logp
                generated_steps = step_idx + 1

                if eos_id_int is not None:
                    finished = finished | sampled_ids.eq(eos_id_int)
                    if bool(torch.all(finished).item()):
                        break

                current_len += 1
                outputs = self.model(
                    input_ids=sampled_ids.unsqueeze(1),
                    attention_mask=full_attention_mask[:, :current_len],
                    past_key_values=past_key_values,
                    use_cache=True,
                )
                next_logits = outputs.logits[:, -1, :].float()
                past_key_values = outputs.past_key_values

            generated_rows = generated[:, :generated_steps].detach().cpu().tolist()
            logp_values = logp_sums.detach().cpu().tolist()
            for row_idx, (orig_idx, sample_idx) in enumerate(flat_meta):
                row_tokens: List[int] = []
                for tok in generated_rows[row_idx]:
                    tok_int = int(tok)
                    row_tokens.append(tok_int)
                    if eos_id_int is not None and tok_int == eos_id_int:
                        break
                continuation_rows[orig_idx][sample_idx] = row_tokens
                logp_rows[orig_idx][sample_idx] = float(logp_values[row_idx])

        return continuation_rows, logp_rows


class GenericSampler:
    """Shared sampling engine that cleanly separates sampling from task-specific evaluation."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.autoreg_sampler = AutoregressiveSampler(model, tokenizer, device)
        self.approx_power_scorer = _ApproxPowerHFScorer(model, tokenizer, device)
        self._pad_token_id = tokenizer.pad_token_id
        if self._pad_token_id is None:
            self._pad_token_id = tokenizer.eos_token_id
        if self._pad_token_id is None:
            self._pad_token_id = 0
        self._method_registry = {
            "power": self._sample_power,
            "power_approx": self._sample_power_approx,
            "power_smc": self._sample_power_smc,
            "power_smc_apf": self._sample_power_smc,
        }

    @property
    def available_methods(self) -> List[str]:
        return sorted(self._method_registry.keys())

    @torch.inference_mode()
    def sample_standard(self, input_ids, max_new_tokens: int = 3072) -> SamplingOutput:
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        start = perf_counter()
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=False,
            output_scores=False,
            do_sample=True,
            pad_token_id=self._pad_token_id,
        )
        latency_seconds = perf_counter() - start
        generated_ids = output[0][prompt_len:].detach().cpu().tolist()
        completion = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return SamplingOutput(
            method="standard",
            completion=completion,
            token_ids=generated_ids,
            latency_seconds=latency_seconds,
        )

    @torch.inference_mode()
    def sample_temperature(self, input_ids, temperature: float, max_new_tokens: int = 3072) -> SamplingOutput:
        prompt_len = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)
        start = perf_counter()
        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=False,
            output_scores=False,
            do_sample=True,
            temperature=temperature,
            pad_token_id=self._pad_token_id,
        )
        latency_seconds = perf_counter() - start
        generated_ids = output[0][prompt_len:].detach().cpu().tolist()
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
            lookahead_tokens=_as_opt_int("lookahead_tokens", 192),
            block_size=_as_int("block_size", 192),
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

    def _sample_power_smc(
        self,
        input_ids,
        temperature: float,
        mcmc_steps: int,
        max_new_tokens: int,
        method_config: Optional[Dict[str, Any]] = None,
    ) -> SamplingOutput:
        """SMC/Feynman--Kac particle filter sampler for the power distribution."""

        del mcmc_steps  # Unused for SMC.
        method_config = method_config or {}

        def _as_int(name: str, default: int) -> int:
            value = method_config.get(name, default)
            return int(value)

        def _as_opt_int(name: str, default: Optional[int]) -> Optional[int]:
            value = method_config.get(name, default)
            if value is None:
                return None
            return int(value)

        def _as_float(name: str, default: float) -> float:
            value = method_config.get(name, default)
            return float(value)

        def _as_opt_float(name: str, default: Optional[float]) -> Optional[float]:
            value = method_config.get(name, default)
            if value is None:
                return None
            return float(value)

        def _as_bool(name: str, default: bool) -> bool:
            value = method_config.get(name, default)
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}

        cfg = PowerSamplerSMCConfig(
            temp=float(temperature),
            num_particles=_as_int("num_particles", 8),
            ess_threshold=_as_float("ess_threshold", 0.5),
            resample_interval=_as_int("resample_interval", 8),
            resample_method=str(method_config.get("resample_method", "systematic")),
            proposal_temperature=_as_float("proposal_temperature", 1.0),
            proposal_top_k=_as_opt_int("proposal_top_k", None),
            proposal_top_p=_as_float("proposal_top_p", 1.0),
            max_logw_step=_as_float("max_logw_step", 50.0),
            stop_on_all_eos=_as_bool("stop_on_all_eos", True),
            use_auxiliary=_as_bool("use_auxiliary", False),
            auxiliary_resample_always=_as_bool("auxiliary_resample_always", False),
            auxiliary_temperature=_as_opt_float("auxiliary_temperature", None),
            seed=method_config.get("seed"),
        )

        prompt_token_ids = input_ids[0].detach().cpu().tolist()
        eos_id = method_config.get("eos_token_id", self.tokenizer.eos_token_id)
        pad_id = method_config.get("pad_token_id", self._pad_token_id)

        start = perf_counter()
        sampled_token_ids, diag = smc_power_sample(
            self.model,
            self.tokenizer,
            self.device,
            prompt_token_ids,
            cfg=cfg,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
        )
        latency_seconds = perf_counter() - start

        generated_token_ids = sampled_token_ids
        if sampled_token_ids[: len(prompt_token_ids)] == prompt_token_ids:
            generated_token_ids = sampled_token_ids[len(prompt_token_ids) :]

        completion = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
        full_completion = self.tokenizer.decode(sampled_token_ids, skip_special_tokens=True)

        metadata: Dict[str, Any] = {
            "temperature": float(temperature),
            "smc_config": asdict(cfg),
        }
        metadata.update(diag)

        return SamplingOutput(
            method="power_smc",
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
    use_auto_device_map = str(device).startswith("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    model_kwargs: Dict[str, Any] = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if use_auto_device_map:
        model_kwargs["device_map"] = "auto"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if not use_auto_device_map:
        model = model.to(device)
    model.eval()
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
