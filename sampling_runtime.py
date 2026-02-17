import json
import math
import os
from dataclasses import dataclass, field
from numbers import Number
from time import perf_counter
from typing import Any, Dict, List, Optional

import torch
import transformers

from power_samp_utils import AutoregressiveSampler, mcmc_power_samp


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


class GenericSampler:
    """Shared sampling engine that cleanly separates sampling from task-specific evaluation."""

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.autoreg_sampler = AutoregressiveSampler(model, tokenizer, device)
        self._method_registry = {
            "power": self._sample_power,
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
    ) -> SamplingOutput:
        if method not in self._method_registry:
            supported = ", ".join(self.available_methods)
            raise ValueError(f"Unsupported sampling method '{method}'. Available methods: {supported}")
        return self._method_registry[method](
            input_ids=input_ids,
            temperature=temperature,
            mcmc_steps=mcmc_steps,
            max_new_tokens=max_new_tokens,
        )

    def _sample_power(self, input_ids, temperature: float, mcmc_steps: int, max_new_tokens: int) -> SamplingOutput:
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
