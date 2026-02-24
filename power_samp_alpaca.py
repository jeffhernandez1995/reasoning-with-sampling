import argparse
import json
import os
import random
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from sampling_runtime import GenericSampler, WandbSampleLogger, load_model_and_tokenizer


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def truncate_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", type=str, default="results/")
    parser.add_argument(
        "--model",
        default="qwen",
        type=str,
        choices=["qwen", "qwen_math", "phi", "tulu", "qwen_grpo", "qwen_math_grpo", "phi_grpo"],
    )
    parser.add_argument("--temperature", "--temp", dest="temperature", type=float, default=0.25)
    parser.add_argument("--dataset", type=str, default="ALPACA")
    parser.add_argument("--cot", type=parse_bool, default=True)
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--sampling_method", type=str, default="power", choices=["power"])
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_log_samples", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.dataset != "ALPACA":
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    save_str = os.path.join(args.save_str, args.model)
    os.makedirs(save_str, exist_ok=True)

    with open("data/ALPACA.json", "r") as f:
        dataset = json.load(f)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, trust_remote_code=True)
    sampler = GenericSampler(model, tokenizer, args.device)

    wandb_logger = WandbSampleLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=args.wandb_run_name,
        sample_log_limit=args.wandb_log_samples,
        config={
            "dataset": args.dataset,
            "model": args.model,
            "temperature": args.temperature,
            "mcmc_steps": args.mcmc_steps,
            "sampling_method": args.sampling_method,
            "batch_idx": args.batch_idx,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
        },
    )

    start = 58 * args.batch_idx
    end = min(58 * (args.batch_idx + 1), len(dataset))

    results = []
    total_power_seconds = 0.0
    total_power_samples = 0

    for step_idx, data in enumerate(tqdm(dataset[start:end], desc="Benchmark on ALPACA")):
        source = data["dataset"]
        instruction = data["instruction"]

        input_ids = tokenizer.encode(instruction, return_tensors="pt").to(args.device)

        naive_sample = sampler.sample_temperature(
            input_ids=input_ids,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        std_sample = sampler.sample_standard(input_ids=input_ids, max_new_tokens=args.max_new_tokens)
        method_sample = sampler.sample_method(
            input_ids=input_ids,
            method=args.sampling_method,
            temperature=args.temperature,
            mcmc_steps=args.mcmc_steps,
            max_new_tokens=args.max_new_tokens,
        )

        power_seconds = method_sample.latency_seconds
        temp_output_tokens = len(naive_sample.token_ids)
        std_output_tokens = len(std_sample.token_ids)
        acceptance_ratio = method_sample.metadata.get("acceptance_ratio")
        sampling_tokens = method_sample.metadata.get("sampling_tokens")
        output_tokens = method_sample.metadata.get("output_tokens")
        internal_sampling_tokens = method_sample.metadata.get("internal_sampling_tokens")
        total_power_seconds += power_seconds
        total_power_samples += 1
        avg_power_seconds = total_power_seconds / max(total_power_samples, 1)

        results.append(
            {
                "dataset": source,
                "instruction": instruction,
                "naive_completion": naive_sample.completion,
                "std_completion": std_sample.completion,
                "mcmc_completion": method_sample.full_completion or method_sample.completion,
                "mcmc_generated_completion": method_sample.completion,
                "sampling_method": args.sampling_method,
                "power_sampling_seconds": power_seconds,
                "power_sampling_avg_seconds_so_far": avg_power_seconds,
                "temp_output_tokens": temp_output_tokens,
                "temp_sampling_tokens": temp_output_tokens,
                "std_output_tokens": std_output_tokens,
                "std_sampling_tokens": std_output_tokens,
                "power_acceptance_ratio": acceptance_ratio,
                "power_sampling_tokens": sampling_tokens,
                "power_output_tokens": output_tokens,
                "power_internal_sampling_tokens": internal_sampling_tokens,
            }
        )

        wandb_logger.log_metrics(
            {
                "latency/power_sampling_seconds": power_seconds,
                "latency/power_sampling_avg_seconds": avg_power_seconds,
                "sampling/temp_output_tokens": temp_output_tokens,
                "sampling/temp_sampling_tokens": temp_output_tokens,
                "sampling/std_output_tokens": std_output_tokens,
                "sampling/std_sampling_tokens": std_output_tokens,
                "sampling/acceptance_ratio": acceptance_ratio,
                "sampling/power_sampling_tokens": sampling_tokens,
                "sampling/power_output_tokens": output_tokens,
                "sampling/power_internal_sampling_tokens": internal_sampling_tokens,
                "tokens/naive": len(naive_sample.token_ids),
                "tokens/std": len(std_sample.token_ids),
                "tokens/mcmc": len(method_sample.token_ids),
            },
            step=step_idx,
        )
        wandb_logger.log_sample(
            {
                "dataset_index": start + step_idx,
                "source": source,
                "instruction": truncate_text(instruction),
                "naive_completion": truncate_text(naive_sample.completion),
                "std_completion": truncate_text(std_sample.completion),
                "mcmc_completion": truncate_text(method_sample.completion),
                "power_sampling_seconds": power_seconds,
                "temp_output_tokens": temp_output_tokens,
                "temp_sampling_tokens": temp_output_tokens,
                "std_output_tokens": std_output_tokens,
                "std_sampling_tokens": std_output_tokens,
                "power_acceptance_ratio": acceptance_ratio,
                "power_sampling_tokens": sampling_tokens,
                "power_output_tokens": output_tokens,
            }
        )

    output_path = os.path.join(
        save_str,
        f"{args.model}_alpaca_base_power_samp_results_{args.mcmc_steps}_{args.temperature}_{args.batch_idx}_{args.seed}.csv",
    )
    pd.DataFrame(results).to_csv(output_path, index=False)

    avg_power_seconds = total_power_seconds / max(total_power_samples, 1)
    print(f"Saved results to: {output_path}")
    print(f"Average power sampling time per sample: {avg_power_seconds:.4f} seconds")

    wandb_logger.finish(
        summary={
            "summary/num_samples": total_power_samples,
            "summary/avg_power_sampling_seconds": avg_power_seconds,
        }
    )
