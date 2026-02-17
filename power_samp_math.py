import argparse
import json
import os
import random
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from grader_utils.math_grader import grade_answer
from grader_utils.parse_utils import parse_answer
from power_samp_utils import format_prompt
from sampling_runtime import GenericSampler, WandbSampleLogger, load_model_and_tokenizer


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def truncate_text(text: str, max_chars: int = 1200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def safe_grade(prediction: str, answer: str) -> int:
    try:
        return int(grade_answer(prediction, answer))
    except Exception:
        return 0


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
    parser.add_argument("--dataset", type=str, default="MATH")
    parser.add_argument("--cot", type=parse_bool, default=True)
    parser.add_argument("--mcmc_steps", type=int, default=10)
    parser.add_argument("--sampling_method", type=str, default="power", choices=["power"])
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "reasoning-with-sampling"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--wandb_log_samples", type=int, default=20)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.dataset != "MATH":
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    save_str = os.path.join(args.save_str, args.model)
    os.makedirs(save_str, exist_ok=True)

    with open("data/MATH500.json", "r") as f:
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

    start = 100 * args.batch_idx
    end = min(100 * (args.batch_idx + 1), len(dataset))

    results = []
    total_power_seconds = 0.0
    total_power_samples = 0

    for step_idx, data in enumerate(tqdm(dataset[start:end], desc="Benchmark on MATH")):
        question = data["prompt"]
        answer = data["answer"]

        input_text = format_prompt(question, args.model, tokenizer, args.cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(args.device)

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

        naive_answer = parse_answer(naive_sample.completion)
        std_answer = parse_answer(std_sample.completion)
        method_answer = parse_answer(method_sample.completion)

        naive_reward = safe_grade(naive_answer, answer)
        std_reward = safe_grade(std_answer, answer)
        method_reward = safe_grade(method_answer, answer)

        power_seconds = method_sample.latency_seconds
        acceptance_ratio = method_sample.metadata.get("acceptance_ratio")
        total_power_seconds += power_seconds
        total_power_samples += 1
        avg_power_seconds = total_power_seconds / max(total_power_samples, 1)

        results.append(
            {
                "question": question,
                "correct_answer": answer,
                "naive_completion": naive_sample.completion,
                "naive_answer": naive_answer,
                "std_completion": std_sample.completion,
                "std_answer": std_answer,
                "mcmc_completion": method_sample.full_completion or method_sample.completion,
                "mcmc_generated_completion": method_sample.completion,
                "mcmc_answer": method_answer,
                "sampling_method": args.sampling_method,
                "power_sampling_seconds": power_seconds,
                "power_sampling_avg_seconds_so_far": avg_power_seconds,
                "power_acceptance_ratio": acceptance_ratio,
                "naive_reward": naive_reward,
                "std_reward": std_reward,
                "mcmc_reward": method_reward,
            }
        )

        wandb_logger.log_metrics(
            {
                "latency/power_sampling_seconds": power_seconds,
                "latency/power_sampling_avg_seconds": avg_power_seconds,
                "reward/naive": naive_reward,
                "reward/std": std_reward,
                "reward/mcmc": method_reward,
                "sampling/acceptance_ratio": acceptance_ratio,
            },
            step=step_idx,
        )
        wandb_logger.log_sample(
            {
                "dataset_index": start + step_idx,
                "question": truncate_text(question),
                "correct_answer": answer,
                "naive_completion": truncate_text(naive_sample.completion),
                "std_completion": truncate_text(std_sample.completion),
                "mcmc_completion": truncate_text(method_sample.completion),
                "naive_reward": naive_reward,
                "std_reward": std_reward,
                "mcmc_reward": method_reward,
                "power_sampling_seconds": power_seconds,
                "power_acceptance_ratio": acceptance_ratio,
            }
        )

    output_path = os.path.join(
        save_str,
        f"{args.model}_math_base_power_samp_results_{args.mcmc_steps}_{args.temperature}_{args.batch_idx}_{args.seed}.csv",
    )
    pd.DataFrame(results).to_csv(output_path, index=False)
    wandb_logger.log_file(output_path)

    avg_power_seconds = total_power_seconds / max(total_power_samples, 1)
    print(f"Saved results to: {output_path}")
    print(f"Average power sampling time per sample: {avg_power_seconds:.4f} seconds")

    wandb_logger.finish(
        summary={
            "summary/num_samples": total_power_samples,
            "summary/avg_power_sampling_seconds": avg_power_seconds,
        }
    )
