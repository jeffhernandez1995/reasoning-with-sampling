import argparse
import json
import os
import random
from typing import Any, Dict

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


def _float_token(value: float) -> str:
    return str(value).replace(".", "p")


def default_wandb_run_name(args) -> str:
    return (
        f"{args.dataset.lower()}_{args.model}_{args.sampling_method}"
        f"_t{_float_token(args.temperature)}_k{args.top_k}"
        f"_m{args.rollouts_per_candidate}_b{args.block_size}"
        f"_shard{args.batch_idx:02d}_seed{args.seed:02d}"
    )


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
    parser.add_argument("--sampling_method", type=str, default="power_approx", choices=["power_approx"])
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--candidate_pool_size", type=int, default=32)
    parser.add_argument("--rollouts_per_candidate", type=int, default=8)
    parser.add_argument("--lookahead_tokens", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=1)
    parser.add_argument("--use_jackknife", type=parse_bool, default=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "reasoning-with-sampling"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--wandb_log_samples", type=int, default=20)
    args = parser.parse_args()
    wandb_run_name = args.wandb_run_name or default_wandb_run_name(args)

    random.seed(args.seed)

    if args.dataset != "MATH":
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    save_str = os.path.join(args.save_str, args.sampling_method, args.model)
    os.makedirs(save_str, exist_ok=True)

    with open("data/MATH500.json", "r") as f:
        dataset = json.load(f)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, trust_remote_code=True)
    sampler = GenericSampler(model, tokenizer, args.device)

    approx_method_config: Dict[str, Any] = {
        "top_k": args.top_k,
        "candidate_pool_size": args.candidate_pool_size,
        "rollouts_per_candidate": args.rollouts_per_candidate,
        "lookahead_tokens": args.lookahead_tokens,
        "block_size": args.block_size,
        "use_jackknife": args.use_jackknife,
        "seed": args.seed,
        "eos_token_id": tokenizer.eos_token_id,
    }

    wandb_logger = WandbSampleLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=wandb_run_name,
        sample_log_limit=args.wandb_log_samples,
        config={
            "dataset": args.dataset,
            "model": args.model,
            "temperature": args.temperature,
            "sampling_method": args.sampling_method,
            "batch_idx": args.batch_idx,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "top_k": args.top_k,
            "candidate_pool_size": args.candidate_pool_size,
            "rollouts_per_candidate": args.rollouts_per_candidate,
            "lookahead_tokens": args.lookahead_tokens,
            "block_size": args.block_size,
            "use_jackknife": args.use_jackknife,
            "wandb_run_name": wandb_run_name,
        },
    )

    start = 100 * args.batch_idx
    end = min(100 * (args.batch_idx + 1), len(dataset))

    results = []
    total_base_seconds = 0.0
    total_temp_seconds = 0.0
    total_approx_seconds = 0.0
    total_approx_samples = 0

    for step_idx, data in enumerate(tqdm(dataset[start:end], desc="Benchmark on MATH (power_approx)")):
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
            mcmc_steps=0,
            max_new_tokens=args.max_new_tokens,
            method_config=approx_method_config,
        )

        naive_answer = parse_answer(naive_sample.completion)
        std_answer = parse_answer(std_sample.completion)
        method_answer = parse_answer(method_sample.completion)

        naive_reward = safe_grade(naive_answer, answer)
        std_reward = safe_grade(std_answer, answer)
        method_reward = safe_grade(method_answer, answer)

        base_seconds = std_sample.latency_seconds
        temp_seconds = naive_sample.latency_seconds
        approx_seconds = method_sample.latency_seconds
        total_base_seconds += base_seconds
        total_temp_seconds += temp_seconds
        total_approx_seconds += approx_seconds
        total_approx_samples += 1
        avg_base_seconds = total_base_seconds / max(total_approx_samples, 1)
        avg_temp_seconds = total_temp_seconds / max(total_approx_samples, 1)
        avg_approx_seconds = total_approx_seconds / max(total_approx_samples, 1)

        approx_rollouts = method_sample.metadata.get("rollouts")
        approx_rollout_tokens = method_sample.metadata.get("rollout_tokens")
        approx_steps = method_sample.metadata.get("steps")

        results.append(
            {
                "question": question,
                "correct_answer": answer,
                "naive_completion": naive_sample.completion,
                "naive_answer": naive_answer,
                "std_completion": std_sample.completion,
                "std_answer": std_answer,
                # Keep mcmc_* column names so existing eval scripts keep working.
                "mcmc_completion": method_sample.full_completion or method_sample.completion,
                "mcmc_generated_completion": method_sample.completion,
                "mcmc_answer": method_answer,
                "sampling_method": args.sampling_method,
                "base_sampling_seconds": base_seconds,
                "base_sampling_avg_seconds_so_far": avg_base_seconds,
                "temp_sampling_seconds": temp_seconds,
                "temp_sampling_avg_seconds_so_far": avg_temp_seconds,
                "power_sampling_seconds": approx_seconds,
                "power_sampling_avg_seconds_so_far": avg_approx_seconds,
                "power_acceptance_ratio": None,
                "approx_steps": approx_steps,
                "approx_rollouts": approx_rollouts,
                "approx_rollout_tokens": approx_rollout_tokens,
                "approx_config": json.dumps(method_sample.metadata.get("approx_config", {})),
                "naive_reward": naive_reward,
                "std_reward": std_reward,
                "mcmc_reward": method_reward,
            }
        )

        wandb_logger.log_metrics(
            {
                "latency/base_sampling_seconds": base_seconds,
                "latency/base_sampling_avg_seconds": avg_base_seconds,
                "latency/temp_sampling_seconds": temp_seconds,
                "latency/temp_sampling_avg_seconds": avg_temp_seconds,
                "latency/power_sampling_seconds": approx_seconds,
                "latency/power_sampling_avg_seconds": avg_approx_seconds,
                "sampling/approx_steps": approx_steps,
                "sampling/approx_rollouts": approx_rollouts,
                "sampling/approx_rollout_tokens": approx_rollout_tokens,
                "reward/naive": naive_reward,
                "reward/std": std_reward,
                "reward/mcmc": method_reward,
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
                "base_sampling_seconds": base_seconds,
                "temp_sampling_seconds": temp_seconds,
                "power_sampling_seconds": approx_seconds,
                "approx_steps": approx_steps,
                "approx_rollouts": approx_rollouts,
            }
        )

    output_path = os.path.join(
        save_str,
        (
            f"{args.model}_math_base_power_approx_results_"
            f"k{args.top_k}_m{args.rollouts_per_candidate}_"
            f"b{args.block_size}_{args.temperature}_{args.batch_idx}_{args.seed}.csv"
        ),
    )
    pd.DataFrame(results).to_csv(output_path, index=False)
    wandb_logger.log_file(output_path)

    avg_base_seconds = total_base_seconds / max(total_approx_samples, 1)
    avg_temp_seconds = total_temp_seconds / max(total_approx_samples, 1)
    avg_approx_seconds = total_approx_seconds / max(total_approx_samples, 1)
    print(f"Saved results to: {output_path}")
    print(f"Average base sampling time per sample: {avg_base_seconds:.4f} seconds")
    print(f"Average temp sampling time per sample: {avg_temp_seconds:.4f} seconds")
    print(f"Average power_approx sampling time per sample: {avg_approx_seconds:.4f} seconds")

    wandb_logger.finish(
        summary={
            "summary/num_samples": total_approx_samples,
            "summary/avg_base_sampling_seconds": avg_base_seconds,
            "summary/avg_temp_sampling_seconds": avg_temp_seconds,
            "summary/avg_power_sampling_seconds": avg_approx_seconds,
        }
    )
