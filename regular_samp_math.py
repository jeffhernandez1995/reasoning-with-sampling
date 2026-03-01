import argparse
import json
import os
import random
import re
from typing import Any

import numpy as np
import pandas as pd
import torch
import transformers
from tqdm import tqdm

from grader_utils.math_grader import grade_answer
from grader_utils.parse_utils import parse_answer
from power_samp_utils import format_prompt
from sampling_runtime import GenericSampler, WandbSampleLogger


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


def _slugify_model_id(model_id: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", model_id)
    slug = slug.strip("-")
    return slug or "hf-model"


def load_hf_model_and_tokenizer(
    model_id: str,
    device: str,
    trust_remote_code: bool = True,
    local_files_only: bool = False,
):
    use_auto_device_map = str(device).startswith("cuda")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": "auto",
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if use_auto_device_map:
        model_kwargs["device_map"] = "auto"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    if not use_auto_device_map:
        model = model.to(device)
    model.eval()
    return model, tokenizer


def default_wandb_run_name(args, model_slug: str) -> str:
    return (
        f"{args.dataset.lower()}_{model_slug}_{args.sampling_method}"
        f"_t{_float_token(args.temperature)}"
        f"_shard{args.batch_idx:02d}_seed{args.seed:02d}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_str", type=str, default="results/")
    parser.add_argument(
        "--hf_model_id",
        type=str,
        default="stellalisy/rethink_rlvr_reproduce-ground_truth-qwen2.5_math_7b-lr5e-7-kl0.00-step150",
    )
    parser.add_argument("--temperature", "--temp", dest="temperature", type=float, default=0.25)
    parser.add_argument("--dataset", type=str, default="MATH")
    parser.add_argument("--cot", type=parse_bool, default=True)
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--sampling_method", type=str, default="regular", choices=["regular"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trust_remote_code", type=parse_bool, default=True)
    parser.add_argument("--local_files_only", type=parse_bool, default=False)
    parser.add_argument("--wandb_project", type=str, default=os.environ.get("WANDB_PROJECT", "reasoning-with-sampling"))
    parser.add_argument("--wandb_entity", type=str, default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb_run_name", type=str, default=os.environ.get("WANDB_RUN_NAME"))
    parser.add_argument("--wandb_log_samples", type=int, default=20)
    args = parser.parse_args()

    model_slug = _slugify_model_id(args.hf_model_id)
    wandb_run_name = args.wandb_run_name or default_wandb_run_name(args, model_slug)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if args.dataset != "MATH":
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    save_str = os.path.join(args.save_str, args.sampling_method, model_slug)
    os.makedirs(save_str, exist_ok=True)

    with open("data/MATH500.json", "r") as f:
        dataset = json.load(f)

    model, tokenizer = load_hf_model_and_tokenizer(
        args.hf_model_id,
        args.device,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    sampler = GenericSampler(model, tokenizer, args.device)

    wandb_logger = WandbSampleLogger(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=wandb_run_name,
        sample_log_limit=args.wandb_log_samples,
        config={
            "dataset": args.dataset,
            "hf_model_id": args.hf_model_id,
            "model_slug": model_slug,
            "temperature": args.temperature,
            "sampling_method": args.sampling_method,
            "batch_idx": args.batch_idx,
            "seed": args.seed,
            "max_new_tokens": args.max_new_tokens,
            "trust_remote_code": args.trust_remote_code,
            "local_files_only": args.local_files_only,
            "wandb_run_name": wandb_run_name,
        },
    )

    start = 100 * args.batch_idx
    end = min(100 * (args.batch_idx + 1), len(dataset))

    results = []
    total_base_seconds = 0.0
    total_temp_seconds = 0.0
    total_regular_seconds = 0.0
    total_samples = 0

    for step_idx, data in enumerate(tqdm(dataset[start:end], desc="Benchmark on MATH")):
        question = data["prompt"]
        answer = data["answer"]

        # Keep prompting identical to qwen_math so runs are directly comparable.
        input_text = format_prompt(question, "qwen_math", tokenizer, args.cot)
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(args.device)

        naive_sample = sampler.sample_temperature(
            input_ids=input_ids,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
        )
        std_sample = sampler.sample_standard(input_ids=input_ids, max_new_tokens=args.max_new_tokens)

        # "mcmc_*" columns are kept for downstream compatibility with eval_math/passk scripts.
        regular_sample = std_sample

        naive_answer = parse_answer(naive_sample.completion)
        std_answer = parse_answer(std_sample.completion)
        regular_answer = parse_answer(regular_sample.completion)

        naive_reward = safe_grade(naive_answer, answer)
        std_reward = safe_grade(std_answer, answer)
        regular_reward = safe_grade(regular_answer, answer)

        base_seconds = std_sample.latency_seconds
        temp_seconds = naive_sample.latency_seconds
        regular_seconds = regular_sample.latency_seconds
        temp_output_tokens = len(naive_sample.token_ids)
        std_output_tokens = len(std_sample.token_ids)
        regular_output_tokens = len(regular_sample.token_ids)

        total_base_seconds += base_seconds
        total_temp_seconds += temp_seconds
        total_regular_seconds += regular_seconds
        total_samples += 1
        avg_base_seconds = total_base_seconds / max(total_samples, 1)
        avg_temp_seconds = total_temp_seconds / max(total_samples, 1)
        avg_regular_seconds = total_regular_seconds / max(total_samples, 1)

        results.append(
            {
                "question": question,
                "correct_answer": answer,
                "hf_model_id": args.hf_model_id,
                "prompt_style": "qwen_math",
                "naive_completion": naive_sample.completion,
                "naive_answer": naive_answer,
                "std_completion": std_sample.completion,
                "std_answer": std_answer,
                "regular_completion": regular_sample.completion,
                "regular_answer": regular_answer,
                "mcmc_completion": regular_sample.completion,
                "mcmc_generated_completion": regular_sample.completion,
                "mcmc_answer": regular_answer,
                "sampling_method": args.sampling_method,
                "base_sampling_seconds": base_seconds,
                "base_sampling_avg_seconds_so_far": avg_base_seconds,
                "temp_sampling_seconds": temp_seconds,
                "temp_sampling_avg_seconds_so_far": avg_temp_seconds,
                "temp_output_tokens": temp_output_tokens,
                "temp_sampling_tokens": temp_output_tokens,
                "power_sampling_seconds": regular_seconds,
                "power_sampling_avg_seconds_so_far": avg_regular_seconds,
                "std_output_tokens": std_output_tokens,
                "std_sampling_tokens": std_output_tokens,
                "regular_output_tokens": regular_output_tokens,
                "regular_sampling_tokens": regular_output_tokens,
                "power_acceptance_ratio": None,
                "power_sampling_tokens": regular_output_tokens,
                "power_output_tokens": regular_output_tokens,
                "power_internal_sampling_tokens": None,
                "naive_reward": naive_reward,
                "std_reward": std_reward,
                "regular_reward": regular_reward,
                "mcmc_reward": regular_reward,
            }
        )

        wandb_logger.log_metrics(
            {
                "latency/base_sampling_seconds": base_seconds,
                "latency/base_sampling_avg_seconds": avg_base_seconds,
                "latency/temp_sampling_seconds": temp_seconds,
                "latency/temp_sampling_avg_seconds": avg_temp_seconds,
                "latency/regular_sampling_seconds": regular_seconds,
                "latency/regular_sampling_avg_seconds": avg_regular_seconds,
                "reward/naive": naive_reward,
                "reward/std": std_reward,
                "reward/regular": regular_reward,
                "reward/mcmc_compat": regular_reward,
                "sampling/temp_output_tokens": temp_output_tokens,
                "sampling/std_output_tokens": std_output_tokens,
                "sampling/regular_output_tokens": regular_output_tokens,
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
                "regular_completion": truncate_text(regular_sample.completion),
                "naive_reward": naive_reward,
                "std_reward": std_reward,
                "regular_reward": regular_reward,
                "base_sampling_seconds": base_seconds,
                "temp_sampling_seconds": temp_seconds,
                "regular_sampling_seconds": regular_seconds,
                "temp_output_tokens": temp_output_tokens,
                "std_output_tokens": std_output_tokens,
                "regular_output_tokens": regular_output_tokens,
            }
        )

    output_path = os.path.join(
        save_str,
        f"{model_slug}_math_base_regular_samp_results_{args.temperature}_{args.batch_idx}_{args.seed}.csv",
    )
    pd.DataFrame(results).to_csv(output_path, index=False)
    wandb_logger.log_file(output_path)

    avg_base_seconds = total_base_seconds / max(total_samples, 1)
    avg_temp_seconds = total_temp_seconds / max(total_samples, 1)
    avg_regular_seconds = total_regular_seconds / max(total_samples, 1)
    print(f"Saved results to: {output_path}")
    print(f"Average base sampling time per sample: {avg_base_seconds:.4f} seconds")
    print(f"Average temp sampling time per sample: {avg_temp_seconds:.4f} seconds")
    print(f"Average regular sampling time per sample: {avg_regular_seconds:.4f} seconds")

    wandb_logger.finish(
        summary={
            "summary/num_samples": total_samples,
            "summary/avg_base_sampling_seconds": avg_base_seconds,
            "summary/avg_temp_sampling_seconds": avg_temp_seconds,
            "summary/avg_regular_sampling_seconds": avg_regular_seconds,
        }
    )
