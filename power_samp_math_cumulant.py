import argparse
import json
import os
import random
import sys
import traceback
from typing import Any, Dict, Optional

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


def _cuda_device_index(device: str) -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    if not str(device).startswith("cuda"):
        return None
    try:
        cuda_device = torch.device(device)
        if cuda_device.index is not None:
            return int(cuda_device.index)
        return int(torch.cuda.current_device())
    except Exception:
        return None


def cuda_memory_snapshot(device: str) -> Dict[str, float]:
    device_idx = _cuda_device_index(device)
    if device_idx is None:
        return {}
    gib = float(1024**3)
    try:
        return {
            "allocated_gb": float(torch.cuda.memory_allocated(device_idx)) / gib,
            "reserved_gb": float(torch.cuda.memory_reserved(device_idx)) / gib,
            "max_allocated_gb": float(torch.cuda.max_memory_allocated(device_idx)) / gib,
            "max_reserved_gb": float(torch.cuda.max_memory_reserved(device_idx)) / gib,
        }
    except Exception as exc:
        return {"error": str(exc)}


def format_cuda_snapshot(snapshot: Dict[str, float]) -> str:
    if not snapshot:
        return "cuda=unavailable"
    if "error" in snapshot:
        return f"cuda_error={snapshot['error']}"
    return (
        f"alloc={snapshot['allocated_gb']:.2f}GB "
        f"reserved={snapshot['reserved_gb']:.2f}GB "
        f"max_alloc={snapshot['max_allocated_gb']:.2f}GB "
        f"max_reserved={snapshot['max_reserved_gb']:.2f}GB"
    )


def default_wandb_run_name(args) -> str:
    return (
        f"{args.dataset.lower()}_{args.model}_{args.sampling_method}"
        f"_t{_float_token(args.temperature)}_k{args.top_k}"
        f"_mr{args.moment_rollouts}_co{args.cumulant_order}_b{args.block_size}"
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
    parser.add_argument("--sampling_method", type=str, default="power_cumulant", choices=["power_cumulant"])
    parser.add_argument("--max_new_tokens", type=int, default=3072)

    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--candidate_pool_size", type=int, default=32)
    parser.add_argument("--lookahead_tokens", type=int, default=192)
    parser.add_argument("--block_size", type=int, default=192)

    parser.add_argument("--moment_rollouts", type=int, default=1)
    parser.add_argument("--cumulant_order", type=int, default=2)
    parser.add_argument("--zeta_weight", type=float, default=1.0)
    parser.add_argument("--varentropy_coef", type=float, default=1.0)
    parser.add_argument("--length_normalize_logp", type=parse_bool, default=False)
    parser.add_argument("--length_penalty", type=float, default=1.0)
    parser.add_argument("--rollout_stop_on_eos", type=parse_bool, default=True)

    parser.add_argument("--rollout_temperature", type=float, default=1.0)
    parser.add_argument("--rollout_top_p", type=float, default=1.0)
    parser.add_argument("--rollout_top_k", type=int, default=None)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_idx", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_questions", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--debug_verbose", type=parse_bool, default=True)
    parser.add_argument("--cuda_sync", type=parse_bool, default=False)
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

    cumulant_method_config: Dict[str, Any] = {
        "top_k": args.top_k,
        "candidate_pool_size": args.candidate_pool_size,
        "lookahead_tokens": args.lookahead_tokens,
        "block_size": args.block_size,
        "moment_rollouts": args.moment_rollouts,
        "cumulant_order": args.cumulant_order,
        "zeta_weight": args.zeta_weight,
        "varentropy_coef": args.varentropy_coef,
        "length_normalize_logp": args.length_normalize_logp,
        "length_penalty": args.length_penalty,
        "rollout_stop_on_eos": args.rollout_stop_on_eos,
        "rollout_temperature": args.rollout_temperature,
        "rollout_top_p": args.rollout_top_p,
        "rollout_top_k": args.rollout_top_k,
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
            **{k: v for k, v in cumulant_method_config.items() if k not in {"eos_token_id"}},
            "wandb_run_name": wandb_run_name,
        },
    )

    start = 100 * args.batch_idx
    end = min(100 * (args.batch_idx + 1), len(dataset))
    if args.max_questions is not None:
        end = min(end, start + max(0, int(args.max_questions)))
    total_questions = max(0, end - start)
    if total_questions == 0:
        print(
            f"No questions selected for batch_idx={args.batch_idx} with max_questions={args.max_questions}.",
            flush=True,
        )
        sys.exit(0)

    output_path = os.path.join(
        save_str,
        (
            f"{args.model}_math_base_power_cumulant_results_"
            f"k{args.top_k}_mr{args.moment_rollouts}_co{args.cumulant_order}_"
            f"b{args.block_size}_t{args.temperature}_{args.batch_idx}_{args.seed}.csv"
        ),
    )
    partial_output_path = f"{output_path}.partial"

    print(
        (
            f"Starting MATH(power_cumulant): batch_idx={args.batch_idx} seed={args.seed} "
            f"questions={total_questions} device={args.device} save_every={args.save_every}"
        ),
        flush=True,
    )
    print(f"Output path: {output_path}", flush=True)

    results = []
    total_base_seconds = 0.0
    total_temp_seconds = 0.0
    total_cumulant_seconds = 0.0
    total_samples = 0

    for step_idx, data in enumerate(tqdm(dataset[start:end], desc=f"Benchmark on MATH ({args.sampling_method})")):
        dataset_index = start + step_idx
        question = data["prompt"]
        answer = data["answer"]

        if args.debug_verbose:
            print(
                (
                    f"[step {step_idx + 1}/{total_questions}] dataset_index={dataset_index} "
                    f"question_chars={len(question)}"
                ),
                flush=True,
            )
            print(f"[cuda] before-step {format_cuda_snapshot(cuda_memory_snapshot(args.device))}", flush=True)

        try:
            input_text = format_prompt(question, args.model, tokenizer, args.cot)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(args.device)
            if args.debug_verbose:
                print(f"[step {step_idx + 1}] prompt_tokens={int(input_ids.shape[1])}", flush=True)

            if args.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            naive_sample = sampler.sample_temperature(
                input_ids=input_ids,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
            )
            if args.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            if args.debug_verbose:
                print(
                    (
                        f"[step {step_idx + 1}] temperature latency={naive_sample.latency_seconds:.2f}s "
                        f"tokens={len(naive_sample.token_ids)}"
                    ),
                    flush=True,
                )

            if args.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            std_sample = sampler.sample_standard(input_ids=input_ids, max_new_tokens=args.max_new_tokens)
            if args.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            if args.debug_verbose:
                print(
                    (
                        f"[step {step_idx + 1}] standard latency={std_sample.latency_seconds:.2f}s "
                        f"tokens={len(std_sample.token_ids)}"
                    ),
                    flush=True,
                )

            if args.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            method_sample = sampler.sample_method(
                input_ids=input_ids,
                method=args.sampling_method,
                temperature=args.temperature,
                mcmc_steps=0,
                max_new_tokens=args.max_new_tokens,
                method_config=cumulant_method_config,
            )
            if args.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            if args.debug_verbose:
                print(
                    (
                        f"[step {step_idx + 1}] power_cumulant latency={method_sample.latency_seconds:.2f}s "
                        f"tokens={len(method_sample.token_ids)}"
                    ),
                    flush=True,
                )

            naive_answer = parse_answer(naive_sample.completion)
            std_answer = parse_answer(std_sample.completion)
            method_answer = parse_answer(method_sample.completion)

            naive_reward = safe_grade(naive_answer, answer)
            std_reward = safe_grade(std_answer, answer)
            method_reward = safe_grade(method_answer, answer)

            base_seconds = std_sample.latency_seconds
            temp_seconds = naive_sample.latency_seconds
            cumulant_seconds = method_sample.latency_seconds
            temp_output_tokens = len(naive_sample.token_ids)
            std_output_tokens = len(std_sample.token_ids)
            total_base_seconds += base_seconds
            total_temp_seconds += temp_seconds
            total_cumulant_seconds += cumulant_seconds
            total_samples += 1
            avg_base_seconds = total_base_seconds / max(total_samples, 1)
            avg_temp_seconds = total_temp_seconds / max(total_samples, 1)
            avg_cumulant_seconds = total_cumulant_seconds / max(total_samples, 1)

            method_metadata = method_sample.metadata
            cumulant_steps = method_metadata.get("steps")
            cumulant_moment_rollouts = method_metadata.get("moment_rollouts")
            cumulant_rollout_tokens = method_metadata.get("rollout_tokens")
            cumulant_candidate_tokens = method_metadata.get("candidate_tokens")
            cumulant_sampling_tokens = method_metadata.get("sampling_tokens")
            cumulant_output_tokens = method_metadata.get("output_tokens")
            cumulant_internal_sampling_tokens = method_metadata.get("internal_sampling_tokens")

            if cumulant_output_tokens is None:
                cumulant_output_tokens = len(method_sample.token_ids)
            if cumulant_sampling_tokens is None:
                cumulant_sampling_tokens = (
                    float(cumulant_output_tokens)
                    + float(cumulant_rollout_tokens or 0.0)
                    + float(cumulant_candidate_tokens or 0.0)
                )
            if cumulant_internal_sampling_tokens is None:
                cumulant_internal_sampling_tokens = max(
                    float(cumulant_sampling_tokens) - float(cumulant_output_tokens),
                    0.0,
                )

            results.append(
                {
                    "dataset_index": dataset_index,
                    "question": question,
                    "correct_answer": answer,
                    "naive_completion": naive_sample.completion,
                    "naive_answer": naive_answer,
                    "std_completion": std_sample.completion,
                    "std_answer": std_answer,
                    # Compatibility columns for existing evaluation scripts.
                    "mcmc_completion": method_sample.full_completion or method_sample.completion,
                    "mcmc_generated_completion": method_sample.completion,
                    "mcmc_answer": method_answer,
                    "sampling_method": args.sampling_method,
                    "base_sampling_seconds": base_seconds,
                    "base_sampling_avg_seconds_so_far": avg_base_seconds,
                    "temp_sampling_seconds": temp_seconds,
                    "temp_sampling_avg_seconds_so_far": avg_temp_seconds,
                    "temp_output_tokens": temp_output_tokens,
                    "temp_sampling_tokens": temp_output_tokens,
                    "power_sampling_seconds": cumulant_seconds,
                    "power_sampling_avg_seconds_so_far": avg_cumulant_seconds,
                    "std_output_tokens": std_output_tokens,
                    "std_sampling_tokens": std_output_tokens,
                    "power_acceptance_ratio": None,
                    "power_sampling_tokens": cumulant_sampling_tokens,
                    "power_output_tokens": cumulant_output_tokens,
                    "power_internal_sampling_tokens": cumulant_internal_sampling_tokens,
                    # Cumulant-native columns.
                    "cumulant_completion": method_sample.full_completion or method_sample.completion,
                    "cumulant_generated_completion": method_sample.completion,
                    "cumulant_answer": method_answer,
                    "cumulant_steps": cumulant_steps,
                    "cumulant_moment_rollouts": cumulant_moment_rollouts,
                    "cumulant_rollout_tokens": cumulant_rollout_tokens,
                    "cumulant_candidate_tokens": cumulant_candidate_tokens,
                    "cumulant_sampling_tokens": cumulant_sampling_tokens,
                    "cumulant_output_tokens": cumulant_output_tokens,
                    "cumulant_internal_sampling_tokens": cumulant_internal_sampling_tokens,
                    "cumulant_config": json.dumps(method_metadata.get("cumulant_config", {})),
                    "naive_reward": naive_reward,
                    "std_reward": std_reward,
                    "mcmc_reward": method_reward,
                    "cumulant_reward": method_reward,
                }
            )

            wandb_logger.log_metrics(
                {
                    "latency/base_sampling_seconds": base_seconds,
                    "latency/base_sampling_avg_seconds": avg_base_seconds,
                    "latency/temp_sampling_seconds": temp_seconds,
                    "latency/temp_sampling_avg_seconds": avg_temp_seconds,
                    "latency/power_sampling_seconds": cumulant_seconds,
                    "latency/power_sampling_avg_seconds": avg_cumulant_seconds,
                    "sampling/temp_output_tokens": temp_output_tokens,
                    "sampling/temp_sampling_tokens": temp_output_tokens,
                    "sampling/std_output_tokens": std_output_tokens,
                    "sampling/std_sampling_tokens": std_output_tokens,
                    "sampling/cumulant_steps": cumulant_steps,
                    "sampling/cumulant_moment_rollouts": cumulant_moment_rollouts,
                    "sampling/cumulant_rollout_tokens": cumulant_rollout_tokens,
                    "sampling/cumulant_candidate_tokens": cumulant_candidate_tokens,
                    "sampling/cumulant_sampling_tokens": cumulant_sampling_tokens,
                    "sampling/cumulant_output_tokens": cumulant_output_tokens,
                    "sampling/cumulant_internal_sampling_tokens": cumulant_internal_sampling_tokens,
                    "reward/naive": naive_reward,
                    "reward/std": std_reward,
                    "reward/mcmc": method_reward,
                },
                step=step_idx,
            )
            wandb_logger.log_sample(
                {
                    "dataset_index": dataset_index,
                    "question": truncate_text(question),
                    "correct_answer": answer,
                    "naive_completion": truncate_text(naive_sample.completion),
                    "std_completion": truncate_text(std_sample.completion),
                    "cumulant_completion": truncate_text(method_sample.completion),
                    "naive_reward": naive_reward,
                    "std_reward": std_reward,
                    "cumulant_reward": method_reward,
                    "base_sampling_seconds": base_seconds,
                    "temp_sampling_seconds": temp_seconds,
                    "power_sampling_seconds": cumulant_seconds,
                    "temp_output_tokens": temp_output_tokens,
                    "temp_sampling_tokens": temp_output_tokens,
                    "std_output_tokens": std_output_tokens,
                    "std_sampling_tokens": std_output_tokens,
                    "cumulant_steps": cumulant_steps,
                    "cumulant_moment_rollouts": cumulant_moment_rollouts,
                    "cumulant_rollout_tokens": cumulant_rollout_tokens,
                    "cumulant_sampling_tokens": cumulant_sampling_tokens,
                    "cumulant_output_tokens": cumulant_output_tokens,
                }
            )

            if args.save_every > 0 and (step_idx + 1) % args.save_every == 0:
                pd.DataFrame(results).to_csv(partial_output_path, index=False)
                if args.debug_verbose:
                    print(f"[checkpoint] wrote {len(results)} rows to {partial_output_path}", flush=True)

            if args.debug_verbose:
                print(f"[cuda] after-step {format_cuda_snapshot(cuda_memory_snapshot(args.device))}", flush=True)

        except Exception as exc:
            print(
                (
                    f"[error] step_idx={step_idx} dataset_index={dataset_index} "
                    f"type={type(exc).__name__}: {exc}"
                ),
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc()
            raise

    if results and (args.save_every <= 0 or len(results) % args.save_every != 0):
        pd.DataFrame(results).to_csv(partial_output_path, index=False)
        if args.debug_verbose:
            print(f"[checkpoint] wrote {len(results)} rows to {partial_output_path}", flush=True)

    pd.DataFrame(results).to_csv(output_path, index=False)
    if os.path.exists(partial_output_path):
        os.remove(partial_output_path)
    wandb_logger.log_file(output_path)

    avg_base_seconds = total_base_seconds / max(total_samples, 1)
    avg_temp_seconds = total_temp_seconds / max(total_samples, 1)
    avg_cumulant_seconds = total_cumulant_seconds / max(total_samples, 1)
    print(f"Saved results to: {output_path}")
    print(f"Average base sampling time per sample: {avg_base_seconds:.4f} seconds")
    print(f"Average temp sampling time per sample: {avg_temp_seconds:.4f} seconds")
    print(f"Average power_cumulant sampling time per sample: {avg_cumulant_seconds:.4f} seconds")

    wandb_logger.finish(
        summary={
            "summary/num_samples": total_samples,
            "summary/avg_base_sampling_seconds": avg_base_seconds,
            "summary/avg_temp_sampling_seconds": avg_temp_seconds,
            "summary/avg_power_sampling_seconds": avg_cumulant_seconds,
        }
    )
