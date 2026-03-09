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
    rollout_tag = (
        f"mr{args.moment_rollouts}"
        if args.sampling_method == "power_cumulant"
        else f"m{args.rollouts_per_candidate}"
    )
    budget_suffix = ""
    stop_suffix = ""
    strategy_suffix = ""
    if args.sampling_method == "power_approx":
        budget_suffix = "" if args.max_sampling_tokens is None else f"_budget{args.max_sampling_tokens}"
        stop_suffix = "" if args.approx_stop_mode == "eos" else f"_stop{args.approx_stop_mode}"
        if args.approx_stop_mode == "budget":
            strategy_suffix = f"_{args.approx_budget_strategy}"
            if args.approx_budget_strategy == "restart":
                strategy_suffix += f"_{args.approx_selection_mode}"
    return (
        f"{args.dataset.lower()}_{args.model}_{args.sampling_method}"
        f"_t{_float_token(args.temperature)}_k{args.top_k}"
        f"_{rollout_tag}_b{args.block_size}{budget_suffix}{stop_suffix}{strategy_suffix}"
        f"_shard{args.batch_idx:02d}_seed{args.seed:02d}"
    )


def build_method_config(args, eos_token_id: Optional[int]) -> Dict[str, Any]:
    method_config: Dict[str, Any] = {
        "top_k": args.top_k,
        "candidate_pool_size": args.candidate_pool_size,
        "lookahead_tokens": args.lookahead_tokens,
        "block_size": args.block_size,
        "seed": args.seed,
        "eos_token_id": eos_token_id,
    }
    if args.sampling_method == "power_approx":
        method_config.update(
            {
                "rollouts_per_candidate": args.rollouts_per_candidate,
                "use_jackknife": args.use_jackknife,
                "stop_mode": args.approx_stop_mode,
                "budget_strategy": args.approx_budget_strategy,
                "selection_mode": args.approx_selection_mode,
                "answer_extractor": parse_answer,
                "max_sampling_tokens": args.max_sampling_tokens,
            }
        )
    elif args.sampling_method == "power_cumulant":
        method_config.update(
            {
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
            }
        )
    elif args.sampling_method == "power_tilted_cgf":
        method_config.update(
            {
                "rollouts_per_candidate": args.rollouts_per_candidate,
                "proposal_temperature": args.proposal_temperature,
                "proposal_top_p": args.proposal_top_p,
                "proposal_top_k": args.proposal_top_k,
                "zeta_weight": args.zeta_weight,
                "length_normalize_logp": args.length_normalize_logp,
                "length_penalty": args.length_penalty,
                "rollout_stop_on_eos": args.rollout_stop_on_eos,
            }
        )
    else:
        raise ValueError(f"Unsupported sampling_method: {args.sampling_method}")
    return method_config


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
    parser.add_argument(
        "--sampling_method",
        type=str,
        default="power_approx",
        choices=["power_approx", "power_cumulant", "power_tilted_cgf"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=3072)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--candidate_pool_size", type=int, default=32)
    parser.add_argument("--rollouts_per_candidate", type=int, default=8)
    parser.add_argument("--lookahead_tokens", type=int, default=192)
    parser.add_argument("--block_size", type=int, default=192)
    parser.add_argument("--use_jackknife", type=parse_bool, default=True)
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
    parser.add_argument("--proposal_temperature", type=float, default=None)
    parser.add_argument("--proposal_top_p", type=float, default=1.0)
    parser.add_argument("--proposal_top_k", type=int, default=None)
    parser.add_argument("--approx_stop_mode", type=str, default="eos", choices=["eos", "budget"])
    parser.add_argument("--approx_budget_strategy", type=str, default="restart", choices=["restart"])
    parser.add_argument(
        "--approx_selection_mode",
        type=str,
        default="weighted_vote",
        choices=["last", "best_logp", "vote", "weighted_vote"],
    )
    parser.add_argument(
        "--max_sampling_tokens",
        "--max_total_effort_tokens",
        dest="max_sampling_tokens",
        type=int,
        default=None,
        help="Hard cap on sampler compute tokens (`sampling_tokens` = output + internal proposal tokens).",
    )
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

    if args.sampling_method != "power_approx" and args.approx_stop_mode != "eos":
        raise ValueError(f"{args.sampling_method} does not support --approx_stop_mode yet.")
    if args.sampling_method != "power_approx" and args.max_sampling_tokens is not None:
        raise ValueError(f"{args.sampling_method} does not support --max_sampling_tokens yet.")
    if args.approx_stop_mode == "budget" and args.max_sampling_tokens is None:
        raise ValueError("--approx_stop_mode budget requires --max_sampling_tokens.")

    if args.dataset != "MATH":
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    save_str = os.path.join(args.save_str, args.sampling_method, args.model)
    os.makedirs(save_str, exist_ok=True)

    with open("data/MATH500.json", "r") as f:
        dataset = json.load(f)

    model, tokenizer = load_model_and_tokenizer(args.model, args.device, trust_remote_code=True)
    sampler = GenericSampler(model, tokenizer, args.device)

    method_config = build_method_config(args, tokenizer.eos_token_id)

    print(
        (
            f"Approx setup: method={args.sampling_method} temperature={args.temperature} "
            f"top_k={args.top_k} candidate_pool_size={args.candidate_pool_size} "
            f"rollouts_per_candidate={args.rollouts_per_candidate} block_size={args.block_size} "
            f"approx_stop_mode={args.approx_stop_mode} "
            f"approx_budget_strategy={args.approx_budget_strategy} "
            f"approx_selection_mode={args.approx_selection_mode} "
            f"max_new_tokens={args.max_new_tokens} max_sampling_tokens={args.max_sampling_tokens}"
        ),
        flush=True,
    )

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
            **{k: v for k, v in method_config.items() if k not in {"eos_token_id", "answer_extractor"}},
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

    output_suffix_parts = [
        f"k{args.top_k}",
        f"m{args.rollouts_per_candidate}",
        f"b{args.block_size}",
        str(args.temperature),
    ]
    if args.sampling_method == "power_approx" and args.max_sampling_tokens is not None:
        output_suffix_parts.append(f"budget{args.max_sampling_tokens}")
    if args.sampling_method == "power_approx" and args.approx_stop_mode != "eos":
        output_suffix_parts.append(f"stop{args.approx_stop_mode}")
        output_suffix_parts.append(args.approx_budget_strategy)
        if args.approx_budget_strategy == "restart":
            output_suffix_parts.append(args.approx_selection_mode)
    output_suffix_parts.extend([str(args.batch_idx), str(args.seed)])
    output_path = os.path.join(
        save_str,
        f"{args.model}_math_base_{args.sampling_method}_results_{'_'.join(output_suffix_parts)}.csv",
    )
    partial_output_path = f"{output_path}.partial"

    print(
        (
            f"Starting MATH({args.sampling_method}): batch_idx={args.batch_idx} seed={args.seed} "
            f"questions={total_questions} device={args.device} save_every={args.save_every}"
        ),
        flush=True,
    )
    print(f"Output path: {output_path}", flush=True)

    results = []
    total_base_seconds = 0.0
    total_temp_seconds = 0.0
    total_approx_seconds = 0.0
    total_approx_samples = 0

    for step_idx, data in enumerate(
        tqdm(dataset[start:end], desc=f"Benchmark on MATH ({args.sampling_method})")
    ):
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
                method_config=method_config,
            )
            if args.cuda_sync and torch.cuda.is_available():
                torch.cuda.synchronize()
            if args.debug_verbose:
                print(
                    (
                        f"[step {step_idx + 1}] {args.sampling_method} latency={method_sample.latency_seconds:.2f}s "
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
            approx_seconds = method_sample.latency_seconds
            temp_output_tokens = len(naive_sample.token_ids)
            std_output_tokens = len(std_sample.token_ids)
            total_base_seconds += base_seconds
            total_temp_seconds += temp_seconds
            total_approx_seconds += approx_seconds
            total_approx_samples += 1
            avg_base_seconds = total_base_seconds / max(total_approx_samples, 1)
            avg_temp_seconds = total_temp_seconds / max(total_approx_samples, 1)
            avg_approx_seconds = total_approx_seconds / max(total_approx_samples, 1)

            approx_rollouts = method_sample.metadata.get("rollouts")
            if approx_rollouts is None:
                approx_rollouts = method_sample.metadata.get("moment_rollouts")
            approx_rollout_tokens = method_sample.metadata.get("rollout_tokens")
            approx_steps = method_sample.metadata.get("steps")
            approx_candidate_tokens = method_sample.metadata.get("candidate_tokens")
            approx_sampling_tokens = method_sample.metadata.get("sampling_tokens")
            approx_output_tokens = method_sample.metadata.get("output_tokens")
            approx_internal_sampling_tokens = method_sample.metadata.get("internal_sampling_tokens")
            approx_stop_mode = method_sample.metadata.get("approx_stop_mode")
            approx_budget_strategy = method_sample.metadata.get("approx_budget_strategy")
            approx_selection_mode = method_sample.metadata.get("approx_selection_mode")
            approx_budget_remaining_tokens = method_sample.metadata.get("budget_remaining_sampling_tokens")
            approx_budget_exhausted = method_sample.metadata.get("budget_exhausted")
            approx_budget_stop_reason = method_sample.metadata.get("budget_stop_reason")
            approx_chains_started = method_sample.metadata.get("approx_chains_started")
            approx_completed_candidates = method_sample.metadata.get("approx_completed_candidates")
            approx_selected_candidate_source = method_sample.metadata.get("approx_selected_candidate_source")
            method_specific_config = (
                method_sample.metadata.get("approx_config")
                or method_sample.metadata.get("cumulant_config")
                or method_sample.metadata.get("tilted_cgf_config")
                or {}
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
                    # Keep mcmc_* column names so existing eval scripts keep working.
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
                    "power_sampling_seconds": approx_seconds,
                    "power_sampling_avg_seconds_so_far": avg_approx_seconds,
                    "std_output_tokens": std_output_tokens,
                    "std_sampling_tokens": std_output_tokens,
                    "power_acceptance_ratio": None,
                    "approx_steps": approx_steps,
                    "approx_rollouts": approx_rollouts,
                    "approx_rollout_tokens": approx_rollout_tokens,
                    "approx_candidate_tokens": approx_candidate_tokens,
                    "approx_sampling_tokens": approx_sampling_tokens,
                    "approx_output_tokens": approx_output_tokens,
                    "approx_internal_sampling_tokens": approx_internal_sampling_tokens,
                    "approx_stop_mode": approx_stop_mode,
                    "approx_budget_strategy": approx_budget_strategy,
                    "approx_selection_mode": approx_selection_mode,
                    "approx_budget_remaining_sampling_tokens": approx_budget_remaining_tokens,
                    "approx_budget_exhausted": approx_budget_exhausted,
                    "approx_budget_stop_reason": approx_budget_stop_reason,
                    "approx_chains_started": approx_chains_started,
                    "approx_completed_candidates": approx_completed_candidates,
                    "approx_selected_candidate_source": approx_selected_candidate_source,
                    "power_sampling_tokens": approx_sampling_tokens,
                    "power_output_tokens": approx_output_tokens,
                    "power_internal_sampling_tokens": approx_internal_sampling_tokens,
                    "approx_config": json.dumps(method_specific_config),
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
                    "sampling/temp_output_tokens": temp_output_tokens,
                    "sampling/temp_sampling_tokens": temp_output_tokens,
                    "sampling/std_output_tokens": std_output_tokens,
                    "sampling/std_sampling_tokens": std_output_tokens,
                    "sampling/approx_steps": approx_steps,
                    "sampling/approx_rollouts": approx_rollouts,
                    "sampling/approx_rollout_tokens": approx_rollout_tokens,
                    "sampling/approx_candidate_tokens": approx_candidate_tokens,
                    "sampling/approx_sampling_tokens": approx_sampling_tokens,
                    "sampling/approx_output_tokens": approx_output_tokens,
                    "sampling/approx_internal_sampling_tokens": approx_internal_sampling_tokens,
                    "sampling/approx_budget_remaining_sampling_tokens": approx_budget_remaining_tokens,
                    "sampling/approx_budget_exhausted": int(bool(approx_budget_exhausted)),
                    "sampling/power_sampling_tokens": approx_sampling_tokens,
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
                    "mcmc_completion": truncate_text(method_sample.completion),
                    "naive_reward": naive_reward,
                    "std_reward": std_reward,
                    "mcmc_reward": method_reward,
                    "base_sampling_seconds": base_seconds,
                    "temp_sampling_seconds": temp_seconds,
                    "power_sampling_seconds": approx_seconds,
                    "temp_output_tokens": temp_output_tokens,
                    "temp_sampling_tokens": temp_output_tokens,
                    "std_output_tokens": std_output_tokens,
                    "std_sampling_tokens": std_output_tokens,
                    "approx_steps": approx_steps,
                    "approx_rollouts": approx_rollouts,
                    "approx_sampling_tokens": approx_sampling_tokens,
                    "approx_output_tokens": approx_output_tokens,
                    "approx_stop_mode": approx_stop_mode,
                    "approx_budget_strategy": approx_budget_strategy,
                    "approx_selection_mode": approx_selection_mode,
                    "approx_budget_stop_reason": approx_budget_stop_reason,
                }
            )

            if args.save_every > 0 and (step_idx + 1) % args.save_every == 0:
                pd.DataFrame(results).to_csv(partial_output_path, index=False)
                if args.debug_verbose:
                    print(
                        f"[checkpoint] wrote {len(results)} rows to {partial_output_path}",
                        flush=True,
                    )

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

    avg_base_seconds = total_base_seconds / max(total_approx_samples, 1)
    avg_temp_seconds = total_temp_seconds / max(total_approx_samples, 1)
    avg_approx_seconds = total_approx_seconds / max(total_approx_samples, 1)
    print(f"Saved results to: {output_path}")
    print(f"Average base sampling time per sample: {avg_base_seconds:.4f} seconds")
    print(f"Average temp sampling time per sample: {avg_temp_seconds:.4f} seconds")
    print(f"Average {args.sampling_method} sampling time per sample: {avg_approx_seconds:.4f} seconds")

    wandb_logger.finish(
        summary={
            "summary/num_samples": total_approx_samples,
            "summary/avg_base_sampling_seconds": avg_base_seconds,
            "summary/avg_temp_sampling_seconds": avg_temp_seconds,
            "summary/avg_power_sampling_seconds": avg_approx_seconds,
        }
    )
