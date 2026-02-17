#!/usr/bin/env python3
"""Rename W&B runs to a standardized math naming convention.

Format:
math_<model>_<sampling_method>_t<temp>_k<top_k>_m<rollouts>_b<block_size>_shard<batch_idx>_seed<seed>
"""

from __future__ import annotations

import argparse
import math
import os
import re
from typing import Any, Dict, Iterable, Optional

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "jeffhernandez1995"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "reasoning-with-sampling"))
    parser.add_argument(
        "--states",
        default="running",
        help="Comma-separated W&B run states to consider (e.g. running,finished,crashed).",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Rename all matching runs, including ones already starting with math_.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply changes. Without this flag, script performs a dry run.",
    )
    return parser.parse_args()


def pick(config: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in config and config[key] is not None:
            return config[key]
    return None


def format_float_token(value: Any) -> str:
    if value is None:
        return "na"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    try:
        val = float(value)
        if not math.isfinite(val):
            return "na"
        if val.is_integer():
            return str(int(val))
        return f"{val:.6g}"
    except Exception:
        return str(value)


def sanitize_token(value: Any) -> str:
    token = "na" if value is None else str(value).strip()
    if token == "" or token.lower() in {"none", "null"}:
        token = "na"
    token = token.replace("/", "-")
    token = re.sub(r"\s+", "", token)
    token = re.sub(r"[^A-Za-z0-9._-]+", "-", token).strip("-")
    return token or "na"


def build_name(config: Dict[str, Any]) -> str:
    model = sanitize_token(pick(config, "model"))
    sampling_method = sanitize_token(pick(config, "sampling_method"))
    temp = sanitize_token(format_float_token(pick(config, "temperature", "temp")))
    top_k = sanitize_token(pick(config, "top_k", "k"))
    rollouts = sanitize_token(pick(config, "rollouts_per_candidate", "rollouts", "mcmc_steps", "m"))
    block_size = sanitize_token(pick(config, "block_size", "b"))
    shard = sanitize_token(pick(config, "batch_idx", "shard"))
    seed = sanitize_token(pick(config, "seed"))
    return (
        f"math_{model}_{sampling_method}_t{temp}_k{top_k}"
        f"_m{rollouts}_b{block_size}_shard{shard}_seed{seed}"
    )


def iter_runs(api: wandb.Api, entity: str, project: str, states: Iterable[str]):
    wanted_states = {s.strip().lower() for s in states if s.strip()}
    for run in api.runs(f"{entity}/{project}"):
        if wanted_states and (run.state or "").lower() not in wanted_states:
            continue
        yield run


def main() -> int:
    args = parse_args()
    states = [s.strip() for s in args.states.split(",") if s.strip()]
    api = wandb.Api()

    total = 0
    renamed = 0
    skipped = 0

    for run in iter_runs(api, args.entity, args.project, states):
        total += 1
        old_name = run.name or ""
        if (not args.all_runs) and old_name.startswith("math_"):
            skipped += 1
            continue

        config = {k: v for k, v in (run.config or {}).items() if not str(k).startswith("_")}
        new_name = build_name(config)
        if new_name == old_name:
            skipped += 1
            continue

        print(f"{run.id}: {old_name} -> {new_name}")
        if args.apply:
            run.name = new_name
            run.update()
            renamed += 1

    mode = "APPLY" if args.apply else "DRY-RUN"
    print(f"[{mode}] inspected={total} renamed={renamed} skipped={skipped} states={','.join(states) or 'all'}")
    if not args.apply:
        print("Re-run with --apply to perform renames.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

