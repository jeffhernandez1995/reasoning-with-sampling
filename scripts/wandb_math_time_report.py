#!/usr/bin/env python3
"""Summarize MATH timing from W&B runs.

Computes per-question timing averaged over seeds for a selected sampling method
and prints a report alongside provided accuracy numbers.
"""

from __future__ import annotations

import argparse
import math
import os
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional

import wandb


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default=os.environ.get("WANDB_ENTITY", "jeffhernandez1995"))
    parser.add_argument("--project", default=os.environ.get("WANDB_PROJECT", "reasoning-with-sampling"))
    parser.add_argument("--model", default="qwen_math")
    parser.add_argument("--sampling_method", default="power")
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--dataset", default="MATH")
    parser.add_argument("--states", default="finished")
    parser.add_argument("--metric", default="summary/avg_power_sampling_seconds")
    parser.add_argument("--expected_seeds", type=int, default=8)
    parser.add_argument("--expected_shards_per_seed", type=int, default=5)
    parser.add_argument("--base_acc", type=float, default=0.484)
    parser.add_argument("--temp_acc", type=float, default=0.683)
    parser.add_argument("--mcmc_acc", type=float, default=0.729)
    return parser.parse_args()


def is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def matches_filters(config: Dict[str, Any], args: argparse.Namespace) -> bool:
    if config.get("dataset") != args.dataset:
        return False
    if config.get("model") != args.model:
        return False
    if config.get("sampling_method") != args.sampling_method:
        return False
    temp = config.get("temperature")
    if not is_finite_number(temp):
        return False
    return abs(float(temp) - float(args.temperature)) < 1e-12


def extract_metric(run: Any, metric_key: str) -> Optional[float]:
    summary = dict(run.summary) if run.summary is not None else {}
    val = summary.get(metric_key)
    if is_finite_number(val):
        return float(val)

    # Fallback: mean from run history if summary key is missing.
    history_key = metric_key.replace("summary/", "")
    try:
        hist = run.history(keys=[history_key], pandas=True)
    except Exception:
        return None
    if history_key not in hist:
        return None
    series = hist[history_key].dropna()
    if series.empty:
        return None
    return float(series.mean())


def main() -> int:
    args = parse_args()
    wanted_states = {s.strip().lower() for s in args.states.split(",") if s.strip()}

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}")

    rows: List[Dict[str, Any]] = []
    for run in runs:
        state = (run.state or "").lower()
        if wanted_states and state not in wanted_states:
            continue

        config = {k: v for k, v in dict(run.config).items() if not str(k).startswith("_")}
        if not matches_filters(config, args):
            continue

        value = extract_metric(run, args.metric)
        if value is None:
            continue

        seed = config.get("seed")
        shard = config.get("batch_idx")
        if not isinstance(seed, int) or not isinstance(shard, int):
            continue

        rows.append(
            {
                "id": run.id,
                "name": run.name,
                "seed": seed,
                "shard": shard,
                "time_per_q": value,
            }
        )

    if not rows:
        print("No matching runs with timing metric were found.")
        return 1

    by_seed: Dict[int, List[float]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row["time_per_q"])

    seed_means = {seed: mean(vals) for seed, vals in by_seed.items() if vals}
    overall_seed_avg = mean(seed_means.values())
    overall_run_avg = mean([row["time_per_q"] for row in rows])

    print("Timing source:")
    print(f"- entity/project: {args.entity}/{args.project}")
    print(f"- filters: dataset={args.dataset} model={args.model} sampling_method={args.sampling_method} temperature={args.temperature}")
    print(f"- runs used: {len(rows)}")
    print()
    print("Per-seed average time (s/question):")
    for seed in sorted(seed_means.keys()):
        count = len(by_seed[seed])
        print(f"- seed {seed}: {seed_means[seed]:.4f}  (runs={count})")
    print()
    print("Aggregate timing:")
    print(f"- mean over seed means: {overall_seed_avg:.4f} s/question")
    print(f"- mean over all runs:   {overall_run_avg:.4f} s/question")
    print()

    # These are not logged separately in current power_samp_math.py runs.
    base_time = None
    temp_time = None
    mcmc_time = overall_seed_avg

    print("Requested summary:")
    print(f"- Base accuracy: {args.base_acc:.3f}, time={'N/A (not logged separately in W&B)' if base_time is None else f'{base_time:.4f}s'}")
    print(f"- Temp accuracy: {args.temp_acc:.3f}, time={'N/A (not logged separately in W&B)' if temp_time is None else f'{temp_time:.4f}s'}")
    print(f"- MCMC accuracy: {args.mcmc_acc:.3f}, time={mcmc_time:.4f}s")
    print()
    print("Note: current logging includes `latency/power_sampling_seconds` (MCMC method),")
    print("but does not include separate latency metrics for base (`std`) or temp (`naive`) decoding.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

