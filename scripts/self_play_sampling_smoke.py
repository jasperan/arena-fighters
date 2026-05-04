#!/usr/bin/env python
"""Verify historical-opponent sampling without running PPO training."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena_fighters.config import IDLE, NUM_ACTIONS, PLATFORM_LAYOUTS, Config
from arena_fighters.evaluation import artifact_metadata, write_eval_summary
from arena_fighters.self_play import OpponentPool, SelfPlayWrapper


DEFAULT_MAP_POOL = ("classic", "flat", "split", "tower")


class RecordingOpponentPolicy:
    """Small load_state_dict target used to exercise wrapper sampling."""

    def __init__(self):
        self.loaded_snapshots: list[dict[str, Any]] = []
        self.training_mode: bool | None = None
        self.action = IDLE

    def load_state_dict(self, state_dict: dict) -> None:
        self.loaded_snapshots.append(dict(state_dict))
        self.action = int(state_dict.get("action", IDLE))

    def set_training_mode(self, mode: bool) -> None:
        self.training_mode = mode

    def predict(self, obs, deterministic: bool = False):
        return self.action, None


def parse_map_pool(value: str) -> tuple[str, ...]:
    maps = tuple(item.strip() for item in value.split(",") if item.strip())
    if not maps:
        raise ValueError("map pool must include at least one map")
    unknown = [map_name for map_name in maps if map_name not in PLATFORM_LAYOUTS]
    if unknown:
        raise ValueError(f"Unknown map names: {', '.join(unknown)}")
    return maps


def _check(check_id: str, passed: bool, **details) -> dict:
    return {"id": check_id, "passed": bool(passed), **details}


def build_self_play_sampling_summary(
    *,
    pool_seed: int = 123,
    reset_seed: int = 1000,
    snapshot_count: int = 5,
    reset_count: int = 64,
    latest_opponent_prob: float = 0.8,
    min_historical_samples: int = 1,
    min_maps_seen: int = 2,
    map_pool: tuple[str, ...] = DEFAULT_MAP_POOL,
) -> dict:
    if snapshot_count < 2:
        raise ValueError("snapshot_count must be at least 2")
    if reset_count < 1:
        raise ValueError("reset_count must be at least 1")
    if not 0.0 <= latest_opponent_prob <= 1.0:
        raise ValueError("latest_opponent_prob must be between 0.0 and 1.0")
    if min_historical_samples < 0:
        raise ValueError("min_historical_samples must be non-negative")
    if min_maps_seen < 0:
        raise ValueError("min_maps_seen must be non-negative")

    cfg = Config(
        training=replace(
            Config().training,
            latest_opponent_prob=latest_opponent_prob,
        )
    )
    pool = OpponentPool(max_size=snapshot_count, seed=pool_seed)
    for snapshot_index in range(snapshot_count):
        pool.add(
            {
                "snapshot_index": snapshot_index,
                "action": snapshot_index % NUM_ACTIONS,
            }
        )

    opponent_policy = RecordingOpponentPolicy()
    wrapper = SelfPlayWrapper(
        config=cfg,
        opponent_pool=pool,
        opponent_policy=opponent_policy,
    )
    wrapper.set_map_pool(map_pool)

    samples = []
    map_counts: Counter[str] = Counter()
    loaded_reset_count = 0
    for reset_index in range(reset_count):
        _, info = wrapper.reset(seed=reset_seed + reset_index)
        state = wrapper.get_state()
        map_name = state.get("map_name")
        map_counts[str(map_name)] += 1
        pool_info = info.get("opponent_pool", {})
        if info.get("opponent_snapshot_loaded"):
            loaded_reset_count += 1
        loaded_snapshot = (
            opponent_policy.loaded_snapshots[-1]
            if opponent_policy.loaded_snapshots
            else {}
        )
        samples.append(
            {
                "reset_index": reset_index,
                "reset_seed": reset_seed + reset_index,
                "map_name": map_name,
                "sample_kind": pool_info.get("last_sample_kind"),
                "sample_id": pool_info.get("last_sample_id"),
                "sample_index": pool_info.get("last_sample_index"),
                "loaded_snapshot_index": loaded_snapshot.get("snapshot_index"),
                "loaded_action": loaded_snapshot.get("action"),
            }
        )

    pool_stats = pool.stats()
    unique_maps_seen = len([count for count in map_counts.values() if count > 0])
    needs_latest_sample = 0.0 < latest_opponent_prob <= 1.0
    checks = [
        _check(
            "snapshot_count_at_least_two",
            snapshot_count >= 2,
            snapshot_count=snapshot_count,
        ),
        _check(
            "opponent_snapshot_loaded_each_reset",
            loaded_reset_count == reset_count,
            loaded_reset_count=loaded_reset_count,
            reset_count=reset_count,
        ),
        _check(
            "historical_samples_meet_minimum",
            pool_stats["historical_samples"] >= min_historical_samples,
            historical_samples=pool_stats["historical_samples"],
            min_historical_samples=min_historical_samples,
        ),
        _check(
            "latest_samples_present_when_probability_allows",
            (not needs_latest_sample) or pool_stats["latest_samples"] > 0,
            latest_samples=pool_stats["latest_samples"],
            latest_opponent_prob=latest_opponent_prob,
        ),
        _check(
            "map_pool_coverage_meets_minimum",
            unique_maps_seen >= min_maps_seen,
            unique_maps_seen=unique_maps_seen,
            min_maps_seen=min_maps_seen,
            map_counts=dict(sorted(map_counts.items())),
        ),
    ]

    return {
        "artifact": artifact_metadata("self_play_sampling_smoke"),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
        "pool_config": {
            "seed": pool_seed,
            "snapshot_count": snapshot_count,
            "latest_opponent_prob": latest_opponent_prob,
            "reset_count": reset_count,
            "reset_seed": reset_seed,
            "min_historical_samples": min_historical_samples,
        },
        "map_pool": list(map_pool),
        "map_counts": dict(sorted(map_counts.items())),
        "unique_maps_seen": unique_maps_seen,
        "loaded_reset_count": loaded_reset_count,
        "latest_samples": pool_stats["latest_samples"],
        "historical_samples": pool_stats["historical_samples"],
        "historical_sample_rate": pool_stats["historical_sample_rate"],
        "pool_stats": pool_stats,
        "samples": samples,
    }


def validate_self_play_sampling_summary(summary: dict) -> None:
    if summary.get("passed") is not True:
        failed = [
            str(check.get("id"))
            for check in summary.get("checks", [])
            if isinstance(check, dict) and check.get("passed") is False
        ]
        raise RuntimeError(
            "Self-play sampling smoke failed checks: " + ", ".join(failed)
        )


def write_sampling_summary(summary: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run self-play sampling smoke")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory for timestamped smoke artifact output",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional path for saving the smoke summary JSON",
    )
    parser.add_argument("--pool-seed", type=int, default=123)
    parser.add_argument("--reset-seed", type=int, default=1000)
    parser.add_argument("--snapshot-count", type=int, default=5)
    parser.add_argument("--reset-count", type=int, default=64)
    parser.add_argument("--latest-opponent-prob", type=float, default=0.8)
    parser.add_argument("--min-historical-samples", type=int, default=1)
    parser.add_argument("--min-maps-seen", type=int, default=2)
    parser.add_argument(
        "--map-pool",
        type=str,
        default=",".join(DEFAULT_MAP_POOL),
        help="Comma-separated map pool to sample during resets",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        map_pool = parse_map_pool(args.map_pool)
        summary = build_self_play_sampling_summary(
            pool_seed=args.pool_seed,
            reset_seed=args.reset_seed,
            snapshot_count=args.snapshot_count,
            reset_count=args.reset_count,
            latest_opponent_prob=args.latest_opponent_prob,
            min_historical_samples=args.min_historical_samples,
            min_maps_seen=args.min_maps_seen,
            map_pool=map_pool,
        )
        validate_self_play_sampling_summary(summary)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        output_dir = Path(tempfile.gettempdir()) / f"arena-self-play-smoke-{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.summary_output:
        path = Path(args.summary_output)
        summary["summary_output"] = str(path)
        summary["artifact_path"] = str(path)
        write_sampling_summary(summary, path)
    else:
        path = write_eval_summary(
            summary,
            output_dir,
            label="self-play-sampling-smoke",
        )
        summary["artifact_path"] = str(path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
