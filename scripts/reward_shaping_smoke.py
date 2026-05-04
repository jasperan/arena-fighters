#!/usr/bin/env python
"""Run a short reward-shaping artifact smoke workflow.

The smoke avoids training. It uses deterministic built-in idle policies to make
anti-stall reward changes visible in saved eval, compare, suite, strategy, and
artifact-index JSON.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def run_command(cmd: list[str], cwd: Path, stdout_path: Path) -> None:
    with stdout_path.open("w") as stdout:
        subprocess.run(
            cmd,
            cwd=cwd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=True,
        )


def latest_artifact(output_dir: Path, label: str) -> Path:
    matches = sorted(output_dir.glob(f"*_{label}.json"))
    if not matches:
        raise FileNotFoundError(f"Missing artifact for label: {label}")
    return matches[-1]


def build_smoke_summary(output_dir: Path) -> dict:
    default_eval = latest_artifact(output_dir, "idle-default")
    anti_eval = latest_artifact(output_dir, "idle-anti-stall")
    comparison = json.loads(
        latest_artifact(output_dir, "idle-reward-compare").read_text()
    )
    strategy_report = json.loads(
        latest_artifact(output_dir, "strategy-report").read_text()
    )
    artifact_index = json.loads(
        latest_artifact(output_dir, "artifact-index").read_text()
    )
    return {
        "output_dir": str(output_dir),
        "default_eval": str(default_eval),
        "anti_stall_eval": str(anti_eval),
        "reward_delta_agent_0": comparison["deltas"].get("avg_rewards.agent_0"),
        "reward_delta_agent_1": comparison["deltas"].get("avg_rewards.agent_1"),
        "draw_rate_delta": comparison["deltas"].get("draw_rate"),
        "strategy_issue_count": strategy_report["issue_count"],
        "indexed_artifact_count": artifact_index["index_config"]["artifact_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run deterministic reward-shaping smoke artifacts"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for artifacts (default: system temp dir)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Eval/suite rounds per matchup (default: 1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Eval seed (default: 1)",
    )
    parser.add_argument(
        "--map",
        dest="map_name",
        default="flat",
        help="Map used for the smoke (default: flat)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_py = repo_root / "scripts" / "train.py"
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = Path(
        args.output_dir
        or Path(tempfile.gettempdir()) / f"arena-reward-shaping-smoke-{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [sys.executable, str(train_py)]
    common_eval = [
        "--mode",
        "eval",
        "--agent-policy",
        "idle",
        "--opponent",
        "idle",
        "--rounds",
        str(args.rounds),
        "--seed",
        str(args.seed),
        "--map",
        args.map_name,
        "--eval-output-dir",
        str(output_dir),
    ]

    run_command(
        base_cmd + common_eval + ["--eval-label", "idle-default"],
        repo_root,
        output_dir / "idle-default.out",
    )
    run_command(
        base_cmd
        + common_eval
        + [
            "--reward-preset",
            "anti_stall",
            "--eval-label",
            "idle-anti-stall",
        ],
        repo_root,
        output_dir / "idle-anti-stall.out",
    )

    default_eval = latest_artifact(output_dir, "idle-default")
    anti_eval = latest_artifact(output_dir, "idle-anti-stall")
    run_command(
        base_cmd
        + [
            "--mode",
            "compare",
            "--before",
            str(default_eval),
            "--after",
            str(anti_eval),
            "--eval-output-dir",
            str(output_dir),
            "--eval-label",
            "idle-reward-compare",
        ],
        repo_root,
        output_dir / "idle-reward-compare.out",
    )

    run_command(
        base_cmd
        + [
            "--mode",
            "suite",
            "--agent-policy",
            "idle",
            "--suite-opponents",
            "idle",
            "--suite-maps",
            args.map_name,
            "--rounds",
            str(args.rounds),
            "--seed",
            str(args.seed),
            "--reward-preset",
            "anti_stall",
            "--eval-output-dir",
            str(output_dir),
            "--eval-label",
            "idle-suite",
        ],
        repo_root,
        output_dir / "idle-suite.out",
    )

    run_command(
        base_cmd
        + [
            "--mode",
            "strategy_report",
            "--artifact-dir",
            str(output_dir),
            "--eval-output-dir",
            str(output_dir),
            "--eval-label",
            "strategy-report",
        ],
        repo_root,
        output_dir / "strategy-report.out",
    )
    run_command(
        base_cmd
        + [
            "--mode",
            "artifact_index",
            "--artifact-dir",
            str(output_dir),
            "--eval-output-dir",
            str(output_dir),
            "--eval-label",
            "artifact-index",
        ],
        repo_root,
        output_dir / "artifact-index.out",
    )

    summary = build_smoke_summary(output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
