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

from arena_fighters.evaluation import artifact_metadata


def run_command(
    cmd: list[str],
    cwd: Path,
    stdout_path: Path,
    *,
    timeout_seconds: float | None = None,
) -> None:
    try:
        with stdout_path.open("w") as stdout:
            subprocess.run(
                cmd,
                cwd=cwd,
                stdout=stdout,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=timeout_seconds,
            )
    except subprocess.TimeoutExpired as exc:
        with stdout_path.open("a") as stdout:
            stdout.write(f"\nCommand timed out after {timeout_seconds} seconds\n")
        raise RuntimeError(
            f"Reward-shaping smoke command timed out after {timeout_seconds} "
            f"seconds: {cmd}"
        ) from exc


def latest_artifact(output_dir: Path, label: str) -> Path:
    matches = sorted(output_dir.glob(f"*_{label}.json"))
    if not matches:
        raise FileNotFoundError(f"Missing artifact for label: {label}")
    return matches[-1]


def _number(value: object) -> float | None:
    if type(value) in {int, float}:
        return float(value)
    return None


def _check_result(check_id: str, passed: bool, details: dict) -> dict:
    return {
        "id": check_id,
        "passed": passed,
        "details": details,
    }


def reward_smoke_checks(summary: dict) -> list[dict]:
    checks = []
    for agent_name in ("agent_0", "agent_1"):
        metric = f"reward_delta_{agent_name}"
        reward_delta = _number(summary.get(metric))
        checks.append(
            _check_result(
                f"{metric}_negative",
                reward_delta is not None and reward_delta < 0.0,
                {
                    "metric": metric,
                    "value": reward_delta,
                    "threshold": 0.0,
                },
            )
        )

    draw_rate_delta = _number(summary.get("draw_rate_delta"))
    checks.append(
        _check_result(
            "draw_rate_delta_not_positive",
            draw_rate_delta is not None and draw_rate_delta <= 0.0,
            {
                "metric": "draw_rate_delta",
                "value": draw_rate_delta,
                "threshold": 0.0,
            },
        )
    )
    return checks


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
    deltas = comparison["deltas"]
    summary = {
        "artifact": artifact_metadata("reward_shaping_smoke"),
        "output_dir": str(output_dir),
        "default_eval": str(default_eval),
        "anti_stall_eval": str(anti_eval),
        "reward_delta_agent_0": deltas.get("avg_rewards.agent_0"),
        "reward_delta_agent_1": deltas.get("avg_rewards.agent_1"),
        "draw_rate_delta": deltas.get("draw_rate"),
        "idle_rate_delta_agent_0": deltas.get("behavior.avg_idle_rate.agent_0"),
        "dominant_action_rate_delta_agent_0": deltas.get(
            "behavior.avg_dominant_action_rate.agent_0"
        ),
        "no_damage_episodes_delta": deltas.get("behavior.no_damage_episodes"),
        "low_engagement_episodes_delta": deltas.get(
            "behavior.low_engagement_episodes"
        ),
        "damage_events_delta_agent_0": deltas.get("behavior.damage_events.agent_0"),
        "strategy_issue_count": strategy_report["issue_count"],
        "indexed_artifact_count": artifact_index["index_config"]["artifact_count"],
    }
    checks = reward_smoke_checks(summary)
    summary["checks"] = checks
    summary["passed"] = all(check["passed"] for check in checks)
    return summary


def write_smoke_summary(summary: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output_path


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
        "--summary-output",
        type=str,
        default=None,
        help="Optional path for saving the reward-shaping smoke summary JSON",
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
    parser.add_argument(
        "--command-timeout-seconds",
        type=float,
        default=300.0,
        help="Per-command timeout in seconds (default: 300)",
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
        timeout_seconds=args.command_timeout_seconds,
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
        timeout_seconds=args.command_timeout_seconds,
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
        timeout_seconds=args.command_timeout_seconds,
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
        timeout_seconds=args.command_timeout_seconds,
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
        timeout_seconds=args.command_timeout_seconds,
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
        timeout_seconds=args.command_timeout_seconds,
    )

    summary = build_smoke_summary(output_dir)
    if args.summary_output:
        summary["summary_output"] = str(Path(args.summary_output))
        write_smoke_summary(summary, args.summary_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
