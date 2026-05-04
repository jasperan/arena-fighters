#!/usr/bin/env python
"""Run cheap smoke workflows in compute-cost order.

By default this suite runs only no-training checks:

1. reward-shaping artifact smoke
2. long-run manifest/status/league-health artifact smoke

The tiny train/eval smoke is opt-in because it starts a short training job.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena_fighters.evaluation import artifact_metadata
from scripts.long_run_artifact_smoke import build_artifact_smoke_summary
from scripts.reward_shaping_smoke import build_smoke_summary as build_reward_summary
from scripts.train_eval_smoke import build_train_eval_summary


def run_command(cmd: list[str], cwd: Path, stdout_path: Path) -> None:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("w") as stdout:
        subprocess.run(
            cmd,
            cwd=cwd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=True,
        )


def build_smoke_commands(
    repo_root: Path,
    output_dir: Path,
    *,
    include_train_eval: bool = False,
    reward_rounds: int = 1,
    reward_map: str = "flat",
    train_eval_timesteps: int = 128,
    train_eval_rounds: int = 1,
) -> list[dict]:
    scripts_dir = repo_root / "scripts"
    commands = [
        {
            "id": "reward_shaping",
            "compute_class": "no_training_eval",
            "output_dir": output_dir / "reward-shaping",
            "stdout_path": output_dir / "reward-shaping.out",
            "cmd": [
                sys.executable,
                str(scripts_dir / "reward_shaping_smoke.py"),
                "--output-dir",
                str(output_dir / "reward-shaping"),
                "--rounds",
                str(reward_rounds),
                "--map",
                reward_map,
            ],
        },
        {
            "id": "long_run_artifact",
            "compute_class": "no_training_artifact",
            "output_dir": output_dir / "long-run-artifact",
            "stdout_path": output_dir / "long-run-artifact.out",
            "run_id": "artifact-smoke",
            "cmd": [
                sys.executable,
                str(scripts_dir / "long_run_artifact_smoke.py"),
                "--output-dir",
                str(output_dir / "long-run-artifact"),
                "--run-id",
                "artifact-smoke",
            ],
        },
    ]
    if include_train_eval:
        commands.append(
            {
                "id": "train_eval",
                "compute_class": "tiny_training",
                "output_dir": output_dir / "train-eval",
                "stdout_path": output_dir / "train-eval.out",
                "cmd": [
                    sys.executable,
                    str(scripts_dir / "train_eval_smoke.py"),
                    "--output-dir",
                    str(output_dir / "train-eval"),
                    "--timesteps",
                    str(train_eval_timesteps),
                    "--rounds",
                    str(train_eval_rounds),
                ],
            }
        )
    return commands


def build_smoke_suite_summary(output_dir: Path, commands: list[dict]) -> dict:
    command_by_id = {command["id"]: command for command in commands}
    smokes = {}
    reward_command = command_by_id.get("reward_shaping")
    if reward_command:
        smokes["reward_shaping"] = build_reward_summary(reward_command["output_dir"])
    artifact_command = command_by_id.get("long_run_artifact")
    if artifact_command:
        artifact_output = artifact_command["output_dir"]
        run_id = artifact_command.get("run_id", "artifact-smoke")
        smokes["long_run_artifact"] = build_artifact_smoke_summary(
            artifact_output,
            artifact_output / "evals",
            run_id,
        )
    train_eval_command = command_by_id.get("train_eval")
    if train_eval_command:
        smokes["train_eval"] = build_train_eval_summary(
            train_eval_command["output_dir"]
        )

    return {
        "artifact": artifact_metadata("smoke_suite"),
        "output_dir": str(output_dir),
        "smoke_count": len(commands),
        "smoke_order": [command["id"] for command in commands],
        "compute_classes": {
            command["id"]: command["compute_class"] for command in commands
        },
        "stdout_paths": {
            command["id"]: str(command["stdout_path"]) for command in commands
        },
        "smokes": smokes,
    }


def write_smoke_suite_summary(summary: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Arena Fighters smoke suite")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: /tmp timestamped dir)",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional path for saving the combined smoke-suite summary JSON",
    )
    parser.add_argument(
        "--reward-rounds",
        type=int,
        default=1,
        help="Reward-shaping smoke rounds per matchup (default: 1)",
    )
    parser.add_argument(
        "--reward-map",
        type=str,
        default="flat",
        help="Reward-shaping smoke map (default: flat)",
    )
    parser.add_argument(
        "--include-train-eval",
        action="store_true",
        help="Also run the tiny train/eval smoke; starts a short training job",
    )
    parser.add_argument(
        "--train-eval-timesteps",
        type=int,
        default=128,
        help="Tiny train/eval smoke timesteps when enabled (default: 128)",
    )
    parser.add_argument(
        "--train-eval-rounds",
        type=int,
        default=1,
        help="Tiny train/eval smoke rounds when enabled (default: 1)",
    )
    args = parser.parse_args()

    repo_root = REPO_ROOT
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = Path(args.output_dir or f"/tmp/arena-smoke-suite-{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    commands = build_smoke_commands(
        repo_root,
        output_dir,
        include_train_eval=args.include_train_eval,
        reward_rounds=args.reward_rounds,
        reward_map=args.reward_map,
        train_eval_timesteps=args.train_eval_timesteps,
        train_eval_rounds=args.train_eval_rounds,
    )

    for command in commands:
        run_command(command["cmd"], repo_root, command["stdout_path"])

    summary = build_smoke_suite_summary(output_dir, commands)
    if args.summary_output:
        summary["summary_output"] = str(Path(args.summary_output))
        write_smoke_suite_summary(summary, args.summary_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
