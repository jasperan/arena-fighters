#!/usr/bin/env python
"""Run a tiny train-to-eval smoke workflow.

This script is for wiring checks, not learning-quality claims. It trains for a
small timestep budget, evaluates the produced final checkpoint against a compact
baseline suite, runs a relaxed promotion audit, then writes strategy-report and
artifact-index summaries.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path


DEFAULT_SMOKE_SUITE_OPPONENTS = "idle,scripted,aggressive,evasive"
DEFAULT_SMOKE_SUITE_MAPS = "classic,flat,split,tower"
ALLOWED_SMOKE_LONG_RUN_FAILURES = {
    "no_candidate_bad_strategy_issues",
    "no_replay_bad_strategy_issues",
}


def run_command(
    cmd: list[str],
    cwd: Path,
    stdout_path: Path,
    *,
    check: bool = True,
) -> int:
    with stdout_path.open("w") as stdout:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return result.returncode


def latest_artifact(output_dir: Path, label: str) -> Path:
    matches = sorted(output_dir.glob(f"*_{label}.json"))
    if not matches:
        raise FileNotFoundError(f"Missing artifact for label: {label}")
    return matches[-1]


def build_promotion_audit_command(
    base_cmd: list[str],
    checkpoint: Path,
    suite_opponents: str,
    suite_maps: str,
    rounds: int,
    eval_dir: Path,
) -> list[str]:
    return base_cmd + [
        "--mode",
        "promotion_audit",
        "--rank-checkpoints",
        str(checkpoint),
        "--suite-opponents",
        suite_opponents,
        "--suite-maps",
        suite_maps,
        "--rounds",
        str(rounds),
        "--rank-min-score",
        "-1",
        "--rank-max-draw-rate",
        "1",
        "--rank-max-no-damage-rate",
        "1",
        "--rank-max-low-engagement-rate",
        "1",
        "--eval-output-dir",
        str(eval_dir),
        "--eval-label",
        "train-smoke-promotion",
    ]


def _csv_count(value: str) -> int:
    return len([item for item in value.split(",") if item])


def build_long_run_check_command(
    base_cmd: list[str],
    promotion_audit: Path,
    strategy_report: Path,
    artifact_index: Path,
    suite_opponents: str,
    suite_maps: str,
    rounds: int,
    eval_dir: Path,
) -> list[str]:
    map_count = _csv_count(suite_maps)
    opponent_count = _csv_count(suite_opponents)
    return base_cmd + [
        "--mode",
        "long_run_check",
        "--promotion-audit-summary",
        str(promotion_audit),
        "--strategy-report-summary",
        str(strategy_report),
        "--artifact-index-summary",
        str(artifact_index),
        "--long-run-min-maps",
        str(map_count),
        "--long-run-min-eval-episodes",
        str(rounds * map_count * opponent_count),
        "--long-run-min-map-episodes",
        str(rounds * opponent_count),
        "--long-run-require-candidate-checkpoint",
        "--long-run-require-candidate-metadata",
        "--long-run-require-candidate-integrity",
        "--eval-output-dir",
        str(eval_dir),
        "--eval-label",
        "train-smoke-long-run-check",
    ]


def long_run_check_failed_required_checks(summary: dict) -> list[str]:
    return [
        check["id"]
        for check in summary.get("checks", [])
        if check.get("required", True) and not check.get("passed")
    ]


def validate_smoke_long_run_check(long_run_check: Path, exit_code: int) -> None:
    summary = json.loads(long_run_check.read_text())
    failed_checks = set(long_run_check_failed_required_checks(summary))
    unexpected_failures = sorted(failed_checks - ALLOWED_SMOKE_LONG_RUN_FAILURES)
    if unexpected_failures:
        raise RuntimeError(
            "Diagnostic long_run_check failed unexpected required checks: "
            + ", ".join(unexpected_failures)
        )
    passed = bool(summary.get("passed"))
    if passed and exit_code != 0:
        raise RuntimeError(
            "Diagnostic long_run_check passed but returned non-zero exit code "
            f"{exit_code}"
        )
    if not passed and exit_code == 0:
        raise RuntimeError("Diagnostic long_run_check failed but returned exit code 0")


def build_train_eval_summary(output_dir: Path) -> dict:
    checkpoint = output_dir / "checkpoints" / "ppo_final.zip"
    metadata = output_dir / "checkpoints" / "ppo_final.meta.json"
    suite = latest_artifact(output_dir / "evals", "train-smoke-suite")
    suite_data = json.loads(suite.read_text())
    suite_config = suite_data.get("suite_config", {})
    suite_overview = suite_data.get("overview", {})
    promotion_audit = None
    try:
        promotion_audit = latest_artifact(
            output_dir / "evals",
            "train-smoke-promotion",
        )
    except FileNotFoundError:
        pass
    long_run_check = None
    long_run_check_data = None
    try:
        long_run_check = latest_artifact(
            output_dir / "evals",
            "train-smoke-long-run-check",
        )
        long_run_check_data = json.loads(long_run_check.read_text())
    except FileNotFoundError:
        pass
    long_run_check_exit_code = None
    long_run_exit_code_path = output_dir / "long-run-check.exitcode"
    if long_run_exit_code_path.exists():
        long_run_check_exit_code = int(long_run_exit_code_path.read_text().strip())
    strategy_report = latest_artifact(output_dir / "evals", "train-smoke-strategy")
    artifact_index = latest_artifact(output_dir / "evals", "train-smoke-index")
    replay_batch = latest_artifact(output_dir / "evals", "train-smoke-replay-batch")
    strategy = json.loads(strategy_report.read_text())
    index = json.loads(artifact_index.read_text())
    replay_analysis_count = index["artifact_counts"].get("replay_analysis", 0)
    return {
        "output_dir": str(output_dir),
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "metadata": str(metadata),
        "metadata_exists": metadata.exists(),
        "replay_count": len(list((output_dir / "replays").glob("*.json"))),
        "replay_analysis_batch": str(replay_batch),
        "replay_analysis_count": replay_analysis_count,
        "suite": str(suite),
        "suite_opponents": suite_config.get("opponents", []),
        "suite_maps": suite_config.get("maps", []),
        "suite_total_matchups": suite_overview.get("total_matchups", 0),
        "suite_total_episodes": suite_overview.get("total_episodes", 0),
        "promotion_audit": str(promotion_audit) if promotion_audit else None,
        "long_run_check": str(long_run_check) if long_run_check else None,
        "long_run_check_passed": (
            long_run_check_data.get("passed") if long_run_check_data else None
        ),
        "long_run_check_exit_code": long_run_check_exit_code,
        "long_run_check_failed_checks": long_run_check_failed_required_checks(
            long_run_check_data or {}
        ),
        "strategy_report": str(strategy_report),
        "strategy_issue_count": strategy["issue_count"],
        "artifact_index": str(artifact_index),
        "indexed_artifact_count": index["index_config"]["artifact_count"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tiny train/eval smoke")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: system temp dir)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=128,
        help="Requested training timesteps (default: 128)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Suite rounds per matchup (default: 1)",
    )
    parser.add_argument(
        "--suite-opponents",
        type=str,
        default=DEFAULT_SMOKE_SUITE_OPPONENTS,
        help=(
            "Suite opponents "
            f"(default: {DEFAULT_SMOKE_SUITE_OPPONENTS})"
        ),
    )
    parser.add_argument(
        "--suite-maps",
        type=str,
        default=DEFAULT_SMOKE_SUITE_MAPS,
        help=f"Suite maps (default: {DEFAULT_SMOKE_SUITE_MAPS})",
    )
    parser.add_argument(
        "--skip-promotion-audit",
        action="store_true",
        help="Skip relaxed promotion-audit smoke over the generated checkpoint",
    )
    parser.add_argument(
        "--skip-long-run-check",
        action="store_true",
        help=(
            "Skip diagnostic long_run_check smoke. It is allowed to fail "
            "because tiny runs often have expected strategy issues."
        ),
    )
    parser.add_argument(
        "--replay-save-interval",
        type=int,
        default=1,
        help="Save one training replay every N completed episodes (default: 1)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_py = repo_root / "scripts" / "train.py"
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = Path(
        args.output_dir
        or Path(tempfile.gettempdir()) / f"arena-train-eval-smoke-{timestamp}"
    )
    checkpoint_dir = output_dir / "checkpoints"
    replay_dir = output_dir / "replays"
    eval_dir = output_dir / "evals"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    replay_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    base_cmd = [sys.executable, str(train_py)]
    run_command(
        base_cmd
        + [
            "--mode",
            "train",
            "--timesteps",
            str(args.timesteps),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--replay-dir",
            str(replay_dir),
            "--curriculum",
            "map_progression",
            "--replay-save-interval",
            str(args.replay_save_interval),
        ],
        repo_root,
        output_dir / "train.out",
    )

    checkpoint = checkpoint_dir / "ppo_final.zip"
    run_command(
        base_cmd
        + [
            "--mode",
            "suite",
            "--checkpoint",
            str(checkpoint),
            "--suite-opponents",
            args.suite_opponents,
            "--suite-maps",
            args.suite_maps,
            "--rounds",
            str(args.rounds),
            "--eval-output-dir",
            str(eval_dir),
            "--eval-label",
            "train-smoke-suite",
        ],
        repo_root,
        output_dir / "suite.out",
    )
    if not args.skip_promotion_audit:
        run_command(
            build_promotion_audit_command(
                base_cmd,
                checkpoint,
                args.suite_opponents,
                args.suite_maps,
                args.rounds,
                eval_dir,
            ),
            repo_root,
            output_dir / "promotion-audit.out",
        )
    run_command(
        base_cmd
        + [
            "--mode",
            "analyze",
            "--replay-dir",
            str(replay_dir),
            "--replay-samples-per-bucket",
            "1",
            "--eval-output-dir",
            str(eval_dir),
            "--eval-label",
            "train-smoke-replay",
        ],
        repo_root,
        output_dir / "replay-analysis.out",
    )
    run_command(
        base_cmd
        + [
            "--mode",
            "strategy_report",
            "--artifact-dir",
            str(eval_dir),
            "--eval-output-dir",
            str(eval_dir),
            "--eval-label",
            "train-smoke-strategy",
        ],
        repo_root,
        output_dir / "strategy.out",
    )
    run_command(
        base_cmd
        + [
            "--mode",
            "artifact_index",
            "--artifact-dir",
            str(eval_dir),
            "--eval-output-dir",
            str(eval_dir),
            "--eval-label",
            "train-smoke-index",
        ],
        repo_root,
        output_dir / "artifact-index.out",
    )

    if not args.skip_promotion_audit and not args.skip_long_run_check:
        long_run_exit_code = run_command(
            build_long_run_check_command(
                base_cmd,
                latest_artifact(eval_dir, "train-smoke-promotion"),
                latest_artifact(eval_dir, "train-smoke-strategy"),
                latest_artifact(eval_dir, "train-smoke-index"),
                args.suite_opponents,
                args.suite_maps,
                args.rounds,
                eval_dir,
            ),
            repo_root,
            output_dir / "long-run-check.out",
            check=False,
        )
        (output_dir / "long-run-check.exitcode").write_text(
            f"{long_run_exit_code}\n"
        )
        validate_smoke_long_run_check(
            latest_artifact(eval_dir, "train-smoke-long-run-check"),
            long_run_exit_code,
        )
        run_command(
            base_cmd
            + [
                "--mode",
                "artifact_index",
                "--artifact-dir",
                str(eval_dir),
                "--eval-output-dir",
                str(eval_dir),
                "--eval-label",
                "train-smoke-index",
            ],
            repo_root,
            output_dir / "artifact-index-final.out",
        )

    print(json.dumps(build_train_eval_summary(output_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
