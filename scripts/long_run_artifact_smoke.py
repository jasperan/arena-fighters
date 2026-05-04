#!/usr/bin/env python
"""Run a no-training long-run artifact plumbing smoke workflow.

This smoke generates a long-run manifest, writes long-run status and league
health artifacts, then indexes the bundle. It intentionally does not execute the
generated launcher or start training.
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


ALLOWED_NO_TRAINING_BLOCKED_REASONS = {
    "latest_launcher_not_executed",
    "latest_manifest_source_stale",
}


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


def latest_artifact(output_dir: Path, label: str) -> Path:
    matches = sorted(output_dir.glob(f"*_{label}.json"))
    if not matches:
        raise FileNotFoundError(f"Missing artifact for label: {label}")
    return matches[-1]


def build_artifact_smoke_summary(
    output_dir: Path,
    artifact_root: Path,
    run_id: str,
) -> dict:
    eval_dir = artifact_root / run_id
    manifest = latest_artifact(artifact_root, "artifact-smoke-plan")
    status = latest_artifact(eval_dir, "long-run-status")
    health = latest_artifact(eval_dir, "league-health")
    artifact_index = latest_artifact(eval_dir, "artifact-smoke-index")
    status_data = json.loads(status.read_text())
    health_data = json.loads(health.read_text())
    index_data = json.loads(artifact_index.read_text())
    return {
        "artifact": artifact_metadata("long_run_artifact_smoke"),
        "output_dir": str(output_dir),
        "artifact_root": str(artifact_root),
        "run_id": run_id,
        "manifest": str(manifest),
        "long_run_status": str(status),
        "league_health": str(health),
        "artifact_index": str(artifact_index),
        "status_blocked_reason": status_data.get("blocked_reason"),
        "status_missing_evidence": status_data.get("missing_evidence", []),
        "health_ready": health_data.get("health", {}).get("ready"),
        "health_blockers": health_data.get("health", {}).get("blockers", []),
        "health_warnings": health_data.get("health", {}).get("warnings", []),
        "health_artifact_scope_dir": health_data.get("health_config", {}).get(
            "artifact_scope_dir"
        ),
        "indexed_artifact_counts": index_data.get("artifact_counts", {}),
        "indexed_artifact_count": index_data.get("index_config", {}).get(
            "artifact_count"
        ),
    }


def validate_artifact_smoke_summary(summary: dict, eval_dir: Path) -> None:
    counts = summary["indexed_artifact_counts"]
    required_counts = {
        "long_run_manifest": 1,
        "long_run_status": 1,
        "league_health": 1,
    }
    missing = [
        artifact_type
        for artifact_type, minimum in required_counts.items()
        if counts.get(artifact_type, 0) < minimum
    ]
    if missing:
        raise RuntimeError(
            "Artifact smoke index is missing required artifact types: "
            + ", ".join(missing)
        )
    if summary["health_artifact_scope_dir"] != str(eval_dir):
        raise RuntimeError(
            "League health did not scope artifacts to the generated run eval dir: "
            f"{summary['health_artifact_scope_dir']!r}"
        )
    if summary["status_blocked_reason"] not in ALLOWED_NO_TRAINING_BLOCKED_REASONS:
        raise RuntimeError(
            "Expected no-training status to stop before launcher execution, got "
            f"{summary['status_blocked_reason']!r}"
        )


def write_artifact_smoke_summary(summary: dict, path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run no-training long-run artifact smoke"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: system temp dir)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for the generated manifest (default: timestamped)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=128,
        help="Manifest timesteps value without executing training (default: 128)",
    )
    parser.add_argument(
        "--summary-output",
        type=str,
        default=None,
        help="Optional path for saving the long-run artifact smoke summary JSON",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    train_py = repo_root / "scripts" / "train.py"
    timestamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output_dir = Path(
        args.output_dir
        or Path(tempfile.gettempdir()) / f"arena-long-run-artifact-smoke-{timestamp}"
    )
    run_id = args.run_id or f"artifact-smoke-{timestamp}"
    artifact_root = output_dir / "evals"
    eval_dir = artifact_root / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    base_cmd = [sys.executable, str(train_py)]
    run_command(
        base_cmd
        + [
            "--mode",
            "long_run_manifest",
            "--run-id",
            run_id,
            "--timesteps",
            str(args.timesteps),
            "--artifact-dir",
            str(artifact_root),
            "--eval-output-dir",
            str(artifact_root),
            "--eval-label",
            "artifact-smoke-plan",
        ],
        repo_root,
        output_dir / "long-run-manifest.out",
    )
    run_command(
        base_cmd
        + [
            "--mode",
            "long_run_status",
            "--artifact-dir",
            str(artifact_root),
            "--eval-output-dir",
            str(eval_dir),
            "--eval-label",
            "long-run-status",
        ],
        repo_root,
        output_dir / "long-run-status.out",
    )
    run_command(
        base_cmd
        + [
            "--mode",
            "league_health",
            "--artifact-dir",
            str(artifact_root),
            "--eval-output-dir",
            str(eval_dir),
            "--eval-label",
            "league-health",
        ],
        repo_root,
        output_dir / "league-health.out",
    )
    run_command(
        base_cmd
        + [
            "--mode",
            "artifact_index",
            "--artifact-dir",
            str(artifact_root),
            "--recursive-artifacts",
            "--eval-output-dir",
            str(eval_dir),
            "--eval-label",
            "artifact-smoke-index",
        ],
        repo_root,
        output_dir / "artifact-index.out",
    )

    summary = build_artifact_smoke_summary(output_dir, artifact_root, run_id)
    validate_artifact_smoke_summary(summary, eval_dir)
    if args.summary_output:
        summary["summary_output"] = str(Path(args.summary_output))
        write_artifact_smoke_summary(summary, args.summary_output)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
