#!/usr/bin/env python
"""CLI entrypoint for arena-fighters: train, watch, and replay modes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import hmac
import json
import re
import shutil
import shlex
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from arena_fighters.config import (
    Config,
    CURRICULUMS,
    IDLE,
    NUM_ACTIONS,
    PLATFORM_LAYOUTS,
    REWARD_PRESET_ALIASES,
    REWARD_PRESETS,
    curriculum_stage_for_step,
    reward_config_for_preset,
)
from arena_fighters.env import ArenaFightersEnv
from arena_fighters.evaluation import (
    artifact_metadata,
    BUILTIN_POLICY_NAMES,
    ModelPolicy,
    compare_eval_summaries,
    evaluate_baseline_suite,
    evaluate_matchup,
    evaluate_pairwise_suite,
    gate_eval_comparison,
    gate_rank_summary,
    load_eval_summary,
    make_builtin_policy,
    rank_baseline_suites,
    ranking_per_map_score_details,
    score_baseline_suite,
    validate_artifact,
    write_eval_summary,
)
from arena_fighters.network import ArenaFeaturesExtractor
from arena_fighters.replay import ReplayLogger, analyze_replay, load_replay
from arena_fighters.self_play import OpponentPool, SelfPlayWrapper
from stable_baselines3.common.callbacks import BaseCallback


_SECRET_NAME_PATTERN = (
    r"[A-Za-z0-9_.-]*(?:api[_-]?key|access[_-]?key|secret|token|"
    r"password|passwd|pwd|private[_-]?key|database[_-]?url|db[_-]?url)"
    r"[A-Za-z0-9_.-]*"
)
_SECRET_VALUE_PATTERN = r'"[^"]*"|\'[^\']*\'|[^\s]+'
_SECRET_ASSIGNMENT_RE = re.compile(
    rf"(?i)\b({_SECRET_NAME_PATTERN})(\s*[:=]\s*)({_SECRET_VALUE_PATTERN})"
)
_BEARER_TOKEN_RE = re.compile(r"(?i)\b(authorization\s*[:=]\s*bearer\s+)[^\s]+")
_BASIC_AUTH_TOKEN_RE = re.compile(
    r"(?i)\b(authorization\s*[:=]\s*basic\s+)[^\s]+"
)
_HEADER_SECRET_RE = re.compile(
    r"(?i)\b((?:proxy-authorization|cookie|set-cookie|x-api-key)\s*[:=]\s*)[^\s]+"
)
_URL_CREDENTIAL_RE = re.compile(r"://([^:/\s]+):([^@\s]+)@")
_PRIVATE_KEY_LINE_RE = re.compile(r"(?i)-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")
_PRIVATE_KEY_ASSIGNMENT_RE = re.compile(
    rf"(?i)\b({_SECRET_NAME_PATTERN})(\s*[:=]\s*).*(PRIVATE KEY).*$"
)
_SECRET_ARG_RE = re.compile(
    rf"(?i)(--{_SECRET_NAME_PATTERN}(?:=|\s+))({_SECRET_VALUE_PATTERN})"
)
_JSON_SECRET_RE = re.compile(
    rf'(?i)("({_SECRET_NAME_PATTERN})"\s*:\s*")[^"]+(")'
)
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
REPLAY_STRATEGY_METRICS = {
    "replay_no_damage",
    "replay_low_engagement",
    "replay_no_attacks",
    "replay_idle_rate_agent_0",
    "replay_dominant_action_rate_agent_0",
}


def clear_terminal() -> None:
    print("\033[2J\033[H", end="")


def clone_policy_for_opponent(model):
    """Return a detached policy copy used only for frozen opponent snapshots."""
    policy = copy.deepcopy(model.policy)
    if hasattr(policy, "set_training_mode"):
        policy.set_training_mode(False)
    elif hasattr(policy, "eval"):
        policy.eval()
    return policy


def mirror_obs(obs: dict) -> dict:
    """Mirror observation for the opponent agent.

    Reuses the same logic as SelfPlayWrapper._mirror_obs but as a
    standalone function so watch mode doesn't need a full wrapper.
    """
    import numpy as np

    grid = obs["grid"].copy()
    vector = obs["vector"].copy()

    # Flip grid horizontally
    grid = np.flip(grid, axis=2).copy()
    # Swap own/opp position channels
    grid[[1, 2]] = grid[[2, 1]]
    # Swap own/opp bullet channels
    grid[[3, 4]] = grid[[4, 3]]
    # Swap own/opp HP in vector
    vector[0], vector[1] = vector[1], vector[0]

    return {"grid": grid, "vector": vector}


def curriculum_metadata(cfg: Config, step: int = 0) -> dict | None:
    if cfg.training.curriculum_name is None:
        return None

    stage = curriculum_stage_for_step(cfg.training.curriculum_name, step)
    return {
        "name": cfg.training.curriculum_name,
        "step": step,
        "stage": {
            "name": stage.name,
            "start_step": stage.start_step,
            "map_choices": list(stage.map_choices),
            "reward_preset": stage.reward_preset,
        },
        "active_map_pool": list(stage.map_choices),
        "active_reward_preset": stage.reward_preset,
    }


def effective_reward_config(cfg: Config, step: int = 0):
    if cfg.training.curriculum_name is None:
        return cfg.reward

    stage = curriculum_stage_for_step(cfg.training.curriculum_name, step)
    return reward_config_for_preset(stage.reward_preset)


def checkpoint_file_path(path: str | Path) -> Path:
    checkpoint_path = Path(path)
    if checkpoint_path.is_file():
        return checkpoint_path
    if checkpoint_path.suffix != ".zip":
        zip_path = Path(f"{checkpoint_path}.zip")
        if zip_path.is_file():
            return zip_path
    return checkpoint_path


def checkpoint_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as checkpoint_file:
        for chunk in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_file_metadata(path: str | Path) -> dict | None:
    checkpoint_path = checkpoint_file_path(path)
    if not checkpoint_path.is_file():
        return None
    stat = checkpoint_path.stat()
    return {
        "file_name": checkpoint_path.name,
        "size_bytes": stat.st_size,
        "sha256": checkpoint_file_sha256(checkpoint_path),
    }


def json_non_negative_int(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def json_number(value: object) -> float | None:
    if type(value) in {int, float}:
        return float(value)
    return None


def checkpoint_metadata(
    cfg: Config,
    num_timesteps: int,
    opponent_pool_stats: dict | None = None,
) -> dict:
    metadata = {
        "num_timesteps": num_timesteps,
        "map_name": cfg.arena.map_name,
        "randomize_maps": cfg.arena.randomize_maps,
        "map_choices": list(cfg.arena.map_choices),
        "reward": effective_reward_config(cfg, num_timesteps).__dict__,
        "curriculum": curriculum_metadata(cfg, num_timesteps),
        "opponent_pool_config": {
            "max_size": cfg.training.opponent_pool_size,
            "latest_opponent_prob": cfg.training.latest_opponent_prob,
            "seed": cfg.training.opponent_pool_seed,
        },
    }
    if opponent_pool_stats is not None:
        metadata["opponent_pool"] = opponent_pool_stats
    return metadata


def write_checkpoint_metadata(
    path: str | Path,
    cfg: Config,
    num_timesteps: int,
    opponent_pool_stats: dict | None = None,
) -> Path:
    metadata_path = Path(f"{path}.meta.json")
    metadata = checkpoint_metadata(cfg, num_timesteps, opponent_pool_stats)
    file_metadata = checkpoint_file_metadata(path)
    if file_metadata is not None:
        metadata["checkpoint_file"] = file_metadata
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata_path


def checkpoint_metadata_paths(path: str | Path | None) -> tuple[Path, ...]:
    if path is None:
        return ()

    checkpoint_path = Path(path)
    candidates = [Path(f"{checkpoint_path}.meta.json")]
    if checkpoint_path.suffix == ".zip":
        candidates.append(Path(f"{checkpoint_path.with_suffix('')}.meta.json"))
    else:
        candidates.append(Path(f"{checkpoint_path}.zip.meta.json"))

    seen: set[Path] = set()
    unique_candidates = []
    for metadata_path in candidates:
        if metadata_path in seen:
            continue
        seen.add(metadata_path)
        unique_candidates.append(metadata_path)
    return tuple(unique_candidates)


def checkpoint_metadata_path(path: str | Path | None) -> Path | None:
    for metadata_path in checkpoint_metadata_paths(path):
        if metadata_path.exists():
            return metadata_path
    return None


def read_checkpoint_metadata(path: str | Path | None) -> dict | None:
    metadata_path = checkpoint_metadata_path(path)
    if metadata_path is None:
        return None
    return json.loads(metadata_path.read_text())


def checkpoint_metadata_integrity(
    path: str | Path | None,
    metadata: dict | None,
) -> dict:
    details = {
        "checkpoint": str(path) if path is not None else None,
        "metadata_present": isinstance(metadata, dict),
    }
    if path is None:
        return {**details, "passed": False, "reason": "missing_checkpoint_path"}

    checkpoint_path = checkpoint_file_path(path)
    details["checkpoint_path"] = str(checkpoint_path)
    if not checkpoint_path.is_file():
        return {**details, "passed": False, "reason": "checkpoint_missing"}
    if not isinstance(metadata, dict):
        return {**details, "passed": False, "reason": "metadata_missing"}

    checkpoint_file = metadata.get("checkpoint_file") or {}
    expected_sha256 = checkpoint_file.get("sha256")
    expected_size = checkpoint_file.get("size_bytes")
    actual_size = checkpoint_path.stat().st_size
    actual_sha256 = checkpoint_file_sha256(checkpoint_path)
    details.update(
        {
            "expected_sha256": expected_sha256,
            "actual_sha256": actual_sha256,
            "expected_size_bytes": expected_size,
            "actual_size_bytes": actual_size,
        }
    )
    if not expected_sha256:
        return {**details, "passed": False, "reason": "metadata_sha256_missing"}

    sha256_matches = expected_sha256 == actual_sha256
    size_matches = expected_size is None or expected_size == actual_size
    if sha256_matches and size_matches:
        return {**details, "passed": True}
    if not sha256_matches:
        reason = "sha256_mismatch"
    else:
        reason = "size_mismatch"
    return {**details, "passed": False, "reason": reason}


def _normalize_sha256(value: object, *, source: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"Invalid SHA-256 digest in {source}")
    return value.lower()


def load_checkpoint_trust_manifest(path: str | Path) -> dict[str, str]:
    """Load checkpoint SHA-256 allowlist entries from a trusted JSON file.

    Accepted shapes:
    - {"checkpoints": {"checkpoints/ppo_final.zip": "<sha256>"}}
    - {"checkpoints": [{"path": "checkpoints/ppo_final.zip", "sha256": "<sha256>"}]}
    - {"checkpoints/ppo_final.zip": "<sha256>"}
    """
    manifest_path = Path(path)
    data = json.loads(manifest_path.read_text())
    raw_entries = data.get("checkpoints", data) if isinstance(data, dict) else data
    trusted: dict[str, str] = {}

    if isinstance(raw_entries, dict):
        for checkpoint_key, entry in raw_entries.items():
            sha256 = entry.get("sha256") if isinstance(entry, dict) else entry
            trusted[str(checkpoint_key)] = _normalize_sha256(
                sha256,
                source=f"{manifest_path}:{checkpoint_key}",
            )
    elif isinstance(raw_entries, list):
        for idx, entry in enumerate(raw_entries):
            if not isinstance(entry, dict) or "path" not in entry:
                raise ValueError(
                    f"Invalid checkpoint trust manifest entry at index {idx}"
                )
            trusted[str(entry["path"])] = _normalize_sha256(
                entry.get("sha256"),
                source=f"{manifest_path}:checkpoints[{idx}]",
            )
    else:
        raise ValueError("Checkpoint trust manifest must be a mapping or list")

    if not trusted:
        raise ValueError("Checkpoint trust manifest must include at least one entry")
    return trusted


def _checkpoint_trust_manifest_keys(path: str | Path) -> tuple[str, ...]:
    raw_path = Path(path)
    checkpoint_path = checkpoint_file_path(raw_path)
    keys = [
        str(raw_path),
        str(checkpoint_path),
        checkpoint_path.name,
    ]
    try:
        keys.append(str(checkpoint_path.resolve()))
    except OSError:
        pass
    if checkpoint_path.suffix == ".zip":
        keys.append(checkpoint_path.with_suffix("").name)
    return tuple(dict.fromkeys(keys))


def expected_checkpoint_sha256(
    path: str | Path,
    trusted_checkpoint_manifest: dict[str, str] | None,
) -> str | None:
    if trusted_checkpoint_manifest is None:
        return None
    for key in _checkpoint_trust_manifest_keys(path):
        expected = trusted_checkpoint_manifest.get(key)
        if expected is not None:
            return expected
    return None


def checkpoint_trust_manifest(checkpoints: tuple[str | Path, ...]) -> dict:
    """Build an explicit SHA-256 allowlist for checkpoints from a trusted run."""
    entries: dict[str, dict[str, str]] = {}

    def add_entry(key: str, sha256: str, checkpoint_path: Path) -> None:
        existing = entries.get(key)
        if existing is not None and existing["sha256"] != sha256:
            raise ValueError(
                "Checkpoint trust manifest key collision for "
                f"{key}: {checkpoint_path}"
            )
        entries[key] = {"sha256": sha256}

    for checkpoint in checkpoints:
        raw_checkpoint = Path(checkpoint)
        checkpoint_path = checkpoint_file_path(checkpoint)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

        sha256 = checkpoint_file_sha256(checkpoint_path)
        add_entry(str(raw_checkpoint), sha256, checkpoint_path)
        add_entry(str(checkpoint_path), sha256, checkpoint_path)
        add_entry(checkpoint_path.name, sha256, checkpoint_path)
        try:
            add_entry(str(checkpoint_path.resolve()), sha256, checkpoint_path)
        except OSError:
            pass
        if checkpoint_path.suffix == ".zip":
            add_entry(checkpoint_path.with_suffix("").name, sha256, checkpoint_path)

    if not entries:
        raise ValueError("Checkpoint trust manifest requires at least one checkpoint")
    return {
        "artifact": artifact_metadata("checkpoint_trust_manifest"),
        "checkpoints": entries,
    }


def write_checkpoint_trust_manifest(
    checkpoints: tuple[str | Path, ...],
    path: str | Path,
) -> Path:
    manifest = checkpoint_trust_manifest(checkpoints)
    manifest_path = Path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def verify_checkpoint_trust(
    path: str | Path,
    *,
    trusted_checkpoint_manifest: dict[str, str] | None = None,
    allow_unverified: bool = False,
) -> dict:
    checkpoint_path = checkpoint_file_path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    expected = expected_checkpoint_sha256(path, trusted_checkpoint_manifest)
    if expected is not None:
        expected = _normalize_sha256(expected, source="trusted_checkpoint_manifest")
        actual = checkpoint_file_sha256(checkpoint_path)
        if not hmac.compare_digest(actual, expected):
            raise ValueError(
                "Checkpoint SHA-256 mismatch for "
                f"{checkpoint_path}: expected {expected}, got {actual}"
            )
        return {
            "path": str(checkpoint_path),
            "verified": True,
            "verification_source": "trusted_manifest",
            "sha256": actual,
        }

    if trusted_checkpoint_manifest is not None:
        raise ValueError(f"Checkpoint is not listed in trust manifest: {checkpoint_path}")

    if allow_unverified:
        return {
            "path": str(checkpoint_path),
            "verified": False,
            "verification_source": "explicit_unverified_override",
        }

    metadata = read_checkpoint_metadata(checkpoint_path)
    integrity = checkpoint_metadata_integrity(checkpoint_path, metadata)
    raise ValueError(
        "Refusing to load checkpoint before trust verification. "
        "Checkpoint sidecar metadata only proves file integrity and is not a "
        "trust source for deserialization. Provide --trusted-checkpoint-manifest "
        "with an expected SHA-256, or pass "
        "--allow-unverified-checkpoints only for known-local legacy checkpoints. "
        f"Verification failure: {integrity.get('reason')}"
    )


def load_trusted_ppo_checkpoint(
    path: str | Path,
    *,
    trusted_checkpoint_manifest: dict[str, str] | None = None,
    allow_unverified: bool = False,
):
    """Load a trusted SB3 checkpoint.

    Stable-Baselines3 checkpoints may deserialize cloudpickle payloads, so callers
    must verify provenance before loading checkpoints from outside this project.
    """
    from stable_baselines3 import PPO

    checkpoint_path = checkpoint_file_path(path)
    verify_checkpoint_trust(
        path,
        trusted_checkpoint_manifest=trusted_checkpoint_manifest,
        allow_unverified=allow_unverified,
    )
    return PPO.load(str(checkpoint_path))


def discover_checkpoints(checkpoint_dir: str | Path) -> tuple[str, ...]:
    root = Path(checkpoint_dir)
    if not root.exists():
        return ()

    candidates = []
    for path in root.iterdir():
        if not path.is_file() or path.name.endswith(".meta.json"):
            continue
        if path.suffix == ".zip" or (path.suffix == "" and path.name.startswith("ppo_")):
            metadata = read_checkpoint_metadata(path) or {}
            candidates.append(
                (
                    int(metadata.get("num_timesteps", -1)),
                    path.name,
                    str(path),
                )
            )

    return tuple(item[2] for item in sorted(candidates))


def parse_csv_tuple(value: str, flag_name: str) -> tuple[str, ...]:
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    if not items:
        raise ValueError(f"{flag_name} must include at least one value")
    return items


def parse_builtin_opponents(value: str | None) -> tuple[str, ...]:
    if value is None:
        return BUILTIN_POLICY_NAMES

    opponents = parse_csv_tuple(value, "--suite-opponents")
    unknown = [name for name in opponents if name not in BUILTIN_POLICY_NAMES]
    if unknown:
        raise ValueError(f"Unknown opponent names: {', '.join(unknown)}")
    return opponents


def parse_suite_maps(value: str | None, cfg: Config) -> tuple[str, ...]:
    if value is None:
        return cfg.arena.map_choices if cfg.arena.randomize_maps else (cfg.arena.map_name,)

    maps = parse_csv_tuple(value, "--suite-maps")
    unknown = [name for name in maps if name not in PLATFORM_LAYOUTS]
    if unknown:
        raise ValueError(f"Unknown map names: {', '.join(unknown)}")
    return maps


def parse_rank_checkpoints(value: str | None) -> tuple[str, ...] | None:
    if value is None:
        return None
    return parse_csv_tuple(value, "--rank-checkpoints")


def validate_run_id(run_id: str) -> str:
    if not _RUN_ID_RE.fullmatch(run_id):
        raise ValueError(
            "run_id must start with a letter or number and contain only "
            "letters, numbers, dots, underscores, or hyphens"
        )
    return run_id


class SelfPlayCallback(BaseCallback):
    """Snapshots weights into the opponent pool at regular intervals."""

    def __init__(
        self,
        wrapper: SelfPlayWrapper,
        opponent_pool: OpponentPool,
        cfg: Config,
        snapshot_interval: int = 50,
        checkpoint_dir: str = "checkpoints",
        curriculum_name: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.wrapper = wrapper
        self.opponent_pool = opponent_pool
        self.cfg = cfg
        self.snapshot_interval = snapshot_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.curriculum_name = curriculum_name
        self._curriculum_stage_name: str | None = None
        self._rollout_count = 0
        self._milestones = {100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
                            50_000_000, 100_000_000}
        self._milestones_hit: set[int] = set()

    def _on_step(self) -> bool:
        self._apply_curriculum()

        # Check for milestone checkpoints based on total timesteps
        steps = self.num_timesteps
        for m in self._milestones:
            if m not in self._milestones_hit and steps >= m:
                self._milestones_hit.add(m)
                label = f"{m // 1_000_000}M" if m >= 1_000_000 else f"{m // 1_000}K"
                path = self.checkpoint_dir / f"ppo_{label}"
                self.model.save(str(path))
                write_checkpoint_metadata(
                    path,
                    self.cfg,
                    steps,
                    opponent_pool_stats=self.opponent_pool.stats(),
                )
                if self.verbose:
                    print(f"[Milestone] {label} steps reached, saved to {path}")
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.snapshot_interval == 0:
            # Snapshot current weights into the pool
            state_dict = self.model.policy.state_dict()
            self.opponent_pool.add(state_dict)

            # Save checkpoint
            ckpt_path = self.checkpoint_dir / f"ppo_snap_{self._rollout_count}"
            self.model.save(str(ckpt_path))
            write_checkpoint_metadata(
                ckpt_path,
                self.cfg,
                self.num_timesteps,
                opponent_pool_stats=self.opponent_pool.stats(),
            )

            if self.wrapper.opponent_policy is None:
                self.wrapper.opponent_policy = clone_policy_for_opponent(self.model)

            if self.verbose:
                pool_stats = self.opponent_pool.stats()
                print(
                    f"[Snapshot] rollout={self._rollout_count}  "
                    f"pool_size={len(self.opponent_pool)}  "
                    f"latest_samples={pool_stats['latest_samples']}  "
                    f"historical_samples={pool_stats['historical_samples']}  "
                    f"last_snapshot_id={pool_stats['last_sample_id']}  "
                    f"saved={ckpt_path}"
                )
        self._record_self_play_stats()

    def _record_self_play_stats(self) -> None:
        pool_stats = self.opponent_pool.stats()
        self.logger.record("self_play/opponent_pool_size", pool_stats["size"])
        self.logger.record(
            "self_play/latest_opponent_samples",
            pool_stats["latest_samples"],
        )
        self.logger.record(
            "self_play/historical_opponent_samples",
            pool_stats["historical_samples"],
        )
        self.logger.record(
            "self_play/historical_sample_rate",
            pool_stats["historical_sample_rate"],
        )
        self.logger.record(
            "self_play/latest_opponent_snapshot_id",
            (
                pool_stats["latest_snapshot_id"]
                if pool_stats["latest_snapshot_id"] is not None
                else -1
            ),
        )
        self.logger.record(
            "self_play/last_opponent_snapshot_id",
            (
                pool_stats["last_sample_id"]
                if pool_stats["last_sample_id"] is not None
                else -1
            ),
        )
        self.logger.record(
            "self_play/last_sample_was_historical",
            1.0 if pool_stats["last_sample_kind"] == "historical" else 0.0,
        )

    def _apply_curriculum(self) -> None:
        if self.curriculum_name is None:
            return

        stage = curriculum_stage_for_step(self.curriculum_name, self.num_timesteps)
        if stage.name == self._curriculum_stage_name:
            return

        self.wrapper.set_map_pool(stage.map_choices)
        self.wrapper.set_reward_config(reward_config_for_preset(stage.reward_preset))
        self._curriculum_stage_name = stage.name
        if self.verbose:
            maps = ",".join(stage.map_choices)
            print(
                f"[Curriculum] step={self.num_timesteps} "
                f"stage={stage.name} maps={maps} reward={stage.reward_preset}"
            )


def build_training_wrapper(
    cfg: Config,
    replay_dir: str,
) -> tuple[SelfPlayWrapper, OpponentPool]:
    pool = OpponentPool(
        max_size=cfg.training.opponent_pool_size,
        seed=cfg.training.opponent_pool_seed,
    )
    replay_logger = ReplayLogger(
        replay_dir=replay_dir,
        save_every_n=cfg.training.replay_save_interval,
    )
    wrapper = SelfPlayWrapper(
        config=cfg,
        opponent_pool=pool,
        replay_logger=replay_logger,
    )
    return wrapper, pool


def run_train(cfg: Config, checkpoint_dir: str, replay_dir: str) -> None:
    """Headless PPO self-play training."""
    from stable_baselines3 import PPO

    wrapper, pool = build_training_wrapper(cfg, replay_dir)

    policy_kwargs = {
        "features_extractor_class": ArenaFeaturesExtractor,
        "features_extractor_kwargs": {"features_dim": 128},
    }

    model = PPO(
        "MultiInputPolicy",
        wrapper,
        learning_rate=cfg.training.learning_rate,
        n_steps=cfg.training.batch_size,
        batch_size=cfg.training.mini_batch_size,
        n_epochs=cfg.training.n_epochs,
        gamma=cfg.training.gamma,
        gae_lambda=cfg.training.gae_lambda,
        clip_range=cfg.training.clip_range,
        policy_kwargs=policy_kwargs,
        tensorboard_log="./tb_logs",
        verbose=1,
    )

    # Seed the pool with the initial (random) weights
    pool.add(model.policy.state_dict())
    wrapper.opponent_policy = clone_policy_for_opponent(model)
    if cfg.training.curriculum_name is not None:
        stage = curriculum_stage_for_step(cfg.training.curriculum_name, 0)
        wrapper.set_map_pool(stage.map_choices)
        wrapper.set_reward_config(reward_config_for_preset(stage.reward_preset))

    callback = SelfPlayCallback(
        wrapper=wrapper,
        opponent_pool=pool,
        cfg=cfg,
        snapshot_interval=cfg.training.snapshot_interval,
        checkpoint_dir=checkpoint_dir,
        curriculum_name=cfg.training.curriculum_name,
        verbose=1,
    )

    model.learn(total_timesteps=cfg.training.total_timesteps, callback=callback)

    final_path = str(Path(checkpoint_dir) / "ppo_final")
    model.save(final_path)
    write_checkpoint_metadata(
        final_path,
        cfg,
        model.num_timesteps,
        opponent_pool_stats=pool.stats(),
    )
    trust_manifest_path = write_checkpoint_trust_manifest(
        discover_checkpoints(checkpoint_dir),
        Path(checkpoint_dir) / "checkpoint-trust-manifest.json",
    )
    print(f"Training complete. Final model saved to {final_path}")
    print(f"Checkpoint trust manifest saved to {trust_manifest_path}")


def run_watch(
    cfg: Config,
    checkpoint: str | None,
    num_rounds: int = 0,
    *,
    trusted_checkpoint_manifest: dict[str, str] | None = None,
    allow_unverified_checkpoints: bool = False,
) -> None:
    """Load a checkpoint and play games in ASCII with score tracking.

    Args:
        num_rounds: number of rounds to play. 0 = infinite (Ctrl+C to stop).
    """
    import numpy as np

    model = None
    checkpoint_label = checkpoint
    if checkpoint:
        checkpoint_path = checkpoint_file_path(checkpoint)
        if not checkpoint_path.exists():
            print(f"Checkpoint not found: {checkpoint}")
            sys.exit(1)
        model = load_trusted_ppo_checkpoint(
            checkpoint,
            trusted_checkpoint_manifest=trusted_checkpoint_manifest,
            allow_unverified=allow_unverified_checkpoints,
        )
        checkpoint_label = str(checkpoint_path)

    score = [0, 0]  # [agent_0 wins, agent_1 wins]
    draws = 0
    round_num = 0

    try:
        while num_rounds == 0 or round_num < num_rounds:
            round_num += 1
            env = ArenaFightersEnv(config=cfg, render_mode="ansi")
            obs_dict, _ = env.reset()

            # Round start splash
            clear_terminal()
            print(f"\033[1;36m{'ARENA FIGHTERS':^44}\033[0m")
            print(f"\033[90m{'=' * 44}\033[0m")
            print(f"\n  \033[1mRound {round_num}\033[0m")
            print(f"  Score: \033[1;33m@\033[0m {score[0]}  -  {score[1]} \033[1;35mX\033[0m  (draws: {draws})")
            if model:
                print(f"  \033[90mCheckpoint: {checkpoint_label}\033[0m")
            else:
                print(f"  \033[90mRandom agents (no checkpoint)\033[0m")
            print(f"\n  \033[90mStarting in 2s...\033[0m")
            time.sleep(2)

            winner = None
            while env.agents:
                actions = {}
                for agent_name in env.agents:
                    agent_obs = obs_dict[agent_name]
                    if model is not None:
                        if agent_name == "agent_1":
                            agent_obs = mirror_obs(agent_obs)
                        action, _ = model.predict(agent_obs, deterministic=False)
                        actions[agent_name] = int(action)
                    else:
                        actions[agent_name] = np.random.randint(0, NUM_ACTIONS)

                obs_dict, rewards, terminations, truncations, infos = env.step(actions)

                clear_terminal()
                frame = env._render_ansi(score=tuple(score))
                print(frame)
                time.sleep(0.08)

                # Determine winner from rewards
                if any(terminations.values()) or any(truncations.values()):
                    r0 = rewards.get("agent_0", 0)
                    r1 = rewards.get("agent_1", 0)
                    if r0 > r1:
                        winner = "agent_0"
                    elif r1 > r0:
                        winner = "agent_1"

            # Update score
            if winner == "agent_0":
                score[0] += 1
            elif winner == "agent_1":
                score[1] += 1
            else:
                draws += 1

            # Round result splash
            clear_terminal()
            print(f"\033[1;36m{'ARENA FIGHTERS':^44}\033[0m")
            print(f"\033[90m{'=' * 44}\033[0m")
            print(f"\n  \033[1mRound {round_num} Result\033[0m\n")
            if winner == "agent_0":
                print(f"  \033[1;33m>>> @ WINS! <<<\033[0m")
            elif winner == "agent_1":
                print(f"  \033[1;35m>>> X WINS! <<<\033[0m")
            else:
                print(f"  \033[90m>>> DRAW <<<\033[0m")
            print(f"\n  Score: \033[1;33m@\033[0m {score[0]}  -  {score[1]} \033[1;35mX\033[0m  (draws: {draws})")
            print(f"  Win rates: \033[33m@\033[0m {score[0]*100/round_num:.0f}%  \033[35mX\033[0m {score[1]*100/round_num:.0f}%")
            print(f"\n  \033[90mNext round in 3s... (Ctrl+C to quit)\033[0m")
            time.sleep(3)
            env.close()

    except KeyboardInterrupt:
        pass

    # Final summary
    print(f"\n\033[1;36m{'FINAL RESULTS':^44}\033[0m")
    print(f"\033[90m{'=' * 44}\033[0m")
    print(f"  Rounds played: {round_num}")
    print(f"  Score: \033[1;33m@\033[0m {score[0]}  -  {score[1]} \033[1;35mX\033[0m  (draws: {draws})")
    if round_num > 0:
        print(f"  Win rates: \033[33m@\033[0m {score[0]*100/round_num:.0f}%  \033[35mX\033[0m {score[1]*100/round_num:.0f}%")
    print()


def run_replay(cfg: Config, episode_path: str) -> None:
    """Play back a JSON replay file frame by frame in ASCII."""
    path = Path(episode_path)
    if not path.exists():
        print(f"Replay file not found: {episode_path}")
        sys.exit(1)

    data = load_replay(path)
    frames = data["frames"]
    winner = data.get("winner", "unknown")
    length = data.get("length", len(frames))
    map_name = data.get("map_name", "classic")
    event_totals = data.get("event_totals", {})

    print(
        f"Replay: {path.name}  Winner: {winner}  "
        f"Length: {length} ticks  Map: {map_name}"
    )
    if event_totals:
        print(f"Event totals: {json.dumps(event_totals, sort_keys=True)}")
    time.sleep(1)

    h = cfg.arena.height
    w = cfg.arena.width

    for frame in frames:
        # Build ASCII display from frame state
        display = [["." for _ in range(w)] for _ in range(h)]

        # Draw platforms
        from arena_fighters.config import PLATFORM_LAYOUT, PLATFORM_LAYOUTS

        map_name = frame.get("map_name", "classic")
        layout = PLATFORM_LAYOUTS.get(map_name, PLATFORM_LAYOUT)
        for x_start, x_end, y in layout:
            for x in range(x_start, x_end + 1):
                if 0 <= y < h and 0 <= x < w:
                    display[y][x] = "="

        # Draw bullets
        for b in frame.get("bullets", []):
            bx, by = int(round(b["x"])), int(round(b["y"]))
            if 0 <= bx < w and 0 <= by < h:
                display[by][bx] = "*"

        # Draw agents
        agents = frame.get("agents", {})
        if "agent_0" in agents:
            a0 = agents["agent_0"]
            if 0 <= a0["y"] < h and 0 <= a0["x"] < w:
                display[a0["y"]][a0["x"]] = "@"
        if "agent_1" in agents:
            a1 = agents["agent_1"]
            if 0 <= a1["y"] < h and 0 <= a1["x"] < w:
                display[a1["y"]][a1["x"]] = "X"

        # Header
        tick = frame.get("tick", "?")
        hp0 = agents.get("agent_0", {}).get("hp", "?")
        hp1 = agents.get("agent_1", {}).get("hp", "?")
        header = f"Tick {tick}/{cfg.arena.max_ticks}  HP: @={hp0}  X={hp1}"

        clear_terminal()
        print(header)
        for row in display:
            print("".join(row))
        time.sleep(0.1)

    print(f"\nReplay complete. Winner: {winner}")


def run_analyze_replay(
    episode_path: str,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    path = Path(episode_path)
    if not path.exists():
        print(f"Replay file not found: {episode_path}")
        sys.exit(1)

    data = load_replay(path)
    analysis = analyze_replay(data)
    analysis["artifact"] = artifact_metadata("replay_analysis")
    print(json.dumps(analysis, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or f"replay-analysis-{path.stem}"
        saved_path = write_eval_summary(analysis, output_dir, label=label)
        print(f"Saved replay analysis to {saved_path}")


REPLAY_SAMPLE_BUCKETS = (
    "agent_0_win",
    "agent_1_win",
    "draw",
    "combat",
    "no_damage",
    "no_attacks",
    "idle_agent_0",
    "dominant_action_agent_0",
)
REPLAY_IDLE_BUCKET_THRESHOLD = 0.75
REPLAY_DOMINANT_ACTION_BUCKET_THRESHOLD = 0.95


def replay_analysis_buckets(analysis: dict) -> list[str]:
    buckets = []
    winner = analysis.get("winner")
    if winner == "agent_0":
        buckets.append("agent_0_win")
    elif winner == "agent_1":
        buckets.append("agent_1_win")
    elif winner == "draw":
        buckets.append("draw")

    flags = analysis.get("flags", {})
    if flags.get("no_damage") is True:
        buckets.append("no_damage")
    elif flags.get("no_damage") is False:
        buckets.append("combat")
        map_name = analysis.get("map_name")
        if map_name:
            buckets.append(f"combat_map:{map_name}")
    if flags.get("no_attacks") is True:
        buckets.append("no_attacks")
    behavior = analysis.get("behavior", {})
    idle_rate = behavior.get("avg_idle_rate", {}).get("agent_0")
    if idle_rate is not None and idle_rate >= REPLAY_IDLE_BUCKET_THRESHOLD:
        buckets.append("idle_agent_0")
    dominant_action_rate = behavior.get("avg_dominant_action_rate", {}).get("agent_0")
    if (
        dominant_action_rate is not None
        and dominant_action_rate >= REPLAY_DOMINANT_ACTION_BUCKET_THRESHOLD
    ):
        buckets.append("dominant_action_agent_0")
    return buckets


def _relative_to_root(path: Path, root: Path) -> str | None:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return None


def build_replay_analysis_batch(
    replay_dir: str | Path,
    samples_per_bucket: int = 1,
) -> dict:
    root = Path(replay_dir)
    bucket_counts = {bucket: 0 for bucket in REPLAY_SAMPLE_BUCKETS}
    selected_by_path: dict[str, dict] = {}
    skipped = []
    scanned = 0

    for path in sorted(root.glob("*.json")):
        if not path.is_file():
            continue
        relative_path = _relative_to_root(path, root)
        try:
            analysis = analyze_replay(load_replay(path))
        except Exception as exc:
            skipped.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        scanned += 1
        buckets = replay_analysis_buckets(analysis)
        needed_buckets = [
            bucket
            for bucket in buckets
            if bucket_counts.get(bucket, 0) < samples_per_bucket
        ]
        if not needed_buckets:
            continue

        analysis["artifact"] = artifact_metadata("replay_analysis")
        entry = selected_by_path.setdefault(
            str(path),
            {
                "path": str(path),
                "relative_path": relative_path,
                "buckets": buckets,
                "selected_for": [],
                "analysis": analysis,
            },
        )
        for bucket in needed_buckets:
            if bucket not in entry["selected_for"]:
                entry["selected_for"].append(bucket)
                bucket_counts.setdefault(bucket, 0)
                bucket_counts[bucket] += 1

    return {
        "artifact": artifact_metadata("replay_analysis_batch"),
        "batch_config": {
            "replay_dir": str(root),
            "samples_per_bucket": samples_per_bucket,
            "bucket_names": list(REPLAY_SAMPLE_BUCKETS),
            "dynamic_bucket_prefixes": ["combat_map:"],
        },
        "scanned_replays": scanned,
        "selected_count": len(selected_by_path),
        "bucket_counts": bucket_counts,
        "selected": list(selected_by_path.values()),
        "skipped_replays": skipped,
    }


def run_analyze_replay_dir(
    replay_dir: str,
    samples_per_bucket: int,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    if samples_per_bucket < 1:
        raise ValueError("--replay-samples-per-bucket must be at least 1")

    batch = build_replay_analysis_batch(
        replay_dir,
        samples_per_bucket=samples_per_bucket,
    )
    if output_dir is not None:
        base_label = output_label or "replay-sample"
        for item in batch["selected"]:
            replay_stem = Path(item["path"]).stem
            label = f"{base_label}-{replay_stem}"
            saved_path = write_eval_summary(item["analysis"], output_dir, label=label)
            item["analysis_artifact_path"] = str(saved_path)
        batch_path = write_eval_summary(
            batch,
            output_dir,
            label=f"{base_label}-batch",
        )
        batch["batch_artifact_path"] = str(batch_path)

    print(json.dumps(batch, indent=2, sort_keys=True))
    if output_dir is not None:
        print(f"Saved replay analysis batch to {batch['batch_artifact_path']}")


def run_eval(
    cfg: Config,
    checkpoint: str | None,
    opponent: str,
    num_rounds: int,
    seed: int | None,
    deterministic: bool,
    reward_preset: str,
    output_dir: str | None,
    output_label: str | None,
    agent_policy: str = "random",
    trusted_checkpoint_manifest: dict[str, str] | None = None,
    allow_unverified_checkpoints: bool = False,
) -> None:
    """Evaluate a checkpoint or built-in policy against a built-in baseline."""
    if checkpoint:
        path = checkpoint_file_path(checkpoint)
        if not path.exists():
            print(f"Checkpoint not found: {checkpoint}")
            sys.exit(1)
        model = load_trusted_ppo_checkpoint(
            checkpoint,
            trusted_checkpoint_manifest=trusted_checkpoint_manifest,
            allow_unverified=allow_unverified_checkpoints,
        )
        agent0_policy = ModelPolicy(model=model, deterministic=deterministic)
        agent0_label = str(path)
    else:
        agent0_policy = make_builtin_policy(agent_policy, seed=seed)
        agent0_label = agent_policy

    opponent_seed = None if seed is None else seed + 100_000
    agent1_policy = make_builtin_policy(opponent, seed=opponent_seed)
    episodes = num_rounds if num_rounds > 0 else 20
    summary = evaluate_matchup(
        cfg=cfg,
        agent0_policy=agent0_policy,
        agent1_policy=agent1_policy,
        episodes=episodes,
        seed=seed,
    )
    summary["agent_0_policy"] = agent0_label
    summary["agent_1_policy"] = opponent
    summary["artifact"] = artifact_metadata("eval")
    summary["eval_config"] = {
        "checkpoint": checkpoint,
        "checkpoint_metadata": read_checkpoint_metadata(checkpoint),
        "agent_policy": agent_policy if checkpoint is None else "checkpoint",
        "opponent": opponent,
        "episodes": episodes,
        "seed": seed,
        "deterministic": deterministic,
        "map_name": cfg.arena.map_name,
        "randomize_maps": cfg.arena.randomize_maps,
        "map_choices": list(cfg.arena.map_choices),
        "reward_preset": reward_preset,
        "curriculum": curriculum_metadata(cfg, 0),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or f"{opponent}_{episodes}ep"
        path = write_eval_summary(summary, output_dir, label=label)
        print(f"Saved eval summary to {path}")


def run_compare(
    before_path: str,
    after_path: str,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    before = load_eval_summary(before_path)
    after = load_eval_summary(after_path)
    validate_artifact(before, "eval")
    validate_artifact(after, "eval")
    comparison = compare_eval_summaries(before, after)
    comparison["before_path"] = before_path
    comparison["after_path"] = after_path
    print(json.dumps(comparison, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or "comparison"
        path = write_eval_summary(comparison, output_dir, label=label)
        print(f"Saved comparison summary to {path}")


def run_gate(
    before_path: str,
    after_path: str,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    before = load_eval_summary(before_path)
    after = load_eval_summary(after_path)
    validate_artifact(before, "eval")
    validate_artifact(after, "eval")
    comparison = compare_eval_summaries(before, after)
    gate = gate_eval_comparison(comparison)
    gate["comparison"] = comparison
    gate["before_path"] = before_path
    gate["after_path"] = after_path
    print(json.dumps(gate, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or "gate"
        path = write_eval_summary(gate, output_dir, label=label)
        print(f"Saved gate summary to {path}")
    if not gate["passed"]:
        sys.exit(1)


def run_rank_gate(
    rank_summary_path: str,
    min_score: float,
    min_win_rate: float,
    max_draw_rate: float,
    max_no_damage_rate: float,
    max_low_engagement_rate: float,
    min_head_to_head_elo: float | None = None,
    min_head_to_head_score: float | None = None,
    min_map_score: float | None = None,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    summary = load_eval_summary(rank_summary_path)
    validate_artifact(summary, "rank")
    gate = gate_rank_summary(
        summary,
        min_score=min_score,
        min_win_rate=min_win_rate,
        max_draw_rate=max_draw_rate,
        max_no_damage_rate=max_no_damage_rate,
        max_low_engagement_rate=max_low_engagement_rate,
        min_map_score=min_map_score,
        min_head_to_head_elo=min_head_to_head_elo,
        min_head_to_head_score=min_head_to_head_score,
    )
    gate["rank_summary_path"] = rank_summary_path
    print(json.dumps(gate, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or "rank-gate"
        path = write_eval_summary(gate, output_dir, label=label)
        print(f"Saved rank gate summary to {path}")
    if not gate["passed"]:
        sys.exit(1)


def run_suite(
    cfg: Config,
    checkpoint: str | None,
    agent_policy: str,
    opponents: tuple[str, ...],
    maps: tuple[str, ...],
    num_rounds: int,
    seed: int | None,
    deterministic: bool,
    reward_preset: str,
    output_dir: str | None,
    output_label: str | None,
    trusted_checkpoint_manifest: dict[str, str] | None = None,
    allow_unverified_checkpoints: bool = False,
) -> None:
    agent0_label = agent_policy
    model = None
    if checkpoint:
        path = checkpoint_file_path(checkpoint)
        if not path.exists():
            print(f"Checkpoint not found: {checkpoint}")
            sys.exit(1)
        model = load_trusted_ppo_checkpoint(
            checkpoint,
            trusted_checkpoint_manifest=trusted_checkpoint_manifest,
            allow_unverified=allow_unverified_checkpoints,
        )
        agent0_label = str(path)

    def agent0_policy_factory(policy_seed: int | None):
        if model is not None:
            return ModelPolicy(model=model, deterministic=deterministic)
        return make_builtin_policy(agent_policy, seed=policy_seed)

    episodes = num_rounds if num_rounds > 0 else 5
    suite = evaluate_baseline_suite(
        cfg=cfg,
        agent0_policy_factory=agent0_policy_factory,
        agent0_label=agent0_label,
        opponents=opponents,
        maps=maps,
        episodes=episodes,
        seed=seed,
        reward_preset=reward_preset,
    )
    suite["suite_config"]["checkpoint_metadata"] = (
        read_checkpoint_metadata(path) if checkpoint else None
    )
    suite["suite_config"]["curriculum"] = curriculum_metadata(cfg, 0)
    suite["artifact"] = artifact_metadata("suite")
    print(json.dumps(suite, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or f"suite_{episodes}ep"
        path = write_eval_summary(suite, output_dir, label=label)
        print(f"Saved suite summary to {path}")


def build_rank_summary(
    cfg: Config,
    checkpoints: tuple[str, ...] | None,
    checkpoint_dir: str,
    opponents: tuple[str, ...],
    maps: tuple[str, ...],
    num_rounds: int,
    seed: int | None,
    deterministic: bool,
    reward_preset: str,
    draw_weight: float,
    no_damage_penalty: float,
    low_engagement_penalty: float,
    include_head_to_head: bool,
    initial_elo: float,
    elo_k_factor: float,
    trusted_checkpoint_manifest: dict[str, str] | None = None,
    allow_unverified_checkpoints: bool = False,
) -> dict:
    checkpoint_paths = checkpoints or discover_checkpoints(checkpoint_dir)
    if not checkpoint_paths:
        print(f"No checkpoints found in {checkpoint_dir}")
        sys.exit(1)

    episodes = num_rounds if num_rounds > 0 else 5
    entries = []
    loaded_models = {}
    used_labels = set()
    for checkpoint_idx, checkpoint in enumerate(checkpoint_paths):
        path = checkpoint_file_path(checkpoint)
        if not path.exists():
            print(f"Checkpoint not found: {checkpoint}")
            sys.exit(1)

        model = load_trusted_ppo_checkpoint(
            checkpoint,
            trusted_checkpoint_manifest=trusted_checkpoint_manifest,
            allow_unverified=allow_unverified_checkpoints,
        )
        label = path.stem if path.suffix == ".zip" else path.name
        if label in used_labels:
            label = f"{label}_{checkpoint_idx}"
        used_labels.add(label)
        loaded_models[label] = model

        def agent0_policy_factory(policy_seed: int | None, model=model):
            return ModelPolicy(model=model, deterministic=deterministic)

        checkpoint_seed = None if seed is None else seed + checkpoint_idx * 1_000_000
        suite = evaluate_baseline_suite(
            cfg=cfg,
            agent0_policy_factory=agent0_policy_factory,
            agent0_label=str(path),
            opponents=opponents,
            maps=maps,
            episodes=episodes,
            seed=checkpoint_seed,
            reward_preset=reward_preset,
        )
        checkpoint_metadata = read_checkpoint_metadata(path)
        suite["suite_config"]["checkpoint_metadata"] = checkpoint_metadata
        suite["suite_config"]["curriculum"] = curriculum_metadata(cfg, 0)
        entries.append(
            {
                "label": label,
                "checkpoint": str(path),
                "checkpoint_metadata": checkpoint_metadata,
                "suite": suite,
            }
        )

    result = {
        "artifact": artifact_metadata("rank"),
        "rank_config": {
            "checkpoints": list(checkpoint_paths),
            "checkpoint_dir": checkpoint_dir,
            "opponents": list(opponents),
            "maps": list(maps),
            "episodes_per_matchup": episodes,
            "seed": seed,
            "deterministic": deterministic,
            "reward_preset": reward_preset,
            "draw_weight": draw_weight,
            "no_damage_penalty": no_damage_penalty,
            "low_engagement_penalty": low_engagement_penalty,
            "include_head_to_head": include_head_to_head,
            "initial_elo": initial_elo,
            "elo_k_factor": elo_k_factor,
            "curriculum": curriculum_metadata(cfg, 0),
        },
        **rank_baseline_suites(
            entries,
            draw_weight=draw_weight,
            no_damage_penalty=no_damage_penalty,
            low_engagement_penalty=low_engagement_penalty,
        ),
        "suites": entries,
    }
    if include_head_to_head and len(entries) >= 2:
        policy_factories = {
            label: (
                lambda policy_seed, model=model: ModelPolicy(
                    model=model,
                    deterministic=deterministic,
                )
            )
            for label, model in loaded_models.items()
        }
        result["head_to_head"] = evaluate_pairwise_suite(
            cfg=cfg,
            policy_factories=policy_factories,
            maps=maps,
            episodes=episodes,
            seed=seed,
            initial_elo=initial_elo,
            elo_k_factor=elo_k_factor,
        )
    elif include_head_to_head:
        result["head_to_head"] = {
            "skipped": "requires_at_least_two_checkpoints",
            "checkpoint_count": len(entries),
        }
    return result


def run_rank(
    cfg: Config,
    checkpoints: tuple[str, ...] | None,
    checkpoint_dir: str,
    opponents: tuple[str, ...],
    maps: tuple[str, ...],
    num_rounds: int,
    seed: int | None,
    deterministic: bool,
    reward_preset: str,
    draw_weight: float,
    no_damage_penalty: float,
    low_engagement_penalty: float,
    include_head_to_head: bool,
    initial_elo: float,
    elo_k_factor: float,
    output_dir: str | None,
    output_label: str | None,
    trusted_checkpoint_manifest: dict[str, str] | None = None,
    allow_unverified_checkpoints: bool = False,
) -> dict:
    result = build_rank_summary(
        cfg=cfg,
        checkpoints=checkpoints,
        checkpoint_dir=checkpoint_dir,
        opponents=opponents,
        maps=maps,
        num_rounds=num_rounds,
        seed=seed,
        deterministic=deterministic,
        reward_preset=reward_preset,
        draw_weight=draw_weight,
        no_damage_penalty=no_damage_penalty,
        low_engagement_penalty=low_engagement_penalty,
        include_head_to_head=include_head_to_head,
        initial_elo=initial_elo,
        elo_k_factor=elo_k_factor,
        trusted_checkpoint_manifest=trusted_checkpoint_manifest,
        allow_unverified_checkpoints=allow_unverified_checkpoints,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if output_dir is not None:
        checkpoint_paths = result["rank_config"]["checkpoints"]
        episodes = result["rank_config"]["episodes_per_matchup"]
        label = output_label or f"rank_{len(checkpoint_paths)}ckpt_{episodes}ep"
        path = write_eval_summary(result, output_dir, label=label)
        print(f"Saved rank summary to {path}")
    return result


def run_promotion_audit(
    cfg: Config,
    checkpoints: tuple[str, ...] | None,
    checkpoint_dir: str,
    opponents: tuple[str, ...],
    maps: tuple[str, ...],
    num_rounds: int,
    seed: int | None,
    deterministic: bool,
    reward_preset: str,
    draw_weight: float,
    no_damage_penalty: float,
    low_engagement_penalty: float,
    include_head_to_head: bool,
    initial_elo: float,
    elo_k_factor: float,
    min_score: float,
    min_win_rate: float,
    max_draw_rate: float,
    max_no_damage_rate: float,
    max_low_engagement_rate: float,
    min_head_to_head_elo: float | None,
    min_head_to_head_score: float | None,
    output_dir: str | None,
    output_label: str | None,
    include_nested: bool = False,
    trusted_checkpoint_manifest: dict[str, str] | None = None,
    allow_unverified_checkpoints: bool = False,
    min_map_score: float | None = None,
) -> None:
    base_label = output_label or "promotion-audit"
    rank_summary = build_rank_summary(
        cfg=cfg,
        checkpoints=checkpoints,
        checkpoint_dir=checkpoint_dir,
        opponents=opponents,
        maps=maps,
        num_rounds=num_rounds,
        seed=seed,
        deterministic=deterministic,
        reward_preset=reward_preset,
        draw_weight=draw_weight,
        no_damage_penalty=no_damage_penalty,
        low_engagement_penalty=low_engagement_penalty,
        include_head_to_head=include_head_to_head,
        initial_elo=initial_elo,
        elo_k_factor=elo_k_factor,
        trusted_checkpoint_manifest=trusted_checkpoint_manifest,
        allow_unverified_checkpoints=allow_unverified_checkpoints,
    )
    validate_artifact(rank_summary, "rank")

    rank_path = None
    if output_dir is not None:
        rank_path = write_eval_summary(rank_summary, output_dir, f"{base_label}-rank")
        print(f"Saved promotion audit rank summary to {rank_path}")

    gate = gate_rank_summary(
        rank_summary,
        min_score=min_score,
        min_win_rate=min_win_rate,
        max_draw_rate=max_draw_rate,
        max_no_damage_rate=max_no_damage_rate,
        max_low_engagement_rate=max_low_engagement_rate,
        min_map_score=min_map_score,
        min_head_to_head_elo=min_head_to_head_elo,
        min_head_to_head_score=min_head_to_head_score,
    )
    gate["rank_summary_path"] = str(rank_path) if rank_path is not None else None

    gate_path = None
    if output_dir is not None:
        gate_path = write_eval_summary(gate, output_dir, f"{base_label}-rank-gate")
        print(f"Saved promotion audit rank gate summary to {gate_path}")

    rankings = rank_summary.get("rankings", [])
    audit = {
        "artifact": artifact_metadata("promotion_audit"),
        "passed": gate["passed"],
        "audit_config": {
            "include_nested": include_nested,
        },
        "rank_artifact_path": str(rank_path) if rank_path is not None else None,
        "rank_gate_artifact_path": str(gate_path) if gate_path is not None else None,
        "rank_config": rank_summary.get("rank_config", {}),
        "ranking_metric": rank_summary.get("ranking_metric"),
        "ranking_labels": [row.get("label") for row in rankings],
        "rules": gate["rules"],
        "candidate": gate["candidate"],
        "failures": gate["failures"],
    }
    if include_nested:
        audit["rank"] = rank_summary
        audit["rank_gate"] = gate
    print(json.dumps(audit, indent=2, sort_keys=True))

    if output_dir is not None:
        audit_path = write_eval_summary(audit, output_dir, base_label)
        print(f"Saved promotion audit summary to {audit_path}")

    if not gate["passed"]:
        sys.exit(1)


def summarize_promotion_audit(summary: dict) -> dict:
    validate_artifact(summary, "promotion_audit")
    candidate = summary.get("candidate") or {}
    return {
        "artifact": artifact_metadata("audit_summary"),
        "source_artifact": summary["artifact"],
        "passed": bool(summary.get("passed")),
        "candidate": {
            "label": candidate.get("label"),
            "checkpoint": candidate.get("checkpoint"),
            "rank": candidate.get("rank"),
            "score": candidate.get("score"),
            "mean_win_rate_agent_0": candidate.get("mean_win_rate_agent_0"),
            "mean_no_damage_rate": candidate.get("mean_no_damage_rate"),
            "mean_low_engagement_rate": candidate.get("mean_low_engagement_rate"),
        },
        "failures": summary.get("failures", []),
        "rank_artifact_path": summary.get("rank_artifact_path"),
        "rank_gate_artifact_path": summary.get("rank_gate_artifact_path"),
        "ranking_labels": summary.get("ranking_labels", []),
        "rules": summary.get("rules", {}),
    }


def run_audit_summary(
    audit_summary_path: str,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    summary = load_eval_summary(audit_summary_path)
    audit_summary = summarize_promotion_audit(summary)
    audit_summary["audit_summary_path"] = audit_summary_path
    print(json.dumps(audit_summary, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or "audit-summary"
        path = write_eval_summary(audit_summary, output_dir, label=label)
        print(f"Saved audit summary to {path}")


def summarize_artifact_file(path: str | Path, root: str | Path | None = None) -> dict:
    artifact_path = Path(path)
    relative_path = None
    if root is not None:
        try:
            relative_path = str(artifact_path.relative_to(Path(root)))
        except ValueError:
            relative_path = None

    entry = {
        "path": str(artifact_path),
        "relative_path": relative_path,
        "file_size_bytes": artifact_path.stat().st_size,
        "artifact_type": "unknown",
        "schema_version": None,
        "summary": {},
        "links": {},
    }
    if artifact_path.suffix == ".exitcode":
        raw_code = artifact_path.read_text().strip()
        exit_code = None
        try:
            exit_code = int(raw_code)
        except ValueError:
            pass
        entry["artifact_type"] = "exit_code"
        entry["summary"] = {
            "exit_code": exit_code,
            "passed": exit_code == 0 if exit_code is not None else None,
            "raw": raw_code,
        }
        return entry

    if artifact_path.suffix == ".sh":
        text = artifact_path.read_text()
        entry["artifact_type"] = "shell_script"
        entry["summary"] = {
            "line_count": len(text.splitlines()),
            "executable": bool(artifact_path.stat().st_mode & 0o111),
            "starts_with_shebang": text.startswith("#!"),
        }
        return entry

    if artifact_path.suffix == ".out":
        tail_byte_limit = 8192
        size = artifact_path.stat().st_size
        with artifact_path.open("rb") as log_file:
            if size > tail_byte_limit:
                log_file.seek(size - tail_byte_limit)
            raw_tail = log_file.read()
        tail_text = raw_tail.decode("utf-8", errors="replace")
        tail_lines = tail_text.splitlines()
        if size > tail_byte_limit and tail_lines:
            tail_lines = tail_lines[1:]
        entry["artifact_type"] = "command_log"
        entry["summary"] = {
            "tail_byte_limit": tail_byte_limit,
            "tail_truncated": size > tail_byte_limit,
            "tail_lines": [redact_log_line(line) for line in tail_lines[-20:]],
        }
        return entry

    try:
        data = load_eval_summary(artifact_path)
    except Exception as exc:
        entry["error"] = f"{type(exc).__name__}: {exc}"
        return entry

    artifact = data.get("artifact", {})
    if isinstance(artifact, dict):
        entry["artifact_type"] = artifact.get("artifact_type") or "unknown"
        entry["schema_version"] = artifact.get("schema_version")

    artifact_type = entry["artifact_type"]
    entry["summary"] = compact_artifact_summary(data, artifact_type)
    entry["links"] = artifact_links(data, artifact_type)
    return entry


def redact_log_line(line: str) -> str:
    private_key_redacted = _PRIVATE_KEY_ASSIGNMENT_RE.sub(r"\1\2<redacted>", line)
    if private_key_redacted != line:
        return private_key_redacted
    if _PRIVATE_KEY_LINE_RE.search(line):
        return "<redacted private key>"
    redacted = _BEARER_TOKEN_RE.sub(r"\1<redacted>", line)
    redacted = _BASIC_AUTH_TOKEN_RE.sub(r"\1<redacted>", redacted)
    redacted = _HEADER_SECRET_RE.sub(r"\1<redacted>", redacted)
    redacted = _URL_CREDENTIAL_RE.sub(r"://\1:<redacted>@", redacted)
    redacted = _SECRET_ASSIGNMENT_RE.sub(r"\1\2<redacted>", redacted)
    redacted = _SECRET_ARG_RE.sub(r"\1<redacted>", redacted)
    return _JSON_SECRET_RE.sub(r"\1<redacted>\3", redacted)


def _is_indexable_artifact_path(path: Path, root: Path) -> bool:
    if path.is_symlink() or not path.is_file():
        return False
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def compact_artifact_summary(data: dict, artifact_type: str) -> dict:
    if artifact_type == "eval":
        cfg = data.get("eval_config", {})
        return {
            "checkpoint": cfg.get("checkpoint"),
            "opponent": cfg.get("opponent"),
            "episodes": cfg.get("episodes"),
            "seed": cfg.get("seed"),
            "map_name": cfg.get("map_name"),
            "randomize_maps": cfg.get("randomize_maps"),
            "win_rate_agent_0": data.get("win_rate_agent_0"),
            "draw_rate": data.get("draw_rate"),
            "avg_length": data.get("avg_length"),
            "avg_rewards": data.get("avg_rewards"),
        }
    if artifact_type == "suite":
        cfg = data.get("suite_config", {})
        return {
            "agent_0_policy": cfg.get("agent_0_policy"),
            "opponents": cfg.get("opponents"),
            "maps": cfg.get("maps"),
            "episodes_per_matchup": cfg.get("episodes_per_matchup"),
            "overview": data.get("overview", {}),
        }
    if artifact_type == "rank":
        rankings = data.get("rankings", [])
        top_ranking = rankings[0] if rankings else {}
        per_map_scores, invalid_map_scores = ranking_per_map_score_details(
            top_ranking
        )
        worst_map = min(
            per_map_scores,
            key=lambda item: (item["mean_score"], item["map_name"]),
            default=None,
        )
        return {
            "checkpoint_count": len(rankings),
            "top_label": top_ranking.get("label") if top_ranking else None,
            "top_score": top_ranking.get("score") if top_ranking else None,
            "top_map_count": len(per_map_scores),
            "top_worst_map_name": worst_map.get("map_name") if worst_map else None,
            "top_worst_map_score": (
                worst_map.get("mean_score") if worst_map else None
            ),
            "top_invalid_map_score_count": len(invalid_map_scores),
            "ranking_labels": [row.get("label") for row in rankings],
            "rank_config": data.get("rank_config", {}),
        }
    if artifact_type == "comparison":
        return {
            "delta_count": len(data.get("deltas", {})),
            "win_rate_delta": data.get("deltas", {}).get("win_rate_agent_0"),
            "draw_rate_delta": data.get("deltas", {}).get("draw_rate"),
        }
    if artifact_type == "gate":
        return {
            "passed": data.get("passed"),
            "failure_count": len(data.get("failures", [])),
            "failed_metrics": [
                failure.get("metric") for failure in data.get("failures", [])
            ],
        }
    if artifact_type == "rank_gate":
        candidate = data.get("candidate") or {}
        return {
            "passed": data.get("passed"),
            "candidate_label": candidate.get("label"),
            "candidate_score": candidate.get("score"),
            "failure_count": len(data.get("failures", [])),
            "failed_metrics": [
                failure.get("metric") for failure in data.get("failures", [])
            ],
        }
    if artifact_type in {"promotion_audit", "audit_summary"}:
        candidate = data.get("candidate") or {}
        return {
            "passed": data.get("passed"),
            "candidate_label": candidate.get("label"),
            "candidate_score": candidate.get("score"),
            "failure_count": len(data.get("failures", [])),
            "ranking_labels": data.get("ranking_labels", []),
        }
    if artifact_type == "strategy_report":
        issues = data.get("issues", [])
        weaknesses = data.get("weaknesses", [])
        skipped_artifacts = data.get("skipped_artifacts", [])
        return {
            "issue_count": data.get("issue_count", len(issues)),
            "weakness_count": data.get("weakness_count", len(weaknesses)),
            "worst_weakness": weaknesses[0] if weaknesses else None,
            "skipped_artifact_count": len(skipped_artifacts),
            "candidate_issue_count": len(
                [
                    issue
                    for issue in issues
                    if str(issue.get("scope", "")).startswith("candidate:")
                ]
            ),
            "smoke_issue_count": len(
                [
                    issue
                    for issue in issues
                    if str(issue.get("scope", "")).startswith("smoke:")
                ]
            ),
            "issue_metrics": sorted(
                {issue.get("metric") for issue in issues if issue.get("metric")}
            ),
            "scanned_artifacts": data.get("scanned_artifacts"),
        }
    if artifact_type == "replay_analysis":
        return {
            "episode_id": data.get("episode_id"),
            "winner": data.get("winner"),
            "length": data.get("length"),
            "map_name": data.get("map_name"),
            "flags": data.get("flags", {}),
            "totals": data.get("totals", {}),
            "behavior": data.get("behavior", {}),
        }
    if artifact_type == "replay_analysis_batch":
        return {
            "scanned_replays": data.get("scanned_replays"),
            "selected_count": data.get("selected_count"),
            "bucket_counts": data.get("bucket_counts", {}),
            "skipped_count": len(data.get("skipped_replays", [])),
        }
    if artifact_type == "long_run_manifest":
        cfg = data.get("manifest_config", {})
        commands = data.get("commands", [])
        source_control = cfg.get("source_control", {})
        return {
            "run_id": cfg.get("run_id"),
            "timesteps": cfg.get("timesteps"),
            "checkpoint_dir": cfg.get("checkpoint_dir"),
            "eval_dir": cfg.get("eval_dir"),
            "replay_dir": cfg.get("replay_dir"),
            "replay_save_interval": cfg.get("replay_save_interval"),
            "replay_save_interval_source": cfg.get("replay_save_interval_source"),
            "opponent_pool_seed": cfg.get("opponent_pool_seed"),
            "min_eval_episodes": cfg.get("min_eval_episodes"),
            "min_map_episodes": cfg.get("min_map_episodes"),
            "min_replay_combat_maps": cfg.get("min_replay_combat_maps"),
            "min_opponent_historical_samples": cfg.get(
                "min_opponent_historical_samples"
            ),
            "min_head_to_head_episodes": cfg.get("min_head_to_head_episodes"),
            "min_head_to_head_map_episodes": cfg.get(
                "min_head_to_head_map_episodes"
            ),
            "require_candidate_checkpoint": cfg.get("require_candidate_checkpoint"),
            "require_candidate_metadata": cfg.get("require_candidate_metadata"),
            "require_candidate_integrity": cfg.get("require_candidate_integrity"),
            "required_curriculum_stage": cfg.get("required_curriculum_stage"),
            "required_reward_preset": cfg.get("required_reward_preset"),
            "require_head_to_head": cfg.get("require_head_to_head"),
            "source_commit": source_control.get("commit"),
            "source_dirty": source_control.get("dirty"),
            "source_status_short_count": source_control.get(
                "status_short_count"
            ),
            "preflight_shell_script_path": data.get("preflight_shell_script_path"),
            "has_preflight_shell_script": bool(data.get("preflight_shell_script")),
            "rank_gate": cfg.get("rank_gate", {}),
            "strategy_report": cfg.get("strategy_report", {}),
            "command_count": len(commands),
            "expensive_command_ids": [
                command.get("id")
                for command in commands
                if command.get("expensive")
            ],
        }
    if artifact_type == "checkpoint_trust_manifest":
        checkpoints = data.get("checkpoints", {})
        return {
            "trusted_checkpoint_count": (
                len(checkpoints) if isinstance(checkpoints, dict) else None
            ),
        }
    if artifact_type == "long_run_check":
        candidate = data.get("candidate") or {}
        checks = data.get("checks", [])
        required_checks = [check for check in checks if check.get("required", True)]
        failed_required_checks = [
            check for check in required_checks if not check.get("passed")
        ]
        return {
            "passed": data.get("passed"),
            "candidate_label": candidate.get("label"),
            "candidate_score": candidate.get("score"),
            "required_check_count": len(required_checks),
            "failed_required_check_count": len(failed_required_checks),
            "failed_required_checks": [
                check.get("id") for check in failed_required_checks
            ],
        }
    if artifact_type == "long_run_status":
        latest = data.get("latest_manifest") or {}
        checkpoint_opponent_pool = latest.get("checkpoint_opponent_pool") or {}
        return {
            "latest_run_id": latest.get("run_id"),
            "latest_launcher_path": latest.get("launcher_path"),
            "latest_preflight_launcher_path": latest.get("preflight_launcher_path"),
            "latest_eval_dir_exists": latest.get("eval_dir_exists"),
            "latest_preflight_dir_exists": latest.get("preflight_dir_exists"),
            "latest_passing_long_run_check_count": latest.get(
                "passing_long_run_check_count"
            ),
            "candidate_evidence_ready": data.get("candidate_evidence_ready"),
            "blocked_reason": data.get("blocked_reason"),
            "missing_evidence": data.get("missing_evidence", []),
            "latest_manifest_source_safe_to_launch": latest.get(
                "source_safe_to_launch"
            ),
            "latest_manifest_source_stale_reasons": latest.get(
                "source_stale_reasons", []
            ),
            "latest_checkpoint_max_historical_samples": (
                checkpoint_opponent_pool.get("max_historical_samples")
            ),
            "latest_checkpoint_historical_sample_ready": (
                checkpoint_opponent_pool.get("meets_min_opponent_historical_samples")
            ),
            "next_command": data.get("next_command"),
            "next_preflight_command": data.get("next_preflight_command"),
        }
    if artifact_type == "league_health":
        health = data.get("health", {})
        signals = data.get("signals", {})
        return {
            "ready": health.get("ready"),
            "blockers": health.get("blockers", []),
            "warnings": health.get("warnings", []),
            "candidate_label": signals.get("candidate", {}).get("label"),
            "strategy_issue_count": signals.get("strategy", {}).get("issue_count"),
            "candidate_strategy_issue_count": signals.get("strategy", {}).get(
                "candidate_issue_count"
            ),
            "strategy_skipped_artifact_count": signals.get("strategy", {}).get(
                "skipped_artifact_count"
            ),
            "historical_sample_ready": signals.get("opponent_pool", {}).get(
                "historical_sample_ready"
            ),
            "max_historical_samples": signals.get("opponent_pool", {}).get(
                "max_historical_samples"
            ),
            "weakness_count": signals.get("map_weaknesses", {}).get("count"),
            "worst_weakness": signals.get("map_weaknesses", {}).get("worst"),
            "head_to_head_candidate_elo": signals.get("head_to_head", {}).get(
                "candidate_elo"
            ),
            "long_run_check_passed": signals.get("long_run", {}).get(
                "latest_check_passed"
            ),
            "replay_strategy_issue_count": signals.get("strategy", {}).get(
                "replay_issue_count"
            ),
            "smoke_strategy_issue_count": signals.get("strategy", {}).get(
                "smoke_issue_count"
            ),
            "self_play_sampling_passed": signals.get("self_play_sampling", {}).get(
                "passed"
            ),
            "self_play_sampling_historical_samples": signals.get(
                "self_play_sampling", {}
            ).get("historical_samples"),
        }
    if artifact_type == "reward_shaping_smoke":
        return {
            "reward_delta_agent_0": data.get("reward_delta_agent_0"),
            "reward_delta_agent_1": data.get("reward_delta_agent_1"),
            "draw_rate_delta": data.get("draw_rate_delta"),
            "idle_rate_delta_agent_0": data.get("idle_rate_delta_agent_0"),
            "dominant_action_rate_delta_agent_0": data.get(
                "dominant_action_rate_delta_agent_0"
            ),
            "no_damage_episodes_delta": data.get("no_damage_episodes_delta"),
            "low_engagement_episodes_delta": data.get(
                "low_engagement_episodes_delta"
            ),
            "damage_events_delta_agent_0": data.get(
                "damage_events_delta_agent_0"
            ),
            "strategy_issue_count": data.get("strategy_issue_count"),
            "indexed_artifact_count": data.get("indexed_artifact_count"),
        }
    if artifact_type == "long_run_artifact_smoke":
        counts = data.get("indexed_artifact_counts", {})
        preflight = data.get("status_self_play_sampling_preflight") or {}
        if not isinstance(preflight, dict):
            preflight = {}
        return {
            "run_id": data.get("run_id"),
            "status_blocked_reason": data.get("status_blocked_reason"),
            "status_missing_evidence": data.get("status_missing_evidence", []),
            "health_ready": data.get("health_ready"),
            "health_blockers": data.get("health_blockers", []),
            "health_warnings": data.get("health_warnings", []),
            "health_artifact_scope_dir": data.get("health_artifact_scope_dir"),
            "self_play_sampling_preflight_state": data.get(
                "self_play_sampling_preflight_state"
            ),
            "self_play_sampling_preflight_passed": preflight.get("passed"),
            "self_play_sampling_preflight_failed_checks": preflight.get(
                "failed_checks", []
            ),
            "indexed_artifact_count": data.get("indexed_artifact_count"),
            "indexed_long_run_manifest_count": counts.get("long_run_manifest"),
            "indexed_long_run_status_count": counts.get("long_run_status"),
            "indexed_league_health_count": counts.get("league_health"),
        }
    if artifact_type == "self_play_sampling_smoke":
        return {
            "passed": data.get("passed"),
            "latest_samples": data.get("latest_samples"),
            "historical_samples": data.get("historical_samples"),
            "historical_sample_rate": data.get("historical_sample_rate"),
            "unique_maps_seen": data.get("unique_maps_seen"),
            "map_counts": data.get("map_counts", {}),
        }
    if artifact_type == "smoke_suite":
        smokes = data.get("smokes", {})
        reward = smokes.get("reward_shaping", {})
        self_play_sampling = smokes.get("self_play_sampling", {})
        long_run_artifact = smokes.get("long_run_artifact", {})
        train_eval = smokes.get("train_eval", {})
        return {
            "smoke_count": data.get("smoke_count"),
            "smoke_order": data.get("smoke_order", []),
            "compute_classes": data.get("compute_classes", {}),
            "summary_paths": data.get("summary_paths", {}),
            "reward_strategy_issue_count": reward.get("strategy_issue_count"),
            "reward_indexed_artifact_count": reward.get("indexed_artifact_count"),
            "reward_idle_rate_delta_agent_0": reward.get(
                "idle_rate_delta_agent_0"
            ),
            "reward_dominant_action_rate_delta_agent_0": reward.get(
                "dominant_action_rate_delta_agent_0"
            ),
            "reward_no_damage_episodes_delta": reward.get(
                "no_damage_episodes_delta"
            ),
            "reward_low_engagement_episodes_delta": reward.get(
                "low_engagement_episodes_delta"
            ),
            "reward_damage_events_delta_agent_0": reward.get(
                "damage_events_delta_agent_0"
            ),
            "self_play_sampling_passed": self_play_sampling.get("passed"),
            "self_play_sampling_historical_samples": self_play_sampling.get(
                "historical_samples"
            ),
            "self_play_sampling_unique_maps_seen": self_play_sampling.get(
                "unique_maps_seen"
            ),
            "long_run_artifact_health_ready": long_run_artifact.get("health_ready"),
            "long_run_artifact_health_blockers": long_run_artifact.get(
                "health_blockers",
                [],
            ),
            "long_run_artifact_health_warnings": long_run_artifact.get(
                "health_warnings",
                [],
            ),
            "train_eval_long_run_check_passed": train_eval.get(
                "long_run_check_passed"
            ),
        }
    return {}


def artifact_links(data: dict, artifact_type: str) -> dict:
    if artifact_type in {"comparison", "gate"}:
        return {
            "before_path": data.get("before_path"),
            "after_path": data.get("after_path"),
        }
    if artifact_type == "rank_gate":
        return {"rank_summary_path": data.get("rank_summary_path")}
    if artifact_type == "promotion_audit":
        return {
            "rank_artifact_path": data.get("rank_artifact_path"),
            "rank_gate_artifact_path": data.get("rank_gate_artifact_path"),
        }
    if artifact_type == "audit_summary":
        return {
            "promotion_audit_path": data.get("audit_summary_path"),
            "rank_artifact_path": data.get("rank_artifact_path"),
            "rank_gate_artifact_path": data.get("rank_gate_artifact_path"),
        }
    return {}


def build_artifact_index(artifact_dir: str | Path, recursive: bool = False) -> dict:
    root = Path(artifact_dir)
    patterns = (
        ("**/*.json", "**/*.exitcode", "**/*.sh", "**/*.out")
        if recursive
        else ("*.json", "*.exitcode", "*.sh", "*.out")
    )
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in root.glob(pattern)
            if _is_indexable_artifact_path(path, root)
        }
    )
    artifacts = [summarize_artifact_file(path, root=root) for path in paths]
    counts: dict[str, int] = {}
    for entry in artifacts:
        artifact_type = entry["artifact_type"]
        counts[artifact_type] = counts.get(artifact_type, 0) + 1

    return {
        "artifact": artifact_metadata("artifact_index"),
        "index_config": {
            "artifact_dir": str(root),
            "recursive": recursive,
            "artifact_count": len(artifacts),
        },
        "artifact_counts": counts,
        "artifacts": artifacts,
    }


def run_artifact_index(
    artifact_dir: str,
    recursive: bool,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    index = build_artifact_index(artifact_dir, recursive=recursive)
    print(json.dumps(index, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or "artifact-index"
        path = write_eval_summary(index, output_dir, label=label)
        print(f"Saved artifact index to {path}")


def add_rate_issue(
    issues: list[dict],
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
    scope: str,
    metric: str,
    value,
    threshold: float,
    reason: str,
) -> None:
    if value is None:
        return
    value = float(value)
    if value > threshold:
        issues.append(
            {
                "path": path,
                "relative_path": relative_path,
                "artifact_type": artifact_type,
                "scope": scope,
                "metric": metric,
                "value": value,
                "threshold": threshold,
                "reason": reason,
            }
        )


def eval_strategy_issues(
    summary: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
    scope: str,
    thresholds: dict[str, float],
) -> list[dict]:
    episodes = int(summary.get("episodes", 0) or 0)
    behavior = summary.get("behavior", {})
    no_damage_rate = None
    low_engagement_rate = None
    if episodes > 0:
        no_damage_rate = float(behavior.get("no_damage_episodes", 0)) / episodes
        low_engagement_rate = (
            float(behavior.get("low_engagement_episodes", 0)) / episodes
        )

    issues: list[dict] = []
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric="draw_rate",
        value=summary.get("draw_rate"),
        threshold=thresholds["max_draw_rate"],
        reason="draw_rate_above_threshold",
    )
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric="no_damage_rate",
        value=no_damage_rate,
        threshold=thresholds["max_no_damage_rate"],
        reason="no_damage_rate_above_threshold",
    )
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric="low_engagement_rate",
        value=low_engagement_rate,
        threshold=thresholds["max_low_engagement_rate"],
        reason="low_engagement_rate_above_threshold",
    )
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric="idle_rate_agent_0",
        value=behavior.get("avg_idle_rate", {}).get("agent_0"),
        threshold=thresholds["max_idle_rate"],
        reason="agent_0_idle_rate_above_threshold",
    )
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric="dominant_action_rate_agent_0",
        value=behavior.get("avg_dominant_action_rate", {}).get("agent_0"),
        threshold=thresholds["max_dominant_action_rate"],
        reason="agent_0_dominant_action_rate_above_threshold",
    )
    return issues


def rank_strategy_issues(
    summary: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
    thresholds: dict[str, float],
) -> list[dict]:
    issues: list[dict] = []
    for ranking in summary.get("rankings", []):
        scope = f"rank:{ranking.get('label')}"
        add_rate_issue(
            issues,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            scope=scope,
            metric="mean_draw_rate",
            value=ranking.get("mean_draw_rate"),
            threshold=thresholds["max_draw_rate"],
            reason="mean_draw_rate_above_threshold",
        )
        add_rate_issue(
            issues,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            scope=scope,
            metric="mean_no_damage_rate",
            value=ranking.get("mean_no_damage_rate"),
            threshold=thresholds["max_no_damage_rate"],
            reason="mean_no_damage_rate_above_threshold",
        )
        add_rate_issue(
            issues,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            scope=scope,
            metric="mean_low_engagement_rate",
            value=ranking.get("mean_low_engagement_rate"),
            threshold=thresholds["max_low_engagement_rate"],
            reason="mean_low_engagement_rate_above_threshold",
        )
    for entry in summary.get("suites", []):
        label = entry.get("label")
        suite = entry.get("suite") or {}
        for map_name, opponents in suite.get("matchups", {}).items():
            for opponent, matchup in opponents.items():
                issues.extend(
                    eval_strategy_issues(
                        matchup,
                        path=path,
                        relative_path=relative_path,
                        artifact_type=artifact_type,
                        scope=f"candidate:{label}:rank_suite:{map_name}/{opponent}",
                        thresholds=thresholds,
                    )
                )
    return issues


def candidate_strategy_issues(
    summary: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
    thresholds: dict[str, float],
) -> list[dict]:
    candidate = summary.get("candidate") or {}
    scope = f"candidate:{candidate.get('label')}"
    issues: list[dict] = []
    draw_metric = "mean_draw_rate"
    draw_rate = candidate.get(draw_metric)
    if draw_rate is None:
        draw_metric = "draw_rate"
        draw_rate = candidate.get(draw_metric)
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric=draw_metric,
        value=draw_rate,
        threshold=thresholds["max_draw_rate"],
        reason=f"candidate_{draw_metric}_above_threshold",
    )
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric="mean_no_damage_rate",
        value=candidate.get("mean_no_damage_rate"),
        threshold=thresholds["max_no_damage_rate"],
        reason="candidate_no_damage_rate_above_threshold",
    )
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric="mean_low_engagement_rate",
        value=candidate.get("mean_low_engagement_rate"),
        threshold=thresholds["max_low_engagement_rate"],
        reason="candidate_low_engagement_rate_above_threshold",
    )
    return issues


def replay_analysis_strategy_issues(
    summary: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
    thresholds: dict[str, float],
) -> list[dict]:
    episode_id = summary.get("episode_id")
    scope = f"replay:{episode_id if episode_id is not None else Path(path).stem}"
    flags = summary.get("flags", {})
    totals = summary.get("totals", {})
    behavior = summary.get("behavior", {})
    no_damage = flags.get("no_damage")
    if no_damage is None and "damage_dealt" in totals:
        no_damage = int(totals.get("damage_dealt", 0)) == 0

    issues: list[dict] = []
    if no_damage is not None:
        add_rate_issue(
            issues,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            scope=scope,
            metric="replay_no_damage",
            value=1.0 if no_damage else 0.0,
            threshold=thresholds["max_no_damage_rate"],
            reason="replay_no_damage",
        )
    if no_damage is True and summary.get("winner") == "draw":
        add_rate_issue(
            issues,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            scope=scope,
            metric="replay_low_engagement",
            value=1.0,
            threshold=thresholds["max_low_engagement_rate"],
            reason="replay_no_damage_draw",
        )
    if flags.get("no_attacks") is True and summary.get("winner") == "draw":
        add_rate_issue(
            issues,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            scope=scope,
            metric="replay_no_attacks",
            value=1.0,
            threshold=thresholds["max_low_engagement_rate"],
            reason="replay_no_attacks_draw",
        )
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric="replay_idle_rate_agent_0",
        value=behavior.get("avg_idle_rate", {}).get("agent_0"),
        threshold=thresholds["max_idle_rate"],
        reason="replay_agent_0_idle_rate_above_threshold",
    )
    add_rate_issue(
        issues,
        path=path,
        relative_path=relative_path,
        artifact_type=artifact_type,
        scope=scope,
        metric="replay_dominant_action_rate_agent_0",
        value=behavior.get("avg_dominant_action_rate", {}).get("agent_0"),
        threshold=thresholds["max_dominant_action_rate"],
        reason="replay_agent_0_dominant_action_rate_above_threshold",
    )
    return issues


def long_run_status_strategy_issues(
    summary: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
) -> list[dict]:
    latest = summary.get("latest_manifest")
    if not isinstance(latest, dict):
        latest = {}
    checkpoint_opponent_pool = latest.get("checkpoint_opponent_pool")
    if not isinstance(checkpoint_opponent_pool, dict):
        checkpoint_opponent_pool = {}
    min_historical_samples = json_non_negative_int(
        latest.get("min_opponent_historical_samples")
    )
    if min_historical_samples is None:
        min_historical_samples = (
            json_non_negative_int(
                checkpoint_opponent_pool.get("min_opponent_historical_samples")
            )
            or 0
        )
    max_historical_samples = json_non_negative_int(
        checkpoint_opponent_pool.get("max_historical_samples")
    )
    missing_evidence = summary.get("missing_evidence", [])
    if not isinstance(missing_evidence, list):
        missing_evidence = []
    missing_historical_samples = (
        "checkpoint_historical_opponent_samples" in missing_evidence
    )
    ready = checkpoint_opponent_pool.get("meets_min_opponent_historical_samples")

    if min_historical_samples <= 0:
        return []
    if ready is not False and not missing_historical_samples:
        return []

    latest_check = latest.get("latest_long_run_check") or {}
    latest_candidate = (
        latest_check.get("candidate") if isinstance(latest_check, dict) else None
    )
    scope_label = (
        latest_candidate.get("label")
        if isinstance(latest_candidate, dict)
        else None
    )
    scope_label = scope_label or latest.get("run_id") or "latest_manifest"
    return [
        {
            "path": path,
            "relative_path": relative_path,
            "artifact_type": artifact_type,
            "scope": f"candidate:{scope_label}:checkpoint_opponent_pool",
            "metric": "checkpoint_historical_opponent_samples",
            "value": max_historical_samples,
            "threshold": min_historical_samples,
            "reason": "checkpoint_historical_opponent_samples_below_min",
            "missing_evidence": missing_evidence,
        }
    ]


def smoke_suite_strategy_issues(
    summary: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
) -> list[dict]:
    smokes = summary.get("smokes")
    if not isinstance(smokes, dict):
        smokes = {}

    reward = smokes.get("reward_shaping")
    if not isinstance(reward, dict):
        reward = {}
    long_run_artifact = smokes.get("long_run_artifact")
    if not isinstance(long_run_artifact, dict):
        long_run_artifact = {}
    self_play_sampling = smokes.get("self_play_sampling")
    if not isinstance(self_play_sampling, dict):
        self_play_sampling = {}
    train_eval = smokes.get("train_eval")
    if not isinstance(train_eval, dict):
        train_eval = {}

    issues: list[dict] = []
    reward_failed_checks = failed_smoke_check_ids(reward)
    if reward.get("passed") is False:
        issues.append(
            {
                "path": path,
                "relative_path": relative_path,
                "artifact_type": artifact_type,
                "scope": "smoke:reward_shaping",
                "metric": "smoke_reward_shaping_failed",
                "value": len(reward_failed_checks),
                "threshold": 0,
                "reason": "smoke_reward_shaping_checks_failed",
                "failed_checks": reward_failed_checks,
            }
        )

    long_run_artifact_failed_checks = failed_smoke_check_ids(long_run_artifact)
    health_blockers = long_run_artifact.get("health_blockers", [])
    if not isinstance(health_blockers, list):
        health_blockers = []
    health_warnings = long_run_artifact.get("health_warnings", [])
    if not isinstance(health_warnings, list):
        health_warnings = []
    if long_run_artifact.get("passed") is False:
        issues.append(
            {
                "path": path,
                "relative_path": relative_path,
                "artifact_type": artifact_type,
                "scope": "smoke:long_run_artifact",
                "metric": "smoke_long_run_artifact_failed",
                "value": len(long_run_artifact_failed_checks),
                "threshold": 0,
                "reason": "smoke_long_run_artifact_checks_failed",
                "failed_checks": long_run_artifact_failed_checks,
                "blockers": health_blockers,
                "warnings": health_warnings,
            }
        )

    self_play_failed_checks = failed_smoke_check_ids(self_play_sampling)
    if self_play_sampling.get("passed") is False:
        issues.append(
            {
                "path": path,
                "relative_path": relative_path,
                "artifact_type": artifact_type,
                "scope": "smoke:self_play_sampling",
                "metric": "smoke_self_play_sampling_failed",
                "value": len(self_play_failed_checks),
                "threshold": 0,
                "reason": "smoke_self_play_sampling_checks_failed",
                "failed_checks": self_play_failed_checks,
            }
        )

    train_eval_issue_count = json_non_negative_int(
        train_eval.get("strategy_issue_count")
    )
    if train_eval_issue_count and train_eval_issue_count > 0:
        issues.append(
            {
                "path": path,
                "relative_path": relative_path,
                "artifact_type": artifact_type,
                "scope": "smoke:train_eval",
                "metric": "smoke_train_eval_strategy_issue_count",
                "value": train_eval_issue_count,
                "threshold": 0,
                "reason": "smoke_train_eval_strategy_issues_present",
            }
        )

    failed_checks = train_eval.get("long_run_check_failed_checks", [])
    if not isinstance(failed_checks, list):
        failed_checks = []
    if train_eval.get("long_run_check_passed") is False:
        issues.append(
            {
                "path": path,
                "relative_path": relative_path,
                "artifact_type": artifact_type,
                "scope": "smoke:train_eval",
                "metric": "smoke_train_eval_long_run_check_failed",
                "value": 1,
                "threshold": 0,
                "reason": "smoke_train_eval_long_run_check_failed",
                "failed_checks": failed_checks,
            }
        )

    return issues


def self_play_sampling_smoke_strategy_issues(
    summary: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
) -> list[dict]:
    failed_checks = failed_smoke_check_ids(summary)
    if summary.get("passed") is not False:
        return []
    return [
        {
            "path": path,
            "relative_path": relative_path,
            "artifact_type": artifact_type,
            "scope": "smoke:self_play_sampling",
            "metric": "self_play_sampling_smoke_failed",
            "value": len(failed_checks),
            "threshold": 0,
            "reason": "self_play_sampling_smoke_checks_failed",
            "failed_checks": failed_checks,
        }
    ]


def failed_smoke_check_ids(summary: dict) -> list[str]:
    checks = summary.get("checks", [])
    if not isinstance(checks, list):
        return []
    return [
        str(check.get("id"))
        for check in checks
        if isinstance(check, dict) and check.get("passed") is False
    ]


def reward_shaping_smoke_strategy_issues(
    summary: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
) -> list[dict]:
    issues: list[dict] = []
    failed_checks = failed_smoke_check_ids(summary)
    if summary.get("passed") is False:
        issues.append(
            {
                "path": path,
                "relative_path": relative_path,
                "artifact_type": artifact_type,
                "scope": "smoke:reward_shaping",
                "metric": "reward_shaping_smoke_failed",
                "value": len(failed_checks),
                "threshold": 0,
                "reason": "reward_shaping_smoke_checks_failed",
                "failed_checks": failed_checks,
            }
        )

    for agent_name in ("agent_0", "agent_1"):
        metric = f"reward_delta_{agent_name}"
        reward_delta = json_number(summary.get(metric))
        if reward_delta is not None and reward_delta >= 0.0:
            issues.append(
                {
                    "path": path,
                    "relative_path": relative_path,
                    "artifact_type": artifact_type,
                    "scope": f"smoke:reward_shaping:{agent_name}",
                    "metric": metric,
                    "value": reward_delta,
                    "threshold": 0.0,
                    "reason": "anti_stall_idle_reward_not_reduced",
                }
            )

    draw_rate_delta = json_number(summary.get("draw_rate_delta"))
    if draw_rate_delta is not None and draw_rate_delta > 0.0:
        issues.append(
            {
                "path": path,
                "relative_path": relative_path,
                "artifact_type": artifact_type,
                "scope": "smoke:reward_shaping",
                "metric": "draw_rate_delta",
                "value": draw_rate_delta,
                "threshold": 0.0,
                "reason": "anti_stall_draw_rate_increased",
            }
        )
    return issues


def long_run_artifact_smoke_strategy_issues(
    summary: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
) -> list[dict]:
    health_blockers = summary.get("health_blockers", [])
    if not isinstance(health_blockers, list):
        health_blockers = []
    health_warnings = summary.get("health_warnings", [])
    if not isinstance(health_warnings, list):
        health_warnings = []
    failed_checks = failed_smoke_check_ids(summary)
    if summary.get("passed") is True:
        return []
    if summary.get("passed") is False:
        return [
            {
                "path": path,
                "relative_path": relative_path,
                "artifact_type": artifact_type,
                "scope": "smoke:long_run_artifact",
                "metric": "long_run_artifact_smoke_failed",
                "value": len(failed_checks),
                "threshold": 0,
                "reason": "long_run_artifact_smoke_checks_failed",
                "failed_checks": failed_checks,
                "blockers": health_blockers,
                "warnings": health_warnings,
            }
        ]
    if summary.get("health_ready") is not False or not health_blockers:
        return []
    return [
        {
            "path": path,
            "relative_path": relative_path,
            "artifact_type": artifact_type,
            "scope": "smoke:long_run_artifact",
            "metric": "long_run_artifact_smoke_health_blockers",
            "value": len(health_blockers),
            "threshold": 0,
            "reason": "long_run_artifact_smoke_health_blocked",
            "blockers": health_blockers,
            "warnings": health_warnings,
        }
    ]


def strategy_issues_for_artifact(
    data: dict,
    *,
    artifact_type: str,
    path: str,
    relative_path: str | None,
    thresholds: dict[str, float],
) -> list[dict]:
    if artifact_type == "eval":
        return eval_strategy_issues(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            scope="eval",
            thresholds=thresholds,
        )
    if artifact_type == "suite":
        issues = []
        for map_name, opponents in data.get("matchups", {}).items():
            for opponent, matchup in opponents.items():
                issues.extend(
                    eval_strategy_issues(
                        matchup,
                        path=path,
                        relative_path=relative_path,
                        artifact_type=artifact_type,
                        scope=f"suite:{map_name}/{opponent}",
                        thresholds=thresholds,
                    )
                )
        return issues
    if artifact_type == "rank":
        return rank_strategy_issues(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            thresholds=thresholds,
        )
    if artifact_type in {"rank_gate", "promotion_audit", "audit_summary"}:
        return candidate_strategy_issues(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            thresholds=thresholds,
        )
    if artifact_type == "replay_analysis":
        return replay_analysis_strategy_issues(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            thresholds=thresholds,
        )
    if artifact_type == "long_run_status":
        return long_run_status_strategy_issues(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
        )
    if artifact_type == "smoke_suite":
        return smoke_suite_strategy_issues(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
        )
    if artifact_type == "reward_shaping_smoke":
        return reward_shaping_smoke_strategy_issues(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
        )
    if artifact_type == "long_run_artifact_smoke":
        return long_run_artifact_smoke_strategy_issues(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
        )
    if artifact_type == "self_play_sampling_smoke":
        return self_play_sampling_smoke_strategy_issues(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
        )
    return []


def _weakness_from_matchup_score(
    item: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
    scope: str,
    label: str | None = None,
) -> dict:
    weakness = {
        "path": path,
        "relative_path": relative_path,
        "artifact_type": artifact_type,
        "scope": scope,
        "map_name": item.get("map_name"),
        "opponent": item.get("opponent"),
        "score": float(item.get("score", 0.0)),
        "episodes": int(item.get("episodes", 0) or 0),
        "win_rate_agent_0": float(item.get("win_rate_agent_0", 0.0)),
        "draw_rate": float(item.get("draw_rate", 0.0)),
        "no_damage_rate": float(item.get("no_damage_rate", 0.0)),
        "low_engagement_rate": float(item.get("low_engagement_rate", 0.0)),
        "avg_length": float(item.get("avg_length", 0.0)),
    }
    if label is not None:
        weakness["label"] = label
    return weakness


def _suite_weaknesses(
    suite: dict,
    *,
    path: str,
    relative_path: str | None,
    artifact_type: str,
    scope_prefix: str,
    label: str | None = None,
) -> list[dict]:
    scored = score_baseline_suite(suite)
    return [
        _weakness_from_matchup_score(
            item,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            scope=f"{scope_prefix}:{item.get('map_name')}/{item.get('opponent')}",
            label=label,
        )
        for item in scored["matchup_scores"]
    ]


def strategy_weaknesses_for_artifact(
    data: dict,
    *,
    artifact_type: str,
    path: str,
    relative_path: str | None,
) -> list[dict]:
    if artifact_type == "suite":
        return _suite_weaknesses(
            data,
            path=path,
            relative_path=relative_path,
            artifact_type=artifact_type,
            scope_prefix="suite",
        )
    if artifact_type == "rank":
        weaknesses = []
        for ranking in data.get("rankings", []):
            label = ranking.get("label")
            for item in ranking.get("matchup_scores", []):
                weaknesses.append(
                    _weakness_from_matchup_score(
                        item,
                        path=path,
                        relative_path=relative_path,
                        artifact_type=artifact_type,
                        scope=(
                            f"rank:{label}:{item.get('map_name')}/"
                            f"{item.get('opponent')}"
                        ),
                        label=label,
                    )
                )
        for entry in data.get("suites", []):
            label = entry.get("label")
            suite = entry.get("suite") or {}
            weaknesses.extend(
                _suite_weaknesses(
                    suite,
                    path=path,
                    relative_path=relative_path,
                    artifact_type=artifact_type,
                    scope_prefix=f"rank_suite:{label}",
                    label=label,
                )
            )
        return weaknesses
    if artifact_type in {"rank_gate", "promotion_audit", "audit_summary"}:
        candidate = data.get("candidate") or {}
        label = candidate.get("label")
        return [
            _weakness_from_matchup_score(
                item,
                path=path,
                relative_path=relative_path,
                artifact_type=artifact_type,
                scope=(
                    f"candidate:{label}:{item.get('map_name')}/"
                    f"{item.get('opponent')}"
                ),
                label=label,
            )
            for item in candidate.get("matchup_scores", [])
        ]
    return []


def build_strategy_report(
    artifact_dir: str | Path,
    recursive: bool = False,
    max_draw_rate: float = 0.9,
    max_no_damage_rate: float = 0.75,
    max_low_engagement_rate: float = 0.5,
    max_idle_rate: float = 0.75,
    max_dominant_action_rate: float = 0.95,
    max_weaknesses: int = 10,
) -> dict:
    root = Path(artifact_dir)
    thresholds = {
        "max_draw_rate": max_draw_rate,
        "max_no_damage_rate": max_no_damage_rate,
        "max_low_engagement_rate": max_low_engagement_rate,
        "max_idle_rate": max_idle_rate,
        "max_dominant_action_rate": max_dominant_action_rate,
    }
    pattern = "**/*.json" if recursive else "*.json"
    paths = sorted(path for path in root.glob(pattern) if path.is_file())
    issues = []
    weaknesses = []
    scanned = 0
    skipped = []
    for path in paths:
        relative_path = None
        try:
            relative_path = str(path.relative_to(root))
            data = load_eval_summary(path)
        except Exception as exc:
            skipped.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue

        artifact = data.get("artifact", {})
        artifact_type = "unknown"
        if isinstance(artifact, dict):
            artifact_type = artifact.get("artifact_type") or "unknown"
        try:
            artifact_issues = strategy_issues_for_artifact(
                data,
                artifact_type=artifact_type,
                path=str(path),
                relative_path=relative_path,
                thresholds=thresholds,
            )
            artifact_weaknesses = strategy_weaknesses_for_artifact(
                data,
                artifact_type=artifact_type,
                path=str(path),
                relative_path=relative_path,
            )
        except Exception as exc:
            skipped.append(
                {
                    "path": str(path),
                    "relative_path": relative_path,
                    "artifact_type": artifact_type,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        scanned += 1
        issues.extend(artifact_issues)
        weaknesses.extend(artifact_weaknesses)

    weaknesses.sort(
        key=lambda item: (
            item["score"],
            item["win_rate_agent_0"],
            -item["draw_rate"],
            -item["no_damage_rate"],
            -item["low_engagement_rate"],
            item["scope"],
        )
    )
    weakness_count = len(weaknesses)
    if max_weaknesses >= 0:
        weaknesses = weaknesses[:max_weaknesses]

    return {
        "artifact": artifact_metadata("strategy_report"),
        "report_config": {
            "artifact_dir": str(root),
            "recursive": recursive,
            "max_weaknesses": max_weaknesses,
            **thresholds,
        },
        "scanned_artifacts": scanned,
        "skipped_artifacts": skipped,
        "issue_count": len(issues),
        "issues": issues,
        "weakness_count": weakness_count,
        "weaknesses": weaknesses,
    }


def run_strategy_report(
    artifact_dir: str,
    recursive: bool,
    max_draw_rate: float,
    max_no_damage_rate: float,
    max_low_engagement_rate: float,
    max_idle_rate: float,
    max_dominant_action_rate: float,
    max_weaknesses: int = 10,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    report = build_strategy_report(
        artifact_dir,
        recursive=recursive,
        max_draw_rate=max_draw_rate,
        max_no_damage_rate=max_no_damage_rate,
        max_low_engagement_rate=max_low_engagement_rate,
        max_idle_rate=max_idle_rate,
        max_dominant_action_rate=max_dominant_action_rate,
        max_weaknesses=max_weaknesses,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or "strategy-report"
        path = write_eval_summary(report, output_dir, label=label)
        print(f"Saved strategy report to {path}")


def check_result(
    check_id: str,
    passed: bool,
    details: dict,
    required: bool = True,
) -> dict:
    return {
        "id": check_id,
        "required": required,
        "passed": bool(passed),
        "details": details,
    }


def candidate_map_names(candidate: dict) -> list[str]:
    maps = {
        item.get("map_name")
        for item in candidate.get("matchup_scores", [])
        if item.get("map_name")
    }
    return sorted(maps)


def missing_required_maps(
    candidate_maps: list[str],
    required_maps: tuple[str, ...],
) -> list[str]:
    present = set(candidate_maps)
    return [map_name for map_name in required_maps if map_name not in present]


def checkpoint_metadata_maps(metadata: dict | None) -> list[str]:
    if not isinstance(metadata, dict):
        return []

    curriculum = metadata.get("curriculum") or {}
    curriculum_maps = set()
    if isinstance(curriculum, dict):
        curriculum_maps.update(curriculum.get("active_map_pool", []))
        stage = curriculum.get("stage") or {}
        if isinstance(stage, dict):
            curriculum_maps.update(stage.get("map_choices", []))
    if curriculum_maps:
        return sorted(curriculum_maps)

    maps = set()
    if metadata.get("randomize_maps"):
        maps.update(metadata.get("map_choices", []))
    elif metadata.get("map_name"):
        maps.add(metadata["map_name"])
    return sorted(maps)


def candidate_per_map_scores(candidate: dict) -> list[dict]:
    per_map_scores, _ = ranking_per_map_score_details(candidate)
    return per_map_scores


def candidate_strategy_issues_for_check(
    strategy_report: dict,
    candidate_label: str | None = None,
) -> list[dict]:
    candidate_metrics = {
        "draw_rate",
        "mean_draw_rate",
        "no_damage_rate",
        "low_engagement_rate",
        "mean_no_damage_rate",
        "mean_low_engagement_rate",
        "idle_rate_agent_0",
        "dominant_action_rate_agent_0",
        "checkpoint_historical_opponent_samples",
    }

    def matches_candidate_scope(issue: dict) -> bool:
        scope = str(issue.get("scope", ""))
        if candidate_label is None:
            return scope.startswith("candidate:")
        candidate_scope = f"candidate:{candidate_label}"
        return scope == candidate_scope or scope.startswith(f"{candidate_scope}:")

    return [
        issue
        for issue in strategy_report.get("issues", [])
        if matches_candidate_scope(issue)
        and issue.get("metric") in candidate_metrics
    ]


def replay_strategy_issues_for_check(strategy_report: dict) -> list[dict]:
    return [
        issue
        for issue in strategy_report.get("issues", [])
        if str(issue.get("scope", "")).startswith("replay:")
        and issue.get("metric") in REPLAY_STRATEGY_METRICS
    ]


def replay_analysis_has_combat(artifact_index: dict) -> bool:
    return replay_analysis_combat_summary(artifact_index)["combat_replay_count"] > 0


def replay_analysis_combat_summary(artifact_index: dict) -> dict:
    combat_maps = set()
    combat_replay_count = 0
    replay_analysis_count = 0
    for entry in artifact_index.get("artifacts", []):
        if entry.get("artifact_type") != "replay_analysis":
            continue
        replay_analysis_count += 1
        summary = entry.get("summary", {})
        flags = summary.get("flags", {})
        totals = summary.get("totals", {})
        has_combat = flags.get("no_damage") is False or int(
            totals.get("damage_dealt", 0)
        ) > 0
        if not has_combat:
            continue
        combat_replay_count += 1
        map_name = summary.get("map_name")
        if map_name:
            combat_maps.add(map_name)
    return {
        "replay_analysis_count": replay_analysis_count,
        "combat_replay_count": combat_replay_count,
        "combat_maps": sorted(combat_maps),
        "combat_map_count": len(combat_maps),
    }


def artifact_index_contains_path(artifact_index: dict, artifact_path: str | Path) -> bool:
    target = Path(artifact_path).resolve()
    root = artifact_index.get("index_config", {}).get("artifact_dir")
    root_path = Path(root) if root else None

    for entry in artifact_index.get("artifacts", []):
        candidates = []
        if entry.get("path"):
            candidates.append(Path(entry["path"]))
        if root_path is not None and entry.get("relative_path"):
            candidates.append(root_path / entry["relative_path"])
        for candidate in candidates:
            if candidate.resolve() == target:
                return True
    return False


def _summary_episode_count(summary: dict) -> int:
    return int(summary.get("episodes", 0) or 0)


def _suite_matchup_episode_count(suite: dict) -> tuple[int, int]:
    total = 0
    counted_matchups = 0
    matchups = suite.get("matchups", {})
    if not isinstance(matchups, dict):
        return total, counted_matchups

    for opponents in matchups.values():
        if not isinstance(opponents, dict):
            continue
        for summary in opponents.values():
            if not isinstance(summary, dict) or "episodes" not in summary:
                continue
            total += _summary_episode_count(summary)
            counted_matchups += 1
    return total, counted_matchups


def _head_to_head_episode_count(
    head_to_head: dict,
) -> tuple[int, int, dict[str, int]]:
    total = 0
    counted_sides = 0
    per_map: dict[str, int] = {}
    matchups = head_to_head.get("matchups", {})
    if not isinstance(matchups, dict):
        return total, counted_sides, per_map

    for map_name, map_matchups in matchups.items():
        if not isinstance(map_matchups, dict):
            continue
        for pairing in map_matchups.values():
            if not isinstance(pairing, dict):
                continue
            for side in ("forward", "reverse"):
                summary = pairing.get(side)
                if not isinstance(summary, dict) or "episodes" not in summary:
                    continue
                episodes = _summary_episode_count(summary)
                total += episodes
                per_map[map_name] = per_map.get(map_name, 0) + episodes
                counted_sides += 1
    return total, counted_sides, per_map


def rank_evaluation_episode_counts(
    rank_summary: dict | None,
    candidate_label: str | None = None,
) -> dict:
    if rank_summary is None:
        return {
            "baseline_episodes": 0,
            "candidate_baseline_episodes": 0,
            "head_to_head_episodes": 0,
            "total_episodes": 0,
            "configured_baseline_episodes": 0,
            "configured_head_to_head_episodes": 0,
            "configured_total_episodes": 0,
            "baseline_matchups_counted": 0,
            "candidate_baseline_matchups_counted": 0,
            "head_to_head_sides_counted": 0,
            "head_to_head_map_episodes": {},
        }

    if candidate_label is None:
        rankings = rank_summary.get("rankings", [])
        if rankings:
            candidate_label = rankings[0].get("label")

    cfg = rank_summary.get("rank_config", {})
    checkpoints = cfg.get("checkpoints", [])
    checkpoint_count = len(checkpoints) or len(rank_summary.get("rankings", []))
    map_count = len(cfg.get("maps", []))
    opponent_count = len(cfg.get("opponents", []))
    episodes_per_matchup = int(cfg.get("episodes_per_matchup", 0) or 0)
    baseline_episodes = (
        checkpoint_count * map_count * opponent_count * episodes_per_matchup
    )
    configured_head_to_head_episodes = int(
        rank_summary.get("head_to_head", {})
        .get("overview", {})
        .get("total_episodes", 0)
        or 0
    )
    actual_baseline_episodes = 0
    candidate_baseline_episodes = 0
    baseline_matchups_counted = 0
    candidate_baseline_matchups_counted = 0
    for entry in rank_summary.get("suites", []):
        suite = entry.get("suite", {}) if isinstance(entry, dict) else {}
        suite_episodes, suite_matchups = _suite_matchup_episode_count(suite)
        actual_baseline_episodes += suite_episodes
        baseline_matchups_counted += suite_matchups
        if isinstance(entry, dict) and entry.get("label") == candidate_label:
            candidate_baseline_episodes += suite_episodes
            candidate_baseline_matchups_counted += suite_matchups

    head_to_head = rank_summary.get("head_to_head", {})
    (
        actual_head_to_head_episodes,
        head_to_head_sides_counted,
        head_to_head_map_episodes,
    ) = (
        _head_to_head_episode_count(head_to_head)
        if isinstance(head_to_head, dict)
        else (0, 0, {})
    )

    return {
        "baseline_episodes": actual_baseline_episodes,
        "candidate_baseline_episodes": candidate_baseline_episodes,
        "head_to_head_episodes": actual_head_to_head_episodes,
        "total_episodes": actual_baseline_episodes + actual_head_to_head_episodes,
        "configured_baseline_episodes": baseline_episodes,
        "configured_head_to_head_episodes": configured_head_to_head_episodes,
        "configured_total_episodes": (
            baseline_episodes + configured_head_to_head_episodes
        ),
        "baseline_matchups_counted": baseline_matchups_counted,
        "candidate_baseline_matchups_counted": candidate_baseline_matchups_counted,
        "head_to_head_sides_counted": head_to_head_sides_counted,
        "head_to_head_map_episodes": dict(sorted(head_to_head_map_episodes.items())),
    }


def load_rank_for_promotion(promotion_audit: dict) -> dict | None:
    if "rank" in promotion_audit:
        return promotion_audit["rank"]
    rank_path = promotion_audit.get("rank_artifact_path")
    if not rank_path:
        return None
    path = Path(rank_path)
    if not path.exists():
        return None
    summary = load_eval_summary(path)
    validate_artifact(summary, "rank")
    return summary


def build_long_run_check(
    promotion_audit: dict,
    strategy_report: dict,
    artifact_index: dict,
    *,
    promotion_audit_path: str | None = None,
    strategy_report_path: str | None = None,
    min_maps: int = 2,
    required_maps: tuple[str, ...] = (),
    min_eval_episodes: int = 0,
    min_map_episodes: int | None = None,
    min_map_score: float | None = None,
    require_replay_analysis: bool = False,
    min_replay_combat_maps: int = 0,
    min_opponent_historical_samples: int = 0,
    require_candidate_checkpoint: bool = False,
    require_candidate_metadata: bool = False,
    require_candidate_integrity: bool = False,
    required_curriculum_stage: str | None = None,
    required_reward_preset: str | None = None,
    require_head_to_head: bool = False,
    min_head_to_head_episodes: int = 0,
    min_head_to_head_map_episodes: int | None = None,
) -> dict:
    validate_artifact(promotion_audit, "promotion_audit")
    validate_artifact(strategy_report, "strategy_report")
    validate_artifact(artifact_index, "artifact_index")

    candidate = promotion_audit.get("candidate") or {}
    candidate_label = candidate.get("label")
    map_names = candidate_map_names(candidate)
    per_map_scores, invalid_map_scores = ranking_per_map_score_details(candidate)
    counts = artifact_index.get("artifact_counts", {})
    bad_strategy_issues = candidate_strategy_issues_for_check(
        strategy_report,
        candidate_label=candidate_label,
    )
    replay_strategy_issues = replay_strategy_issues_for_check(strategy_report)
    strategy_skipped_artifacts = strategy_report.get("skipped_artifacts", [])
    if not isinstance(strategy_skipped_artifacts, list):
        strategy_skipped_artifacts = []
    rank_summary = load_rank_for_promotion(promotion_audit)
    eval_episode_counts = rank_evaluation_episode_counts(
        rank_summary,
        candidate_label=candidate_label,
    )
    replay_combat = replay_analysis_combat_summary(artifact_index)
    candidate_checkpoint = candidate.get("checkpoint")
    candidate_metadata = None
    candidate_metadata_error = None
    if candidate_checkpoint:
        try:
            candidate_metadata = read_checkpoint_metadata(candidate_checkpoint)
        except (OSError, json.JSONDecodeError) as exc:
            candidate_metadata_error = f"{type(exc).__name__}: {exc}"
    candidate_integrity = checkpoint_metadata_integrity(
        candidate_checkpoint,
        candidate_metadata,
    )
    candidate_integrity["metadata_error"] = candidate_metadata_error
    opponent_pool_metadata = (
        candidate_metadata.get("opponent_pool")
        if isinstance(candidate_metadata, dict)
        else None
    )
    opponent_historical_samples = (
        json_non_negative_int(opponent_pool_metadata.get("historical_samples"))
        if isinstance(opponent_pool_metadata, dict)
        else None
    )

    checks = [
        check_result(
            "promotion_audit_passed",
            bool(promotion_audit.get("passed")),
            {
                "failures": promotion_audit.get("failures", []),
            },
        ),
        check_result(
            "no_candidate_bad_strategy_issues",
            not bad_strategy_issues,
            {
                "issue_count": len(bad_strategy_issues),
                "issues": bad_strategy_issues,
            },
        ),
        check_result(
            "strategy_report_analyzed_all_artifacts",
            not strategy_skipped_artifacts,
            {
                "skipped_artifact_count": len(strategy_skipped_artifacts),
                "skipped_artifacts": strategy_skipped_artifacts,
            },
        ),
        check_result(
            "candidate_map_coverage",
            len(map_names) >= min_maps,
            {
                "map_count": len(map_names),
                "maps": map_names,
                "min_maps": min_maps,
            },
        ),
        check_result(
            "artifact_index_has_required_artifacts",
            all(
                counts.get(artifact_type, 0) > 0
                for artifact_type in (
                    "promotion_audit",
                    "strategy_report",
                    "rank",
                    "rank_gate",
                )
            ),
            {
                "artifact_counts": counts,
                "required_types": [
                    "promotion_audit",
                    "strategy_report",
                    "rank",
                    "rank_gate",
                ],
            },
        ),
    ]

    if required_maps:
        missing_maps = missing_required_maps(map_names, required_maps)
        checks.append(
            check_result(
                "candidate_required_maps",
                not missing_maps,
                {
                    "maps": map_names,
                    "required_maps": list(required_maps),
                    "missing_maps": missing_maps,
                },
            )
        )

    if invalid_map_scores:
        checks.append(
            check_result(
                "candidate_map_scores_valid",
                False,
                {
                    "invalid_map_scores": invalid_map_scores,
                    "per_map_scores": per_map_scores,
                },
            )
        )

    input_paths = {
        "promotion_audit": promotion_audit_path,
        "strategy_report": strategy_report_path,
    }
    checked_inputs = {
        artifact_type: path
        for artifact_type, path in input_paths.items()
        if path is not None
    }
    if checked_inputs:
        missing_inputs = {
            artifact_type: path
            for artifact_type, path in checked_inputs.items()
            if not artifact_index_contains_path(artifact_index, path)
        }
        checks.append(
            check_result(
                "artifact_index_contains_input_artifacts",
                not missing_inputs,
                {
                    "checked_inputs": checked_inputs,
                    "missing_inputs": missing_inputs,
                },
            )
        )

    if require_replay_analysis:
        checks.append(
            check_result(
                "replay_analysis_has_combat",
                counts.get("replay_analysis", 0) > 0
                and replay_combat["combat_replay_count"] > 0,
                {
                    "replay_analysis_count": counts.get("replay_analysis", 0),
                    **replay_combat,
                },
            )
        )
        checks.append(
            check_result(
                "no_replay_bad_strategy_issues",
                not replay_strategy_issues,
                {
                    "issue_count": len(replay_strategy_issues),
                    "issues": replay_strategy_issues,
                },
            )
        )

    if min_replay_combat_maps > 0:
        required_map_set = set(required_maps)
        eligible_combat_maps = [
            map_name
            for map_name in replay_combat["combat_maps"]
            if not required_map_set or map_name in required_map_set
        ]
        ignored_combat_maps = [
            map_name
            for map_name in replay_combat["combat_maps"]
            if required_map_set and map_name not in required_map_set
        ]
        checks.append(
            check_result(
                "replay_combat_map_coverage",
                len(eligible_combat_maps) >= min_replay_combat_maps,
                {
                    **replay_combat,
                    "eligible_combat_maps": eligible_combat_maps,
                    "ignored_combat_maps": ignored_combat_maps,
                    "required_maps": list(required_maps),
                    "min_replay_combat_maps": min_replay_combat_maps,
                },
            )
        )

    if min_opponent_historical_samples > 0:
        checks.append(
            check_result(
                "candidate_historical_opponent_samples",
                isinstance(opponent_historical_samples, int)
                and opponent_historical_samples >= min_opponent_historical_samples,
                {
                    "min_opponent_historical_samples": (
                        min_opponent_historical_samples
                    ),
                    "opponent_pool": {
                        "historical_samples": opponent_historical_samples,
                    }
                    if isinstance(opponent_pool_metadata, dict)
                    else None,
                    "historical_samples": opponent_historical_samples,
                    "metadata_present": isinstance(candidate_metadata, dict),
                    "opponent_pool_metadata_present": isinstance(
                        opponent_pool_metadata,
                        dict,
                    ),
                },
            )
        )

    if min_eval_episodes > 0:
        checks.append(
            check_result(
                "minimum_rank_eval_episodes",
                eval_episode_counts["candidate_baseline_episodes"]
                >= min_eval_episodes,
                {
                    **eval_episode_counts,
                    "min_eval_episodes": min_eval_episodes,
                },
            )
        )

    if min_head_to_head_episodes > 0:
        checks.append(
            check_result(
                "minimum_head_to_head_episodes",
                eval_episode_counts["head_to_head_episodes"]
                >= min_head_to_head_episodes,
                {
                    **eval_episode_counts,
                    "min_head_to_head_episodes": min_head_to_head_episodes,
                },
            )
        )

    if min_head_to_head_map_episodes is not None:
        head_to_head_map_episodes = eval_episode_counts[
            "head_to_head_map_episodes"
        ]
        target_maps = required_maps or tuple(sorted(head_to_head_map_episodes))
        low_head_to_head_maps = [
            {
                "map_name": map_name,
                "episode_count": head_to_head_map_episodes.get(map_name, 0),
            }
            for map_name in target_maps
            if head_to_head_map_episodes.get(map_name, 0)
            < min_head_to_head_map_episodes
        ]
        checks.append(
            check_result(
                "head_to_head_min_map_episodes",
                bool(target_maps) and not low_head_to_head_maps,
                {
                    **eval_episode_counts,
                    "required_maps": list(required_maps),
                    "target_maps": list(target_maps),
                    "min_head_to_head_map_episodes": (
                        min_head_to_head_map_episodes
                    ),
                    "low_head_to_head_maps": low_head_to_head_maps,
                },
            )
        )

    if min_map_episodes is not None:
        low_episode_maps = [
            item
            for item in per_map_scores
            if item["episode_count"] < min_map_episodes
        ]
        checks.append(
            check_result(
                "candidate_min_map_episodes",
                bool(per_map_scores) and not low_episode_maps,
                {
                    "min_map_episodes": min_map_episodes,
                    "per_map_scores": per_map_scores,
                    "low_episode_maps": low_episode_maps,
                },
            )
        )

    if min_map_score is not None:
        low_score_maps = [
            item
            for item in per_map_scores
            if item["mean_score"] < min_map_score
        ]
        checks.append(
            check_result(
                "candidate_min_map_score",
                bool(per_map_scores) and not low_score_maps,
                {
                    "min_map_score": min_map_score,
                    "per_map_scores": per_map_scores,
                    "low_score_maps": low_score_maps,
                },
            )
        )

    if require_candidate_checkpoint:
        candidate_checkpoint_exists = bool(
            candidate_checkpoint and Path(candidate_checkpoint).exists()
        )
        checks.append(
            check_result(
                "candidate_checkpoint_exists",
                candidate_checkpoint_exists,
                {
                    "checkpoint": candidate_checkpoint,
                },
            )
        )

    if require_candidate_metadata:
        checks.append(
            check_result(
                "candidate_checkpoint_metadata_exists",
                isinstance(candidate_metadata, dict),
                {
                    "checkpoint": candidate_checkpoint,
                    "metadata_present": isinstance(candidate_metadata, dict),
                    "metadata_keys": (
                        sorted(candidate_metadata)
                        if isinstance(candidate_metadata, dict)
                        else []
                    ),
                    "metadata_error": candidate_metadata_error,
                },
            )
        )

    if require_candidate_integrity:
        checks.append(
            check_result(
                "candidate_checkpoint_integrity",
                bool(candidate_integrity.get("passed")),
                candidate_integrity,
            )
        )

    if require_candidate_metadata and required_maps:
        metadata_maps = checkpoint_metadata_maps(candidate_metadata)
        missing_metadata_maps = missing_required_maps(metadata_maps, required_maps)
        checks.append(
            check_result(
                "candidate_metadata_required_maps",
                isinstance(candidate_metadata, dict) and not missing_metadata_maps,
                {
                    "metadata_maps": metadata_maps,
                    "required_maps": list(required_maps),
                    "missing_maps": missing_metadata_maps,
                },
            )
        )

    if required_curriculum_stage is not None:
        curriculum = candidate_metadata.get("curriculum") if isinstance(
            candidate_metadata, dict
        ) else None
        stage = (curriculum or {}).get("stage") or {}
        actual_stage = stage.get("name")
        checks.append(
            check_result(
                "candidate_metadata_curriculum_stage",
                actual_stage == required_curriculum_stage,
                {
                    "required_curriculum_stage": required_curriculum_stage,
                    "actual_curriculum_stage": actual_stage,
                    "metadata_present": isinstance(candidate_metadata, dict),
                },
            )
        )

    if required_reward_preset is not None:
        curriculum = candidate_metadata.get("curriculum") if isinstance(
            candidate_metadata, dict
        ) else None
        actual_reward_preset = (curriculum or {}).get("active_reward_preset")
        checks.append(
            check_result(
                "candidate_metadata_reward_preset",
                actual_reward_preset == required_reward_preset,
                {
                    "required_reward_preset": required_reward_preset,
                    "actual_reward_preset": actual_reward_preset,
                    "metadata_present": isinstance(candidate_metadata, dict),
                },
            )
        )

    if rank_summary is None:
        checks.append(
            check_result(
                "head_to_head_candidate_not_worse",
                False,
                {"reason": "missing_rank_artifact"},
                required=require_head_to_head,
            )
        )
    else:
        head_to_head = rank_summary.get("head_to_head", {})
        standings = head_to_head.get("standings", [])
        if standings:
            checks.append(
                check_result(
                    "head_to_head_candidate_not_worse",
                    standings[0].get("label") == candidate_label,
                    {
                        "candidate_label": candidate_label,
                        "top_head_to_head_label": standings[0].get("label"),
                    },
                )
            )
        else:
            checks.append(
                check_result(
                "head_to_head_candidate_not_worse",
                False,
                {
                    "reason": head_to_head.get(
                        "skipped",
                        "missing_head_to_head_standings",
                    ),
                },
                required=require_head_to_head,
            )
        )

    passed = all(check["passed"] for check in checks if check["required"])
    return {
        "artifact": artifact_metadata("long_run_check"),
        "passed": passed,
        "candidate": {
            "label": candidate_label,
            "checkpoint": candidate.get("checkpoint"),
            "score": candidate.get("score"),
            "rank": candidate.get("rank"),
        },
        "check_config": {
            "min_maps": min_maps,
            "required_maps": list(required_maps),
            "min_eval_episodes": min_eval_episodes,
            "min_map_episodes": min_map_episodes,
            "min_map_score": min_map_score,
            "require_replay_analysis": require_replay_analysis,
            "min_replay_combat_maps": min_replay_combat_maps,
            "min_opponent_historical_samples": min_opponent_historical_samples,
            "min_head_to_head_episodes": min_head_to_head_episodes,
            "min_head_to_head_map_episodes": min_head_to_head_map_episodes,
            "require_candidate_checkpoint": require_candidate_checkpoint,
            "require_candidate_metadata": require_candidate_metadata,
            "require_candidate_integrity": require_candidate_integrity,
            "required_curriculum_stage": required_curriculum_stage,
            "required_reward_preset": required_reward_preset,
            "require_head_to_head": require_head_to_head,
        },
        "checks": checks,
    }


def build_long_run_input_error_check(
    input_errors: list[dict],
    *,
    min_maps: int,
    required_maps: tuple[str, ...],
    min_eval_episodes: int,
    min_map_episodes: int | None,
    min_map_score: float | None = None,
    require_replay_analysis: bool,
    min_replay_combat_maps: int = 0,
    min_opponent_historical_samples: int = 0,
    require_candidate_checkpoint: bool = False,
    require_candidate_metadata: bool = False,
    require_candidate_integrity: bool = False,
    required_curriculum_stage: str | None = None,
    required_reward_preset: str | None = None,
    require_head_to_head: bool = False,
    min_head_to_head_episodes: int = 0,
    min_head_to_head_map_episodes: int | None = None,
) -> dict:
    return {
        "artifact": artifact_metadata("long_run_check"),
        "passed": False,
        "candidate": {
            "label": None,
            "checkpoint": None,
            "score": None,
            "rank": None,
        },
        "check_config": {
            "min_maps": min_maps,
            "required_maps": list(required_maps),
            "min_eval_episodes": min_eval_episodes,
            "min_map_episodes": min_map_episodes,
            "min_map_score": min_map_score,
            "require_replay_analysis": require_replay_analysis,
            "min_replay_combat_maps": min_replay_combat_maps,
            "min_opponent_historical_samples": min_opponent_historical_samples,
            "min_head_to_head_episodes": min_head_to_head_episodes,
            "min_head_to_head_map_episodes": min_head_to_head_map_episodes,
            "require_candidate_checkpoint": require_candidate_checkpoint,
            "require_candidate_metadata": require_candidate_metadata,
            "require_candidate_integrity": require_candidate_integrity,
            "required_curriculum_stage": required_curriculum_stage,
            "required_reward_preset": required_reward_preset,
            "require_head_to_head": require_head_to_head,
        },
        "checks": [
            check_result(
                "input_artifacts_loadable",
                False,
                {"errors": input_errors},
            )
        ],
    }


def load_long_run_check_input(
    name: str,
    path: str,
    expected_artifact_type: str,
) -> tuple[dict | None, dict | None]:
    try:
        summary = load_eval_summary(path)
        validate_artifact(summary, expected_artifact_type)
    except json.JSONDecodeError as exc:
        return None, {
            "name": name,
            "path": path,
            "error_type": type(exc).__name__,
            "message": f"{exc.msg} at line {exc.lineno} column {exc.colno}",
        }
    except (OSError, ValueError) as exc:
        return None, {
            "name": name,
            "path": path,
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
    return summary, None


def run_long_run_check(
    promotion_audit_path: str,
    strategy_report_path: str,
    artifact_index_path: str,
    min_maps: int,
    required_maps: tuple[str, ...],
    min_eval_episodes: int,
    min_map_score: float | None,
    require_replay_analysis: bool,
    min_map_episodes: int | None = None,
    min_replay_combat_maps: int = 0,
    min_opponent_historical_samples: int = 0,
    require_candidate_checkpoint: bool = False,
    require_candidate_metadata: bool = False,
    require_candidate_integrity: bool = False,
    required_curriculum_stage: str | None = None,
    required_reward_preset: str | None = None,
    require_head_to_head: bool = False,
    min_head_to_head_episodes: int = 0,
    min_head_to_head_map_episodes: int | None = None,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    input_specs = (
        ("promotion_audit", promotion_audit_path, "promotion_audit"),
        ("strategy_report", strategy_report_path, "strategy_report"),
        ("artifact_index", artifact_index_path, "artifact_index"),
    )
    summaries = {}
    input_errors = []
    for name, path, expected_type in input_specs:
        summary, error = load_long_run_check_input(name, path, expected_type)
        if error is not None:
            input_errors.append(error)
        else:
            summaries[name] = summary

    if input_errors:
        result = build_long_run_input_error_check(
            input_errors,
            min_maps=min_maps,
            required_maps=required_maps,
            min_eval_episodes=min_eval_episodes,
            min_map_episodes=min_map_episodes,
            min_map_score=min_map_score,
            require_replay_analysis=require_replay_analysis,
            min_replay_combat_maps=min_replay_combat_maps,
            min_opponent_historical_samples=min_opponent_historical_samples,
            require_candidate_checkpoint=require_candidate_checkpoint,
            require_candidate_metadata=require_candidate_metadata,
            require_candidate_integrity=require_candidate_integrity,
            required_curriculum_stage=required_curriculum_stage,
            required_reward_preset=required_reward_preset,
            require_head_to_head=require_head_to_head,
            min_head_to_head_episodes=min_head_to_head_episodes,
            min_head_to_head_map_episodes=min_head_to_head_map_episodes,
        )
    else:
        result = build_long_run_check(
            summaries["promotion_audit"],
            summaries["strategy_report"],
            summaries["artifact_index"],
            promotion_audit_path=promotion_audit_path,
            strategy_report_path=strategy_report_path,
            min_maps=min_maps,
            required_maps=required_maps,
            min_eval_episodes=min_eval_episodes,
            min_map_episodes=min_map_episodes,
            min_map_score=min_map_score,
            require_replay_analysis=require_replay_analysis,
            min_replay_combat_maps=min_replay_combat_maps,
            min_opponent_historical_samples=min_opponent_historical_samples,
            require_candidate_checkpoint=require_candidate_checkpoint,
            require_candidate_metadata=require_candidate_metadata,
            require_candidate_integrity=require_candidate_integrity,
            required_curriculum_stage=required_curriculum_stage,
            required_reward_preset=required_reward_preset,
            require_head_to_head=require_head_to_head,
            min_head_to_head_episodes=min_head_to_head_episodes,
            min_head_to_head_map_episodes=min_head_to_head_map_episodes,
        )
    result["inputs"] = {
        "promotion_audit": promotion_audit_path,
        "strategy_report": strategy_report_path,
        "artifact_index": artifact_index_path,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or "long-run-check"
        path = write_eval_summary(result, output_dir, label=label)
        print(f"Saved long-run check to {path}")
    if not result["passed"]:
        sys.exit(1)


def failed_required_check_ids(long_run_check: dict) -> list[str]:
    return [
        check.get("id")
        for check in long_run_check.get("checks", [])
        if check.get("required", True) and not check.get("passed")
    ]


def _path_is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _manifest_sidecar_path(manifest_path: Path, path_text: str | None) -> str | None:
    if not isinstance(path_text, str) or not path_text:
        return None

    raw_path = Path(path_text)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        candidates.append(manifest_path.parent / raw_path)

    for candidate in candidates:
        if _path_is_relative_to(candidate, manifest_path.parent):
            return str(candidate)
    return None


def _existing_launcher_path(manifest_path: Path, manifest: dict) -> str | None:
    explicit_path = _manifest_sidecar_path(
        manifest_path,
        manifest.get("shell_script_path"),
    )
    if explicit_path:
        return explicit_path
    sibling_path = manifest_path.with_suffix(".sh")
    if sibling_path.exists():
        return str(sibling_path)
    return None


def _existing_preflight_launcher_path(
    manifest_path: Path,
    manifest: dict,
) -> str | None:
    explicit_path = _manifest_sidecar_path(
        manifest_path,
        manifest.get("preflight_shell_script_path"),
    )
    if explicit_path:
        return explicit_path
    sibling_path = manifest_path.with_suffix(".preflight.sh")
    if sibling_path.exists():
        return str(sibling_path)
    return None


def _directory_file_count(path: Path | None) -> int:
    if path is None or not path.exists():
        return 0
    return sum(1 for child in path.iterdir() if child.is_file())


def _checkpoint_artifact_paths(path: Path | None) -> tuple[Path, ...]:
    if path is None or not path.exists():
        return ()
    return tuple(
        sorted(
            child
            for child in path.iterdir()
            if child.is_file()
            and not child.name.endswith(".meta.json")
            and (
                child.suffix == ".zip"
                or (child.suffix == "" and child.name.startswith("ppo_"))
            )
        )
    )


def _is_valid_training_replay_file(path: Path) -> bool:
    if path.suffix != ".json" or not path.name.startswith("episode_"):
        return False
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, dict) and isinstance(data.get("frames"), list)


def _training_replay_paths(path: Path | None) -> tuple[Path, ...]:
    if path is None or not path.exists():
        return ()
    return tuple(
        sorted(
            child
            for child in path.iterdir()
            if child.is_file() and _is_valid_training_replay_file(child)
        )
    )


def checkpoint_opponent_pool_status(
    checkpoint_path: Path | None,
    min_historical_samples: int = 0,
) -> dict:
    metadata_paths = (
        sorted(checkpoint_path.glob("*.meta.json"))
        if checkpoint_path is not None and checkpoint_path.exists()
        else []
    )
    best_metadata = None
    load_errors = []
    metadata_with_opponent_pool = 0
    metadata_with_historical_samples = 0
    metadata_meeting_min = 0
    max_historical_samples = None

    for metadata_path in metadata_paths:
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            load_errors.append(
                {
                    "path": str(metadata_path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        if not isinstance(metadata, dict):
            continue

        opponent_pool = metadata.get("opponent_pool")
        if not isinstance(opponent_pool, dict):
            continue
        metadata_with_opponent_pool += 1

        historical_samples = json_non_negative_int(
            opponent_pool.get("historical_samples")
        )
        if historical_samples is not None:
            metadata_with_historical_samples += 1
            if max_historical_samples is None:
                max_historical_samples = historical_samples
            else:
                max_historical_samples = max(
                    max_historical_samples,
                    historical_samples,
                )
            if historical_samples >= min_historical_samples:
                metadata_meeting_min += 1

        checkpoint_file = metadata.get("checkpoint_file")
        entry = {
            "path": str(metadata_path),
            "checkpoint_file": (
                checkpoint_file.get("file_name")
                if isinstance(checkpoint_file, dict)
                else None
            ),
            "num_timesteps": json_non_negative_int(metadata.get("num_timesteps")),
            "pool_size": json_non_negative_int(opponent_pool.get("size")),
            "historical_samples": historical_samples,
        }
        best_key = (
            historical_samples if historical_samples is not None else -1,
            entry["num_timesteps"] if entry["num_timesteps"] is not None else -1,
            str(metadata_path),
        )
        if best_metadata is None or best_key > best_metadata["_key"]:
            best_metadata = {**entry, "_key": best_key}

    if best_metadata is not None:
        best_metadata.pop("_key", None)

    meets_min = (
        True
        if min_historical_samples <= 0
        else (
            max_historical_samples is not None
            and max_historical_samples >= min_historical_samples
        )
    )
    return {
        "metadata_file_count": len(metadata_paths),
        "metadata_with_opponent_pool_count": metadata_with_opponent_pool,
        "metadata_with_historical_samples_count": metadata_with_historical_samples,
        "metadata_meeting_min_count": metadata_meeting_min,
        "metadata_load_error_count": len(load_errors),
        "metadata_load_errors": load_errors[:5],
        "min_opponent_historical_samples": min_historical_samples,
        "max_historical_samples": max_historical_samples,
        "meets_min_opponent_historical_samples": meets_min,
        "best_checkpoint_metadata": best_metadata,
    }


def _manifest_source_status(manifest_source: dict, current_source: dict) -> dict:
    source_commit = manifest_source.get("commit")
    current_commit = current_source.get("commit")
    manifest_dirty = manifest_source.get("dirty")
    current_dirty = current_source.get("dirty")
    source_available = bool(manifest_source.get("available", source_commit is not None))
    current_available = bool(current_source.get("available"))
    commit_matches_current = None
    if source_commit is not None and current_commit is not None:
        commit_matches_current = source_commit == current_commit

    stale_reasons = []
    if not source_available or not current_available:
        stale_reasons.append("source_unavailable")
    if commit_matches_current is False:
        stale_reasons.append("commit_mismatch")
    if manifest_dirty is True:
        stale_reasons.append("manifest_created_from_dirty_worktree")
    if current_dirty is True:
        stale_reasons.append("current_worktree_dirty")

    return {
        "source_current_commit": current_commit,
        "source_current_dirty": current_dirty,
        "source_commit_matches_current": commit_matches_current,
        "source_manifest_clean": manifest_dirty is False,
        "source_current_clean": current_dirty is False,
        "source_safe_to_launch": not stale_reasons,
        "source_stale_reasons": stale_reasons,
    }


def _preflight_self_play_sampling_status(preflight_dir: str | None) -> dict:
    status = {
        "available": False,
        "path": None,
        "passed": None,
        "historical_samples": None,
        "historical_sample_rate": None,
        "latest_samples": None,
        "unique_maps_seen": None,
        "failed_checks": [],
    }
    if not preflight_dir:
        return status

    path = Path(preflight_dir) / "self-play-sampling-summary.json"
    status["path"] = str(path)
    if not path.exists():
        return status

    try:
        summary = load_eval_summary(path)
        validate_artifact(summary, "self_play_sampling_smoke")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        status["load_error"] = f"{type(exc).__name__}: {exc}"
        return status

    status.update(
        {
            "available": True,
            "passed": summary.get("passed"),
            "historical_samples": summary.get("historical_samples"),
            "historical_sample_rate": summary.get("historical_sample_rate"),
            "latest_samples": summary.get("latest_samples"),
            "unique_maps_seen": summary.get("unique_maps_seen"),
            "failed_checks": failed_smoke_check_ids(summary),
        }
    )
    return status


def _manifest_status_entry(
    manifest_path: Path,
    manifest: dict,
    current_source: dict,
) -> dict:
    cfg = manifest.get("manifest_config", {})
    eval_dir = cfg.get("eval_dir")
    eval_path = Path(eval_dir) if eval_dir else None
    checkpoint_dir = cfg.get("checkpoint_dir")
    checkpoint_path = Path(checkpoint_dir) if checkpoint_dir else None
    replay_dir = cfg.get("replay_dir")
    replay_path = Path(replay_dir) if replay_dir else None
    preflight_dir = cfg.get("preflight_dir")
    preflight_self_play_sampling = _preflight_self_play_sampling_status(
        preflight_dir if isinstance(preflight_dir, str) else None
    )
    launcher_path = _existing_launcher_path(manifest_path, manifest)
    preflight_launcher_path = _existing_preflight_launcher_path(
        manifest_path,
        manifest,
    )
    source_control = cfg.get("source_control", {})
    command_ids = {
        command.get("id")
        for command in manifest.get("commands", [])
        if isinstance(command, dict)
    }
    expects_self_play_sampling_preflight = (
        "self_play_sampling_smoke_preflight" in command_ids
    )
    source_status = _manifest_source_status(source_control, current_source)
    min_opponent_historical_samples = json_non_negative_int(
        cfg.get("min_opponent_historical_samples")
    )
    checkpoint_pool_status = checkpoint_opponent_pool_status(
        checkpoint_path,
        min_historical_samples=min_opponent_historical_samples or 0,
    )
    checkpoint_files = _checkpoint_artifact_paths(checkpoint_path)
    replay_files = _training_replay_paths(replay_path)
    return {
        "path": str(manifest_path),
        "run_id": cfg.get("run_id"),
        "timesteps": cfg.get("timesteps"),
        "eval_dir": eval_dir,
        "eval_dir_exists": eval_path.exists() if eval_path else False,
        "eval_file_count": _directory_file_count(eval_path),
        "expects_self_play_sampling_preflight": expects_self_play_sampling_preflight,
        "self_play_sampling_preflight_exitcode_exists": (
            (eval_path / "self-play-sampling-preflight.exitcode").exists()
            if eval_path
            else False
        ),
        "self_play_sampling_preflight": preflight_self_play_sampling,
        "preflight_exitcode_exists": (
            (eval_path / "preflight.exitcode").exists() if eval_path else False
        ),
        "train_exitcode_exists": (
            (eval_path / "train.exitcode").exists() if eval_path else False
        ),
        "promotion_audit_exitcode_exists": (
            (eval_path / "promotion-audit.exitcode").exists() if eval_path else False
        ),
        "long_run_check_exitcode_exists": (
            (eval_path / "long-run-check.exitcode").exists() if eval_path else False
        ),
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_dir_exists": checkpoint_path.exists() if checkpoint_path else False,
        "checkpoint_file_count": len(checkpoint_files),
        "checkpoint_total_file_count": _directory_file_count(checkpoint_path),
        "replay_dir": replay_dir,
        "replay_dir_exists": replay_path.exists() if replay_path else False,
        "replay_file_count": len(replay_files),
        "replay_total_file_count": _directory_file_count(replay_path),
        "preflight_dir": preflight_dir,
        "preflight_dir_exists": (
            Path(preflight_dir).exists() if preflight_dir else False
        ),
        "launcher_path": launcher_path,
        "launcher_exists": Path(launcher_path).exists() if launcher_path else False,
        "preflight_launcher_path": preflight_launcher_path,
        "preflight_launcher_exists": (
            Path(preflight_launcher_path).exists()
            if preflight_launcher_path
            else False
        ),
        "required_maps": cfg.get("required_maps", []),
        "min_eval_episodes": cfg.get("min_eval_episodes"),
        "min_map_episodes": cfg.get("min_map_episodes"),
        "min_replay_combat_maps": cfg.get("min_replay_combat_maps"),
        "min_opponent_historical_samples": min_opponent_historical_samples,
        "require_head_to_head": cfg.get("require_head_to_head"),
        "min_head_to_head_episodes": cfg.get("min_head_to_head_episodes"),
        "min_head_to_head_map_episodes": cfg.get("min_head_to_head_map_episodes"),
        "checkpoint_opponent_pool": checkpoint_pool_status,
        "source_commit": source_control.get("commit"),
        "require_candidate_integrity": cfg.get("require_candidate_integrity"),
        "source_dirty": source_control.get("dirty"),
        "source_status_short_count": source_control.get("status_short_count"),
        **source_status,
    }


def _long_run_check_status_entry(check_path: Path, check: dict) -> dict:
    return {
        "path": str(check_path),
        "passed": bool(check.get("passed")),
        "candidate": check.get("candidate", {}),
        "failed_required_checks": failed_required_check_ids(check),
    }


def long_run_missing_evidence(
    latest_manifest: dict | None,
    latest_checks: list[dict],
    latest_passing_checks: list[dict],
) -> list[str]:
    if latest_manifest is None:
        return ["long_run_manifest"]

    missing = []
    if not latest_manifest.get("launcher_exists"):
        missing.append("long_run_launcher")
    if not latest_manifest.get("preflight_launcher_exists"):
        missing.append("long_run_preflight_launcher")
    if (
        latest_manifest.get("expects_self_play_sampling_preflight")
        and not latest_manifest.get("self_play_sampling_preflight_exitcode_exists")
    ):
        missing.append("self_play_sampling_preflight_exitcode")
    if latest_manifest.get("expects_self_play_sampling_preflight"):
        preflight_sampling = latest_manifest.get("self_play_sampling_preflight") or {}
        if not preflight_sampling.get("available"):
            missing.append("self_play_sampling_preflight_summary")
    if not latest_manifest.get("preflight_exitcode_exists"):
        missing.append("preflight_exitcode")
    if not latest_manifest.get("train_exitcode_exists"):
        missing.append("train_exitcode")
    if not latest_manifest.get("promotion_audit_exitcode_exists"):
        missing.append("promotion_audit_exitcode")
    if not latest_manifest.get("long_run_check_exitcode_exists"):
        missing.append("long_run_check_exitcode")
    if latest_manifest.get("checkpoint_file_count", 0) == 0:
        missing.append("candidate_checkpoint_files")
    if latest_manifest.get("replay_file_count", 0) == 0:
        missing.append("real_training_replay_files")
    if (
        json_non_negative_int(latest_manifest.get("min_opponent_historical_samples"))
        or 0
    ) > 0:
        checkpoint_pool_status = latest_manifest.get("checkpoint_opponent_pool") or {}
        if not checkpoint_pool_status.get("meets_min_opponent_historical_samples"):
            missing.append("checkpoint_historical_opponent_samples")
    if not latest_checks:
        missing.append("latest_run_long_run_check")
    elif not latest_passing_checks:
        missing.append("passing_latest_long_run_check")
        failed_ids = sorted(
            {
                failed_check
                for check in latest_checks
                for failed_check in check.get("failed_required_checks", [])
                if failed_check
            }
        )
        missing.extend(f"failed_check:{failed_id}" for failed_id in failed_ids)
    return missing


def build_long_run_status(
    artifact_dir: str | Path,
    *,
    recursive: bool = True,
) -> dict:
    root = Path(artifact_dir)
    index = build_artifact_index(root, recursive=recursive)
    current_source = source_control_snapshot()
    manifests = []
    long_run_checks = []

    for entry in index.get("artifacts", []):
        artifact_path = Path(entry["path"])
        if entry.get("artifact_type") == "long_run_manifest":
            try:
                manifest = load_eval_summary(artifact_path)
                validate_artifact(manifest, "long_run_manifest")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                manifests.append(
                    {
                        "path": str(artifact_path),
                        "load_error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            status_entry = _manifest_status_entry(
                artifact_path,
                manifest,
                current_source,
            )
            status_entry["mtime"] = artifact_path.stat().st_mtime
            manifests.append(status_entry)
        elif entry.get("artifact_type") == "long_run_check":
            try:
                check = load_eval_summary(artifact_path)
                validate_artifact(check, "long_run_check")
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            status_entry = _long_run_check_status_entry(artifact_path, check)
            status_entry["mtime"] = artifact_path.stat().st_mtime
            long_run_checks.append(status_entry)

    manifests.sort(key=lambda item: (item.get("mtime", 0), item.get("path", "")))
    long_run_checks.sort(key=lambda item: (item.get("mtime", 0), item["path"]))
    latest_manifest = manifests[-1] if manifests else None

    latest_checks = []
    if latest_manifest and latest_manifest.get("eval_dir"):
        latest_eval_dir = Path(latest_manifest["eval_dir"])
        latest_checks = [
            check
            for check in long_run_checks
            if _path_is_relative_to(Path(check["path"]), latest_eval_dir)
        ]

    passing_checks = [check for check in long_run_checks if check["passed"]]
    latest_passing_checks = [check for check in latest_checks if check["passed"]]
    missing_evidence = long_run_missing_evidence(
        latest_manifest,
        latest_checks,
        latest_passing_checks,
    )

    blocked_reason = None
    next_command = None
    next_preflight_command = None
    if latest_manifest is None:
        blocked_reason = "no_long_run_manifest_found"
    elif not latest_manifest.get("eval_dir_exists"):
        blocked_reason = "latest_launcher_not_executed"
    elif not latest_checks and not latest_manifest.get("train_exitcode_exists"):
        blocked_reason = "latest_preflight_only"
    elif not latest_checks:
        blocked_reason = "latest_long_run_check_missing"
    elif not latest_passing_checks:
        blocked_reason = "latest_long_run_check_not_passing"
    if (
        latest_manifest is not None
        and not latest_passing_checks
        and latest_manifest.get("source_safe_to_launch") is False
    ):
        blocked_reason = "latest_manifest_source_stale"

    if (
        latest_manifest
        and blocked_reason
        in {"latest_launcher_not_executed", "latest_preflight_only"}
        and latest_manifest.get("source_safe_to_launch") is not False
        and latest_manifest.get("launcher_exists")
    ):
        next_command = f"bash {_shell_arg(latest_manifest['launcher_path'])}"
    if (
        latest_manifest
        and blocked_reason == "latest_launcher_not_executed"
        and not latest_manifest.get("preflight_dir_exists")
        and latest_manifest.get("source_safe_to_launch") is not False
        and latest_manifest.get("preflight_launcher_exists")
    ):
        next_preflight_command = (
            f"bash {_shell_arg(latest_manifest['preflight_launcher_path'])}"
        )

    latest_manifest_summary = dict(latest_manifest) if latest_manifest else None
    if latest_manifest_summary is not None:
        latest_manifest_summary.pop("mtime", None)
        latest_manifest_summary["long_run_check_count"] = len(latest_checks)
        latest_manifest_summary["passing_long_run_check_count"] = len(
            latest_passing_checks
        )
        latest_manifest_summary["latest_long_run_check"] = (
            {key: value for key, value in latest_checks[-1].items() if key != "mtime"}
            if latest_checks
            else None
        )

    return {
        "artifact": artifact_metadata("long_run_status"),
        "status_config": {
            "artifact_dir": str(root),
            "recursive": recursive,
            "source_control": current_source,
        },
        "manifest_count": len(manifests),
        "long_run_check_count": len(long_run_checks),
        "passing_long_run_check_count": len(passing_checks),
        "latest_manifest": latest_manifest_summary,
        "passing_long_run_checks": [
            {key: value for key, value in check.items() if key != "mtime"}
            for check in passing_checks
        ],
        "candidate_evidence_ready": bool(latest_passing_checks),
        "missing_evidence": missing_evidence,
        "blocked_reason": blocked_reason,
        "next_command": next_command,
        "next_preflight_command": next_preflight_command,
    }


def run_long_run_status(
    artifact_dir: str,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    status = build_long_run_status(artifact_dir, recursive=True)
    print(json.dumps(status, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or "long-run-status"
        path = write_eval_summary(status, output_dir, label=label)
        print(f"Saved long-run status to {path}")


def _latest_artifact_entry(
    index: dict,
    artifact_type: str,
    *,
    scope_dir: str | Path | None = None,
) -> dict | None:
    entries = [
        entry
        for entry in index.get("artifacts", [])
        if entry.get("artifact_type") == artifact_type
    ]
    if scope_dir is not None:
        scope_path = Path(scope_dir)
        entries = [
            entry
            for entry in entries
            if _path_is_relative_to(Path(entry["path"]), scope_path)
        ]
    if not entries:
        return None
    return max(
        entries,
        key=lambda entry: (
            Path(entry["path"]).stat().st_mtime,
            entry.get("relative_path") or entry["path"],
        ),
    )


def _load_indexed_artifact(entry: dict | None, expected_type: str) -> dict | None:
    if entry is None:
        return None
    try:
        artifact = load_eval_summary(entry["path"])
        validate_artifact(artifact, expected_type)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return artifact


def _candidate_head_to_head_signal(rank: dict | None) -> dict:
    if not isinstance(rank, dict):
        return {
            "available": False,
            "candidate_label": None,
            "candidate_elo": None,
            "candidate_score": None,
            "standing_rank": None,
        }
    rankings = rank.get("rankings", [])
    if not isinstance(rankings, list):
        rankings = []
    top_ranking = rankings[0] if rankings and isinstance(rankings[0], dict) else {}
    candidate_label = top_ranking.get("label")
    head_to_head = rank.get("head_to_head", {})
    if not isinstance(head_to_head, dict):
        head_to_head = {}
    standings = head_to_head.get("standings", [])
    if not isinstance(standings, list):
        standings = []
    candidate_standing = None
    for index, standing in enumerate(standings, start=1):
        if not isinstance(standing, dict):
            continue
        if standing.get("label") == candidate_label:
            candidate_standing = standing
            standing_rank = index
            break
    else:
        standing_rank = None
    overview = head_to_head.get("overview", {})
    if not isinstance(overview, dict):
        overview = {}
    return {
        "available": candidate_standing is not None,
        "candidate_label": candidate_label,
        "candidate_elo": (
            candidate_standing.get("elo") if candidate_standing else None
        ),
        "candidate_score": (
            candidate_standing.get("score") if candidate_standing else None
        ),
        "standing_rank": standing_rank,
        "total_episodes": overview.get("total_episodes"),
    }


def _candidate_rank_map_score_signal(rank_top: dict | None) -> dict:
    if not isinstance(rank_top, dict) or not rank_top:
        return {
            "available": False,
            "candidate_label": None,
            "map_count": 0,
            "per_map_scores": [],
            "invalid_map_scores": [],
            "worst": None,
        }

    per_map_scores, invalid_map_scores = ranking_per_map_score_details(rank_top)
    worst_map = min(
        per_map_scores,
        key=lambda item: (item["mean_score"], item["map_name"]),
        default=None,
    )
    return {
        "available": bool(per_map_scores or invalid_map_scores),
        "candidate_label": rank_top.get("label"),
        "map_count": len(per_map_scores),
        "per_map_scores": per_map_scores,
        "invalid_map_scores": invalid_map_scores,
        "worst": worst_map,
    }


def build_league_health_report(
    artifact_dir: str | Path,
    *,
    recursive: bool = True,
) -> dict:
    root = Path(artifact_dir)
    index = build_artifact_index(root, recursive=recursive)
    status_entry = _latest_artifact_entry(index, "long_run_status")
    status_artifact = _load_indexed_artifact(status_entry, "long_run_status")
    status = status_artifact or {}
    latest_manifest = status.get("latest_manifest") or {}
    if not isinstance(latest_manifest, dict):
        latest_manifest = {}
    artifact_scope_dir = latest_manifest.get("eval_dir")
    if not isinstance(artifact_scope_dir, str) or not artifact_scope_dir:
        artifact_scope_dir = None
    latest_entries = {"long_run_status": status_entry}
    for artifact_type in (
        "strategy_report",
        "rank",
        "promotion_audit",
        "long_run_check",
        "self_play_sampling_smoke",
    ):
        latest_entries[artifact_type] = _latest_artifact_entry(
            index,
            artifact_type,
            scope_dir=artifact_scope_dir,
        )
    latest = {
        artifact_type: _load_indexed_artifact(entry, artifact_type)
        for artifact_type, entry in latest_entries.items()
    }
    latest["long_run_status"] = status_artifact

    strategy = latest["strategy_report"] or {}
    rank = latest["rank"] or {}
    promotion = latest["promotion_audit"] or {}
    long_run_check = latest["long_run_check"] or {}

    strategy_issues = strategy.get("issues", [])
    if not isinstance(strategy_issues, list):
        strategy_issues = []
    candidate_issues = [
        issue
        for issue in strategy_issues
        if str(issue.get("scope", "")).startswith("candidate:")
    ]
    historical_sampling_issues = [
        issue
        for issue in strategy_issues
        if issue.get("metric") == "checkpoint_historical_opponent_samples"
    ]
    replay_strategy_issues = [
        issue
        for issue in strategy_issues
        if str(issue.get("scope", "")).startswith("replay:")
        and issue.get("metric") in REPLAY_STRATEGY_METRICS
    ]
    smoke_strategy_issues = [
        issue
        for issue in strategy_issues
        if str(issue.get("scope", "")).startswith("smoke:")
    ]
    strategy_skipped_artifacts = strategy.get("skipped_artifacts", [])
    if not isinstance(strategy_skipped_artifacts, list):
        strategy_skipped_artifacts = []
    weaknesses = strategy.get("weaknesses", [])
    if not isinstance(weaknesses, list):
        weaknesses = []
    weakness_maps = sorted(
        {weakness.get("map_name") for weakness in weaknesses if weakness.get("map_name")}
    )

    checkpoint_pool = latest_manifest.get("checkpoint_opponent_pool") or {}
    if not isinstance(checkpoint_pool, dict):
        checkpoint_pool = {}
    preflight_self_play_sampling = (
        latest_manifest.get("self_play_sampling_preflight") or {}
    )
    if not isinstance(preflight_self_play_sampling, dict):
        preflight_self_play_sampling = {}
    long_run_check_failed = failed_required_check_ids(long_run_check)
    self_play_sampling = latest["self_play_sampling_smoke"] or {}
    self_play_sampling_failed_checks = failed_smoke_check_ids(self_play_sampling)
    promotion_candidate = promotion.get("candidate") or {}
    rankings = rank.get("rankings") or []
    if not isinstance(rankings, list) or not rankings:
        rank_top = {}
    else:
        rank_top = rankings[0] if isinstance(rankings[0], dict) else {}
    candidate_label = (
        promotion_candidate.get("label")
        or rank_top.get("label")
        or long_run_check.get("candidate", {}).get("label")
    )

    blockers = []
    warnings = []
    if latest["strategy_report"] is None:
        warnings.append("missing_strategy_report")
    if latest["rank"] is None:
        warnings.append("missing_rank")
    if latest["promotion_audit"] is None:
        warnings.append("missing_promotion_audit")
    if latest["long_run_status"] is None:
        warnings.append("missing_long_run_status")
    if latest["long_run_check"] is None:
        warnings.append("missing_long_run_check")
    if strategy_skipped_artifacts:
        warnings.append("strategy_report_skipped_artifacts")
    if status and (
        status.get("blocked_reason") or status.get("candidate_evidence_ready") is False
    ):
        blockers.append("long_run_status_blocked")
    if promotion and promotion.get("passed") is False:
        blockers.append("promotion_audit_failed")
    if candidate_issues:
        blockers.append("candidate_strategy_issues")
    if replay_strategy_issues:
        blockers.append("replay_strategy_issues")
    if smoke_strategy_issues:
        blockers.append("smoke_strategy_issues")
    if (
        preflight_self_play_sampling.get("available")
        and preflight_self_play_sampling.get("passed") is False
    ):
        blockers.append("self_play_sampling_preflight_failed")
    if self_play_sampling and self_play_sampling.get("passed") is False:
        blockers.append("self_play_sampling_smoke_failed")
    missing_evidence = status.get("missing_evidence", [])
    if not isinstance(missing_evidence, list):
        missing_evidence = []
    if (
        historical_sampling_issues
        or "checkpoint_historical_opponent_samples" in missing_evidence
    ):
        blockers.append("historical_opponent_sampling")
    if long_run_check and not long_run_check.get("passed"):
        blockers.append("long_run_check_failed")

    source_artifacts = {
        artifact_type: entry.get("path") if entry else None
        for artifact_type, entry in latest_entries.items()
    }
    source_artifacts["self_play_sampling_preflight"] = (
        preflight_self_play_sampling.get("path")
    )
    return {
        "artifact": artifact_metadata("league_health"),
        "health_config": {
            "artifact_dir": str(root),
            "recursive": recursive,
            "artifact_scope_dir": artifact_scope_dir,
        },
        "source_artifacts": source_artifacts,
        "health": {
            "ready": not blockers and not warnings,
            "blockers": blockers,
            "warnings": warnings,
        },
        "signals": {
            "candidate": {
                "label": candidate_label,
                "promotion_passed": promotion.get("passed"),
                "rank_score": rank_top.get("score"),
            },
            "opponent_pool": {
                "historical_sample_ready": checkpoint_pool.get(
                    "meets_min_opponent_historical_samples"
                ),
                "max_historical_samples": checkpoint_pool.get(
                    "max_historical_samples"
                ),
                "min_historical_samples": checkpoint_pool.get(
                    "min_opponent_historical_samples"
                ),
            },
            "strategy": {
                "issue_count": strategy.get("issue_count", len(strategy_issues)),
                "candidate_issue_count": len(candidate_issues),
                "historical_sampling_issue_count": len(historical_sampling_issues),
                "replay_issue_count": len(replay_strategy_issues),
                "smoke_issue_count": len(smoke_strategy_issues),
                "skipped_artifact_count": len(strategy_skipped_artifacts),
                "skipped_artifacts": strategy_skipped_artifacts,
                "issue_metrics": sorted(
                    {
                        issue.get("metric")
                        for issue in strategy_issues
                        if issue.get("metric")
                    }
                ),
            },
            "map_weaknesses": {
                "count": strategy.get("weakness_count", len(weaknesses)),
                "reported_count": len(weaknesses),
                "maps": weakness_maps,
                "worst": weaknesses[0] if weaknesses else None,
            },
            "rank_map_scores": _candidate_rank_map_score_signal(rank_top),
            "head_to_head": _candidate_head_to_head_signal(rank),
            "self_play_sampling": {
                "available": bool(self_play_sampling),
                "passed": self_play_sampling.get("passed"),
                "historical_samples": self_play_sampling.get("historical_samples"),
                "historical_sample_rate": self_play_sampling.get(
                    "historical_sample_rate"
                ),
                "unique_maps_seen": self_play_sampling.get("unique_maps_seen"),
                "failed_checks": self_play_sampling_failed_checks,
            },
            "self_play_sampling_preflight": {
                "available": bool(preflight_self_play_sampling.get("available")),
                "passed": preflight_self_play_sampling.get("passed"),
                "historical_samples": preflight_self_play_sampling.get(
                    "historical_samples"
                ),
                "historical_sample_rate": preflight_self_play_sampling.get(
                    "historical_sample_rate"
                ),
                "latest_samples": preflight_self_play_sampling.get("latest_samples"),
                "unique_maps_seen": preflight_self_play_sampling.get(
                    "unique_maps_seen"
                ),
                "failed_checks": preflight_self_play_sampling.get(
                    "failed_checks", []
                ),
            },
            "long_run": {
                "status_candidate_evidence_ready": status.get(
                    "candidate_evidence_ready"
                ),
                "status_blocked_reason": status.get("blocked_reason"),
                "missing_evidence": missing_evidence,
                "latest_check_passed": long_run_check.get("passed"),
                "failed_required_checks": long_run_check_failed,
            },
        },
    }


def run_league_health(
    artifact_dir: str,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    report = build_league_health_report(artifact_dir, recursive=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    if output_dir is not None:
        label = output_label or "league-health"
        path = write_eval_summary(report, output_dir, label=label)
        print(f"Saved league health report to {path}")


def run_checkpoint_trust_manifest(
    checkpoint_dir: str,
    checkpoints: tuple[str, ...] | None,
    output_path: str,
) -> None:
    checkpoint_paths = checkpoints or discover_checkpoints(checkpoint_dir)
    if not checkpoint_paths:
        raise ValueError(f"No checkpoints found in {checkpoint_dir}")
    path = write_checkpoint_trust_manifest(tuple(checkpoint_paths), output_path)
    manifest = json.loads(path.read_text())
    print(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"Saved checkpoint trust manifest to {path}")


def _shell_assign(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def _shell_arg(value: object) -> str:
    return shlex.quote(str(value))


def _with_output_redirect(parts: list[str], output_path: str) -> list[str]:
    redirected = list(parts)
    redirected[-1] = f"{redirected[-1]} > {output_path} 2>&1"
    return redirected


def _run_git_metadata(args: list[str]) -> tuple[str | None, str | None]:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None, "git executable not found"
    try:
        result = subprocess.run(
            [git_executable, *args],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except OSError as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if result.returncode != 0:
        return None, result.stderr.strip() or f"git exited {result.returncode}"
    return result.stdout.strip(), None


def source_control_snapshot() -> dict:
    root, root_error = _run_git_metadata(["rev-parse", "--show-toplevel"])
    if root_error is not None:
        return {
            "vcs": "git",
            "available": False,
            "error": root_error,
        }

    commit, commit_error = _run_git_metadata(["rev-parse", "HEAD"])
    branch, branch_error = _run_git_metadata(["rev-parse", "--abbrev-ref", "HEAD"])
    status, status_error = _run_git_metadata(["status", "--short"])
    status_lines = status.splitlines() if status is not None else []
    return {
        "vcs": "git",
        "available": True,
        "root": root,
        "commit": commit,
        "commit_error": commit_error,
        "branch": branch,
        "branch_error": branch_error,
        "dirty": bool(status_lines) if status_error is None else None,
        "status_short_count": len(status_lines),
        "status_short_sample": status_lines[:20],
        "status_error": status_error,
    }


def build_long_run_manifest(
    *,
    run_id: str,
    checkpoint_root: str | Path = "checkpoints",
    eval_root: str | Path = "evals",
    replay_root: str | Path = "replays",
    timesteps: int = 5_000_000,
    suite_opponents: str = "idle,scripted,aggressive,evasive",
    suite_maps: str = "classic,flat,split,tower",
    rounds: int = 20,
    replay_samples_per_bucket: int = 2,
    replay_save_interval: int | None = None,
    opponent_pool_seed: int | None = None,
    rank_min_score: float = 0.1,
    rank_min_win_rate: float = 0.0,
    rank_max_draw_rate: float = 0.9,
    rank_max_no_damage_rate: float = 0.75,
    rank_max_low_engagement_rate: float = 0.5,
    rank_min_map_score: float | None = 0.0,
    strategy_max_draw_rate: float = 0.9,
    strategy_max_no_damage_rate: float = 0.75,
    strategy_max_low_engagement_rate: float = 0.5,
    strategy_max_idle_rate: float = 0.75,
    strategy_max_dominant_action_rate: float = 0.95,
    strategy_max_weaknesses: int = 10,
    require_replay_analysis: bool = True,
    min_maps: int = 2,
    required_maps: tuple[str, ...] | None = None,
    min_eval_episodes: int | None = None,
    min_map_episodes: int | None = None,
    min_map_score: float | None = 0.0,
    min_replay_combat_maps: int | None = None,
    min_opponent_historical_samples: int | None = None,
    min_head_to_head_episodes: int | None = None,
    min_head_to_head_map_episodes: int | None = None,
    require_candidate_checkpoint: bool = True,
    require_candidate_metadata: bool = True,
    require_candidate_integrity: bool = True,
    required_curriculum_stage: str | None = "full_map_pool",
    required_reward_preset: str | None = "anti_stall",
    require_head_to_head: bool | None = None,
) -> dict:
    if replay_save_interval is not None and replay_save_interval < 1:
        raise ValueError("replay_save_interval must be at least 1")
    if opponent_pool_seed is not None and opponent_pool_seed < 0:
        raise ValueError("opponent_pool_seed must be non-negative")
    if min_map_episodes is not None and min_map_episodes < 0:
        raise ValueError("min_map_episodes must be non-negative")
    if min_replay_combat_maps is not None and min_replay_combat_maps < 0:
        raise ValueError("min_replay_combat_maps must be non-negative")
    if (
        min_opponent_historical_samples is not None
        and min_opponent_historical_samples < 0
    ):
        raise ValueError("min_opponent_historical_samples must be non-negative")
    if min_head_to_head_episodes is not None and min_head_to_head_episodes < 0:
        raise ValueError("min_head_to_head_episodes must be non-negative")
    if (
        min_head_to_head_map_episodes is not None
        and min_head_to_head_map_episodes < 0
    ):
        raise ValueError("min_head_to_head_map_episodes must be non-negative")
    validate_run_id(run_id)

    suite_opponent_names = parse_builtin_opponents(suite_opponents)
    suite_map_names = parse_suite_maps(suite_maps, Config())
    if required_maps is None:
        effective_required_maps = suite_map_names
    else:
        unknown_required_maps = [
            name for name in required_maps if name not in PLATFORM_LAYOUTS
        ]
        if unknown_required_maps:
            raise ValueError(
                f"Unknown required map names: {', '.join(unknown_required_maps)}"
            )
        effective_required_maps = required_maps

    if required_curriculum_stage is not None:
        allowed_stages = {
            stage.name for stages in CURRICULUMS.values() for stage in stages
        }
        if required_curriculum_stage not in allowed_stages:
            raise ValueError(f"Unknown curriculum stage: {required_curriculum_stage}")
    if required_reward_preset is not None:
        reward_config_for_preset(required_reward_preset)

    suite_opponents_csv = ",".join(suite_opponent_names)
    suite_maps_csv = ",".join(suite_map_names)
    required_maps_csv = ",".join(effective_required_maps)

    effective_replay_save_interval = replay_save_interval
    replay_save_interval_source = "user" if replay_save_interval is not None else "config"
    if effective_replay_save_interval is None and timesteps <= 10_000:
        effective_replay_save_interval = 1
        replay_save_interval_source = "auto_small_run"
    rank_gate_config = {
        "min_score": rank_min_score,
        "min_win_rate": rank_min_win_rate,
        "max_draw_rate": rank_max_draw_rate,
        "max_no_damage_rate": rank_max_no_damage_rate,
        "max_low_engagement_rate": rank_max_low_engagement_rate,
        "min_map_score": rank_min_map_score,
    }
    strategy_report_config = {
        "max_draw_rate": strategy_max_draw_rate,
        "max_no_damage_rate": strategy_max_no_damage_rate,
        "max_low_engagement_rate": strategy_max_low_engagement_rate,
        "max_idle_rate": strategy_max_idle_rate,
        "max_dominant_action_rate": strategy_max_dominant_action_rate,
        "max_weaknesses": strategy_max_weaknesses,
    }

    checkpoint_dir = Path(checkpoint_root) / run_id
    eval_dir = Path(eval_root) / run_id
    replay_dir = Path(replay_root) / run_id
    preflight_dir = Path(eval_root) / f"{run_id}-preflight-smoke"
    preflight_timesteps = 128
    preflight_rounds = 1
    map_count = len(suite_map_names)
    self_play_sampling_preflight_min_maps = min(2, map_count)
    opponent_count = len(suite_opponent_names)
    effective_min_eval_episodes = (
        min_eval_episodes
        if min_eval_episodes is not None
        else rounds * map_count * opponent_count
    )
    effective_min_map_episodes = (
        min_map_episodes if min_map_episodes is not None else rounds * opponent_count
    )
    default_min_replay_combat_maps = (
        len(effective_required_maps) if effective_required_maps else min_maps
    )
    effective_min_replay_combat_maps = (
        min_replay_combat_maps
        if min_replay_combat_maps is not None
        else (default_min_replay_combat_maps if require_replay_analysis else 0)
    )
    effective_min_opponent_historical_samples = (
        min_opponent_historical_samples
        if min_opponent_historical_samples is not None
        else (1 if timesteps > 10_000 else 0)
    )
    effective_require_head_to_head = (
        timesteps > 10_000 if require_head_to_head is None else require_head_to_head
    )
    effective_min_head_to_head_episodes = (
        min_head_to_head_episodes
        if min_head_to_head_episodes is not None
        else (rounds * map_count * 2 if effective_require_head_to_head else 0)
    )
    effective_min_head_to_head_map_episodes = (
        min_head_to_head_map_episodes
        if min_head_to_head_map_episodes is not None
        else (rounds * 2 if effective_require_head_to_head else None)
    )

    long_run_check_parts = [
        "set +e",
        'python scripts/train.py --mode long_run_check \\',
        '  --promotion-audit-summary "$PROMOTION_AUDIT" \\',
        '  --strategy-report-summary "$STRATEGY_REPORT" \\',
        '  --artifact-index-summary "$ARTIFACT_INDEX" \\',
        f"  --long-run-min-maps {_shell_arg(min_maps)} \\",
        f"  --long-run-required-maps {_shell_arg(required_maps_csv)} \\",
        f"  --long-run-min-eval-episodes {_shell_arg(effective_min_eval_episodes)} \\",
        f"  --long-run-min-map-episodes {_shell_arg(effective_min_map_episodes)} \\",
    ]
    if min_map_score is not None:
        long_run_check_parts.append(
            f"  --long-run-min-map-score {_shell_arg(min_map_score)} \\"
        )
    if require_replay_analysis:
        long_run_check_parts.append("  --long-run-require-replay-analysis \\")
    if effective_min_replay_combat_maps > 0:
        long_run_check_parts.append(
            "  --long-run-min-replay-combat-maps "
            f"{_shell_arg(effective_min_replay_combat_maps)} \\"
        )
    if effective_min_opponent_historical_samples > 0:
        long_run_check_parts.append(
            "  --long-run-min-opponent-historical-samples "
            f"{_shell_arg(effective_min_opponent_historical_samples)} \\"
        )
    if require_candidate_checkpoint:
        long_run_check_parts.append("  --long-run-require-candidate-checkpoint \\")
    if require_candidate_metadata:
        long_run_check_parts.append("  --long-run-require-candidate-metadata \\")
    if require_candidate_integrity:
        long_run_check_parts.append("  --long-run-require-candidate-integrity \\")
    if required_curriculum_stage is not None:
        long_run_check_parts.append(
            "  --long-run-required-curriculum-stage "
            f"{_shell_arg(required_curriculum_stage)} \\"
        )
    if required_reward_preset is not None:
        long_run_check_parts.append(
            f"  --long-run-required-reward-preset {_shell_arg(required_reward_preset)} \\"
        )
    if effective_require_head_to_head:
        long_run_check_parts.append("  --long-run-require-head-to-head \\")
    if effective_min_head_to_head_episodes > 0:
        long_run_check_parts.append(
            "  --long-run-min-head-to-head-episodes "
            f"{_shell_arg(effective_min_head_to_head_episodes)} \\"
        )
    if effective_min_head_to_head_map_episodes is not None:
        long_run_check_parts.append(
            "  --long-run-min-head-to-head-map-episodes "
            f"{_shell_arg(effective_min_head_to_head_map_episodes)} \\"
        )
    long_run_check_parts.extend(
        [
            '  --eval-output-dir "$EVAL_DIR" \\',
            '  --eval-label long-run-check > "$EVAL_DIR/long-run-check.out" 2>&1',
            "LONG_RUN_CHECK_EXIT=$?",
            'printf "%s\\n" "$LONG_RUN_CHECK_EXIT" > "$EVAL_DIR/long-run-check.exitcode"',
            "set -e",
        ]
    )

    train_parts = [
        "python scripts/train.py --mode train \\",
        f"  --timesteps {_shell_arg(timesteps)} \\",
        "  --curriculum map_progression \\",
        "  --randomize-maps \\",
        f"  --map-choices {_shell_arg(suite_maps_csv)} \\",
        '  --checkpoint-dir "$CHECKPOINT_DIR" \\',
    ]
    if effective_replay_save_interval is not None:
        train_parts.append(
            f"  --replay-save-interval {_shell_arg(effective_replay_save_interval)} \\"
        )
    if opponent_pool_seed is not None:
        train_parts.append(
            f"  --opponent-pool-seed {_shell_arg(opponent_pool_seed)} \\"
        )
    train_parts.append('  --replay-dir "$REPLAY_DIR"')

    preflight_parts = [
        "python scripts/train_eval_smoke.py \\",
        '  --output-dir "$PREFLIGHT_DIR" \\',
        f"  --timesteps {_shell_arg(preflight_timesteps)} \\",
        f"  --rounds {_shell_arg(preflight_rounds)} \\",
        f"  --suite-opponents {_shell_arg(suite_opponents_csv)} \\",
        f"  --suite-maps {_shell_arg(suite_maps_csv)}",
    ]
    if opponent_pool_seed is not None:
        preflight_parts[-1] = f"{preflight_parts[-1]} \\"
        preflight_parts.append(
            f"  --opponent-pool-seed {_shell_arg(opponent_pool_seed)}"
        )
    self_play_sampling_preflight_parts = [
        "python scripts/self_play_sampling_smoke.py \\",
        '  --output-dir "$PREFLIGHT_DIR/self-play-sampling" \\',
        '  --summary-output "$PREFLIGHT_DIR/self-play-sampling-summary.json" \\',
        f"  --map-pool {_shell_arg(suite_maps_csv)} \\",
        "  --min-historical-samples 1 \\",
        f"  --min-maps-seen {_shell_arg(self_play_sampling_preflight_min_maps)}",
    ]
    if opponent_pool_seed is not None:
        self_play_sampling_preflight_parts[-1] = (
            f"{self_play_sampling_preflight_parts[-1]} \\"
        )
        self_play_sampling_preflight_parts.append(
            f"  --pool-seed {_shell_arg(opponent_pool_seed)}"
        )
    final_artifact_index_parts = [
        "python scripts/train.py --mode artifact_index \\",
        '  --artifact-dir "$EVAL_DIR" \\',
        "  --recursive-artifacts \\",
        '  --eval-output-dir "$EVAL_DIR" \\',
        "  --eval-label final-artifact-index",
    ]
    long_run_status_parts = [
        "python scripts/train.py --mode long_run_status \\",
        '  --artifact-dir "$EVAL_ROOT" \\',
        '  --eval-output-dir "$EVAL_DIR" \\',
        "  --eval-label long-run-status",
    ]
    league_health_parts = [
        "python scripts/train.py --mode league_health \\",
        '  --artifact-dir "$EVAL_ROOT" \\',
        '  --eval-output-dir "$EVAL_DIR" \\',
        "  --eval-label league-health",
    ]
    checkpoint_trust_manifest_parts = [
        "python scripts/train.py --mode checkpoint_trust_manifest \\",
        '  --checkpoint-dir "$CHECKPOINT_DIR" \\',
        '  --checkpoint-trust-manifest-output "$TRUSTED_CHECKPOINT_MANIFEST"',
    ]
    preflight_index_parts = [
        "python scripts/train.py --mode artifact_index \\",
        '  --artifact-dir "$PREFLIGHT_DIR" \\',
        "  --recursive-artifacts \\",
        '  --eval-output-dir "$EVAL_DIR" \\',
        "  --eval-label preflight-artifact-index",
    ]
    resolve_promotion_audit_parts = [
        'PROMOTION_AUDIT=$(ls -1t "$EVAL_DIR"/*_promotion.json 2>/dev/null | head -n 1 || true)',
        'if [ -z "$PROMOTION_AUDIT" ]; then',
        '  PROMOTION_AUDIT="$EVAL_DIR/MISSING_promotion.json"',
        "fi",
    ]
    audit_summary_parts = [
        'if [ -f "$PROMOTION_AUDIT" ]; then',
        "  python scripts/train.py --mode audit_summary \\",
        '    --audit-summary "$PROMOTION_AUDIT" \\',
        '    --eval-output-dir "$EVAL_DIR" \\',
        "    --eval-label promotion-summary",
        "fi",
    ]
    resolve_validation_artifact_parts = [
        'STRATEGY_REPORT=$(ls -1t "$EVAL_DIR"/*_strategy-report.json 2>/dev/null | head -n 1 || true)',
        'if [ -z "$STRATEGY_REPORT" ]; then',
        '  STRATEGY_REPORT="$EVAL_DIR/MISSING_strategy-report.json"',
        "fi",
        'ARTIFACT_INDEX=$(ls -1t "$EVAL_DIR"/*_artifact-index.json 2>/dev/null | head -n 1 || true)',
        'if [ -z "$ARTIFACT_INDEX" ]; then',
        '  ARTIFACT_INDEX="$EVAL_DIR/MISSING_artifact-index.json"',
        "fi",
    ]
    preflight_shell = "\n".join(
        [
            "set +e",
            *_with_output_redirect(preflight_parts, '"$EVAL_DIR/preflight.out"'),
            "PREFLIGHT_EXIT=$?",
            'printf "%s\\n" "$PREFLIGHT_EXIT" > "$EVAL_DIR/preflight.exitcode"',
            "set -e",
            'if [ "$PREFLIGHT_EXIT" -ne 0 ]; then',
            *(
                f"  {line}"
                for line in _with_output_redirect(
                    preflight_index_parts,
                    '"$EVAL_DIR/preflight-artifact-index.out"',
                )
            ),
            *(
                f"  {line}"
                for line in _with_output_redirect(
                    final_artifact_index_parts,
                    '"$EVAL_DIR/final-artifact-index.out"',
                )
            ),
            '  exit "$PREFLIGHT_EXIT"',
            "fi",
        ]
    )
    self_play_sampling_preflight_shell = "\n".join(
        [
            "set +e",
            *_with_output_redirect(
                self_play_sampling_preflight_parts,
                '"$EVAL_DIR/self-play-sampling-preflight.out"',
            ),
            "SELF_PLAY_SAMPLING_PREFLIGHT_EXIT=$?",
            'printf "%s\\n" "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT" > "$EVAL_DIR/self-play-sampling-preflight.exitcode"',
            "set -e",
            'if [ "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT" -ne 0 ]; then',
            '  printf "%s\\n" "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT" > "$EVAL_DIR/preflight.exitcode"',
            *(
                f"  {line}"
                for line in _with_output_redirect(
                    preflight_index_parts,
                    '"$EVAL_DIR/preflight-artifact-index.out"',
                )
            ),
            *(
                f"  {line}"
                for line in _with_output_redirect(
                    final_artifact_index_parts,
                    '"$EVAL_DIR/final-artifact-index.out"',
                )
            ),
            '  exit "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT"',
            "fi",
        ]
    )
    train_shell = "\n".join(
        [
            "set +e",
            *_with_output_redirect(train_parts, '"$EVAL_DIR/train.out"'),
            "TRAIN_EXIT=$?",
            'printf "%s\\n" "$TRAIN_EXIT" > "$EVAL_DIR/train.exitcode"',
            "set -e",
            'if [ "$TRAIN_EXIT" -ne 0 ]; then',
            *(
                f"  {line}"
                for line in _with_output_redirect(
                    final_artifact_index_parts,
                    '"$EVAL_DIR/final-artifact-index.out"',
                )
            ),
            '  exit "$TRAIN_EXIT"',
            "fi",
        ]
    )

    commands = [
        {
            "id": "create_run_dirs",
            "description": "Create isolated artifact directories for this run.",
            "expensive": False,
            "shell": 'mkdir -p "$CHECKPOINT_DIR" "$EVAL_DIR" "$REPLAY_DIR" "$PREFLIGHT_DIR"',
        },
        {
            "id": "archive_launcher",
            "description": "Copy this launcher into the run eval directory.",
            "expensive": False,
            "shell": 'cp "$0" "$EVAL_DIR/long-run-launcher.sh"',
        },
        {
            "id": "self_play_sampling_smoke_preflight",
            "description": "Verify historical opponent sampling before any PPO smoke or real training.",
            "expensive": False,
            "shell": self_play_sampling_preflight_shell,
        },
        {
            "id": "train_eval_smoke_preflight",
            "description": "Run the tiny train/eval/verifier smoke before spending real compute.",
            "expensive": False,
            "shell": preflight_shell,
        },
        {
            "id": "train",
            "description": "Run the real curriculum self-play training job and preserve diagnostics if it fails.",
            "expensive": True,
            "shell": train_shell,
        },
        {
            "id": "checkpoint_trust_manifest",
            "description": "Write a trusted SHA-256 allowlist for checkpoints generated by this run.",
            "expensive": False,
            "shell": "\n".join(
                _with_output_redirect(
                    checkpoint_trust_manifest_parts,
                    '"$EVAL_DIR/checkpoint-trust-manifest.out"',
                )
            ),
        },
        {
            "id": "promotion_audit",
            "description": "Rank checkpoints, gate the top promotion candidate, and capture its exit code.",
            "expensive": False,
            "shell": "\n".join(
                [
                    "set +e",
                    *_with_output_redirect(
                        [
                            "python scripts/train.py --mode promotion_audit \\",
                            '  --checkpoint-dir "$CHECKPOINT_DIR" \\',
                            f"  --suite-opponents {_shell_arg(suite_opponents_csv)} \\",
                            f"  --suite-maps {_shell_arg(suite_maps_csv)} \\",
                            f"  --rounds {_shell_arg(rounds)} \\",
                            "  --rank-head-to-head \\",
                            f"  --rank-min-score {_shell_arg(rank_min_score)} \\",
                            f"  --rank-min-win-rate {_shell_arg(rank_min_win_rate)} \\",
                            f"  --rank-max-draw-rate {_shell_arg(rank_max_draw_rate)} \\",
                            f"  --rank-max-no-damage-rate {_shell_arg(rank_max_no_damage_rate)} \\",
                            "  --rank-max-low-engagement-rate "
                            f"{_shell_arg(rank_max_low_engagement_rate)} \\",
                            *(
                                [
                                    "  --rank-min-map-score "
                                    f"{_shell_arg(rank_min_map_score)} \\"
                                ]
                                if rank_min_map_score is not None
                                else []
                            ),
                            '  --trusted-checkpoint-manifest "$TRUSTED_CHECKPOINT_MANIFEST" \\',
                            '  --eval-output-dir "$EVAL_DIR" \\',
                            "  --eval-label promotion",
                        ],
                        '"$EVAL_DIR/promotion-audit.out"',
                    ),
                    "PROMOTION_AUDIT_EXIT=$?",
                    'printf "%s\\n" "$PROMOTION_AUDIT_EXIT" > "$EVAL_DIR/promotion-audit.exitcode"',
                    "set -e",
                ]
            ),
        },
        {
            "id": "resolve_promotion_audit",
            "description": "Resolve the newest promotion-audit artifact path.",
            "expensive": False,
            "shell": "\n".join(resolve_promotion_audit_parts),
        },
        {
            "id": "audit_summary",
            "description": "Write a compact promotion-audit summary when the audit artifact exists.",
            "expensive": False,
            "shell": "\n".join(
                _with_output_redirect(
                    audit_summary_parts,
                    '"$EVAL_DIR/audit-summary.out"',
                )
            ),
        },
        {
            "id": "sample_replay_analysis",
            "description": "Save representative replay-analysis artifacts.",
            "expensive": False,
            "shell": "\n".join(
                _with_output_redirect(
                    [
                        "python scripts/train.py --mode analyze \\",
                        '  --replay-dir "$REPLAY_DIR" \\',
                        "  --replay-samples-per-bucket "
                        f"{_shell_arg(replay_samples_per_bucket)} \\",
                        '  --eval-output-dir "$EVAL_DIR" \\',
                        "  --eval-label replay-sample",
                    ],
                    '"$EVAL_DIR/replay-analysis.out"',
                )
            ),
        },
        {
            "id": "strategy_report",
            "description": "Scan artifacts for stalled or degenerate strategies.",
            "expensive": False,
            "shell": "\n".join(
                _with_output_redirect(
                    [
                        "python scripts/train.py --mode strategy_report \\",
                        '  --artifact-dir "$EVAL_DIR" \\',
                        "  --recursive-artifacts \\",
                        f"  --strategy-max-draw-rate {_shell_arg(strategy_max_draw_rate)} \\",
                        "  --strategy-max-no-damage-rate "
                        f"{_shell_arg(strategy_max_no_damage_rate)} \\",
                        "  --strategy-max-low-engagement-rate "
                        f"{_shell_arg(strategy_max_low_engagement_rate)} \\",
                        f"  --strategy-max-idle-rate {_shell_arg(strategy_max_idle_rate)} \\",
                        "  --strategy-max-dominant-action-rate "
                        f"{_shell_arg(strategy_max_dominant_action_rate)} \\",
                        "  --strategy-max-weaknesses "
                        f"{_shell_arg(strategy_max_weaknesses)} \\",
                        '  --eval-output-dir "$EVAL_DIR" \\',
                        "  --eval-label strategy-report",
                    ],
                    '"$EVAL_DIR/strategy-report.out"',
                )
            ),
        },
        {
            "id": "artifact_index",
            "description": "Index all saved artifacts for the run.",
            "expensive": False,
            "shell": "\n".join(
                _with_output_redirect(
                    [
                        "python scripts/train.py --mode artifact_index \\",
                        '  --artifact-dir "$EVAL_DIR" \\',
                        "  --recursive-artifacts \\",
                        '  --eval-output-dir "$EVAL_DIR" \\',
                        "  --eval-label artifact-index",
                    ],
                    '"$EVAL_DIR/artifact-index.out"',
                )
            ),
        },
        {
            "id": "resolve_validation_artifacts",
            "description": "Resolve artifacts needed by long_run_check.",
            "expensive": False,
            "shell": "\n".join(resolve_validation_artifact_parts),
        },
        {
            "id": "long_run_check",
            "description": "Validate promotion evidence against long-run criteria and capture its exit code.",
            "expensive": False,
            "shell": "\n".join(long_run_check_parts),
        },
        {
            "id": "long_run_status",
            "description": "Summarize launcher execution and missing long-run evidence.",
            "expensive": False,
            "shell": "\n".join(
                _with_output_redirect(
                    long_run_status_parts,
                    '"$EVAL_DIR/long-run-status.out"',
                )
            ),
        },
        {
            "id": "league_health",
            "description": "Combine strategy, status, rank, and verifier signals into a health summary.",
            "expensive": False,
            "shell": "\n".join(
                _with_output_redirect(
                    league_health_parts,
                    '"$EVAL_DIR/league-health.out"',
                )
            ),
        },
        {
            "id": "final_artifact_index",
            "description": "Index final artifacts, including status and league-health output.",
            "expensive": False,
            "shell": "\n".join(
                _with_output_redirect(
                    final_artifact_index_parts,
                    '"$EVAL_DIR/final-artifact-index.out"',
                )
            ),
        },
        {
            "id": "exit_with_long_run_check_status",
            "description": "Preserve the long_run_check pass/fail status after final indexing.",
            "expensive": False,
            "shell": 'exit "$LONG_RUN_CHECK_EXIT"',
        },
    ]

    shell_header = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        _shell_assign("RUN_ID", run_id),
        _shell_assign("EVAL_ROOT", str(eval_root)),
        _shell_assign("CHECKPOINT_DIR", str(checkpoint_dir)),
        _shell_assign("EVAL_DIR", str(eval_dir)),
        _shell_assign("REPLAY_DIR", str(replay_dir)),
        _shell_assign("PREFLIGHT_DIR", str(preflight_dir)),
        'TRUSTED_CHECKPOINT_MANIFEST="$EVAL_DIR/checkpoint-trust-manifest.json"',
    ]
    shell_script = "\n\n".join(
        [
            *shell_header,
            *(command["shell"] for command in commands),
        ]
    )
    preflight_shell_script = "\n\n".join(
        [
            *shell_header,
            commands[0]["shell"],
            self_play_sampling_preflight_shell,
            preflight_shell,
        ]
    )

    return {
        "artifact": artifact_metadata("long_run_manifest"),
        "manifest_config": {
            "run_id": run_id,
            "checkpoint_dir": str(checkpoint_dir),
            "eval_dir": str(eval_dir),
            "replay_dir": str(replay_dir),
            "preflight_dir": str(preflight_dir),
            "trusted_checkpoint_manifest": str(
                eval_dir / "checkpoint-trust-manifest.json"
            ),
            "preflight_timesteps": preflight_timesteps,
            "preflight_rounds": preflight_rounds,
            "self_play_sampling_preflight_min_maps": (
                self_play_sampling_preflight_min_maps
            ),
            "timesteps": timesteps,
            "suite_opponents": suite_opponents_csv,
            "suite_maps": suite_maps_csv,
            "rounds": rounds,
            "replay_samples_per_bucket": replay_samples_per_bucket,
            "replay_save_interval": effective_replay_save_interval,
            "replay_save_interval_source": replay_save_interval_source,
            "opponent_pool_seed": opponent_pool_seed,
            "rank_gate": rank_gate_config,
            "strategy_report": strategy_report_config,
            "source_control": source_control_snapshot(),
            "require_replay_analysis": require_replay_analysis,
            "min_maps": min_maps,
            "required_maps": list(effective_required_maps),
            "min_eval_episodes": effective_min_eval_episodes,
            "min_map_episodes": effective_min_map_episodes,
            "min_map_score": min_map_score,
            "min_replay_combat_maps": effective_min_replay_combat_maps,
            "min_opponent_historical_samples": (
                effective_min_opponent_historical_samples
            ),
            "min_head_to_head_episodes": effective_min_head_to_head_episodes,
            "min_head_to_head_map_episodes": (
                effective_min_head_to_head_map_episodes
            ),
            "require_candidate_checkpoint": require_candidate_checkpoint,
            "require_candidate_metadata": require_candidate_metadata,
            "require_candidate_integrity": require_candidate_integrity,
            "required_curriculum_stage": required_curriculum_stage,
            "required_reward_preset": required_reward_preset,
            "require_head_to_head": effective_require_head_to_head,
        },
        "guardrails": {
            "executes_training": False,
            "deletes_artifacts": False,
            "contains_expensive_training_command": True,
        },
        "commands": commands,
        "shell_script": shell_script,
        "preflight_shell_script": preflight_shell_script,
    }


def run_long_run_manifest(
    *,
    run_id: str | None,
    checkpoint_root: str,
    eval_root: str,
    replay_root: str,
    timesteps: int,
    suite_opponents: str,
    suite_maps: str,
    rounds: int,
    replay_samples_per_bucket: int,
    replay_save_interval: int | None = None,
    opponent_pool_seed: int | None = None,
    rank_min_score: float = 0.1,
    rank_min_win_rate: float = 0.0,
    rank_max_draw_rate: float = 0.9,
    rank_max_no_damage_rate: float = 0.75,
    rank_max_low_engagement_rate: float = 0.5,
    rank_min_map_score: float | None = 0.0,
    strategy_max_draw_rate: float = 0.9,
    strategy_max_no_damage_rate: float = 0.75,
    strategy_max_low_engagement_rate: float = 0.5,
    strategy_max_idle_rate: float = 0.75,
    strategy_max_dominant_action_rate: float = 0.95,
    strategy_max_weaknesses: int = 10,
    require_replay_analysis: bool,
    min_maps: int,
    required_maps: tuple[str, ...] | None,
    min_eval_episodes: int | None,
    min_map_episodes: int | None = None,
    min_map_score: float | None = None,
    min_replay_combat_maps: int | None = None,
    min_opponent_historical_samples: int | None = None,
    min_head_to_head_episodes: int | None = None,
    min_head_to_head_map_episodes: int | None = None,
    require_candidate_checkpoint: bool = True,
    require_candidate_metadata: bool = True,
    require_candidate_integrity: bool = True,
    required_curriculum_stage: str | None = "full_map_pool",
    required_reward_preset: str | None = "anti_stall",
    require_head_to_head: bool | None = None,
    output_dir: str | None = None,
    output_label: str | None = None,
) -> None:
    if run_id is None:
        run_id = f"arena-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    manifest = build_long_run_manifest(
        run_id=run_id,
        checkpoint_root=checkpoint_root,
        eval_root=eval_root,
        replay_root=replay_root,
        timesteps=timesteps,
        suite_opponents=suite_opponents,
        suite_maps=suite_maps,
        rounds=rounds,
        replay_samples_per_bucket=replay_samples_per_bucket,
        replay_save_interval=replay_save_interval,
        opponent_pool_seed=opponent_pool_seed,
        rank_min_score=rank_min_score,
        rank_min_win_rate=rank_min_win_rate,
        rank_max_draw_rate=rank_max_draw_rate,
        rank_max_no_damage_rate=rank_max_no_damage_rate,
        rank_max_low_engagement_rate=rank_max_low_engagement_rate,
        rank_min_map_score=rank_min_map_score,
        strategy_max_draw_rate=strategy_max_draw_rate,
        strategy_max_no_damage_rate=strategy_max_no_damage_rate,
        strategy_max_low_engagement_rate=strategy_max_low_engagement_rate,
        strategy_max_idle_rate=strategy_max_idle_rate,
        strategy_max_dominant_action_rate=strategy_max_dominant_action_rate,
        strategy_max_weaknesses=strategy_max_weaknesses,
        require_replay_analysis=require_replay_analysis,
        min_maps=min_maps,
        required_maps=required_maps,
        min_eval_episodes=min_eval_episodes,
        min_map_episodes=min_map_episodes,
        min_map_score=min_map_score,
        min_replay_combat_maps=min_replay_combat_maps,
        min_opponent_historical_samples=min_opponent_historical_samples,
        min_head_to_head_episodes=min_head_to_head_episodes,
        min_head_to_head_map_episodes=min_head_to_head_map_episodes,
        require_candidate_checkpoint=require_candidate_checkpoint,
        require_candidate_metadata=require_candidate_metadata,
        require_candidate_integrity=require_candidate_integrity,
        required_curriculum_stage=required_curriculum_stage,
        required_reward_preset=required_reward_preset,
        require_head_to_head=require_head_to_head,
    )

    if output_dir is not None:
        label = output_label or f"{run_id}-long-run-manifest"
        path = write_eval_summary(manifest, output_dir, label=label)
        script_path = path.with_suffix(".sh")
        preflight_script_path = path.with_suffix(".preflight.sh")
        script_path.write_text(manifest["shell_script"] + "\n")
        preflight_script_path.write_text(manifest["preflight_shell_script"] + "\n")
        script_path.chmod(0o755)
        preflight_script_path.chmod(0o755)
        manifest["manifest_artifact_path"] = str(path)
        manifest["shell_script_path"] = str(script_path)
        manifest["preflight_shell_script_path"] = str(preflight_script_path)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    if output_dir is not None:
        print(f"Saved long-run manifest to {manifest['manifest_artifact_path']}")
        print(f"Saved long-run launcher to {manifest['shell_script_path']}")
        print(
            "Saved long-run preflight launcher to "
            f"{manifest['preflight_shell_script_path']}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Arena Fighters: train, watch, replay, eval, or rank"
    )
    parser.add_argument(
        "--mode",
        choices=[
            "train",
            "watch",
            "replay",
            "analyze",
            "eval",
            "compare",
            "gate",
            "suite",
            "rank",
            "rank_gate",
            "promotion_audit",
            "audit_summary",
            "artifact_index",
            "strategy_report",
            "long_run_manifest",
            "long_run_check",
            "long_run_status",
            "league_health",
            "checkpoint_trust_manifest",
        ],
        default="train",
        help="Operating mode (default: train)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a trusted model checkpoint (for watch/eval/suite modes)",
    )
    parser.add_argument(
        "--episode",
        type=str,
        default=None,
        help="Path to replay JSON file (for replay mode)",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from config or long-run manifest",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Run ID for long_run_manifest mode (default: generated UTC timestamp)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving or discovering trusted checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default="replays",
        help="Directory for replay files or sampled replay analysis (default: replays)",
    )
    parser.add_argument(
        "--replay-samples-per-bucket",
        type=int,
        default=1,
        help="Replay samples per outcome/behavior bucket for analyze mode without --episode",
    )
    parser.add_argument(
        "--replay-save-interval",
        type=int,
        default=None,
        help="Save one training replay every N completed episodes (default: config value)",
    )
    parser.add_argument(
        "--opponent-pool-seed",
        type=int,
        default=None,
        help="Seed for reproducible opponent-pool sampling in train/manifest modes",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Number of rounds to watch/evaluate (watch: 0 = infinite, eval: 0 = 20)",
    )
    parser.add_argument(
        "--opponent",
        choices=BUILTIN_POLICY_NAMES,
        default="random",
        help="Built-in opponent for eval mode (default: random)",
    )
    parser.add_argument(
        "--agent-policy",
        choices=BUILTIN_POLICY_NAMES,
        default="random",
        help="Built-in agent 0 policy for eval/suite modes when --checkpoint is omitted",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for eval mode",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic model actions in eval mode",
    )
    parser.add_argument(
        "--map",
        dest="map_name",
        choices=sorted(PLATFORM_LAYOUTS),
        default=None,
        help="Arena map to use",
    )
    parser.add_argument(
        "--randomize-maps",
        action="store_true",
        help="Randomly choose a map on each reset",
    )
    parser.add_argument(
        "--map-choices",
        type=str,
        default=None,
        help="Comma-separated map names used with --randomize-maps",
    )
    parser.add_argument(
        "--reward-preset",
        choices=sorted(set(REWARD_PRESETS) | set(REWARD_PRESET_ALIASES)),
        default=None,
        help="Reward preset for train/watch/eval modes",
    )
    parser.add_argument(
        "--curriculum",
        choices=sorted(CURRICULUMS),
        default=None,
        help="Curriculum schedule for train mode",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=None,
        help="Directory for timestamped eval/artifact JSON summaries",
    )
    parser.add_argument(
        "--eval-label",
        type=str,
        default=None,
        help="Optional label used in eval summary filenames",
    )
    parser.add_argument(
        "--before",
        type=str,
        default=None,
        help="Earlier eval JSON summary for compare mode",
    )
    parser.add_argument(
        "--after",
        type=str,
        default=None,
        help="Later eval JSON summary for compare mode",
    )
    parser.add_argument(
        "--suite-opponents",
        type=str,
        default=None,
        help="Comma-separated built-in opponents for suite/rank modes",
    )
    parser.add_argument(
        "--suite-maps",
        type=str,
        default=None,
        help="Comma-separated maps for suite/rank modes",
    )
    parser.add_argument(
        "--rank-checkpoints",
        type=str,
        default=None,
        help=(
            "Comma-separated trusted checkpoint paths for rank mode; "
            "defaults to --checkpoint-dir discovery"
        ),
    )
    parser.add_argument(
        "--trusted-checkpoint-manifest",
        type=str,
        default=None,
        help=(
            "JSON allowlist of trusted checkpoint paths/names and SHA-256 digests "
            "checked before SB3 deserialization"
        ),
    )
    parser.add_argument(
        "--checkpoint-trust-manifest-output",
        type=str,
        default=None,
        help="Output path for checkpoint_trust_manifest mode",
    )
    parser.add_argument(
        "--allow-unverified-checkpoints",
        action="store_true",
        help=(
            "Allow loading checkpoints without a trusted manifest; use only for "
            "known-local legacy checkpoints"
        ),
    )
    parser.add_argument(
        "--rank-draw-weight",
        type=float,
        default=0.5,
        help="Rank score credit for draw rate (default: 0.5)",
    )
    parser.add_argument(
        "--rank-no-damage-penalty",
        type=float,
        default=0.25,
        help="Rank score penalty for no-damage episode rate (default: 0.25)",
    )
    parser.add_argument(
        "--rank-low-engagement-penalty",
        type=float,
        default=0.25,
        help="Rank score penalty for low-engagement episode rate (default: 0.25)",
    )
    parser.add_argument(
        "--rank-head-to-head",
        action="store_true",
        help="Include pairwise checkpoint-vs-checkpoint matchups in rank mode",
    )
    parser.add_argument(
        "--rank-initial-elo",
        type=float,
        default=1000.0,
        help="Initial Elo rating for head-to-head rank standings (default: 1000)",
    )
    parser.add_argument(
        "--rank-elo-k",
        type=float,
        default=32.0,
        help="Elo K-factor for head-to-head rank standings (default: 32)",
    )
    parser.add_argument(
        "--rank-summary",
        type=str,
        default=None,
        help="Rank JSON summary for rank_gate mode",
    )
    parser.add_argument(
        "--audit-summary",
        dest="audit_summary_path",
        type=str,
        default=None,
        help="Promotion-audit JSON summary for audit_summary mode",
    )
    parser.add_argument(
        "--promotion-audit-summary",
        dest="promotion_audit_path",
        type=str,
        default=None,
        help="Promotion-audit JSON summary for long_run_check mode",
    )
    parser.add_argument(
        "--strategy-report-summary",
        dest="strategy_report_path",
        type=str,
        default=None,
        help="Strategy-report JSON summary for long_run_check mode",
    )
    parser.add_argument(
        "--artifact-index-summary",
        dest="artifact_index_path",
        type=str,
        default=None,
        help="Artifact-index JSON summary for long_run_check mode",
    )
    parser.add_argument(
        "--artifact-dir",
        type=str,
        default="evals",
        help="Directory of saved JSON artifacts for artifact report modes",
    )
    parser.add_argument(
        "--recursive-artifacts",
        action="store_true",
        help=(
            "Recursively scan --artifact-dir in artifact_index and strategy_report "
            "modes; long_run_status and league_health always scan recursively"
        ),
    )
    parser.add_argument(
        "--strategy-max-draw-rate",
        type=float,
        default=0.9,
        help="Draw-rate threshold for strategy_report mode (default: 0.9)",
    )
    parser.add_argument(
        "--strategy-max-no-damage-rate",
        type=float,
        default=0.75,
        help="No-damage rate threshold for strategy_report mode (default: 0.75)",
    )
    parser.add_argument(
        "--strategy-max-low-engagement-rate",
        type=float,
        default=0.5,
        help="Low-engagement rate threshold for strategy_report mode (default: 0.5)",
    )
    parser.add_argument(
        "--strategy-max-idle-rate",
        type=float,
        default=0.75,
        help="Agent 0 idle-rate threshold for strategy_report mode (default: 0.75)",
    )
    parser.add_argument(
        "--strategy-max-dominant-action-rate",
        type=float,
        default=0.95,
        help=(
            "Agent 0 dominant-action-rate threshold for strategy_report mode "
            "(default: 0.95)"
        ),
    )
    parser.add_argument(
        "--strategy-max-weaknesses",
        type=int,
        default=10,
        help=(
            "Maximum low-score map/opponent weaknesses to include in "
            "strategy_report output; use -1 for all (default: 10)"
        ),
    )
    parser.add_argument(
        "--rank-min-score",
        type=float,
        default=0.1,
        help="Minimum top-checkpoint rank score for rank_gate mode (default: 0.1)",
    )
    parser.add_argument(
        "--rank-min-win-rate",
        type=float,
        default=0.0,
        help="Minimum top-checkpoint mean win rate for rank_gate mode (default: 0.0)",
    )
    parser.add_argument(
        "--rank-max-draw-rate",
        type=float,
        default=0.9,
        help="Maximum top-checkpoint draw rate for rank_gate mode (default: 0.9)",
    )
    parser.add_argument(
        "--rank-max-no-damage-rate",
        type=float,
        default=0.75,
        help="Maximum top-checkpoint no-damage rate for rank_gate mode (default: 0.75)",
    )
    parser.add_argument(
        "--rank-max-low-engagement-rate",
        type=float,
        default=0.5,
        help="Maximum top-checkpoint low-engagement rate for rank_gate mode (default: 0.5)",
    )
    parser.add_argument(
        "--rank-min-map-score",
        type=float,
        default=None,
        help=(
            "Optional minimum top-checkpoint mean score on every evaluated map "
            "for rank_gate and promotion_audit modes"
        ),
    )
    parser.add_argument(
        "--rank-min-head-to-head-elo",
        type=float,
        default=None,
        help="Optional minimum top-checkpoint head-to-head Elo for rank_gate mode",
    )
    parser.add_argument(
        "--rank-min-head-to-head-score",
        type=float,
        default=None,
        help="Optional minimum top-checkpoint head-to-head score for rank_gate mode",
    )
    parser.add_argument(
        "--audit-include-nested",
        action="store_true",
        help="Include full nested rank and rank-gate JSON in promotion_audit output",
    )
    parser.add_argument(
        "--long-run-min-maps",
        type=int,
        default=2,
        help="Minimum candidate map coverage for long_run_check mode (default: 2)",
    )
    parser.add_argument(
        "--long-run-required-maps",
        type=str,
        default=None,
        help="Comma-separated maps that must appear in long_run_check candidate scores",
    )
    parser.add_argument(
        "--long-run-min-eval-episodes",
        type=int,
        default=0,
        help="Minimum candidate baseline-suite episode count for long_run_check mode",
    )
    parser.add_argument(
        "--long-run-min-map-episodes",
        type=int,
        default=None,
        help="Optional minimum candidate baseline-suite episodes on every evaluated map",
    )
    parser.add_argument(
        "--long-run-min-map-score",
        type=float,
        default=None,
        help="Optional minimum candidate mean score on every evaluated map",
    )
    parser.add_argument(
        "--long-run-require-replay-analysis",
        action="store_true",
        help="Require at least one replay-analysis artifact with combat in long_run_check mode",
    )
    parser.add_argument(
        "--long-run-min-replay-combat-maps",
        type=int,
        default=0,
        help="Minimum distinct maps with combat replay-analysis evidence",
    )
    parser.add_argument(
        "--long-run-min-opponent-historical-samples",
        type=int,
        default=None,
        help=(
            "Minimum historical-opponent samples recorded in candidate checkpoint "
            "metadata"
        ),
    )
    parser.add_argument(
        "--long-run-min-head-to-head-episodes",
        type=int,
        default=0,
        help="Minimum actual checkpoint-vs-checkpoint episodes in long_run_check mode",
    )
    parser.add_argument(
        "--long-run-min-head-to-head-map-episodes",
        type=int,
        default=None,
        help="Optional minimum checkpoint-vs-checkpoint episodes on each required map",
    )
    parser.add_argument(
        "--long-run-require-candidate-checkpoint",
        action="store_true",
        help="Require the promoted candidate checkpoint path to exist",
    )
    parser.add_argument(
        "--long-run-require-candidate-metadata",
        action="store_true",
        help="Require the promoted candidate checkpoint metadata sidecar to exist",
    )
    parser.add_argument(
        "--long-run-require-candidate-integrity",
        action="store_true",
        help="Require candidate metadata SHA-256 to match the checkpoint file",
    )
    parser.add_argument(
        "--long-run-require-head-to-head",
        action="store_true",
        help="Require head-to-head checkpoint standings in long_run_check mode",
    )
    parser.add_argument(
        "--long-run-required-curriculum-stage",
        type=str,
        default=None,
        help="Optional required candidate checkpoint metadata curriculum stage",
    )
    parser.add_argument(
        "--long-run-required-reward-preset",
        type=str,
        default=None,
        help="Optional required candidate checkpoint metadata reward preset",
    )
    args = parser.parse_args()

    cfg = Config()

    arena_overrides = {}
    if args.map_name is not None:
        arena_overrides["map_name"] = args.map_name
    if args.randomize_maps:
        arena_overrides["randomize_maps"] = True
    if args.map_choices is not None:
        map_choices = tuple(
            name.strip() for name in args.map_choices.split(",") if name.strip()
        )
        if not map_choices:
            parser.error("--map-choices must include at least one map name")
        unknown_maps = [name for name in map_choices if name not in PLATFORM_LAYOUTS]
        if unknown_maps:
            parser.error(f"Unknown map names: {', '.join(unknown_maps)}")
        arena_overrides["map_choices"] = map_choices
    if arena_overrides:
        cfg = replace(cfg, arena=replace(cfg.arena, **arena_overrides))
    if args.reward_preset is not None:
        cfg = replace(cfg, reward=reward_config_for_preset(args.reward_preset))

    # Override timesteps if provided
    if args.timesteps is not None:
        cfg = replace(
            cfg,
            training=replace(cfg.training, total_timesteps=args.timesteps),
        )
    if args.curriculum is not None:
        cfg = replace(
            cfg,
            training=replace(cfg.training, curriculum_name=args.curriculum),
        )
    if args.replay_save_interval is not None:
        if args.replay_save_interval < 1:
            parser.error("--replay-save-interval must be at least 1")
        cfg = replace(
            cfg,
            training=replace(
                cfg.training,
                replay_save_interval=args.replay_save_interval,
            ),
        )
    if args.opponent_pool_seed is not None:
        if args.opponent_pool_seed < 0:
            parser.error("--opponent-pool-seed must be non-negative")
        cfg = replace(
            cfg,
            training=replace(
                cfg.training,
                opponent_pool_seed=args.opponent_pool_seed,
            ),
        )
    if (
        args.long_run_min_opponent_historical_samples is not None
        and args.long_run_min_opponent_historical_samples < 0
    ):
        parser.error("--long-run-min-opponent-historical-samples must be non-negative")
    try:
        trusted_checkpoint_manifest = (
            load_checkpoint_trust_manifest(args.trusted_checkpoint_manifest)
            if args.trusted_checkpoint_manifest
            else None
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(f"Invalid --trusted-checkpoint-manifest: {exc}")

    if args.mode == "train":
        run_train(cfg, args.checkpoint_dir, args.replay_dir)
    elif args.mode == "watch":
        run_watch(
            cfg,
            args.checkpoint,
            num_rounds=args.rounds,
            trusted_checkpoint_manifest=trusted_checkpoint_manifest,
            allow_unverified_checkpoints=args.allow_unverified_checkpoints,
        )
    elif args.mode == "replay":
        if not args.episode:
            parser.error("--episode is required for replay mode")
        run_replay(cfg, args.episode)
    elif args.mode == "analyze":
        if args.episode:
            run_analyze_replay(
                args.episode,
                output_dir=args.eval_output_dir,
                output_label=args.eval_label,
            )
        else:
            run_analyze_replay_dir(
                args.replay_dir,
                samples_per_bucket=args.replay_samples_per_bucket,
                output_dir=args.eval_output_dir,
                output_label=args.eval_label,
            )
    elif args.mode == "eval":
        run_eval(
            cfg,
            args.checkpoint,
            args.opponent,
            num_rounds=args.rounds,
            seed=args.seed,
            deterministic=not args.stochastic,
            reward_preset=args.reward_preset or "default",
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
            agent_policy=args.agent_policy,
            trusted_checkpoint_manifest=trusted_checkpoint_manifest,
            allow_unverified_checkpoints=args.allow_unverified_checkpoints,
        )
    elif args.mode == "compare":
        if not args.before or not args.after:
            parser.error("--before and --after are required for compare mode")
        run_compare(
            args.before,
            args.after,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
        )
    elif args.mode == "gate":
        if not args.before or not args.after:
            parser.error("--before and --after are required for gate mode")
        run_gate(
            args.before,
            args.after,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
        )
    elif args.mode == "rank_gate":
        if not args.rank_summary:
            parser.error("--rank-summary is required for rank_gate mode")
        run_rank_gate(
            args.rank_summary,
            min_score=args.rank_min_score,
            min_win_rate=args.rank_min_win_rate,
            max_draw_rate=args.rank_max_draw_rate,
            max_no_damage_rate=args.rank_max_no_damage_rate,
            max_low_engagement_rate=args.rank_max_low_engagement_rate,
            min_map_score=args.rank_min_map_score,
            min_head_to_head_elo=args.rank_min_head_to_head_elo,
            min_head_to_head_score=args.rank_min_head_to_head_score,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
        )
    elif args.mode == "promotion_audit":
        try:
            suite_opponents = parse_builtin_opponents(args.suite_opponents)
            suite_maps = parse_suite_maps(args.suite_maps, cfg)
            rank_checkpoints = parse_rank_checkpoints(args.rank_checkpoints)
        except ValueError as exc:
            parser.error(str(exc))

        run_promotion_audit(
            cfg,
            rank_checkpoints,
            args.checkpoint_dir,
            suite_opponents,
            suite_maps,
            num_rounds=args.rounds,
            seed=args.seed,
            deterministic=not args.stochastic,
            reward_preset=args.reward_preset or "default",
            draw_weight=args.rank_draw_weight,
            no_damage_penalty=args.rank_no_damage_penalty,
            low_engagement_penalty=args.rank_low_engagement_penalty,
            include_head_to_head=args.rank_head_to_head,
            initial_elo=args.rank_initial_elo,
            elo_k_factor=args.rank_elo_k,
            min_score=args.rank_min_score,
            min_win_rate=args.rank_min_win_rate,
            max_draw_rate=args.rank_max_draw_rate,
            max_no_damage_rate=args.rank_max_no_damage_rate,
            max_low_engagement_rate=args.rank_max_low_engagement_rate,
            min_head_to_head_elo=args.rank_min_head_to_head_elo,
            min_head_to_head_score=args.rank_min_head_to_head_score,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
            include_nested=args.audit_include_nested,
            trusted_checkpoint_manifest=trusted_checkpoint_manifest,
            allow_unverified_checkpoints=args.allow_unverified_checkpoints,
            min_map_score=args.rank_min_map_score,
        )
    elif args.mode == "audit_summary":
        if not args.audit_summary_path:
            parser.error("--audit-summary is required for audit_summary mode")
        run_audit_summary(
            args.audit_summary_path,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
        )
    elif args.mode == "artifact_index":
        run_artifact_index(
            args.artifact_dir,
            recursive=args.recursive_artifacts,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
        )
    elif args.mode == "strategy_report":
        run_strategy_report(
            args.artifact_dir,
            recursive=args.recursive_artifacts,
            max_draw_rate=args.strategy_max_draw_rate,
            max_no_damage_rate=args.strategy_max_no_damage_rate,
            max_low_engagement_rate=args.strategy_max_low_engagement_rate,
            max_idle_rate=args.strategy_max_idle_rate,
            max_dominant_action_rate=args.strategy_max_dominant_action_rate,
            max_weaknesses=args.strategy_max_weaknesses,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
        )
    elif args.mode == "long_run_check":
        missing = []
        if not args.promotion_audit_path:
            missing.append("--promotion-audit-summary")
        if not args.strategy_report_path:
            missing.append("--strategy-report-summary")
        if not args.artifact_index_path:
            missing.append("--artifact-index-summary")
        if missing:
            parser.error(f"{', '.join(missing)} required for long_run_check mode")
        try:
            required_maps = (
                parse_suite_maps(args.long_run_required_maps, cfg)
                if args.long_run_required_maps
                else ()
            )
        except ValueError as exc:
            parser.error(str(exc))
        run_long_run_check(
            args.promotion_audit_path,
            args.strategy_report_path,
            args.artifact_index_path,
            min_maps=args.long_run_min_maps,
            required_maps=required_maps,
            min_eval_episodes=args.long_run_min_eval_episodes,
            min_map_episodes=args.long_run_min_map_episodes,
            min_map_score=args.long_run_min_map_score,
            require_replay_analysis=args.long_run_require_replay_analysis,
            min_replay_combat_maps=args.long_run_min_replay_combat_maps,
            min_opponent_historical_samples=(
                args.long_run_min_opponent_historical_samples or 0
            ),
            min_head_to_head_episodes=args.long_run_min_head_to_head_episodes,
            min_head_to_head_map_episodes=(
                args.long_run_min_head_to_head_map_episodes
            ),
            require_candidate_checkpoint=args.long_run_require_candidate_checkpoint,
            require_candidate_metadata=args.long_run_require_candidate_metadata,
            require_candidate_integrity=args.long_run_require_candidate_integrity,
            required_curriculum_stage=args.long_run_required_curriculum_stage,
            required_reward_preset=args.long_run_required_reward_preset,
            require_head_to_head=args.long_run_require_head_to_head,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
        )
    elif args.mode == "long_run_status":
        run_long_run_status(
            args.artifact_dir,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
        )
    elif args.mode == "league_health":
        run_league_health(
            args.artifact_dir,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
        )
    elif args.mode == "checkpoint_trust_manifest":
        if not args.checkpoint_trust_manifest_output:
            parser.error(
                "--checkpoint-trust-manifest-output is required for "
                "checkpoint_trust_manifest mode"
            )
        try:
            rank_checkpoints = parse_rank_checkpoints(args.rank_checkpoints)
            explicit_checkpoints = (
                rank_checkpoints
                if rank_checkpoints is not None
                else ((args.checkpoint,) if args.checkpoint else None)
            )
            run_checkpoint_trust_manifest(
                args.checkpoint_dir,
                explicit_checkpoints,
                args.checkpoint_trust_manifest_output,
            )
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    elif args.mode == "long_run_manifest":
        try:
            suite_opponents = (
                parse_builtin_opponents(args.suite_opponents)
                if args.suite_opponents
                else ("idle", "scripted", "aggressive", "evasive")
            )
            suite_maps = (
                parse_suite_maps(args.suite_maps, cfg)
                if args.suite_maps
                else ("classic", "flat", "split", "tower")
            )
            required_maps = (
                parse_suite_maps(args.long_run_required_maps, cfg)
                if args.long_run_required_maps
                else suite_maps
            )
        except ValueError as exc:
            parser.error(str(exc))

        try:
            run_long_run_manifest(
                run_id=args.run_id,
                checkpoint_root=args.checkpoint_dir,
                eval_root=args.artifact_dir,
                replay_root=args.replay_dir,
                timesteps=args.timesteps or 5_000_000,
                suite_opponents=",".join(suite_opponents),
                suite_maps=",".join(suite_maps),
                rounds=args.rounds or 20,
                replay_samples_per_bucket=args.replay_samples_per_bucket,
                replay_save_interval=args.replay_save_interval,
                opponent_pool_seed=args.opponent_pool_seed,
                rank_min_score=args.rank_min_score,
                rank_min_win_rate=args.rank_min_win_rate,
                rank_max_draw_rate=args.rank_max_draw_rate,
                rank_max_no_damage_rate=args.rank_max_no_damage_rate,
                rank_max_low_engagement_rate=args.rank_max_low_engagement_rate,
                rank_min_map_score=(
                    args.rank_min_map_score
                    if args.rank_min_map_score is not None
                    else 0.0
                ),
                strategy_max_draw_rate=args.strategy_max_draw_rate,
                strategy_max_no_damage_rate=args.strategy_max_no_damage_rate,
                strategy_max_low_engagement_rate=args.strategy_max_low_engagement_rate,
                strategy_max_idle_rate=args.strategy_max_idle_rate,
                strategy_max_dominant_action_rate=args.strategy_max_dominant_action_rate,
                strategy_max_weaknesses=args.strategy_max_weaknesses,
                require_replay_analysis=True,
                min_maps=args.long_run_min_maps,
                required_maps=required_maps,
                min_eval_episodes=args.long_run_min_eval_episodes or None,
                min_map_episodes=args.long_run_min_map_episodes,
                min_map_score=(
                    args.long_run_min_map_score
                    if args.long_run_min_map_score is not None
                    else 0.0
                ),
                min_replay_combat_maps=(
                    args.long_run_min_replay_combat_maps or None
                ),
                min_opponent_historical_samples=(
                    args.long_run_min_opponent_historical_samples
                ),
                min_head_to_head_episodes=(
                    args.long_run_min_head_to_head_episodes or None
                ),
                min_head_to_head_map_episodes=(
                    args.long_run_min_head_to_head_map_episodes
                ),
                require_candidate_checkpoint=True,
                require_candidate_metadata=True,
                require_candidate_integrity=True,
                required_curriculum_stage=(
                    args.long_run_required_curriculum_stage or "full_map_pool"
                ),
                required_reward_preset=(
                    args.long_run_required_reward_preset or "anti_stall"
                ),
                require_head_to_head=(
                    True if args.long_run_require_head_to_head else None
                ),
                output_dir=args.eval_output_dir,
                output_label=args.eval_label,
            )
        except ValueError as exc:
            parser.error(str(exc))
    elif args.mode == "suite":
        try:
            suite_opponents = parse_builtin_opponents(args.suite_opponents)
            suite_maps = parse_suite_maps(args.suite_maps, cfg)
        except ValueError as exc:
            parser.error(str(exc))

        run_suite(
            cfg,
            args.checkpoint,
            args.agent_policy,
            suite_opponents,
            suite_maps,
            num_rounds=args.rounds,
            seed=args.seed,
            deterministic=not args.stochastic,
            reward_preset=args.reward_preset or "default",
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
            trusted_checkpoint_manifest=trusted_checkpoint_manifest,
            allow_unverified_checkpoints=args.allow_unverified_checkpoints,
        )
    elif args.mode == "rank":
        try:
            suite_opponents = parse_builtin_opponents(args.suite_opponents)
            suite_maps = parse_suite_maps(args.suite_maps, cfg)
            rank_checkpoints = parse_rank_checkpoints(args.rank_checkpoints)
        except ValueError as exc:
            parser.error(str(exc))

        run_rank(
            cfg,
            rank_checkpoints,
            args.checkpoint_dir,
            suite_opponents,
            suite_maps,
            num_rounds=args.rounds,
            seed=args.seed,
            deterministic=not args.stochastic,
            reward_preset=args.reward_preset or "default",
            draw_weight=args.rank_draw_weight,
            no_damage_penalty=args.rank_no_damage_penalty,
            low_engagement_penalty=args.rank_low_engagement_penalty,
            include_head_to_head=args.rank_head_to_head,
            initial_elo=args.rank_initial_elo,
            elo_k_factor=args.rank_elo_k,
            output_dir=args.eval_output_dir,
            output_label=args.eval_label,
            trusted_checkpoint_manifest=trusted_checkpoint_manifest,
            allow_unverified_checkpoints=args.allow_unverified_checkpoints,
        )


if __name__ == "__main__":
    main()
