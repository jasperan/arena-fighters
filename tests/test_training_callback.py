from dataclasses import replace
import json
from pathlib import Path
import shlex
import sys

from arena_fighters.config import Config, IDLE, reward_config_for_preset
from arena_fighters.evaluation import artifact_metadata
from arena_fighters.self_play import OpponentPool
from scripts.train import (
    SelfPlayCallback,
    artifact_index_contains_path,
    build_artifact_index,
    build_long_run_check,
    build_long_run_manifest,
    build_long_run_status,
    build_league_health_report,
    build_replay_analysis_batch,
    build_strategy_report,
    build_training_wrapper,
    candidate_per_map_scores,
    checkpoint_file_sha256,
    checkpoint_metadata_integrity,
    checkpoint_metadata_maps,
    checkpoint_metadata,
    checkpoint_trust_manifest,
    curriculum_metadata,
    discover_checkpoints,
    effective_reward_config,
    load_checkpoint_trust_manifest,
    load_trusted_ppo_checkpoint,
    main,
    missing_required_maps,
    parse_builtin_opponents,
    parse_csv_tuple,
    parse_rank_checkpoints,
    parse_suite_maps,
    rank_evaluation_episode_counts,
    read_checkpoint_metadata,
    run_artifact_index,
    run_audit_summary,
    run_analyze_replay,
    run_analyze_replay_dir,
    run_compare,
    run_eval,
    run_gate,
    run_long_run_check,
    run_long_run_manifest,
    run_long_run_status,
    run_league_health,
    run_promotion_audit,
    run_rank_gate,
    run_suite,
    run_strategy_report,
    summarize_promotion_audit,
    verify_checkpoint_trust,
    write_checkpoint_metadata,
)


class FakeWrapper:
    def __init__(self):
        self.map_pools = []
        self.reward_configs = []

    def set_map_pool(self, map_choices):
        self.map_pools.append(map_choices)

    def set_reward_config(self, reward_config):
        self.reward_configs.append(reward_config)


class FakeLogger:
    def __init__(self):
        self.records = {}

    def record(self, key, value):
        self.records[key] = value


class FakeModelWithLogger:
    def __init__(self):
        self.logger = FakeLogger()


class FakePredictModel:
    def predict(self, obs, deterministic=True):
        return IDLE, None


def _clean_source_snapshot():
    return {
        "vcs": "git",
        "available": True,
        "commit": "clean-commit",
        "dirty": False,
        "status_short_count": 0,
    }


def _write_replay(
    path: Path,
    *,
    episode_id: int,
    winner: str,
    damage_dealt: int,
    map_name: str = "classic",
) -> None:
    path.write_text(
        json.dumps(
            {
                "episode_id": episode_id,
                "winner": winner,
                "length": 3,
                "map_name": map_name,
                "event_totals": {
                    "agent_0": {
                        "shots_fired": 1 if damage_dealt else 0,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 1 if damage_dealt else 0,
                        "damage_dealt": damage_dealt,
                        "damage_taken": 0,
                    },
                    "agent_1": {
                        "shots_fired": 0,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 0,
                        "damage_dealt": 0,
                        "damage_taken": damage_dealt,
                    },
                },
                "frames": [
                    {
                        "tick": 3,
                        "map_name": map_name,
                        "agents": {
                            "agent_0": {"hp": 30},
                            "agent_1": {"hp": 20},
                        },
                    }
                ],
            }
        )
        + "\n"
    )


def _eval_summary(
    label: str,
    win_rate: float = 0.5,
    draw_rate: float = 0.0,
    idle_rate: float = 0.1,
    no_damage_episodes: int = 0,
    low_engagement_episodes: int = 0,
):
    return {
        "artifact": artifact_metadata("eval"),
        "eval_config": {"label": label},
        "episodes": 4,
        "win_rate_agent_0": win_rate,
        "win_rate_agent_1": 1.0 - win_rate - draw_rate,
        "draw_rate": draw_rate,
        "avg_length": 10.0,
        "avg_rewards": {"agent_0": -1.0, "agent_1": -1.0},
        "damage_dealt": {"agent_0": 2.0, "agent_1": 1.0},
        "behavior": {
            "avg_idle_rate": {"agent_0": idle_rate, "agent_1": 0.1},
            "avg_dominant_action_rate": {"agent_0": 0.4, "agent_1": 0.4},
            "damage_events": {"agent_0": 2, "agent_1": 1},
            "no_damage_episodes": no_damage_episodes,
            "low_engagement_episodes": low_engagement_episodes,
        },
    }


def _rank_summary(
    label: str = "candidate",
    score: float = 0.5,
    win_rate: float = 0.5,
    no_damage_rate: float = 0.0,
    low_engagement_rate: float = 0.0,
):
    return {
        "artifact": artifact_metadata("rank"),
        "rankings": [
            {
                "label": label,
                "score": score,
                "mean_win_rate_agent_0": win_rate,
                "mean_no_damage_rate": no_damage_rate,
                "mean_low_engagement_rate": low_engagement_rate,
            }
        ],
    }


def _promotion_audit_summary(passed: bool = True):
    return {
        "artifact": artifact_metadata("promotion_audit"),
        "passed": passed,
        "rank_artifact_path": "evals/rank.json",
        "rank_gate_artifact_path": "evals/rank-gate.json",
        "ranking_labels": ["candidate", "older"],
        "rules": {"min_score": 0.1},
        "candidate": {
            "label": "candidate",
            "checkpoint": "checkpoints/candidate.zip",
            "rank": 1,
            "score": 0.5,
            "mean_win_rate_agent_0": 0.5,
            "mean_no_damage_rate": 0.0,
            "mean_low_engagement_rate": 0.0,
        },
        "failures": [] if passed else [{"metric": "score", "reason": "below"}],
    }


def _long_run_promotion_audit():
    summary = _promotion_audit_summary()
    summary["candidate"]["matchup_scores"] = [
        {"map_name": "classic", "score": 0.5, "episodes": 20},
        {"map_name": "flat", "score": 0.5, "episodes": 20},
    ]
    summary["rank"] = {
        "artifact": artifact_metadata("rank"),
        "rank_config": {
            "checkpoints": ["checkpoints/candidate.zip"],
            "maps": ["classic", "flat"],
            "opponents": ["idle", "scripted"],
            "episodes_per_matchup": 10,
        },
        "rankings": [
            {
                "label": "candidate",
                "checkpoint": "checkpoints/candidate.zip",
                "rank": 1,
                "score": 0.5,
            }
        ],
        "suites": [
            {
                "label": "candidate",
                "checkpoint": "checkpoints/candidate.zip",
                "suite": {
                    "matchups": {
                        "classic": {
                            "idle": {"episodes": 10},
                            "scripted": {"episodes": 10},
                        },
                        "flat": {
                            "idle": {"episodes": 10},
                            "scripted": {"episodes": 10},
                        },
                    }
                },
            }
        ],
        "head_to_head": {
            "overview": {"total_episodes": 4},
            "standings": [
                {"label": "candidate", "score": 1.0},
                {"label": "older", "score": 0.0},
            ],
            "matchups": {
                "classic": {
                    "candidate__vs__older": {
                        "forward": {"episodes": 1},
                        "reverse": {"episodes": 1},
                    }
                },
                "flat": {
                    "candidate__vs__older": {
                        "forward": {"episodes": 1},
                        "reverse": {"episodes": 1},
                    }
                },
            },
        },
    }
    return summary


def _long_run_strategy_report(
    candidate_issue: bool = False,
    replay_issue: bool = False,
):
    issues = []
    if candidate_issue:
        issues.append(
            {
                "scope": "candidate:candidate",
                "metric": "mean_no_damage_rate",
                "value": 1.0,
            }
        )
    if replay_issue:
        issues.append(
            {
                "scope": "replay:episode_0001",
                "metric": "replay_idle_rate_agent_0",
                "value": 1.0,
                "threshold": 0.75,
            }
        )
    return {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": len(issues),
        "issues": issues,
    }


def _long_run_artifact_index(
    replay_analysis: bool = True,
    replay_maps: tuple[str, ...] = ("classic",),
):
    artifacts = []
    counts = {
        "promotion_audit": 1,
        "strategy_report": 1,
        "rank": 1,
        "rank_gate": 1,
    }
    if replay_analysis:
        counts["replay_analysis"] = len(replay_maps)
        for map_name in replay_maps:
            artifacts.append(
                {
                    "artifact_type": "replay_analysis",
                    "summary": {
                        "map_name": map_name,
                        "flags": {"no_damage": False},
                        "totals": {"damage_dealt": 10},
                    },
                }
            )
    return {
        "artifact": artifact_metadata("artifact_index"),
        "artifact_counts": counts,
        "artifacts": artifacts,
    }


def test_self_play_callback_applies_curriculum_stages():
    wrapper = FakeWrapper()
    callback = SelfPlayCallback(
        wrapper=wrapper,
        opponent_pool=OpponentPool(),
        cfg=replace(
            Config(),
            training=replace(Config().training, curriculum_name="map_progression"),
        ),
        curriculum_name="map_progression",
    )

    callback.num_timesteps = 0
    callback._apply_curriculum()
    callback.num_timesteps = 250_000
    callback._apply_curriculum()

    assert wrapper.map_pools == [
        ("flat",),
        ("flat", "classic"),
    ]
    assert wrapper.reward_configs == [
        reward_config_for_preset("default"),
        reward_config_for_preset("default"),
    ]


def test_self_play_callback_applies_curriculum_reward_presets():
    wrapper = FakeWrapper()
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(cfg.training, curriculum_name="map_progression"),
    )
    callback = SelfPlayCallback(
        wrapper=wrapper,
        opponent_pool=OpponentPool(),
        cfg=cfg,
        curriculum_name="map_progression",
    )

    callback.num_timesteps = 0
    callback._apply_curriculum()
    callback.num_timesteps = 1_000_000
    callback._apply_curriculum()

    assert wrapper.map_pools == [
        ("flat",),
        ("classic", "split"),
    ]
    assert wrapper.reward_configs == [
        reward_config_for_preset("default"),
        reward_config_for_preset("anti_stall"),
    ]


def test_build_training_wrapper_wires_replay_logger(tmp_path):
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(
            cfg.training,
            replay_save_interval=7,
            opponent_pool_seed=123,
        ),
    )

    wrapper, pool = build_training_wrapper(cfg, str(tmp_path))
    expected_pool = OpponentPool(max_size=cfg.training.opponent_pool_size, seed=123)
    for i in range(5):
        pool.add({"weight": i})
        expected_pool.add({"weight": i})

    assert isinstance(pool, OpponentPool)
    assert [pool.sample(latest_prob=0.4)["weight"] for _ in range(10)] == [
        expected_pool.sample(latest_prob=0.4)["weight"] for _ in range(10)
    ]
    assert wrapper.replay_logger is not None
    assert wrapper.replay_logger.replay_dir == tmp_path
    assert wrapper.replay_logger.save_every_n == 7


def test_self_play_callback_records_opponent_pool_stats():
    pool = OpponentPool()
    pool.add({"weight": 0})
    pool.add({"weight": 1})
    pool.sample(latest_prob=1.0)
    pool.sample(latest_prob=0.0)
    callback = SelfPlayCallback(
        wrapper=FakeWrapper(),
        opponent_pool=pool,
        cfg=Config(),
    )
    model = FakeModelWithLogger()
    callback.model = model

    callback._record_self_play_stats()

    assert model.logger.records == {
        "self_play/opponent_pool_size": 2,
        "self_play/latest_opponent_samples": 1,
        "self_play/historical_opponent_samples": 1,
        "self_play/historical_sample_rate": 0.5,
        "self_play/latest_opponent_snapshot_id": 1,
        "self_play/last_opponent_snapshot_id": 0,
        "self_play/last_sample_was_historical": 1.0,
    }


def test_curriculum_metadata_records_active_stage():
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(cfg.training, curriculum_name="map_progression"),
    )

    metadata = curriculum_metadata(cfg, step=1_500_000)

    assert metadata["name"] == "map_progression"
    assert metadata["stage"]["name"] == "mixed_routes"
    assert metadata["active_map_pool"] == ["classic", "split"]


def test_checkpoint_metadata_includes_curriculum_state():
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(
            cfg.training,
            curriculum_name="map_progression",
            opponent_pool_seed=123,
        ),
    )

    metadata = checkpoint_metadata(cfg, num_timesteps=3_000_000)

    assert metadata["num_timesteps"] == 3_000_000
    assert metadata["curriculum"]["stage"]["name"] == "full_map_pool"
    assert metadata["curriculum"]["active_reward_preset"] == "anti_stall"
    assert metadata["reward"] == reward_config_for_preset("anti_stall").__dict__
    assert metadata["opponent_pool_config"] == {
        "max_size": cfg.training.opponent_pool_size,
        "latest_opponent_prob": cfg.training.latest_opponent_prob,
        "seed": 123,
    }


def test_checkpoint_metadata_can_include_opponent_pool_stats():
    metadata = checkpoint_metadata(
        Config(),
        num_timesteps=100,
        opponent_pool_stats={
            "size": 3,
            "historical_samples": 7,
            "historical_sample_rate": 0.35,
        },
    )

    assert metadata["opponent_pool"] == {
        "size": 3,
        "historical_samples": 7,
        "historical_sample_rate": 0.35,
    }


def test_effective_reward_config_uses_curriculum_stage_reward():
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(cfg.training, curriculum_name="map_progression"),
    )

    assert effective_reward_config(cfg, 0) == reward_config_for_preset("default")
    assert effective_reward_config(cfg, 1_000_000) == reward_config_for_preset(
        "anti_stall"
    )


def test_checkpoint_metadata_round_trip_supports_zip_checkpoint_paths(tmp_path):
    cfg = Config()
    cfg = replace(
        cfg,
        training=replace(cfg.training, curriculum_name="map_progression"),
    )
    checkpoint_path = tmp_path / "ppo_final"

    metadata_path = write_checkpoint_metadata(
        checkpoint_path,
        cfg,
        num_timesteps=250_000,
    )
    loaded = read_checkpoint_metadata(tmp_path / "ppo_final.zip")

    assert metadata_path.name == "ppo_final.meta.json"
    assert loaded["num_timesteps"] == 250_000
    assert loaded["curriculum"]["stage"]["name"] == "classic_duel"


def test_checkpoint_metadata_records_checkpoint_file_digest(tmp_path):
    checkpoint = tmp_path / "ppo_final.zip"
    checkpoint.write_bytes(b"checkpoint-bytes")

    metadata_path = write_checkpoint_metadata(
        tmp_path / "ppo_final",
        Config(),
        num_timesteps=100,
    )
    loaded = read_checkpoint_metadata(checkpoint)

    assert metadata_path.name == "ppo_final.meta.json"
    assert loaded["checkpoint_file"] == {
        "file_name": "ppo_final.zip",
        "size_bytes": len(b"checkpoint-bytes"),
        "sha256": checkpoint_file_sha256(checkpoint),
    }


def test_checkpoint_metadata_integrity_detects_stale_sidecar(tmp_path):
    checkpoint = tmp_path / "ppo_final.zip"
    checkpoint.write_bytes(b"original")
    write_checkpoint_metadata(tmp_path / "ppo_final", Config(), num_timesteps=100)
    metadata = read_checkpoint_metadata(checkpoint)

    passing = checkpoint_metadata_integrity(checkpoint, metadata)
    checkpoint.write_bytes(b"changed")
    failing = checkpoint_metadata_integrity(checkpoint, metadata)

    assert passing["passed"] is True
    assert failing["passed"] is False
    assert failing["reason"] == "sha256_mismatch"


def test_verify_checkpoint_trust_rejects_sidecar_metadata_as_trust(tmp_path):
    checkpoint = tmp_path / "ppo_final.zip"
    checkpoint.write_bytes(b"checkpoint-bytes")
    write_checkpoint_metadata(tmp_path / "ppo_final", Config(), num_timesteps=100)

    try:
        verify_checkpoint_trust(checkpoint)
    except ValueError as exc:
        assert "sidecar metadata only proves file integrity" in str(exc)
    else:
        raise AssertionError("expected sidecar metadata not to establish trust")


def test_checkpoint_trust_manifest_records_resolved_checkpoint_keys(tmp_path):
    checkpoint = tmp_path / "ppo_final.zip"
    checkpoint.write_bytes(b"checkpoint-bytes")
    digest = checkpoint_file_sha256(checkpoint)

    manifest = checkpoint_trust_manifest((tmp_path / "ppo_final",))

    assert manifest["artifact"] == {
        "artifact_type": "checkpoint_trust_manifest",
        "schema_version": 1,
    }
    assert manifest["checkpoints"][str(tmp_path / "ppo_final")]["sha256"] == digest
    assert manifest["checkpoints"][checkpoint.name]["sha256"] == digest
    assert manifest["checkpoints"]["ppo_final"]["sha256"] == digest


def test_load_checkpoint_trust_manifest_accepts_mapping_shapes(tmp_path):
    checkpoint = tmp_path / "ppo_final.zip"
    checkpoint.write_bytes(b"checkpoint-bytes")
    digest = checkpoint_file_sha256(checkpoint)
    manifest_path = tmp_path / "trusted-checkpoints.json"
    manifest_path.write_text(
        json.dumps({"checkpoints": {checkpoint.name: {"sha256": digest}}}) + "\n"
    )

    trusted = load_checkpoint_trust_manifest(manifest_path)
    trust = verify_checkpoint_trust(
        checkpoint,
        trusted_checkpoint_manifest=trusted,
    )

    assert trusted == {checkpoint.name: digest}
    assert trust["verified"] is True
    assert trust["verification_source"] == "trusted_manifest"


def test_load_trusted_ppo_checkpoint_rejects_unverified_before_load(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "external.zip"
    checkpoint.write_bytes(b"external-checkpoint")
    load_calls = []

    def fake_load(path):
        load_calls.append(path)
        return object()

    monkeypatch.setattr("stable_baselines3.PPO.load", fake_load)

    try:
        load_trusted_ppo_checkpoint(checkpoint)
    except ValueError as exc:
        assert "Refusing to load checkpoint before trust verification" in str(exc)
    else:
        raise AssertionError("expected unverified checkpoint to be rejected")

    assert load_calls == []


def test_load_trusted_ppo_checkpoint_allows_explicit_unverified_override(
    tmp_path,
    monkeypatch,
):
    checkpoint = tmp_path / "legacy-local.zip"
    checkpoint.write_bytes(b"legacy-local-checkpoint")
    sentinel = object()
    load_calls = []

    def fake_load(path):
        load_calls.append(path)
        return sentinel

    monkeypatch.setattr("stable_baselines3.PPO.load", fake_load)

    loaded = load_trusted_ppo_checkpoint(checkpoint, allow_unverified=True)

    assert loaded is sentinel
    assert load_calls == [str(checkpoint)]


def test_discover_checkpoints_uses_metadata_order_and_ignores_sidecars(tmp_path):
    cfg = Config()
    first = tmp_path / "ppo_100.zip"
    second = tmp_path / "ppo_200.zip"
    first.touch()
    second.touch()
    (tmp_path / "notes.txt").write_text("ignore me\n")
    write_checkpoint_metadata(tmp_path / "ppo_100", cfg, num_timesteps=100)
    write_checkpoint_metadata(tmp_path / "ppo_200", cfg, num_timesteps=200)

    discovered = discover_checkpoints(tmp_path)

    assert [Path(path).name for path in discovered] == [
        "ppo_100.zip",
        "ppo_200.zip",
    ]


def test_parse_csv_tuple_rejects_empty_values():
    assert parse_csv_tuple("idle, scripted", "--suite-opponents") == (
        "idle",
        "scripted",
    )

    try:
        parse_csv_tuple(" , ", "--suite-opponents")
    except ValueError as exc:
        assert "--suite-opponents must include at least one value" in str(exc)
    else:
        raise AssertionError("expected empty csv value to fail")


def test_parse_builtin_opponents_validates_names():
    assert parse_builtin_opponents("idle,evasive") == ("idle", "evasive")

    try:
        parse_builtin_opponents("idle,missing")
    except ValueError as exc:
        assert "Unknown opponent names: missing" in str(exc)
    else:
        raise AssertionError("expected unknown opponent to fail")


def test_parse_suite_maps_defaults_and_validates_names():
    cfg = Config()
    cfg = replace(
        cfg,
        arena=replace(
            cfg.arena,
            randomize_maps=True,
            map_choices=("classic", "flat"),
        ),
    )

    assert parse_suite_maps(None, cfg) == ("classic", "flat")
    assert parse_suite_maps("flat,tower", cfg) == ("flat", "tower")

    try:
        parse_suite_maps("flat,missing", cfg)
    except ValueError as exc:
        assert "Unknown map names: missing" in str(exc)
    else:
        raise AssertionError("expected unknown map to fail")


def test_parse_rank_checkpoints_reuses_csv_validation():
    assert parse_rank_checkpoints(None) is None
    assert parse_rank_checkpoints("a.zip,b.zip") == ("a.zip", "b.zip")

    try:
        parse_rank_checkpoints(" , ")
    except ValueError as exc:
        assert "--rank-checkpoints must include at least one value" in str(exc)
    else:
        raise AssertionError("expected empty rank checkpoint list to fail")


def test_run_eval_includes_curriculum_metadata(capsys):
    cfg = Config()
    cfg = replace(
        cfg,
        arena=replace(cfg.arena, max_ticks=3),
        training=replace(cfg.training, curriculum_name="map_progression"),
    )

    run_eval(
        cfg,
        checkpoint=None,
        opponent="idle",
        num_rounds=1,
        seed=7,
        deterministic=True,
        reward_preset="default",
        output_dir=None,
        output_label=None,
    )
    summary = json.loads(capsys.readouterr().out)

    assert summary["artifact"] == {"artifact_type": "eval", "schema_version": 1}
    assert summary["eval_config"]["checkpoint_metadata"] is None
    assert summary["eval_config"]["curriculum"]["stage"]["name"] == "flat_intro"


def test_run_eval_can_use_builtin_agent_policy_and_cumulative_rewards(capsys):
    cfg = Config()
    cfg = replace(
        cfg,
        arena=replace(cfg.arena, max_ticks=3),
        reward=reward_config_for_preset("anti_stall"),
    )

    run_eval(
        cfg,
        checkpoint=None,
        opponent="idle",
        num_rounds=1,
        seed=7,
        deterministic=True,
        reward_preset="anti_stall",
        output_dir=None,
        output_label=None,
        agent_policy="idle",
    )
    summary = json.loads(capsys.readouterr().out)

    expected_reward = (
        cfg.reward.draw
        + cfg.reward.no_damage_draw_penalty
        + cfg.arena.max_ticks * cfg.reward.idle_penalty
    )
    assert summary["agent_0_policy"] == "idle"
    assert summary["eval_config"]["agent_policy"] == "idle"
    assert abs(summary["avg_rewards"]["agent_0"] - expected_reward) < 1e-9
    assert summary["behavior"]["avg_idle_rate"]["agent_0"] == 1.0


def test_run_eval_resolves_extensionless_checkpoint_before_loading(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint = tmp_path / "ppo_final.zip"
    checkpoint.write_bytes(b"checkpoint-bytes")
    trusted = {str(tmp_path / "ppo_final"): checkpoint_file_sha256(checkpoint)}
    load_calls = []

    def fake_load(path):
        load_calls.append(path)
        return FakePredictModel()

    monkeypatch.setattr("stable_baselines3.PPO.load", fake_load)
    cfg = replace(Config(), arena=replace(Config().arena, max_ticks=1))

    run_eval(
        cfg,
        checkpoint=str(tmp_path / "ppo_final"),
        opponent="idle",
        num_rounds=1,
        seed=7,
        deterministic=True,
        reward_preset="default",
        output_dir=None,
        output_label=None,
        trusted_checkpoint_manifest=trusted,
    )
    summary = json.loads(capsys.readouterr().out)

    assert load_calls == [str(checkpoint)]
    assert summary["agent_0_policy"] == str(checkpoint)


def test_run_suite_includes_curriculum_metadata(capsys):
    cfg = Config()
    cfg = replace(
        cfg,
        arena=replace(cfg.arena, max_ticks=3),
        training=replace(cfg.training, curriculum_name="map_progression"),
    )

    run_suite(
        cfg,
        checkpoint=None,
        agent_policy="random",
        opponents=("idle",),
        maps=("flat",),
        num_rounds=1,
        seed=7,
        deterministic=True,
        reward_preset="default",
        output_dir=None,
        output_label=None,
    )
    suite = json.loads(capsys.readouterr().out)

    assert suite["artifact"] == {"artifact_type": "suite", "schema_version": 1}
    assert suite["suite_config"]["checkpoint_metadata"] is None
    assert suite["suite_config"]["curriculum"]["stage"]["name"] == "flat_intro"


def test_run_suite_can_use_builtin_agent_policy(capsys):
    cfg = Config()
    cfg = replace(
        cfg,
        arena=replace(cfg.arena, max_ticks=3),
        reward=reward_config_for_preset("anti_stall"),
    )

    run_suite(
        cfg,
        checkpoint=None,
        agent_policy="idle",
        opponents=("idle",),
        maps=("flat",),
        num_rounds=1,
        seed=7,
        deterministic=True,
        reward_preset="anti_stall",
        output_dir=None,
        output_label=None,
    )
    suite = json.loads(capsys.readouterr().out)
    matchup = suite["matchups"]["flat"]["idle"]

    expected_reward = (
        cfg.reward.draw
        + cfg.reward.no_damage_draw_penalty
        + cfg.arena.max_ticks * cfg.reward.idle_penalty
    )
    assert suite["suite_config"]["agent_0_policy"] == "idle"
    assert matchup["agent_0_policy"] == "idle"
    assert abs(matchup["avg_rewards"]["agent_0"] - expected_reward) < 1e-9


def test_run_suite_resolves_extensionless_checkpoint_metadata(
    tmp_path,
    monkeypatch,
    capsys,
):
    checkpoint = tmp_path / "ppo_final.zip"
    checkpoint.write_bytes(b"checkpoint-bytes")
    write_checkpoint_metadata(tmp_path / "ppo_final", Config(), num_timesteps=100)
    trusted = {str(tmp_path / "ppo_final"): checkpoint_file_sha256(checkpoint)}

    monkeypatch.setattr(
        "stable_baselines3.PPO.load",
        lambda path: FakePredictModel(),
    )
    cfg = replace(Config(), arena=replace(Config().arena, max_ticks=1))

    run_suite(
        cfg,
        checkpoint=str(tmp_path / "ppo_final"),
        agent_policy="random",
        opponents=("idle",),
        maps=("flat",),
        num_rounds=1,
        seed=7,
        deterministic=True,
        reward_preset="default",
        output_dir=None,
        output_label=None,
        trusted_checkpoint_manifest=trusted,
    )
    suite = json.loads(capsys.readouterr().out)

    assert suite["suite_config"]["agent_0_policy"] == str(checkpoint)
    assert suite["suite_config"]["checkpoint_metadata"]["num_timesteps"] == 100


def test_run_compare_can_save_comparison_artifact(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_eval_summary("before", win_rate=0.25)) + "\n")
    after.write_text(json.dumps(_eval_summary("after", win_rate=0.5)) + "\n")
    output_dir = tmp_path / "outputs"

    run_compare(
        str(before),
        str(after),
        output_dir=str(output_dir),
        output_label="cmp",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_cmp.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved comparison summary to" in stdout
    assert saved["artifact"] == {"artifact_type": "comparison", "schema_version": 1}
    assert saved["before_path"] == str(before)
    assert saved["after_path"] == str(after)
    assert saved["deltas"]["win_rate_agent_0"] == 0.25


def test_run_gate_saves_passing_gate_artifact(tmp_path, capsys):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_eval_summary("before")) + "\n")
    after.write_text(json.dumps(_eval_summary("after")) + "\n")
    output_dir = tmp_path / "outputs"

    run_gate(
        str(before),
        str(after),
        output_dir=str(output_dir),
        output_label="promotion gate",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_promotion-gate.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved gate summary to" in stdout
    assert saved["artifact"] == {"artifact_type": "gate", "schema_version": 1}
    assert saved["passed"] is True
    assert saved["comparison"]["before_config"]["label"] == "before"


def test_run_gate_saves_failing_gate_before_exit(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text(json.dumps(_eval_summary("before", win_rate=0.5)) + "\n")
    after.write_text(
        json.dumps(
            _eval_summary(
                "after",
                win_rate=0.0,
                draw_rate=0.25,
                idle_rate=0.25,
                no_damage_episodes=1,
                low_engagement_episodes=1,
            )
        )
        + "\n"
    )
    output_dir = tmp_path / "outputs"

    try:
        run_gate(
            str(before),
            str(after),
            output_dir=str(output_dir),
            output_label="failed-gate",
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected failing gate to exit non-zero")

    [saved_path] = output_dir.glob("*_failed-gate.json")
    saved = json.loads(saved_path.read_text())
    assert saved["passed"] is False
    assert saved["failures"]


def test_run_rank_gate_can_save_passing_artifact(tmp_path, capsys):
    rank_summary = tmp_path / "rank.json"
    rank_summary.write_text(json.dumps(_rank_summary()) + "\n")
    output_dir = tmp_path / "outputs"

    run_rank_gate(
        str(rank_summary),
        min_score=0.1,
        min_win_rate=0.0,
        max_draw_rate=0.9,
        max_no_damage_rate=0.75,
        max_low_engagement_rate=0.5,
        min_head_to_head_elo=None,
        min_head_to_head_score=None,
        output_dir=str(output_dir),
        output_label="rank promotion",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_rank-promotion.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved rank gate summary to" in stdout
    assert saved["artifact"] == {"artifact_type": "rank_gate", "schema_version": 1}
    assert saved["passed"] is True
    assert saved["rank_summary_path"] == str(rank_summary)


def test_run_rank_gate_saves_failing_artifact_before_exit(tmp_path):
    rank_summary = tmp_path / "rank.json"
    rank_summary.write_text(
        json.dumps(
            _rank_summary(
                label="stalled",
                score=0.0,
                no_damage_rate=1.0,
                low_engagement_rate=1.0,
            )
        )
        + "\n"
    )
    output_dir = tmp_path / "outputs"

    try:
        run_rank_gate(
            str(rank_summary),
            min_score=0.1,
            min_win_rate=0.0,
            max_draw_rate=0.9,
            max_no_damage_rate=0.75,
            max_low_engagement_rate=0.5,
            min_head_to_head_elo=None,
            min_head_to_head_score=None,
            output_dir=str(output_dir),
            output_label="failed-rank-gate",
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected failing rank gate to exit non-zero")

    [saved_path] = output_dir.glob("*_failed-rank-gate.json")
    saved = json.loads(saved_path.read_text())
    assert saved["passed"] is False
    assert saved["candidate"]["label"] == "stalled"
    assert saved["failures"]


def test_run_promotion_audit_saves_rank_gate_and_audit_artifacts(
    tmp_path,
    monkeypatch,
    capsys,
):
    def fake_build_rank_summary(**kwargs):
        assert kwargs["checkpoints"] == ("fake.zip",)
        assert kwargs["opponents"] == ("idle",)
        assert kwargs["maps"] == ("flat",)
        return _rank_summary()

    monkeypatch.setattr("scripts.train.build_rank_summary", fake_build_rank_summary)
    output_dir = tmp_path / "outputs"

    run_promotion_audit(
        cfg=Config(),
        checkpoints=("fake.zip",),
        checkpoint_dir="checkpoints",
        opponents=("idle",),
        maps=("flat",),
        num_rounds=1,
        seed=7,
        deterministic=True,
        reward_preset="default",
        draw_weight=0.5,
        no_damage_penalty=0.25,
        low_engagement_penalty=0.25,
        include_head_to_head=False,
        initial_elo=1000.0,
        elo_k_factor=32.0,
        min_score=0.1,
        min_win_rate=0.0,
        max_draw_rate=0.9,
        max_no_damage_rate=0.75,
        max_low_engagement_rate=0.5,
        min_head_to_head_elo=None,
        min_head_to_head_score=None,
        output_dir=str(output_dir),
        output_label="promotion audit",
    )

    stdout = capsys.readouterr().out
    [rank_path] = output_dir.glob("*_promotion-audit-rank.json")
    [gate_path] = output_dir.glob("*_promotion-audit-rank-gate.json")
    [audit_path] = output_dir.glob("*_promotion-audit.json")
    rank = json.loads(rank_path.read_text())
    gate = json.loads(gate_path.read_text())
    audit = json.loads(audit_path.read_text())

    assert "Saved promotion audit summary to" in stdout
    assert rank["artifact"] == {"artifact_type": "rank", "schema_version": 1}
    assert gate["artifact"] == {"artifact_type": "rank_gate", "schema_version": 1}
    assert audit["artifact"] == {
        "artifact_type": "promotion_audit",
        "schema_version": 1,
    }
    assert audit["passed"] is True
    assert audit["audit_config"] == {"include_nested": False}
    assert audit["rank_artifact_path"] == str(rank_path)
    assert audit["rank_gate_artifact_path"] == str(gate_path)
    assert audit["ranking_labels"] == ["candidate"]
    assert "rank" not in audit
    assert "rank_gate" not in audit


def test_run_promotion_audit_can_include_nested_artifacts(tmp_path, monkeypatch):
    def fake_build_rank_summary(**kwargs):
        return _rank_summary()

    monkeypatch.setattr("scripts.train.build_rank_summary", fake_build_rank_summary)
    output_dir = tmp_path / "outputs"

    run_promotion_audit(
        cfg=Config(),
        checkpoints=("fake.zip",),
        checkpoint_dir="checkpoints",
        opponents=("idle",),
        maps=("flat",),
        num_rounds=1,
        seed=7,
        deterministic=True,
        reward_preset="default",
        draw_weight=0.5,
        no_damage_penalty=0.25,
        low_engagement_penalty=0.25,
        include_head_to_head=False,
        initial_elo=1000.0,
        elo_k_factor=32.0,
        min_score=0.1,
        min_win_rate=0.0,
        max_draw_rate=0.9,
        max_no_damage_rate=0.75,
        max_low_engagement_rate=0.5,
        min_head_to_head_elo=None,
        min_head_to_head_score=None,
        output_dir=str(output_dir),
        output_label="nested audit",
        include_nested=True,
    )

    [audit_path] = output_dir.glob("*_nested-audit.json")
    audit = json.loads(audit_path.read_text())
    assert audit["audit_config"] == {"include_nested": True}
    assert audit["rank"]["artifact"] == {"artifact_type": "rank", "schema_version": 1}
    assert audit["rank_gate"]["artifact"] == {
        "artifact_type": "rank_gate",
        "schema_version": 1,
    }


def test_run_promotion_audit_saves_failing_artifacts_before_exit(
    tmp_path,
    monkeypatch,
):
    def fake_build_rank_summary(**kwargs):
        return _rank_summary(
            label="stalled",
            score=0.0,
            no_damage_rate=1.0,
            low_engagement_rate=1.0,
        )

    monkeypatch.setattr("scripts.train.build_rank_summary", fake_build_rank_summary)
    output_dir = tmp_path / "outputs"

    try:
        run_promotion_audit(
            cfg=Config(),
            checkpoints=("fake.zip",),
            checkpoint_dir="checkpoints",
            opponents=("idle",),
            maps=("flat",),
            num_rounds=1,
            seed=7,
            deterministic=True,
            reward_preset="default",
            draw_weight=0.5,
            no_damage_penalty=0.25,
            low_engagement_penalty=0.25,
            include_head_to_head=False,
            initial_elo=1000.0,
            elo_k_factor=32.0,
            min_score=0.1,
            min_win_rate=0.0,
            max_draw_rate=0.9,
            max_no_damage_rate=0.75,
            max_low_engagement_rate=0.5,
            min_head_to_head_elo=None,
            min_head_to_head_score=None,
            output_dir=str(output_dir),
            output_label="failed audit",
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected failing promotion audit to exit non-zero")

    [gate_path] = output_dir.glob("*_failed-audit-rank-gate.json")
    [audit_path] = output_dir.glob("*_failed-audit.json")
    gate = json.loads(gate_path.read_text())
    audit = json.loads(audit_path.read_text())
    assert gate["passed"] is False
    assert audit["passed"] is False
    assert audit["candidate"]["label"] == "stalled"
    assert audit["failures"]


def test_summarize_promotion_audit_returns_compact_fields():
    summary = summarize_promotion_audit(_promotion_audit_summary(passed=False))

    assert summary["artifact"] == {
        "artifact_type": "audit_summary",
        "schema_version": 1,
    }
    assert summary["source_artifact"] == {
        "artifact_type": "promotion_audit",
        "schema_version": 1,
    }
    assert summary["passed"] is False
    assert summary["candidate"] == {
        "label": "candidate",
        "checkpoint": "checkpoints/candidate.zip",
        "rank": 1,
        "score": 0.5,
        "mean_win_rate_agent_0": 0.5,
        "mean_no_damage_rate": 0.0,
        "mean_low_engagement_rate": 0.0,
    }
    assert summary["failures"] == [{"metric": "score", "reason": "below"}]
    assert summary["rank_artifact_path"] == "evals/rank.json"
    assert summary["rank_gate_artifact_path"] == "evals/rank-gate.json"


def test_run_audit_summary_can_save_summary_artifact(tmp_path, capsys):
    audit_path = tmp_path / "promotion.json"
    audit_path.write_text(json.dumps(_promotion_audit_summary()) + "\n")
    output_dir = tmp_path / "outputs"

    run_audit_summary(
        str(audit_path),
        output_dir=str(output_dir),
        output_label="audit skim",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_audit-skim.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved audit summary to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "audit_summary",
        "schema_version": 1,
    }
    assert saved["audit_summary_path"] == str(audit_path)
    assert saved["passed"] is True


def test_run_audit_summary_rejects_non_promotion_audit_artifacts(tmp_path):
    audit_path = tmp_path / "rank.json"
    audit_path.write_text(json.dumps(_rank_summary()) + "\n")

    try:
        run_audit_summary(str(audit_path))
    except ValueError as exc:
        assert "Expected promotion_audit artifact, got rank" in str(exc)
    else:
        raise AssertionError("expected audit summary to reject rank artifact")


def test_build_artifact_index_summarizes_artifacts_and_links(tmp_path):
    eval_path = tmp_path / "before.json"
    comparison_path = tmp_path / "comparison.json"
    audit_path = tmp_path / "promotion.json"
    strategy_path = tmp_path / "strategy.json"
    long_run_check_path = tmp_path / "long-run-check.json"
    eval_path.write_text(json.dumps(_eval_summary("before")) + "\n")
    comparison_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("comparison"),
                "before_path": "evals/before.json",
                "after_path": "evals/after.json",
                "deltas": {
                    "win_rate_agent_0": 0.25,
                    "draw_rate": -0.1,
                },
            }
        )
        + "\n"
    )
    audit_path.write_text(json.dumps(_promotion_audit_summary()) + "\n")
    strategy_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("strategy_report"),
                "issue_count": 2,
                "issues": [
                    {"scope": "candidate:ppo_final", "metric": "draw_rate"},
                    {"scope": "suite:flat/idle", "metric": "no_damage_rate"},
                ],
                "scanned_artifacts": 4,
            }
        )
        + "\n"
    )
    long_run_check_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("long_run_check"),
                "passed": False,
                "candidate": {"label": "ppo_final", "score": 0.25},
                "checks": [
                    {"id": "promotion_audit_passed", "required": True, "passed": True},
                    {
                        "id": "no_candidate_bad_strategy_issues",
                        "required": True,
                        "passed": False,
                    },
                    {
                        "id": "head_to_head_candidate_not_worse",
                        "required": False,
                        "passed": False,
                    },
                ],
            }
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    assert index["artifact"] == {
        "artifact_type": "artifact_index",
        "schema_version": 1,
    }
    assert index["index_config"]["artifact_count"] == 5
    assert index["artifact_counts"] == {
        "comparison": 1,
        "eval": 1,
        "long_run_check": 1,
        "promotion_audit": 1,
        "strategy_report": 1,
    }

    comparison = next(
        entry for entry in index["artifacts"] if entry["artifact_type"] == "comparison"
    )
    assert comparison["relative_path"] == "comparison.json"
    assert comparison["summary"] == {
        "delta_count": 2,
        "win_rate_delta": 0.25,
        "draw_rate_delta": -0.1,
    }
    assert comparison["links"] == {
        "before_path": "evals/before.json",
        "after_path": "evals/after.json",
    }

    audit = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "promotion_audit"
    )
    assert audit["summary"]["passed"] is True
    assert audit["links"]["rank_artifact_path"] == "evals/rank.json"

    strategy = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "strategy_report"
    )
    assert strategy["summary"] == {
        "issue_count": 2,
        "weakness_count": 0,
        "worst_weakness": None,
        "candidate_issue_count": 1,
        "smoke_issue_count": 0,
        "issue_metrics": ["draw_rate", "no_damage_rate"],
        "scanned_artifacts": 4,
    }

    long_run_check = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "long_run_check"
    )
    assert long_run_check["summary"] == {
        "passed": False,
        "candidate_label": "ppo_final",
        "candidate_score": 0.25,
        "required_check_count": 2,
        "failed_required_check_count": 1,
        "failed_required_checks": ["no_candidate_bad_strategy_issues"],
    }


def test_build_artifact_index_summarizes_smoke_suite_artifacts(tmp_path):
    smoke_suite_path = tmp_path / "smoke-suite-summary.json"
    smoke_suite_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("smoke_suite"),
                "smoke_count": 3,
                "smoke_order": [
                    "reward_shaping",
                    "self_play_sampling",
                    "long_run_artifact",
                ],
                "compute_classes": {
                    "reward_shaping": "no_training_eval",
                    "self_play_sampling": "no_training_self_play",
                    "long_run_artifact": "no_training_artifact",
                },
                "smokes": {
                    "reward_shaping": {
                        "strategy_issue_count": 15,
                        "indexed_artifact_count": 11,
                        "idle_rate_delta_agent_0": -0.25,
                        "dominant_action_rate_delta_agent_0": -0.1,
                        "no_damage_episodes_delta": -1,
                        "low_engagement_episodes_delta": -1,
                        "damage_events_delta_agent_0": 2,
                    },
                    "self_play_sampling": {
                        "passed": True,
                        "historical_samples": 12,
                        "unique_maps_seen": 4,
                    },
                    "long_run_artifact": {
                        "health_ready": False,
                        "health_blockers": ["long_run_status_blocked"],
                        "health_warnings": ["missing_rank"],
                    },
                },
            }
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"smoke_suite": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "smoke_suite"
    assert entry["summary"] == {
        "smoke_count": 3,
        "smoke_order": [
            "reward_shaping",
            "self_play_sampling",
            "long_run_artifact",
        ],
        "compute_classes": {
            "reward_shaping": "no_training_eval",
            "self_play_sampling": "no_training_self_play",
            "long_run_artifact": "no_training_artifact",
        },
        "summary_paths": {},
        "reward_strategy_issue_count": 15,
        "reward_indexed_artifact_count": 11,
        "reward_idle_rate_delta_agent_0": -0.25,
        "reward_dominant_action_rate_delta_agent_0": -0.1,
        "reward_no_damage_episodes_delta": -1,
        "reward_low_engagement_episodes_delta": -1,
        "reward_damage_events_delta_agent_0": 2,
        "self_play_sampling_passed": True,
        "self_play_sampling_historical_samples": 12,
        "self_play_sampling_unique_maps_seen": 4,
        "long_run_artifact_health_ready": False,
        "long_run_artifact_health_blockers": ["long_run_status_blocked"],
        "long_run_artifact_health_warnings": ["missing_rank"],
        "train_eval_long_run_check_passed": None,
    }


def test_build_artifact_index_summarizes_self_play_sampling_smoke_artifacts(tmp_path):
    sampling_path = tmp_path / "sampling-summary.json"
    sampling_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("self_play_sampling_smoke"),
                "passed": True,
                "latest_samples": 52,
                "historical_samples": 12,
                "historical_sample_rate": 0.1875,
                "unique_maps_seen": 4,
                "map_counts": {"classic": 12, "flat": 20},
            }
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"self_play_sampling_smoke": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "self_play_sampling_smoke"
    assert entry["summary"] == {
        "passed": True,
        "latest_samples": 52,
        "historical_samples": 12,
        "historical_sample_rate": 0.1875,
        "unique_maps_seen": 4,
        "map_counts": {"classic": 12, "flat": 20},
    }


def test_build_artifact_index_summarizes_reward_shaping_smoke_artifacts(tmp_path):
    reward_summary_path = tmp_path / "reward-summary.json"
    reward_summary_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("reward_shaping_smoke"),
                "reward_delta_agent_0": -13.5,
                "reward_delta_agent_1": -13.5,
                "draw_rate_delta": 0.0,
                "idle_rate_delta_agent_0": -0.25,
                "dominant_action_rate_delta_agent_0": -0.1,
                "no_damage_episodes_delta": -1,
                "low_engagement_episodes_delta": -1,
                "damage_events_delta_agent_0": 2,
                "strategy_issue_count": 15,
                "indexed_artifact_count": 11,
            }
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"reward_shaping_smoke": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "reward_shaping_smoke"
    assert entry["summary"] == {
        "reward_delta_agent_0": -13.5,
        "reward_delta_agent_1": -13.5,
        "draw_rate_delta": 0.0,
        "idle_rate_delta_agent_0": -0.25,
        "dominant_action_rate_delta_agent_0": -0.1,
        "no_damage_episodes_delta": -1,
        "low_engagement_episodes_delta": -1,
        "damage_events_delta_agent_0": 2,
        "strategy_issue_count": 15,
        "indexed_artifact_count": 11,
    }


def test_build_artifact_index_summarizes_long_run_artifact_smoke_artifacts(tmp_path):
    smoke_path = tmp_path / "artifact-smoke-summary.json"
    smoke_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("long_run_artifact_smoke"),
                "run_id": "artifact-smoke",
                "status_blocked_reason": "latest_launcher_not_executed",
                "status_missing_evidence": ["train_exitcode"],
                "health_ready": False,
                "health_blockers": ["long_run_status_blocked"],
                "health_warnings": ["missing_rank"],
                "health_artifact_scope_dir": "/tmp/evals/artifact-smoke",
                "indexed_artifact_counts": {
                    "long_run_manifest": 1,
                    "long_run_status": 1,
                    "league_health": 1,
                },
                "indexed_artifact_count": 3,
            }
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"long_run_artifact_smoke": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "long_run_artifact_smoke"
    assert entry["summary"] == {
        "run_id": "artifact-smoke",
        "status_blocked_reason": "latest_launcher_not_executed",
        "status_missing_evidence": ["train_exitcode"],
        "health_ready": False,
        "health_blockers": ["long_run_status_blocked"],
        "health_warnings": ["missing_rank"],
        "health_artifact_scope_dir": "/tmp/evals/artifact-smoke",
        "indexed_artifact_count": 3,
        "indexed_long_run_manifest_count": 1,
        "indexed_long_run_status_count": 1,
        "indexed_league_health_count": 1,
    }


def test_run_artifact_index_can_save_manifest(tmp_path, capsys):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    (artifact_dir / "promotion.json").write_text(
        json.dumps(_promotion_audit_summary()) + "\n"
    )
    output_dir = tmp_path / "indexes"

    run_artifact_index(
        str(artifact_dir),
        recursive=False,
        output_dir=str(output_dir),
        output_label="artifact manifest",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_artifact-manifest.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved artifact index to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "artifact_index",
        "schema_version": 1,
    }
    assert saved["artifact_counts"] == {"promotion_audit": 1}


def test_build_artifact_index_summarizes_exit_code_sidecars(tmp_path):
    (tmp_path / "promotion-audit.exitcode").write_text("0\n")
    (tmp_path / "long-run-check.exitcode").write_text("1\n")

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"exit_code": 2}
    summaries = {
        Path(entry["path"]).name: entry["summary"]
        for entry in index["artifacts"]
        if entry["artifact_type"] == "exit_code"
    }
    assert summaries == {
        "long-run-check.exitcode": {"exit_code": 1, "passed": False, "raw": "1"},
        "promotion-audit.exitcode": {"exit_code": 0, "passed": True, "raw": "0"},
    }


def test_build_artifact_index_summarizes_shell_script_sidecars(tmp_path):
    launcher = tmp_path / "long-run-launcher.sh"
    launcher.write_text("#!/usr/bin/env bash\nset -euo pipefail\n")
    launcher.chmod(0o755)

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"shell_script": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "shell_script"
    assert entry["summary"] == {
        "line_count": 2,
        "executable": True,
        "starts_with_shebang": True,
    }


def test_build_artifact_index_summarizes_command_logs(tmp_path):
    log_path = tmp_path / "train.out"
    log_path.write_text("\n".join(f"line {idx}" for idx in range(30)) + "\n")

    index = build_artifact_index(tmp_path)

    assert index["artifact_counts"] == {"command_log": 1}
    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "command_log"
    assert entry["summary"]["tail_truncated"] is False
    assert entry["summary"]["tail_byte_limit"] == 8192
    assert entry["summary"]["tail_lines"] == [f"line {idx}" for idx in range(10, 30)]


def test_build_artifact_index_redacts_command_log_secrets(tmp_path):
    log_path = tmp_path / "train.out"
    log_path.write_text(
        "\n".join(
            [
                "api_key=abc123",
                "TOKEN: xyz789",
                "OPENAI_API_KEY=sk-local",
                "AWS_SECRET_ACCESS_KEY = aws-secret",
                "MY_TOKEN: custom-token",
                "client_secret='quoted-secret'",
                "Authorization: Bearer opaque-token",
                "Authorization: Basic opaque-basic-token",
                "Cookie: session=abc123",
                "DATABASE_URL=postgres://user:db-pass@localhost/db",
                "PRIVATE_KEY=-----BEGIN PRIVATE KEY-----abc",
                "password = swordfish",
                "python script.py --api-key abc123 --safe value",
                "python script.py --api-key=abc123 --safe value",
                '{"token":"json-token","safe":"value"}',
                "safe line",
            ]
        )
        + "\n"
    )

    index = build_artifact_index(tmp_path)

    [entry] = index["artifacts"]
    assert entry["artifact_type"] == "command_log"
    assert entry["summary"]["tail_lines"] == [
        "api_key=<redacted>",
        "TOKEN: <redacted>",
        "OPENAI_API_KEY=<redacted>",
        "AWS_SECRET_ACCESS_KEY = <redacted>",
        "MY_TOKEN: <redacted>",
        "client_secret=<redacted>",
        "Authorization: Bearer <redacted>",
        "Authorization: Basic <redacted>",
        "Cookie: <redacted>",
        "DATABASE_URL=<redacted>",
        "PRIVATE_KEY=<redacted>",
        "password = <redacted>",
        "python script.py --api-key <redacted> --safe value",
        "python script.py --api-key=<redacted> --safe value",
        '{"token":"<redacted>","safe":"value"}',
        "safe line",
    ]


def test_build_artifact_index_skips_symlinked_artifacts(tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    outside_log = tmp_path / "outside.out"
    outside_log.write_text("secret_token=outside\n")
    symlink_path = artifact_dir / "train.out"
    try:
        symlink_path.symlink_to(outside_log)
    except OSError:
        return

    index = build_artifact_index(artifact_dir)

    assert index["artifact_counts"] == {}
    assert index["artifacts"] == []


def test_run_analyze_replay_can_save_indexable_artifact(tmp_path, capsys):
    replay_path = tmp_path / "episode_0007.json"
    replay_path.write_text(
        json.dumps(
            {
                "episode_id": 7,
                "winner": "agent_0",
                "length": 3,
                "map_name": "split",
                "event_totals": {
                    "agent_0": {
                        "shots_fired": 1,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 1,
                        "damage_dealt": 10,
                        "damage_taken": 0,
                    },
                    "agent_1": {
                        "shots_fired": 0,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 0,
                        "damage_dealt": 0,
                        "damage_taken": 10,
                    },
                },
                "frames": [
                    {
                        "tick": 3,
                        "map_name": "split",
                        "agents": {
                            "agent_0": {"hp": 30},
                            "agent_1": {"hp": 20},
                        },
                    }
                ],
            }
        )
        + "\n"
    )
    output_dir = tmp_path / "evals"

    run_analyze_replay(
        str(replay_path),
        output_dir=str(output_dir),
        output_label="sample replay",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_sample-replay.json")
    saved = json.loads(saved_path.read_text())
    index = build_artifact_index(output_dir)

    assert "Saved replay analysis to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "replay_analysis",
        "schema_version": 1,
    }
    assert saved["flags"]["no_damage"] is False
    assert index["artifact_counts"] == {"replay_analysis": 1}
    assert index["artifacts"][0]["summary"]["flags"]["no_damage"] is False


def test_build_replay_analysis_batch_selects_representative_buckets(tmp_path):
    _write_replay(
        tmp_path / "agent0_win.json",
        episode_id=1,
        winner="agent_0",
        damage_dealt=10,
    )
    _write_replay(
        tmp_path / "agent1_win.json",
        episode_id=2,
        winner="agent_1",
        damage_dealt=10,
        map_name="split",
    )
    _write_replay(
        tmp_path / "draw_no_damage.json",
        episode_id=3,
        winner="draw",
        damage_dealt=0,
        map_name="tower",
    )

    batch = build_replay_analysis_batch(tmp_path, samples_per_bucket=1)

    assert batch["artifact"] == {
        "artifact_type": "replay_analysis_batch",
        "schema_version": 1,
    }
    assert batch["scanned_replays"] == 3
    assert batch["selected_count"] == 3
    assert {
        bucket: batch["bucket_counts"][bucket]
        for bucket in (
            "agent_0_win",
            "agent_1_win",
            "draw",
            "combat",
            "no_damage",
            "no_attacks",
        )
    } == {
        "agent_0_win": 1,
        "agent_1_win": 1,
        "draw": 1,
        "combat": 1,
        "no_damage": 1,
        "no_attacks": 1,
    }
    assert batch["bucket_counts"]["combat_map:classic"] == 1
    assert batch["bucket_counts"]["combat_map:split"] == 1
    selected_for = {
        bucket
        for item in batch["selected"]
        for bucket in item["selected_for"]
    }
    assert {
        "agent_0_win",
        "agent_1_win",
        "draw",
        "combat",
        "no_damage",
        "no_attacks",
        "combat_map:classic",
        "combat_map:split",
    } == selected_for


def test_build_replay_analysis_batch_selects_combat_samples_per_map(tmp_path):
    _write_replay(
        tmp_path / "flat_combat.json",
        episode_id=1,
        winner="agent_0",
        damage_dealt=10,
        map_name="flat",
    )
    _write_replay(
        tmp_path / "classic_combat.json",
        episode_id=2,
        winner="agent_0",
        damage_dealt=10,
        map_name="classic",
    )

    batch = build_replay_analysis_batch(tmp_path, samples_per_bucket=1)

    assert batch["selected_count"] == 2
    selected_for = {
        bucket
        for item in batch["selected"]
        for bucket in item["selected_for"]
    }
    assert "combat_map:flat" in selected_for
    assert "combat_map:classic" in selected_for


def test_build_replay_analysis_batch_selects_action_collapse_samples(tmp_path):
    replay_path = tmp_path / "idle_heavy.json"
    replay_path.write_text(
        json.dumps(
            {
                "episode_id": 5,
                "winner": "draw",
                "length": 3,
                "map_name": "flat",
                "event_totals": {
                    "agent_0": {
                        "shots_fired": 1,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 1,
                        "damage_dealt": 10,
                        "damage_taken": 0,
                    },
                    "agent_1": {
                        "shots_fired": 0,
                        "melee_attempts": 0,
                        "melee_hits": 0,
                        "projectile_hits": 0,
                        "damage_dealt": 0,
                        "damage_taken": 10,
                    },
                },
                "frames": [
                    {"tick": 0, "map_name": "flat"},
                    {
                        "tick": 1,
                        "map_name": "flat",
                        "actions": {"agent_0": IDLE},
                    },
                    {
                        "tick": 2,
                        "map_name": "flat",
                        "actions": {"agent_0": IDLE},
                    },
                    {
                        "tick": 3,
                        "map_name": "flat",
                        "actions": {"agent_0": IDLE},
                    },
                ],
            }
        )
        + "\n"
    )

    batch = build_replay_analysis_batch(tmp_path, samples_per_bucket=1)

    assert batch["bucket_counts"]["idle_agent_0"] == 1
    assert batch["bucket_counts"]["dominant_action_agent_0"] == 1
    assert {
        "idle_agent_0",
        "dominant_action_agent_0",
    }.issubset(set(batch["selected"][0]["selected_for"]))


def test_run_analyze_replay_dir_saves_selected_artifacts_and_batch(
    tmp_path,
    capsys,
):
    replay_dir = tmp_path / "replays"
    replay_dir.mkdir()
    _write_replay(
        replay_dir / "agent0_win.json",
        episode_id=1,
        winner="agent_0",
        damage_dealt=10,
    )
    _write_replay(
        replay_dir / "draw_no_damage.json",
        episode_id=2,
        winner="draw",
        damage_dealt=0,
    )
    output_dir = tmp_path / "evals"

    run_analyze_replay_dir(
        str(replay_dir),
        samples_per_bucket=1,
        output_dir=str(output_dir),
        output_label="sampled-replays",
    )

    stdout = capsys.readouterr().out
    index = build_artifact_index(output_dir)

    assert "Saved replay analysis batch to" in stdout
    assert index["artifact_counts"] == {
        "replay_analysis": 2,
        "replay_analysis_batch": 1,
    }
    batch_entry = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "replay_analysis_batch"
    )
    assert batch_entry["summary"]["selected_count"] == 2
    assert batch_entry["summary"]["bucket_counts"]["no_damage"] == 1
    assert batch_entry["summary"]["bucket_counts"]["no_attacks"] == 1


def test_build_strategy_report_flags_stalled_eval_and_rank_artifacts(tmp_path):
    eval_summary = _eval_summary(
        "stalled",
        win_rate=0.0,
        draw_rate=1.0,
        idle_rate=0.9,
        no_damage_episodes=4,
        low_engagement_episodes=4,
    )
    eval_summary["behavior"]["avg_dominant_action_rate"]["agent_0"] = 0.99
    rank_summary = _rank_summary(
        label="stalled",
        score=0.0,
        no_damage_rate=1.0,
        low_engagement_rate=1.0,
    )
    rank_summary["rankings"][0]["mean_draw_rate"] = 1.0
    (tmp_path / "stalled-eval.json").write_text(json.dumps(eval_summary) + "\n")
    (tmp_path / "stalled-rank.json").write_text(json.dumps(rank_summary) + "\n")

    report = build_strategy_report(tmp_path)

    assert report["artifact"] == {
        "artifact_type": "strategy_report",
        "schema_version": 1,
    }
    assert report["scanned_artifacts"] == 2
    issue_metrics = {issue["metric"] for issue in report["issues"]}
    assert {
        "draw_rate",
        "no_damage_rate",
        "low_engagement_rate",
        "idle_rate_agent_0",
        "dominant_action_rate_agent_0",
        "mean_draw_rate",
        "mean_no_damage_rate",
        "mean_low_engagement_rate",
    }.issubset(issue_metrics)


def test_build_strategy_report_scans_rank_embedded_suite_behavior(tmp_path):
    rank_summary = _rank_summary()
    rank_summary["suites"] = [
        {
            "label": "candidate",
            "suite": {
                "matchups": {
                    "classic": {
                        "idle": _eval_summary("candidate", idle_rate=0.9),
                    }
                }
            },
        }
    ]
    (tmp_path / "rank.json").write_text(json.dumps(rank_summary) + "\n")

    report = build_strategy_report(tmp_path)

    assert any(
        issue["artifact_type"] == "rank"
        and issue["scope"] == "candidate:candidate:rank_suite:classic/idle"
        and issue["metric"] == "idle_rate_agent_0"
        for issue in report["issues"]
    )


def test_build_strategy_report_ranks_low_score_matchup_weaknesses(tmp_path):
    suite = {
        "artifact": artifact_metadata("suite"),
        "matchups": {
            "flat": {"idle": _eval_summary("weak", win_rate=0.0)},
            "tower": {"scripted": _eval_summary("strong", win_rate=1.0)},
        },
    }
    (tmp_path / "suite.json").write_text(json.dumps(suite) + "\n")

    report = build_strategy_report(tmp_path, max_weaknesses=1)

    assert report["issue_count"] == 0
    assert report["weakness_count"] == 2
    assert report["weaknesses"] == [
        {
            "path": str(tmp_path / "suite.json"),
            "relative_path": "suite.json",
            "artifact_type": "suite",
            "scope": "suite:flat/idle",
            "map_name": "flat",
            "opponent": "idle",
            "score": 0.0,
            "episodes": 4,
            "win_rate_agent_0": 0.0,
            "draw_rate": 0.0,
            "no_damage_rate": 0.0,
            "low_engagement_rate": 0.0,
            "avg_length": 10.0,
        }
    ]


def test_build_strategy_report_extracts_rank_matchup_weaknesses(tmp_path):
    rank = _rank_summary()
    rank["rankings"][0]["matchup_scores"] = [
        {
            "map_name": "classic",
            "opponent": "scripted",
            "score": -0.25,
            "episodes": 4,
            "win_rate_agent_0": 0.0,
            "draw_rate": 0.0,
            "no_damage_rate": 1.0,
            "low_engagement_rate": 0.0,
            "avg_length": 10.0,
        }
    ]
    (tmp_path / "rank.json").write_text(json.dumps(rank) + "\n")

    report = build_strategy_report(tmp_path)

    assert {
        "artifact_type": "rank",
        "scope": "rank:candidate:classic/scripted",
        "label": "candidate",
        "map_name": "classic",
        "opponent": "scripted",
        "score": -0.25,
    }.items() <= report["weaknesses"][0].items()


def test_build_strategy_report_flags_no_damage_replay_analysis(tmp_path):
    replay_summary = {
        "artifact": artifact_metadata("replay_analysis"),
        "episode_id": 7,
        "winner": "draw",
        "map_name": "flat",
        "flags": {"no_damage": True, "no_attacks": True},
        "totals": {"damage_dealt": 0},
    }
    (tmp_path / "no-damage-replay.json").write_text(
        json.dumps(replay_summary) + "\n"
    )

    report = build_strategy_report(tmp_path)

    issue_metrics = {issue["metric"] for issue in report["issues"]}
    assert {
        "replay_no_damage",
        "replay_low_engagement",
        "replay_no_attacks",
    }.issubset(issue_metrics)
    assert all(issue["artifact_type"] == "replay_analysis" for issue in report["issues"])


def test_build_strategy_report_flags_replay_action_collapse(tmp_path):
    replay_summary = {
        "artifact": artifact_metadata("replay_analysis"),
        "episode_id": 11,
        "winner": "draw",
        "map_name": "flat",
        "flags": {"no_damage": False, "no_attacks": False},
        "totals": {"damage_dealt": 10},
        "behavior": {
            "avg_idle_rate": {"agent_0": 1.0, "agent_1": 0.0},
            "avg_dominant_action_rate": {"agent_0": 1.0, "agent_1": 0.5},
        },
    }
    (tmp_path / "idle-replay.json").write_text(json.dumps(replay_summary) + "\n")

    report = build_strategy_report(tmp_path)

    issue_metrics = {issue["metric"] for issue in report["issues"]}
    assert {
        "replay_idle_rate_agent_0",
        "replay_dominant_action_rate_agent_0",
    }.issubset(issue_metrics)


def test_build_strategy_report_flags_candidate_draw_rate(tmp_path):
    promotion = _promotion_audit_summary()
    promotion["candidate"]["mean_draw_rate"] = 1.0
    (tmp_path / "promotion.json").write_text(json.dumps(promotion) + "\n")

    report = build_strategy_report(tmp_path)

    assert {
        "scope": "candidate:candidate",
        "metric": "mean_draw_rate",
        "value": 1.0,
    }.items() <= report["issues"][0].items()


def test_build_strategy_report_flags_long_run_status_missing_historical_samples(
    tmp_path,
):
    status = {
        "artifact": artifact_metadata("long_run_status"),
        "missing_evidence": ["checkpoint_historical_opponent_samples"],
        "latest_manifest": {
            "run_id": "status-run",
            "min_opponent_historical_samples": 1,
            "checkpoint_opponent_pool": {
                "min_opponent_historical_samples": 1,
                "max_historical_samples": 0,
                "meets_min_opponent_historical_samples": False,
            },
        },
    }
    (tmp_path / "status.json").write_text(json.dumps(status) + "\n")

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 1
    assert {
        "artifact_type": "long_run_status",
        "scope": "candidate:status-run:checkpoint_opponent_pool",
        "metric": "checkpoint_historical_opponent_samples",
        "value": 0,
        "threshold": 1,
        "reason": "checkpoint_historical_opponent_samples_below_min",
    }.items() <= report["issues"][0].items()


def test_build_strategy_report_flags_smoke_suite_failures(tmp_path):
    smoke_suite = {
        "artifact": artifact_metadata("smoke_suite"),
        "smokes": {
            "reward_shaping": {
                "strategy_issue_count": 15,
                "indexed_artifact_count": 11,
                "passed": False,
                "checks": [
                    {"id": "reward_delta_agent_0_negative", "passed": False},
                ],
            },
            "long_run_artifact": {
                "health_ready": False,
                "health_blockers": ["long_run_status_blocked"],
                "health_warnings": ["missing_rank"],
                "passed": False,
                "checks": [
                    {"id": "required_artifacts_indexed", "passed": False},
                ],
            },
            "self_play_sampling": {
                "passed": False,
                "checks": [
                    {"id": "historical_samples_meet_minimum", "passed": False},
                ],
            },
            "train_eval": {
                "long_run_check_passed": False,
                "long_run_check_failed_checks": [
                    "no_candidate_bad_strategy_issues",
                ],
                "strategy_issue_count": 2,
            },
        },
    }
    (tmp_path / "smoke-suite-summary.json").write_text(
        json.dumps(smoke_suite) + "\n"
    )

    report = build_strategy_report(tmp_path)

    issue_by_metric = {issue["metric"]: issue for issue in report["issues"]}
    assert report["issue_count"] == 5
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:reward_shaping",
        "metric": "smoke_reward_shaping_failed",
        "value": 1,
        "threshold": 0,
        "reason": "smoke_reward_shaping_checks_failed",
        "failed_checks": ["reward_delta_agent_0_negative"],
    }.items() <= issue_by_metric["smoke_reward_shaping_failed"].items()
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:long_run_artifact",
        "metric": "smoke_long_run_artifact_failed",
        "value": 1,
        "threshold": 0,
        "reason": "smoke_long_run_artifact_checks_failed",
        "failed_checks": ["required_artifacts_indexed"],
        "blockers": ["long_run_status_blocked"],
        "warnings": ["missing_rank"],
    }.items() <= issue_by_metric["smoke_long_run_artifact_failed"].items()
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:self_play_sampling",
        "metric": "smoke_self_play_sampling_failed",
        "value": 1,
        "threshold": 0,
        "reason": "smoke_self_play_sampling_checks_failed",
        "failed_checks": ["historical_samples_meet_minimum"],
    }.items() <= issue_by_metric["smoke_self_play_sampling_failed"].items()
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:train_eval",
        "metric": "smoke_train_eval_strategy_issue_count",
        "value": 2,
        "threshold": 0,
        "reason": "smoke_train_eval_strategy_issues_present",
    }.items() <= issue_by_metric["smoke_train_eval_strategy_issue_count"].items()
    assert {
        "artifact_type": "smoke_suite",
        "scope": "smoke:train_eval",
        "metric": "smoke_train_eval_long_run_check_failed",
        "value": 1,
        "threshold": 0,
        "reason": "smoke_train_eval_long_run_check_failed",
        "failed_checks": ["no_candidate_bad_strategy_issues"],
    }.items() <= issue_by_metric[
        "smoke_train_eval_long_run_check_failed"
    ].items()


def test_build_strategy_report_ignores_healthy_smoke_suite(tmp_path):
    smoke_suite = {
        "artifact": artifact_metadata("smoke_suite"),
        "smokes": {
            "reward_shaping": {
                "strategy_issue_count": 15,
                "passed": True,
            },
            "long_run_artifact": {
                "health_ready": False,
                "health_blockers": ["long_run_status_blocked"],
                "health_warnings": ["missing_rank"],
                "passed": True,
            },
            "self_play_sampling": {
                "passed": True,
                "historical_samples": 12,
            },
            "train_eval": {
                "long_run_check_passed": True,
                "long_run_check_failed_checks": [],
                "strategy_issue_count": 0,
            },
        },
    }
    (tmp_path / "smoke-suite-summary.json").write_text(
        json.dumps(smoke_suite) + "\n"
    )

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 0
    assert report["issues"] == []


def test_build_strategy_report_flags_self_play_sampling_smoke_failures(tmp_path):
    sampling_smoke = {
        "artifact": artifact_metadata("self_play_sampling_smoke"),
        "passed": False,
        "checks": [
            {"id": "historical_samples_meet_minimum", "passed": False},
        ],
    }
    (tmp_path / "sampling-summary.json").write_text(
        json.dumps(sampling_smoke) + "\n"
    )

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 1
    assert {
        "artifact_type": "self_play_sampling_smoke",
        "scope": "smoke:self_play_sampling",
        "metric": "self_play_sampling_smoke_failed",
        "value": 1,
        "threshold": 0,
        "reason": "self_play_sampling_smoke_checks_failed",
        "failed_checks": ["historical_samples_meet_minimum"],
    }.items() <= report["issues"][0].items()


def test_build_strategy_report_flags_reward_shaping_smoke_failures(tmp_path):
    reward_smoke = {
        "artifact": artifact_metadata("reward_shaping_smoke"),
        "reward_delta_agent_0": 0.0,
        "reward_delta_agent_1": 1.25,
        "draw_rate_delta": 0.5,
        "strategy_issue_count": 3,
        "passed": False,
        "checks": [
            {"id": "reward_delta_agent_0_negative", "passed": False},
            {"id": "reward_delta_agent_1_negative", "passed": False},
            {"id": "draw_rate_delta_not_positive", "passed": False},
        ],
    }
    (tmp_path / "reward-summary.json").write_text(json.dumps(reward_smoke) + "\n")

    report = build_strategy_report(tmp_path)

    issue_by_metric = {issue["metric"]: issue for issue in report["issues"]}
    assert report["issue_count"] == 4
    assert {
        "artifact_type": "reward_shaping_smoke",
        "scope": "smoke:reward_shaping",
        "metric": "reward_shaping_smoke_failed",
        "value": 3,
        "threshold": 0,
        "reason": "reward_shaping_smoke_checks_failed",
        "failed_checks": [
            "reward_delta_agent_0_negative",
            "reward_delta_agent_1_negative",
            "draw_rate_delta_not_positive",
        ],
    }.items() <= issue_by_metric["reward_shaping_smoke_failed"].items()
    assert {
        "artifact_type": "reward_shaping_smoke",
        "scope": "smoke:reward_shaping:agent_0",
        "metric": "reward_delta_agent_0",
        "value": 0.0,
        "threshold": 0.0,
        "reason": "anti_stall_idle_reward_not_reduced",
    }.items() <= issue_by_metric["reward_delta_agent_0"].items()
    assert {
        "artifact_type": "reward_shaping_smoke",
        "scope": "smoke:reward_shaping:agent_1",
        "metric": "reward_delta_agent_1",
        "value": 1.25,
        "threshold": 0.0,
        "reason": "anti_stall_idle_reward_not_reduced",
    }.items() <= issue_by_metric["reward_delta_agent_1"].items()
    assert {
        "artifact_type": "reward_shaping_smoke",
        "scope": "smoke:reward_shaping",
        "metric": "draw_rate_delta",
        "value": 0.5,
        "threshold": 0.0,
        "reason": "anti_stall_draw_rate_increased",
    }.items() <= issue_by_metric["draw_rate_delta"].items()


def test_build_strategy_report_ignores_healthy_reward_shaping_smoke(tmp_path):
    reward_smoke = {
        "artifact": artifact_metadata("reward_shaping_smoke"),
        "reward_delta_agent_0": -13.5,
        "reward_delta_agent_1": -13.5,
        "draw_rate_delta": 0.0,
        "strategy_issue_count": 3,
        "passed": True,
    }
    (tmp_path / "reward-summary.json").write_text(json.dumps(reward_smoke) + "\n")

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 0
    assert report["issues"] == []


def test_build_strategy_report_flags_long_run_artifact_smoke_failures(tmp_path):
    artifact_smoke = {
        "artifact": artifact_metadata("long_run_artifact_smoke"),
        "health_ready": False,
        "health_blockers": ["long_run_status_blocked"],
        "health_warnings": ["missing_rank"],
        "passed": False,
        "checks": [
            {"id": "required_artifacts_indexed", "passed": False},
        ],
    }
    (tmp_path / "artifact-smoke-summary.json").write_text(
        json.dumps(artifact_smoke) + "\n"
    )

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 1
    assert {
        "artifact_type": "long_run_artifact_smoke",
        "scope": "smoke:long_run_artifact",
        "metric": "long_run_artifact_smoke_failed",
        "value": 1,
        "threshold": 0,
        "reason": "long_run_artifact_smoke_checks_failed",
        "failed_checks": ["required_artifacts_indexed"],
        "blockers": ["long_run_status_blocked"],
        "warnings": ["missing_rank"],
    }.items() <= report["issues"][0].items()


def test_build_strategy_report_ignores_healthy_long_run_artifact_smoke(tmp_path):
    artifact_smoke = {
        "artifact": artifact_metadata("long_run_artifact_smoke"),
        "health_ready": False,
        "health_blockers": ["long_run_status_blocked"],
        "health_warnings": ["missing_rank"],
        "passed": True,
    }
    (tmp_path / "artifact-smoke-summary.json").write_text(
        json.dumps(artifact_smoke) + "\n"
    )

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 0
    assert report["issues"] == []


def test_build_strategy_report_allows_values_at_max_thresholds(tmp_path):
    eval_summary = _eval_summary(
        "threshold",
        win_rate=0.0,
        draw_rate=0.9,
        idle_rate=0.75,
        no_damage_episodes=3,
        low_engagement_episodes=2,
    )
    eval_summary["behavior"]["avg_dominant_action_rate"]["agent_0"] = 0.95
    rank_summary = _rank_summary(
        label="threshold",
        score=0.1,
        win_rate=0.0,
        no_damage_rate=0.75,
        low_engagement_rate=0.5,
    )
    rank_summary["rankings"][0]["mean_draw_rate"] = 0.9
    (tmp_path / "threshold-eval.json").write_text(json.dumps(eval_summary) + "\n")
    (tmp_path / "threshold-rank.json").write_text(json.dumps(rank_summary) + "\n")

    report = build_strategy_report(tmp_path)

    assert report["issue_count"] == 0
    assert report["issues"] == []


def test_run_strategy_report_can_save_report(tmp_path, capsys):
    eval_summary = _eval_summary(
        "stalled",
        win_rate=0.0,
        draw_rate=1.0,
        no_damage_episodes=4,
    )
    (tmp_path / "stalled-eval.json").write_text(json.dumps(eval_summary) + "\n")
    output_dir = tmp_path / "reports"

    run_strategy_report(
        str(tmp_path),
        recursive=False,
        max_draw_rate=0.9,
        max_no_damage_rate=0.75,
        max_low_engagement_rate=0.5,
        max_idle_rate=0.75,
        max_dominant_action_rate=0.95,
        output_dir=str(output_dir),
        output_label="bad-strategy",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_bad-strategy.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved strategy report to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "strategy_report",
        "schema_version": 1,
    }
    assert saved["issue_count"] >= 2


def test_build_long_run_check_passes_documented_promotion_criteria():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    assert result["artifact"] == {
        "artifact_type": "long_run_check",
        "schema_version": 1,
    }
    assert result["passed"] is True
    assert {check["id"] for check in result["checks"] if check["required"]} == {
        "promotion_audit_passed",
        "no_candidate_bad_strategy_issues",
        "candidate_map_coverage",
        "artifact_index_has_required_artifacts",
        "replay_analysis_has_combat",
        "no_replay_bad_strategy_issues",
        "head_to_head_candidate_not_worse",
    }


def test_build_long_run_check_fails_bad_strategy_and_low_map_coverage():
    promotion = _long_run_promotion_audit()
    promotion["candidate"]["matchup_scores"] = [
        {"map_name": "flat", "score": 0.0, "episodes": 20}
    ]

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(candidate_issue=True),
        _long_run_artifact_index(replay_analysis=False),
        min_maps=2,
        require_replay_analysis=True,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert {
        "no_candidate_bad_strategy_issues",
        "candidate_map_coverage",
        "replay_analysis_has_combat",
    }.issubset(failed)


def test_build_long_run_check_fails_replay_strategy_issues_when_required():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(replay_issue=True),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "no_replay_bad_strategy_issues"
    )
    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["issue_count"] == 1
    assert check["details"]["issues"][0]["metric"] == "replay_idle_rate_agent_0"


def test_build_long_run_check_can_require_replay_combat_map_coverage():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(replay_maps=("classic",)),
        min_maps=2,
        min_replay_combat_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "replay_combat_map_coverage"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["combat_maps"] == ["classic"]
    assert check["details"]["min_replay_combat_maps"] == 2


def test_build_long_run_check_only_counts_required_replay_combat_maps():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(replay_maps=("debug_a", "debug_b")),
        min_maps=2,
        required_maps=("classic", "flat"),
        min_replay_combat_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "replay_combat_map_coverage"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["eligible_combat_maps"] == []
    assert check["details"]["ignored_combat_maps"] == ["debug_a", "debug_b"]


def test_build_long_run_check_treats_candidate_draw_rate_as_bad_strategy():
    strategy_report = _long_run_strategy_report()
    strategy_report["issue_count"] = 1
    strategy_report["issues"] = [
        {
            "scope": "candidate:candidate",
            "metric": "mean_draw_rate",
            "value": 1.0,
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert "no_candidate_bad_strategy_issues" in failed


def test_build_long_run_check_ignores_other_candidate_strategy_issues():
    strategy_report = _long_run_strategy_report()
    strategy_report["issue_count"] = 1
    strategy_report["issues"] = [
        {
            "scope": "candidate:older",
            "metric": "mean_no_damage_rate",
            "value": 1.0,
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "no_candidate_bad_strategy_issues"
    )
    assert result["passed"] is True
    assert check["details"]["issue_count"] == 0


def test_build_long_run_check_treats_candidate_idle_as_bad_strategy():
    strategy_report = _long_run_strategy_report()
    strategy_report["issue_count"] = 1
    strategy_report["issues"] = [
        {
            "scope": "candidate:candidate:rank_suite:classic/idle",
            "metric": "idle_rate_agent_0",
            "value": 0.9,
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert "no_candidate_bad_strategy_issues" in failed


def test_build_long_run_check_treats_candidate_historical_status_as_bad_strategy():
    strategy_report = _long_run_strategy_report()
    strategy_report["issue_count"] = 1
    strategy_report["issues"] = [
        {
            "scope": "candidate:candidate:checkpoint_opponent_pool",
            "metric": "checkpoint_historical_opponent_samples",
            "value": 0,
            "threshold": 1,
        }
    ]

    result = build_long_run_check(
        _long_run_promotion_audit(),
        strategy_report,
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )

    failed = {check["id"] for check in result["checks"] if not check["passed"]}
    assert result["passed"] is False
    assert "no_candidate_bad_strategy_issues" in failed


def test_build_long_run_check_can_require_head_to_head_standings():
    promotion = _long_run_promotion_audit()
    promotion["rank"]["head_to_head"] = {
        "skipped": "requires_at_least_two_checkpoints",
        "checkpoint_count": 1,
    }

    optional = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
    )
    required = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
        require_head_to_head=True,
    )

    optional_check = next(
        check
        for check in optional["checks"]
        if check["id"] == "head_to_head_candidate_not_worse"
    )
    required_check = next(
        check
        for check in required["checks"]
        if check["id"] == "head_to_head_candidate_not_worse"
    )

    assert optional["passed"] is True
    assert optional_check["required"] is False
    assert optional_check["passed"] is False
    assert required["passed"] is False
    assert required_check["required"] is True
    assert required_check["passed"] is False
    assert required_check["details"]["reason"] == "requires_at_least_two_checkpoints"


def test_run_long_run_check_can_save_result(tmp_path, capsys):
    promotion_path = tmp_path / "promotion.json"
    strategy_path = tmp_path / "strategy.json"
    index_path = tmp_path / "index.json"
    promotion_path.write_text(json.dumps(_long_run_promotion_audit()) + "\n")
    strategy_path.write_text(json.dumps(_long_run_strategy_report()) + "\n")
    index = _long_run_artifact_index()
    index["artifacts"].extend(
        [
            {
                "artifact_type": "promotion_audit",
                "path": str(promotion_path),
                "relative_path": promotion_path.name,
                "summary": {},
            },
            {
                "artifact_type": "strategy_report",
                "path": str(strategy_path),
                "relative_path": strategy_path.name,
                "summary": {},
            },
        ]
    )
    index_path.write_text(json.dumps(index) + "\n")
    output_dir = tmp_path / "outputs"

    run_long_run_check(
        str(promotion_path),
        str(strategy_path),
        str(index_path),
        min_maps=2,
        required_maps=("classic", "flat"),
        min_eval_episodes=1,
        min_map_score=0.0,
        require_replay_analysis=True,
        output_dir=str(output_dir),
        output_label="long-run-check",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_long-run-check.json")
    saved = json.loads(saved_path.read_text())
    assert "Saved long-run check to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "long_run_check",
        "schema_version": 1,
    }
    assert saved["passed"] is True
    assert saved["inputs"] == {
        "promotion_audit": str(promotion_path),
        "strategy_report": str(strategy_path),
        "artifact_index": str(index_path),
    }
    assert any(
        check["id"] == "artifact_index_contains_input_artifacts"
        and check["passed"]
        for check in saved["checks"]
    )


def test_run_long_run_check_saves_missing_input_failure(tmp_path, capsys):
    output_dir = tmp_path / "outputs"
    promotion_path = tmp_path / "missing-promotion.json"
    strategy_path = tmp_path / "missing-strategy.json"
    index_path = tmp_path / "missing-index.json"

    try:
        run_long_run_check(
            str(promotion_path),
            str(strategy_path),
            str(index_path),
            min_maps=2,
            required_maps=("classic", "flat"),
            min_eval_episodes=20,
            min_map_score=0.0,
            require_replay_analysis=True,
            output_dir=str(output_dir),
            output_label="missing-long-run-check",
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected missing long-run inputs to exit non-zero")

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_missing-long-run-check.json")
    saved = json.loads(saved_path.read_text())
    assert "Traceback" not in stdout
    assert "Saved long-run check to" in stdout
    assert saved["passed"] is False
    assert saved["check_config"]["required_maps"] == ["classic", "flat"]
    assert saved["inputs"] == {
        "promotion_audit": str(promotion_path),
        "strategy_report": str(strategy_path),
        "artifact_index": str(index_path),
    }
    [check] = saved["checks"]
    assert check["id"] == "input_artifacts_loadable"
    assert check["required"] is True
    assert check["passed"] is False
    assert {error["name"] for error in check["details"]["errors"]} == {
        "promotion_audit",
        "strategy_report",
        "artifact_index",
    }
    assert {error["error_type"] for error in check["details"]["errors"]} == {
        "FileNotFoundError"
    }


def test_artifact_index_contains_path_matches_absolute_and_relative_entries(tmp_path):
    artifact_path = tmp_path / "promotion.json"
    artifact_path.write_text(json.dumps(_promotion_audit_summary()) + "\n")
    index = build_artifact_index(tmp_path)

    assert artifact_index_contains_path(index, artifact_path)
    assert artifact_index_contains_path(index, str(artifact_path))


def test_build_long_run_check_fails_when_index_omits_input_artifacts(tmp_path):
    promotion_path = tmp_path / "promotion.json"
    strategy_path = tmp_path / "strategy.json"
    promotion_path.write_text(json.dumps(_long_run_promotion_audit()) + "\n")
    strategy_path.write_text(json.dumps(_long_run_strategy_report()) + "\n")

    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        promotion_audit_path=str(promotion_path),
        strategy_report_path=str(strategy_path),
        min_maps=2,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "artifact_index_contains_input_artifacts"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["missing_inputs"] == {
        "promotion_audit": str(promotion_path),
        "strategy_report": str(strategy_path),
    }


def test_rank_evaluation_episode_counts_include_baseline_and_head_to_head():
    counts = rank_evaluation_episode_counts(_long_run_promotion_audit()["rank"])

    assert counts["baseline_episodes"] == 40
    assert counts["candidate_baseline_episodes"] == 40
    assert counts["head_to_head_episodes"] == 4
    assert counts["total_episodes"] == 44
    assert counts["configured_baseline_episodes"] == 40
    assert counts["configured_head_to_head_episodes"] == 4
    assert counts["configured_total_episodes"] == 44
    assert counts["baseline_matchups_counted"] == 4
    assert counts["candidate_baseline_matchups_counted"] == 4
    assert counts["head_to_head_sides_counted"] == 4
    assert counts["head_to_head_map_episodes"] == {"classic": 2, "flat": 2}


def test_rank_evaluation_episode_counts_uses_nested_matchups_over_config():
    rank = _long_run_promotion_audit()["rank"]
    rank["rank_config"]["episodes_per_matchup"] = 999
    rank["head_to_head"]["overview"]["total_episodes"] = 999

    counts = rank_evaluation_episode_counts(rank)

    assert counts["baseline_episodes"] == 40
    assert counts["candidate_baseline_episodes"] == 40
    assert counts["head_to_head_episodes"] == 4
    assert counts["total_episodes"] == 44
    assert counts["configured_baseline_episodes"] == 3996
    assert counts["configured_head_to_head_episodes"] == 999
    assert counts["configured_total_episodes"] == 4995
    assert counts["head_to_head_map_episodes"] == {"classic": 2, "flat": 2}


def test_build_long_run_check_can_require_minimum_head_to_head_episodes():
    passing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
        require_head_to_head=True,
        min_head_to_head_episodes=4,
    )
    failing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_replay_analysis=True,
        require_head_to_head=True,
        min_head_to_head_episodes=5,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "minimum_head_to_head_episodes"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "minimum_head_to_head_episodes"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert passing_check["details"]["head_to_head_episodes"] == 4
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["head_to_head_episodes"] == 4
    assert failing_check["details"]["min_head_to_head_episodes"] == 5


def test_build_long_run_check_can_require_head_to_head_map_episodes():
    passing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("classic", "flat"),
        require_replay_analysis=True,
        require_head_to_head=True,
        min_head_to_head_map_episodes=2,
    )
    failing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("classic", "flat"),
        require_replay_analysis=True,
        require_head_to_head=True,
        min_head_to_head_map_episodes=3,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "head_to_head_min_map_episodes"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "head_to_head_min_map_episodes"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert passing_check["details"]["head_to_head_map_episodes"] == {
        "classic": 2,
        "flat": 2,
    }
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["low_head_to_head_maps"] == [
        {"map_name": "classic", "episode_count": 2},
        {"map_name": "flat", "episode_count": 2},
    ]


def test_build_long_run_check_rejects_config_only_episode_counts():
    promotion = _long_run_promotion_audit()
    promotion["rank"]["rank_config"]["episodes_per_matchup"] = 999
    promotion["rank"]["head_to_head"]["overview"]["total_episodes"] = 999
    promotion["rank"].pop("suites")
    promotion["rank"]["head_to_head"].pop("matchups")

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_eval_episodes=1,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "minimum_rank_eval_episodes"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["candidate_baseline_episodes"] == 0
    assert check["details"]["total_episodes"] == 0
    assert check["details"]["configured_total_episodes"] == 4995


def test_candidate_per_map_scores_average_matchup_scores():
    candidate = _long_run_promotion_audit()["candidate"]
    candidate["matchup_scores"].append(
        {"map_name": "flat", "score": 0.25, "episodes": 5}
    )

    assert candidate_per_map_scores(candidate) == [
        {
            "map_name": "classic",
            "mean_score": 0.5,
            "matchup_count": 1,
            "episode_count": 20,
        },
        {
            "map_name": "flat",
            "mean_score": 0.375,
            "matchup_count": 2,
            "episode_count": 25,
        },
    ]


def test_missing_required_maps_preserves_requested_order():
    assert missing_required_maps(
        ["classic", "tower"],
        ("flat", "classic", "split"),
    ) == ["flat", "split"]


def test_build_long_run_check_can_require_minimum_eval_episodes():
    passing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_eval_episodes=40,
        require_replay_analysis=True,
    )
    failing = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_eval_episodes=41,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "minimum_rank_eval_episodes"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "minimum_rank_eval_episodes"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["candidate_baseline_episodes"] == 40
    assert failing_check["details"]["total_episodes"] == 44
    assert failing_check["details"]["min_eval_episodes"] == 41


def test_build_long_run_check_requires_candidate_eval_episodes_not_rank_total():
    promotion = _long_run_promotion_audit()
    promotion["rank"]["suites"].append(
        {
            "label": "older",
            "checkpoint": "checkpoints/older.zip",
            "suite": {
                "matchups": {
                    "classic": {
                        "idle": {"episodes": 100},
                    }
                }
            },
        }
    )

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_eval_episodes=41,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "minimum_rank_eval_episodes"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["baseline_episodes"] == 140
    assert check["details"]["candidate_baseline_episodes"] == 40


def test_build_long_run_check_can_require_minimum_per_map_score():
    promotion = _long_run_promotion_audit()
    promotion["candidate"]["matchup_scores"] = [
        {"map_name": "classic", "score": 0.5, "episodes": 20},
        {"map_name": "flat", "score": -0.1, "episodes": 20},
    ]

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_map_score=0.0,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "candidate_min_map_score"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["low_score_maps"] == [
        {
            "map_name": "flat",
            "mean_score": -0.1,
            "matchup_count": 1,
            "episode_count": 20,
        }
    ]


def test_build_long_run_check_can_require_candidate_checkpoint(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_checkpoint=True,
        require_replay_analysis=True,
    )

    promotion["candidate"]["checkpoint"] = str(tmp_path / "missing.zip")
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_checkpoint=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_checkpoint_exists"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_checkpoint_exists"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert failing["passed"] is False
    assert failing_check["passed"] is False


def test_build_long_run_check_can_require_candidate_metadata(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    write_checkpoint_metadata(tmp_path / "candidate", Config(), num_timesteps=100)
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )

    missing_metadata_checkpoint = tmp_path / "missing_metadata.zip"
    missing_metadata_checkpoint.touch()
    promotion["candidate"]["checkpoint"] = str(missing_metadata_checkpoint)
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_checkpoint_metadata_exists"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_checkpoint_metadata_exists"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert "num_timesteps" in passing_check["details"]["metadata_keys"]
    assert failing["passed"] is False
    assert failing_check["passed"] is False


def test_build_long_run_check_can_require_candidate_integrity(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.write_bytes(b"candidate-v1")
    write_checkpoint_metadata(tmp_path / "candidate", Config(), num_timesteps=100)
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        require_candidate_integrity=True,
        require_replay_analysis=True,
    )
    checkpoint.write_bytes(b"candidate-v2")
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        require_candidate_integrity=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_checkpoint_integrity"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_checkpoint_integrity"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["reason"] == "sha256_mismatch"


def test_build_long_run_check_can_require_historical_opponent_samples(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    write_checkpoint_metadata(
        tmp_path / "candidate",
        Config(),
        num_timesteps=100,
        opponent_pool_stats={
            "size": 3,
            "latest_samples": 8,
            "historical_samples": 2,
            "historical_sample_rate": 0.2,
        },
    )
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_opponent_historical_samples=1,
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_opponent_historical_samples=3,
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_historical_opponent_samples"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_historical_opponent_samples"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert passing_check["details"]["historical_samples"] == 2
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["min_opponent_historical_samples"] == 3


def test_build_long_run_check_rejects_invalid_historical_opponent_samples(tmp_path):
    for index, historical_samples in enumerate((True, -1)):
        promotion = _long_run_promotion_audit()
        checkpoint = tmp_path / f"candidate-{index}.zip"
        checkpoint.touch()
        write_checkpoint_metadata(
            tmp_path / f"candidate-{index}",
            Config(),
            num_timesteps=100,
            opponent_pool_stats={
                "size": 3,
                "latest_samples": 8,
                "historical_samples": historical_samples,
            },
        )
        promotion["candidate"]["checkpoint"] = str(checkpoint)

        result = build_long_run_check(
            promotion,
            _long_run_strategy_report(),
            _long_run_artifact_index(),
            min_maps=2,
            min_opponent_historical_samples=1,
            require_candidate_metadata=True,
            require_replay_analysis=True,
        )

        check = next(
            check
            for check in result["checks"]
            if check["id"] == "candidate_historical_opponent_samples"
        )

        assert result["passed"] is False
        assert check["passed"] is False
        assert check["details"]["historical_samples"] is None


def test_build_long_run_check_validates_candidate_metadata_required_maps(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    cfg = replace(
        Config(),
        arena=replace(
            Config().arena,
            randomize_maps=True,
            map_choices=("flat", "classic"),
        ),
    )
    write_checkpoint_metadata(tmp_path / "candidate", cfg, num_timesteps=100)
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("flat", "classic"),
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("flat", "tower"),
        require_candidate_metadata=True,
        require_replay_analysis=True,
    )

    passing_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_metadata_required_maps"
    )
    failing_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_metadata_required_maps"
    )

    assert passing["passed"] is True
    assert passing_check["passed"] is True
    assert failing["passed"] is False
    assert failing_check["passed"] is False
    assert failing_check["details"]["missing_maps"] == ["tower"]


def test_checkpoint_metadata_maps_prefers_curriculum_stage_coverage():
    metadata = {
        "map_name": "classic",
        "randomize_maps": True,
        "map_choices": ["classic", "flat", "split", "tower"],
        "curriculum": {
            "active_map_pool": ["flat"],
            "stage": {"map_choices": ["flat"]},
        },
    }

    assert checkpoint_metadata_maps(metadata) == ["flat"]


def test_build_long_run_check_validates_candidate_curriculum_metadata(tmp_path):
    promotion = _long_run_promotion_audit()
    checkpoint = tmp_path / "candidate.zip"
    checkpoint.touch()
    cfg = replace(
        Config(),
        training=replace(Config().training, curriculum_name="map_progression"),
    )
    write_checkpoint_metadata(tmp_path / "candidate", cfg, num_timesteps=3_000_000)
    promotion["candidate"]["checkpoint"] = str(checkpoint)

    passing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        required_curriculum_stage="full_map_pool",
        required_reward_preset="anti_stall",
        require_replay_analysis=True,
    )
    failing = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        require_candidate_metadata=True,
        required_curriculum_stage="full_map_pool",
        required_reward_preset="default",
        require_replay_analysis=True,
    )

    stage_check = next(
        check
        for check in passing["checks"]
        if check["id"] == "candidate_metadata_curriculum_stage"
    )
    reward_check = next(
        check
        for check in failing["checks"]
        if check["id"] == "candidate_metadata_reward_preset"
    )

    assert passing["passed"] is True
    assert stage_check["details"]["actual_curriculum_stage"] == "full_map_pool"
    assert failing["passed"] is False
    assert reward_check["passed"] is False
    assert reward_check["details"]["actual_reward_preset"] == "anti_stall"


def test_build_long_run_check_can_require_minimum_per_map_episodes():
    promotion = _long_run_promotion_audit()
    promotion["candidate"]["matchup_scores"] = [
        {"map_name": "classic", "score": 0.5, "episodes": 20},
        {"map_name": "flat", "score": 0.5, "episodes": 3},
    ]

    result = build_long_run_check(
        promotion,
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        min_map_episodes=10,
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "candidate_min_map_episodes"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["low_episode_maps"] == [
        {
            "map_name": "flat",
            "mean_score": 0.5,
            "matchup_count": 1,
            "episode_count": 3,
        }
    ]


def test_build_long_run_check_can_require_specific_maps():
    result = build_long_run_check(
        _long_run_promotion_audit(),
        _long_run_strategy_report(),
        _long_run_artifact_index(),
        min_maps=2,
        required_maps=("classic", "flat", "tower"),
        require_replay_analysis=True,
    )

    check = next(
        check
        for check in result["checks"]
        if check["id"] == "candidate_required_maps"
    )

    assert result["passed"] is False
    assert check["passed"] is False
    assert check["details"]["missing_maps"] == ["tower"]


def test_build_long_run_manifest_emits_non_executing_command_bundle():
    manifest = build_long_run_manifest(
        run_id="arena-test",
        timesteps=1234,
        suite_opponents="idle,scripted",
        suite_maps="flat,tower",
        rounds=3,
        replay_samples_per_bucket=2,
        replay_save_interval=5,
        opponent_pool_seed=123,
        rank_min_score=0.2,
        rank_min_win_rate=0.1,
        rank_max_draw_rate=0.8,
        rank_max_no_damage_rate=0.6,
        rank_max_low_engagement_rate=0.4,
        strategy_max_draw_rate=0.85,
        strategy_max_no_damage_rate=0.65,
        strategy_max_low_engagement_rate=0.45,
        strategy_max_idle_rate=0.7,
        strategy_max_dominant_action_rate=0.9,
        min_maps=2,
    )

    command_ids = [command["id"] for command in manifest["commands"]]

    assert manifest["artifact"] == {
        "artifact_type": "long_run_manifest",
        "schema_version": 1,
    }
    assert manifest["guardrails"] == {
        "executes_training": False,
        "deletes_artifacts": False,
        "contains_expensive_training_command": True,
    }
    assert "python scripts/self_play_sampling_smoke.py" in manifest[
        "preflight_shell_script"
    ]
    assert "python scripts/train_eval_smoke.py" in manifest["preflight_shell_script"]
    assert manifest["preflight_shell_script"].index(
        "python scripts/self_play_sampling_smoke.py"
    ) < manifest["preflight_shell_script"].index("python scripts/train_eval_smoke.py")
    assert "--opponent-pool-seed 123" in manifest["preflight_shell_script"]
    assert "--pool-seed 123" in manifest["preflight_shell_script"]
    assert "--map-pool flat,tower" in manifest["preflight_shell_script"]
    assert "self-play-sampling-summary.json" in manifest["preflight_shell_script"]
    assert "self-play-sampling-preflight.exitcode" in manifest[
        "preflight_shell_script"
    ]
    assert "python scripts/train.py --mode train" not in manifest[
        "preflight_shell_script"
    ]
    assert "TRAIN_EXIT=$?" not in manifest["preflight_shell_script"]
    assert "promotion_audit" not in manifest["preflight_shell_script"]
    assert 'exit "$PREFLIGHT_EXIT"' in manifest["preflight_shell_script"]
    assert command_ids == [
        "create_run_dirs",
        "archive_launcher",
        "self_play_sampling_smoke_preflight",
        "train_eval_smoke_preflight",
        "train",
        "checkpoint_trust_manifest",
        "promotion_audit",
        "resolve_promotion_audit",
        "audit_summary",
        "sample_replay_analysis",
        "strategy_report",
        "artifact_index",
        "resolve_validation_artifacts",
        "long_run_check",
        "long_run_status",
        "league_health",
        "final_artifact_index",
        "exit_with_long_run_check_status",
    ]
    assert "--timesteps 1234" in manifest["shell_script"]
    assert "EVAL_ROOT=evals" in manifest["shell_script"]
    assert 'cp "$0" "$EVAL_DIR/long-run-launcher.sh"' in manifest["shell_script"]
    assert "python scripts/train_eval_smoke.py" in manifest["shell_script"]
    assert 'PREFLIGHT_DIR=evals/arena-test-preflight-smoke' in manifest["shell_script"]
    assert '--output-dir "$PREFLIGHT_DIR"' in manifest["shell_script"]
    assert "--timesteps 128" in manifest["shell_script"]
    assert "--rounds 1" in manifest["shell_script"]
    assert "PREFLIGHT_EXIT=$?" in manifest["shell_script"]
    assert "preflight.exitcode" in manifest["shell_script"]
    assert "preflight.out" in manifest["shell_script"]
    assert "SELF_PLAY_SAMPLING_PREFLIGHT_EXIT=$?" in manifest["shell_script"]
    assert "self-play-sampling-preflight.out" in manifest["shell_script"]
    assert "self-play-sampling-preflight.exitcode" in manifest["shell_script"]
    assert 'if [ "$PREFLIGHT_EXIT" -ne 0 ]; then' in manifest["shell_script"]
    assert "--eval-label preflight-artifact-index" in manifest["shell_script"]
    assert "preflight-artifact-index.out" in manifest["shell_script"]
    assert "--replay-save-interval 5" in manifest["shell_script"]
    assert "--opponent-pool-seed 123" in manifest["shell_script"]
    assert "TRAIN_EXIT=$?" in manifest["shell_script"]
    assert "train.exitcode" in manifest["shell_script"]
    assert "train.out" in manifest["shell_script"]
    assert 'if [ "$TRAIN_EXIT" -ne 0 ]; then' in manifest["shell_script"]
    assert "python scripts/train.py --mode checkpoint_trust_manifest" in manifest[
        "shell_script"
    ]
    assert 'TRUSTED_CHECKPOINT_MANIFEST="$EVAL_DIR/checkpoint-trust-manifest.json"' in (
        manifest["shell_script"]
    )
    assert "--trusted-checkpoint-manifest \"$TRUSTED_CHECKPOINT_MANIFEST\"" in (
        manifest["shell_script"]
    )
    assert "--suite-maps flat,tower" in manifest["shell_script"]
    assert "--rank-min-score 0.2" in manifest["shell_script"]
    assert "--rank-min-win-rate 0.1" in manifest["shell_script"]
    assert "--rank-max-draw-rate 0.8" in manifest["shell_script"]
    assert "--rank-max-no-damage-rate 0.6" in manifest["shell_script"]
    assert "--rank-max-low-engagement-rate 0.4" in manifest["shell_script"]
    assert "PROMOTION_AUDIT_EXIT=$?" in manifest["shell_script"]
    assert "promotion-audit.exitcode" in manifest["shell_script"]
    assert "promotion-audit.out" in manifest["shell_script"]
    assert "MISSING_promotion.json" in manifest["shell_script"]
    assert 'if [ -f "$PROMOTION_AUDIT" ]; then' in manifest["shell_script"]
    assert "audit-summary.out" in manifest["shell_script"]
    assert "--strategy-max-draw-rate 0.85" in manifest["shell_script"]
    assert "--strategy-max-no-damage-rate 0.65" in manifest["shell_script"]
    assert "--strategy-max-low-engagement-rate 0.45" in manifest["shell_script"]
    assert "--strategy-max-idle-rate 0.7" in manifest["shell_script"]
    assert "--strategy-max-dominant-action-rate 0.9" in manifest["shell_script"]
    assert "--strategy-max-weaknesses 10" in manifest["shell_script"]
    assert "--long-run-required-maps flat,tower" in manifest["shell_script"]
    assert "--long-run-min-eval-episodes 12" in manifest["shell_script"]
    assert "--long-run-min-map-episodes 6" in manifest["shell_script"]
    assert "--long-run-min-map-score 0.0" in manifest["shell_script"]
    assert "--long-run-min-replay-combat-maps 2" in manifest["shell_script"]
    assert "--long-run-min-opponent-historical-samples" not in manifest["shell_script"]
    assert "--long-run-min-head-to-head-episodes" not in manifest["shell_script"]
    assert "--long-run-min-head-to-head-map-episodes" not in manifest["shell_script"]
    assert "--long-run-require-candidate-checkpoint" in manifest["shell_script"]
    assert "--long-run-require-candidate-metadata" in manifest["shell_script"]
    assert "--long-run-require-candidate-integrity" in manifest["shell_script"]
    assert "--long-run-require-head-to-head" not in manifest["shell_script"]
    assert "--long-run-required-curriculum-stage full_map_pool" in manifest["shell_script"]
    assert "--long-run-required-reward-preset anti_stall" in manifest["shell_script"]
    assert "LONG_RUN_CHECK_EXIT=$?" in manifest["shell_script"]
    assert "long-run-check.exitcode" in manifest["shell_script"]
    assert "long-run-check.out" in manifest["shell_script"]
    assert "python scripts/train.py --mode long_run_status" in manifest["shell_script"]
    assert '--artifact-dir "$EVAL_ROOT"' in manifest["shell_script"]
    assert "--eval-label long-run-status" in manifest["shell_script"]
    assert "long-run-status.out" in manifest["shell_script"]
    assert "python scripts/train.py --mode league_health" in manifest["shell_script"]
    assert "--eval-label league-health" in manifest["shell_script"]
    assert "league-health.out" in manifest["shell_script"]
    assert "--eval-label final-artifact-index" in manifest["shell_script"]
    assert "final-artifact-index.out" in manifest["shell_script"]
    assert 'exit "$LONG_RUN_CHECK_EXIT"' in manifest["shell_script"]
    assert manifest["manifest_config"]["required_maps"] == ["flat", "tower"]
    assert manifest["manifest_config"]["min_eval_episodes"] == 12
    assert manifest["manifest_config"]["min_map_episodes"] == 6
    assert manifest["manifest_config"]["min_map_score"] == 0.0
    assert manifest["manifest_config"]["min_replay_combat_maps"] == 2
    assert manifest["manifest_config"]["min_opponent_historical_samples"] == 0
    assert manifest["manifest_config"]["min_head_to_head_episodes"] == 0
    assert manifest["manifest_config"]["min_head_to_head_map_episodes"] is None
    assert manifest["manifest_config"]["require_candidate_checkpoint"] is True
    assert manifest["manifest_config"]["require_candidate_metadata"] is True
    assert manifest["manifest_config"]["require_candidate_integrity"] is True
    assert manifest["manifest_config"]["require_head_to_head"] is False
    assert manifest["manifest_config"]["required_curriculum_stage"] == "full_map_pool"
    assert manifest["manifest_config"]["required_reward_preset"] == "anti_stall"
    assert manifest["manifest_config"]["preflight_dir"] == (
        "evals/arena-test-preflight-smoke"
    )
    assert not manifest["manifest_config"]["preflight_dir"].startswith(
        f"{manifest['manifest_config']['eval_dir']}/"
    )
    assert manifest["manifest_config"]["preflight_timesteps"] == 128
    assert manifest["manifest_config"]["preflight_rounds"] == 1
    assert manifest["manifest_config"]["self_play_sampling_preflight_min_maps"] == 2
    assert manifest["manifest_config"]["replay_save_interval"] == 5
    assert manifest["manifest_config"]["replay_save_interval_source"] == "user"
    assert manifest["manifest_config"]["opponent_pool_seed"] == 123
    assert manifest["manifest_config"]["rank_gate"] == {
        "min_score": 0.2,
        "min_win_rate": 0.1,
        "max_draw_rate": 0.8,
        "max_no_damage_rate": 0.6,
        "max_low_engagement_rate": 0.4,
    }
    assert manifest["manifest_config"]["strategy_report"] == {
        "max_draw_rate": 0.85,
        "max_no_damage_rate": 0.65,
        "max_low_engagement_rate": 0.45,
        "max_idle_rate": 0.7,
        "max_dominant_action_rate": 0.9,
        "max_weaknesses": 10,
    }
    source_control = manifest["manifest_config"]["source_control"]
    assert source_control["vcs"] == "git"
    assert isinstance(source_control["available"], bool)
    if source_control["available"]:
        assert "commit" in source_control
        assert "dirty" in source_control
        assert "status_short_count" in source_control
    assert "--long-run-require-replay-analysis" in manifest["shell_script"]
    assert 'RUN_ID=arena-test' in manifest["shell_script"]


def test_build_long_run_manifest_rejects_shell_injection_values():
    cases = [
        (
            {"required_curriculum_stage": "full_map_pool; echo injected >&2"},
            "Unknown curriculum stage",
        ),
        (
            {"required_reward_preset": "anti_stall; echo injected >&2"},
            "Unknown reward preset",
        ),
        (
            {"suite_maps": "classic; echo injected >&2"},
            "Unknown map names",
        ),
        (
            {"suite_opponents": "idle; echo injected >&2"},
            "Unknown opponent names",
        ),
        (
            {"opponent_pool_seed": -1},
            "opponent_pool_seed must be non-negative",
        ),
    ]

    for kwargs, expected_message in cases:
        try:
            build_long_run_manifest(run_id="arena-test", timesteps=1234, **kwargs)
        except ValueError as exc:
            assert expected_message in str(exc)
        else:
            raise AssertionError(f"Expected {kwargs} to be rejected")


def test_build_long_run_manifest_rejects_unsafe_run_ids():
    for run_id in ("", "../escape", "arena/test", "arena test", "arena;echo injected"):
        try:
            build_long_run_manifest(run_id=run_id, timesteps=1234)
        except ValueError as exc:
            assert "run_id must start" in str(exc)
        else:
            raise AssertionError(f"Expected run_id={run_id!r} to be rejected")


def test_build_long_run_manifest_indexes_early_failure_artifacts():
    manifest = build_long_run_manifest(run_id="arena-test", timesteps=1234)
    script = manifest["shell_script"]

    assert (
        'printf "%s\\n" "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT" > '
        '"$EVAL_DIR/self-play-sampling-preflight.exitcode"'
    ) in script
    assert 'printf "%s\\n" "$PREFLIGHT_EXIT" > "$EVAL_DIR/preflight.exitcode"' in script
    assert 'printf "%s\\n" "$TRAIN_EXIT" > "$EVAL_DIR/train.exitcode"' in script
    assert 'if [ "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT" -ne 0 ]; then' in script
    assert 'if [ "$PREFLIGHT_EXIT" -ne 0 ]; then' in script
    assert 'if [ "$TRAIN_EXIT" -ne 0 ]; then' in script
    assert '--artifact-dir "$PREFLIGHT_DIR"' in script
    assert "--eval-label preflight-artifact-index" in script
    assert script.count("--eval-label final-artifact-index") == 4
    assert 'exit "$SELF_PLAY_SAMPLING_PREFLIGHT_EXIT"' in script
    assert 'exit "$PREFLIGHT_EXIT"' in script
    assert 'exit "$TRAIN_EXIT"' in script


def test_build_long_run_manifest_uses_missing_artifact_placeholders():
    manifest = build_long_run_manifest(run_id="arena-test", timesteps=1234)
    script = manifest["shell_script"]

    assert 'ls -1t "$EVAL_DIR"/*_promotion.json 2>/dev/null' in script
    assert 'PROMOTION_AUDIT="$EVAL_DIR/MISSING_promotion.json"' in script
    assert 'if [ -f "$PROMOTION_AUDIT" ]; then' in script
    assert 'ls -1t "$EVAL_DIR"/*_strategy-report.json 2>/dev/null' in script
    assert 'STRATEGY_REPORT="$EVAL_DIR/MISSING_strategy-report.json"' in script
    assert 'ls -1t "$EVAL_DIR"/*_artifact-index.json 2>/dev/null' in script
    assert 'ARTIFACT_INDEX="$EVAL_DIR/MISSING_artifact-index.json"' in script


def test_build_long_run_manifest_redirects_command_logs():
    manifest = build_long_run_manifest(run_id="arena-test", timesteps=1234)
    script = manifest["shell_script"]

    for log_name in (
        "self-play-sampling-preflight.out",
        "preflight.out",
        "train.out",
        "checkpoint-trust-manifest.out",
        "promotion-audit.out",
        "audit-summary.out",
        "replay-analysis.out",
        "strategy-report.out",
        "artifact-index.out",
        "long-run-check.out",
        "long-run-status.out",
        "league-health.out",
        "final-artifact-index.out",
    ):
        assert f' > "$EVAL_DIR/{log_name}" 2>&1' in script


def test_run_long_run_manifest_saves_json_and_launcher(tmp_path, capsys):
    output_dir = tmp_path / "manifests"

    run_long_run_manifest(
        run_id="arena-test",
        checkpoint_root="ckpts",
        eval_root="evals",
        replay_root="replays",
        timesteps=1234,
        suite_opponents="idle,scripted",
        suite_maps="flat,tower",
        rounds=3,
        replay_samples_per_bucket=1,
        replay_save_interval=5,
        opponent_pool_seed=123,
        rank_min_score=0.2,
        rank_min_win_rate=0.1,
        rank_max_draw_rate=0.8,
        rank_max_no_damage_rate=0.6,
        rank_max_low_engagement_rate=0.4,
        strategy_max_draw_rate=0.85,
        strategy_max_no_damage_rate=0.65,
        strategy_max_low_engagement_rate=0.45,
        strategy_max_idle_rate=0.7,
        strategy_max_dominant_action_rate=0.9,
        require_replay_analysis=True,
        min_maps=2,
        required_maps=("flat", "tower"),
        min_eval_episodes=None,
        min_map_episodes=None,
        min_map_score=0.0,
        min_replay_combat_maps=None,
        require_candidate_checkpoint=True,
        require_candidate_metadata=True,
        required_curriculum_stage="full_map_pool",
        required_reward_preset="anti_stall",
        require_head_to_head=True,
        output_dir=str(output_dir),
        output_label="long-run-plan",
    )

    stdout = capsys.readouterr().out
    [manifest_path] = output_dir.glob("*_long-run-plan.json")
    [script_path] = output_dir.glob("*_long-run-plan.sh")
    [preflight_script_path] = output_dir.glob("*_long-run-plan.preflight.sh")
    saved = json.loads(manifest_path.read_text())
    index = build_artifact_index(output_dir)

    assert "Saved long-run manifest to" in stdout
    assert "Saved long-run launcher to" in stdout
    assert "Saved long-run preflight launcher to" in stdout
    assert saved["artifact"] == {
        "artifact_type": "long_run_manifest",
        "schema_version": 1,
    }
    assert saved["manifest_artifact_path"] == str(manifest_path)
    assert saved["shell_script_path"] == str(script_path)
    assert saved["preflight_shell_script_path"] == str(preflight_script_path)
    assert script_path.read_text().startswith("#!/usr/bin/env bash")
    assert preflight_script_path.read_text().startswith("#!/usr/bin/env bash")
    assert "python scripts/self_play_sampling_smoke.py" in (
        preflight_script_path.read_text()
    )
    assert "python scripts/train_eval_smoke.py" in preflight_script_path.read_text()
    assert preflight_script_path.read_text().index(
        "python scripts/self_play_sampling_smoke.py"
    ) < preflight_script_path.read_text().index("python scripts/train_eval_smoke.py")
    assert "python scripts/train.py --mode train" not in (
        preflight_script_path.read_text()
    )
    assert script_path.stat().st_mode & 0o111
    assert preflight_script_path.stat().st_mode & 0o111
    assert index["artifact_counts"] == {
        "long_run_manifest": 1,
        "shell_script": 2,
    }
    manifest_entry = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "long_run_manifest"
    )
    assert manifest_entry["summary"]["run_id"] == "arena-test"
    assert manifest_entry["summary"]["replay_save_interval"] == 5
    assert manifest_entry["summary"]["replay_save_interval_source"] == "user"
    assert manifest_entry["summary"]["opponent_pool_seed"] == 123
    assert manifest_entry["summary"]["min_eval_episodes"] == 12
    assert manifest_entry["summary"]["min_map_episodes"] == 6
    assert manifest_entry["summary"]["min_replay_combat_maps"] == 2
    assert manifest_entry["summary"]["min_opponent_historical_samples"] == 0
    assert manifest_entry["summary"]["min_head_to_head_episodes"] == 12
    assert manifest_entry["summary"]["min_head_to_head_map_episodes"] == 6
    assert manifest_entry["summary"]["require_candidate_checkpoint"] is True
    assert manifest_entry["summary"]["require_candidate_metadata"] is True
    assert manifest_entry["summary"]["require_candidate_integrity"] is True
    assert manifest_entry["summary"]["require_head_to_head"] is True
    assert manifest_entry["summary"]["required_curriculum_stage"] == "full_map_pool"
    assert manifest_entry["summary"]["required_reward_preset"] == "anti_stall"
    assert "source_dirty" in manifest_entry["summary"]
    assert "source_status_short_count" in manifest_entry["summary"]
    assert manifest_entry["summary"]["has_preflight_shell_script"] is True
    assert manifest_entry["summary"]["preflight_shell_script_path"] == str(
        preflight_script_path
    )
    assert manifest_entry["summary"]["rank_gate"]["max_draw_rate"] == 0.8
    assert manifest_entry["summary"]["strategy_report"]["max_draw_rate"] == 0.85
    assert manifest_entry["summary"]["expensive_command_ids"] == ["train"]
    script_entry = next(
        entry
        for entry in index["artifacts"]
        if entry["artifact_type"] == "shell_script"
    )
    assert script_entry["summary"]["starts_with_shebang"] is True
    assert saved["manifest_config"]["replay_save_interval"] == 5
    assert saved["manifest_config"]["opponent_pool_seed"] == 123
    assert saved["manifest_config"]["min_map_episodes"] == 6
    assert saved["manifest_config"]["min_replay_combat_maps"] == 2
    assert saved["manifest_config"]["min_opponent_historical_samples"] == 0
    assert saved["manifest_config"]["min_head_to_head_episodes"] == 12
    assert saved["manifest_config"]["min_head_to_head_map_episodes"] == 6
    assert saved["manifest_config"]["require_candidate_checkpoint"] is True
    assert saved["manifest_config"]["require_candidate_metadata"] is True
    assert saved["manifest_config"]["require_candidate_integrity"] is True
    assert saved["manifest_config"]["require_head_to_head"] is True
    assert saved["manifest_config"]["required_curriculum_stage"] == "full_map_pool"
    assert saved["manifest_config"]["required_reward_preset"] == "anti_stall"
    assert saved["manifest_config"]["preflight_dir"] == (
        "evals/arena-test-preflight-smoke"
    )
    assert saved["manifest_config"]["preflight_timesteps"] == 128
    assert saved["manifest_config"]["preflight_rounds"] == 1
    assert saved["manifest_config"]["self_play_sampling_preflight_min_maps"] == 2
    assert saved["manifest_config"]["replay_save_interval_source"] == "user"
    assert saved["manifest_config"]["rank_gate"]["max_draw_rate"] == 0.8
    assert saved["manifest_config"]["strategy_report"]["max_draw_rate"] == 0.85


def test_long_run_manifest_cli_honors_required_maps(tmp_path, monkeypatch, capsys):
    output_dir = tmp_path / "manifests"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "long_run_manifest",
            "--run-id",
            "cli-required",
            "--timesteps",
            "1234",
            "--suite-maps",
            "classic,flat,split,tower",
            "--long-run-required-maps",
            "flat,tower",
            "--eval-output-dir",
            str(output_dir),
            "--eval-label",
            "cli-required",
        ],
    )

    main()

    capsys.readouterr()
    [manifest_path] = output_dir.glob("*_cli-required.json")
    saved = json.loads(manifest_path.read_text())
    assert saved["manifest_config"]["suite_maps"] == "classic,flat,split,tower"
    assert saved["manifest_config"]["required_maps"] == ["flat", "tower"]
    assert "--long-run-required-maps flat,tower" in saved["shell_script"]


def test_long_run_manifest_cli_reports_invalid_run_id(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "long_run_manifest",
            "--run-id",
            "../escape",
        ],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected parser error for invalid run ID")

    assert "run_id must start" in capsys.readouterr().err


def test_long_run_check_cli_rejects_negative_historical_opponent_threshold(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--mode",
            "long_run_check",
            "--long-run-min-opponent-historical-samples",
            "-1",
        ],
    )

    try:
        main()
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("Expected parser error for negative historical threshold")

    assert (
        "--long-run-min-opponent-historical-samples must be non-negative"
        in capsys.readouterr().err
    )


def test_build_long_run_status_reports_latest_manifest_execution_state(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    launcher_path = manifest_path.with_suffix(".sh")
    preflight_launcher_path = manifest_path.with_suffix(".preflight.sh")
    manifest["manifest_artifact_path"] = str(manifest_path)
    manifest["shell_script_path"] = str(launcher_path)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    launcher_path.write_text(manifest["shell_script"] + "\n")
    preflight_launcher_path.write_text(manifest["preflight_shell_script"] + "\n")
    launcher_path.chmod(0o755)
    preflight_launcher_path.chmod(0o755)

    status = build_long_run_status(artifact_dir)

    assert status["artifact"] == {
        "artifact_type": "long_run_status",
        "schema_version": 1,
    }
    assert status["manifest_count"] == 1
    assert status["long_run_check_count"] == 0
    assert status["candidate_evidence_ready"] is False
    assert status["blocked_reason"] == "latest_launcher_not_executed"
    assert status["next_command"] == f"bash {launcher_path}"
    assert status["next_preflight_command"] == f"bash {preflight_launcher_path}"
    assert set(status["missing_evidence"]) >= {
        "preflight_exitcode",
        "train_exitcode",
        "promotion_audit_exitcode",
        "long_run_check_exitcode",
        "candidate_checkpoint_files",
        "real_training_replay_files",
        "latest_run_long_run_check",
    }
    latest = status["latest_manifest"]
    assert latest["run_id"] == "status-run"
    assert latest["launcher_exists"] is True
    assert latest["preflight_launcher_exists"] is True
    assert latest["preflight_launcher_path"] == str(preflight_launcher_path)
    assert latest["eval_dir_exists"] is False
    assert latest["checkpoint_file_count"] == 0
    assert latest["replay_file_count"] == 0
    assert latest["long_run_check_count"] == 0
    assert latest["passing_long_run_check_count"] == 0
    assert latest["source_dirty"] in {True, False, None}


def test_build_long_run_status_quotes_copy_paste_launcher_commands(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals with spaces"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status plan.json"
    launcher_path = manifest_path.with_suffix(".sh")
    preflight_launcher_path = manifest_path.with_suffix(".preflight.sh")
    manifest["manifest_artifact_path"] = str(manifest_path)
    manifest["shell_script_path"] = str(launcher_path)
    manifest["preflight_shell_script_path"] = str(preflight_launcher_path)
    manifest_path.write_text(json.dumps(manifest) + "\n")
    launcher_path.write_text(manifest["shell_script"] + "\n")
    preflight_launcher_path.write_text(manifest["preflight_shell_script"] + "\n")

    status = build_long_run_status(artifact_dir)

    assert status["next_command"] == f"bash {shlex.quote(str(launcher_path))}"
    assert status["next_preflight_command"] == (
        f"bash {shlex.quote(str(preflight_launcher_path))}"
    )


def test_build_long_run_status_ignores_unsafe_manifest_launcher_paths(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    outside_launcher = tmp_path / "outside.sh"
    outside_preflight = tmp_path / "outside.preflight.sh"
    outside_launcher.write_text("#!/usr/bin/env bash\n")
    outside_preflight.write_text("#!/usr/bin/env bash\n")
    manifest["shell_script_path"] = str(outside_launcher)
    manifest["preflight_shell_script_path"] = str(outside_preflight)
    manifest_path.write_text(json.dumps(manifest) + "\n")

    status = build_long_run_status(artifact_dir)

    latest = status["latest_manifest"]
    assert latest["launcher_path"] is None
    assert latest["preflight_launcher_path"] is None
    assert latest["launcher_exists"] is False
    assert latest["preflight_launcher_exists"] is False
    assert status["next_command"] is None
    assert status["next_preflight_command"] is None


def test_build_long_run_status_reports_manifest_source_freshness(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="stale-source-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest["manifest_config"]["source_control"] = {
        "vcs": "git",
        "available": True,
        "commit": "old-commit",
        "dirty": True,
        "status_short_count": 1,
    }
    manifest_path = artifact_dir / "stale-source-plan.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    monkeypatch.setattr(
        "scripts.train.source_control_snapshot",
        lambda: {
            "vcs": "git",
            "available": True,
            "commit": "new-commit",
            "dirty": False,
            "status_short_count": 0,
        },
    )

    status = build_long_run_status(artifact_dir)

    latest = status["latest_manifest"]
    assert status["blocked_reason"] == "latest_manifest_source_stale"
    assert status["next_command"] is None
    assert status["next_preflight_command"] is None
    assert latest["source_current_commit"] == "new-commit"
    assert latest["source_current_dirty"] is False
    assert latest["source_commit_matches_current"] is False
    assert latest["source_manifest_clean"] is False
    assert latest["source_current_clean"] is True
    assert latest["source_safe_to_launch"] is False
    assert latest["source_stale_reasons"] == [
        "commit_mismatch",
        "manifest_created_from_dirty_worktree",
    ]
    assert status["status_config"]["source_control"]["commit"] == "new-commit"


def test_build_long_run_status_distinguishes_preflight_only_run(tmp_path, monkeypatch):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    launcher_path = manifest_path.with_suffix(".sh")
    preflight_launcher_path = manifest_path.with_suffix(".preflight.sh")
    manifest_path.write_text(json.dumps(manifest) + "\n")
    launcher_path.write_text(manifest["shell_script"] + "\n")
    preflight_launcher_path.write_text(manifest["preflight_shell_script"] + "\n")
    eval_dir = Path(manifest["manifest_config"]["eval_dir"])
    preflight_dir = Path(manifest["manifest_config"]["preflight_dir"])
    eval_dir.mkdir(parents=True)
    preflight_dir.mkdir(parents=True)
    (eval_dir / "preflight.exitcode").write_text("0\n")

    status = build_long_run_status(artifact_dir)

    assert status["blocked_reason"] == "latest_preflight_only"
    assert status["next_command"] == f"bash {launcher_path}"
    assert status["next_preflight_command"] is None
    assert "preflight_exitcode" not in status["missing_evidence"]
    assert set(status["missing_evidence"]) >= {
        "train_exitcode",
        "promotion_audit_exitcode",
        "long_run_check_exitcode",
        "candidate_checkpoint_files",
        "real_training_replay_files",
        "latest_run_long_run_check",
    }
    latest = status["latest_manifest"]
    assert latest["eval_dir_exists"] is True
    assert latest["preflight_dir_exists"] is True
    assert latest["preflight_exitcode_exists"] is True
    assert latest["train_exitcode_exists"] is False
    assert latest["checkpoint_file_count"] == 0
    assert latest["replay_file_count"] == 0


def test_build_long_run_status_requires_usable_checkpoint_and_replay_files(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    checkpoint_dir = Path(manifest["manifest_config"]["checkpoint_dir"])
    replay_dir = Path(manifest["manifest_config"]["replay_dir"])
    checkpoint_dir.mkdir(parents=True)
    replay_dir.mkdir(parents=True)
    (checkpoint_dir / "ppo_final.meta.json").write_text("{}\n")
    (checkpoint_dir / "notes.txt").write_text("not a checkpoint\n")
    (replay_dir / "notes.txt").write_text("not a replay\n")
    (replay_dir / "episode_0001.json").write_text('{"episode_id": 1}\n')

    status = build_long_run_status(artifact_dir)

    latest = status["latest_manifest"]
    assert latest["checkpoint_file_count"] == 0
    assert latest["checkpoint_total_file_count"] == 2
    assert latest["replay_file_count"] == 0
    assert latest["replay_total_file_count"] == 2
    assert "candidate_checkpoint_files" in status["missing_evidence"]
    assert "real_training_replay_files" in status["missing_evidence"]

    (checkpoint_dir / "ppo_final.zip").write_bytes(b"checkpoint")
    (replay_dir / "episode_0002.json").write_text(
        json.dumps({"episode_id": 2, "frames": []}) + "\n"
    )

    passing_status = build_long_run_status(artifact_dir)

    passing_latest = passing_status["latest_manifest"]
    assert passing_latest["checkpoint_file_count"] == 1
    assert passing_latest["checkpoint_total_file_count"] == 3
    assert passing_latest["replay_file_count"] == 1
    assert passing_latest["replay_total_file_count"] == 3
    assert "candidate_checkpoint_files" not in passing_status["missing_evidence"]
    assert "real_training_replay_files" not in passing_status["missing_evidence"]


def test_build_long_run_status_reports_checkpoint_historical_opponent_samples(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr("scripts.train.source_control_snapshot", _clean_source_snapshot)
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    checkpoint_dir = Path(manifest["manifest_config"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True)
    checkpoint = checkpoint_dir / "ppo_final.zip"
    checkpoint.write_bytes(b"candidate")
    write_checkpoint_metadata(
        checkpoint_dir / "ppo_final",
        Config(),
        num_timesteps=100,
        opponent_pool_stats={
            "size": 2,
            "latest_samples": 4,
            "historical_samples": 0,
        },
    )

    status = build_long_run_status(artifact_dir)

    latest = status["latest_manifest"]
    opponent_pool = latest["checkpoint_opponent_pool"]
    assert latest["min_opponent_historical_samples"] == 1
    assert opponent_pool["metadata_file_count"] == 1
    assert opponent_pool["metadata_with_opponent_pool_count"] == 1
    assert opponent_pool["max_historical_samples"] == 0
    assert opponent_pool["meets_min_opponent_historical_samples"] is False
    assert (
        opponent_pool["best_checkpoint_metadata"]["path"]
        == str(checkpoint_dir / "ppo_final.meta.json")
    )
    assert "checkpoint_historical_opponent_samples" in status["missing_evidence"]

    write_checkpoint_metadata(
        checkpoint_dir / "ppo_final",
        Config(),
        num_timesteps=200,
        opponent_pool_stats={
            "size": 3,
            "latest_samples": 8,
            "historical_samples": 2,
        },
    )

    passing_status = build_long_run_status(artifact_dir)
    passing_pool = passing_status["latest_manifest"]["checkpoint_opponent_pool"]
    assert passing_pool["max_historical_samples"] == 2
    assert passing_pool["metadata_meeting_min_count"] == 1
    assert passing_pool["meets_min_opponent_historical_samples"] is True
    assert (
        "checkpoint_historical_opponent_samples"
        not in passing_status["missing_evidence"]
    )


def test_build_long_run_status_detects_latest_passing_check(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    manifest = build_long_run_manifest(
        run_id="status-run",
        checkpoint_root=str(tmp_path / "checkpoints"),
        eval_root=str(artifact_dir),
        replay_root=str(tmp_path / "replays"),
        timesteps=5_000_000,
    )
    manifest_path = artifact_dir / "status-plan.json"
    manifest_path.write_text(json.dumps(manifest) + "\n")
    eval_dir = Path(manifest["manifest_config"]["eval_dir"])
    eval_dir.mkdir(parents=True)
    check_path = eval_dir / "status-long-run-check.json"
    check_path.write_text(
        json.dumps(
            {
                "artifact": artifact_metadata("long_run_check"),
                "passed": True,
                "candidate": {
                    "label": "ppo_final",
                    "checkpoint": "checkpoints/ppo_final.zip",
                    "score": 0.75,
                },
                "checks": [
                    {
                        "id": "promotion_audit_passed",
                        "required": True,
                        "passed": True,
                    }
                ],
            }
        )
        + "\n"
    )

    status = build_long_run_status(artifact_dir)

    assert status["candidate_evidence_ready"] is True
    assert status["blocked_reason"] is None
    assert status["next_command"] is None
    assert status["passing_long_run_check_count"] == 1
    latest = status["latest_manifest"]
    assert latest["eval_dir_exists"] is True
    assert latest["long_run_check_count"] == 1
    assert latest["passing_long_run_check_count"] == 1
    assert latest["latest_long_run_check"]["path"] == str(check_path)


def test_run_long_run_status_can_save_indexable_artifact(tmp_path, capsys):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    output_dir = tmp_path / "status"

    run_long_run_status(
        str(artifact_dir),
        output_dir=str(output_dir),
        output_label="status-report",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_status-report.json")
    saved = json.loads(saved_path.read_text())
    index = build_artifact_index(output_dir)
    [status_entry] = index["artifacts"]
    assert "Saved long-run status to" in stdout
    assert saved["blocked_reason"] == "no_long_run_manifest_found"
    assert index["artifact_counts"] == {"long_run_status": 1}
    assert status_entry["summary"]["candidate_evidence_ready"] is False
    assert status_entry["summary"]["blocked_reason"] == "no_long_run_manifest_found"
    assert status_entry["summary"]["missing_evidence"] == ["long_run_manifest"]


def test_build_league_health_report_summarizes_latest_league_signals(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 1,
        "issues": [
            {
                "scope": "candidate:candidate:checkpoint_opponent_pool",
                "metric": "checkpoint_historical_opponent_samples",
                "value": 0,
                "threshold": 1,
            }
        ],
        "weakness_count": 1,
        "weaknesses": [
            {
                "scope": "suite:flat/idle",
                "map_name": "flat",
                "opponent": "idle",
                "score": -0.25,
                "episodes": 20,
                "win_rate_agent_0": 0.0,
                "draw_rate": 0.5,
                "no_damage_rate": 0.25,
                "low_engagement_rate": 0.5,
                "avg_length": 50.0,
            }
        ],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": False,
        "blocked_reason": "latest_long_run_check_not_passing",
        "missing_evidence": ["checkpoint_historical_opponent_samples"],
        "latest_manifest": {
            "run_id": "status-run",
            "checkpoint_opponent_pool": {
                "min_opponent_historical_samples": 1,
                "max_historical_samples": 0,
                "meets_min_opponent_historical_samples": False,
            },
        },
    }
    rank = _rank_summary(label="candidate", score=0.5)
    rank["head_to_head"] = {
        "overview": {"total_episodes": 4},
        "standings": [
            {"label": "candidate", "elo": 1012.0, "score": 0.6},
            {"label": "older", "elo": 988.0, "score": 0.4},
        ],
    }
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": False,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [
            {
                "id": "no_candidate_bad_strategy_issues",
                "required": True,
                "passed": False,
            }
        ],
    }
    promotion = _promotion_audit_summary()
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["artifact"] == {
        "artifact_type": "league_health",
        "schema_version": 1,
    }
    assert report["health"] == {
        "ready": False,
        "blockers": [
            "long_run_status_blocked",
            "candidate_strategy_issues",
            "historical_opponent_sampling",
            "long_run_check_failed",
        ],
        "warnings": [],
    }
    assert report["signals"]["candidate"]["label"] == "candidate"
    assert report["signals"]["opponent_pool"] == {
        "historical_sample_ready": False,
        "max_historical_samples": 0,
        "min_historical_samples": 1,
    }
    assert report["signals"]["strategy"]["candidate_issue_count"] == 1
    assert report["signals"]["strategy"]["historical_sampling_issue_count"] == 1
    assert report["signals"]["map_weaknesses"]["maps"] == ["flat"]
    assert report["signals"]["map_weaknesses"]["worst"]["scope"] == "suite:flat/idle"
    assert report["signals"]["head_to_head"]["candidate_elo"] == 1012.0
    assert report["signals"]["head_to_head"]["standing_rank"] == 1
    assert report["signals"]["long_run"]["failed_required_checks"] == [
        "no_candidate_bad_strategy_issues"
    ]


def test_build_league_health_blocks_on_replay_strategy_issues(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 1,
        "issues": [
            {
                "scope": "replay:episode_0001",
                "metric": "replay_dominant_action_rate_agent_0",
                "value": 1.0,
                "threshold": 0.95,
            }
        ],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "latest_manifest": {"run_id": "status-run"},
    }
    rank = _rank_summary(label="candidate", score=0.5)
    promotion = _promotion_audit_summary()
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["health"]["ready"] is False
    assert "replay_strategy_issues" in report["health"]["blockers"]
    assert report["signals"]["strategy"]["replay_issue_count"] == 1


def test_build_league_health_blocks_on_smoke_strategy_issues(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 1,
        "issues": [
            {
                "scope": "smoke:reward_shaping",
                "metric": "reward_smoke_strategy_issue_count",
                "value": 3,
                "threshold": 0,
            }
        ],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "latest_manifest": {"run_id": "status-run"},
    }
    rank = _rank_summary(label="candidate", score=0.5)
    promotion = _promotion_audit_summary()
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["health"]["ready"] is False
    assert "smoke_strategy_issues" in report["health"]["blockers"]
    assert report["signals"]["strategy"]["smoke_issue_count"] == 1


def test_build_league_health_blocks_on_failed_self_play_sampling_smoke(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 0,
        "issues": [],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "latest_manifest": {"run_id": "status-run"},
    }
    sampling_smoke = {
        "artifact": artifact_metadata("self_play_sampling_smoke"),
        "passed": False,
        "historical_samples": 0,
        "unique_maps_seen": 4,
        "checks": [
            {"id": "historical_samples_meet_minimum", "passed": False},
        ],
    }
    rank = _rank_summary(label="candidate", score=0.5)
    promotion = _promotion_audit_summary()
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "sampling.json").write_text(json.dumps(sampling_smoke) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(rank) + "\n")
    (artifact_dir / "promotion.json").write_text(json.dumps(promotion) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")

    report = build_league_health_report(artifact_dir)

    assert report["health"]["ready"] is False
    assert "self_play_sampling_smoke_failed" in report["health"]["blockers"]
    assert report["signals"]["self_play_sampling"] == {
        "available": True,
        "passed": False,
        "historical_samples": 0,
        "historical_sample_rate": None,
        "unique_maps_seen": 4,
        "failed_checks": ["historical_samples_meet_minimum"],
    }
    assert report["source_artifacts"]["self_play_sampling_smoke"] == str(
        artifact_dir / "sampling.json"
    )


def test_build_league_health_blocks_on_latest_long_run_status(tmp_path):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 0,
        "issues": [],
        "weakness_count": 0,
        "weaknesses": [],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": False,
        "blocked_reason": "latest_long_run_check_not_passing",
        "missing_evidence": ["passing_latest_long_run_check"],
        "latest_manifest": {
            "run_id": "latest-run",
            "checkpoint_opponent_pool": {
                "min_opponent_historical_samples": 1,
                "max_historical_samples": 2,
                "meets_min_opponent_historical_samples": True,
            },
        },
    }
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "older", "score": 0.5},
        "checks": [
            {
                "id": "promotion_audit_passed",
                "required": True,
                "passed": True,
            }
        ],
    }
    (artifact_dir / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (artifact_dir / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (artifact_dir / "rank.json").write_text(json.dumps(_rank_summary()) + "\n")
    (artifact_dir / "check.json").write_text(json.dumps(long_run_check) + "\n")
    (artifact_dir / "promotion.json").write_text(
        json.dumps(_promotion_audit_summary()) + "\n"
    )

    report = build_league_health_report(artifact_dir)

    assert report["health"]["ready"] is False
    assert report["health"]["blockers"] == ["long_run_status_blocked"]
    assert report["health"]["warnings"] == []
    assert report["signals"]["long_run"]["status_blocked_reason"] == (
        "latest_long_run_check_not_passing"
    )
    assert report["signals"]["long_run"]["latest_check_passed"] is True


def test_build_league_health_scopes_sources_to_latest_status_run(tmp_path):
    artifact_dir = tmp_path / "evals"
    older_run = artifact_dir / "older-run"
    latest_run = artifact_dir / "latest-run"
    older_run.mkdir(parents=True)
    latest_run.mkdir(parents=True)
    strategy_report = {
        "artifact": artifact_metadata("strategy_report"),
        "issue_count": 0,
        "issues": [],
        "weakness_count": 0,
        "weaknesses": [],
    }
    long_run_status = {
        "artifact": artifact_metadata("long_run_status"),
        "candidate_evidence_ready": True,
        "blocked_reason": None,
        "missing_evidence": [],
        "latest_manifest": {
            "run_id": "latest-run",
            "eval_dir": str(latest_run),
            "checkpoint_opponent_pool": {
                "min_opponent_historical_samples": 1,
                "max_historical_samples": 2,
                "meets_min_opponent_historical_samples": True,
            },
        },
    }
    long_run_check = {
        "artifact": artifact_metadata("long_run_check"),
        "passed": True,
        "candidate": {"label": "candidate", "score": 0.5},
        "checks": [
            {
                "id": "promotion_audit_passed",
                "required": True,
                "passed": True,
            }
        ],
    }
    (latest_run / "strategy.json").write_text(json.dumps(strategy_report) + "\n")
    (latest_run / "status.json").write_text(json.dumps(long_run_status) + "\n")
    (latest_run / "check.json").write_text(json.dumps(long_run_check) + "\n")
    (latest_run / "promotion.json").write_text(
        json.dumps(_promotion_audit_summary()) + "\n"
    )
    (older_run / "newer-but-unrelated-rank.json").write_text(
        json.dumps(_rank_summary(label="unrelated", score=0.99)) + "\n"
    )

    report = build_league_health_report(artifact_dir)

    assert report["health_config"]["artifact_scope_dir"] == str(latest_run)
    assert report["source_artifacts"]["rank"] is None
    assert report["source_artifacts"]["strategy_report"] == str(
        latest_run / "strategy.json"
    )
    assert report["health"]["ready"] is False
    assert report["health"]["blockers"] == []
    assert report["health"]["warnings"] == ["missing_rank"]
    assert report["signals"]["candidate"]["rank_score"] is None
    assert report["signals"]["head_to_head"]["candidate_label"] is None


def test_run_league_health_can_save_indexable_artifact(tmp_path, capsys):
    artifact_dir = tmp_path / "evals"
    artifact_dir.mkdir()
    output_dir = tmp_path / "health"

    run_league_health(
        str(artifact_dir),
        output_dir=str(output_dir),
        output_label="league-health",
    )

    stdout = capsys.readouterr().out
    [saved_path] = output_dir.glob("*_league-health.json")
    saved = json.loads(saved_path.read_text())
    index = build_artifact_index(output_dir)
    [health_entry] = index["artifacts"]
    assert "Saved league health report to" in stdout
    assert saved["health"]["ready"] is False
    assert set(saved["health"]["warnings"]) == {
        "missing_strategy_report",
        "missing_rank",
        "missing_promotion_audit",
        "missing_long_run_status",
        "missing_long_run_check",
    }
    assert index["artifact_counts"] == {"league_health": 1}
    assert health_entry["summary"]["ready"] is False
    assert "missing_rank" in health_entry["summary"]["warnings"]


def test_build_long_run_manifest_auto_pins_replay_interval_for_tiny_runs():
    tiny = build_long_run_manifest(run_id="tiny-run", timesteps=128)
    long = build_long_run_manifest(run_id="long-run", timesteps=5_000_000)

    assert "--replay-save-interval 1" in tiny["shell_script"]
    assert "--long-run-require-head-to-head" not in tiny["shell_script"]
    assert "--long-run-min-head-to-head-episodes" not in tiny["shell_script"]
    assert "--long-run-min-head-to-head-map-episodes" not in tiny["shell_script"]
    assert "--long-run-min-replay-combat-maps 4" in tiny["shell_script"]
    assert "--long-run-min-opponent-historical-samples" not in tiny["shell_script"]
    assert "--long-run-require-candidate-integrity" in tiny["shell_script"]
    assert tiny["manifest_config"]["replay_save_interval"] == 1
    assert tiny["manifest_config"]["replay_save_interval_source"] == "auto_small_run"
    assert tiny["manifest_config"]["require_head_to_head"] is False
    assert tiny["manifest_config"]["min_replay_combat_maps"] == 4
    assert tiny["manifest_config"]["min_opponent_historical_samples"] == 0
    assert tiny["manifest_config"]["min_head_to_head_episodes"] == 0
    assert tiny["manifest_config"]["min_head_to_head_map_episodes"] is None
    assert tiny["manifest_config"]["require_candidate_integrity"] is True
    assert "--replay-save-interval" not in long["shell_script"]
    assert "--long-run-require-head-to-head" in long["shell_script"]
    assert "--long-run-min-head-to-head-episodes 160" in long["shell_script"]
    assert "--long-run-min-head-to-head-map-episodes 40" in long["shell_script"]
    assert "--long-run-min-replay-combat-maps 4" in long["shell_script"]
    assert "--long-run-min-opponent-historical-samples 1" in long["shell_script"]
    assert "--long-run-require-candidate-integrity" in long["shell_script"]
    assert long["manifest_config"]["replay_save_interval"] is None
    assert long["manifest_config"]["replay_save_interval_source"] == "config"
    assert long["manifest_config"]["require_head_to_head"] is True
    assert long["manifest_config"]["min_replay_combat_maps"] == 4
    assert long["manifest_config"]["min_opponent_historical_samples"] == 1
    assert long["manifest_config"]["min_head_to_head_episodes"] == 160
    assert long["manifest_config"]["min_head_to_head_map_episodes"] == 40
    assert long["manifest_config"]["require_candidate_integrity"] is True


def test_run_compare_rejects_non_eval_artifacts(tmp_path):
    before = tmp_path / "before.json"
    after = tmp_path / "after.json"
    before.write_text('{"artifact": {"artifact_type": "rank", "schema_version": 1}}\n')
    after.write_text('{"artifact": {"artifact_type": "eval", "schema_version": 1}}\n')

    try:
        run_compare(str(before), str(after))
    except ValueError as exc:
        assert "Expected eval artifact, got rank" in str(exc)
    else:
        raise AssertionError("expected compare to reject rank artifact")


def test_run_rank_gate_rejects_non_rank_artifacts(tmp_path):
    path = tmp_path / "eval.json"
    path.write_text('{"artifact": {"artifact_type": "eval", "schema_version": 1}}\n')

    try:
        run_rank_gate(
            str(path),
            min_score=0.1,
            min_win_rate=0.0,
            max_draw_rate=0.9,
            max_no_damage_rate=0.75,
            max_low_engagement_rate=0.5,
            min_head_to_head_elo=None,
            min_head_to_head_score=None,
        )
    except ValueError as exc:
        assert "Expected rank artifact, got eval" in str(exc)
    else:
        raise AssertionError("expected rank gate to reject eval artifact")
