"""Tests split from the former test_training_callback catch-all.

Shared fixtures, fake doubles, and artifact builders live in
``tests._training_helpers``.
"""

from tests._training_helpers import *  # noqa: F401,F403


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
