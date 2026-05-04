import json

from scripts.self_play_sampling_smoke import (
    build_self_play_sampling_summary,
    parse_map_pool,
    validate_self_play_sampling_summary,
    write_sampling_summary,
)


def test_build_self_play_sampling_summary_records_historical_samples_and_maps():
    summary = build_self_play_sampling_summary(
        pool_seed=123,
        reset_seed=1000,
        snapshot_count=5,
        reset_count=32,
        latest_opponent_prob=0.8,
        min_historical_samples=1,
        min_maps_seen=2,
        map_pool=("classic", "flat", "split", "tower"),
    )

    assert summary["artifact"] == {
        "artifact_type": "self_play_sampling_smoke",
        "schema_version": 1,
    }
    assert summary["passed"] is True
    assert summary["historical_samples"] >= 1
    assert summary["latest_samples"] >= 1
    assert summary["unique_maps_seen"] >= 2
    assert summary["loaded_reset_count"] == 32
    assert all(check["passed"] for check in summary["checks"])
    assert {sample["sample_kind"] for sample in summary["samples"]} >= {
        "latest",
        "historical",
    }


def test_self_play_sampling_summary_is_reproducible_with_seed():
    first = build_self_play_sampling_summary(pool_seed=123, reset_count=16)
    second = build_self_play_sampling_summary(pool_seed=123, reset_count=16)

    assert first["samples"] == second["samples"]
    assert first["pool_stats"] == second["pool_stats"]


def test_validate_self_play_sampling_summary_rejects_failed_checks():
    summary = build_self_play_sampling_summary(
        pool_seed=123,
        reset_count=4,
        latest_opponent_prob=1.0,
        min_historical_samples=1,
    )

    try:
        validate_self_play_sampling_summary(summary)
    except RuntimeError as exc:
        assert "historical_samples_meet_minimum" in str(exc)
    else:
        raise AssertionError("Expected failed self-play sampling smoke to raise")


def test_parse_map_pool_validates_names():
    assert parse_map_pool("flat,tower") == ("flat", "tower")

    try:
        parse_map_pool("flat,missing")
    except ValueError as exc:
        assert "Unknown map names: missing" in str(exc)
    else:
        raise AssertionError("Expected unknown map to fail")


def test_write_sampling_summary_creates_parent_dirs(tmp_path):
    summary = {"artifact": {"artifact_type": "self_play_sampling_smoke"}}
    path = tmp_path / "nested" / "sampling-summary.json"

    written = write_sampling_summary(summary, path)

    assert written == path
    assert json.loads(path.read_text()) == summary
