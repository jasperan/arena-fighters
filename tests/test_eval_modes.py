"""Tests split from the former test_training_callback catch-all.

Shared fixtures, fake doubles, and artifact builders live in
``tests._training_helpers``.
"""

from tests._training_helpers import *  # noqa: F401,F403


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
        min_map_score=None,
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
            min_map_score=None,
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


def test_run_rank_gate_can_require_minimum_per_map_score(tmp_path):
    rank_summary = tmp_path / "rank.json"
    rank = _rank_summary(label="aggregate-good", score=0.5, win_rate=0.5)
    rank["rankings"][0]["matchup_scores"] = [
        {"map_name": "classic", "opponent": "idle", "score": 0.5, "episodes": 10},
        {"map_name": "flat", "opponent": "idle", "score": -0.25, "episodes": 10},
    ]
    rank_summary.write_text(json.dumps(rank) + "\n")
    output_dir = tmp_path / "outputs"

    try:
        run_rank_gate(
            str(rank_summary),
            min_score=0.1,
            min_win_rate=0.0,
            max_draw_rate=0.9,
            max_no_damage_rate=0.75,
            max_low_engagement_rate=0.5,
            min_map_score=0.0,
            min_head_to_head_elo=None,
            min_head_to_head_score=None,
            output_dir=str(output_dir),
            output_label="map-gate",
        )
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("expected weak map rank gate to exit non-zero")

    [saved_path] = output_dir.glob("*_map-gate.json")
    saved = json.loads(saved_path.read_text())
    assert saved["passed"] is False
    assert saved["rules"]["min_map_score"] == 0.0
    assert saved["failures"] == [
        {
            "metric": "per_map_score",
            "value": -0.25,
            "min": 0.0,
            "reason": "below_minimum",
            "low_score_maps": [
                {
                    "map_name": "flat",
                    "mean_score": -0.25,
                    "matchup_count": 1,
                    "episode_count": 10,
                },
            ],
            "per_map_scores": [
                {
                    "map_name": "classic",
                    "mean_score": 0.5,
                    "matchup_count": 1,
                    "episode_count": 10,
                },
                {
                    "map_name": "flat",
                    "mean_score": -0.25,
                    "matchup_count": 1,
                    "episode_count": 10,
                },
            ],
        }
    ]


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
            min_map_score=None,
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
