#!/usr/bin/env python
"""Render an arena-fighters match to PNG frames, a GIF, and a contact sheet.

Examples:
    # Built-in policies, classic map
    python scripts/render_episode.py --agent-policy scripted --opponent aggressive

    # A trusted checkpoint against a baseline
    python scripts/render_episode.py --checkpoint checkpoints/ppo_final \\
        --trusted-checkpoint-manifest checkpoints/checkpoint-trust-manifest.json \\
        --opponent scripted --output-dir renders/ppo-final

    # Render a saved replay (frames are already logged)
    python scripts/render_episode.py --replay replays/episode_0100.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arena_fighters.config import Config
from arena_fighters.env import ArenaFightersEnv
from arena_fighters.evaluation import ModelPolicy, make_builtin_policy
from arena_fighters.render import (
    render_state,
    save_contact_sheet,
    save_gif,
)
from arena_fighters.replay import load_replay
from scripts.train import load_trusted_ppo_checkpoint


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--trusted-checkpoint-manifest", type=str, default=None)
    parser.add_argument("--allow-unverified-checkpoints", action="store_true")
    parser.add_argument("--agent-policy", type=str, default="scripted")
    parser.add_argument("--opponent", type=str, default="scripted")
    parser.add_argument("--map", dest="map_name", type=str, default="classic")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-ticks", type=int, default=None)
    parser.add_argument("--replay", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="renders/episode")
    parser.add_argument("--cell", type=int, default=24)
    parser.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    parser.add_argument("--contact-frames", type=int, default=6)
    parser.add_argument("--gif", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gif-duration-ms", type=int, default=90)
    parser.add_argument("--frames", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args(argv)


def _load_manifest(path: str | None) -> dict[str, str] | None:
    if path is None:
        return None
    from scripts.train import load_checkpoint_trust_manifest

    return load_checkpoint_trust_manifest(path)


def collect_env_frames(
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    cfg = Config()
    if args.max_ticks is not None:
        from dataclasses import replace

        cfg = replace(cfg, arena=replace(cfg.arena, max_ticks=args.max_ticks))
    from dataclasses import replace

    cfg = replace(cfg, arena=replace(cfg.arena, map_name=args.map_name))

    env = ArenaFightersEnv(config=cfg)
    obs, info = env.reset(seed=args.seed)

    if args.checkpoint:
        manifest = _load_manifest(args.trusted_checkpoint_manifest)
        model = load_trusted_ppo_checkpoint(
            args.checkpoint,
            trusted_checkpoint_manifest=manifest,
            allow_unverified=args.allow_unverified_checkpoints,
        )
        agent_policy = ModelPolicy(model=model, deterministic=True)
    else:
        agent_policy = make_builtin_policy(args.agent_policy, seed=args.seed)
    opponent_policy = make_builtin_policy(args.opponent, seed=args.seed + 1)

    frames = [env.get_state()]
    while env.agents:
        actions = {
            "agent_0": agent_policy.act("agent_0", obs["agent_0"], env),
            "agent_1": opponent_policy.act("agent_1", obs["agent_1"], env),
        }
        obs, _rewards, terminations, truncations, _info = env.step(actions)
        frames.append(env.get_state())
        if any(terminations.values()) or any(truncations.values()):
            break
    return frames, info


def collect_replay_frames(path: str) -> list[dict]:
    replay = load_replay(Path(path))
    frames = replay.get("frames") or []
    if not frames:
        raise ValueError(f"Replay has no frames: {path}")
    return frames


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.replay:
        frames = collect_replay_frames(args.replay)
        map_name = str(frames[0].get("map_name", args.map_name))
        max_hp = 1
        title = args.title or f"replay {Path(args.replay).name}"
    else:
        frames, info = collect_env_frames(args)
        map_name = str(info.get("map_name", args.map_name))
        max_hp = Config().agent.start_hp
        agent_label = args.checkpoint or args.agent_policy
        title = args.title or f"{agent_label} vs {args.opponent} @ {map_name}"

    stride = max(1, args.stride)
    selected = frames[::stride]
    if selected[-1] is not frames[-1]:
        selected.append(frames[-1])

    rendered = [
        render_state(frame, cell=args.cell, max_hp=max_hp, title=title)
        for frame in selected
    ]

    if args.frames:
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for index, image in enumerate(rendered):
            image.save(frames_dir / f"frame_{index:04d}.png")

    if args.gif and len(rendered) > 1:
        save_gif(
            rendered,
            output_dir / "match.gif",
            duration_ms=args.gif_duration_ms,
        )

    if args.contact_frames > 0:
        count = min(args.contact_frames, len(rendered))
        if count == 1:
            picks = [rendered[0]]
        else:
            step = (len(rendered) - 1) / (count - 1)
            picks = [rendered[round(index * step)] for index in range(count)]
        save_contact_sheet(picks, output_dir / "contact-sheet.png", columns=3)

    summary = {
        "map_name": map_name,
        "frames": len(frames),
        "rendered_frames": len(rendered),
        "stride": stride,
        "output_dir": str(output_dir),
        "gif": str(output_dir / "match.gif") if args.gif and len(rendered) > 1 else None,
        "contact_sheet": (
            str(output_dir / "contact-sheet.png") if args.contact_frames > 0 else None
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
