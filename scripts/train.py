#!/usr/bin/env python
"""CLI entrypoint for arena-fighters: train, watch, and replay modes."""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

from arena_fighters.config import Config, IDLE, NUM_ACTIONS
from arena_fighters.env import ArenaFightersEnv
from arena_fighters.network import ArenaFeaturesExtractor
from arena_fighters.replay import load_replay
from arena_fighters.self_play import OpponentPool, SelfPlayWrapper
from stable_baselines3.common.callbacks import BaseCallback


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


class SelfPlayCallback(BaseCallback):
    """Snapshots weights into the opponent pool at regular intervals."""

    def __init__(
        self,
        wrapper: SelfPlayWrapper,
        opponent_pool: OpponentPool,
        snapshot_interval: int = 50,
        checkpoint_dir: str = "checkpoints",
        verbose: int = 0,
    ):
        super().__init__(verbose=verbose)
        self.wrapper = wrapper
        self.opponent_pool = opponent_pool
        self.snapshot_interval = snapshot_interval
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._rollout_count = 0
        self._milestones = {100_000, 500_000, 1_000_000, 5_000_000, 10_000_000,
                            50_000_000, 100_000_000}
        self._milestones_hit: set[int] = set()

    def _on_step(self) -> bool:
        # Check for milestone checkpoints based on total timesteps
        steps = self.num_timesteps
        for m in self._milestones:
            if m not in self._milestones_hit and steps >= m:
                self._milestones_hit.add(m)
                label = f"{m // 1_000_000}M" if m >= 1_000_000 else f"{m // 1_000}K"
                path = self.checkpoint_dir / f"ppo_{label}"
                self.model.save(str(path))
                if self.verbose:
                    print(f"[Milestone] {label} steps reached, saved to {path}")
        return True

    def _on_rollout_end(self) -> None:
        self._rollout_count += 1
        if self._rollout_count % self.snapshot_interval != 0:
            return

        # Snapshot current weights into the pool
        state_dict = self.model.policy.state_dict()
        self.opponent_pool.add(state_dict)

        # Save checkpoint
        ckpt_path = self.checkpoint_dir / f"ppo_snap_{self._rollout_count}"
        self.model.save(str(ckpt_path))

        # Update the wrapper's opponent policy to the current model
        self.wrapper.opponent_policy = self.model

        if self.verbose:
            print(
                f"[Snapshot] rollout={self._rollout_count}  "
                f"pool_size={len(self.opponent_pool)}  "
                f"saved={ckpt_path}"
            )


def run_train(cfg: Config, checkpoint_dir: str, replay_dir: str) -> None:
    """Headless PPO self-play training."""
    from stable_baselines3 import PPO

    pool = OpponentPool(max_size=cfg.training.opponent_pool_size)
    wrapper = SelfPlayWrapper(config=cfg, opponent_pool=pool)

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
    wrapper.opponent_policy = model

    callback = SelfPlayCallback(
        wrapper=wrapper,
        opponent_pool=pool,
        snapshot_interval=cfg.training.snapshot_interval,
        checkpoint_dir=checkpoint_dir,
        verbose=1,
    )

    model.learn(total_timesteps=cfg.training.total_timesteps, callback=callback)

    final_path = os.path.join(checkpoint_dir, "ppo_final")
    model.save(final_path)
    print(f"Training complete. Final model saved to {final_path}")


def run_watch(cfg: Config, checkpoint: str | None, num_rounds: int = 0) -> None:
    """Load a checkpoint and play games in ASCII with score tracking.

    Args:
        num_rounds: number of rounds to play. 0 = infinite (Ctrl+C to stop).
    """
    import numpy as np

    model = None
    if checkpoint and Path(checkpoint).exists():
        from stable_baselines3 import PPO

        model = PPO.load(checkpoint)

    score = [0, 0]  # [agent_0 wins, agent_1 wins]
    draws = 0
    round_num = 0

    try:
        while num_rounds == 0 or round_num < num_rounds:
            round_num += 1
            env = ArenaFightersEnv(config=cfg, render_mode="ansi")
            obs_dict, _ = env.reset()

            # Round start splash
            os.system("clear" if os.name != "nt" else "cls")
            print(f"\033[1;36m{'ARENA FIGHTERS':^44}\033[0m")
            print(f"\033[90m{'=' * 44}\033[0m")
            print(f"\n  \033[1mRound {round_num}\033[0m")
            print(f"  Score: \033[1;33m@\033[0m {score[0]}  -  {score[1]} \033[1;35mX\033[0m  (draws: {draws})")
            if model:
                print(f"  \033[90mCheckpoint: {checkpoint}\033[0m")
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

                os.system("clear" if os.name != "nt" else "cls")
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
            os.system("clear" if os.name != "nt" else "cls")
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

    print(f"Replay: {path.name}  Winner: {winner}  Length: {length} ticks")
    time.sleep(1)

    h = cfg.arena.height
    w = cfg.arena.width

    for frame in frames:
        # Build ASCII display from frame state
        display = [["." for _ in range(w)] for _ in range(h)]

        # Draw platforms
        from arena_fighters.config import PLATFORM_LAYOUT

        for x_start, x_end, y in PLATFORM_LAYOUT:
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

        os.system("clear" if os.name != "nt" else "cls")
        print(header)
        for row in display:
            print("".join(row))
        time.sleep(0.1)

    print(f"\nReplay complete. Winner: {winner}")


def main():
    parser = argparse.ArgumentParser(
        description="Arena Fighters: train, watch, or replay"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "watch", "replay"],
        default="train",
        help="Operating mode (default: train)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for watch mode)",
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
        help="Override total_timesteps from config",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory for saving checkpoints (default: checkpoints)",
    )
    parser.add_argument(
        "--replay-dir",
        type=str,
        default="replays",
        help="Directory for replay files (default: replays)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=0,
        help="Number of rounds to watch (0 = infinite, Ctrl+C to stop)",
    )
    args = parser.parse_args()

    cfg = Config()

    # Override timesteps if provided
    if args.timesteps is not None:
        cfg = replace(
            cfg,
            training=replace(cfg.training, total_timesteps=args.timesteps),
        )

    if args.mode == "train":
        run_train(cfg, args.checkpoint_dir, args.replay_dir)
    elif args.mode == "watch":
        run_watch(cfg, args.checkpoint, num_rounds=args.rounds)
    elif args.mode == "replay":
        if not args.episode:
            parser.error("--episode is required for replay mode")
        run_replay(cfg, args.episode)


if __name__ == "__main__":
    main()
