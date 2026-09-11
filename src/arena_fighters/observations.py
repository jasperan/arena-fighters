"""Observation and action transformation helpers for shared-weight self-play.

One network plays both sides. To do that soundly, every observation handed to
the network must be expressed in the *canonical* frame the network was trained
in: agent 0's frame, where the controlled fighter sits on the left and faces
right. Reflecting the grid horizontally is exactly that transformation, and it
is an isometry, so every geometric relation the policy reads -- own position,
opponent position, facing marker, bullet positions, platform spans -- stays
consistent with the physical situation.

Because the reflection flips the meaning of "left" and "right", the action the
policy returns is expressed in the mirrored frame too and must be converted
back before it reaches the environment. Only the two horizontal movement
actions change meaning; jumps, ducks, shots, and melee are mirror-symmetric.
"""

from __future__ import annotations

import numpy as np

from arena_fighters.config import MOVE_LEFT, MOVE_RIGHT


def mirror_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Reflect an observation into the canonical (agent 0) frame.

    Every channel is reflected: positions, own/opponent bullets, and the facing
    marker. Channel *labels* stay attached to the fighter they describe, so a
    policy reasoning about "the opponent bullet 3 tiles to my left" sees the
    same relation after the transform -- no channel is exchanged.

    Note that the raw observation already labels channels from the acting
    agent's point of view, so this function only reflects geometry; it must not
    also swap own/opponent channels (doing both cancels the reflection for
    positions while double-transforming bullets, which blinds the agent to
    incoming fire).
    """
    return {
        "grid": np.flip(obs["grid"], axis=2).copy(),
        "vector": obs["vector"].copy(),
    }


def mirror_action(action: int) -> int:
    """Convert a canonical-frame action back into the true arena frame."""
    if action == MOVE_LEFT:
        return MOVE_RIGHT
    if action == MOVE_RIGHT:
        return MOVE_LEFT
    return action
