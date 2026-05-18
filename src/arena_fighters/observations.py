"""Observation transformation helpers."""

from __future__ import annotations

import numpy as np

from arena_fighters.config import (
    CH_OPP_BULLETS,
    CH_OPP_POS,
    CH_OWN_BULLETS,
    CH_OWN_POS,
    VEC_OPP_HP,
    VEC_OWN_HP,
)


def mirror_obs(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Mirror an observation so agent 1 can reuse agent 0 policy semantics."""
    grid = obs["grid"].copy()
    vector = obs["vector"].copy()

    grid = np.flip(grid, axis=2).copy()
    grid[[CH_OWN_POS, CH_OPP_POS]] = grid[[CH_OPP_POS, CH_OWN_POS]]
    grid[[CH_OWN_BULLETS, CH_OPP_BULLETS]] = grid[
        [CH_OPP_BULLETS, CH_OWN_BULLETS]
    ]
    vector[VEC_OWN_HP], vector[VEC_OPP_HP] = (
        vector[VEC_OPP_HP],
        vector[VEC_OWN_HP],
    )

    return {"grid": grid, "vector": vector}
