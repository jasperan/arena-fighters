from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArenaConfig:
    width: int = 40
    height: int = 20
    max_ticks: int = 500


@dataclass(frozen=True)
class AgentConfig:
    start_hp: int = 100
    bullet_damage: int = 10
    melee_damage: int = 15
    shoot_cooldown: int = 5
    melee_cooldown: int = 3
    duck_duration: int = 2
    jump_height: int = 3
    bullet_speed: int = 2


@dataclass(frozen=True)
class RewardConfig:
    win: float = 10.0
    lose: float = -10.0
    draw: float = -1.0
    deal_damage_per_hp: float = 0.1
    take_damage_per_hp: float = -0.05
    idle_penalty: float = -0.001


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 3e-4
    batch_size: int = 2048
    mini_batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    total_timesteps: int = 10_000_000
    snapshot_interval: int = 50
    opponent_pool_size: int = 20
    latest_opponent_prob: float = 0.8


@dataclass(frozen=True)
class Config:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


# Platform layout: list of (x_start, x_end, y) tuples
# y=0 is top of grid, y=19 is bottom
PLATFORM_LAYOUT = [
    # Ground floor (y=19)
    (0, 39, 19),
    # Low side platforms (y=15)
    (1, 4, 15),
    (31, 34, 15),
    # Mid platform (y=12)
    (14, 21, 12),
    # High side platforms (y=9)
    (4, 13, 9),
    (26, 35, 9),
]

# Action enum
IDLE = 0
MOVE_LEFT = 1
MOVE_RIGHT = 2
JUMP = 3
DUCK = 4
SHOOT_FORWARD = 5
SHOOT_DIAG_UP = 6
SHOOT_DIAG_DOWN = 7
MELEE = 8
NUM_ACTIONS = 9

# Observation grid channels
CH_PLATFORMS = 0
CH_OWN_POS = 1
CH_OPP_POS = 2
CH_OWN_BULLETS = 3
CH_OPP_BULLETS = 4
CH_OWN_FACING = 5
NUM_CHANNELS = 6
NUM_VECTOR_OBS = 6
