from dataclasses import dataclass, field


@dataclass(frozen=True)
class ArenaConfig:
    width: int = 40
    height: int = 20
    max_ticks: int = 500
    map_name: str = "classic"
    randomize_maps: bool = False
    map_choices: tuple[str, ...] = ("classic", "flat", "split", "tower")


@dataclass(frozen=True)
class AgentConfig:
    start_hp: int = 1
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
    no_damage_draw_penalty: float = 0.0
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
    opponent_pool_seed: int | None = None
    curriculum_name: str | None = None
    replay_save_interval: int = 100


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    start_step: int
    map_choices: tuple[str, ...]
    reward_preset: str = "default"


@dataclass(frozen=True)
class Config:
    arena: ArenaConfig = field(default_factory=ArenaConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)


REWARD_PRESETS = {
    "default": RewardConfig(),
    "anti_stall": RewardConfig(
        win=12.0,
        lose=-12.0,
        draw=-5.0,
        no_damage_draw_penalty=-5.0,
        deal_damage_per_hp=0.2,
        take_damage_per_hp=-0.1,
        idle_penalty=-0.01,
    ),
}
REWARD_PRESET_ALIASES = {
    "anti-stall": "anti_stall",
}


def reward_config_for_preset(name: str) -> RewardConfig:
    preset_name = REWARD_PRESET_ALIASES.get(name, name)
    if preset_name not in REWARD_PRESETS:
        raise ValueError(f"Unknown reward preset: {name}")
    return REWARD_PRESETS[preset_name]


# Platform layouts: each tuple is (x_start, x_end, y).
# y=0 is top of grid, y=19 is bottom.
PLATFORM_LAYOUTS = {
    "classic": (
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
    ),
    "flat": (
        (0, 39, 19),
    ),
    "split": (
        (0, 39, 19),
        (2, 10, 15),
        (29, 37, 15),
        (13, 26, 12),
        (5, 15, 8),
        (24, 34, 8),
    ),
    "tower": (
        (0, 39, 19),
        (17, 22, 16),
        (14, 25, 13),
        (11, 28, 10),
        (8, 31, 7),
    ),
}

# Backwards-compatible alias for the default map.
PLATFORM_LAYOUT = list(PLATFORM_LAYOUTS["classic"])

CURRICULUMS = {
    "map_progression": (
        CurriculumStage(
            name="flat_intro",
            start_step=0,
            map_choices=("flat",),
        ),
        CurriculumStage(
            name="classic_duel",
            start_step=250_000,
            map_choices=("flat", "classic"),
        ),
        CurriculumStage(
            name="mixed_routes",
            start_step=1_000_000,
            map_choices=("classic", "split"),
            reward_preset="anti_stall",
        ),
        CurriculumStage(
            name="full_map_pool",
            start_step=2_500_000,
            map_choices=("classic", "flat", "split", "tower"),
            reward_preset="anti_stall",
        ),
    ),
}


def curriculum_for_name(name: str) -> tuple[CurriculumStage, ...]:
    if name not in CURRICULUMS:
        raise ValueError(f"Unknown curriculum: {name}")
    validate_curriculum(name)
    return CURRICULUMS[name]


def curriculum_stage_for_step(name: str, step: int) -> CurriculumStage:
    if step < 0:
        raise ValueError("step must be non-negative")

    stages = curriculum_for_name(name)
    current = stages[0]
    for stage in stages:
        if stage.start_step <= step:
            current = stage
        else:
            break
    return current


def validate_curriculum(name: str) -> bool:
    if name not in CURRICULUMS:
        raise ValueError(f"Unknown curriculum: {name}")

    stages = CURRICULUMS[name]
    if not stages or stages[0].start_step != 0:
        raise ValueError(f"Curriculum must start at step 0: {name}")

    previous_step = -1
    for stage in stages:
        if stage.start_step <= previous_step:
            raise ValueError(f"Curriculum stages must be ordered: {name}")
        previous_step = stage.start_step
        for map_name in stage.map_choices:
            if map_name not in PLATFORM_LAYOUTS:
                raise ValueError(f"Unknown map in curriculum {name}: {map_name}")
        reward_config_for_preset(stage.reward_preset)
    return True

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
