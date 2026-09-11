from pathlib import Path
import subprocess
import sys

from PIL import Image

from arena_fighters.config import Config, PLATFORM_LAYOUTS
from arena_fighters.env import ArenaFightersEnv, Bullet
from arena_fighters.render import (
    AGENT_COLORS,
    HUD_TOP,
    SIDE_PADDING,
    render_env_frame,
    render_state,
    save_contact_sheet,
    save_gif,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pixel_for_agent(image: Image.Image, x: int, y: int, cell: int, top_row: int):
    origin_x = SIDE_PADDING
    origin_y = HUD_TOP - top_row * cell
    return (origin_x + int((x + 0.5) * cell), origin_y + int((y + 0.5) * cell))


def test_render_state_contains_agents_and_platforms():
    env = ArenaFightersEnv(config=Config())
    env.reset(seed=1)
    image = render_state(env.get_state(), cell=24)

    assert image.width == 40 * 24 + SIDE_PADDING * 2
    assert image.height > HUD_TOP
    colors = {color for _, color in image.getcolors(maxcolors=2**24)}
    assert AGENT_COLORS["agent_0"] in colors
    assert AGENT_COLORS["agent_1"] in colors


def test_render_state_paints_agents_at_their_grid_positions():
    env = ArenaFightersEnv(config=Config())
    env.reset(seed=1)
    state = env.get_state()
    cell = 24
    top_row = 6  # classic: highest platform at y=9 minus 3 rows headroom
    image = render_state(state, cell=cell, top_row=top_row)

    a0 = state["agents"]["agent_0"]
    a1 = state["agents"]["agent_1"]
    at_a0 = image.getpixel(_pixel_for_agent(image, a0["x"], a0["y"], cell, top_row))
    at_a1 = image.getpixel(_pixel_for_agent(image, a1["x"], a1["y"], cell, top_row))
    assert at_a0 == AGENT_COLORS["agent_0"]
    assert at_a1 == AGENT_COLORS["agent_1"]


def test_render_state_does_not_accumulate_previous_frames():
    """Successive frames must not inherit earlier drawings from cached bases."""
    env = ArenaFightersEnv(config=Config())
    env.reset(seed=1)
    cell = 24
    top_row = 6

    first = render_state(env.get_state(), cell=cell, top_row=top_row)
    env._agent_states["agent_0"].x = 30
    second = render_state(env.get_state(), cell=cell, top_row=top_row)

    old_position = _pixel_for_agent(first, 5, 18, cell, top_row)
    new_position = _pixel_for_agent(second, 30, 18, cell, top_row)
    assert first.getpixel(old_position) == AGENT_COLORS["agent_0"]
    assert second.getpixel(new_position) == AGENT_COLORS["agent_0"]
    # The stale agent must not still be painted at the old cell.
    assert second.getpixel(old_position) != AGENT_COLORS["agent_0"]

    repeat = render_state(env.get_state(), cell=cell, top_row=top_row)
    assert second.tobytes() == repeat.tobytes()


def test_render_env_frame_covers_every_map():
    for map_name in sorted(PLATFORM_LAYOUTS):
        env = ArenaFightersEnv(config=Config())
        env.reset(seed=2)
        env._map_name = map_name
        image = render_env_frame(env, cell=24)
        assert image.width == 40 * 24 + SIDE_PADDING * 2
        assert image.height > HUD_TOP


def test_render_state_draws_bullets():
    env = ArenaFightersEnv(config=Config())
    env.reset(seed=1)
    env._bullets = [Bullet(x=20, y=12, dx=2, dy=0, owner="agent_0")]
    image = render_state(env.get_state(), cell=24, top_row=6)
    from arena_fighters.render import BULLET_COLORS

    cell = 24
    origin_x = SIDE_PADDING
    origin_y = HUD_TOP - 6 * cell
    core = image.getpixel(
        (origin_x + int((20 + 0.5) * cell), origin_y + int((12 + 0.5) * cell))
    )
    assert core == BULLET_COLORS["agent_0"]


def test_save_contact_sheet_and_gif(tmp_path: Path):
    env = ArenaFightersEnv(config=Config())
    env.reset(seed=1)
    frames = []
    for tick in range(4):
        env._agent_states["agent_0"].x = 5 + tick
        frames.append(render_env_frame(env, cell=16))

    sheet_path = save_contact_sheet(frames, tmp_path / "sheet.png", columns=2)
    gif_path = save_gif(frames, tmp_path / "match.gif", duration_ms=60)

    assert sheet_path.is_file() and sheet_path.stat().st_size > 0
    assert gif_path.is_file() and gif_path.stat().st_size > 0
    with Image.open(gif_path) as gif:
        assert gif.n_frames == 4


def test_render_episode_cli_end_to_end(tmp_path: Path):
    output_dir = tmp_path / "render-out"
    result = subprocess.run(
        [
            sys.executable,
            "scripts/render_episode.py",
            "--agent-policy",
            "scripted",
            "--opponent",
            "idle",
            "--map",
            "flat",
            "--seed",
            "1",
            "--max-ticks",
            "12",
            "--stride",
            "1",
            "--cell",
            "16",
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "contact-sheet.png").is_file()
    assert (output_dir / "match.gif").is_file()
    assert any((output_dir / "frames").glob("*.png"))


def test_ansi_render_contains_agents():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    output = env.render()
    assert "@" in output
    assert "X" in output


def test_ansi_render_contains_hp():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    output = env.render()
    assert "1HP" in output


def test_ansi_render_contains_platforms():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    output = env.render()
    assert "platform" in output  # legend text


def test_ansi_render_shows_bullets():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    env._bullets.append(Bullet(x=20, y=10, dx=2, dy=0, owner="agent_0"))
    output = env.render()
    assert "-" in output  # horizontal bullet char


def test_ansi_render_shows_tick():
    env = ArenaFightersEnv(config=Config(), render_mode="ansi")
    env.reset()
    env._tick = 42
    output = env.render()
    assert "42" in output
