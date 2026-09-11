"""Raster (PNG/GIF) renderer for arena-fighters matches.

The terminal renderer in :mod:`arena_fighters.env` is fine for live watching,
but strategy review needs frames that survive outside a terminal. This module
turns a serializable state dict (``ArenaFightersEnv.get_state()``) or replay
frames into Pillow images, so the same renderer can draw live episodes and
saved replays.

Pillow is already available through ``matplotlib``, which Stable-Baselines3
depends on directly, so no new dependency is introduced.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from arena_fighters.config import PLATFORM_LAYOUTS

# --- Theme -----------------------------------------------------------------

BACKGROUND_TOP = (12, 16, 38)
BACKGROUND_BOTTOM = (32, 40, 84)
PLATFORM_FILL = (52, 66, 112)
PLATFORM_TOP = (122, 176, 255)
PLATFORM_BORDER = (28, 36, 66)
AGENT_COLORS = {
    "agent_0": (255, 179, 71),
    "agent_1": (255, 93, 143),
}
AGENT_OUTLINE = (16, 18, 30)
BULLET_COLORS = {
    "agent_0": (255, 236, 120),
    "agent_1": (255, 148, 196),
}
HUD_TEXT = (222, 230, 255)
HUD_MUTED = (170, 184, 222)
HP_TRACK = (44, 52, 80)
HP_FULL = (116, 231, 149)
HP_LOW = (255, 113, 113)

DEFAULT_CELL = 24
HUD_TOP = 62
HUD_BOTTOM = 16
SIDE_PADDING = 18


@lru_cache(maxsize=8)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load DejaVu from matplotlib when available; fall back to PIL's bitmap."""
    try:
        import matplotlib

        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / name
        if path.is_file():
            return ImageFont.truetype(str(path), size=size)
    except Exception:  # pragma: no cover - font is cosmetic, never fatal
        pass
    return ImageFont.load_default()


def _platform_cells(map_name: str) -> set[tuple[int, int]]:
    layout = PLATFORM_LAYOUTS.get(map_name, PLATFORM_LAYOUTS["classic"])
    cells: set[tuple[int, int]] = set()
    for x_start, x_end, y in layout:
        for x in range(x_start, x_end + 1):
            cells.add((x, y))
    return cells


def _platform_runs(cells: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    """Group platform cells into contiguous horizontal runs (start_x, end_x, y)."""
    by_row: dict[int, list[int]] = {}
    for x, y in cells:
        by_row.setdefault(y, []).append(x)
    runs: list[tuple[int, int, int]] = []
    for y, columns in sorted(by_row.items()):
        xs = sorted(columns)
        start = previous = xs[0]
        for x in xs[1:]:
            if x == previous + 1:
                previous = x
                continue
            runs.append((start, previous, y))
            start = previous = x
        runs.append((start, previous, y))
    return runs


@lru_cache(maxsize=16)
def _background(width: int, height: int) -> Image.Image:
    """Cached gradient+vignette base. Callers MUST ``.copy()`` before drawing."""
    ramp = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    top = np.array(BACKGROUND_TOP, dtype=np.float32)
    bottom = np.array(BACKGROUND_BOTTOM, dtype=np.float32)
    gradient = top[None, :] * (1.0 - ramp) + bottom[None, :] * ramp
    array = np.repeat(gradient[:, None, :], width, axis=1).astype(np.float32)
    # Soft vignette keeps the eye on the arena instead of the corners.
    ys = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xs = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    radius = np.sqrt(xs**2 + ys**2) / np.sqrt(2.0)
    array *= (1.0 - 0.32 * radius**2)[:, :, None]
    return Image.fromarray(np.clip(array, 0, 255).astype(np.uint8), mode="RGB")


def _draw_grid(
    image: Image.Image,
    arena_width: int,
    arena_height: int,
    top_row: int,
    cell: int,
    origin_x: int,
    origin_y: int,
) -> None:
    """Faint cell grid so positions/trajectories are easy to read off."""
    grid = Image.new("RGBA", image.size, (0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(grid)
    top = max(0, origin_y + top_row * cell)
    bottom = min(image.height, origin_y + arena_height * cell)
    right = origin_x + arena_width * cell
    for x in range(arena_width + 1):
        px = origin_x + x * cell
        grid_draw.line((px, top, px, bottom), fill=(255, 255, 255, 10), width=1)
    for y in range(top_row, arena_height + 1):
        py = origin_y + y * cell
        grid_draw.line((origin_x, py, right, py), fill=(255, 255, 255, 10), width=1)
    image.paste(Image.alpha_composite(image.convert("RGBA"), grid).convert("RGB"), (0, 0))


def _draw_platforms(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    cells: set[tuple[int, int]],
    cell: int,
    origin_x: int,
    origin_y: int,
) -> None:
    """Draw platforms as continuous runs so tiles read as solid surfaces."""
    del draw
    radius = max(3, cell // 3)
    for start_x, end_x, y in _platform_runs(cells):
        left = origin_x + start_x * cell
        top = origin_y + y * cell
        right = origin_x + (end_x + 1) * cell
        body = Image.new("RGBA", (right - left, cell), (0, 0, 0, 0))
        body_draw = ImageDraw.Draw(body)
        body_draw.rounded_rectangle(
            (0, 0, right - left - 1, cell - 1),
            radius=radius,
            fill=PLATFORM_FILL,
            outline=PLATFORM_BORDER,
            width=1,
        )
        # Bright top rim only where the surface is exposed from above.
        exposed_segments: list[tuple[int, int]] = []
        segment_start: int | None = None
        for x in range(start_x, end_x + 2):
            exposed = x <= end_x and (x, y - 1) not in cells
            if exposed and segment_start is None:
                segment_start = x
            elif not exposed and segment_start is not None:
                exposed_segments.append((segment_start, x - 1))
                segment_start = None
        rim_h = max(2, cell // 8)
        for rim_start, rim_end in exposed_segments:
            rim_left = (rim_start - start_x) * cell + 2
            rim_right = (rim_end - start_x + 1) * cell - 2
            if rim_right - rim_left < 2:
                rim_left, rim_right = 0, right - left - 1
            body_draw.rounded_rectangle(
                (rim_left, 1, rim_right, rim_h),
                radius=max(1, rim_h // 2),
                fill=PLATFORM_TOP,
            )
        image.paste(body, (left, top), body)


def _draw_bullet_glow(
    image: Image.Image,
    bullets: Sequence[dict[str, Any]],
    cell: int,
    origin_x: int,
    origin_y: int,
) -> None:
    """Soft glow halo plus a fading comet trail behind each bullet."""
    if not bullets:
        return
    glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    for bullet in bullets:
        cx = origin_x + (bullet["x"] + 0.5) * cell
        cy = origin_y + (bullet["y"] + 0.5) * cell
        dx = int(np.sign(bullet.get("dx", 0)))
        dy = int(np.sign(bullet.get("dy", 0)))
        color = BULLET_COLORS.get(bullet.get("owner", "agent_0"), BULLET_COLORS["agent_0"])
        # Trail dots fade with distance and add to the comet glow before blur.
        for index, alpha in ((1, 120), (2, 70), (3, 35)):
            tx = cx - dx * cell * 0.5 * index
            ty = cy - dy * cell * 0.5 * index
            trail_r = max(1.0, cell * (0.2 - 0.04 * index))
            glow_draw.ellipse(
                (tx - trail_r, ty - trail_r, tx + trail_r, ty + trail_r),
                fill=(*color, alpha),
            )
        radius = cell * 0.36
        glow_draw.ellipse(
            (cx - radius, cy - radius, cx + radius, cy + radius),
            fill=(*color, 150),
        )
    glow = glow.filter(ImageFilter.GaussianBlur(radius=cell * 0.2))
    image.paste(Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB"), (0, 0))


def _draw_bullet_cores(
    draw: ImageDraw.ImageDraw,
    bullets: Sequence[dict[str, Any]],
    cell: int,
    origin_x: int,
    origin_y: int,
) -> None:
    for bullet in bullets:
        cx = origin_x + (bullet["x"] + 0.5) * cell
        cy = origin_y + (bullet["y"] + 0.5) * cell
        color = BULLET_COLORS.get(bullet.get("owner", "agent_0"), BULLET_COLORS["agent_0"])
        core = cell * 0.2
        draw.ellipse((cx - core, cy - core, cx + core, cy + core), fill=color)
        shine = cell * 0.08
        draw.ellipse(
            (cx - core * 0.4 - shine, cy - core * 0.4 - shine,
             cx - core * 0.4 + shine, cy - core * 0.4 + shine),
            fill=(255, 255, 255),
        )


def _draw_agent(
    image: Image.Image,
    state: dict[str, Any],
    cell: int,
    origin_x: int,
    origin_y: int,
    *,
    label: str | None = None,
) -> None:
    color = AGENT_COLORS.get(label or "agent_0", AGENT_COLORS["agent_0"])
    cx = origin_x + (state["x"] + 0.5) * cell
    feet_y = origin_y + (state["y"] + 1) * cell
    ducking = state.get("duck_ticks", 0) > 0
    body_h = cell * (0.62 if ducking else 1.0)
    body_w = cell * (0.94 if ducking else 0.7)
    top = feet_y - body_h
    left = cx - body_w / 2

    # Contact shadow so the fighter reads as standing on the surface.
    shadow = Image.new("RGBA", image.size, (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    shadow_draw.ellipse(
        (cx - cell * 0.42, feet_y - cell * 0.18, cx + cell * 0.42, feet_y + cell * 0.12),
        fill=(0, 0, 0, 130),
    )
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=cell * 0.12))
    image.paste(Image.alpha_composite(image.convert("RGBA"), shadow).convert("RGB"), (0, 0))

    body = Image.new("RGBA", image.size, (0, 0, 0, 0))
    body_draw = ImageDraw.Draw(body)
    radius = max(3, int(cell * 0.22))
    body_draw.rounded_rectangle(
        (left, top, left + body_w, feet_y - cell * 0.08),
        radius=radius,
        fill=color,
        outline=AGENT_OUTLINE,
        width=max(1, cell // 12),
    )
    highlight = tuple(int(channel + (255 - channel) * 0.35) for channel in color)
    body_draw.rounded_rectangle(
        (
            left + body_w * 0.18,
            top + body_h * 0.12,
            left + body_w * 0.82,
            top + body_h * 0.38,
        ),
        radius=max(1, int(radius * 0.7)),
        fill=highlight,
    )
    facing = 1 if state.get("facing", 1) >= 0 else -1
    # Legs and a stubby weapon keep the silhouette readable as a fighter.
    leg_color = tuple(int(channel * 0.55) for channel in color)
    leg_w = max(2.0, cell * 0.16)
    leg_top = feet_y - cell * 0.18
    for offset in (-cell * 0.2, cell * 0.06):
        body_draw.rounded_rectangle(
            (cx + offset, leg_top, cx + offset + leg_w, feet_y - cell * 0.02),
            radius=max(1, int(leg_w / 2)),
            fill=leg_color,
        )
    arm_y = top + body_h * 0.58
    arm_w = cell * 0.3
    body_draw.rounded_rectangle(
        (
            min(cx, cx + facing * arm_w),
            arm_y,
            max(cx, cx + facing * arm_w),
            arm_y + max(2.0, cell * 0.12),
        ),
        radius=max(1, int(cell * 0.06)),
        fill=(214, 220, 236),
        outline=AGENT_OUTLINE,
        width=1,
    )
    # Face marker: eye + chevron in the facing direction.
    eye_r = max(1.5, cell * 0.1)
    eye_x = cx + facing * cell * 0.14
    eye_y = top + body_h * 0.3
    body_draw.ellipse(
        (eye_x - eye_r, eye_y - eye_r, eye_x + eye_r, eye_y + eye_r),
        fill=(255, 255, 255, 235),
    )
    chevron_x = cx + facing * cell * 0.42
    chevron_y = top + body_h * 0.62
    body_draw.polygon(
        [
            (chevron_x, chevron_y - cell * 0.12),
            (chevron_x + facing * cell * 0.16, chevron_y),
            (chevron_x, chevron_y + cell * 0.12),
        ],
        fill=(255, 255, 255, 200),
    )
    image.paste(Image.alpha_composite(image.convert("RGBA"), body).convert("RGB"), (0, 0))


def _draw_hp_bar(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    width: int,
    height: int,
    hp: int,
    max_hp: int,
    *,
    right_aligned: bool = False,
) -> None:
    draw.rounded_rectangle(
        (x, y, x + width, y + height), radius=height // 2, fill=HP_TRACK
    )
    fraction = max(0.0, min(1.0, hp / max(1, max_hp)))
    if fraction <= 0.0:
        return
    fill_color = HP_FULL if fraction > 0.34 else HP_LOW
    bar_w = int((width - 2) * fraction)
    left = x + width - 1 - bar_w if right_aligned else x + 1
    draw.rounded_rectangle(
        (left, y + 1, left + bar_w, y + height - 1), radius=height // 2, fill=fill_color
    )


def _draw_hud(
    draw: ImageDraw.ImageDraw,
    image_width: int,
    state: dict[str, Any],
    score: tuple[int, int] | None,
    map_name: str,
    *,
    max_hp: int,
    tick: int | None,
    title: str | None,
) -> None:
    font_label = _font(15, bold=True)
    font_small = _font(13)
    font_title = _font(13)

    agents = state.get("agents", {})
    bar_w, bar_h = 150, 12

    for name, color, left_side in (
        ("agent_0", AGENT_COLORS["agent_0"], True),
        ("agent_1", AGENT_COLORS["agent_1"], False),
    ):
        agent = agents.get(name, {})
        hp = int(agent.get("hp", 0))
        label = "AGENT 0" if name == "agent_0" else "AGENT 1"
        if left_side:
            draw.text((SIDE_PADDING, 10), label, font=font_label, fill=color)
            _draw_hp_bar(draw, SIDE_PADDING, 32, bar_w, bar_h, hp, max_hp)
            draw.text(
                (SIDE_PADDING + bar_w + 8, 29),
                f"{max(0, hp)}/{max_hp}",
                font=font_small,
                fill=HUD_TEXT,
            )
        else:
            text_w = draw.textlength(label, font=font_label)
            draw.text(
                (image_width - SIDE_PADDING - text_w, 10),
                label,
                font=font_label,
                fill=color,
            )
            bar_x = image_width - SIDE_PADDING - bar_w
            _draw_hp_bar(draw, bar_x, 32, bar_w, bar_h, hp, max_hp, right_aligned=True)
            hp_text = f"{max(0, hp)}/{max_hp}"
            draw.text(
                (bar_x - 8 - draw.textlength(hp_text, font=font_small), 29),
                hp_text,
                font=font_small,
                fill=HUD_TEXT,
            )

    center = image_width // 2
    headline = title or map_name
    headline_w = draw.textlength(headline, font=font_title)
    draw.text((center - headline_w / 2, 6), headline, font=font_title, fill=HUD_MUTED)
    clock = tick if tick is not None else int(state.get("tick", 0))
    clock_text = f"tick {clock}"
    clock_w = draw.textlength(clock_text, font=font_small)
    draw.text((center - clock_w / 2, 26), clock_text, font=font_small, fill=HUD_TEXT)
    if score is not None:
        score_text = f"{score[0]} : {score[1]}"
        score_w = draw.textlength(score_text, font=font_label)
        draw.text(
            (center - score_w / 2, 42),
            score_text,
            font=font_label,
            fill=HUD_TEXT,
        )


def render_state(
    state: dict[str, Any],
    *,
    cell: int = DEFAULT_CELL,
    score: tuple[int, int] | None = None,
    max_hp: int = 1,
    tick: int | None = None,
    title: str | None = None,
    top_row: int | None = None,
    clip_headroom: int = 3,
    max_top_row: int = 6,
) -> Image.Image:
    """Render one serializable env/replay state dict into a Pillow image.

    ``top_row`` crops empty sky above the highest platform (auto-detected by
    default) so the arena fills the frame instead of occupying the lower half.
    """
    map_name = str(state.get("map_name", "classic"))
    layout = PLATFORM_LAYOUTS.get(map_name, PLATFORM_LAYOUTS["classic"])
    arena_height = max((y for _, _, y in layout), default=19) + 1
    # The playable grid is always 20 rows; derive width from the widest platform.
    arena_width = max((x_end for _, x_end, _ in layout), default=39) + 1
    if top_row is None:
        highest_platform = min(y for _, _, y in layout)
        top_row = max(0, min(max_top_row, highest_platform - clip_headroom))
    top_row = max(0, min(top_row, arena_height - 1))

    image_width = arena_width * cell + SIDE_PADDING * 2
    image_height = HUD_TOP + (arena_height - top_row) * cell + HUD_BOTTOM
    origin_x = SIDE_PADDING
    origin_y = HUD_TOP - top_row * cell

    image = _background(image_width, image_height).copy()
    _draw_grid(
        image,
        arena_width,
        arena_height,
        top_row,
        cell,
        origin_x,
        origin_y,
    )
    draw = ImageDraw.Draw(image)
    _draw_platforms(image, draw, _platform_cells(map_name), cell, origin_x, origin_y)

    bullets = list(state.get("bullets", []))
    _draw_bullet_glow(image, bullets, cell, origin_x, origin_y)
    draw = ImageDraw.Draw(image)
    _draw_bullet_cores(draw, bullets, cell, origin_x, origin_y)

    for name in ("agent_0", "agent_1"):
        agent = state.get("agents", {}).get(name)
        if agent is not None:
            _draw_agent(image, agent, cell, origin_x, origin_y, label=name)

    _draw_hud(
        ImageDraw.Draw(image),
        image_width,
        state,
        score,
        map_name,
        max_hp=max_hp,
        tick=tick,
        title=title,
    )
    return image


def render_env_frame(
    env: Any,
    *,
    cell: int = DEFAULT_CELL,
    score: tuple[int, int] | None = None,
    title: str | None = None,
) -> Image.Image:
    """Render a live ``ArenaFightersEnv`` (uses its serializable state)."""
    return render_state(
        env.get_state(),
        cell=cell,
        score=score,
        max_hp=env.cfg.agent.start_hp,
        title=title,
    )


def save_contact_sheet(
    frames: Sequence[Image.Image],
    path: str | Path,
    *,
    columns: int = 3,
    padding: int = 8,
    background: tuple[int, int, int] = (8, 10, 24),
) -> Path:
    """Compose frames into a single overview image."""
    if not frames:
        raise ValueError("frames must not be empty")
    columns = max(1, min(columns, len(frames)))
    rows = (len(frames) + columns - 1) // columns
    frame_w = max(frame.width for frame in frames)
    frame_h = max(frame.height for frame in frames)
    sheet = Image.new(
        "RGB",
        (
            columns * frame_w + (columns + 1) * padding,
            rows * frame_h + (rows + 1) * padding,
        ),
        background,
    )
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        sheet.paste(
            frame,
            (
                padding + column * (frame_w + padding),
                padding + row * (frame_h + padding),
            ),
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output


def save_gif(
    frames: Sequence[Image.Image],
    path: str | Path,
    *,
    duration_ms: int = 80,
    loop: int = 0,
    max_width: int | None = 720,
) -> Path:
    """Write an animated GIF, downscaling frames with nearest-neighbour clarity."""
    if not frames:
        raise ValueError("frames must not be empty")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    prepared = []
    for source in frames:
        converted = source.convert("P", palette=Image.ADAPTIVE, colors=128)
        if max_width is not None and converted.width > max_width:
            scale = max_width / converted.width
            converted = converted.resize(
                (max_width, max(1, int(converted.height * scale))), Image.NEAREST
            )
        prepared.append(converted)
    prepared[0].save(
        output,
        save_all=True,
        append_images=prepared[1:],
        duration=duration_ms,
        loop=loop,
    )
    return output
