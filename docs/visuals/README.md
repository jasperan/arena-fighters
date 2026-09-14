# Broadcast replay renderer

## Capture notes

PNG/GIF files come from the actual ArenaFightersEnv and Pillow renderer: classic map, seed 12, NumPy random actions, 100 ticks maximum, one captured frame per 10 ticks, 24px cells. This is visual rendering evidence, not trained-agent performance. No checkpoints were created or training run.

Web captures use Chromium at 1440 × 1080 (desktop) and 390 × 1080 (mobile), with reduced motion enabled. Screenshots are real rendered interfaces, not image-generated UI mockups. Raster renderer examples retain their native dimensions.

## Verification — 2026-09-14

`PYTHONPATH=src python -m pytest tests/test_renderer.py -q` passed 12 tests before and after, using the project's environment. Tests cover fighter/bullet coordinates, all maps, cached-frame isolation, GIF/contact-sheet output, and the short renderer CLI flow.

Only the existing visual surfaces were changed. The screenshots are not evidence of end-to-end service availability, accessibility certification, or production performance.
