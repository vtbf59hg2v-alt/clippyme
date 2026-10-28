"""Clip style presets: how long a clip may run and which selection rules apply.

``CLIP_STYLE`` picks the preset: ``default`` (15–60s viral moments, the stock
prompt) or ``comedy`` (sketch shows and stand-up: whole bits of 40–120s, a
self-contained joke may run 30s). ``target`` is what the prompt asks for and
the sentence snap may grow a clip to; ``hard`` is what validation still
accepts, a little wider so near-misses survive.
"""
import os

STYLES = {
    "default": {"target": (15, 60), "hard": (10, 75)},
    "comedy": {"target": (40, 120), "hard": (25, 130)},
}


def current_style() -> str:
    """The configured style name, ``default`` when unset or unknown."""
    name = (os.getenv("CLIP_STYLE") or "default").strip().lower()
    return name if name in STYLES else "default"


def target_bounds(style: str | None = None) -> tuple[int, int]:
    return STYLES[style or current_style()]["target"]


def hard_bounds(style: str | None = None) -> tuple[int, int]:
    return STYLES[style or current_style()]["hard"]
