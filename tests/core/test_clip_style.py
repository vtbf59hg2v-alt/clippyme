"""CLIP_STYLE presets: duration bounds and what validation accepts."""
import pytest

from clippyme.core.clip_schema import ViralClip
from clippyme.core.clip_style import current_style, hard_bounds, target_bounds


def _clip(duration):
    return {"start": 100.0, "end": 100.0 + duration, "viral_score": 80,
            "viral_reason": "a complete bit with setup, punchline and laugh"}


def test_unknown_or_missing_style_is_default(monkeypatch):
    monkeypatch.delenv("CLIP_STYLE", raising=False)
    assert current_style() == "default"
    monkeypatch.setenv("CLIP_STYLE", "nonsense")
    assert current_style() == "default"
    monkeypatch.setenv("CLIP_STYLE", " Comedy ")
    assert current_style() == "comedy"


def test_bounds_per_style():
    assert target_bounds("default") == (15, 60)
    assert target_bounds("comedy") == (40, 120)
    assert hard_bounds("comedy")[1] > 120


def test_default_style_still_rejects_two_minute_clips(monkeypatch):
    monkeypatch.delenv("CLIP_STYLE", raising=False)
    with pytest.raises(ValueError):
        ViralClip.model_validate(_clip(110))


def test_comedy_style_accepts_whole_bits_and_short_jokes(monkeypatch):
    monkeypatch.setenv("CLIP_STYLE", "comedy")
    assert ViralClip.model_validate(_clip(110)).end == 210.0
    assert ViralClip.model_validate(_clip(30)).end == 130.0
    with pytest.raises(ValueError):
        ViralClip.model_validate(_clip(200))


def test_visual_gag_flag_survives_validation():
    clip = ViralClip.model_validate({**_clip(40), "visual_gag": True})
    assert clip.model_dump()["visual_gag"] is True
