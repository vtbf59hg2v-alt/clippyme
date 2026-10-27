"""Tests for the ads layer (clippyme.editing.ads) and its compose wiring.

Pure helpers (settings clamping, banner window, break choice, filter graph)
run anywhere; the two render tests need ffmpeg and are skipped without it.
"""
import asyncio
import json
import os
import shutil
import subprocess

import pytest

from clippyme.editing import ads, compose


@pytest.fixture(autouse=True)
def _isolated_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(ads, "CONFIG_PATH", str(tmp_path / "ads.json"))
    monkeypatch.setattr(ads, "BANNER_PATH", str(tmp_path / "ad_banner.png"))
    monkeypatch.setattr(ads, "VIDEO_PATH", str(tmp_path / "ad_video.mp4"))


# --- settings --------------------------------------------------------------

def test_defaults_are_off():
    cfg = ads.load_config()
    assert cfg["enabled"] is False
    assert cfg["banner"]["enabled"] is False and cfg["video"]["enabled"] is False


def test_normalize_clamps_and_rejects_unknown_values():
    cfg = ads.normalize_config({
        "enabled": 1,
        "banner": {"start": -5, "duration": 99999, "position": "evil;rm", "size": "XL",
                   "opacity": 7, "anchor": "middle"},
        "video": {"placement": "x", "target_pct": 2},
    })
    b, v = cfg["banner"], cfg["video"]
    assert cfg["enabled"] is True
    assert b["start"] == 0.0 and b["duration"] == 600.0 and b["opacity"] == 1.0
    assert b["position"] == "bottom-center" and b["size"] == "M" and b["anchor"] == "start"
    assert v["placement"] == "smart" and v["target_pct"] == 0.9


def test_save_load_round_trip():
    ads.save_config({"enabled": True, "video": {"enabled": True, "placement": "end"}})
    cfg = ads.load_config()
    assert cfg["enabled"] and cfg["video"]["placement"] == "end"
    with open(ads.CONFIG_PATH, encoding="utf-8") as f:
        assert json.load(f) == cfg


def test_active_parts_need_master_switch_and_asset(tmp_path):
    cfg = ads.normalize_config({"enabled": True, "banner": {"enabled": True},
                                "video": {"enabled": True}})
    assert ads.active_parts(cfg) == {"banner": False, "video": False}
    open(ads.BANNER_PATH, "wb").close()
    assert ads.active_parts(cfg) == {"banner": True, "video": False}
    cfg["enabled"] = False
    assert ads.active_parts(cfg) == {"banner": False, "video": False}


# --- banner window ---------------------------------------------------------

def test_banner_window_from_start_and_end():
    b = ads.normalize_config({"banner": {"start": 3, "duration": 8}})["banner"]
    assert ads.banner_window(b, 30) == (3.0, 11.0)
    b = ads.normalize_config({"banner": {"anchor": "end", "start": 0, "duration": 5}})["banner"]
    assert ads.banner_window(b, 30) == (25.0, 30.0)


def test_banner_window_clamps_or_skips():
    b = ads.normalize_config({"banner": {"start": 25, "duration": 10}})["banner"]
    assert ads.banner_window(b, 30) == (25.0, 30.0)
    b = ads.normalize_config({"banner": {"start": 40, "duration": 10}})["banner"]
    assert ads.banner_window(b, 30) is None


# --- break point -----------------------------------------------------------

def test_choose_break_prefers_pause_near_target():
    gaps = [(5.0, 5.5), (14.0, 14.6), (22.0, 22.4)]
    assert ads.choose_break(30, gaps, [], 0.5) == pytest.approx(14.5)


def test_choose_break_prefers_pause_after_laugh():
    gaps = [(13.0, 13.5), (17.0, 19.0)]
    laughs = [(17.2, 18.6)]
    # 13.4 is closer to 15 than 18.9, but the post-laugh pause wins.
    assert ads.choose_break(30, gaps, laughs, 0.5) == pytest.approx(18.9)


def test_choose_break_ignores_edges_and_falls_back_to_target():
    assert ads.choose_break(30, [(1.0, 2.0), (28.0, 29.0)], [], 0.5) == 15.0


def test_break_point_placements_and_transcript_mapping():
    transcript = {
        "segments": [{"words": [
            {"word": "a", "start": 100.0, "end": 104.0},
            {"word": "b", "start": 106.0, "end": 110.0},
            {"word": "c", "start": 110.1, "end": 120.0},
        ]}],
        "audio_events": [{"text": "(laughter)", "start": 104.2, "end": 105.8}],
    }
    clip = {"start": 100.0, "end": 120.0}
    v = ads.normalize_config({"video": {"placement": "smart"}})["video"]
    assert ads.break_point(v, 20, transcript, clip, True) == pytest.approx(5.9)
    # After Smart Cut the transcript no longer maps → bare target.
    assert ads.break_point(v, 18, transcript, clip, False) == 9.0
    assert ads.break_point({**v, "placement": "start"}, 20, transcript, clip, True) == 0.0
    assert ads.break_point({**v, "placement": "end"}, 20, transcript, clip, True) == 20


# --- filter graph ----------------------------------------------------------

MAIN = {"width": 1080, "height": 1920, "fps": 30.0, "duration": 20.0,
        "has_audio": True, "sample_rate": 48000}
AD = {"width": 1920, "height": 1080, "fps": 25.0, "duration": 5.0,
      "has_audio": True, "sample_rate": 44100}


def test_filter_mid_roll_has_three_segments():
    filt, v, a = ads.build_insert_filter(MAIN, AD, 8.0)
    assert "concat=n=3:v=1:a=1" in filt and (v, a) == ("[vout]", "[aout]")
    assert "[0:v]split[mv0][mv1]" in filt and "[0:a]asplit[ma0][ma1]" in filt
    assert "scale=1080:1920:force_original_aspect_ratio=decrease" in filt


def test_filter_pre_roll_and_silent_ad():
    filt, _, _ = ads.build_insert_filter(MAIN, {**AD, "has_audio": False}, 0.0)
    assert "concat=n=2" in filt and "split[mv0]" not in filt
    assert "anullsrc" in filt and "[1:a]" not in filt


# --- compose wiring --------------------------------------------------------

def _touch(path):
    with open(path, "wb") as f:
        f.write(b"x")
    return path


def test_ads_run_last_even_with_no_toggles(tmp_path, monkeypatch):
    ads.save_config({"enabled": True, "video": {"enabled": True}})
    _touch(ads.VIDEO_PATH)
    seen = {}

    async def fake_ads(current_input, job_dir, clip_index, ad_cfg, ad_parts, metadata,
                       clip_info, *, timeline_intact, intermediate_files):
        seen.update(parts=ad_parts, intact=timeline_intact)
        out = _touch(os.path.join(job_dir, f"composed_adinsert_{clip_index}.mp4"))
        intermediate_files.append(out)
        return out

    monkeypatch.setattr(compose, "_apply_ads", fake_ads)
    base = _touch(str(tmp_path / "clip_0.mp4"))
    result = asyncio.run(compose.compose_layers(
        base_clip=base, job_dir=str(tmp_path), clip_index=0,
        metadata={"transcript": {}}, clip_info={"start": 0, "end": 30},
        toggles={}, hook_params={}, subtitle_params={}))
    assert result == "composed_clip_0.mp4"
    assert seen == {"parts": {"banner": False, "video": True}, "intact": True}


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_render_banner_and_mid_roll(tmp_path, monkeypatch):
    monkeypatch.setenv("CLIPPYME_X264_PRESET", "ultrafast")
    main = str(tmp_path / "main.mp4")
    ad = str(tmp_path / "ad.mp4")
    run = lambda *a: subprocess.run(["ffmpeg", "-v", "error", "-y", *a], check=True)
    run("-f", "lavfi", "-i", "testsrc2=s=360x640:r=30:d=6", "-f", "lavfi",
        "-i", "sine=f=440:d=6", "-shortest", "-c:v", "libx264", "-preset", "ultrafast",
        "-c:a", "aac", main)
    run("-f", "lavfi", "-i", "testsrc=s=640x360:r=25:d=2", "-c:v", "libx264",
        "-preset", "ultrafast", ad)  # silent ad
    run("-f", "lavfi", "-i", "color=c=yellow:s=300x100", "-frames:v", "1", ads.BANNER_PATH)

    b = ads.normalize_config({"banner": {"start": 1, "duration": 3}})["banner"]
    banner_out = str(tmp_path / "b.mp4")
    assert ads.add_timed_banner(main, ads.BANNER_PATH, banner_out, b)
    assert ads.probe_media(banner_out)["duration"] == pytest.approx(6.0, abs=0.1)

    out = str(tmp_path / "o.mp4")
    ads.insert_ad_video(banner_out, ad, out, 3.0)
    info = ads.probe_media(out)
    assert info["duration"] == pytest.approx(8.0, abs=0.15)
    assert (info["width"], info["height"], info["has_audio"]) == (360, 640, True)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_validate_ad_video_rejects_non_video(tmp_path):
    bad = tmp_path / "x.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(ValueError):
        ads.validate_ad_video(str(bad))
