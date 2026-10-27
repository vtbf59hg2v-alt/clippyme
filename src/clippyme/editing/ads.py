"""Advertising layer — timed sponsor banner and mid-roll ad insert.

Global, not per-clip: the settings live in ``data/ads.json`` and the assets
next to it, and every composed clip (download, publish, live monitor) gets
them, so a sponsor deal is set up once instead of clip by clip. Runs as the
very last compose layer because the ad insert changes the clip's length.

Two independent parts:

- **Timed banner** — a PNG shown for a fixed window (e.g. seconds 3–11, or
  the last 8 s), with an alpha fade. A PNG for the whole clip is the
  existing Logo layer.
- **Ad insert** — the clip pauses, the sponsor video plays with its own
  audio, then the clip resumes from the same frame. ``smart`` placement puts
  the break in a pause between phrases near the middle, preferring the moment
  right after an audience laugh, so it never cuts through a joke.
"""
from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import subprocess
import tempfile
import threading
from typing import Any

from clippyme.media.encode import ffmpeg_timeout, x264_video_args

logger = logging.getLogger("clippyme")

DATA_DIR = "data"
CONFIG_PATH = os.environ.get("CLIPPYME_ADS_CONFIG") or os.path.join(DATA_DIR, "ads.json")
BANNER_PATH = os.environ.get("CLIPPYME_AD_BANNER_PATH") or os.path.join(DATA_DIR, "ad_banner.png")
VIDEO_PATH = os.environ.get("CLIPPYME_AD_VIDEO_PATH") or os.path.join(DATA_DIR, "ad_video.mp4")

POSITIONS = ("top-left", "top-right", "top-center", "bottom-left",
             "bottom-right", "bottom-center", "center")
PLACEMENTS = ("smart", "start", "end")
SIZE_MAP = {"S": 0.25, "M": 0.45, "L": 0.7}

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": False,
    "banner": {
        "enabled": False,
        "anchor": "start",      # count `start` from the clip's start or end
        "start": 3.0,
        "duration": 8.0,
        "position": "bottom-center",
        "size": "M",
        "opacity": 1.0,
        "fade": 0.4,
    },
    "video": {
        "enabled": False,
        "placement": "smart",
        "target_pct": 0.5,      # smart: aim for this point of the clip
    },
}

MIN_GAP_SEC = 0.25              # shortest pause between phrases to break in
SMART_WINDOW = (0.25, 0.75)     # never break in the first/last quarter
LAUGH_BONUS = 0.35              # post-laugh pause: distance to target × this
_LOCK = threading.RLock()


# --- settings --------------------------------------------------------------

def _clamp(value, lo, hi, default):
    try:
        return max(lo, min(hi, float(value)))
    except (TypeError, ValueError):
        return default


def normalize_config(raw: dict | None) -> dict[str, Any]:
    """Merge user input over the defaults and clamp every field, so whatever
    the dashboard (or a hand-edited file) sends can't reach ffmpeg unchecked."""
    raw = raw if isinstance(raw, dict) else {}
    rb = raw.get("banner") if isinstance(raw.get("banner"), dict) else {}
    rv = raw.get("video") if isinstance(raw.get("video"), dict) else {}
    db, dv = DEFAULT_CONFIG["banner"], DEFAULT_CONFIG["video"]
    return {
        "enabled": bool(raw.get("enabled", False)),
        "banner": {
            "enabled": bool(rb.get("enabled", False)),
            "anchor": rb.get("anchor") if rb.get("anchor") in ("start", "end") else db["anchor"],
            "start": _clamp(rb.get("start", db["start"]), 0.0, 600.0, db["start"]),
            "duration": _clamp(rb.get("duration", db["duration"]), 0.5, 600.0, db["duration"]),
            "position": rb.get("position") if rb.get("position") in POSITIONS else db["position"],
            "size": rb.get("size") if rb.get("size") in SIZE_MAP else db["size"],
            "opacity": _clamp(rb.get("opacity", db["opacity"]), 0.1, 1.0, db["opacity"]),
            "fade": _clamp(rb.get("fade", db["fade"]), 0.0, 2.0, db["fade"]),
        },
        "video": {
            "enabled": bool(rv.get("enabled", False)),
            "placement": rv.get("placement") if rv.get("placement") in PLACEMENTS else dv["placement"],
            "target_pct": _clamp(rv.get("target_pct", dv["target_pct"]), 0.1, 0.9, dv["target_pct"]),
        },
    }


def load_config() -> dict[str, Any]:
    with _LOCK:
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                return normalize_config(json.load(f))
        except FileNotFoundError:
            return normalize_config({})
        except (OSError, ValueError) as exc:
            logger.warning("Could not read %s (%s) — ads disabled", CONFIG_PATH, exc)
            return normalize_config({})


def save_config(raw: dict) -> dict[str, Any]:
    cfg = normalize_config(raw)
    with _LOCK:
        os.makedirs(os.path.dirname(CONFIG_PATH) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CONFIG_PATH) or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
        os.replace(tmp, CONFIG_PATH)
    return cfg


def active_parts(cfg: dict | None = None) -> dict[str, bool]:
    """Which parts will actually render: enabled AND their asset exists."""
    cfg = cfg or load_config()
    on = bool(cfg.get("enabled"))
    return {
        "banner": on and cfg["banner"]["enabled"] and os.path.exists(BANNER_PATH),
        "video": on and cfg["video"]["enabled"] and os.path.exists(VIDEO_PATH),
    }


# --- probing ---------------------------------------------------------------

def probe_media(path: str) -> dict[str, Any]:
    """Width, height, fps, duration, audio presence and sample rate."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format",
         "-show_streams", path],
        check=True, capture_output=True, timeout=60,
    ).stdout
    info = json.loads(out or b"{}")
    streams = info.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    fps = 30.0
    if video and video.get("avg_frame_rate", "0/0") != "0/0":
        num, _, den = video["avg_frame_rate"].partition("/")
        with contextlib.suppress(ValueError, ZeroDivisionError):
            fps = float(num) / float(den or 1)
    return {
        "width": int(video["width"]) if video else 0,
        "height": int(video["height"]) if video else 0,
        "fps": fps if 1 <= fps <= 120 else 30.0,
        "duration": float(info.get("format", {}).get("duration") or 0.0),
        "has_audio": audio is not None,
        "sample_rate": int(audio.get("sample_rate") or 48000) if audio else 48000,
    }


def _run(cmd: list[str], what: str) -> None:
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=ffmpeg_timeout())
    except subprocess.CalledProcessError as exc:
        tail = (exc.stderr or b"").decode(errors="replace")[-600:]
        logger.error("❌ %s ffmpeg error: %s", what, tail)
        raise


AD_VIDEO_MAX_SEC = 180.0


def validate_ad_video(path: str) -> dict[str, Any]:
    """Reject anything ffprobe can't read as a video of sane size/length."""
    try:
        info = probe_media(path)
    except (subprocess.SubprocessError, ValueError, OSError) as exc:
        raise ValueError("not a readable video") from exc
    if not (16 <= info["width"] <= 7680 and 16 <= info["height"] <= 7680):
        raise ValueError("no usable video stream")
    if not 0.5 <= info["duration"] <= AD_VIDEO_MAX_SEC:
        raise ValueError(f"ad must be 0.5–{AD_VIDEO_MAX_SEC:.0f} s long")
    return info


def status() -> dict[str, Any]:
    """Settings + which assets are uploaded, for the dashboard."""
    cfg = load_config()
    out: dict[str, Any] = {**cfg, "banner_uploaded": os.path.exists(BANNER_PATH),
                           "video_uploaded": os.path.exists(VIDEO_PATH)}
    if out["video_uploaded"]:
        with contextlib.suppress(Exception):
            out["video_duration"] = round(probe_media(VIDEO_PATH)["duration"], 1)
    return out


# --- timed banner ----------------------------------------------------------

def banner_window(cfg_banner: dict, clip_duration: float) -> tuple[float, float] | None:
    """(start, end) seconds of the banner inside the clip, or None if it
    can't fit at all."""
    dur = float(cfg_banner["duration"])
    if cfg_banner.get("anchor") == "end":
        end = clip_duration - float(cfg_banner["start"])
        start = end - dur
    else:
        start = float(cfg_banner["start"])
        end = start + dur
    start, end = max(0.0, start), min(clip_duration, end)
    return (round(start, 3), round(end, 3)) if end - start >= 0.5 else None


def add_timed_banner(video_path: str, banner_path: str, output_path: str,
                     cfg_banner: dict) -> bool:
    """Overlay the sponsor PNG for the configured window. Returns False (and
    writes nothing) when the window doesn't fit the clip."""
    from clippyme.editing.logo import logo_filter_chain

    info = probe_media(video_path)
    window = banner_window(cfg_banner, info["duration"])
    if window is None:
        logger.warning("Ad banner window doesn't fit a %.1fs clip — skipped", info["duration"])
        return False
    start, end = window
    chain, x_expr, y_expr = logo_filter_chain(
        info["width"] or 1080, SIZE_MAP[cfg_banner["size"]], cfg_banner["opacity"],
        0.04, cfg_banner["position"])
    fade = min(float(cfg_banner["fade"]), (end - start) / 3)
    if fade > 0:
        chain += (f",fade=t=in:st={start}:d={fade:.3f}:alpha=1"
                  f",fade=t=out:st={end - fade:.3f}:d={fade:.3f}:alpha=1")
    filt = (f"[1:v]{chain}[ad];"
            f"[0:v][ad]overlay={x_expr}:{y_expr}:enable='between(t,{start},{end})':shortest=1")
    _run(["ffmpeg", "-y", "-i", video_path,
          "-loop", "1", "-t", f"{info['duration']:.3f}", "-i", banner_path,
          "-filter_complex", filt, "-map", "0:a?", "-c:a", "copy",
          *x264_video_args(), output_path], "Ad banner")
    logger.info("✅ Ad banner %.1f–%.1fs → %s", start, end, os.path.basename(output_path))
    return True


# --- ad insert -------------------------------------------------------------

def _word_gaps(transcript: dict, clip_start: float, clip_end: float) -> list[tuple[float, float]]:
    """Pauses between consecutive words inside the clip, clip-relative."""
    words = [w for seg in (transcript or {}).get("segments") or []
             for w in seg.get("words") or []
             if clip_start <= float(w["start"]) < clip_end]
    words.sort(key=lambda w: float(w["start"]))
    gaps = []
    for a, b in itertools.pairwise(words):
        g0, g1 = float(a["end"]), float(b["start"])
        if g1 - g0 >= MIN_GAP_SEC:
            gaps.append((g0 - clip_start, g1 - clip_start))
    return gaps


def choose_break(duration: float, gaps: list[tuple[float, float]],
                 laughs: list[tuple[float, float]], target_pct: float) -> float:
    """Pick the clip-relative second to cut to the ad.

    Candidate = just before the next phrase starts (so a laugh inside the
    pause plays out first). Closest to the target wins; a pause that contains
    the end of an audience laugh gets a bonus — that's where the room has
    just reset. No usable pause → the bare target.
    """
    target = duration * target_pct
    lo, hi = duration * SMART_WINDOW[0], duration * SMART_WINDOW[1]
    best, best_score = None, None
    for g0, g1 in gaps:
        t = max(g0, g1 - 0.1)
        if not lo <= t <= hi:
            continue
        score = abs(t - target)
        if any(g0 - 0.5 <= l_end <= g1 + 0.1 for _, l_end in laughs):
            score *= LAUGH_BONUS
        if best_score is None or score < best_score:
            best, best_score = t, score
    return round(best if best is not None else target, 3)


def break_point(cfg_video: dict, duration: float, transcript: dict | None,
                clip_info: dict, timeline_intact: bool) -> float:
    """Clip-relative insert time for the configured placement."""
    if cfg_video["placement"] == "start":
        return 0.0
    if cfg_video["placement"] == "end":
        return duration
    gaps: list[tuple[float, float]] = []
    laughs: list[tuple[float, float]] = []
    if timeline_intact and transcript:
        # Transcript times map 1:1 onto the clip only while no Smart Cut has
        # removed pauses; otherwise fall back to the target point.
        c0 = float(clip_info.get("start", 0) or 0)
        c1 = float(clip_info.get("end", 0) or 0) or c0 + duration
        gaps = _word_gaps(transcript, c0, c1)
        laughs = [(float(e["start"]) - c0, float(e["end"]) - c0)
                  for e in transcript.get("audio_events") or []
                  if c0 <= float(e["end"]) <= c1]
    return choose_break(duration, gaps, laughs, cfg_video["target_pct"])


def build_insert_filter(main: dict, ad: dict, at: float) -> tuple[str, str, str]:
    """filter_complex + its video/audio output labels for main[:at] → ad → main[at:].

    The ad is fitted into the clip's frame over a blurred, cropped copy of
    itself (no black bars), conformed to the clip's fps and audio format so
    the parts concatenate cleanly. A silent ad (or clip) gets a silent track.
    """
    w, h, fps, sr = main["width"], main["height"], round(main["fps"], 3), main["sample_rate"]
    norm_v = f"setsar=1,fps={fps},format=yuv420p,setpts=PTS-STARTPTS"
    norm_a = f"aresample={sr},aformat=sample_fmts=fltp:channel_layouts=stereo,asetpts=PTS-STARTPTS"
    silence = f"anullsrc=r={sr}:cl=stereo,atrim=0:{{dur:.3f}},"

    parts = [
        "[1:v]split[adbg][adfg]",
        f"[adbg]scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},boxblur=20:2[bg]",
        f"[adfg]scale={w}:{h}:force_original_aspect_ratio=decrease[fg]",
        f"[bg][fg]overlay=(W-w)/2:(H-h)/2,{norm_v}[va]",
        f"[1:a]{norm_a}[aa]" if ad["has_audio"]
        else f"{silence.format(dur=ad['duration'])}{norm_a}[aa]",
    ]

    before = at > 0.05
    after = at < main["duration"] - 0.05
    v0 = v1 = "[0:v]"
    a0 = a1 = "[0:a]"
    if before and after:
        # The clip feeds two trims → split its streams first.
        parts.append("[0:v]split[mv0][mv1]")
        v0, v1 = "[mv0]", "[mv1]"
        if main["has_audio"]:
            parts.append("[0:a]asplit[ma0][ma1]")
            a0, a1 = "[ma0]", "[ma1]"
    if not main["has_audio"]:
        a0 = a1 = silence.format(dur=main["duration"])

    segments = []
    if before:
        parts.append(f"{v0}trim=0:{at},{norm_v}[v0]")
        parts.append(f"{a0}atrim=0:{at},{norm_a}[a0]")
        segments.append("[v0][a0]")
    segments.append("[va][aa]")
    if after:
        parts.append(f"{v1}trim=start={at},{norm_v}[v1]")
        parts.append(f"{a1}atrim=start={at},{norm_a}[a1]")
        segments.append("[v1][a1]")
    parts.append(f"{''.join(segments)}concat=n={len(segments)}:v=1:a=1[vout][aout]")
    return ";".join(parts), "[vout]", "[aout]"


def insert_ad_video(video_path: str, ad_path: str, output_path: str, at: float) -> float:
    """Render main[:at] → ad → main[at:]. Returns the insert time used."""
    main = probe_media(video_path)
    ad = probe_media(ad_path)
    at = max(0.0, min(float(at), main["duration"]))
    filt, vout, aout = build_insert_filter(main, ad, at)
    _run(["ffmpeg", "-y", "-i", video_path, "-i", ad_path,
          "-filter_complex", filt, "-map", vout, "-map", aout,
          "-c:a", "aac", "-b:a", "192k", *x264_video_args(), output_path], "Ad insert")
    logger.info("✅ Ad insert (%.1fs ad) at %.2fs → %s",
                ad["duration"], at, os.path.basename(output_path))
    return at
