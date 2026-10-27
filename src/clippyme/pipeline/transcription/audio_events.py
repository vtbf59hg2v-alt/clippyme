"""Audience-reaction detection (laughter / applause) from the audio track.

The Gemini prompt treats ``(laughter)`` / ``(applause)`` markers in the
transcript as strong payoff evidence, but only ElevenLabs Scribe emits them.
Whisper-family providers (local Faster-Whisper, VoiceStudio, Deepgram) drop
non-speech sounds entirely — which loses the single best signal for studio
comedy, where the room tells you which line landed.

This module recovers that signal locally with plain signal processing, no
model download: a live-audience reaction is loud, noise-like (spectrally
flat, unlike voiced speech) and mostly falls in the gaps between words.
Detected events are returned in the same ``audio_events`` shape as the
ElevenLabs backend and woven into the transcript ``text`` the prompt reads.
The word stream (subtitles, Smart Cut) is never touched.

It is a heuristic: loud broadband sound with nobody talking (crowd noise,
engines, wind) can also trigger it. Tune or disable it per source with the
env vars below.

Env vars
~~~~~~~~
- ``CLIPPYME_AUDIENCE_EVENTS`` (default ``true``; ``false`` disables)
- ``CLIPPYME_AUDIENCE_FLATNESS`` (default 0.25; minimum spectral flatness
  of a reaction frame, 0–1 — raise it if speech is flagged)
- ``CLIPPYME_AUDIENCE_LOUDNESS_DB`` (default -6; how loud a reaction frame
  must be relative to the median speech level)
- ``CLIPPYME_AUDIENCE_MIN_SEC`` (default 0.8; shortest reaction kept)
"""
from __future__ import annotations

import logging
import os
import subprocess
from typing import Any

import numpy as np

logger = logging.getLogger("clippyme")

SAMPLE_RATE = 16000
WIN = 1024                     # 64 ms analysis window
HOP = 800                      # 50 ms hop
HOP_SEC = HOP / SAMPLE_RATE
BAND_HZ = (300.0, 6000.0)      # where crowd noise and speech both live
MERGE_GAP_SEC = 0.3            # join reaction runs split by a short dip
MAX_SPEECH_OVERLAP = 0.6       # reactions mostly sit between words
APPLAUSE_MIN_SEC = 2.5         # long and steady → applause
APPLAUSE_MAX_STD_DB = 3.0
LOUD_EXTRA_DB = 3.0            # this far above speech → "loud laughter"
_READ_BYTES = SAMPLE_RATE * 2 * 30  # decode 30 s of s16le at a time


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "") or default)
    except ValueError:
        return default


def enabled() -> bool:
    return (os.getenv("CLIPPYME_AUDIENCE_EVENTS") or "true").strip().lower() not in (
        "0", "false", "no", "off")


def frame_features(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame loudness (dBFS) and in-band spectral flatness (0–1)."""
    if samples.size < WIN:
        return np.zeros(0), np.zeros(0)
    n_frames = 1 + (samples.size - WIN) // HOP
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = samples[idx] * np.hanning(WIN)[None, :]

    rms = np.sqrt(np.mean(frames ** 2, axis=1))
    rms_db = 20.0 * np.log10(np.maximum(rms, 1e-6))

    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    freqs = np.fft.rfftfreq(WIN, 1.0 / SAMPLE_RATE)
    band = power[:, (freqs >= BAND_HZ[0]) & (freqs <= BAND_HZ[1])] + 1e-12
    flatness = np.exp(np.mean(np.log(band), axis=1)) / np.mean(band, axis=1)
    return rms_db, flatness


def decode_audio_features(media_path: str) -> tuple[np.ndarray, np.ndarray]:
    """Stream the audio through ffmpeg and compute frame features chunk by
    chunk, so a multi-hour episode never sits in memory as raw samples."""
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-i", media_path, "-vn",
           "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    rms_parts: list[np.ndarray] = []
    flat_parts: list[np.ndarray] = []
    carry = np.zeros(0, dtype=np.float32)
    try:
        assert proc.stdout is not None
        while True:
            chunk = proc.stdout.read(_READ_BYTES)
            if not chunk:
                break
            data = np.frombuffer(chunk[: len(chunk) // 2 * 2], dtype="<i2")
            buf = np.concatenate([carry, data.astype(np.float32) / 32768.0])
            if buf.size < WIN:
                carry = buf
                continue
            n_frames = 1 + (buf.size - WIN) // HOP
            used = (n_frames - 1) * HOP + WIN
            rms_db, flat = frame_features(buf[:used])
            rms_parts.append(rms_db)
            flat_parts.append(flat)
            carry = buf[n_frames * HOP:]
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()
    if proc.returncode not in (0, None):
        raise RuntimeError(f"ffmpeg exited with code {proc.returncode}")
    if not rms_parts:
        return np.zeros(0), np.zeros(0)
    return np.concatenate(rms_parts), np.concatenate(flat_parts)


def _speech_mask(n_frames: int, segments: list[dict[str, Any]]) -> np.ndarray:
    mask = np.zeros(n_frames, dtype=bool)
    for seg in segments or []:
        for w in seg.get("words") or []:
            a = int(float(w["start"]) / HOP_SEC)
            b = int(np.ceil(float(w["end"]) / HOP_SEC))
            mask[max(0, a):max(0, min(n_frames, b))] = True
    return mask


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame index pairs of consecutive True values."""
    if flags.size == 0:
        return []
    padded = np.concatenate([[False], flags, [False]]).astype(np.int8)
    edges = np.flatnonzero(np.diff(padded))
    return list(zip(edges[0::2].tolist(), edges[1::2].tolist()))


def detect_events(rms_db: np.ndarray, flatness: np.ndarray,
                  segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Find audience reactions in precomputed frame features."""
    n = rms_db.size
    if n == 0:
        return []
    speech = _speech_mask(n, segments)
    ref_db = float(np.median(rms_db[speech])) if speech.any() else float(np.median(rms_db))

    loud_db = ref_db + _env_float("CLIPPYME_AUDIENCE_LOUDNESS_DB", -6.0)
    flat_min = _env_float("CLIPPYME_AUDIENCE_FLATNESS", 0.25)
    min_frames = max(1, round(_env_float("CLIPPYME_AUDIENCE_MIN_SEC", 0.8) / HOP_SEC))

    # 5-frame median smoothing so a single voiced frame inside a laugh, or a
    # single noisy consonant inside speech, doesn't split or create a run.
    k = 2
    padded = np.pad(flatness, k, mode="edge")
    smooth_flat = np.median(np.lib.stride_tricks.sliding_window_view(padded, 2 * k + 1), axis=1)
    candidate = (rms_db >= loud_db) & (smooth_flat >= flat_min)

    # Close short gaps between runs.
    gap = round(MERGE_GAP_SEC / HOP_SEC)
    runs = _runs(candidate)
    merged: list[list[int]] = []
    for a, b in runs:
        if merged and a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    events: list[dict[str, Any]] = []
    for a, b in merged:
        if b - a < min_frames:
            continue
        if speech[a:b].mean() > MAX_SPEECH_OVERLAP:
            continue
        seg_db = rms_db[a:b]
        dur = (b - a) * HOP_SEC
        if dur >= APPLAUSE_MIN_SEC and float(np.std(seg_db)) <= APPLAUSE_MAX_STD_DB:
            label = "(applause)"
        elif float(np.mean(seg_db)) >= ref_db + LOUD_EXTRA_DB:
            label = "(loud laughter)"
        else:
            label = "(laughter)"
        events.append({
            "text": label,
            "start": round(a * HOP_SEC, 2),
            "end": round(b * HOP_SEC + (WIN - HOP) / SAMPLE_RATE, 2),
        })
    return events


def weave_text(segments: list[dict[str, Any]], events: list[dict[str, Any]]) -> str:
    """Transcript string with event markers placed by timestamp."""
    tokens = [(float(w["start"]), str(w["word"]).strip())
              for seg in segments for w in seg.get("words") or []]
    tokens += [(float(e["start"]), e["text"]) for e in events]
    tokens.sort(key=lambda t: t[0])
    return " ".join(tok for _, tok in tokens if tok)


def annotate_audio_events(media_path: str, result: dict[str, Any]) -> dict[str, Any]:
    """Add detected audience reactions to a transcription result in place.

    No-op when disabled, when the provider already tagged events (ElevenLabs),
    or when the result has no word timings to weave against. Never raises:
    detection is a bonus signal, not a pipeline dependency.
    """
    if not enabled() or not result or result.get("audio_events"):
        return result
    segments = result.get("segments") or []
    if not any(seg.get("words") for seg in segments):
        return result
    try:
        rms_db, flatness = decode_audio_features(media_path)
        events = detect_events(rms_db, flatness, segments)
    except Exception as exc:  # noqa: BLE001 — best-effort enrichment
        logger.warning("Audience-reaction detection failed (%s) — skipping", exc)
        return result

    result["audio_events"] = events
    if events:
        result["text"] = weave_text(segments, events) or result.get("text", "")
    counts: dict[str, int] = {}
    for e in events:
        counts[e["text"]] = counts.get(e["text"], 0) + 1
    summary = ", ".join(f"{k} ×{v}" for k, v in sorted(counts.items())) or "none"
    print(f"   😂 Audience reactions detected: {summary}")
    return result
