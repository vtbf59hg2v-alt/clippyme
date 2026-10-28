"""VoiceStudio transcription backend for ClippyMe.

Fourth transcription provider, alongside Faster-Whisper (in-process), Deepgram
and ElevenLabs. VoiceStudio is a desktop app that serves an OpenAI-compatible
``POST /v1/audio/transcriptions`` endpoint on the host (default port 3900).
It runs Whisper on the host's own accelerator (e.g. MLX on Apple Silicon),
which a Docker container on macOS cannot reach — so offloading ASR there is
much faster than the in-container CPU Whisper, and it costs nothing.

Returns the **same dict shape** as ``main.transcribe_video``::

    {
        "text": str,
        "segments": [
            {"text": str, "start": float, "end": float,
             "words": [{"word": str, "start": float, "end": float,
                        "probability": float}, ...]},
            ...
        ],
        "language": str,
    }

The OpenAI ``verbose_json`` response carries ``segments`` and a flat
top-level ``words`` list (some servers nest ``words`` per segment instead);
both layouts are handled. When the server returns words but no segments, the
words are chunked into sentences the same way as the ElevenLabs path.

Env vars
~~~~~~~~
- ``VOICESTUDIO_BASE_URL`` (default ``http://host.docker.internal:3900/v1``;
  use ``http://127.0.0.1:3900/v1`` when ClippyMe runs outside Docker)
- ``VOICESTUDIO_API_KEY`` (optional; sent as a Bearer token)
- ``VOICESTUDIO_MODEL`` (default ``whisper-1``; the server picks its engine)
- ``VOICESTUDIO_LANGUAGE`` (ISO code; falls back to ``CLIPPYME_LANGUAGE``;
  blank / ``multi`` → auto-detect)
- ``VOICESTUDIO_HTTP_TIMEOUT`` (default 14400 s — a long episode on a CPU
  engine is slow)
- ``VOICESTUDIO_MAX_RETRIES`` (default 2)
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import requests

from clippyme.pipeline.transcription.elevenlabs_transcribe import (
    _compute_backoff,
    _segments_from_words,
    _should_retry,
)

logger = logging.getLogger("clippyme")

DEFAULT_BASE_URL = "http://host.docker.internal:3900/v1"
DEFAULT_MODEL = "whisper-1"
DEFAULT_TIMEOUT = 14400  # a 3-hour show on a CPU engine needs well over 2 h
DEFAULT_MAX_RETRIES = 2


class VoiceStudioError(RuntimeError):
    """Raised when VoiceStudio transcription fails unrecoverably."""


def _endpoint() -> str:
    base = (os.getenv("VOICESTUDIO_BASE_URL") or DEFAULT_BASE_URL).strip().rstrip("/")
    return f"{base}/audio/transcriptions"


def _language() -> str:
    lang = (os.getenv("VOICESTUDIO_LANGUAGE") or os.getenv("CLIPPYME_LANGUAGE") or "").strip()
    return "" if lang.lower() == "multi" else lang


def _build_form(model: str, language: str) -> list[tuple[str, str]]:
    """Multipart text fields. A list of tuples so the repeated
    ``timestamp_granularities[]`` key is sent twice."""
    form = [
        ("model", model),
        ("response_format", "verbose_json"),
        ("timestamp_granularities[]", "word"),
        ("timestamp_granularities[]", "segment"),
    ]
    if language:
        form.append(("language", language))
    return form


def _post_with_retries(url: str, headers: dict[str, str], form: list[tuple[str, str]],
                       file_path: str, timeout: float, max_retries: int) -> requests.Response:
    attempt = 0
    while True:
        try:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
                response = requests.post(url, headers=headers, data=form, files=files,
                                         timeout=timeout)
        except requests.RequestException as exc:
            if attempt >= max_retries:
                raise VoiceStudioError(
                    f"Cannot reach VoiceStudio at {url} after {attempt + 1} attempts: {exc}"
                ) from exc
            wait = _compute_backoff(attempt + 1, None)
            print(f"   ⚠️  VoiceStudio network error ({exc}); retrying in {wait:.1f}s…")
            time.sleep(wait)
            attempt += 1
            continue

        if response.status_code == 200:
            return response
        if _should_retry(response.status_code) and attempt < max_retries:
            wait = _compute_backoff(attempt + 1, response.headers.get("Retry-After"))
            print(f"   ⚠️  VoiceStudio HTTP {response.status_code}; retrying in {wait:.1f}s…")
            time.sleep(wait)
            attempt += 1
            continue
        snippet = re.sub(r"[\x00-\x1f\x7f]", " ", (response.text or "")[:400])
        raise VoiceStudioError(f"VoiceStudio returned HTTP {response.status_code}: {snippet}")


def _to_word(raw: dict[str, Any]) -> dict[str, Any] | None:
    text = str(raw.get("word") or raw.get("text") or "").strip()
    if not text:
        return None
    try:
        start = float(raw.get("start", 0.0) or 0.0)
        end = float(raw.get("end", start) or start)
    except (TypeError, ValueError):
        return None
    prob = raw.get("probability")
    try:
        prob = max(0.0, min(1.0, float(prob))) if prob is not None else 1.0
    except (TypeError, ValueError):
        prob = 1.0
    return {"word": text, "start": start, "end": max(start, end), "probability": prob}


def _parse_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Map a verbose_json body onto pipeline segments with word timings."""
    raw_segments = payload.get("segments") or []
    flat_words = [w for w in (_to_word(r) for r in payload.get("words") or []) if w]

    if not raw_segments:
        return _segments_from_words(flat_words)

    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        try:
            start = float(raw.get("start", 0.0) or 0.0)
            end = float(raw.get("end", start) or start)
        except (TypeError, ValueError):
            continue
        nested = [w for w in (_to_word(r) for r in raw.get("words") or []) if w]
        segments.append({
            "text": str(raw.get("text") or "").strip(),
            "start": start,
            "end": end,
            "words": nested,
        })

    if flat_words and not any(seg["words"] for seg in segments):
        # Assign each flat word to the segment that contains its midpoint;
        # a word in a gap goes to the nearest preceding segment.
        idx = 0
        for w in flat_words:
            mid = (w["start"] + w["end"]) / 2
            while idx + 1 < len(segments) and mid >= segments[idx + 1]["start"]:
                idx += 1
            segments[idx]["words"].append(w)

    for seg in segments:
        if not seg["text"] and seg["words"]:
            seg["text"] = " ".join(w["word"] for w in seg["words"])
    return [seg for seg in segments if seg["text"]]


def transcribe_with_voicestudio(audio_path: str) -> dict[str, Any]:
    """Transcribe a local audio/video file through VoiceStudio.

    Raises :class:`VoiceStudioError` on failure so the caller can fall back
    to in-process Faster-Whisper.
    """
    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise VoiceStudioError(f"File missing or empty: {audio_path}")

    url = _endpoint()
    model = (os.getenv("VOICESTUDIO_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL
    language = _language()
    timeout = float(os.getenv("VOICESTUDIO_HTTP_TIMEOUT", str(DEFAULT_TIMEOUT)))
    max_retries = int(os.getenv("VOICESTUDIO_MAX_RETRIES", str(DEFAULT_MAX_RETRIES)))
    headers = {"User-Agent": "ClippyMe/1.0"}
    api_key = (os.getenv("VOICESTUDIO_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    size_mb = os.path.getsize(audio_path) / 1024 / 1024
    print(f"🎙️  Transcribing with VoiceStudio [{url}, lang={language or 'auto'}] "
          f"— uploading {size_mb:.1f} MB…")

    t0 = time.monotonic()
    response = _post_with_retries(url, headers, _build_form(model, language),
                                  audio_path, timeout, max_retries)
    elapsed = time.monotonic() - t0

    try:
        payload = response.json()
    except ValueError as exc:
        raise VoiceStudioError(f"VoiceStudio returned non-JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise VoiceStudioError("VoiceStudio returned an unexpected JSON body")

    segments = _parse_payload(payload)
    if not segments:
        raise VoiceStudioError("VoiceStudio returned no segments")
    n_words = sum(len(seg["words"]) for seg in segments)
    if n_words == 0:
        # Without word timings subtitles and cut boundaries break — let the
        # caller fall back to Whisper instead of producing degraded clips.
        raise VoiceStudioError("VoiceStudio returned no word timestamps")

    full_text = (payload.get("text") or "").strip() or " ".join(s["text"] for s in segments)
    language_detected = str(payload.get("language") or language or "unknown")
    duration = float(payload.get("duration") or segments[-1]["end"])
    speed = f" ({duration / elapsed:.1f}× realtime)" if elapsed > 0 and duration else ""
    print(f"   ✅ VoiceStudio OK — {len(segments)} segments, {n_words} words, "
          f"audio={duration:.1f}s, wall={elapsed:.1f}s{speed}, lang='{language_detected}'")
    logger.info("VoiceStudio transcription done: audio_s=%.1f wall_s=%.1f segments=%d words=%d",
                duration, elapsed, len(segments), n_words)

    return {"text": full_text, "segments": segments, "language": language_detected}
