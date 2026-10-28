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
- ``VOICESTUDIO_MAX_RETRIES`` (default 4, per request)
- ``VOICESTUDIO_CHUNK_SEC`` (default 300; ``0`` sends the file whole)

Long files are sent in chunks of about ``VOICESTUDIO_CHUNK_SEC`` seconds, cut
at the quietest moment near each boundary so no word is split. One request
per chunk keeps each call short — a three-hour show in one synchronous
request can outlive a gateway timeout (HTTP 504) — and a failure only
retries that chunk.
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
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
DEFAULT_MAX_RETRIES = 4
DEFAULT_CHUNK_SEC = 300  # VoiceStudio gives one request at most 300 s by default
CUT_SEARCH_SEC = 20.0  # look this far either side of a boundary for a pause


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
            # Usually the app is closed or still loading its model: give the
            # user time to open it rather than giving up after a few seconds.
            wait = 15.0 * (attempt + 1)
            print(f"   ⚠️  Can't reach VoiceStudio ({exc}) — is the app open? "
                  f"Retrying in {wait:.0f}s…")
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


def _probe_duration(path: str) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            check=False, capture_output=True, text=True, timeout=60)
        return float(out.stdout.strip() or 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def plan_cuts(duration: float, chunk_sec: float, rms_db=None, hop_sec: float = 0.05) -> list[float]:
    """Chunk boundaries in seconds (excluding 0 and the end). With per-frame
    loudness, each boundary moves to the quietest frame within
    ``CUT_SEARCH_SEC`` so the cut lands in a pause, not mid-word."""
    if chunk_sec <= 0 or duration <= chunk_sec * 1.2:
        return []
    # The search radius stays under a quarter chunk, so every cut lands at
    # least half a chunk after the previous one and the loop always advances.
    search = min(CUT_SEARCH_SEC, chunk_sec / 4)
    cuts: list[float] = []
    t = chunk_sec
    while t < duration - chunk_sec * 0.2:
        cut = t
        if rms_db is not None and len(rms_db):
            lo = max(0, int((t - search) / hop_sec))
            hi = min(len(rms_db), int((t + search) / hop_sec) + 1)
            if hi > lo:
                window = rms_db[lo:hi]
                cut = (lo + int(min(range(len(window)), key=window.__getitem__))) * hop_sec
        cuts.append(round(cut, 2))
        t = cut + chunk_sec
    return cuts


def _extract_chunk(src: str, start: float, end: float | None, dst: str) -> None:
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", src]
    if end is not None:
        cmd += ["-t", f"{end - start:.3f}"]
    cmd += ["-vn", "-ac", "1", "-ar", "16000", "-c:a", "flac", dst]
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0 or not os.path.exists(dst):
        raise VoiceStudioError(f"ffmpeg could not cut chunk at {start:.0f}s: "
                               f"{(proc.stderr or '').strip()[-300:]}")


def _shift(segments: list[dict[str, Any]], offset: float) -> list[dict[str, Any]]:
    for seg in segments:
        seg["start"] = round(seg["start"] + offset, 3)
        seg["end"] = round(seg["end"] + offset, 3)
        for w in seg["words"]:
            w["start"] = round(w["start"] + offset, 3)
            w["end"] = round(w["end"] + offset, 3)
    return segments


def _chunk_cuts(audio_path: str) -> list[float]:
    try:
        chunk_sec = float(os.getenv("VOICESTUDIO_CHUNK_SEC", str(DEFAULT_CHUNK_SEC)) or 0)
    except ValueError:
        chunk_sec = DEFAULT_CHUNK_SEC
    duration = _probe_duration(audio_path) if chunk_sec > 0 else 0.0
    if not plan_cuts(duration, chunk_sec):
        return []
    rms_db = None
    try:
        from clippyme.pipeline.transcription import audio_events
        rms_db, _ = audio_events.decode_audio_features(audio_path)
        hop = audio_events.HOP_SEC
    except Exception as exc:  # noqa: BLE001 — fall back to fixed-length cuts
        logger.warning("VoiceStudio chunking: loudness scan failed (%s); fixed cuts", exc)
        hop = 0.05
    return plan_cuts(duration, chunk_sec, rms_db, hop)


def _transcribe_one(path: str, url: str, headers: dict[str, str], form: list[tuple[str, str]],
                    timeout: float, max_retries: int) -> dict[str, Any]:
    response = _post_with_retries(url, headers, form, path, timeout, max_retries)
    try:
        payload = response.json()
    except ValueError as exc:
        raise VoiceStudioError(f"VoiceStudio returned non-JSON body: {exc}") from exc
    if not isinstance(payload, dict):
        raise VoiceStudioError("VoiceStudio returned an unexpected JSON body")
    return payload


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
    form = _build_form(model, language)
    cuts = _chunk_cuts(audio_path)
    if not cuts:
        payload = _transcribe_one(audio_path, url, headers, form, timeout, max_retries)
        segments = _parse_payload(payload)
    else:
        bounds = [0.0, *cuts]
        print(f"   ✂️  Long file — sending {len(bounds)} chunks of ~"
              f"{float(os.getenv('VOICESTUDIO_CHUNK_SEC') or DEFAULT_CHUNK_SEC) / 60:.0f} min")
        segments, texts, payload = [], [], {}
        with tempfile.TemporaryDirectory(prefix="voicestudio-") as tmp:
            for i, start in enumerate(bounds):
                end = bounds[i + 1] if i + 1 < len(bounds) else None
                chunk = os.path.join(tmp, f"chunk{i:03d}.flac")
                _extract_chunk(audio_path, start, end, chunk)
                part = _transcribe_one(chunk, url, headers, form, timeout, max_retries)
                payload = payload or part  # first chunk carries the language
                segments += _shift(_parse_payload(part), start)
                texts.append((part.get("text") or "").strip())
                print(f"   🎙️  chunk {i + 1}/{len(bounds)} done "
                      f"({time.monotonic() - t0:.0f}s elapsed)")
        payload = {"language": payload.get("language"), "text": " ".join(t for t in texts if t)}
    elapsed = time.monotonic() - t0

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
