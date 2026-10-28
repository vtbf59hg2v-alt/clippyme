"""Unit tests for clippyme.pipeline.transcription.voicestudio_transcribe.

Mocked network only (no running VoiceStudio): verbose_json parsing for both
word layouts, form shaping, retries and the failure modes that must raise so
the dispatcher falls back to Faster-Whisper.
"""
import itertools
from pathlib import Path

import pytest
import requests

import clippyme.pipeline.transcription.voicestudio_transcribe as vs

VERBOSE = {
    "task": "transcribe",
    "language": "ru",
    "duration": 3.2,
    "text": "Добрый вечер, дорогие друзья! Начинаем.",
    "segments": [
        {"start": 0.0, "end": 1.66, "text": " Добрый вечер, дорогие друзья!",
         "avg_logprob": 0, "no_speech_prob": 0},
        {"start": 2.0, "end": 3.2, "text": " Начинаем."},
    ],
    "words": [
        {"word": "Добрый", "start": 0.0, "end": 0.36},
        {"word": "вечер,", "start": 0.36, "end": 0.8},
        {"word": "дорогие", "start": 0.8, "end": 1.4},
        {"word": "друзья!", "start": 1.4, "end": 1.66},
        {"word": "Начинаем.", "start": 2.0, "end": 3.2},
    ],
}


class _Resp:
    def __init__(self, status=200, payload=None, text="", headers=None):
        self.status_code = status
        self._payload = payload
        self.text = text
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


@pytest.fixture
def audio(tmp_path):
    p = tmp_path / "a.flac"
    p.write_bytes(b"fake-audio")
    return str(p)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in ("VOICESTUDIO_BASE_URL", "VOICESTUDIO_API_KEY", "VOICESTUDIO_LANGUAGE",
                 "CLIPPYME_LANGUAGE", "VOICESTUDIO_MAX_RETRIES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(vs.time, "sleep", lambda s: None)


def test_parse_assigns_flat_words_to_segments():
    segs = vs._parse_payload(VERBOSE)
    assert [s["text"] for s in segs] == ["Добрый вечер, дорогие друзья!", "Начинаем."]
    assert [w["word"] for w in segs[0]["words"]] == ["Добрый", "вечер,", "дорогие", "друзья!"]
    assert [w["word"] for w in segs[1]["words"]] == ["Начинаем."]
    assert segs[0]["words"][0] == {"word": "Добрый", "start": 0.0, "end": 0.36, "probability": 1.0}


def test_parse_nested_words_layout():
    payload = {"segments": [{"start": 0, "end": 1, "text": "Привет",
                             "words": [{"word": " Привет", "start": 0, "end": 1,
                                        "probability": 0.7}]}]}
    segs = vs._parse_payload(payload)
    assert segs[0]["words"] == [{"word": "Привет", "start": 0.0, "end": 1.0, "probability": 0.7}]


def test_parse_words_without_segments_chunks_sentences():
    segs = vs._parse_payload({"words": VERBOSE["words"]})
    assert [s["text"] for s in segs] == ["Добрый вечер, дорогие друзья!", "Начинаем."]


def test_form_language_falls_back_to_job_language(monkeypatch):
    monkeypatch.setenv("CLIPPYME_LANGUAGE", "ru")
    form = vs._build_form("whisper-1", vs._language())
    assert ("language", "ru") in form
    assert ("timestamp_granularities[]", "word") in form
    assert ("response_format", "verbose_json") in form
    monkeypatch.setenv("VOICESTUDIO_LANGUAGE", "multi")
    assert vs._language() == ""


def test_transcribe_full_shape(monkeypatch, audio):
    calls = {}

    def fake_post(url, headers, data, files, timeout):
        calls["url"] = url
        calls["headers"] = headers
        return _Resp(payload=VERBOSE)

    monkeypatch.setenv("VOICESTUDIO_BASE_URL", "http://127.0.0.1:3900/v1/")
    monkeypatch.setenv("VOICESTUDIO_API_KEY", "k")
    monkeypatch.setattr(vs.requests, "post", fake_post)
    out = vs.transcribe_with_voicestudio(audio)
    assert calls["url"] == "http://127.0.0.1:3900/v1/audio/transcriptions"
    assert calls["headers"]["Authorization"] == "Bearer k"
    assert out["language"] == "ru"
    assert out["text"] == VERBOSE["text"]
    assert len(out["segments"]) == 2


def test_retries_then_succeeds(monkeypatch, audio):
    responses = iter([_Resp(503, text="busy"), _Resp(payload=VERBOSE)])
    monkeypatch.setattr(vs.requests, "post", lambda *a, **k: next(responses))
    assert len(vs.transcribe_with_voicestudio(audio)["segments"]) == 2


def test_unreachable_server_raises(monkeypatch, audio):
    def boom(*a, **k):
        raise requests.ConnectionError("refused")
    monkeypatch.setattr(vs.requests, "post", boom)
    with pytest.raises(vs.VoiceStudioError, match="Cannot reach"):
        vs.transcribe_with_voicestudio(audio)


def test_client_error_raises_without_retry(monkeypatch, audio):
    n = {"calls": 0}

    def post(*a, **k):
        n["calls"] += 1
        return _Resp(400, text="bad\x1b[31m request")
    monkeypatch.setattr(vs.requests, "post", post)
    with pytest.raises(vs.VoiceStudioError, match="HTTP 400"):
        vs.transcribe_with_voicestudio(audio)
    assert n["calls"] == 1


def test_missing_word_timestamps_raises(monkeypatch, audio):
    payload = {"text": "x", "segments": [{"start": 0, "end": 1, "text": "x"}]}
    monkeypatch.setattr(vs.requests, "post", lambda *a, **k: _Resp(payload=payload))
    with pytest.raises(vs.VoiceStudioError, match="word timestamps"):
        vs.transcribe_with_voicestudio(audio)


def test_plan_cuts_moves_boundaries_into_pauses():
    assert vs.plan_cuts(700, 600) == []           # short enough for one request
    assert vs.plan_cuts(3000, 0) == []            # chunking disabled
    assert vs.plan_cuts(1800, 600) == [600, 1200]
    hop = 0.05
    rms = [-20.0] * int(1800 / hop)
    rms[int(607 / hop)] = -60.0                   # a pause 7 s after the boundary
    cuts = vs.plan_cuts(1800, 600, rms, hop)
    assert cuts[0] == 607.0 and cuts[1] == pytest.approx(1207.0, abs=20)


def test_long_file_is_sent_in_chunks_with_shifted_times(monkeypatch, audio):
    monkeypatch.setattr(vs, "_chunk_cuts", lambda p: [600.0, 1200.0])
    monkeypatch.setattr(vs, "_extract_chunk", lambda src, a, b, dst: Path(dst).write_bytes(b"x"))
    calls = []

    def post(url, headers, data, files, timeout):
        calls.append(1)
        return _Resp(200, VERBOSE)

    monkeypatch.setattr(vs.requests, "post", post)
    out = vs.transcribe_with_voicestudio(audio)
    assert len(calls) == 3
    starts = [s["start"] for s in out["segments"]]
    assert starts == [0.0, 2.0, 600.0, 602.0, 1200.0, 1202.0]
    assert out["segments"][-1]["words"][-1]["end"] == pytest.approx(1203.2)
    assert out["language"] == "ru"
    assert out["text"].count("Начинаем.") == 3


def test_plan_cuts_always_advances_past_an_early_pause():
    hop = 0.05
    rms = [-20.0] * int(30 / hop)
    for i in range(int(10 / hop), int(12 / hop)):
        rms[i] = -60.0  # one long pause the search could keep snapping back to
    cuts = vs.plan_cuts(30, 8, rms, hop)
    assert cuts == sorted(cuts) and len(cuts) == len(set(cuts)) and len(cuts) <= 4
    assert all(b - a >= 4 for a, b in itertools.pairwise(cuts))
