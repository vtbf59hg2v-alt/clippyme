"""Unit tests for clippyme.pipeline.transcription.audio_events.

Synthetic audio: a voiced "speaker" (harmonic tone, syllable-rate amplitude
modulation) with word timings, a fluctuating noise burst as laughter and a
steady noise block as applause. Covers detection, labelling, the
speech-overlap guard, text weaving and the ffmpeg decode path.
"""
import shutil
import subprocess

import numpy as np
import pytest

import clippyme.pipeline.transcription.audio_events as ae

SR = ae.SAMPLE_RATE
rng = np.random.default_rng(7)


def _voice(sec, amp=0.2):
    t = np.arange(int(sec * SR)) / SR
    tone = sum(np.sin(2 * np.pi * 140 * h * t) / h for h in range(1, 12))
    env = 0.6 + 0.4 * np.sin(2 * np.pi * 4 * t)
    return amp * tone / 3 * env


def _noise(sec, amp, mod_hz=None):
    n = rng.standard_normal(int(sec * SR))
    if mod_hz:
        t = np.arange(n.size) / SR
        n *= 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * mod_hz * t))
    return amp * n / 4


def _words(start, end, step=0.4):
    return [{"word": f"w{i}", "start": round(s, 2), "end": round(min(s + step, end), 2)}
            for i, s in enumerate(np.arange(start, end, step))]


def _scene(laugh_amp=0.15):
    """0–4 s speech, 4–6 s laughter, 6–9 s speech, 9–13 s applause, 13–15 s speech."""
    audio = np.concatenate([
        _voice(4), _noise(2, laugh_amp, mod_hz=5), _voice(3), _noise(4, 0.2), _voice(2),
    ]) + _noise(15, 0.002)
    segments = [{"text": "a", "start": 0, "end": 4, "words": _words(0, 4)},
                {"text": "b", "start": 6, "end": 9, "words": _words(6, 9)},
                {"text": "c", "start": 13, "end": 15, "words": _words(13, 15)}]
    return audio.astype(np.float32), segments


def test_detects_laughter_and_applause_between_words():
    audio, segments = _scene()
    events = ae.detect_events(*ae.frame_features(audio), segments)
    assert [e["text"] for e in events] == ["(laughter)", "(applause)"]
    assert events[0]["start"] == pytest.approx(4.0, abs=0.3)
    assert events[0]["end"] == pytest.approx(6.0, abs=0.3)
    assert events[1]["start"] == pytest.approx(9.0, abs=0.3)
    assert events[1]["end"] == pytest.approx(13.0, abs=0.3)


def test_laughter_well_above_speech_is_loud():
    audio, segments = _scene(laugh_amp=0.4)
    events = ae.detect_events(*ae.frame_features(audio), segments)
    assert events[0]["text"] == "(loud laughter)"


def test_no_events_in_plain_speech():
    audio = _voice(10).astype(np.float32)
    events = ae.detect_events(*ae.frame_features(audio),
                              [{"text": "x", "start": 0, "end": 10, "words": _words(0, 10)}])
    assert events == []


def test_noise_under_words_is_not_a_reaction():
    """Loud broadband sound while someone talks (engine, music) is skipped."""
    audio = (_voice(6) + _noise(6, 0.3, mod_hz=5)).astype(np.float32)
    events = ae.detect_events(*ae.frame_features(audio),
                              [{"text": "x", "start": 0, "end": 6, "words": _words(0, 6)}])
    assert events == []


def test_quiet_room_tone_is_ignored():
    audio = np.concatenate([_voice(3), _noise(3, 0.01), _voice(3)]).astype(np.float32)
    segs = [{"text": "a", "start": 0, "end": 3, "words": _words(0, 3)},
            {"text": "b", "start": 6, "end": 9, "words": _words(6, 9)}]
    assert ae.detect_events(*ae.frame_features(audio), segs) == []


def test_weave_text_orders_markers_by_time():
    segments = [{"words": [{"word": " Шутка", "start": 0.0, "end": 0.5},
                           {"word": " дальше", "start": 3.0, "end": 3.5}]}]
    events = [{"text": "(laughter)", "start": 1.0, "end": 2.5}]
    assert ae.weave_text(segments, events) == "Шутка (laughter) дальше"


def test_annotate_skips_when_provider_tagged_events(monkeypatch):
    result = {"text": "t", "segments": [{"words": [{"word": "a", "start": 0, "end": 1}]}],
              "audio_events": [{"text": "(laughter)", "start": 1, "end": 2}]}
    monkeypatch.setattr(ae, "decode_audio_features",
                        lambda p: pytest.fail("must not decode"))
    assert ae.annotate_audio_events("x.mp4", result)["text"] == "t"


def test_annotate_disabled_by_env(monkeypatch):
    monkeypatch.setenv("CLIPPYME_AUDIENCE_EVENTS", "false")
    result = {"text": "t", "segments": [{"words": [{"word": "a", "start": 0, "end": 1}]}]}
    assert ae.annotate_audio_events("x.mp4", result) == {"text": "t", "segments": result["segments"]}


def test_annotate_never_raises(monkeypatch):
    def boom(path):
        raise RuntimeError("ffmpeg missing")
    monkeypatch.setattr(ae, "decode_audio_features", boom)
    result = {"text": "t", "segments": [{"words": [{"word": "a", "start": 0, "end": 1}]}]}
    assert ae.annotate_audio_events("x.mp4", result)["text"] == "t"


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_annotate_end_to_end_through_ffmpeg(tmp_path):
    audio, segments = _scene()
    raw = tmp_path / "scene.raw"
    (np.clip(audio, -1, 1) * 32767).astype("<i2").tofile(raw)
    wav = tmp_path / "scene.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1",
                    "-i", str(raw), str(wav)], check=True)
    result = {"text": "orig", "segments": segments, "language": "ru"}
    out = ae.annotate_audio_events(str(wav), result)
    assert [e["text"] for e in out["audio_events"]] == ["(laughter)", "(applause)"]
    assert "(laughter)" in out["text"] and "(applause)" in out["text"]
    assert out["text"].index("(laughter)") < out["text"].index("(applause)")
