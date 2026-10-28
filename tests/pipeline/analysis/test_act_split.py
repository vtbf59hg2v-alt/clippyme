"""Two-pass comedy selection: acts first, then clips inside each act."""
import json

from clippyme.pipeline.analysis import act_split


def _show(n_segments=60, step=10.0):
    return {"text": "", "segments": [
        {"start": i * step, "end": i * step + step, "text": f"фраза {i}", "words": [
            {"word": "фраза", "start": i * step, "end": i * step + 1}]}
        for i in range(n_segments)
    ]}


class _Resp:
    def __init__(self, text):
        self.text = text
        self.usage_metadata = None


def test_parse_acts_sorts_clamps_and_merges_slivers():
    text = "thinking…\n### JSON ###\n" + json.dumps({"acts": [
        {"start": 300, "end": 700, "kind": "Sketch", "title": "Воля и Синяков"},
        {"start": 0, "end": 300, "kind": "host", "title": "Открытие"},
        {"start": 700, "end": 720, "kind": "sketch", "title": "обрывок"},
        {"start": 720, "end": 900, "kind": "music", "title": "песня"},
        {"start": "x", "end": 5},
    ]})
    acts = act_split.parse_acts(text, video_duration=850)
    assert [(a["start"], a["end"], a["kind"]) for a in acts] == [
        (0.0, 300.0, "host"), (300.0, 720.0, "sketch"), (720.0, 850.0, "music")]


def test_parse_acts_garbage_is_empty():
    assert act_split.parse_acts("no json here") == []
    assert act_split.parse_acts("") == []


def test_act_windows_skip_music_and_window_long_acts():
    acts = [{"start": 0.0, "end": 2000.0, "kind": "standup", "title": "Т"},
            {"start": 2000.0, "end": 2300.0, "kind": "music", "title": "M"}]
    windows = act_split.act_windows(acts)
    assert [(w["start"], w["end"], w["part"], w["parts"]) for w in windows] == [
        (0.0, 900.0, 1, 3), (780.0, 1680.0, 2, 3), (1560.0, 2000.0, 3, 3)]


def test_slice_transcript_keeps_overlapping_segments():
    sliced = act_split.slice_transcript(_show(), 95.0, 120.0)
    assert [s["start"] for s in sliced["segments"]] == [90.0, 100.0, 110.0]


def test_select_by_acts_asks_once_per_act_and_merges():
    prompts = []

    def ask(prompt):
        prompts.append(prompt)
        if "split it into ACTS" in prompt:
            return _Resp("### JSON ###\n" + json.dumps({"acts": [
                {"start": 0, "end": 200, "kind": "sketch", "title": "Первый"},
                {"start": 200, "end": 260, "kind": "promo", "title": "Реклама"},
                {"start": 260, "end": 600, "kind": "standup", "title": "Второй"}]})), "m"
        start = 10.0 if "Первый" in prompt else 300.0
        return _Resp("### JSON ###\n" + json.dumps({"shorts": [
            {"start": start, "end": start + 50, "viral_score": 80,
             "viral_reason": "complete bit with setup and laugh"}]})), "m"

    response, model, data = act_split.select_by_acts(_show(), 600, None, ask, workers=1)
    assert len(prompts) == 3                       # split + two acts, promo skipped
    assert "SKETCH / STAND-UP MODE" in prompts[1] and "ONE act" in prompts[1]
    assert "фраза 25" not in prompts[1]            # act 1 sees only its own lines
    assert [s["start"] for s in data["shorts"]] == [10.0, 300.0]
    assert json.loads(response.text) == data and len(response.usages) == 3 and model == "m"


def test_select_by_acts_returns_none_without_acts():
    assert act_split.select_by_acts(_show(), 600, None, lambda p: (None, None)) is None
