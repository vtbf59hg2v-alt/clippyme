"""Two-pass selection for long comedy shows (CLIP_STYLE=comedy) — pure, host-testable.

One prompt over a 4-hour show makes the model hunt jokes across thousands of
lines, and it cuts them away from their setups. So the show is handled in two
passes: first the model splits the whole show into ACTS (who is on stage,
where each number starts and ends), then each act is sent on its own, with
the sketch rules, to pick the clips inside it. ``main.get_viral_clips`` makes
the calls
(``ask``); everything here is string/dict work around them.
"""
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor

from clippyme.pipeline.analysis.gemini_parser import (
    _clean_json,
    _extract_json_section,
    parse_gemini_response,
)
from clippyme.pipeline.analysis.gemini_request import (
    build_segment_lines,
    build_viral_prompt,
)

# Acts that never hold a clip worth cutting.
SKIP_KINDS = frozenset({"music", "promo", "ads", "credits"})
# An act shorter than this holds no complete bit; merged into a neighbour.
MIN_ACT_SEC = 45.0
# A longer act is cut into windows so one request stays small and focused.
MAX_ACT_SEC = 900.0
# Overlap between those windows so a bit on the seam is whole in one of them.
WINDOW_OVERLAP_SEC = 120.0

ACT_PROMPT_TEMPLATE = """
You are the editor of a TV comedy show (sketch show / stand-up / guests). Read
the WHOLE show below and split it into ACTS — the separate numbers of the
show, in order, with no gaps between them.

An act is one performer or group on stage doing one thing: a sketch, a
stand-up set, a guest segment or interview, an improvisation, a song. A new
act starts when the host announces someone ("Встречайте", "На сцене", "У нас
в гостях"), when the music/applause between numbers plays, or when the cast
and the topic change together. Keep a long act whole even if it changes topic
midway; do not split a single set into pieces here.

Lines are phrases `[start-end] text` in seconds. Markers such as (laughter),
(applause), (music) are real audience/sound cues. A line flagged
"(ASR repeat ×N)" is a speech-recognition error.

For each act give:
- start: start time of its first line, copied exactly
- end: end time of its last line, copied exactly
- kind: one of sketch, standup, guest, improv, host, music, promo
  (music = a song or band performance; promo = an advert or film plug)
- title: who and what, a few words, in the language of the show

VIDEO_DURATION_SECONDS: {video_duration}

SEGMENTS:
{segments}

## OUTPUT
On its own line emit the LITERAL delimiter `### JSON ###`, then ONLY this
JSON object — no markdown, no code fences, no prose after:
### JSON ###
{{"acts": [{{"start": 12.5, "end": 431.2, "kind": "sketch", "title": "..."}}]}}
"""


def build_act_prompt(transcript_result, video_duration):
    """Prompt for pass 1: the show split into acts."""
    return ACT_PROMPT_TEMPLATE.format(
        video_duration=video_duration,
        segments=build_segment_lines(transcript_result),
    )


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_acts(text, video_duration=None):
    """``[{start, end, kind, title}]`` from the pass-1 answer, time-sorted.

    Unusable entries are dropped, edges are clamped to the source, and an act
    too short to hold a bit is merged into the act before it. Returns ``[]``
    when nothing usable came back — the caller then runs the single pass.
    """
    try:
        data = json.loads(_clean_json(_extract_json_section(text or "")))
    except ValueError:
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(_clean_json(match.group(0)))
        except ValueError:
            return []
    raw = data.get("acts") if isinstance(data, dict) else data
    acts = []
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        start, end = _float(entry.get("start")), _float(entry.get("end"))
        if start is None or end is None:
            continue
        start = max(0.0, start)
        if video_duration:
            end = min(end, float(video_duration))
        if end <= start:
            continue
        acts.append({
            "start": start,
            "end": end,
            "kind": str(entry.get("kind") or "sketch").strip().lower(),
            "title": " ".join(str(entry.get("title") or "").split())[:120],
        })
    acts.sort(key=lambda a: a["start"])
    merged = []
    for act in acts:
        if merged and act["end"] - act["start"] < MIN_ACT_SEC and act["kind"] not in SKIP_KINDS:
            merged[-1]["end"] = max(merged[-1]["end"], act["end"])
            continue
        merged.append(act)
    return merged


def act_windows(acts):
    """Acts to send in pass 2: skipped kinds removed, long acts windowed.

    Each window is ``{start, end, title, part, parts}``; an act up to
    ``MAX_ACT_SEC`` is one window, a longer one overlapping windows.
    """
    windows = []
    for act in acts:
        if act["kind"] in SKIP_KINDS or act["end"] - act["start"] < MIN_ACT_SEC:
            continue
        spans, start = [], act["start"]
        while True:
            end = min(act["end"], start + MAX_ACT_SEC)
            spans.append((start, end))
            if end >= act["end"]:
                break
            start = end - WINDOW_OVERLAP_SEC
        for i, (s, e) in enumerate(spans):
            windows.append({"start": s, "end": e, "title": act["title"],
                            "kind": act["kind"], "part": i + 1, "parts": len(spans)})
    return windows


def slice_transcript(transcript_result, start, end):
    """The segments (and their words) that overlap ``[start, end]``."""
    segments = [
        seg for seg in transcript_result.get("segments", [])
        if seg.get("start") is not None and seg.get("end") is not None
        and seg["end"] > start and seg["start"] < end
    ]
    return {
        "text": " ".join(str(seg.get("text", "")).strip() for seg in segments),
        "segments": segments,
    }


def act_instructions(window, instructions=None):
    """User-instruction text telling pass 2 which act it is looking at."""
    part = f" (part {window['part']} of {window['parts']})" if window["parts"] > 1 else ""
    note = (
        f"This transcript is ONE act of a longer show: {window['title'] or window['kind']}"
        f"{part}, {window['start']:.1f}s–{window['end']:.1f}s. Cut clips only inside it; "
        "timestamps are of the whole show. If the act holds no complete, funny bit, "
        "return an empty shorts list."
    )
    return f"{note}\n{instructions}" if instructions else note


def merge_act_shorts(results):
    """One ``{"shorts": [...]}`` from the pass-2 answers (dicts or None)."""
    shorts = []
    for data in results:
        if isinstance(data, dict) and isinstance(data.get("shorts"), list):
            shorts.extend(s for s in data["shorts"] if isinstance(s, dict))
    return {"shorts": shorts}


class CombinedResponse:
    """The answers of a two-pass run, shaped like one SDK response.

    ``usages`` holds ``(usage_metadata, model_name)`` per call, for pricing.
    """

    def __init__(self, text, usages):
        self.text = text
        self.usages = usages
        self.usage_metadata = None


def select_by_acts(transcript_result, video_duration, instructions, ask, *,
                   creator=None, workers=3, delay=0.0, log_fn=print, sleep_fn=time.sleep):
    """Two-pass comedy selection: ``(response, model_name, data)`` or None.

    ``ask(prompt)`` returns ``(response, model_name)`` or ``(None, None)``.
    Pass 1 asks for the show's acts; pass 2 asks for clips inside each act,
    ``workers`` acts at a time, call ``i`` starting ``delay * i`` seconds in
    (free-tier requests-per-minute). None — pass 1 gave nothing usable, or
    every act failed — means the caller runs the single prompt instead.
    """
    log_fn("🎭 Pass 1: splitting the show into acts...")
    response, model_name = ask(build_act_prompt(transcript_result, video_duration))
    acts = parse_acts(getattr(response, "text", "") or "", video_duration)
    windows = act_windows(acts)
    if not windows:
        log_fn("⚠️  No acts found — falling back to a single pass over the whole show.")
        return None
    skipped = sorted({a["kind"] for a in acts} & SKIP_KINDS)
    log_fn(f"🎭 {len(acts)} acts, {len(windows)} to cut (skipped: {', '.join(skipped) or 'none'})")
    for act in acts:
        log_fn(f"   · {act['start']:.0f}–{act['end']:.0f}s {act['kind']}: {act['title']}")

    def _one(item):
        index, window = item
        if delay:
            sleep_fn(delay * index)
        prompt, _ = build_viral_prompt(
            slice_transcript(transcript_result, window["start"], window["end"]),
            video_duration, act_instructions(window, instructions),
            creator=creator, style="comedy",
        )
        resp, used = ask(prompt)
        if resp is None:
            log_fn(f"⚠️  Act {index + 1}/{len(windows)} got no answer — skipped.")
            return None, None, None
        parsed = parse_gemini_response(resp.text or "", request_id=os.urandom(4).hex())
        count = len((parsed.data or {}).get("shorts") or [])
        log_fn(f"🎬 Act {index + 1}/{len(windows)} ({window['title']}): {count} clip(s) via {used}")
        return parsed.data, getattr(resp, "usage_metadata", None), used

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        results = list(pool.map(_one, enumerate(windows)))
    answered = [r for r in results if r[2]]
    if not answered:
        return None
    usages = [(getattr(response, "usage_metadata", None), model_name)]
    usages += [(usage, used) for _, usage, used in answered]
    data = merge_act_shorts(d for d, _, _ in answered)
    log_fn(f"🎭 Pass 2 done: {len(data['shorts'])} clip candidates from {len(answered)} acts.")
    return CombinedResponse(json.dumps(data, ensure_ascii=False), usages), answered[0][2], data
