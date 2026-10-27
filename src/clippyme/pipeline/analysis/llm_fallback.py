"""Non-Gemini fallback models for the clip-selection call.

When every Gemini model in the chain is rate-limited or failing (or no Gemini
key is set), the same prompt goes to the next endpoint in ``LLM_FALLBACKS``.
Every fallback speaks the OpenAI ``/chat/completions`` dialect, which covers:

- local servers — Ollama, LM Studio, llama.cpp ``llama-server``;
- the subscription bridge in ``scripts/llm_bridge.py``, which answers on the
  host by running the Claude Code / Codex CLIs on the user's own login;
- any paid OpenAI-compatible API, if the user wants one.

``LLM_PRIMARY`` set to one of the fallback names flips the order: that model
answers first, the other fallbacks follow, and Gemini comes last.

``LLM_FALLBACKS`` is a ``;``-separated list of ``name|base_url|model[|api_key]``
entries, tried in order, e.g.::

    claude|http://host.docker.internal:3901/v1|claude;local|http://host.docker.internal:8090/v1|qwen

The response is wrapped to look like a Gemini response (``.text`` and an empty
``usage_metadata``) so parsing, validation and cost logging stay unchanged.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

import requests

DEFAULT_TIMEOUT = 900  # local models on a laptop are slow on a long transcript
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


@dataclass(frozen=True)
class Fallback:
    name: str
    base_url: str
    model: str
    api_key: str = ""


@dataclass
class FallbackResponse:
    text: str
    usage_metadata: None = None


def parse_fallbacks(raw: str | None) -> list[Fallback]:
    out = []
    for entry in (raw or "").split(";"):
        parts = [p.strip() for p in entry.split("|")]
        if len(parts) < 3 or not all(parts[:3]):
            continue
        if not parts[1].startswith(("http://", "https://")):
            continue
        out.append(Fallback(parts[0], parts[1].rstrip("/"), parts[2],
                            parts[3] if len(parts) > 3 else ""))
    return out


def configured() -> list[Fallback]:
    return parse_fallbacks(os.getenv("LLM_FALLBACKS"))


def primary_name() -> str:
    """``LLM_PRIMARY``: ``gemini`` (default) or the name of a fallback entry
    that should answer first, with Gemini kept as the last resort."""
    return (os.getenv("LLM_PRIMARY") or "gemini").strip().lower() or "gemini"


def prefer(fallbacks: list[Fallback], name: str) -> list[Fallback] | None:
    """Move the fallback called ``name`` to the front. ``None`` when no
    entry has that name (the caller then keeps Gemini first)."""
    chosen = [fb for fb in fallbacks if fb.name.lower() == name]
    if not chosen:
        return None
    return chosen + [fb for fb in fallbacks if fb.name.lower() != name]


def call_fallback(fb: Fallback, prompt: str, timeout: float | None = None) -> str:
    headers = {"Content-Type": "application/json"}
    if fb.api_key:
        headers["Authorization"] = f"Bearer {fb.api_key}"
    body = {
        "model": fb.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.4,
        "stream": False,
    }
    timeout = timeout or float(os.getenv("LLM_FALLBACK_TIMEOUT", str(DEFAULT_TIMEOUT)))
    resp = requests.post(f"{fb.base_url}/chat/completions", json=body, headers=headers,
                         timeout=timeout)
    if resp.status_code != 200:
        snippet = re.sub(r"[\x00-\x1f\x7f]", " ", (resp.text or "")[:300])
        raise RuntimeError(f"{fb.name} returned HTTP {resp.status_code}: {snippet}")
    try:
        content = resp.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"{fb.name} returned an unexpected body") from exc
    # Reasoning models (Qwen3, DeepSeek) may inline their thinking.
    content = _THINK_RE.sub("", content).strip()
    if not content:
        raise RuntimeError(f"{fb.name} returned an empty answer")
    return content


def generate_with_fallbacks(prompt: str, fallbacks: list[Fallback] | None = None,
                            log_fn=print) -> tuple[FallbackResponse, str]:
    """Try each fallback in order. Returns ``(response, "name:model")``;
    raises the last error when all of them fail."""
    fallbacks = configured() if fallbacks is None else fallbacks
    if not fallbacks:
        raise RuntimeError("no LLM_FALLBACKS configured")
    last: Exception | None = None
    for fb in fallbacks:
        log_fn(f"🔁 Trying fallback model {fb.name} ({fb.model})…")
        t0 = time.monotonic()
        try:
            text = call_fallback(fb, prompt)
        except Exception as exc:  # noqa: BLE001 — move on to the next one
            last = exc
            log_fn(f"   ⚠️  {fb.name} failed: {exc}")
            continue
        log_fn(f"   ✅ {fb.name} answered in {time.monotonic() - t0:.0f}s")
        return FallbackResponse(text), f"{fb.name}:{fb.model}"
    raise RuntimeError(f"all fallback models failed (last: {last})")
