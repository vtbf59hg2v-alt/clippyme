"""Tests for the non-Gemini fallback chain and the subscription CLI bridge.

The bridge is exercised end to end: a real ThreadingHTTPServer on a free port
with fake ``claude`` / ``codex`` executables on PATH, called through
``llm_fallback`` exactly as the pipeline does.
"""
import importlib.util
import os
import stat
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from clippyme.pipeline.analysis import llm_fallback as lf

ROOT = Path(__file__).resolve().parents[3]


class _Resp:
    def __init__(self, status=200, payload=None, text=""):
        self.status_code = status
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _ok(content):
    return _Resp(payload={"choices": [{"message": {"content": content}}]})


def test_parse_fallbacks_skips_malformed_entries():
    fbs = lf.parse_fallbacks(
        "claude|http://h:3901/v1/|claude; bad entry ;x|ftp://h|m;"
        "paid|https://api.example.com/v1|gpt|sk-1")
    assert [f.name for f in fbs] == ["claude", "paid"]
    assert fbs[0].base_url == "http://h:3901/v1"
    assert fbs[1].api_key == "sk-1"


def test_primary_choice_moves_that_fallback_first(monkeypatch):
    fbs = lf.parse_fallbacks("claude|http://b/v1|claude;codex|http://b/v1|codex;local|http://l/v1|q")
    assert [f.name for f in lf.prefer(fbs, "codex")] == ["codex", "claude", "local"]
    assert lf.prefer(fbs, "missing") is None
    monkeypatch.delenv("LLM_PRIMARY", raising=False)
    assert lf.primary_name() == "gemini"
    monkeypatch.setenv("LLM_PRIMARY", " Claude ")
    assert lf.primary_name() == "claude"


def test_chain_moves_on_and_strips_thinking(monkeypatch):
    calls = []

    def post(url, json, headers, timeout):
        calls.append((url, json["model"], headers.get("Authorization")))
        if "first" in url:
            return _Resp(429, text="limit")
        return _ok("<think>hmm</think>\n{\"shorts\": []}")

    monkeypatch.setattr(lf.requests, "post", post)
    fbs = lf.parse_fallbacks("a|http://first/v1|m1;b|http://second/v1|m2|k")
    resp, used = lf.generate_with_fallbacks("prompt", fbs, log_fn=lambda *_: None)
    assert resp.text == '{"shorts": []}' and resp.usage_metadata is None
    assert used == "b:m2"
    assert calls[1] == ("http://second/v1/chat/completions", "m2", "Bearer k")


def test_all_failing_raises(monkeypatch):
    monkeypatch.setattr(lf.requests, "post", lambda *a, **k: _ok(""))
    with pytest.raises(RuntimeError, match="all fallback models failed"):
        lf.generate_with_fallbacks("p", lf.parse_fallbacks("a|http://x/v1|m"),
                                   log_fn=lambda *_: None)
    with pytest.raises(RuntimeError, match="no LLM_FALLBACKS"):
        lf.generate_with_fallbacks("p", [], log_fn=lambda *_: None)


# --- bridge ------------------------------------------------------------------

def _load_bridge():
    spec = importlib.util.spec_from_file_location("llm_bridge", ROOT / "scripts" / "llm_bridge.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _fake_cli(dir_: Path, name: str, body: str) -> None:
    path = dir_ / name
    path.write_text("#!/bin/sh\n" + body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


@pytest.fixture
def bridge(tmp_path, monkeypatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    # claude: echo the args and the stdin length back.
    _fake_cli(bin_dir, "claude", 'echo "claude $* :: $(wc -c | tr -d " ")"\n')
    # codex: write the final message to the --output-last-message file.
    _fake_cli(bin_dir, "codex",
              'out=""; while [ $# -gt 0 ]; do [ "$1" = "--output-last-message" ] && out="$2"; shift; done\n'
              'cat > /dev/null; echo "noise on stdout"; echo "codex final" > "$out"\n')
    _fake_cli(bin_dir, "broken", 'echo "usage limit reached" >&2; exit 3\n')
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    mod = _load_bridge()
    server = ThreadingHTTPServer(("127.0.0.1", 0), mod.Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield mod, f"http://127.0.0.1:{server.server_address[1]}/v1"
    server.shutdown()


def test_bridge_claude_gets_prompt_on_stdin(bridge):
    _, url = bridge
    text = lf.call_fallback(lf.Fallback("claude", url, "claude"), "x" * 5000)
    assert text.startswith("claude -p --model sonnet")
    assert text.endswith(":: 5000")


def test_bridge_codex_reads_last_message_file(bridge):
    _, url = bridge
    assert lf.call_fallback(lf.Fallback("codex", url, "codex"), "hi") == "codex final"


def test_bridge_cli_failure_is_429_and_unknown_model_400(bridge, monkeypatch):
    _, url = bridge
    monkeypatch.setenv("LLM_BRIDGE_CLAUDE_CMD", "broken")
    with pytest.raises(RuntimeError, match="HTTP 429.*usage limit"):
        lf.call_fallback(lf.Fallback("c", url, "claude"), "hi")
    with pytest.raises(RuntimeError, match="HTTP 400"):
        lf.call_fallback(lf.Fallback("g", url, "gpt-4"), "hi")
