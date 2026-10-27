#!/usr/bin/env python3
"""Subscription LLM bridge: an OpenAI-compatible endpoint backed by the
Claude Code and Codex command-line tools on this machine.

ClippyMe runs in Docker and can't use CLIs logged in on the host. This tiny
server runs on the host instead and answers ``POST /v1/chat/completions`` by
piping the prompt into ``claude -p`` (model ``claude``) or ``codex exec``
(model ``codex``). Both run on the user's own Claude / ChatGPT login, so a
call costs no API money — it uses the subscription's limits instead.

Standard library only. Run it on the host:

    python3 scripts/llm_bridge.py            # listens on 127.0.0.1:3901

and point ClippyMe at it, first in line or as a fallback:

    LLM_FALLBACKS=claude|http://host.docker.internal:3901/v1|claude;codex|http://host.docker.internal:3901/v1|codex

The commands are overridable (``LLM_BRIDGE_CLAUDE_CMD`` / ``LLM_BRIDGE_CODEX_CMD``,
shell-split) in case a CLI version renames a flag. Each runs in an empty
temporary directory with its tools locked down: the prompt is a transcript,
not a task, so the model only needs to answer in text.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

DEFAULT_COMMANDS = {
    # -p: print mode (non-interactive). No tools: answer from the prompt only.
    # Sonnet: plenty for picking clips, and it spares the subscription limits.
    "claude": "claude -p --model sonnet --output-format text --tools ''",
    # exec: non-interactive; '-' reads the prompt from stdin; the final
    # message is written to the file appended as --output-last-message.
    "codex": "codex exec --skip-git-repo-check --sandbox read-only -",
}
# Where the CLIs live when they're not on PATH (Codex ships inside ChatGPT.app).
KNOWN_PATHS = {
    "claude": [os.path.expanduser("~/.local/bin/claude")],
    "codex": ["/Applications/ChatGPT.app/Contents/Resources/codex-cli/bin/codex"],
}
TIMEOUT = float(os.getenv("LLM_BRIDGE_TIMEOUT", "900"))
MAX_BODY = 8 * 1024 * 1024


def _resolve(binary: str) -> str:
    if shutil.which(binary):
        return binary
    for path in KNOWN_PATHS.get(binary, []):
        if os.access(path, os.X_OK):
            return path
    return binary


def command_for(model: str) -> list[str] | None:
    key = "claude" if model.startswith("claude") else "codex" if model.startswith("codex") else None
    if key is None:
        return None
    cmd = shlex.split(os.getenv(f"LLM_BRIDGE_{key.upper()}_CMD") or DEFAULT_COMMANDS[key])
    return [_resolve(cmd[0]), *cmd[1:]]


def flatten(messages: list[dict]) -> str:
    parts = []
    for m in messages or []:
        content = m.get("content")
        if isinstance(content, list):  # OpenAI content parts
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        if content:
            parts.append(str(content) if m.get("role") in (None, "user")
                         else f"[{m.get('role')}]\n{content}")
    return "\n\n".join(parts)


def run_cli(model: str, prompt: str) -> str:
    cmd = command_for(model)
    if cmd is None:
        raise ValueError(f"unknown model {model!r}; use 'claude' or 'codex'")
    if shutil.which(cmd[0]) is None:
        raise FileNotFoundError(f"{cmd[0]} is not installed or not on PATH")
    with tempfile.TemporaryDirectory(prefix="llm-bridge-") as workdir:
        last_msg = os.path.join(workdir, "last_message.txt")
        if os.path.basename(cmd[0]) == "codex" and "--output-last-message" not in cmd:
            cmd = [*cmd[:-1], "--output-last-message", last_msg, cmd[-1]] if cmd[-1] == "-" \
                else [*cmd, "--output-last-message", last_msg]
        proc = subprocess.run(cmd, check=False, input=prompt, capture_output=True, text=True,
                              cwd=workdir, timeout=TIMEOUT)
        if proc.returncode != 0:
            raise RuntimeError(f"{os.path.basename(cmd[0])} exited {proc.returncode}: "
                               f"{(proc.stderr or proc.stdout).strip()[-500:]}")
        if os.path.exists(last_msg):
            with open(last_msg, encoding="utf-8") as f:
                return f.read().strip()
        return proc.stdout.strip()


class Handler(BaseHTTPRequestHandler):
    server_version = "ClippyMeLLMBridge/1.0"

    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            models = [{"id": m, "object": "model"} for m in DEFAULT_COMMANDS
                      if shutil.which(command_for(m)[0])]
            self._send(200, {"object": "list", "data": models})
        else:
            self._send(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._send(404, {"error": {"message": "not found"}})
            return
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_BODY:
            self._send(413, {"error": {"message": "body missing or too large"}})
            return
        try:
            req = json.loads(self.rfile.read(length))
            model = str(req.get("model") or "claude")
            prompt = flatten(req.get("messages"))
            if not prompt:
                raise ValueError("empty prompt")
        except (ValueError, AttributeError) as exc:
            self._send(400, {"error": {"message": str(exc)}})
            return
        t0 = time.monotonic()
        try:
            text = run_cli(model, prompt)
        except (ValueError, FileNotFoundError) as exc:
            self._send(400, {"error": {"message": str(exc)}})
            return
        except subprocess.TimeoutExpired:
            self._send(504, {"error": {"message": f"{model} timed out"}})
            return
        except RuntimeError as exc:
            # Usage-limit messages land here: 429 lets the caller move on.
            self._send(429, {"error": {"message": str(exc)}})
            return
        self.log_message("%s answered in %.0fs (%d chars in, %d out)",
                         model, time.monotonic() - t0, len(prompt), len(text))
        self._send(200, {
            "id": f"bridge-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "model": model,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": text}}],
        })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=3901)
    args = ap.parse_args()
    found = [m for m in DEFAULT_COMMANDS if shutil.which(command_for(m)[0])]
    print(f"LLM bridge on http://{args.host}:{args.port}/v1 — models: {', '.join(found) or 'none found'}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
