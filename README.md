# Local LLM Server with Tool-Calling Backend

A self-hosted AI assistant capable of acting on the filesystem (creating and
running code), with no dependency on Claude Code, Aider, or any paid API.
Built as a fully local alternative for cases where a commercial coding
assistant isn't available or desired.

## Architecture

```
[Browser]                  [Backend]                  [Inference server]
chat-orchestrated.html  →  backend.py (FastAPI)  →    vLLM (Docker)
                            decides: chat or             Qwen2.5-7B
                            action, executes it          with tool-calling
                            write_file / run_command
                                  ↓
                            Real filesystem
                            (create, edit, execute)
```

Three independent components:

1. **vLLM (Docker)** — serves the model through an OpenAI-compatible API.
   Generates text and, when appropriate, a structured function call
   (`tool_calls`). Cannot touch anything on its own.
2. **backend.py** — receives the user request, forwards it to the model. If
   the model requests a tool (`write_file`, `run_command`), the backend
   executes the corresponding action on disk and returns the result to the
   model for the final response.
3. **chat-orchestrated.html** — frontend. Sends/displays text and logs which
   actions were executed and their result.

## Why not just use Aider

Aider is built specifically for code editing in a repo (diff-based edits,
auto-commits), which biases it toward treating every message as a code
change request. A general-purpose assistant (read system state, run a
script, create a file and report back) needs its own orchestration layer
without that bias — hence the custom backend.

## Tool-calling: model/parser compatibility

The initial model tested (`Qwen2.5-Coder-7B-Instruct-AWQ`) understood tasks
correctly but returned the function call as plain text wrapped in invented
tags instead of a structured `tool_calls` field, regardless of which vLLM
parser was configured (`hermes`, `qwen3_coder`).

Root cause: tool-calling requires three pieces to match exactly — the model
(trained for a specific call format), the vLLM parser (translates that
format into the `tool_calls` field), and the chat template embedded in the
checkpoint. This particular AWQ build of the "Coder" variant didn't carry a
correctly aligned template.

Fix: switched to `Qwen2.5-7B-Instruct-AWQ` (general-purpose, non-code-specialized)
with `--tool-call-parser hermes`. Verified with:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "messages": [...], "tools": [...], "tool_choice": "auto"}'
```
Expected: `"content": null`, populated `"tool_calls"`, `"finish_reason": "tool_calls"`.

## Requirements

- NVIDIA GPU, 8GB+ VRAM
- Docker + NVIDIA Container Toolkit (see `RUNBOOK.md` for full setup)
- Python 3.10+ for the backend (virtual environment recommended)

## Usage

**1. Start the model server:**
```bash
docker compose up
```

**2. Start the backend:**
```bash
python3 -m venv backend-env
source backend-env/bin/activate
pip install fastapi uvicorn httpx
python3 backend.py
```

**3. Open the frontend:**
Open `chat-orchestrated.html` in a browser.

## Verifying actions are real

```bash
ls ~/test-vllm/
cat ~/test-vllm/<filename>
```
If the file exists with the expected content, the action was executed by
the backend on disk — not just described by the model.

## Available tools

| Tool | Description |
|---|---|
| `write_file` | Creates or overwrites a file in the workspace |
| `run_command` | Runs a shell command inside the workspace (15s timeout) |

Extend via `TOOLS` and `ACTIONS` in `backend.py`.

## Known limitations

- 7B model: reliable for simple tasks, unreliable for complex reasoning or
  specific factual claims.
- `run_command` executes whatever the model decides; the workspace is
  confined to a single directory but there is no process-level sandboxing.
  Do not expose this to the internet without adding further isolation.
- No persistent memory across browser sessions.

## See also

- `RUNBOOK.md` — exact commands to reproduce the full setup on a new
  machine, including a table of common errors and root causes.
