# helexa-acp

ACP (Agent Client Protocol) bridge for editors like
[Zed](https://zed.dev). Lets you point your editor's agent panel at
**any combination** of OpenAI-compatible, OpenAI Responses, and
Anthropic Messages endpoints — public APIs, private LAN deployments,
local Ollama / LM Studio — and switch between them per session via a
model dropdown.

The "missing ACP binary" for users who don't want to be locked into
one vendor's agent client.

```
       ┌───────────────────────────────────┐
       │  Zed (or any ACP editor client)   │
       └────────────┬──────────────────────┘
                    │  stdio JSON-RPC (ACP)
                    ▼
            ┌─────────────────┐
            │   helexa-acp    │  ← one binary, multi-endpoint
            └─────┬───────────┘
                  │  HTTP / SSE
         ┌────────┼─────────────┬──────────────┬──────────────┐
         ▼        ▼             ▼              ▼              ▼
    cortex/    OpenAI       Anthropic      OpenRouter    LM Studio
    neuron    Responses    Messages
   (self-     (gpt-5,…)    (Claude)
    hosted)
```

## What it does

- **Speaks ACP** over stdio to editor clients (Zed today; any future
  ACP client tomorrow).
- **Multi-endpoint** — one config file lists every LLM endpoint
  you want available; pick one per session via the model dropdown
  (`endpoint:model` selector).
- **Three wire formats**: `openai-chat` (the broadly compatible
  default), `openai-responses` (newer OpenAI surface), and
  `anthropic-messages` (Claude). Each is a separate provider impl
  in `src/provider/`; adding a fourth (Gemini, Ollama native, …) is
  one file plus a `WireApi` enum variant.
- **Built-in tools**: `read_file`, `write_file`, `edit_file`,
  `list_dir`, `bash`. Permission-gated by default; the editor user
  approves writes/shell per-call.
- **Three session modes**: Default (gated), Bypass Permissions
  (auto-allow), and Plan (write-only-to-plan-dir, no shell).
- **Vision** — drag-drop images into the agent panel against any
  vision-capable model.
- **Session resume** — multi-day conversations survive editor
  restarts via on-disk transcript persistence.
- **Context compaction** — rolling history stays inside the model's
  context window automatically so long sessions on small-context
  local models don't fall over.

## Install

### From source

```sh
git clone https://git.lair.cafe/helexa/cortex.git
cd cortex
cargo install --path crates/helexa-acp
# Binary lands at ~/.cargo/bin/helexa-acp
```

### Pre-built RPM (Fedora 43)

```sh
dnf copr enable helexa/helexa
dnf install helexa-acp
```

The COPR project bundles helexa-acp alongside the cortex gateway
and helexa-neuron flavours; install only the package(s) you need.

## Quick start

The fastest path: env-var single-endpoint config.

```sh
export HELEXA_ACP_BASE_URL=http://hanzalova.internal:31313/v1
export HELEXA_ACP_MODEL=Qwen/Qwen3.6-27B
helexa-acp  # speaks ACP over stdin/stdout; not interactive
```

Then in Zed (`~/.config/zed/settings.json`):

```jsonc
{
  "agent_servers": {
    "helexa": {
      "command": "helexa-acp",
      "args": []
    }
  }
}
```

Restart Zed → open the agent panel → pick "helexa" → start
chatting. Tool calls (file reads, writes, bash) prompt for
permission per-call in Default mode.

That's the minimum. The full config story below is what unlocks
the multi-endpoint dropdown.

## Multi-endpoint config

Copy `helexa-acp.example.toml` from this repo to
`$XDG_CONFIG_HOME/helexa-acp/config.toml` (typically
`~/.config/helexa-acp/config.toml`) and edit:

```toml
default_endpoint = "helexa"

[[endpoints]]
name = "helexa"
base_url = "http://hanzalova.internal:31313/v1"
wire_api = "openai-chat"
default_model = "Qwen/Qwen3.6-27B"
max_tokens = 8192
context_window = 32768

[[endpoints]]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "openai-chat"
api_key_env = "OPENROUTER_API_KEY"
default_model = "anthropic/claude-opus-4"

[[endpoints]]
name = "anthropic"
base_url = "https://api.anthropic.com/v1"
wire_api = "anthropic-messages"
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-opus-4"
```

Restart Zed. The model dropdown lists every model from every
configured endpoint with the `endpoint:model` selector
(`helexa:Qwen/Qwen3.6-27B`, `openrouter:anthropic/claude-opus-4`,
…). Switch mid-session; the next prompt routes to the new endpoint.

When only one endpoint is configured the prefix is dropped (model
ids appear bare).

### Selector syntax

The `model` field on every internal request is parsed as
`<endpoint>:<model>`:

- `openrouter:gpt-4o` → routes to the `openrouter` endpoint,
  model `gpt-4o`.
- `helexa/large` → no colon → falls through to whichever endpoint
  is named in `default_endpoint`, model `helexa/large`.
- `:gpt-5` → leading colon → also falls through to default.

## Endpoint cookbook

Copy-pasteable blocks. Mix and match.

### cortex / neuron (self-hosted)

```toml
[[endpoints]]
name = "helexa"
base_url = "http://hanzalova.internal:31313/v1"
wire_api = "openai-chat"
default_model = "Qwen/Qwen3.6-27B"
max_tokens = 8192
context_window = 32768
```

Use `openai-responses` instead of `openai-chat` once cortex 0.1.16+
is deployed and you want the Responses API surface (vision item
shape, structured reasoning items, etc.).

### OpenAI directly

```toml
[[endpoints]]
name = "openai"
base_url = "https://api.openai.com/v1"
wire_api = "openai-responses"
api_key_env = "OPENAI_API_KEY"
default_model = "gpt-5"
```

`openai-responses` is the right choice for current OpenAI models;
`openai-chat` works against legacy GPT-3.5/4 deployments and
anything labelled "chat completions".

### Anthropic directly

```toml
[[endpoints]]
name = "anthropic"
base_url = "https://api.anthropic.com/v1"
wire_api = "anthropic-messages"
api_key_env = "ANTHROPIC_API_KEY"
default_model = "claude-opus-4"
```

helexa-acp sends `x-api-key` + `anthropic-version: 2023-06-01`
automatically. The `api_key_env` indirection keeps your key out of
the config file.

### OpenRouter (multi-vendor proxy)

```toml
[[endpoints]]
name = "openrouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "openai-chat"
api_key_env = "OPENROUTER_API_KEY"
default_model = "anthropic/claude-opus-4"
```

OpenRouter speaks OpenAI-compat for every model it fronts, so
`openai-chat` is the right wire format regardless of the
underlying vendor.

### LM Studio (local)

```toml
[[endpoints]]
name = "lmstudio"
base_url = "http://localhost:1234/v1"
wire_api = "openai-chat"
default_model = "auto"
```

LM Studio's "auto" model id picks whatever's loaded. Same shape
works for Ollama in compat mode (`http://localhost:11434/v1`) and
vLLM.

### Multiple cortex deployments

```toml
[[endpoints]]
name = "lan"
base_url = "http://hanzalova.internal:31313/v1"
wire_api = "openai-chat"
default_model = "Qwen/Qwen3.6-27B"

[[endpoints]]
name = "cloud"
base_url = "https://cortex.example.com/v1"
wire_api = "openai-chat"
api_key_env = "CLOUD_CORTEX_KEY"
default_model = "Qwen/Qwen3-VL-8B"
```

Use the `endpoint:model` selector to switch between them mid-session.

## Zed setup

`~/.config/zed/settings.json`:

```jsonc
{
  "agent_servers": {
    "helexa": {
      "command": "helexa-acp"
    }
  }
}
```

Optional environment overrides for the binary:

```jsonc
{
  "agent_servers": {
    "helexa": {
      "command": "helexa-acp",
      "env": {
        "HELEXA_ACP_LOG_FILE": "/tmp/helexa-acp.log",
        "RUST_LOG": "helexa_acp=debug"
      }
    }
  }
}
```

`HELEXA_ACP_LOG_FILE` is the one you actually want — Zed doesn't
surface the agent's stderr, so without that env var debug output is
invisible. Point it at a file you can `tail -f`.

After restarting Zed: ⌘+? (or wherever your "Open Agent Panel"
binding is) → select "helexa" → the model dropdown populates from
your config → start prompting.

## Modes

Three session modes ship; the user picks via Zed's mode dropdown
on the agent panel.

| Mode | Reads | Writes | Bash | Permission prompts |
|------|-------|--------|------|--------------------|
| **Default** | ✓ | with prompt | with prompt | per call |
| **Bypass Permissions** | ✓ | ✓ | ✓ | never |
| **Plan** | ✓ | only into plan dir | disabled | never (plan-dir writes auto-allow) |

### Default

Reads are always allowed (`read_file`, `list_dir` are
unrestricted). Writes and shell commands prompt the user before
running. The intended baseline for any session where the agent
might do something you'd rather review first.

### Bypass Permissions

Auto-allow every tool call. Use for agentic loops you trust — bulk
edits across many files, scripted workflows, prepared session
templates. Never for code the agent hasn't seen before.

### Plan

The "draft an implementation plan before you write code" mode.
Available tools:

- `read_file`, `list_dir`: unrestricted (read the codebase).
- `write_file`, `edit_file`: allowed *only* under
  `$XDG_DATA_HOME/helexa-acp/plans/<project-id>/`. Any path
  outside that returns "plan mode: writes are restricted to …"
  back to the model so it self-corrects.
- `bash`: disabled outright. Returns "plan mode: shell execution
  is disabled" if attempted.

When the plan is complete, the model presents a 3-option menu:

1. **Bypass Permissions** — implement the plan now, no prompts.
2. **Default** — implement now with per-tool prompts.
3. **Plan** (stay here) — refine the plan with more guidance.

Switch the mode dropdown to your preference and reply to proceed.

## Tools

Five tools, defined in `src/tools.rs`:

| Tool | Args | Gated in Default? |
|------|------|-------------------|
| `read_file` | `path`, `line?`, `limit?` | no |
| `list_dir` | `path` | no |
| `write_file` | `path`, `content` | yes |
| `edit_file` | `path`, `old_text`, `new_text` | yes |
| `bash` | `command`, `cwd?` | yes |

### Path handling

`~`, `~/`, `$HOME`, and `$HOME/` are expanded server-side before
the path reaches ACP or local fs. Lets the model emit
`~/git/repo/file.rs` and have it Just Work.

`read_file` first tries the editor's filesystem (ACP's
`fs/read_text_file` — respects open buffers, workspace overlays,
etc.). If that fails — typically because the path is outside Zed's
workspace boundary — it falls back to `std::fs::read_to_string`.
This lets the agent pull in shared material like
`~/git/architecture/generic.md` from a different project's
session.

The fallback is logged at warn level so you can see when it kicks
in.

### Tool dispatch

Tool descriptions reach the model through a Qwen3 Hermes-format
`# Tools` block injected into the system prompt — cortex/neuron
pass the OpenAI `tools` request field through to the encoder
unread, so we work the model into emitting `<tool_call>{json}</tool_call>`
markers it then parses out of the content stream. This applies to
the helexa wire format; OpenAI / Anthropic endpoints with native
tool support would use their own paths once they're wired in.

The parser is tolerant: malformed JSON (trailing braces, missing
`name`, name nested in `arguments`) gets a repair pass; if that
fails the call surfaces as a "Malformed tool call" card in Zed and
the model gets a synthetic error result so it can self-correct.

## Session resume

helexa-acp persists every session to
`$XDG_DATA_HOME/helexa-acp/sessions/<id>.json`. Zed's `session/list`
RPC asks helexa-acp to enumerate them on workspace open;
`session/load` rehydrates and replays the transcript as
`session/update` notifications so the agent panel renders the
prior conversation.

Behaviour:

- Persisted per-round, so a mid-turn agent stall (long bash, wedged
  ACP roundtrip) doesn't lose earlier rounds.
- Survives editor restart and the helexa-acp binary upgrading
  between versions.
- Project-scoped: only sessions whose `cwd` matches the workspace
  are listed.

To wipe history: `rm -rf $XDG_DATA_HOME/helexa-acp/sessions/`.

## Context compaction

When an endpoint sets `context_window`, helexa-acp projects the
rolling history into a token budget before each request — old
`ToolResult` content (read_file payloads are the worst offenders)
gets elided to one-line markers, preserving `tool_call_id` pairing
so the wire schema stays valid.

System prompts, user turns, and the most recent ~4 messages are
never elided. The full history stays on disk; compaction is a
per-request projection, not a destructive edit.

Set `context_window = 32768` for a 32 K Qwen3, `131072` for a
modern Claude, etc. With `max_tokens` also set, the budget is
`context_window - max_tokens - 512_safety`.

## Troubleshooting

### "default endpoint 'helexa' has no usable provider — check config"

The named default endpoint failed to construct. Usually:

- `api_key_env` references a variable that isn't set in the env
  Zed launched helexa-acp with.
- The TOML's `wire_api` is misspelled (only `openai-chat`,
  `openai-responses`, `anthropic-messages` are accepted).

Test by running `helexa-acp` directly from a shell — startup
errors land on stderr.

### Model dropdown is empty

Each provider's `list_models` failed at startup. Look at
`HELEXA_ACP_LOG_FILE` for "list_models failed; this endpoint's
models won't appear in the picker". Likely the endpoint URL is
wrong, the API key is invalid, or the upstream `/v1/models`
endpoint isn't responding.

The agent still works against `default_model` even when the
dropdown is empty — list-models is for picking, not routing.

### "prompt_too_long" / agent stalls mid-conversation

You hit the model's context window. Set `context_window` on the
endpoint and helexa-acp will compact before sending. The log line
`context compaction applied` confirms it's running; if it fires
but the upstream still rejects, the compaction heuristic
under-counted and the budget needs tuning down.

### Reading files outside the workspace returns "not found"

Zed's `fs/read_text_file` is workspace-scoped. helexa-acp falls
back to local `std::fs` automatically when that fails — look for
`fs/read_text_file failed; falling back to local std::fs` in the
log. If even local read fails, the file genuinely doesn't exist
or the user process lacks permissions.

### Tool calls render as text instead of structured cards

The model is emitting `<tool_call>` markers that the parser can't
decode. Two common causes:

1. The system prompt isn't reaching the model (cortex/neuron's
   tool-block injection didn't fire). Confirm with
   `RUST_LOG=helexa_acp=debug` and look at the outgoing
   `POST /chat/completions` body.
2. The model itself is too small / undertrained to follow the
   Hermes format reliably. helexa-acp has shape-based name
   inference and JSON repair, but there's a floor below which
   nothing helps.

### Plan-mode writes refused even inside the plan dir

The path comparison is byte-for-byte. If the model emits a path
with `~` and the plan_dir has the expanded form, expansion runs
*before* the comparison — but resolved-vs-symlinked-path
mismatches can still bite. The error message names the attempted
path and the expected prefix so you can compare directly.

## Architecture

Source layout under `crates/helexa-acp/src/`:

| File | Responsibility |
|------|----------------|
| `main.rs` | tokio + Stdio transport. Builds providers, hands off to `agent::Agent` |
| `config.rs` | TOML + env-fallback config, endpoint resolver |
| `agent.rs` | ACP handlers (initialize, session/new, session/prompt, session/cancel, session/set_mode, session/set_model, session/load, session/list), prompt loop with tool-call recursion |
| `session.rs` | Per-session state map (Arc<RwLock<HashMap<…>>>) |
| `store.rs` | On-disk session persistence, plan-dir resolution |
| `prompt.rs` | System-prompt assembly, plan-mode addendum |
| `tools.rs` | Tool schemas + shape-based name inference |
| `tool_runner.rs` | Dispatch a single tool call through ACP client RPCs; permission gate |
| `qwen3.rs` | Qwen3 Hermes tool-format parser (`<tool_call>` / `<think>` markers) |
| `compaction.rs` | Token-budget compaction for the rolling history |
| `path_util.rs` | `~` / `$HOME` expansion shared across every path-taking tool |
| `provider/openai_chat.rs` | OpenAI chat completions provider |
| `provider/openai_responses.rs` | OpenAI Responses API provider |
| `provider/anthropic_messages.rs` | Anthropic Messages API provider |

### Adding a new wire format

1. New file under `src/provider/` implementing the `Provider`
   trait (encoder + SSE decoder).
2. Add a `WireApi` variant in `config.rs`.
3. Wire it into `build_provider` in `main.rs`.
4. Done — every other module is wire-format-agnostic.

### Concurrency

- `Arc<RwLock<HashMap<SessionId, Arc<Mutex<SessionState>>>>>` —
  per-session mutex so concurrent requests across sessions don't
  contend; the map's RwLock is read-mostly.
- Every tool call dispatched serially within a session (parallel
  dispatch would require Zed to handle interleaved permission
  prompts).
- Provider streams are back-pressured by the consumer (bounded
  mpsc channels).

### Self-contained

The crate has no workspace-internal dependencies (no
`cortex-core`, no `cortex-gateway`). Migration to a dedicated
GitHub repo for cross-platform CI / cargo-dist binaries is
Cargo.toml-only.

## Status

- Stages 1–6 shipped: scaffold, agent loop, tools, modes, session
  resume, image input, model picker, three wire formats.
- Stage 8 (RPM + multi-platform CI) tracked in the canonical plan;
  Linux x86_64 RPM ships today via the cortex monorepo's Gitea
  Actions.

## Contributing

Repository: https://git.lair.cafe/helexa/cortex (`crates/helexa-acp/`).
Issues / PRs welcome. The canonical staged plan is in
`~/.claude/plans/plan-the-per-device-worker-abstract-micali.md` on
the maintainer's machine; the substages 3a–3e and 6a/6b that the
canonical plan didn't anticipate are documented in commit messages.

CI: `cargo fmt --check --all`, `cargo clippy --workspace -- -D
warnings`, `cargo test --workspace` must all pass before merge.
