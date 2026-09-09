# AGENT-SPLIT-PLAN.md

This file is the single source of truth for the client/server split. 
A fresh session will implement from this document, so it contains the locked decisions, the architecture, the wire protocol, and the implementation roadmap. Read it fully before writing code.

---

## 0. IMPORTANT GENERAL INSTRUCTION !!!
- Always use the codegraph skill to browse, search and inspect flows in code (see the relevant SKILL in ~/SKILLS/codegraph/)
- NEVER COMMIT WITHOUT PRIOR USER APPROVAL
- !!!! NEVER GIT PUSH !!!! Only the user can git push !!!!
- Keep the code simple and clean, don't blidly copy code around; always look at the surroundings, rename symbols to fit new places.
- Don't write comments in code, unless strictly necessary.
- ALWAYS Propose before modify anythhing. Discuss instead of thinking long. User like to discuss with you to find extraordinary solutions TOGETHER.
- We would like to have proper tests for all agent features (where doable)
- Re-read this file a few time per session, to keep the focus on these IMPORTANT instructions.

---

## 1. Goal

Split the current single-process `ds4-agent` (UI + tools + inference all in one)
into two separate binaries connected over TCP:

- **`ds4-agent-server`** (new file `agent/ds4_agent_server.c`): inference. 
  Owns engine, DS4 session state, KV cache, full transcript, disk persistence, compaction, sampling, DSML/GLM parsing + tools orchestration
- **`ds4-agent-client`** (new file `agent/ds4_agent_client.c`): UI + tools execution. 
  Owns the terminal/linenoise editor and tools execution

The existing `ds4-agent` binary is **NOT touched** and stays fully functional.
The two new binaries are built from new files (plus shared header) and reuse the
engine/session/kvstore code in `ds4.c` / `ds4_kvstore.h`.

Requirements (locked):
- Streaming UX must be identical to today (tokens appear live, tool calls are
  visualized live, prefill progress footer, compaction progress, Ctrl+C interrupt).
- Avoid per-token round trips: the protocol must allow the server to stream tokens
  autonomously at decode speed in the common case.
- We cannot loose features on the way, neither the more obscure.

---

## 2. Locked design decisions

0. The goal is to split the current monolitic agent, without loosing a single
   feature, without altering the user experinece and keeping the same code 
   style/conventions of the monolitic agent. Before inventing a solution look
   at how it is done in the monolitic agent.

1. **No edit-upto.** Remove BOTH the `[upto]` marker and the forcer
   (`agent_edit_upto_forcer`, `agent_edit_old_*`, `[upto]` handling in
   `agent_edit_find_old_span`, the `--edit-upto` prompt variant).
   The edit tool becomes exact old/new matching only (`old` must match exactly
   once in the file). Rationale: `[upto]` is ds4-specific, not a standard agent
   convention; removing it eliminates the only source of per-token round trips
   (the forcer inspects the next sampled token before eval, which needs file
   content + next-token text + pre-eval decision = a round trip per token).
   The current agent keeps it; the new client simply does not implement it.

2. **Datetime and system-prompt reminders are server-side**, like the initial
   prompt. The server decides when to inject the datetime context and when to
   re-inject the system prompt reminder (every 50k transcript tokens). 

3. **Server is the sole authority on the transcript.**, and the sole to parse
   render DSML, thinking. Client is model-agnostic, it just render markdown
   and the tokens in the correct format/color given the token kind.

4. **Compaction is server-side.** the server builds the compaction prompt 
   internally, runs generation (think-mode none, greedy, stop at control token), 
   streams the summary tokens to the client for display, rebuilds the compacted 
   transcript. Interrupt during compaction works (server checks interrupt between 
   tokens).

9. **Sessions/persistence are server-side**: sysprompt.kv, saved sessions, titles,
   `/save /switch /del /strip /list`.

10. **One session per TCP connection.** On disconnect the server tears down the
    session and engine. No multi-client handoff (it is a local tool). The client
    opens the connection at startup and keeps it for the process lifetime.

11. **Web approval and bash jobs are client-side** (they are tools). The browser
    (ds4_web) and bash process groups never touch the server.


---

## 3. Ownership table

| Concern                                                 | Server | Client |
|---------------------------------------------------------|--------|--------|
| Engine open/close, backend, model load                  |   ✅   |   —    |
| DS4 session + KV cache + transcript                     |   ✅   |   —    |
| Prefill, sampling, eval, argmax                         |   ✅   |   —    |
| Disk persistence (sysprompt.kv, sessions, titles, list) |   ✅   |   —    |
| Compaction (prompt + summary + rebuild)                 |   ✅   |   —    |
| Greedy/temp + stop decisions                            |   ✅   |   —    |
| System prompt text (build + send)                       |   ✅   |   —    |
| Datetime + system-prompt reminders                      |   ✅   |   —    |
| DSML/GLM parsing + tool-call orchestration              |   ✅   |   —    |
| Marker/think trackers                                   |   ✅   |   —    |
| Terminal/linenoise editor, status footer                |   —    |   ✅   |
| Tools execution: bash, web, read/write/edit/search/list |   —    |   ✅   |
| view_image: read file bytes (client) + image attach (server) |   ✅   |   ✅   |
| Live tool visualization / rendering                     |   —    |   ✅   |
| Queued user prompts, Ctrl+X/ESC, web approval           |   —    |   ✅   |


---

## 4. Wire protocol (binary, super compact)

Transport: TCP. One connection per session. Both directions use the same framing.

### Framing
- 4-byte big-endian length `N` (length of the payload, excluding the 4-byte prefix).
- Then `N` payload bytes.

### Field encodings (payload)
- `varint`: unsigned LEB128 (one byte per 7 bits). Used for small integers.
- `string`: varint byte-length + raw bytes (no NUL terminator).
- `f32`: 4 bytes IEEE-754 little-endian.
- `bool`: 1 byte, 0 or 1.
- `u64`: varint (values fit in 64-bit).


Notes:
- `STATUS` is pushed by the server on every meaningful transition (state change,
  prefill progress, generated count, greedy toggle) so the client footer stays
  live without polling. It is also sent as part of HELLO.
- `TOKEN` carries `id` for tracing (client logs token traces as today), kind and `text`
  for parsing/rendering. The client parses/renders each token as it arrives.
- Vision: `ATTACH_IMAGE` (C2S, generic, not tool-specific) carries the image bytes
  read on the client. The server encodes (engine), builds the multimodal message
  (text + spans), appends to transcript + KV + cache; on failure it sends ERROR.
  Other tools may send images too.

---

## 6. Shared header `ds4_agent_proto.h` + `ds4_agent_utils.h`

Self-contained (no engine, no file I/O, no OS deps beyond malloc/string). 

Both binaries include this headers.

---

## 7. `ds4-agent-server` implementation ideas

File: `ds4_agent_server.c`. Link with ds4.c engine/session + ds4_kvstore.h.

- CLI: reuse the engine-option parsing from ds4_agent.c `parse_options`
  (model, backend, ctx, think, power, mtp, streaming, steering, distributed, etc.)
  but WITHOUT the agent-specific UI/tool options. New option: `--port <port>`
  (default e.g. 9334) and `--host <addr>` (default e.g. 127.0.0.1).
- Main loop: socket accept (single client), then read messages with the framing,
  dispatch to handlers, write framed replies.
- Session lifecycle: create session on NEW_SESSION (ctx from message), load or
  build sysprompt.kv from the received sysprompt text (server-side persistence
  logic copied from ds4_agent.c `agent_worker_reset_to_sysprompt` /
  `agent_kv_load_path` / `agent_kv_save_path`). On disconnect, `ds4_engine_close`.
- Status push: after each state transition / progress callback / greedy toggle,
  build and send STATUS (look at `worker_progress_cb`, `agent_set_status`,
  `agent_set_error` logic).
- Compaction: copy `agent_compact_make_prompt`, `agent_worker_compact` logic
  (build prompt, sync, argmax loop with greedy + stop at control token, rebuild
  transcript, COMPACT_DONE). Stream summary tokens as TOKEN for display.
- Sessions/persistence: copy `agent_session_identity_sha`,
  `agent_session_title_from_text`, `agent_kv_*` helpers, `agent_worker_save_session_now`,
  `agent_worker_find_session/delete/strip/switch`, `agent_worker_list_sessions`,
  `agent_kv_path_for_sha`, `agent_default_cache_dir`.
- Token RPC: `ds4_tokenize_text` on the received text -> COUNT.
- Power: `ds4_session_set_power`.
- Interrupt: a flag checked between tokens; also cancels compaction/sync
  (`ds4_session_set_cancel` with a callback like today).
- Distributed-route wait: keep if relevant (`agent_worker_wait_distributed_route`).
- Vision/ATTACH_IMAGE: port `agent_tool_observation_build/commit` +
  `ds4_chat_append_multimodal_message` + `ds4_engine_vision_encode_memory` +
  `ds4_session_sync_multimodal`; cache like ds4_server `server_image_cache`.
  Observation text built server-side and committed to transcript; sessions with
  images cannot be saved (carry over the restriction).

## 8. `ds4-agent-client` implementation notes

File: `ds4_agent_client.c`. Does NOT link the engine. Links linenoise, ds4_web,
and the shared headers.

- CLI: keep the UI options (prompt, non-interactive, raw, chdir, trace,
  ctx, temp, top-p, min-p, think, edit-upto REMOVED, power, etc.) plus new
  `--server <host:port>` (default localhost:9334). No model/backend options
  (the server owns those).
- Transport: a small TCP client with the framing; a worker thread owns the socket
  and the tool excecution; the UI thread owns the editor (mirror the
  existing worker/UI split in ds4_agent.c).
- UI: copy the editor (agent_editor, scroll regions, CPR, paste, footer) and
  prompt queue from ds4_agent.c unchanged. Session status comes from STATUS.
- Tools: copy agent_tool_* / agent_bash_* / agent_web_* / agent_edit_* /
  agent_search_* / agent_read_* unchanged (all client-side).
- view_image: client reads the image file bytes and sends ATTACH_IMAGE; paints
  `[tool:view_image] path` live from TOOL_CALLS. No `w->images` on the client.
- Web approval: unchanged (client-side ds4_web + UI prompt).
- Trace logging: keep `agent_trace*` (client writes its own trace file).
- Ctrl+C: worker_interrupt sends INTERRUPT;

## 9. Tools
Shared `agent/ds4_agent_tools.h` (tool structs, agent_buf,
fit-context, client worker, dispatch) + one file per group:
`ds4_agent_tools_file.c` (read/more/write/list/edit/search),
`ds4_agent_tools_bash.c` (bash/bash_status/bash_stop),
`ds4_agent_tools_web.c` (google_search/visit_page + approval).
view_image is a special case: it lives on the client (reads file bytes) but the
image attach is a generic server service (ATTACH_IMAGE), not a client tool.
Renamed to fit the new structure (agent_tool_read -> agent_tools_file_read, ...).

## 10. Testing strategy (no real model needed)
Context: the dev machine has 128GB RAM but ds4 weights occupy ~80GB and a live
agent already fills it, so we CANNOT run two real engines here. There is no small
compatible test GGUF. This does NOT block development: the client/server split is
mostly control logic (protocol, state machines, UI, tools, persistence), and only
the innermost inference layer needs a real model. We test by layers with mock
seams; only the last layer requires a machine with RAM.

### Layer 1 — Protocol (pure unit test, no model)
Encode/decode round trips for every message type, framing (4-byte length), varint
edge cases (0, max, multi-byte), string boundaries. Pure byte manipulation in
`ds4_agent_proto.h`; no engine, no RAM. Runs under `DS4_AGENT_TEST`.

### Layer 2 — Client end-to-end against a MOCK SERVER (no model, no second ds4)
The client never touches the engine (TCP only), so a small test harness connects
to the real client as the server and feeds scripted `TOKEN` / `TURN_PAUSED` /
`STATUS` sequences. Verify:
- rendered output on stdout (markdown, tool visualization, think, prefill footer);
- messages the client sends (USER / TOOL / SYSTEM / COMPACT / TOKENS / POWER /
  session commands);
- tool execution (bash / file / web) and fit-context (TOKENS -> COUNT math);
- session command rendering (history / list / save / switch / del / strip) from
  fake replies.
This covers the whole UX + tool loop (the most fragile part) with zero model.

### Layer 3 — Server logic against a MOCK ENGINE (test build only)
The server calls `ds4_session_*` / `ds4_engine_*`. Add a small **backend seam**
(function pointers: sample / eval / sync / argmax / tokenize / token_text) with two
implementations:
- real: wraps ds4 session/engine;
- mock: deterministic, scripted output, compiled ONLY under `#ifdef DS4_AGENT_TEST`.
Under the seam, test (with a mock tokenizer): message handling + framing, state,
sysprompt.kv + session persistence (save/switch/del/strip/list), compaction
(prompt, summary loop, transcript rebuild), sampling loop + TURN_PAUSED
transitions, POWER, TOKENS, interrupt.
The mock never ships in the production binary (respects AGENT.md: no variants
behind flags — diagnostic test seams are fine).

### Layer 4 — Real inference end-to-end (requires a host with RAM)
Only when validating true prefill/sampling: run server + client on a machine with
the model (the CUDA host, or a dedicated test host). The client can even run on
this RAM-bound box since it is TCP-only. Per AGENT.md, ask the user before CUDA /
distributed tests. Do NOT run a second engine while this agent
is live on the dev box.

## 11. Build (Makefile) and testing

- New targets: `ds4-agent-server` (ds4_agent_server.o + ds4.o + ds4_kvstore.o +
  engine objects) and `ds4-agent-client` (ds4_agent_client.o + ds4_web.o +
  linenoise.o). Add to the existing Makefile; do not break existing targets.
- Unit tests: add a `DS4_AGENT_TEST` guard in each new file (like ds4_agent.c) for:
  - protocol encode/decode round trips (every message type);
  - shared parser/detector behavior (DSML + GLM chunked, think-tool ignored,
    greedy boundaries — port the existing test_agent_glm/dsml tests to the
    shared parser);
  - fit-context math.
- Live test: real inference end-to-end happens only with user assistance; on this machine use the mock-server (Layer 2) and mock-engine (Layer 3) seams instead of a real model.

## 12. Prompt build

The prompt round trip is fully server-side. On NEW_SESSION the client sends only
the user `-sys` text (`sys_extra`, plain) plus the sampling/think settings. The
server ports `agent_worker_build_system_tokens` (ds4_agent.c:4822) and builds
all system tokens internally

The client is model-agnostic: it never sees the tools prompt text or schemas, it
only executes tool calls received via TOOL_CALLS (name + args). GLM vs DSML
syntax is decided server-side (`agent_tool_syntax_for_engine`); the client just
renders tokens by kind.

## 13. Constraints carried from AGENT.md

- No C++. Small, readable, elegant code. No dead code / slop.
- Keep the production path whole-model Metal graph inference; do not disturb SSD
  streaming, CUDA, distributed, Metal paths in ds4.c (we only ADD new binaries).
- Preserve correctness before speed; no unexplained attention/KV/logits drift.
- Do not add permanent semantic variants behind flags (edit-upto removal aligns).
- Testing: `make` for build validation; `make test` for unit/regression; live
  server tests only when testing the API surface. At major changes: test Metal
  path, SSD streaming; ask user before distributed/CUDA.
