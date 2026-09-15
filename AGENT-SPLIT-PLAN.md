# Splitting `ds4_agent.c` into `ds4-agent-server` + `ds4-agent-client` over TCP

## Context

`ds4-agent` is today a single binary (`ds4_agent.c`, 13120 lines) that runs the
terminal UI, tool execution and LLM inference in one process. We need to run the
agent UI on a different machine than the one doing inference (the machine with
the ~80GB model / GPU). Goal: two binaries connected by a TCP link, without
losing **any** feature and without altering the user experience (live token
streaming, live tool-call visualisation, prefill progress footer, compaction
progress, Ctrl+C).

This branch (`agent-split-cc`) is a **fresh, independent** attempt: the work on
the `agent-split` branch is to be ignored entirely.

**Status:** T1-T13 implemented; three corrections came out of Layer 4 testing
on real hardware (a ROCm Strix Halo host) and are folded into this document —
see "Server bootstrap / model-loading feedback" in §Architecture and Risks
20-22. If re-executing this plan from scratch, those sections already reflect
the corrected design; do not reintroduce the sequential engine-open-then-
listen order, the unbounded STREAM coalescing buffer, or the single-blocking-
`conn_recv` handshake wait.

## Locked decisions (with the user)

1. **New standalone files.** `ds4_agent.c` and the `ds4-agent` binary stay
   untouched and fully functional. We create self-contained `ds4_agent_server.c`,
   `ds4_agent_client.c` that **copy** the needed code from `ds4_agent.c`, renaming
   symbols where required. Code duplication with the monolith is accepted.
2. **Binary framed transport over TCP.** Every frame = a **12-byte header
   `{u32 magic, u32 type, u32 len}`** + `len` payload bytes, **all big-endian**
   (`htonl`/`ntohl`), matching `ds4_distributed` — the one in-tree protocol that
   is a real cross-machine link. `magic = 0x44533441` ("DS4A", joins `DS4D` /
   `DS4T` / `DS4B`); a byte-swapped or wrong magic ⇒ wrong peer / endianness ⇒
   close. `type` is a flat `enum agent_msg` (one value per message, sub-commands
   included). Payload primitives: fixed-width `u8` / `u16` / `u32` (big-endian);
   `string` = `u32 len` + UTF-8 bytes (no NUL); `blob` = `u32 len` + octets;
   `bool` = 1 byte, rejected if ∉ {0,1}; `array` = `u32 count` + elements.
   **No varint. No f32** — scalar knobs travel as fixed-point `u32` (see §1).
   **Single frame cap `AGENT_PROTO_MAX_FRAME = 32 MiB`**, a hard maximum for
   every frame (images inline, no chunking), rejected before allocation. A
   single **active** connection at a time and a single session (local tool, no
   multi-client). On disconnect the server does **not** destroy the session: it
   parks it in RAM (engine kept) and resumes it on the next reconnect
   (`SESSION resume`) — see §Architecture "Reconnect". SIGINT / server shutdown
   free everything.
   - **Positional bodies, no field tags.** Each message body is a fixed ordered
     sequence of primitives with a hand-written encode/decode pair; no field IDs,
     no optional fields. `HELLO` negotiates `proto_version`; a mismatch is a hard
     error (`ERR` + close, "rebuild both binaries"). Correct because `make`
     always builds and deploys the two binaries together — protobuf-style tagged
     fields would be over-engineering.
3. **Full removal of edit-upto.** No `[upto]` marker, no `agent_edit_upto_forcer`,
   no `agent_preflight_edit_old` / `agent_stream_preflight_closed_param`, no
   `--edit-upto` option, no `[upto]` branch in `agent_edit_find_old_span`. The
   `edit` tool becomes exact `old`/`new` matching; `old` must occur **exactly
   once** in the file, checked **client-side** at execution time. This removes
   the only mid-generation client round-trips.
4. **Minimal message catalogue.** A small orthogonal operation set — **8 client
   verbs + 4 server pushes** (§1). Session management and runtime knobs are
   sub-commands of one verb each (`SESSION`, `CONFIG`) with generic replies
   (`OK/ERR/INT/STR/MAP/ARR`) — no per-command typed codec. Turn accept/reject/
   done and compaction begin/end are carried by `STATUS` (no `TURN_*` /
   `COMPACTION_*` messages); the `✦` notices and compaction summary text are
   `STREAM` kinds. No client-side tokenize-count RPC — the client uses byte
   caps, `agent_tool_observation_fits` is the server-side gate.

## Development and test environment

- **Dev box:** Linux x86_64, ~31 GB RAM, GCC 16.2 / clang 22, **no GPU
  toolchain** (no nvcc, no hipcc, no ROCm). The tree builds clean.
  `UNAME_S=Linux` ⇒ `make` alone prints help; `make test` requires
  `$(DS4_LINK)=nvcc` → **not usable here**.
  - Local build: **`make cpu`** (`-DDS4_NO_GPU`, `$(CPU_CORE_OBJS)`, link
    `$(CC)`).
  - Local tests: binaries built/run **individually** (no `make test`). The new
    tests must have a **CPU form** that builds with `$(CC)` without nvcc/model
    (see §4).
  - **No real inference here** (31 GB < ~80 GB model, no GPU): only Layer 1-3 +
    Layer 2 mock.
- **Layer 4 host (real inference):** AMD AI MAX+ **Strix Halo / gfx1151**,
  128 GB shared RAM, **ROCm backend**. Build there: `make strix-halo` (alias
  `make rocm`); test suite: `make test-rocm`. Layer 4 is done there **with the
  user** (`AGENT.md`: ask before touching ROCm / distributed).
- **macOS / Metal:** only full `make` + `make test` are a gate there; not
  available in this project except on a Mac.

## Scope of changes to existing files

- **`ds4_agent.c`: NEVER touched** (locked decision). Gate: `git diff` doesn't
  show it; `grep -c edit_upto ds4_agent.c` unchanged.
- **Not touched** — only linked / included / copied-from: `ds4.c`,
  `ds4_kvstore.c`, `ds4_web.c`, `linenoise.c`, `ds4_prompt_prefix.c`,
  `ds4_gpu_args.c`, `ds4.h`, `ds4_kvstore.h`, `ds4_tool_text.h`. Caveat: if a
  needed `ds4_*` symbol turns out to be `static` in `ds4.c`/`ds4_kvstore.c`
  (verify in T3/T4), it is **reimplemented server-side**, NOT de-staticised.
- **`ds4_help.h` + `ds4_help.c`: TOUCHED, additive.** `ds4_help.h`: two new enum
  values `DS4_HELP_AGENT_SERVER` / `DS4_HELP_AGENT_CLIENT`. `ds4_help.c`: new
  `case`s in `tool_name` / `tool_usage` / `tool_summary` / `tool_has_topic` /
  `print_more_info` / `print_examples` / `print_topic` / `print_default` +
  `print_agent_server_specific` / `print_agent_client_specific`, reusing the
  existing helpers (`print_model_runtime` / `print_sampling` / `print_steering` /
  `print_distributed` for the server; `print_agent_specific` /
  `print_agent_sessions` for the client). `DS4_HELP_AGENT` (monolith)
  unchanged. Server help in T3, client help in T8, polish in T13. The new
  binaries' `usage()` call `ds4_help_print(fp, DS4_HELP_AGENT_SERVER/_CLIENT,
  topic)` — **not** inline text.
- **`Makefile`: additions only.** New `.o` rules, new binary/test targets, the
  `test-cpu` phony, new entries in the `all:` (Darwin) / `cpu:` / `cuda-*` /
  `strix-halo` / `test:` / `test-frontends` / `test-rocm -B` / `clean:` lists. No
  existing line rewritten in content. Targets join the lists **in the task that
  creates them**.
- **`AGENT.md` / `README.md` / `QA_BEFORE_RELEASES.md`**: modified in T13
  (documents, not code).
- **Consequence:** `git status` at every step = new files + additive hunks in
  `Makefile` + (from T3) in `ds4_help.h`/`ds4_help.c` + (T13) the 3 docs. Since
  `ds4_help.o` is shared by all binaries, after any touch to `ds4_help.*` the
  regression gate (`make cpu` green for all binaries + `make test-cpu`) is
  **mandatory**.

## Architecture: the seam already exists

The monolith has two threads — UI and worker — that communicate **only** through
`w->mu`-guarded fields + a self-pipe wakeup. The contract is already
message-shaped. The network split maps onto that boundary.

| SERVER (inference machine) | CLIENT (user machine) |
|---|---|
| `ds4_engine`, `ds4_session`, `w->transcript` (authority) + image-span array | editor/linenoise (scroll region, CPR filter, bracketed paste, async output above the prompt) |
| KV checkpoint + saved sessions (`agent_kv_*`, `sysprompt.kv`, SHA1(title‖created_at) identity) | prompt queue (`agent_prompt_queue`), Ctrl+C / Ctrl+X / ESC |
| compaction (`agent_worker_compact_transcript` + helpers) | footer/status line (`build_status_text`, progress bar, prefill label), `✦` echoes |
| build system+tools prompt, GLM/DSML selection (`ds4_engine_is_glm_dsa`), datetime/reminder/hints | all painting: markdown, syntax highlight, fenced code, hide `<think>` (`agent_token_renderer` + `agent_tool_visualizer`) |
| DSML/GLM streaming parser (`agent_dsml_parser`, `agent_glm_tool_parse`, `agent_stream_*`) | **tool execution**: read/more/write/list/edit/search/google_search/visit_page/view_image + background bash + web approval (`ds4_web`) |
| greedy vs stochastic sampling decision (`agent_stream_wants_greedy_sampling`) — **stays server-side ⇒ no per-token round-trip** | `more_*` pagination cursor, tool-output byte caps |
| `worker_run_turn` orchestration, speculative/MTP, interrupt in `ds4_session_set_cancel` | |

**Rendering split.** Today the streaming does two distinct things in one pass,
living in the same function: **(1) classification** — the parser
(`agent_dsml_parser` + `renderer_process` for `<think>` +
`agent_stream_tool_events`, ds4_agent.c:3876) decides byte by byte what a chunk
of the stream is: normal assistant text / inside `<think>` / tool name / param
name / param value; **(2) painting** — takes the classification and emits ANSI +
markdown + syntax highlight to the terminal.

The split cuts between them. The **server keeps (1)** (it needs the parser anyway
to extract the tool calls to run) and, instead of painting, emits **stream
fragments** `STREAM{kind, text}` where `kind` **is** the classification result:
`kind ∈ NORMAL / THINK / TOOL_NAME / TOOL_PARAM_NAME / TOOL_PARAM_VALUE` (the same
`STREAM` message also carries `SUMMARY` for compaction summary text and `SYSTEM`
for the `✦` notices — see §1). Adjacent same-kind fragments are coalesced. The
**client does (2)**, driven by `kind`, with painting code **copied verbatim from
the monolith**. ("Render" is a client concern — the wire carries a classified
stream fragment, not a paint directive.)

*Normal vs grey-thinking.* `renderer_process` (3335) today hides the literal
`<think>`/`</think>` tags, keeps the `in_think` flag, and if a tag is split
across two tokens it buffers until it resolves (3363-3372). In the split **all of
this stays server-side**: the server consumes `<think>`/`</think>`, never sends
them, tags every fragment `kind=THINK` while `in_think` then `kind=NORMAL`. The
client receives `(THINK,"...")` → paints grey; `(NORMAL,"...")` → normal colour +
the text goes through the markdown machine (`renderer_markdown_feed`) for
bold/italic/fence/heading. When a `STREAM` fragment arrives, the classification
is already final — the client never guesses.

*Tool call.* The parser decomposes the stanza into tool name → for each param:
name + value. The server emits the structure as a sequence of fragments, e.g.
for `edit`: `STREAM{TOOL_NAME,"edit"}` → `STREAM{TOOL_PARAM_NAME,"path"}` →
`STREAM{TOOL_PARAM_VALUE,"src/foo.c"}` → `STREAM{TOOL_PARAM_NAME,"old"}` →
`STREAM{TOOL_PARAM_VALUE,"<old text, chunked>"}` →
`STREAM{TOOL_PARAM_NAME,"new"}` → `STREAM{TOOL_PARAM_VALUE,"<new text>"}`.
The client keeps only *current tool* + *current param* (updated by the
`TOOL_NAME`/`TOOL_PARAM_NAME` fragments) and rebuilds the exact same
visualisation as today, because the painting is a **pure function of `(tool,
param)`**: `agent_tool_viz_prefix` ("bash"→`"$ "`, "edit"→`"edit "`, … 3496),
`agent_tool_param_kind_for` / `agent_tool_param_color` (path green, edit/old red,
edit/new green, bash/command bold cyan, … 3428/3448), code-body syntax highlight
+ `-`/`+` diff prefixes for write/edit, the summary line "Reading <path>
<start>:<max>..." composed from the received param values
(`agent_read_default_lines` uses `ctx_size` from the `SESSION new`/`resume` reply).

*What does NOT cross the wire:* raw DSML bytes (`<｜DSML｜invoke...>`), `<think>`
tags, ANSI escapes, the `\r\x1b[2K` line-clears — generated/consumed on their
respective sides. The server sends only `(kind, clean text)`; the client
generates the ANSI. The notices (`[tool call ignored: ...]`, `[invalid tool
call: ...]`, `[tool call interrupted]`) travel as `kind=NORMAL`; the `✦` notices
as `kind=SYSTEM`.

**Tool execution over the wire.** Reuses the existing handshake shape for the
user-queue drain (`worker_request_queued_user_drain`, ds4_agent.c:4944 /
worker_run_turn:10376): the server emits `TOOL_CALLS` and **suspends** the turn
loop; the client runs the tools and replies `TOOL_RESULT` (text + optional
images); the server appends the observation to the transcript
(`agent_tool_observation_commit`) and resumes.

**Vision.** The client reads the image file bytes and sends them in the `TURN`
message's `images[]` array (generic, not tool-specific); the server encodes with
`ds4_engine_vision_encode_memory`, builds the multimodal message, appends to
transcript+KV. `view_image`: client reads the bytes, returns them in
`TOOL_RESULT.images[]`, server encodes. Sessions with images cannot be saved (the
`agent_worker_save_session_now`, ds4_agent.c:5204, restriction is carried over).

**CLI split.** engine/model/backend/ctx/mtp/dspark/ssd/power/dir-steering/
distributed/TP options → SERVER + `--port` (default 7878) / `--host` (default
127.0.0.1). UI options (prompt, prompt-file, prefix-file, non-interactive,
raw-prompt, `-sys`, trace, chdir) → CLIENT + `--server host:port`. `ctx_size` is
server-CLI only (`--ctx`), **not** in `SESSION new`. Sampling/think/seed/power/
steer/hints are client flags that populate `SESSION new`. The `-sys` text is
sent by the client; the server builds all the system tokens.

**Security / network (decided).** The protocol is plaintext TCP, **no
authentication**: whoever reaches the port gets a session with inference +
access to the on-disk KV store (`~/.ds4/kvcache`, `/save` `/del` `/strip`).
Supported model: **`--host 127.0.0.1` by default**, cross-machine use **only via
an SSH tunnel** (or a trusted LAN/VPN). No token, no TLS in the protocol.
`AGENT.md` / `README` / `usage()` must say so explicitly (T13). `--host 0.0.0.0`
is possible but at the user's risk.

**Reconnect / parked session (decided).** On socket loss the server does **not**
destroy the session: it **parks** it (engine + `ds4_session` + transcript +
images stay in RAM) and returns to `accept()`. If a turn was in progress, the
drop is handled as `INTERRUPT` (append turn-end, session left at a well-formed
IDLE boundary). On reconnect the client chooses: `SESSION resume` (re-enters the
parked session) or `SESSION new` (the parked one is freed and a fresh one is
built). A single active connection at a time stays the rule; "parked" = no
socket. SIGINT / server shutdown free from any state.

**Server bootstrap / model-loading feedback (decided, added after Layer 4
testing on real hardware — get this right the first time, do not
reintroduce the sequential version).** A naive port of the monolith's
`main()` opens the engine (`ds4_engine_open`, minutes for a large model)
**synchronously before the TCP listener even exists** — the client cannot
connect, let alone see anything, until loading finishes; this was
implemented, shipped, and had to be fixed. The listener **must** come up
first and stay independent of engine-loading state:
- `run_server()` order: `server_listen()` (bind/listen) → spawn a
  background **boot thread** that calls the engine-open sequence, unchanged,
  and then attaches the engine to the one long-lived `agent_worker` → the
  `accept()` loop starts immediately, concurrently with that thread.
- `agent_worker` needs two lifecycle stages, not one: an engine-independent
  base init (wake pipe, mutex/cond, cache dir, sysprompt path — produces a
  valid, lockable `agent_worker*` usable for HELLO replies and the FIFO/push
  mechanism) and a later engine-attach step (`ds4_session_create` + assigning
  the engine pointer **under the worker mutex**, since the accept loop can
  now read it concurrently — then, only then, spawn the existing worker
  thread). A `bool engine_ready` (engine attached **or** permanently failed)
  and a `bool boot_failed` (permanent failure only) sit next to the existing
  `initialized` flag (which still means "engine open **and** sysprompt.kv
  ready" — do not conflate the two).
- `HELLO` is answered **unconditionally**, regardless of loading state (it
  already doesn't wait on `initialized` even without this change): if the
  engine isn't attached yet, the reply's engine-derived fields are zeroed and
  the new `model_loading` bool is set (§1). The reader/writer threads and the
  push FIFO are already stood up right after the HELLO reply in the existing
  design, independent of engine or worker state, so they are usable
  immediately for the next point.
- While the engine loads, a small heartbeat thread pushes a periodic
  `STREAM{SYSTEM}` notice ("Loading model... (Ns elapsed)") — same mechanism
  as every other one-off system notice (`agent_publish_system_status`), same
  ~100ms-poll/named-interval style as the `STATUS`/`STREAM` coalescing
  policies elsewhere in this doc (a **fixed** start timestamp for the printed
  total; only the next-tick threshold advances — a first implementation
  reset the start timestamp on every tick and got a counter stuck at the
  interval value instead of counting up: keep them separate). No new
  `STREAM` kind, no GiB/percentage (that would need a progress-callback hook
  in `ds4.c`/`ds4.h`, explicitly out of scope — those files stay untouched;
  see "Scope of changes to existing files").
- The client prints a one-time "Connected — the model is still loading..."
  line right after decoding the HELLO reply if `model_loading` is true, using
  the same inline stream renderer the handshake already needs for the next
  point. It needs **no other client-side plumbing** for the periodic
  heartbeats themselves — see the very next paragraph.
- **The client's `SESSION new`/`resume` reply wait must not be a single
  blocking `conn_recv`.** It must loop and dispatch (print) any interleaved
  `STREAM`/`STATUS` frame before the actual `SESSION` reply arrives —
  mirroring the tolerance `client_rpc` already has for every later RPC. A
  first implementation did a single `conn_recv` here and hard-failed with
  "expected SESSION reply, got STREAM" the moment `agent_worker_reset_to_
  sysprompt` (building/loading `sysprompt.kv`, itself already emitting
  `STREAM{SYSTEM}` notices — see the compaction/system-prompt sections below)
  or this new loading heartbeat pushed a notice before the reply. This is the
  same interleaving Risk 3 already documents for the distributed-route wait;
  it just wasn't implemented as loop-tolerant on the client the first time.
- `server_apply_session_new`'s existing `worker_is_initialized` poll (up to
  ~5 minutes) now also has to cover the full engine-load time for the first
  client that connects while still booting, not just the sysprompt.kv build —
  add a `boot_failed` check inside the loop so a permanent load failure is
  reported immediately instead of after the full timeout; the budget itself
  can stay at 5 minutes.
- **Accepted consequence, not a bug:** since `ds4.c` stays untouched, there is
  no cancellation point inside the blocking `ds4_engine_open()` call. `run_server`
  must join the boot thread before freeing the worker (which destroys its
  mutex/cond) or returning, so Ctrl+C during a long model load stops the
  listener/any connection immediately but the **process** only exits once
  that in-flight call returns on its own. Document this in `AGENT.md`/`usage()`
  in T13 rather than trying to "fix" it here.
- **STREAM coalescing needs a periodic force-flush, independent of this
  feature but found alongside it in the same Layer 4 pass.** `srv_emit`'s
  coalescing buffer must flush on a timer (a named
  `AGENT_STREAM_FLUSH_INTERVAL_SEC`, ~100ms) in addition to the existing
  triggers (a kind change, `finish=true` at turn end). A first implementation
  only had the latter two: a long, uninterrupted run of plain assistant text
  or `<think>` content — the common case, no kind change for the whole
  turn — sat fully buffered and only reached the client in one lump at the
  very end of generation, i.e. no visible token streaming at all. Coalescing
  inside each interval window is still correct and still the point (it keeps
  batching fast bursts like tool-call name/param bytes); only the *unbounded*
  wait was the bug.

## 1. Wire protocol — message catalogue

Framing (see Locked decision 2): every frame is a **12-byte header
`{u32 magic, u32 type, u32 len}`** + `len` payload bytes, **all big-endian**.
`magic = 0x44533441` ("DS4A"); a wrong / byte-swapped magic ⇒ reject + close.
`type` is a flat `enum agent_msg` in `ds4_agent_proto.h` (one value per message,
`SESSION` / `CONFIG` sub-commands included). **Single frame cap
`AGENT_PROTO_MAX_FRAME = 32 MiB`** — hard maximum for every frame (images
inline, no chunk sub-protocol), over-cap `len` rejected before allocation.
Primitives: fixed-width `u8` / `u16` / `u32` big-endian; `string` = `u32 len` +
UTF-8 bytes; `blob` = `u32 len` + octets; `bool` = 1 byte (∉ {0,1} → reject);
`array` = `u32 count` + elements. **No varint, no f32** — scalar knobs are
fixed-point `u32` (scales below). Bodies are positional; no field tags.

The catalogue is deliberately small: **8 client verbs + 4 server pushes**.
Session management and runtime knobs are sub-commands of one verb each
(`SESSION`, `CONFIG`) with **generic replies** — no per-command typed codec.
Only the messages with real structure (`HELLO` reply, `SESSION new`/`resume`
reply, `TURN`, `STREAM`, `STATUS`, `TOOL_CALLS`, `TOOL_RESULT`, `DRAIN_REPLY`)
get an `ap_encode_*` / `ap_decode_*` pair. The wire stays binary; the
minimalism is in the *operation set*, not the framing.

### Generic reply values

A command reply is `u32 reply_tag` + body, one of: `OK` (no body) · `ERR`
(`string message`) · `INT` (`u32`) · `STR` (`string`) · `MAP` (`u32 n` +
n×(`string key`, `string value`)) · `ARR` (`u32 n` + n×`MAP`). Every
`SESSION` / `CONFIG` reply is one of these; the structured replies below name
their fields but travel as a `MAP`.

### Fixed-point scalar fields

No float crosses the wire. `temperature`, `top_p`, `min_p`, `dir_steering_ffn`
and the `CONFIG` `steer` scale travel as **milli-units** (`u32`, value ×1000 —
`0.7` → `700`); `STATUS.prefill_tps` / `STATUS.gen_tps` (display only) as
**centi-units** (`u32`, value ×100). The `+bool set` flags on the samplers are
unchanged (a `0` milli-value is still a valid explicit `0.0`).

### Client → Server

| Name | When | Body / reply |
|---|---|---|
| `HELLO` | after TCP connect | `u32 proto_version`, `string client_version`, `string cwd` (trace only). **Reply** = `MAP` with `proto_version`, `engine_is_glm`, `has_vision`, `ctx_size_cli` (server's `--ctx`, indicative — the authoritative value comes in the `SESSION new`/`resume` reply), `backend_name`, `power_percent`, `mtp_draft_tokens`, `model_name`, `vocab_size`, `session_parked`, `model_loading` (added in proto v2, see "Server bootstrap / model-loading feedback" below — `true` while the engine hasn't finished loading yet, in which case every other engine-derived field above is zeroed, not a real capability); or `ERR` (version mismatch; a second connection while one is **active** → `ERR "busy"` then close). |
| `SESSION` | session lifecycle + management; every sub-command except `new`/`resume` requires worker IDLE | `u32 subcmd` + args — see the sub-command table. |
| `CONFIG` | runtime knobs | `u32 op` (0 get, 1 set), `u32 key` (0 power, 1 steer, 2 hints); on set the value (`u32` power 1..100 / `u32` steer, milli-units / `bool` hints). **Reply** `MAP {ok, value, error}` — `get` echoes the current value; `power` set is applied between tokens (`worker_apply_pending_power`) and `STATUS` re-echoes it; `steer` set requires IDLE (like ds4_agent.c:12821). |
| `TURN` | the user submits a line | `string text`, `array<image>` { `blob file_bytes`, `string source_path` } (empty unless the line references image files — this replaces the separate `ATTACH_IMAGE`). **No discrete reply** — the first `STATUS` push is the ack. |
| `INTERRUPT` | Ctrl+C / ESC while busy | *no payload*. Outcome visible via `STATUS`. |
| `STOP` | clean client exit | *no payload*. |
| `TOOL_RESULT` | client has run the tools | `u32 request_id`, `array<string> text_parts`, `array<image>` { `blob file_bytes`, `string source_hint` }. Sent in reply to a `TOOL_CALLS` push. |
| `DRAIN_REPLY` | client drains `agent_prompt_queue` | `string text` (empty ⇒ nothing). After a compaction the client **prepends** the "bash jobs still running" note for live jobs (Risk 7, option b). Sent in reply to a `DRAIN_REQUEST` push. |

### `SESSION` sub-commands

| subcmd | Args | Reply |
|---|---|---|
| `new` | `string sys_text`, `string prefix_file_text`, `u32 temperature`+`bool set`, `u32 top_p`+`bool set`, `u32 min_p`+`bool set` (all milli-units), `u32 seed`, `u32 think_mode`, `u32 n_predict`, `bool raw_prompt`, `bool hints_enabled`, `u32 dir_steering_ffn` (milli), `bool power_set`+`u32 power`. **No ctx_size.** Frees any parked session. | `MAP` with `ctx_used`, `ctx_size` (**authoritative** — `agent_worker_effective_ctx_size` after the think-mode adjustment; the client uses it for footer and fit-context), `distributed_route_ready`, `state` (`enum agent_state`), `note`; or `ERR` (sysprompt build failed). Replaces `NEW_SESSION` + `SESSION_READY`. |
| `resume` | *none* — valid only after `HELLO` reported `session_parked=true` | same `MAP` shape (state of the parked session); `state` is always IDLE after cleaning up any interrupted turn. Replaces `RESUME_SESSION` + `SESSION_READY`. |
| `save` | *none* | `MAP {ok, scheduled, sha, tokens, error}` |
| `switch` | `string sha_prefix`, `u32 history_turns` | `MAP {ok, sha, title, ctx_used, error}`, then a `STREAM{NORMAL}` history dump. `sha_prefix` = the current session's sha ⇒ re-dump the current history (covers the old `CMD_HISTORY`). |
| `list` | *none* | `ARR` of `MAP {sha, title, tokens, created_at, last_used, is_current}` |
| `del` | `string sha_prefix`, `bool strip` | `MAP {ok, sha, tokens, error}`. `strip=true` keeps the metadata, drops only the KV (the old `CMD_STRIP`). |
| `compact` | *none* | no discrete reply; drives a `STREAM{SUMMARY}` stream framed by `STATUS{COMPACTING}` … `STATUS{IDLE, ctx_used}`. If the worker is busy: `STREAM{SYSTEM, "compaction scheduled…"}`. |

### Server → Client (push)

| Name | When | Fields |
|---|---|---|
| `STREAM` | generation, history dumps, compaction summaries, the `✦` notices; coalesced per kind | `u32 stream_id`, `u32 kind` (`enum agent_stream_kind`, below), `string text` |
| `STATUS` | every state transition + error; otherwise coalesced ~5 Hz (~10 Hz during PREFILL) | mirror of `agent_status` (ds4_agent.c:104-117): `u32 state` (`enum agent_state` = mirror of `agent_worker_state` 93-102: 0 IDLE, 1 PREFILL, 2 GENERATING, 3 COMPACTING, 4 DRAINING, 5 SAVING, 6 ERROR, 7 STOPPED), `u32 prefill_done`, `u32 prefill_total`, `u32 prefill_label`, `u32 prefill_tps` (centi), `u32 generated`, `u32 gen_tps` (centi), `bool greedy_sampling`, `u32 ctx_used`, `u32 ctx_size`, `u32 power_percent`, `string error` |
| `TOOL_CALLS` | parser at `AGENT_DSML_DONE`; server appends turn-end then **suspends** | `u32 request_id`, `array<call>` { `string name`, `array<arg>` { `string name`, `string value`, `bool is_string` } } |
| `DRAIN_REQUEST` | after a tool result is committed (`worker_request_queued_user_drain`), and after every successful compaction | *no payload* |

**Turn control lives in `STATUS`, not in dedicated messages:**
- `TURN` accepted ⇒ `STATUS{state=PREFILL}`; rejected (worker not idle) ⇒
  `STATUS{state=IDLE, error="busy"}`.
- compaction ⇒ `STATUS{state=COMPACTING}` … `STATUS{state=IDLE, ctx_used}`; the
  human-readable `reason` rides in a `STREAM{SYSTEM}` at entry.
- turn end ⇒ `STATUS{state=IDLE|ERROR, ctx_used = w->transcript.len}`. The client
  unhides the editor on a terminal state reached from PREFILL / GENERATING /
  COMPACTING. There is **no** `TURN_DONE`.

Interrupt while suspended in `TOOL_CALLS`: the client sends `INTERRUPT`; the
server wakes the handshake with `w->interrupt` set, appends no observation, emits
`STREAM{SYSTEM, "Stopped by user"}` + `STATUS{IDLE, ctx_used}`.

### `STREAM.kind` — `enum agent_stream_kind` (in `ds4_agent_proto.h`)

The result of the server parser's classification (see §Architecture). It is a
**semantic** tag — the protocol never names a colour; the client maps kind →
paint. The client keeps *(current tool, current param)* from the last
`TOOL_NAME` / `TOOL_PARAM_NAME` to interpret the `TOOL_PARAM_VALUE`s. The server
coalesces adjacent same-kind fragments; a kind change is an implicit boundary
(flush).

| kind (`enum agent_stream_kind`) | val | meaning (what the fragments tag) | client painting (code copied from the monolith) |
|---|---|---|---|
| `AGENT_STREAM_NORMAL` | 0 | visible assistant text, outside `<think>` and outside a tool stanza; **also** the notices `[tool call ignored: ...]` / `[invalid tool call: ...]` / `[tool call interrupted]` | full markdown machine (`renderer_markdown_feed`: bold/italic/code fence + syntax highlight/heading/hint aside `> **Hint:**`) + default colour |
| `AGENT_STREAM_THINK` | 1 | text inside `<think>...</think>` — the server has already stripped the tags | `renderer_set_grey`, no markdown |
| `AGENT_STREAM_SUMMARY` | 2 | compaction summary tokens | grey, no markdown (same paint as THINK; a distinct kind because the trigger and the client's banner differ) |
| `AGENT_STREAM_SYSTEM` | 3 | the `✦` notices (`agent_publish_system_status`) | `✦ ` prefix line, system colour |
| `AGENT_STREAM_TOOL_NAME` | 4 | the just-recognised tool name (`p->current.name`) | new line `🛠️ ` + per-tool prefix (`$ ` bash, `edit `, `write `, `search `, `Reading ` for read, ... — `agent_tool_viz_prefix` 3496) in bold; for `read` opens the green "Reading " style |
| `AGENT_STREAM_TOOL_PARAM_NAME` | 5 | the name of a tool-call param (`p->param_name`) | param label in the colour of `agent_tool_param_kind_for(tool, param)` → `agent_tool_param_color` (3428/3448) |
| `AGENT_STREAM_TOOL_PARAM_VALUE` | 6 | a chunk of the current param's value | param colour; for `write`/`edit` it is "code body" → syntax highlight + `-`/`+` diff prefixes (`agent_tool_viz_param_is_code_body` / `agent_tool_viz_diff_prefix`); for `read` the bytes are accumulated into the summary line `Reading <path> <start>:<max>...` (`agent_tool_viz_render_read`) |

In the rest of the document these are referred to by their short name (`NORMAL`,
`THINK`, `SUMMARY`, `SYSTEM`, `TOOL_NAME`, `TOOL_PARAM_NAME`,
`TOOL_PARAM_VALUE`).

No separate kinds are needed for "code body", "diff", "read line": they are all
derivable client-side from `(tool, param)`, which the client knows from the
preceding `TOOL_NAME` / `TOOL_PARAM_NAME` directives.

Client-side the `agent_tool_viz_*` functions today read the parser fields
(`sr->parser->current.name`, `->param_name`, `->state`) via
`agent_stream_tool_events`. In the split `client_apply_stream_fragment(kind,
text)` synthesises the same calls from `(kind, text)`, keeping the current
*(tool, param)* in a reduced `agent_stream_renderer` **without a `parser`**.

### Fit-context — no RPC

The client trims tool output with the conservative byte caps
(`AGENT_TOOL_MAX_BYTES` etc.; `read`/`more` bucket on `ctx_size` from `SESSION
new`/`resume`). The **authoritative** fit check stays server-side
(`agent_tool_observation_fits` + compaction retry, worker_run_turn:10322-10368):
a `TOOL_RESULT` that would overflow triggers a mid-turn compaction and, if it
still doesn't fit, a truncated error observation. There is no tokenize-count
round-trip.

### No payload
`INTERRUPT`, `STOP`, `DRAIN_REQUEST`, `SESSION resume`, `SESSION save`, `SESSION
list`, `SESSION compact`.

### Edge cases the protocol test must cover
- `{magic, type, len}` header: `u32` big-endian regardless of host endianness; a
  byte-swapped `DS4A` magic → reject + close; an unknown `magic` → reject.
- fixed-width fields (`u8`/`u16`/`u32`) truncated at the frame end → reject
  (no OOB read).
- `string` / `blob` / `array` with a `u32` length or count past the end of the
  frame → reject (no OOB read); empty `string`, empty `array`.
- fixed-point round-trip: `700` ↔ `0.700` for the milli fields, `1270` ↔ `12.70`
  for the centi tps fields; a `0` milli-value is a valid explicit `0.0`.
- `bool` with a byte ∉ {0,1} → reject.
- partial frame one byte at a time → reassembly; multiple frames in one `read()`;
  frame split on any header/body boundary.
- `len > AGENT_PROTO_MAX_FRAME` (32 MiB) → reject before allocation; a
  `TURN` / `TOOL_RESULT` with inline images just under the cap → accepted.
- `SESSION` / `CONFIG` sub-command or key tag out of range → `ERR`.
- generic reply: each `u32 reply_tag` decoded; `MAP` / `ARR` with a count past
  the frame end → reject; empty `MAP` / `ARR`.
- large `stream_id` / `request_id`; empty payload after the header.

## 2. Turn lifecycle over the wire

**Connection / resume.** At startup: TCP connect → `HELLO` → `HELLO` reply. If
`session_parked=false`: the client sends `SESSION new` → its reply → `SESSION
list` (seed the completion cache). If `session_parked=true`: the client sends
`SESSION resume` (re-enter) or `SESSION new` (restart from scratch); in both
cases the reply + `SESSION list` follow. If a connection drops mid-turn, the
server treats it as `INTERRUPT` (flush + append turn-end, state → IDLE), then
**parks** the session; at the next `SESSION resume` the reply reports
`state=IDLE` and the current `ctx_used`, and the client redraws footer + prompt.
The client does not retry the connect on its own: if the server is unreachable it
prints the error and exits non-zero (the user relaunches the client and does
`SESSION resume`).

A turn with a tool call + mid-turn compaction. Interruptible points **A–E**;
`INTERRUPT` at any of them ends with the server appending the assistant-turn-end
marker (`agent_worker_append_assistant_turn_end`, ds4_agent.c:414) and reporting
the committed length in the terminal `STATUS.ctx_used`.

1. **Client** — linenoise returns the line; the client echoes the prompt
   locally. It sends `TURN{text, images}` (`images` non-empty only if the line
   references image files).
2. **Server** — the reader thread sets `w->cmd_text`; `worker_main` wakes, state
   → PREFILL. Sends `STATUS{PREFILL}` (this is the turn ack; a rejection would be
   `STATUS{IDLE, error="busy"}`). **Client** on the first `STATUS{PREFILL}`:
   hides the editor and takes the footer to busy (symmetric to the unhide on the
   terminal `STATUS`, step 12).
   - **A — pre-turn compaction** (`agent_worker_compact_if_needed`, 9880).
     If it runs: `STATUS{COMPACTING}` + `STREAM{SYSTEM, reason}` →
     `STREAM{SUMMARY}`* → `STATUS{state, new ctx_used}`. On success the server
     sends `DRAIN_REQUEST` and the client replies `DRAIN_REPLY` with the optional
     "bash jobs still running" note prepended (Risk 7, option b); the server
     appends it as a `tool` message before proceeding. `INTERRUPT` here ⇒ the
     `agent_err_is_interrupted` path (9883): clear interrupt, state IDLE,
     `STREAM{SYSTEM, "Compaction interrupted..."}`, `STATUS{IDLE, ctx_used}`
     (transcript unchanged).
3. **Server** — appends datetime/hints/user message to `w->transcript`.
   `ds4_session_sync` prefill; `worker_progress_cb` → `STATUS{PREFILL,
   prefill_done/total/tps}` coalesced.
   - **B — prefill** — `INTERRUPT` ⇒ `ds4_session_sync` returns
     `DS4_SESSION_SYNC_INTERRUPTED` (9995): `STREAM{SYSTEM, "Model reading
     interrupted..."}`, append turn-end, `STATUS{IDLE, ctx_used}`.
4. **Server** — generation loop, state GENERATING, `STATUS{generated, gen_tps,
   greedy_sampling}`. Each accepted token (`worker_finish_generated_token`) feeds
   the streaming state machine, which now calls `srv_emit(kind, bytes)` instead
   of painting; the server coalesces adjacent same-kind fragments and sends
   `STREAM{stream_id++, kind, text}`. `agent_stream_wants_greedy_sampling` stays
   server-side — **no per-token round-trip**.
   - **C — generation** — `INTERRUPT` ⇒ the loop breaks (10057),
     `agent_stream_text(finish)` flushes, `[tool call interrupted]` as
     `STREAM{NORMAL}` if a partial tool was open, append turn-end,
     `STREAM{SYSTEM, "Stopped by user"}`, `STATUS{IDLE, ctx_used}`.
5. **Server** — the model emits a complete DSML/GLM stanza; parser →
   `AGENT_DSML_DONE`. `agent_stream_text(finish)`, append turn-end. Sends
   `TOOL_CALLS{request_id=R, calls}` and **suspends** the turn loop in
   `worker_request_tool_exec` (new, modelled on 4944).
6. **Client** — runs the tools locally:
   - `edit` — reads the file, **exact + unique** match of `old`
     (`agent_edit_find_old_span` without the `[upto]` branch). Non-unique /
     missing ⇒ text `"Tool error: ..."`.
   - `read`/`more` — local `more_*` cursor; buckets on `ctx_size` from the
     `SESSION new`/`resume` reply; conservative byte caps trim to the reserve (no
     tokenize RPC).
   - `bash` — local fork/pipe/execl + monitor thread; returns a head/tail
     observation.
   - `view_image` — reads the bytes; adds them as an `image` in `TOOL_RESULT`.
   - **D — client-side tool execution** — a local Ctrl+C returns a partial
     observation or sends `INTERRUPT`. On `INTERRUPT` the server appends no
     observation; ends as in **C**.
7. **Client** — sends `TOOL_RESULT{request_id=R, text_parts, images}`.
8. **Server** — builds `agent_tool_observation` from `text_parts`; for images it
   calls `ds4_engine_vision_encode_memory` then `agent_tool_observation_build`
   (multimodal `user` role). `..._fits`:
   - doesn't fit ⇒ **mid-turn compaction** `agent_worker_compact "tool result
     would exceed context"` (10330): `STATUS{COMPACTING}` + `STREAM{SYSTEM,
     reason}` → `STREAM{SUMMARY}`* → `STATUS{state, ctx_used}`. On success the
     `DRAIN_REQUEST`/`DRAIN_REPLY` round follows (bash-jobs reminder prepended →
     appended as a `tool` message), as at point **A**.
     - **E — compaction** — `INTERRUPT` as in **A**.
   - re-check; if it still doesn't fit ⇒ the server truncates to an error
     observation (10346-10367).
9. **Server** — `agent_tool_observation_commit` grows `w->transcript`;
   `STATUS{ctx_used}`.
10. **Server** — `DRAIN_REQUEST`; suspends. **Client** replies `DRAIN_REPLY{text}`
    draining `agent_prompt_queue`. The server appends the non-empty text as a
    `user` message.
11. **Server** — loops back to step 3 for the next round. Eventually a round
    produces the final answer with no tool call (`!got_tool && !malformed_tool`,
    10292) ⇒ append turn-end, `STATUS{IDLE, ctx_used}` (terminal — reached from
    GENERATING).
12. **Client** — on a terminal `STATUS` (`IDLE`/`ERROR`/`STOPPED` reached from
    PREFILL/GENERATING/COMPACTING): unhide editor, reprint prompt, flush footer.

**"How much was committed"** — every `STATUS` carries `ctx_used =
w->transcript.len`. Since the server always appends the turn-end marker before
going idle (like the monolith), the terminal `STATUS.ctx_used` is exactly the
length the next turn extends from, and the client uses it as the base for
fit-context and for the footer's `ctx used`.

**Web / Chrome approval** — now entirely client-local (`ds4_web` links into the
client; `agent_web_confirm` prompts the local user inline). No protocol
messages; the server turn stays parked in the `TOOL_CALLS` handshake.

## 3. File / module plan

### `ds4_agent_proto.h` / `.c` (new)
Pure C99, **no `ds4.h`**, libc only. `enum agent_msg` (flat), `enum
agent_stream_kind`, mirror `enum agent_state`; growable `ap_buf`; big-endian
`ap_put_u8/u16/u32/str/blob/bool` + a bounds-checked `ap_reader` cursor with
`ap_get_*` (no varint, no f32); `ap_write_frame` / `ap_read_frame` for the
12-byte BE `{magic, type, len}` header with partial reassembly + the
`AGENT_PROTO_MAX_FRAME` check; a generic reply codec (`ap_put_reply` /
`ap_get_reply` over `OK/ERR/INT/STR/MAP/ARR`) for every `SESSION` / `CONFIG`
reply; one typed `ap_encode_<msg>` / `ap_decode_<msg>` pair per structured
message (`HELLO` reply, `SESSION new`/`resume` reply, `TURN`, `STREAM`,
`STATUS`, `TOOL_CALLS`, `TOOL_RESULT`, `DRAIN_REPLY`).

### `ds4_agent_utils.h` / `.c` (new, recommended)
Pure helpers duplicated by both binaries, copied verbatim minus `static`:
`agent_buf` + append/puts/take (ds4_agent.c:4329-4374), `xmalloc/xstrdup/
xstrndup/xrealloc` (437-469), `write_all` (506-521), `now_sec` (663-668),
`agent_input_buf` (522-552), the various `parse_int/parse_nonnegative_int/
parse_u64/parse_float_range/parse_power_percent/parse_steering_level` (553-635),
`agent_parse_bool_default` (6319-6327), `agent_mkdir_p` (4383-4399),
`set_nonblock` (10722-10728), `drain_wake_fd` (10743-10751). Links: libc only.

### `ds4_agent_server.c` (new)
**Copied from `ds4_agent.c`** (almost verbatim; everything `static` in one TU,
rename only on collision with a proto symbol):
- **CLI**: subset of `parse_options` (681-968) → `server_parse_options` — keeps
  engine/model/backend/gpu/`-t`, `--ctx` (server-only), think, mtp/dspark,
  ssd-streaming, `--power --warm-weights --prefill-chunk --simulate-used-memory`,
  dir-steering, distributed + TP (`ds4_dist_parse_cli_arg`, `ds4_tp_parse_cli_arg`
  + adopt/prepare/validate 930-956). **Adds** `--port` (7878), `--host`
  (127.0.0.1). **Drops** `-p/--prompt`, `--prompt-file`, `--prefix-file`,
  `--non-interactive`, `--raw*`, `-sys`, `--trace` (the server has its own),
  `--chdir`, `--edit-upto`. Structs `agent_generation_options`/`agent_config`
  (64-91) without the purely UI fields (`prompt*`, `trace_path`, `raw_prompt`
  aside, `chdir_path`, `non_interactive`, `edit_upto`); **keeps** `ctx_size`
  (from `--ctx`) and the generation fields (`n_predict`, `temperature*`,
  `top_p*`, `min_p*`, `seed`, `think_mode`, `prefix`) which are populated from
  `SESSION new`, not from the CLI, and rewritten on every `/new`. Helpers
  `default_backend/parse_backend/need_arg`,
  `agent_apply_model_sampling_defaults`, `effective_think_mode`.
- **Engine bootstrap**: body of `main` 13020-13115 (engine open / gpu config /
  TP leader) without chdir; a SIGINT handler that stops the listener.
- **Worker**: struct `agent_worker` (133-191) without the `web*`/linenoise
  fields; `agent_worker_init` (without `ds4_web_create`), `agent_worker_free`,
  `worker_main`, `worker_run_turn` (adapted — §3b), `worker_run_raw_prompt`
  (adapted), deferred save/compact/power (10541-10651), `worker_submit/
  interrupt/stop/consume/get_status/is_idle/is_initialized/status_power_locked` —
  `worker_consume` now drains an **in-order proto-message FIFO** filled under
  `w->mu` instead of the `w->out` bytes.
- **Transcript + images**: `ds4_tokens transcript`,
  `agent_worker_images_clear/append/fit_tokens` (470-505).
- **KV + persistence**: `agent_kv_*` (4410-4818), `agent_session_identity_sha`,
  `agent_default_cache_dir`, `agent_kv_path_for_sha`,
  `agent_worker_save_session_now/save_session/needs_save` (5180-5274),
  `agent_worker_switch_session` (6230),
  `agent_worker_find/delete/strip_session` (6034-6229), `agent_session_list_*`
  (5840-5943), `agent_session_title_*` (4463, 5284-5377),
  `agent_worker_reset_to_sysprompt` (5045),
  `agent_worker_clear_session_identity`. `agent_worker_list_sessions` reworked to
  fill the `SESSION list` reply `ARR` rows instead of `agent_publish`.
- **History rendering**: `agent_history_*` (5378-5839); the `agent_publish` sink
  goes to `STREAM{NORMAL}` frames.
- **Compaction**: `agent_worker_compact_transcript` (9384-9691),
  `agent_worker_compact/if_needed/should_compact`, helpers `agent_compact_*` +
  defines. Emits `STATUS{COMPACTING}` + `STREAM{SYSTEM, reason}` at entry,
  `STREAM{SUMMARY}` for the summary tokens, `STATUS{state, ctx_used}` at the end
  (no dedicated compaction messages). `agent_bash_jobs_compaction_observation`
  (9234) **dropped** server-side; the client recomposes the reminder text and
  prepends it to the `DRAIN_REPLY` that follows a successful compaction (Risk 7,
  option b). The server therefore sends `DRAIN_REQUEST` after every successful
  compaction too, not just after a tool result.
- **System + tools prompt**: macros
  `AGENT_TOOL_CONTRACTS`/`AGENT_EDIT_TARGET_RULE`,
  `agent_build_dsml_tools_prompt`/`agent_build_glm_tools_prompt`/
  `agent_build_tools_prompt` **without the `edit_upto` parameter**,
  `agent_append_system_prompt`, `agent_worker_build_system_tokens`,
  `agent_build_system_prompt_reminder`,
  `agent_worker_maybe_append_system_prompt_reminder`,
  `agent_worker_maybe_append_datetime_context`, hints (1362-1418),
  `agent_tool_syntax_for_engine`, `agent_worker_append_assistant_turn_end`,
  `agent_worker_effective_ctx_size`.
- **Streaming parser**: `agent_dsml_parser` + `agent_glm_tool_parse` + all the
  `agent_dsml_*` (289-319, 1684-2197), `agent_tool_call*`,
  `agent_dsml_marker_detector`. The **painting layer is replaced**: keep the
  control flow of `agent_stream_renderer`/`agent_stream_text`/
  `agent_stream_normal_byte`/`agent_stream_feed_dsml_byte`/`agent_stream_tool_events`
  (353-378, 3876-4261) but every `renderer_write*`/`agent_tool_viz_*` call
  becomes `srv_emit(kind, bytes)` with per-kind coalescing. The notices
  (3839-3872, 3948-3956, 4250-4252) emit `srv_emit(NORMAL, ...)`.
- **Sampling**: `agent_stream_wants_greedy_sampling`,
  `agent_stream_compaction_needs_lookahead`, `worker_sample_with_mode`,
  `worker_set_greedy_sampling`, `worker_finish_generated_token`/
  `worker_accept_generated_token`/`worker_force_generated_text` (these call the
  directive emitter).
- **Speculative / MTP**: inline block 10077-10101 verbatim.
- **Interrupt**: `worker_should_interrupt/clear_interrupt/cancel_session_cb/
  err_is_interrupted` (4304-4328), `worker_progress_cb` (4268).
- **Tool suspend/resume**: **new** `worker_request_tool_exec(w, &calls) →
  observation` (copy of the `worker_request_queued_user_drain` shape); the
  `agent_execute_tool_observation` call at 10320 is replaced by "emit
  `TOOL_CALLS`, wait, receive `TOOL_RESULT`, build the observation". Keep
  `agent_tool_observation*`, `agent_tool_observation_build/fits/commit`,
  `agent_vision_spans_free`. Vision: the `TURN.images[]` + `view_image` paths
  call `ds4_engine_vision_encode_memory` on the wire bytes.
- `agent_trace*` — the server's own trace file (`--trace`).

**Written fresh (~1300 lines):** TCP listener
(`socket/setsockopt(SO_REUSEADDR)/bind/listen/accept`, **one active connection at
a time**; a second connection while one is active → `ERR "busy"` + close);
**session state machine**: `{none}` –`SESSION new`→ `{active}` –disconnect→
`{parked}` –reconnect+`SESSION resume`→ `{active}` ; `{parked}`
–reconnect+`SESSION new`→ `{fresh active}` (frees the parked one). On disconnect:
if a turn is in progress it is closed as `INTERRUPT` (`worker_should_interrupt`
forced, flush, append turn-end), the reader/writer threads for that socket are
stopped, and `w` / engine / `ds4_session` / transcript / images **are kept**;
`ds4_session_free` is NOT called. SIGINT / shutdown free everything. Then: the
framed IO layer over `ds4_agent_proto`; the **reader thread** that decodes
frames and touches `w->cmd_text` / the latched bools / the handshake replies
under `w->mu` exactly where `run_agent` does today; the single **writer thread**
draining the FIFO into frames; the `srv_emit` (→ `STREAM`) / `server_set_status`
(→ `STATUS`) shims; the STATUS coalescer; `--port`/`--host` parsing.

**Link (Makefile):** `ds4_agent_server.o ds4_agent_proto.o ds4_agent_utils.o
ds4_help.o ds4_prompt_prefix.o ds4_kvstore.o ds4_gpu_args.o $(CORE_OBJS)`
(+ `$(METAL_LDLIBS)` on Darwin / `$(DS4_LINK_LIBS)` on Linux). **No linenoise,
no ds4_web.**

**`ds4_*` symbol families the server links** (from `ds4.h` / `ds4_kvstore.h` —
resolve the exact set by `make cpu` link errors, Risk 16): engine
lifecycle/introspection, think-mode, `ds4_session_*` lifecycle/sync/sample/eval,
power/steer, tokenize/chat, vision encode, `ds4_kvstore_*` low-level,
`ds4_prompt_prefix_*`, distributed/TP CLI + adopt/validate, gpu-args.

### `ds4_agent_client.c` (new)
**Copied from `ds4_agent.c`:**
- **Terminal / UI**: `agent_editor` + all `editor_*` (11166-12116), `linenoise*`
  calls, `agent_prompt_queue` (11031-11099), `build_prompt_text/
  build_status_text/build_footer_text/agent_progress_bar/agent_prefill_label/
  agent_power_status_suffix/agent_format_ctx_size/agent_format_welcome_banner/
  editor_write_welcome_banner/runtime_help` (10863-12172),
  `agent_next_prefill_label`, `agent_sigint_handler`, `stdout_is_tty`, user
  prompt echo (10863-10888), CPR + bracketed-paste helpers,
  `agent_yes_no_*`/`agent_prompt_yes_no*`, `agent_maybe_save_before_*` (reworked
  onto `SESSION save`), `agent_noninteractive_marker`,
  `agent_read_stdin_available`, `agent_switch_completion_callback` (5976,
  reworked: filters the in-memory session-list cache — see "written fresh" —
  instead of the local cache dir; Risk 15).
- **Painting**: `agent_token_renderer` + `agent_syntax` tables (208-253,
  2367-3421), all `renderer_*`, `agent_tool_visualizer` + `agent_tool_viz_*`
  (321-351, 3428-3838), `agent_tail_capture`, `agent_markdown_*`. **New**
  `client_apply_stream_fragment(kind, text)` feeding the renderer/visualiser
  the way `agent_stream_text`'s internals do today, driven by the `STREAM`
  frames. The "Reading <path> <start>:<max>..." line
  (`agent_tool_viz_render_read`, 3555) is produced client-side by the tool
  executor right before `read`.
- **Tool execution**: `agent_tool_read/more/write/list/edit/search/
  google_search/visit_page` (6553-8302), `agent_tool_view_image` (without the
  engine encode — sends the bytes), `agent_execute_tool_call`/
  `agent_execute_tool_observation` (reworked to build a `TOOL_RESULT`), the whole
  `agent_bash_*` subsystem (8303-8929) + `agent_bash_jobs_compaction_observation`
  (9234, for the post-compaction reminder prepended to `DRAIN_REPLY` — Risk 7
  option b), file helpers (`agent_open_regular_file/read_file_bytes/split_lines/
  replace_file/copy_file_xattrs/same_file_version/apply_file_splice/edit_result*/
  old_new_line_effect/line_for_offset/find_unique/memmem_simple`),
  **`agent_edit_find_old_span` exact-only** (§3b), the `more_*` cursor,
  fit-context helpers `agent_tool_result_reserve_tokens`,
  `agent_read_default_lines`, `agent_string_head/count_lines/write_temp_text`,
  the `AGENT_*` defines.
- **Web**: `agent_web_confirm/agent_web_log/agent_web_cancel` (4869-4913)
  reworked to prompt the **local** user inline (no worker-mutex handshake);
  `ds4_web_create` config.
- **CLI**: subset of `parse_options` → `client_parse_options`: keeps
  `-p/--prompt`, `--prompt-file`, `--prefix-file` (the client reads the file and
  forwards the raw text in `SESSION new`'s `prefix_file_text`),
  `--non-interactive`, `--raw*`, `-sys`, `--trace` (client trace), `--chdir`
  (client chdir — the tools run here). Keeps `--temp --top-p --min-p --seed
  --think --think-max --nothink --power --dir-steering-ffn --hints -n/--tokens`
  as **client flags that populate `SESSION new`**. **Adds** `--server host:port`
  (default `127.0.0.1:7878`). **Drops** every engine/backend/gpu/mtp/dspark/
  ssd/ctx/distributed/TP flag.
- `agent_trace*` — client trace.
- **Main loop**: `run_agent` and `run_agent_non_interactive` reworked onto the
  protocol: `worker_consume` → read frames; `worker_submit` → `TURN`;
  `worker_interrupt` → `INTERRUPT`; `worker_is_idle` → last `STATUS` state; slash
  commands → `SESSION <subcmd>`; `/steer`/`/power`/`/hints` → `CONFIG`. `main` →
  resolve `--server`, connect, `HELLO` + reply, `SESSION new` + reply, store
  the capabilities, run the loop.

**Written fresh (~800 lines):** socket connect + framed IO over `ds4_agent_proto`;
a receiver turning frames into editor/stream/tool calls; the **client
session-state mirror** (last `agent_status`, `ctx_size` from the `SESSION
new`/`resume` reply, `engine_is_glm`, `has_vision`, `backend_name`, `power`); the
**session-list cache** (rows from `SESSION list`) seeded with a `SESSION list` at
startup after the `SESSION new`/`resume` reply and refreshed after every `SESSION
save`/`del`/`switch`/`new` reply and after `/list` — used by the tab-completion
callback (Risk 15); disconnect handling (print error, exit non-zero);
`TURN.images[]` population for image-referencing prompts; the
`agent_prompt_queue` ↔ `DRAIN_REPLY` bridge.

**Link (Makefile):** `ds4_agent_client.o ds4_agent_proto.o ds4_agent_utils.o
ds4_help.o ds4_web.o linenoise.o` + `$(LDLIBS)`. **No `$(CORE_OBJS)`, no
`ds4.o`, no engine.** The client `#include "ds4.h"` only for the POD types
(`ds4_think_mode`, `ds4_vision_embedding`, `ds4_tokens`); any residual `ds4_*`
*call* in the copied tool code (tokenisation for fit-context) is replaced by a
byte-cap heuristic, with `agent_tool_observation_fits` server-authoritative as
the real gate (Risk 11).

## 3b. edit-upto removal

### Symbols / blocks NOT copied into either binary
The whole `[upto]` machinery in `ds4_agent.c` is left behind, by family:
- **edit-span forcer / preflight** — `agent_edit_upto_forcer*`,
  `agent_edit_old_*_for_upto`, `agent_edit_old_may_be_closing_tag`,
  `agent_edit_old_is_only_space`, `agent_span_has_nonspace`,
  `agent_find_unique_after`, `agent_preflight_edit_old`,
  `agent_stream_preflight_closed_param`, the `agent_stream_renderer`
  `tool_preflight_error` / `_msg` fields and every reference (`early_tool_error`
  stays, set only by the malformed-tool path).
- **CLI / config** — the `--edit-upto` option, `agent_config.edit_upto` and
  every read.
- **prompt** — `agent_tools_prompt_edit_upto` / `agent_glm_tools_prompt_edit_upto`;
  the tools-prompt builders lose the `edit_upto` parameter and always use the
  `..._edit_exact` strings.
- **worker** — the forcer branch in `worker_run_turn` (`worker_force_generated_text(w, "[upto]\n", ...)`).
- **rendering** — `md_code_highlight_upto`, `renderer_syntax_write_upto_marker`,
  `renderer_code_stream_set_upto_marker`, the `[upto]` highlight branch + call.
- **tests** — the three `test_agent_edit_upto_*` cases.

### The one behaviour change — `agent_edit_find_old_span` (client copy)
Signature without `allow_upto` and `anchored`. Body = exact + unique-once:
```
size_t old_len = strlen(old);
if (!agent_find_unique(data, len, old, old_len, match, "old text", err, err_len))
    return false;          /* not found, or found more than once */
*match_len = old_len;
return true;
```
Checked **client-side** at execution time. The callers `agent_tool_edit` (7846)
and the removed `agent_preflight_edit_old` lose the `allow_upto`/`anchored`
locals; the label `"anchored old/new replacement"` in `agent_apply_file_splice`
becomes `"old/new replacement"` (7884).

### Confirmation that nothing server-side references them
Tool execution (incl. `agent_edit_find_old_span` / `agent_tool_edit`) lives
**only on the client** — the server never links edit-span code. After removal the
server's copy of `worker_run_turn` has no forcer, no preflight, no
`tool_preflight_error`. Post-implementation gate: `grep -nE
'upto|preflight|edit_upto' ds4_agent_server.c` → empty; in `ds4_agent_client.c` →
only the exact `agent_edit_find_old_span` with no `[upto]` branch.

## 4. Makefile plan

Mirror the `ds4-agent` wiring; do not rewrite the content of existing lines.

**Binary targets** — Darwin block after line 105, Linux block after line 266
(the server mirrors `$(DS4_LINK)`, the client stays on `$(CC)` — it links no
CUDA):
```
ds4-agent-server: ds4_agent_server.o ds4_agent_proto.o ds4_agent_utils.o ds4_help.o ds4_prompt_prefix.o ds4_kvstore.o ds4_gpu_args.o $(CORE_OBJS)
	$(CC)/$(DS4_LINK) -o $@ $^ $(METAL_LDLIBS)/$(DS4_LINK_LIBS)
ds4-agent-client: ds4_agent_client.o ds4_agent_proto.o ds4_agent_utils.o ds4_help.o ds4_web.o linenoise.o
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
```

**Object rules** (after 351, mirroring 350-351 / 362-363):
`ds4_agent_proto.o`, `ds4_agent_utils.o`, `ds4_agent_server.o` (deps: ds4.h
ds4_ssd.h ds4_distributed.h ds4_tp.h ds4_help.h ds4_prompt_prefix.h ds4_kvstore.h
ds4_agent_proto.h ds4_agent_utils.h ds4_tool_text.h), `ds4_agent_client.o` (deps:
ds4.h ds4_help.h ds4_web.h linenoise.h ds4_agent_proto.h ds4_agent_utils.h
ds4_tool_text.h).

**CPU variant** (mirroring 392-393 and the `cpu:` recipes at 181-186 / 294-299):
`ds4_agent_server_cpu.o` with `-DDS4_NO_GPU`; in the two `cpu:` recipes add the
build of `ds4-agent-server` (from `_cpu.o` + `$(CPU_CORE_OBJS)` + link `$(CC)`)
and `ds4-agent-client` (shared client `.o` — no GPU code). This way the whole
split builds on the Linux dev box without nvcc.

**Aggregate target lists:** the new targets join the lists **in the task that
creates them** ("in `all` when you write them"): `ds4-agent-server`/
`ds4-agent-client` in line 77 (`all:` Darwin), the `cpu:` recipe (181/294),
`cuda-spark/cuda-generic/cuda/strix-halo` (209/212/220/223), `test-rocm -B`
(236-239). T12 = final consistency sweep.

**Test wiring** — the tests **must run on the dev box without nvcc or a model**
(see §Environment):
- `ds4_agent_proto_test` ← `tests/ds4_agent_proto_test.c` (pure, no `.c`
  include). Link: only `$(CC)` + `ds4_agent_proto.o ds4_agent_utils.o`. No
  `$(CORE_OBJS)`, no `$(DS4_LINK)`.
- `ds4_agent_client_test` ← `tests/ds4_agent_client_test.c` doing
  `#define DS4_AGENT_CLIENT_TEST[_NO_MAIN]` + `#include "../ds4_agent_client.c"`;
  drives the client against a **mock server** over `socketpair()` (scripted
  `HELLO` reply / `SESSION new` reply / `STREAM` / `STATUS` / `TOOL_CALLS` /
  `DRAIN_REQUEST` frames, incl. terminal `STATUS`); asserts rendered stdout,
  frames sent, tool execution, fit-context, `SESSION resume`. Link: only `$(CC)`
  + `ds4_agent_proto.o
  ds4_agent_utils.o ds4_web.o linenoise.o ds4_help.o` (the client links no
  engine). Reuses the terminal harness of `tests/ds4_agent_test.c:744`.
- `ds4_agent_server_test` ← `tests/ds4_agent_server_test.c` doing
  `#define DS4_AGENT_TEST[_NO_MAIN]` + `#include "../ds4_agent_server.c"`;
  exercises only the pure/control paths (proto dispatch, persistence helpers,
  `sysprompt.kv` + identity/title/listing, `agent_compact_make_prompt`, session
  state-machine transitions). **Single target, always CPU**: `.o` with
  `-DDS4_NO_GPU`, link `$(CC)` + `$(CPU_CORE_OBJS)` + `ds4_agent_proto.o
  ds4_agent_utils.o ds4_help.o ds4_prompt_prefix.o ds4_kvstore.o
  ds4_gpu_args_cpu.o`. No GPU/ROCm form — Layer 3 never exercises the
  engine-dependent sampling/prefill/sync/tokenize paths (same boundary as the
  monolith).
- **`ds4_agent_test_cpu` + `ds4_test_cpu`** — CPU forms of the monolith's two
  regression tests (not our code, but they must be runnable here because
  `ds4_help.*` is shared by `ds4-agent` **and** `ds4-server`, see §Scope).
  `ds4_agent_test_cpu`: `.o` with `-DDS4_NO_GPU` on `tests/ds4_agent_test.c` +
  link `$(CC)` + `$(CPU_CORE_OBJS)` + `ds4_help.o ds4_prompt_prefix.o ds4_web.o
  ds4_kvstore.o linenoise.o` + `$(LDLIBS)`. `ds4_test_cpu`: `.o` with
  `-DDS4_NO_GPU` on `tests/ds4_test.c` + link `$(CC)` + `$(CPU_CORE_OBJS)` +
  `ds4_help.o ds4_kvstore.o rax.o` + `$(LDLIBS)`. The existing `ds4_agent_test` /
  `ds4_test` forms and `make test` stay unchanged. Verified on the dev box: both
  build clean and pass (`ds4-agent tests: ok`, `ds4 tests: ok` + `--server`
  `server: OK`).
- **New `test-cpu` phony** (additive): prereqs `ds4_agent_proto_test
  ds4_agent_client_test ds4_agent_server_test ds4_agent_test_cpu ds4_test_cpu`;
  recipe runs all the `./...` (for `ds4_test_cpu` also `--server`). It is the
  local gate on the dev box.
- `test:` (703-708): add the three new tests to the prerequisites +
  `./ds4_agent_proto_test`, `./ds4_agent_client_test`, `./ds4_agent_server_test`
  to the recipe; same for `test-frontends` (699-701).
- `test-rocm` (236-246): add the three to the `-B` list and the `./...` to the
  recipe (the ROCm-host suite).
- `clean:` (772): add `ds4-agent-server ds4-agent-client` + the 5 tests
  (`ds4_agent_proto_test ds4_agent_client_test ds4_agent_server_test
  ds4_agent_test_cpu ds4_test_cpu`) + `ds4_agent_server_cpu.o
  ds4_agent_test_cpu.o ds4_test_cpu.o`.
- `ds4_tool_text.h` dependency group (757-758): add `ds4_agent_server.o
  ds4_agent_server_cpu.o ds4_agent_client.o`.

**Green locally (dev box):** `make cpu` (all binaries — proves the rebuild+relink
after the additive `ds4_help.*` touches) + **`make test-cpu`** green, without
nvcc or a model. **Green fully:** `make` + `make test` on macOS;
`make strix-halo` + `make test-rocm` on the ROCm host.

## 5. Task sequence (commit-sized)

| # | Unit | Test |
|---|---|---|
| T1 | `ds4_agent_proto.[ch]` — 12-byte BE `{magic, type, len}` framing + `AGENT_PROTO_MAX_FRAME` (32 MiB) check, big-endian `u8/u16/u32` + `string`/`blob`/`bool` codecs (no varint, no f32), bounds-checked `ap_reader`, `ap_write_frame`/`ap_read_frame` reassembly, `enum agent_msg` (flat) + `enum agent_stream_kind`, the generic reply codec (`OK/ERR/INT/STR/MAP/ARR`), typed encode/decode for the ~9 structured messages, fixed-point helpers for the milli/centi fields. `tests/ds4_agent_proto_test.c` + target. | Layer 1: `./ds4_agent_proto_test` — all §1 edge cases (byte-order magic, fixed-width truncation, fixed-point round-trip, oversize-frame reject, generic-reply tags, `SESSION`/`CONFIG` sub-command tags). |
| T2 | `ds4_agent_utils.[ch]` — copy the pure helpers. `.o` rule. | Layer 1: buffer-growth / int-parse / `mkdir_p` asserts in `ds4_agent_proto_test`. `make` green. |
| T3 | `ds4_agent_server.c` skeleton — `server_parse_options` (`--port --host --ctx` + engine flags) + `-h`/`--help` via new `DS4_HELP_AGENT_SERVER` in `ds4_help.h`/`ds4_help.c` (additive). **Bootstrap order (see "Server bootstrap / model-loading feedback" above — implement it this way from the start, do not do the engine load synchronously before the listener):** `server_listen()` first; `agent_worker` split into an engine-independent base init and a later engine-attach step (`engine_ready`/`boot_failed` flags); a background boot thread runs the engine-open sequence (copy of `main` 13020-13115 without chdir, unchanged) then attaches the engine; a heartbeat thread pushes periodic `STREAM{SYSTEM}` "Loading model... (Ns elapsed)" notices (fixed start timestamp, advancing next-tick threshold) until attached; the `accept()` loop starts immediately, concurrently. One active connection at a time, `HELLO` + reply (with `session_parked`, and the new `model_loading` bool set whenever the engine isn't attached yet — reply unconditionally, never gated on engine/worker state), session state machine `{none/active/parked}` + `SESSION new`/`resume` (its `worker_is_initialized` poll also checks `boot_failed` for a fast failure path), park on disconnect (turn in progress → `INTERRUPT`). Verify Risk 16 (`ds4_engine_power`). Target `ds4-agent-server` (Darwin + non-Darwin + `cpu:`) + `.o` rule + `all:`/`cpu:` lists. `tests/ds4_agent_server_test.c` skeleton. | Layer 3: `server_parse_options` flag mapping, `HELLO` reply packing (both engine-ready and still-loading cases), the boot-failed state transition (no real engine/threads needed), session state-machine transitions. `make cpu` green (incl. monolith rebuild after the `ds4_help.*` touch) + `make test-cpu`. |
| T4 | Server: `agent_worker` + lifecycle (init without web/linenoise, free, `worker_main`, submit/interrupt/stop/consume/get_status/idle), build system+tools prompt (`edit_upto` param already removed), `agent_worker_reset_to_sysprompt`, load/save `sysprompt.kv`, the `SESSION new`/`resume` reply + `ERR`. | Layer 3: `agent_build_tools_prompt` without `[upto]`; `agent_compact_make_prompt` text; `agent_session_identity_sha` stability. |
| T5 | Server: DSML/GLM parser + `srv_emit` fragment emitter replacing `renderer_*`/`agent_tool_viz_*` inside the `agent_stream_*` control flow; per-kind coalescing **plus a periodic force-flush** (`AGENT_STREAM_FLUSH_INTERVAL_SEC`, ~100ms — see "Server bootstrap / model-loading feedback" above: a kind change and `finish=true` are not enough triggers on their own, a long same-kind run must not sit buffered for the whole turn); `STREAM` emission (incl. `SUMMARY`/`SYSTEM` kinds). `agent_stream_wants_greedy_sampling` verbatim. | Layer 3: scripted token text (DSML + GLM, chunked) → assert the `(kind,text)` fragment sequence (mirrors `agent_test_stream_capture`, 7260); a real-sleep test asserting a flush fires mid-run from the timer alone, no kind change, before `finish`. No engine. |
| T6 | Server: `worker_run_turn`/`worker_run_raw_prompt`/deferred, compaction (`agent_worker_compact_transcript` + helpers), `worker_request_tool_exec` (new, on 4944), `agent_tool_observation_build/fits/commit`, `TURN.images[]` + `ds4_engine_vision_encode_memory`, `DRAIN_REQUEST`/`DRAIN_REPLY` (post-tool-result **and** post-successful-compaction, Risk 7 b). Wire `TURN`/`TOOL_CALLS`/`TOOL_RESULT`/`STREAM`/`STATUS` (folding turn-accepted/rejected/done + compaction begin/end into `STATUS`, compaction reason into `STREAM{SYSTEM}`); reader-thread dispatch; STATUS coalescer. | Layer 3: dispatch-mapping units (frame in → worker field), `STATUS` packing, terminal-`STATUS.ctx_used` accounting with a fake transcript. |
| T7 | Server: `SESSION save/new/switch/list/del/compact` sub-commands (generic replies) — `del` carries the `strip` flag, `switch` on the current sha covers history; `CONFIG get/set` for power/steer/hints. No tokenize RPC. | Layer 3: temp `cache_dir` with fake `.kv` (pattern from `tests/ds4_agent_test.c`) → `SESSION list` reply rows, `/switch` prefix resolution, `/del` + `strip`, title/identity SHA. |
| T8 | `ds4_agent_client.c` skeleton — `client_parse_options` (`--server`, UI flags, sampling/think/seed/power/steer/hints/`-sys`/`-n` → `SESSION new`) + `-h`/`--help` via new `DS4_HELP_AGENT_CLIENT` in `ds4_help.h`/`ds4_help.c` (additive), socket connect, `HELLO` + reply, print a one-time "model is still loading" notice if `model_loading` is true, `session_parked` branch → `SESSION resume` vs `SESSION new`, store capabilities, error+exit non-zero if the server is unreachable. **The `SESSION new`/`resume` reply wait must loop and dispatch (print) any interleaved `STREAM`/`STATUS` push before the reply arrives — never a single blocking `conn_recv`** (see "Server bootstrap / model-loading feedback" above; this is what makes the loading heartbeat and the existing sysprompt-build notices actually visible, and its absence is a real bug found in Layer 4, not a hypothetical). Target `ds4-agent-client` (Darwin + `cpu:` — no engine) + `.o` rule + `all:`/`cpu:` lists. `tests/ds4_agent_client_test.c` skeleton with a `socketpair` mock server, including a case with `model_loading=true` plus interleaved `STREAM{SYSTEM}` frames before the `SESSION` reply. | Layer 2 bootstrap: mock with `session_parked=false` → well-formed `SESSION new`; `session_parked=true` → `SESSION resume`; capabilities stored; interleaved pushes before the `SESSION`/`SESSION new` reply are printed, not treated as protocol errors. `make cpu` + `make test-cpu` green. |
| T9 | Client: render stack (`agent_token_renderer`, `agent_syntax`, `renderer_*`, `agent_tool_visualizer`, `agent_tool_viz_*`, `agent_tail_capture`) + `client_apply_stream_fragment`; editor/linenoise/footer/queue/banner/`runtime_help`. Wire `STREAM{NORMAL/THINK/TOOL_*}`→renderer, `STREAM{SYSTEM}`→`✦`, `STREAM{SUMMARY}`→banner+grey, `STATUS`→footer + terminal-state unhide. | Layer 2: the mock feeds scripted `STREAM`/`STATUS` (incl. `SUMMARY`/`SYSTEM` and a `COMPACTING`→`IDLE` sequence); capture client stdout; assert markdown / think-hide / tool-viz bytes and footer text against fixtures ported from `test_unicode_output_and_footer` / `test_markdown_literals` / `test_hint_rendering`. |
| T10 | Client: all tool execution — `agent_tool_read/more/write/list/edit/search/google_search/visit_page/view_image`, `agent_bash_*`, file helpers, **`agent_edit_find_old_span` exact-only** (§3b), `more_*` cursor, fit-context byte caps on `ctx_size` (no tokenize RPC). Wire `TOOL_CALLS`→execute→`TOOL_RESULT` (text + raw image bytes), `TURN.images[]` population, `DRAIN_REQUEST`→`DRAIN_REPLY` from `agent_prompt_queue` (with `agent_bash_jobs_compaction_observation` prepended if the `DRAIN_REQUEST` follows a successful compaction and there are live jobs), local `agent_web_confirm`. | Layer 2: the mock sends `TOOL_CALLS` (read/edit/bash/list/search/view_image); assert the `TOOL_RESULT` payload, the exact-unique `edit` failure text, bash job lifecycle, fit-context truncation, `TURN.images[]` emission, the bash-jobs reminder in the post-compaction `DRAIN_REPLY`. Reuse `test_atomic_file_tools` / `test_streaming_file_tools` / `test_background_jobs` retargeted. |
| T11 | Client: rework `run_agent` + `run_agent_non_interactive` onto the protocol — `TURN`/`INTERRUPT`/`STOP`, `SESSION <subcmd>` for slash commands, `CONFIG` for `/steer`/`/power`/`/hints`, `STATUS`-driven idle tracking, Ctrl+C / Ctrl+X / ESC, queue drain, welcome banner on `/new`, disconnect handling; session-list cache for tab-completion (seed `SESSION list` at startup, refresh after `SESSION save`/`del`/`switch`/`new` replies and `/list` — Risk 15). | Layer 2 e2e: script a full turn (`STATUS{PREFILL}` → `STREAM` → `TOOL_CALLS` → client executes → `STREAM` → terminal `STATUS`); assert the order of client sends (`TURN`, `TOOL_RESULT`, `DRAIN_REPLY`) and the final stdout; script a mid-stream `INTERRUPT`; script `/list` + check the completion cache updates; `/switch` tab-completion from the cache. |
| T12 | Makefile consistency sweep — `all:` (Darwin) / `cpu:` / `cuda-*` / `strix-halo` lists, `test:` / `test-frontends` / `test-rocm -B` lists, the `test-cpu` phony, `clean:`, `ds4_agent_server_cpu.o` + `ds4_agent_test_cpu.o` + `ds4_test_cpu.o`, `ds4_tool_text.h` dep group. | Local gate: `make cpu` + **`make test-cpu`** green (Layer 1-3 + client Layer-2 e2e + `ds4_agent_test_cpu` / `ds4_test_cpu` sentinels; no engine, no nvcc). |
| T13 | Docs — `AGENT.md` / `README` note the split binaries + `--server` / `--port` / `--host`, the **"SSH tunnel only" model** and the `--host 0.0.0.0` warning (no auth), the instance lock (Risk 18), reconnect/`SESSION resume`; a `QA_BEFORE_RELEASES.md` entry for the real-inference e2e (ROCm Strix Halo host). | Layer 4 (manual, with the user, **ROCm host** `make strix-halo`): build `sysprompt.kv`, a turn with a tool, a compaction, `/save`+`/switch`, a vision turn, an interrupt at each point A-E, **a reconnect** (kill the client mid-turn → relaunch → `SESSION resume` → session intact at IDLE). |

**End state before the real-inference e2e:** on the dev box `make cpu` +
**`make test-cpu`** green (the 3 new tests + `ds4_agent_test_cpu` /
`ds4_test_cpu` sentinels). Layer 1-3 and the client layer-2 mock e2e cover the
protocol codec, framing edge cases, rendering, tool execution, session-command
control logic, fit-context, dispatch mapping, state transitions. Only
engine-bound sampling / prefill / compaction-summary / KV-sync remain for
Layer 4 (ROCm host) — exactly the monolith's own boundary.

## 6. Risks / open questions

1. **STATUS cadence vs footer without polling.** **Chosen default:** `STATUS` on
   every state transition + every error, otherwise coalesced ~5 Hz (~10 Hz
   during PREFILL, keep-latest). The client footer keeps its
   `AGENT_STATUS_REDRAW_INTERVAL_SEC` timer (0.20, ds4_agent.c:10896) off the
   last `STATUS`. Tunable in Layer 4 if the prefill bar feels laggy.
2. **`/steer` and `/power` mid-turn.** `/power` is latched and applied between
   tokens — kept: `CONFIG set power` any time, `STATUS` echoes. `/steer` requires
   IDLE in the monolith (12821) — kept: the server rejects `CONFIG set steer`
   unless the state is IDLE.
3. **Distributed-route-ready wait.** `agent_worker_wait_distributed_route` (5138)
   blocks in `worker_main` before `initialized`. The client must not send
   `TURN` before the `SESSION new`/`resume` reply. During the wait the server
   emits periodic `STREAM{SYSTEM, "Waiting for distributed route: ..."}`, then
   the reply.
4. **How the client learns `ctx_size`.** Authoritative value = the `SESSION
   new`/`resume` reply's `ctx_size` (`agent_worker_effective_ctx_size` after the
   think-mode adjustment), re-received on every `/new`; `HELLO.ctx_size_cli` is
   an indicative pre-session hint only; `STATUS.ctx_size` carries the current
   value on every frame.
5. **linenoise multi-line + remote latency.** All the `editor_*` scroll-region /
   CPR maths stays client-local and unchanged. The only new stress: bursty
   `STREAM` frames feeding `editor_write_async` (12050) in large chunks — already
   handled. No protocol impact; flag a soak test over a lossy link.
6. **Trace files on both sides.** Two independent `--trace` files. Include
   `STREAM.stream_id` and `TOOL_CALLS.request_id` in both to correlate them.
7. **Bash-jobs-survive-compaction reminder.** **Decided: option (b).** The
   server sends `DRAIN_REQUEST` after every successful compaction (besides after
   a tool result). The client keeps `agent_bash_jobs_compaction_observation`
   (9234): when it has live bash jobs at the time of the post-compaction
   `DRAIN_REPLY` it prepends that text to the queue-drained content. The server
   appends the non-empty `DRAIN_REPLY` as a `tool` message (post-compaction) or a
   `user` message (post-tool-result), as it already does. No new protocol
   message.
8. **Image size + `server_image_cache`.** **Decided: skip for v1** (like the
   monolith agent — only `ds4_server.c` has the cache). The server calls
   `ds4_engine_vision_encode_memory` whenever needed. Images ride inline in the
   `TURN` / `TOOL_RESULT` frame, bounded by the single 32 MiB
   `AGENT_PROTO_MAX_FRAME` cap (no chunk sub-protocol); a frame past the cap is
   rejected and the sender must downscale. "Sessions containing images cannot be
   saved" (5204) is carried over → `SESSION save` reply `{ok=false, ...}`.
9. **Reconnect / parked session.** Decided — see §Architecture "Reconnect".
   Open (minor): an optional `--session-timeout` to auto-free a session parked
   too long, default infinite.
10. **`agent_publish` vs `STATUS` ordering.** The server MUST use a single writer
    thread draining one in-order FIFO (filled under `w->mu`) so a `COMPACTING`
    banner never reorders past its `STREAM{SUMMARY}` tokens.
11. **`ds4.h` in the client TU.** The client `#include`s `ds4.h` for the POD
    types (`ds4_vision_embedding` / `ds4_think_mode` / `ds4_tokens`) but links no
    `ds4.o`. T10 audit: any real `ds4_*` *call* in the copied tool code
    (tokenisation for fit-context, `ds4_token_*`) is dropped — see §1
    "Fit-context — no RPC".
12. **Non-interactive mode.** The `run_agent_non_interactive` markers and the
    stdin quiet-deadline batching move entirely client-side; the server is
    unaware. The `worker_request_queued_user_drain` handshake still works —
    `DRAIN_REPLY` carries the batched stdin text instead of the linenoise one.
13. **`usage()` / `ds4_help`.** Decided — see §Scope ("`ds4_help.h` + `ds4_help.c`:
    TOUCHED, additive"). Regression gate `make test-cpu` after every
    `ds4_help.*` touch.
14. **Protocol version negotiation.** `HELLO.proto_version` ≠ the server's ⇒
    `ERR` + close; the client prints a clear "rebuild both binaries".
15. **`/switch` tab-completion.** **Decided: it must always work.** The linenoise
    callback (`agent_switch_completion_callback`, 5976) is synchronous on the UI
    thread and cannot do network IO. The client sends a `SESSION list` at
    startup (right after the `SESSION new`/`resume` reply), keeps the rows in
    memory, and the callback filters that cache (the shas from the `SESSION list`
    reply rows). The cache is **refreshed** (a new `SESSION list`) after every
    store-mutating reply: `SESSION save`, `SESSION del` (incl. `strip`), `SESSION
    switch` (updates `last_used`), `SESSION new` (may auto-save), and after an
    explicit `/list`.
16. **`ds4_engine_power` / `ds4_engine_set_power`.** The monolith does not call
    them (power travels in `cfg.engine.power_percent` + `ds4_session_set_power`).
    The server links them only if `main` (13020-13115, copied) actually uses them
    for the bootstrap; otherwise they are dropped from the symbol list. Verify in
    T3 with `make cpu` on the dev box (link errors = missing symbols).
17. **`--power` on two sides.** The server has `--power` as an engine default;
    the client `--power` (via `SESSION new`'s `power_set`/`power`) is a session
    override that wins. Intended behaviour, but to be documented in `usage()`.
18. **Engine instance lock.** `AGENT.md`: "The instance lock is intentional".
    `ds4-agent-server` and a `ds4-agent` monolith (or another `ds4-server`) on
    the same host cannot run together — the lock prevents it, and that is the
    intended behaviour. To be noted in `usage()` / `AGENT.md`.
19. **Security — no auth.** Decided — see §Architecture "Security / network".
    T13: `usage()` + `AGENT.md` must warn that `--host 0.0.0.0` exposes
    inference + the on-disk KV store with no access control.
20. **Server bootstrap order / client feedback while the model loads.**
    Decided — see §Architecture "Server bootstrap / model-loading feedback";
    a first implementation got this wrong (engine opened synchronously before
    the listener existed) and had to be fixed after Layer 4 testing on real
    hardware. Accepted consequence: Ctrl+C during a long model load no longer
    kills the process instantly (§Architecture, same section) — to be
    documented in `AGENT.md`/`usage()` in T13.
21. **STREAM coalescing must force-flush on a timer, not just on a kind
    change / `finish=true`.** Decided — see §Architecture, same section as
    Risk 20. A first implementation only had the latter two triggers, so a
    long same-kind run (plain assistant text, `<think>` content — the common
    case) sat fully buffered until the turn ended: no visible token
    streaming at all until Layer 4 testing caught it.
22. **The client's `SESSION new`/`resume` reply wait must loop-tolerate
    interleaved `STREAM`/`STATUS`, not do a single blocking `conn_recv`.**
    Decided — see §Architecture, same section as Risk 20; this is what makes
    Risk 3's "periodic `STREAM{SYSTEM}`" notices and Risk 20's loading
    heartbeat actually visible instead of a hard "expected SESSION reply,
    got STREAM" failure, which is exactly what a first implementation did.

## End-to-end verification

- **Dev box build (Linux CPU):** `make cpu` green (existing `ds4` / `ds4-server`
  / `ds4-bench` / `ds4-eval` / `ds4-agent` + new `ds4-agent-server` /
  `ds4-agent-client` in the `cpu:` recipe) + **`make test-cpu`** green. No nvcc,
  no model.
- **Full builds (other machines):** `make` + `make test` on macOS;
  `make strix-halo` + `make test-rocm` on the ROCm Strix Halo host — all green,
  existing targets unchanged.
- **Automated tests (no model, via `make test-cpu`):** `ds4_agent_proto_test`
  (Layer 1 — codec + framing), `ds4_agent_client_test` (Layer 2 — client against
  the mock server: rendering, tool execution, fit-context, send order,
  interrupt, session commands, `SESSION resume`), `ds4_agent_server_test`
  (Layer 3 — always CPU: proto dispatch, persistence, `sysprompt.kv`,
  identity/title/listing, compaction prompt builder, session state machine),
  `ds4_agent_test_cpu` + `ds4_test_cpu` (regression sentinels for `ds4-agent` /
  `ds4-server` after the additive `ds4_help.*` touches).
- **Monolith regression:** `git status` = only new files + additive hunks
  (`Makefile`, `ds4_help.h`/`ds4_help.c`, + the 3 docs in T13); `ds4_agent.c`
  never present; `grep -c edit_upto ds4_agent.c` unchanged; `ds4_agent_test_cpu`
  + `ds4_test_cpu` green.
- **Layer 4 — real inference (with the user, ROCm Strix Halo gfx1151 host,
  `make strix-halo`):**
  1. On the ROCm host: `./ds4-agent-server -m <model> --ctx N` (binds
     `127.0.0.1:7878` by default).
  2. From the client: SSH tunnel `ssh -N -L 7878:127.0.0.1:7878 <host>` then
     `./ds4-agent-client --server 127.0.0.1:7878`.
  3. **Connect immediately, before the model has finished loading** (start the
     client right after the server, don't wait): confirm the one-time
     "Connected — the model is still loading..." line, then periodic
     "Loading model... (Ns elapsed)" notices with a **correctly incrementing**
     counter (2, 4, 6, 8s...) — not stuck at one value — until the sysprompt.kv
     build notice takes over and `SESSION new` completes normally.
  4. Verify: a turn with a tool call (`read` + `edit`) with **live, token-by-
     token streaming** (not the whole answer appearing at once at the end —
     regression-check the `AGENT_STREAM_FLUSH_INTERVAL_SEC` periodic flush,
     Risk 21) and visualisation identical to the monolith; prefill/generation
     footer; a compaction (fill the context) with a banner and a streamed
     summary; `/save` then `/switch` (with sha tab-completion); a vision turn
     (`view_image`); Ctrl+C at points A (pre-turn compaction), B (prefill), C
     (generation), D (tool execution), E (mid-turn compaction) — each time a
     clean return to IDLE with a well-formed transcript.
  5. **Reconnect:** kill the client during a turn, relaunch it → the `HELLO`
     reply's `session_parked=true` → `SESSION resume` → the conversation is
     intact, state IDLE, the interrupted turn was closed cleanly.
  6. **Ctrl+C on the server while the model is still loading:** confirm the
     listener/any connection drops immediately, and the process itself exits
     once the in-flight engine-open call returns (Risk 20 — expected, not a
     hang, but visibly delayed).
  7. Eyeball comparison of the output against `./ds4-agent` on the same
     machine/same prompt.
  8. For distributed/TP or changes that could touch the ROCm path: ask the user
     first (AGENT.md).
