# AGENT-TODO.md

This file track the execution and the details of the plan, splitted in numbered tasks.
The progress list must be updated at every task start/completion.
When following this plan you commit at least once at very step, and you don't need user confirmation in this case (the whole plan is approved)



## Plan progress
legend: [ ] todo  [>] in progress  [x] done

- [ ] 1. Protocol tests (Layer 1)
- [ ] 2. Port DSML/GLM parser + marker/think trackers into ds4_agent_server.c
- [ ] 3. Server skeleton + mock engine seam (Layer 3)
- [ ] 4. Tools framework (client-side): header + one file per group
- [ ] 5. Client skeleton + mock-server harness (Layer 2)
- [ ] 6. Makefile targets + build/test integration
- [ ] 7. Real inference e2e (only with user, RAM/CUDA host)


## Task 1 — Protocol tests (Layer 1)
Validate the wire protocol under `DS4_AGENT_TEST`. Encode/decode round trips for
every message type (incl. ATTACH_IMAGE), framing (4-byte length), varint edge
cases (0, max, multi-byte), string boundaries. Pure byte manipulation in
`agent/ds4_agent_proto.h`; no engine, no RAM.


## Task 2 — Port DSML/GLM parser + marker/think trackers
The server-side streaming parser, marker/think trackers, render-directive
producer, greedy boundaries live in `ds4_agent_server.c` (server-internal, per
the proto header note). Port from `ds4_agent.c` (`agent_stream_*`,
`agent_dsml_parser`, tool-call orchestration, `agent_tool_viz`-style render
kinds) and port the existing `test_agent_glm`/`test_agent_dsml` tests.
Most delicate piece; do it early.


## Task 3 — Server skeleton + mock engine seam (Layer 3)
`ds4_agent_server.c`: CLI (`parse_options` minus UI/tool opts + `--port`/`--host`),
socket accept (single client), framing read/dispatch, session lifecycle
(NEW_SESSION), status push, HELLO, TOKEN, TURN_PAUSED, TOOL_CALLS,
TOOL_RESULT/ATTACH_IMAGE, COMPACT, sessions/persistence (sysprompt.kv, save/
switch/del/strip/list), POWER, TOKENS->COUNT, INTERRUPT. Backend seam (function
pointers sample/eval/sync/argmax/tokenize/token_text) with a mock under
`#ifdef DS4_AGENT_TEST`: test state, persistence, compaction, sampling loop +
TURN_PAUSED, POWER, TOKENS, interrupt. Mock never ships (AGENT.md: no variants
behind flags; diagnostic test seams fine).


## Task 4 — Tools framework (client-side)
`agent/ds4_agent_tools.h` (tool structs, agent_buf, fit-context, client worker,
dispatch) + one file per group: `ds4_agent_tools_file.c` (read/more/write/list/
edit/search), `ds4_agent_tools_bash.c` (bash/bash_status/bash_stop),
`ds4_agent_tools_web.c` (google_search/visit_page + approval). Renamed to fit the
new structure (agent_tool_read -> agent_tools_file_read, ...). view_image is a
special case: client reads file bytes and sends ATTACH_IMAGE; no `w->images`.


## Task 5 — Client skeleton + mock-server harness (Layer 2)
`ds4_agent_client.c`: CLI (UI opts + `--server`), TCP client with framing, worker
thread owns socket + tool execution, UI thread owns editor (mirror worker/UI
split), editor copy, prompt queue, STATUS mirroring, token rendering (dumb
painter), tool execution, view_image->ATTACH_IMAGE, session commands, trace,
Ctrl+C->INTERRUPT. Harness acts as server feeding scripted TOKEN/TURN_PAUSED/
STATUS: verify rendered output, messages sent, tool execution + fit-context
(TOKENS->COUNT), session command rendering.


## Task 6 — Makefile targets + integration
Targets `ds4-agent-server` (server.o + ds4.o + ds4_kvstore.o + engine objects)
and `ds4-agent-client` (client.o + tools + ds4_web.o + linenoise.o). Do not
break existing targets. `make` green, `make test` green.


## Task 7 — Real inference e2e
Only with user assistance on a RAM/CUDA host (dev box can't run two engines).
Client can run on the RAM-bound box (TCP-only). Ask before CUDA/distributed tests.
