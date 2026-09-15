/* Layer 3 tests for ds4-agent-server (skeleton scope, T3).
 *
 * Includes ds4_agent_server.c directly with its main() suppressed, and drives
 * the pure control paths: CLI flag mapping, HELLO-reply packing, and the
 * {none, active, parked} session state machine. No sockets, no model. The
 * worker / turn loop / persistence tests land in T4-T7.
 */

#define DS4_AGENT_SERVER_TEST_NO_MAIN
#include "../ds4_agent_server.c"

#include <stdio.h>
#include <string.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

static server_config parse(char **argv, int argc) {
    return server_parse_options(argc, argv);
}

static void test_parse_defaults(void) {
    char *argv[] = { "ds4-agent-server" };
    server_config c = parse(argv, 1);
    CHECK(strcmp(c.engine.model_path, "ds4flash.gguf") == 0);
    CHECK(c.gen.ctx_size == 100000);
    CHECK(strcmp(c.host, AGENT_PROTO_DEFAULT_HOST) == 0);
    CHECK(c.port == AGENT_PROTO_DEFAULT_PORT);
    CHECK(c.trace_path == NULL);
    CHECK(c.engine.power_percent == 0); /* unset -> HELLO reply reports 100 */
}

static void test_parse_engine_flags(void) {
    char *argv[] = {
        "ds4-agent-server",
        "-m", "model.gguf",
        "--ctx", "8192",
        "--cpu",
        "--power", "55",
        "-t", "6",
        "--mtp",
        "--dspark",
        "--vision", "vis.gguf",
        "--trace", "/tmp/srv.trace",
        "--host", "0.0.0.0",
        "--port", "9191",
    };
    server_config c = parse(argv, (int)(sizeof(argv) / sizeof(argv[0])));
    CHECK(strcmp(c.engine.model_path, "model.gguf") == 0);
    CHECK(c.gen.ctx_size == 8192);
    CHECK(c.engine.backend == DS4_BACKEND_CPU);
    CHECK(c.engine.power_percent == 55);
    CHECK(c.engine.n_threads == 6);
    CHECK(c.engine.glm_mtp == true);
    CHECK(c.engine.dspark == true);
    CHECK(strcmp(c.engine.vision_path, "vis.gguf") == 0);
    CHECK(strcmp(c.trace_path, "/tmp/srv.trace") == 0);
    CHECK(strcmp(c.host, "0.0.0.0") == 0);
    CHECK(c.port == 9191);
}

static void test_parse_server_endpoint(void) {
    char *a1[] = { "ds4-agent-server", "--server", "10.0.0.5:5555" };
    server_config c1 = parse(a1, 3);
    CHECK(strcmp(c1.host, "10.0.0.5") == 0 && c1.port == 5555);

    char *a2[] = { "ds4-agent-server", "--server", "namedhost" };
    server_config c2 = parse(a2, 3);
    CHECK(strcmp(c2.host, "namedhost") == 0);
    CHECK(c2.port == AGENT_PROTO_DEFAULT_PORT);

    char *a3[] = { "ds4-agent-server", "--server", ":6000" };
    server_config c3 = parse(a3, 3);
    CHECK(strcmp(c3.host, AGENT_PROTO_DEFAULT_HOST) == 0 && c3.port == 6000);
}

static void test_parse_backend_names(void) {
    char *a[] = { "ds4-agent-server", "--backend", "cpu" };
    CHECK(parse(a, 3).engine.backend == DS4_BACKEND_CPU);
    char *b[] = { "ds4-agent-server", "--backend", "metal" };
    CHECK(parse(b, 3).engine.backend == DS4_BACKEND_METAL);
}

static void test_hello_reply_fill(void) {
    server_config c = { 0 };
    c.engine.backend = DS4_BACKEND_CPU;
    c.gen.ctx_size = 32768;
    c.engine.power_percent = 0;

    ap_hello_reply r;
    server_fill_hello_reply(&r, NULL, &c, false, false);
    CHECK(r.proto_version == AGENT_PROTO_VERSION);
    CHECK(r.ctx_size_cli == 32768);
    CHECK(r.power_percent == 100); /* unset -> 100 */
    CHECK(r.session_parked == false);
    CHECK(r.model_loading == false);
    CHECK(strlen(r.backend_name) > 0);
    CHECK(r.vocab_size == 0 && r.model_name[0] == '\0'); /* no engine */

    c.engine.power_percent = 42;
    server_fill_hello_reply(&r, NULL, &c, true, false);
    CHECK(r.power_percent == 42);
    CHECK(r.session_parked == true);

    /* still loading: engine-derived fields stay zeroed regardless of a real
     * engine being passed, since the caller is expected to pass NULL/false
     * consistently -- here we just check the flag round-trips on its own. */
    server_fill_hello_reply(&r, NULL, &c, false, true);
    CHECK(r.model_loading == true);
    CHECK(r.vocab_size == 0 && r.model_name[0] == '\0');

    /* survives a proto round trip */
    c.engine.power_percent = 42;
    server_fill_hello_reply(&r, NULL, &c, true, false);
    ap_buf body;
    ap_buf_init(&body);
    ap_encode_hello_reply(&body, &r);
    ap_reader rd;
    ap_reader_init(&rd, body.data, body.len);
    ap_hello_reply got;
    char err[128] = { 0 };
    CHECK(ap_decode_hello_reply(&rd, &got, err, sizeof(err)));
    CHECK(got.power_percent == 42 && got.session_parked && got.ctx_size_cli == 32768);
    CHECK(got.model_loading == false);
    ap_buf_free(&body);
}

static void test_session_state_machine(void) {
    server_session s;
    server_session_clear(&s);
    CHECK(s.state == SERVER_SESSION_NONE);

    server_session_begin_new(&s, 4096);
    CHECK(s.state == SERVER_SESSION_ACTIVE);
    CHECK(s.ready.ctx_size == 4096);
    CHECK(s.ready.state == AGENT_STATE_IDLE);
    CHECK(s.ready.distributed_route_ready == true);

    /* resume is only valid from PARKED */
    CHECK(server_session_resume(&s) == false);
    CHECK(s.state == SERVER_SESSION_ACTIVE);

    /* a turn is running when the socket drops */
    s.turn_in_progress = true;
    server_session_park(&s);
    CHECK(s.state == SERVER_SESSION_PARKED);
    CHECK(s.turn_in_progress == true); /* cleaned up on resume, not on park */

    CHECK(server_session_resume(&s) == true);
    CHECK(s.state == SERVER_SESSION_ACTIVE);
    CHECK(s.turn_in_progress == false);
    CHECK(strstr(s.ready.note, "resumed") != NULL);

    /* SESSION new from a parked session discards it */
    server_session_park(&s);
    CHECK(s.state == SERVER_SESSION_PARKED);
    server_session_begin_new(&s, 8192);
    CHECK(s.state == SERVER_SESSION_ACTIVE && s.ready.ctx_size == 8192);

    /* park only acts on an active session */
    server_session_shutdown(&s);
    CHECK(s.state == SERVER_SESSION_NONE);
    server_session_park(&s);
    CHECK(s.state == SERVER_SESSION_NONE);
    CHECK(server_session_resume(&s) == false);
}

/* T4: the system/tool prompt builders drop the exact/anchored edit toggle and
 * always emit the exact-match strings (AGENT-SPLIT-PLAN.md section 3b). */
static void test_tools_prompt_exact_only(void) {
    char *dsml = agent_build_dsml_tools_prompt(false);
    char *glm  = agent_build_glm_tools_prompt(false);
    CHECK(dsml && glm);
    CHECK(strstr(dsml, "[upto]") == NULL);
    CHECK(strstr(glm,  "[upto]") == NULL);
    CHECK(strstr(dsml, "match exactly once") != NULL);
    CHECK(strstr(glm,  "match exactly once") != NULL);
    CHECK(strstr(dsml, "view_image") == NULL); /* no vision schema by default */

    char *dsml_v = agent_build_dsml_tools_prompt(true);
    char *glm_v  = agent_build_glm_tools_prompt(true);
    CHECK(strstr(dsml_v, "view_image") != NULL);
    CHECK(strstr(glm_v,  "view_image") != NULL);

    free(dsml);
    free(glm);
    free(dsml_v);
    free(glm_v);
}

/* T4: the compaction summary prompt is pure text (the rest of compaction is
 * T6). It forbids tool calls and echoes the reason only when one is given. */
static void test_compact_make_prompt(void) {
    char *p = agent_compact_make_prompt("tool result would exceed context");
    CHECK(strstr(p, "context compaction request") != NULL);
    CHECK(strstr(p, "do not call tools") != NULL);
    CHECK(strstr(p, "Compaction reason: tool result would exceed context") != NULL);
    free(p);

    char *q = agent_compact_make_prompt(NULL);
    CHECK(strstr(q, "Compaction reason:") == NULL);
    free(q);

    CHECK(agent_compact_summary_budget(80000) == AGENT_COMPACT_SUMMARY_MAX_TOKENS);
    CHECK(agent_compact_summary_budget(512) == 256); /* clamped up to the floor */
}

/* T4: a session's file name is SHA1(title || created_at_le64); it is stable
 * across resaves and changes only when the title or creation time changes. */
static void test_identity_sha_stable(void) {
    char a[41], b[41], c[41], d[41];
    agent_session_identity_sha("fix the parser", 1710000000ULL, a);
    agent_session_identity_sha("fix the parser", 1710000000ULL, b);
    agent_session_identity_sha("fix the parser", 1710000001ULL, c);
    agent_session_identity_sha("other title",    1710000000ULL, d);
    CHECK(strlen(a) == 40);
    CHECK(strcmp(a, b) == 0);
    CHECK(strcmp(a, c) != 0);
    CHECK(strcmp(a, d) != 0);
}

/* ---- T5: streaming classifier -> (kind, text) fragments ---------------- */

typedef struct {
    uint32_t id;
    uint32_t kind;
    char *text;
} cap_frag;

typedef struct {
    cap_frag *v;
    size_t n;
    size_t cap;
} cap_list;

static void cap_emit(void *ud, uint32_t sid, uint32_t kind,
                     const char *text, size_t len) {
    cap_list *c = ud;
    if (c->n == c->cap) {
        c->cap = c->cap ? c->cap * 2 : 8;
        c->v = realloc(c->v, c->cap * sizeof(c->v[0]));
    }
    char *copy = malloc(len + 1);
    memcpy(copy, text, len);
    copy[len] = '\0';
    c->v[c->n++] = (cap_frag){ .id = sid, .kind = kind, .text = copy };
}

static void cap_free(cap_list *c) {
    for (size_t i = 0; i < c->n; i++) free(c->v[i].text);
    free(c->v);
    memset(c, 0, sizeof(*c));
}

/* Feed scripted chunks through the classifier; returns the captured fragments. */
static void run_stream(agent_tool_syntax syntax, const char **chunks,
                       size_t n, cap_list *out) {
    agent_dsml_parser p = { .syntax = syntax, .state = AGENT_DSML_SEARCH };
    srv_stream s;
    srv_stream_init(&s, &p, syntax, NULL, cap_emit, out);
    for (size_t i = 0; i < n; i++)
        srv_stream_text(&s, chunks[i], strlen(chunks[i]), false);
    srv_stream_text(&s, NULL, 0, true);
    srv_stream_free(&s);
    agent_dsml_parser_free(&p);
}

/* Index of the first fragment of kind k at or after `from`; -1 if none. */
static int frag_of_kind(const cap_list *c, uint32_t k, size_t from) {
    for (size_t i = from; i < c->n; i++)
        if (c->v[i].kind == k) return (int)i;
    return -1;
}

static void test_stream_dsml_tool_call_chunked(void) {
    const char *chunks[] = {
        "hello <\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>\n",
        "<\xef\xbd\x9c" "DSML\xef\xbd\x9cinvoke name=\"read\">\n",
        "<\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter name=\"path\" string=\"true\">src/",
        "foo.c</\xef\xbd\x9c" "DSML\xef\xbd\x9cparameter>\n",
        "</\xef\xbd\x9c" "DSML\xef\xbd\x9cinvoke>\n</\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls>",
    };
    cap_list c = {0};
    run_stream(AGENT_TOOL_SYNTAX_DSML, chunks, 5, &c);

    CHECK(c.n >= 4);
    CHECK(c.v[0].kind == AGENT_STREAM_NORMAL && strcmp(c.v[0].text, "hello ") == 0);
    int tn = frag_of_kind(&c, AGENT_STREAM_TOOL_NAME, 0);
    int pn = frag_of_kind(&c, AGENT_STREAM_TOOL_PARAM_NAME, 0);
    int pv = frag_of_kind(&c, AGENT_STREAM_TOOL_PARAM_VALUE, 0);
    CHECK(tn >= 0 && strcmp(c.v[tn].text, "read") == 0);
    CHECK(pn > tn && strcmp(c.v[pn].text, "path") == 0);
    CHECK(pv > pn && strcmp(c.v[pv].text, "src/foo.c") == 0); /* coalesced, no close tag */
    /* raw DSML markup never crosses as NORMAL */
    for (size_t i = 0; i < c.n; i++)
        CHECK(strstr(c.v[i].text, "DSML") == NULL || c.v[i].kind != AGENT_STREAM_NORMAL);
    /* stream ids are strictly increasing */
    for (size_t i = 1; i < c.n; i++) CHECK(c.v[i].id > c.v[i - 1].id);
    cap_free(&c);
}

static void test_stream_glm_tool_call(void) {
    const char *chunks[] = {
        "prose <tool_call>list<arg_key>path</arg_key>",
        "<arg_value>.</arg_value></tool_call>",
    };
    cap_list c = {0};
    run_stream(AGENT_TOOL_SYNTAX_GLM, chunks, 2, &c);

    CHECK(c.v[0].kind == AGENT_STREAM_NORMAL && strcmp(c.v[0].text, "prose ") == 0);
    int tn = frag_of_kind(&c, AGENT_STREAM_TOOL_NAME, 0);
    int pn = frag_of_kind(&c, AGENT_STREAM_TOOL_PARAM_NAME, 0);
    int pv = frag_of_kind(&c, AGENT_STREAM_TOOL_PARAM_VALUE, 0);
    CHECK(tn >= 0 && strcmp(c.v[tn].text, "list") == 0);
    CHECK(pn > tn && strcmp(c.v[pn].text, "path") == 0);
    CHECK(pv > pn && strcmp(c.v[pv].text, ".") == 0);
    cap_free(&c);
}

static void test_stream_think_is_stripped(void) {
    const char *chunks[] = { "<think>sec", "ret</think>the answer" };
    cap_list c = {0};
    run_stream(AGENT_TOOL_SYNTAX_DSML, chunks, 2, &c);

    int th = frag_of_kind(&c, AGENT_STREAM_THINK, 0);
    CHECK(th >= 0 && strcmp(c.v[th].text, "secret") == 0);
    int nm = frag_of_kind(&c, AGENT_STREAM_NORMAL, 0);
    CHECK(nm > th);
    CHECK(strstr(c.v[nm].text, "the answer") != NULL);
    CHECK(c.v[nm].text[0] == '\n'); /* blank line after thinking */
    /* the literal think tags never appear in any fragment */
    for (size_t i = 0; i < c.n; i++) {
        CHECK(strstr(c.v[i].text, "<think>") == NULL);
        CHECK(strstr(c.v[i].text, "</think>") == NULL);
    }
    cap_free(&c);
}

static void test_stream_normal_coalesced(void) {
    const char *chunks[] = { "abc", "def", "ghi" };
    cap_list c = {0};
    run_stream(AGENT_TOOL_SYNTAX_DSML, chunks, 3, &c);
    CHECK(c.n == 1);
    CHECK(c.v[0].kind == AGENT_STREAM_NORMAL);
    CHECK(strcmp(c.v[0].text, "abcdefghi") == 0);
    cap_free(&c);
}

/* A same-kind run must not sit unflushed for the whole turn: past
 * AGENT_STREAM_FLUSH_INTERVAL_SEC the coalescer force-flushes even with no
 * kind change, so a slow/real generation loop still streams live instead of
 * dumping everything at finish=true (the Layer 4 bug on the ROCm host). */
static void test_stream_periodic_flush(void) {
    agent_dsml_parser p = { .syntax = AGENT_TOOL_SYNTAX_DSML, .state = AGENT_DSML_SEARCH };
    cap_list c = {0};
    srv_stream s;
    srv_stream_init(&s, &p, AGENT_TOOL_SYNTAX_DSML, NULL, cap_emit, &c);

    srv_stream_text(&s, "abc", 3, false);
    CHECK(c.n == 0); /* still buffered, interval not elapsed yet */

    struct timespec d = { 0, (long)(AGENT_STREAM_FLUSH_INTERVAL_SEC * 1.5 * 1e9) };
    nanosleep(&d, NULL);
    srv_stream_text(&s, "def", 3, false); /* same kind, but the interval elapsed */
    /* The proof: a flush already happened here, before finish and with no
     * kind change -- only AGENT_STREAM_FLUSH_INTERVAL_SEC having elapsed
     * explains it (without the fix this stays 0 until finish=true). */
    CHECK(c.n == 1);

    srv_stream_text(&s, NULL, 0, true); /* flushes any lookahead-held tail */
    CHECK(c.n >= 1 && c.n <= 2);
    for (size_t i = 0; i < c.n; i++) CHECK(c.v[i].kind == AGENT_STREAM_NORMAL);
    for (size_t i = 1; i < c.n; i++) CHECK(c.v[i].id > c.v[i - 1].id);
    /* the marker-lookahead holdback can shift bytes across the two fragments
     * (e.g. a trailing byte of "abc" held back and released together with
     * "def"), so only the concatenated content is asserted, not the split. */
    char joined[16] = {0};
    for (size_t i = 0; i < c.n; i++) strcat(joined, c.v[i].text);
    CHECK(strcmp(joined, "abcdef") == 0);

    cap_free(&c);
    srv_stream_free(&s);
    agent_dsml_parser_free(&p);
}

static void test_stream_invalid_tool_call_notice(void) {
    /* a DSML marker in plain assistant text, outside any tool_calls block */
    const char *chunks[] = { "text </\xef\xbd\x9c" "DSML\xef\xbd\x9c more" };
    cap_list c = {0};
    run_stream(AGENT_TOOL_SYNTAX_DSML, chunks, 1, &c);
    bool saw = false;
    for (size_t i = 0; i < c.n; i++)
        if (strstr(c.v[i].text, "[invalid tool call:") &&
            c.v[i].kind == AGENT_STREAM_NORMAL)
            saw = true;
    CHECK(saw);
    cap_free(&c);
}

static void test_stream_greedy_sampling_flag(void) {
    /* inside a DSML structural region the classifier asks for argmax sampling */
    agent_dsml_parser p = { .syntax = AGENT_TOOL_SYNTAX_DSML, .state = AGENT_DSML_SEARCH };
    cap_list c = {0};
    srv_stream s;
    srv_stream_init(&s, &p, AGENT_TOOL_SYNTAX_DSML, NULL, cap_emit, &c);
    const char *pre = "answer text ";
    srv_stream_text(&s, pre, strlen(pre), false);
    CHECK(!agent_stream_wants_greedy_sampling(&s));
    const char *open = "<\xef\xbd\x9c" "DSML\xef\xbd\x9ctool_calls><\xef\xbd\x9c" "DSML\xef\xbd\x9cinvoke name=\"read\">";
    srv_stream_text(&s, open, strlen(open), false);
    CHECK(agent_stream_wants_greedy_sampling(&s));
    srv_stream_free(&s);
    agent_dsml_parser_free(&p);
    cap_free(&c);
}

/* ---- T6a: server -> client FIFO, STATUS packing, reader dispatch ------- */

static server_config g_fake_cfg;

/* A worker with just the sync primitives + config the FIFO/dispatch paths
 * touch. No engine, no session, no worker thread. */
static void fake_worker(agent_worker *w) {
    memset(w, 0, sizeof(*w));
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cond, NULL);
    pthread_cond_init(&w->out_cond, NULL);
    memset(&g_fake_cfg, 0, sizeof(g_fake_cfg));
    g_fake_cfg.gen.ctx_size = 8192;
    w->cfg = &g_fake_cfg;
    w->wake_fd[0] = w->wake_fd[1] = -1;
    w->status.state = AGENT_WORKER_IDLE;
    w->initialized = true;
    w->out_active = true;
}

static void fake_worker_destroy(agent_worker *w) {
    srv_fifo_clear_locked(w);
    pthread_cond_destroy(&w->out_cond);
    pthread_cond_destroy(&w->cond);
    pthread_mutex_destroy(&w->mu);
}

static size_t fifo_count(const agent_worker *w) {
    size_t n = 0;
    for (const srv_msg *m = w->fifo_head; m; m = m->next) n++;
    return n;
}

/* A permanent boot failure (server_boot_engine_main) must unblock anything
 * waiting on engine_ready without needing real threads or an engine. */
static void test_worker_mark_boot_failed(void) {
    agent_worker w;
    fake_worker(&w);
    w.initialized = false;
    w.engine_ready = false;

    agent_worker_mark_boot_failed(&w, "boom");
    CHECK(w.boot_failed == true);
    CHECK(w.engine_ready == true);
    CHECK(w.status.state == AGENT_WORKER_ERROR);
    CHECK(strstr(w.status.error, "boom") != NULL);
    CHECK(worker_boot_failed(&w) == true);

    fake_worker_destroy(&w);
}

static void test_srv_status_packing(void) {
    agent_worker w;
    fake_worker(&w);
    w.status.state = AGENT_WORKER_GENERATING;
    w.status.generated = 42;
    w.status.gen_tps = 12.5;
    w.status.greedy_sampling = true;
    w.transcript.len = 1000;

    pthread_mutex_lock(&w.mu);
    srv_status_publish_locked(&w, true);
    pthread_mutex_unlock(&w.mu);
    CHECK(fifo_count(&w) == 1);
    CHECK(w.fifo_head->type == AGENT_MSG_STATUS);

    ap_reader r;
    ap_reader_init(&r, w.fifo_head->body.data, w.fifo_head->body.len);
    ap_status st;
    CHECK(ap_decode_status(&r, &st));
    CHECK(st.state == AGENT_STATE_GENERATING);
    CHECK(st.generated == 42);
    CHECK(st.greedy_sampling == true);
    CHECK(st.ctx_used == 1000);
    CHECK(st.ctx_size == 8192);
    CHECK(ap_centi_to_double(st.gen_tps) > 12.4 && ap_centi_to_double(st.gen_tps) < 12.6);

    fake_worker_destroy(&w);
}

static void test_srv_status_coalesce(void) {
    agent_worker w;
    fake_worker(&w);
    struct timespec pause = {0, 200 * 1000 * 1000}; /* > the 150ms rate gate */

    pthread_mutex_lock(&w.mu);
    srv_status_publish_locked(&w, true);
    srv_status_publish_locked(&w, true);
    CHECK(fifo_count(&w) == 2); /* forced STATUS is never merged */
    pthread_mutex_unlock(&w.mu);

    nanosleep(&pause, NULL);
    pthread_mutex_lock(&w.mu);
    w.status.generated = 3;
    srv_status_publish_locked(&w, false); /* rate gate passed -> non-forced node */
    CHECK(fifo_count(&w) == 3);

    w.status.generated = 9;
    srv_status_publish_locked(&w, false); /* tail is non-forced STATUS -> in place */
    CHECK(fifo_count(&w) == 3);

    srv_fifo_push_stream_locked(&w, AGENT_STREAM_NORMAL, "x", 1);
    CHECK(fifo_count(&w) == 4);

    srv_status_publish_locked(&w, false); /* tail is STREAM, rate-gated -> dropped */
    CHECK(fifo_count(&w) == 4);
    pthread_mutex_unlock(&w.mu);

    /* the refreshed non-forced STATUS (node index 2) carries the latest state */
    ap_reader r;
    ap_reader_init(&r, w.fifo_head->next->next->body.data,
                   w.fifo_head->next->next->body.len);
    ap_status st;
    CHECK(ap_decode_status(&r, &st));
    CHECK(st.generated == 9);

    fake_worker_destroy(&w);
}

static void test_srv_dispatch_turn(void) {
    agent_worker w;
    fake_worker(&w);
    srv_conn_ctx c = { .w = &w };

    ap_buf body;
    ap_buf_init(&body);
    ap_turn t = { .text = "do the thing", .text_len = 12 };
    ap_encode_turn(&body, &t);

    srv_dispatch_result dr = srv_dispatch_frame(&c, AGENT_MSG_TURN, &body);
    CHECK(dr == SRV_DISPATCH_CONTINUE);
    CHECK(w.cmd_text && strcmp(w.cmd_text, "do the thing") == 0);
    CHECK(w.status.state == AGENT_WORKER_PREFILL);
    /* the PREFILL ack STATUS was enqueued */
    CHECK(fifo_count(&w) >= 1);
    CHECK(w.fifo_tail->type == AGENT_MSG_STATUS);

    ap_buf_free(&body);
    free(w.cmd_text);
    fake_worker_destroy(&w);
}

static void test_srv_dispatch_stop_interrupt(void) {
    agent_worker w;
    fake_worker(&w);
    srv_conn_ctx c = { .w = &w };
    ap_buf empty;
    ap_buf_init(&empty);

    CHECK(srv_dispatch_frame(&c, AGENT_MSG_STOP, &empty) == SRV_DISPATCH_STOP);

    CHECK(srv_dispatch_frame(&c, AGENT_MSG_INTERRUPT, &empty) == SRV_DISPATCH_CONTINUE);
    CHECK(w.interrupt == true);

    CHECK(srv_dispatch_frame(&c, AGENT_MSG_HELLO, &empty) == SRV_DISPATCH_CLOSE);

    ap_buf_free(&empty);
    fake_worker_destroy(&w);
}

static void test_srv_publish_stream(void) {
    agent_worker w;
    fake_worker(&w);

    agent_publish_system_status(&w, "building sysprompt");
    agent_publish(&w, "plain output", 12);
    CHECK(fifo_count(&w) == 2);

    ap_reader r;
    ap_reader_init(&r, w.fifo_head->body.data, w.fifo_head->body.len);
    ap_stream s;
    CHECK(ap_decode_stream(&r, &s));
    CHECK(s.kind == AGENT_STREAM_SYSTEM);
    CHECK(s.text_len == strlen("building sysprompt"));
    CHECK(memcmp(s.text, "building sysprompt", s.text_len) == 0);
    CHECK(s.stream_id == 1);

    ap_reader_init(&r, w.fifo_head->next->body.data, w.fifo_head->next->body.len);
    CHECK(ap_decode_stream(&r, &s));
    CHECK(s.kind == AGENT_STREAM_NORMAL && s.stream_id == 2);

    /* nothing is enqueued once the client detaches */
    pthread_mutex_lock(&w.mu);
    w.out_active = false;
    srv_fifo_clear_locked(&w);
    pthread_mutex_unlock(&w.mu);
    agent_publish(&w, "dropped", 7);
    CHECK(fifo_count(&w) == 0);

    fake_worker_destroy(&w);
}

/* ---- T6b: turn-suspend handshakes (TOOL_CALLS/TOOL_RESULT, DRAIN) ------ */

#include <pthread.h>
#include <time.h>

static void short_sleep(void) {
    struct timespec d = {0, 5 * 1000 * 1000};
    nanosleep(&d, NULL);
}

static const srv_msg *fifo_find(const agent_worker *w, uint32_t type) {
    for (const srv_msg *m = w->fifo_head; m; m = m->next)
        if (m->type == type) return m;
    return NULL;
}

struct te_arg {
    agent_worker *w;
    const agent_tool_calls *calls;
    agent_tool_observation obs;
    bool intr;
    bool done;
};

static void *te_thread(void *p) {
    struct te_arg *a = p;
    a->obs = worker_request_tool_exec(a->w, a->calls, &a->intr);
    a->done = true;
    return NULL;
}

static agent_tool_calls one_call(const char *name, const char *arg, const char *val) {
    agent_tool_calls calls = {0};
    agent_tool_call c = {0};
    c.name = xstrdup(name);
    c.args = xmalloc(sizeof(c.args[0]));
    c.args[0].name = xstrdup(arg);
    c.args[0].value = xstrdup(val);
    c.args[0].is_string = true;
    c.argc = 1;
    c.argcap = 1;
    agent_tool_calls_push(&calls, &c);
    return calls;
}

static void test_tool_exec_handshake(void) {
    agent_worker w;
    fake_worker(&w);
    agent_tool_calls calls = one_call("read", "path", "src/x.c");

    struct te_arg a = { .w = &w, .calls = &calls };
    pthread_t th;
    pthread_create(&th, NULL, te_thread, &a);

    for (int i = 0; i < 400 && !fifo_find(&w, AGENT_MSG_TOOL_CALLS); i++)
        short_sleep();
    const srv_msg *m = fifo_find(&w, AGENT_MSG_TOOL_CALLS);
    CHECK(m != NULL);
    if (m) {
        ap_reader r;
        ap_reader_init(&r, m->body.data, m->body.len);
        ap_tool_calls tc;
        CHECK(ap_decode_tool_calls(&r, &tc));
        CHECK(tc.request_id == 1);
        CHECK(tc.call_count == 1);
        CHECK(tc.calls[0].name_len == 4 &&
              memcmp(tc.calls[0].name, "read", 4) == 0);
        CHECK(tc.calls[0].arg_count == 1);
        CHECK(memcmp(tc.calls[0].args[0].value, "src/x.c", 7) == 0);
    }

    ap_tool_result tr = {0};
    tr.request_id = 1;
    tr.text_parts[0].ptr = "file contents here\n";
    tr.text_parts[0].len = strlen("file contents here\n");
    tr.text_part_count = 1;
    worker_answer_tool_exec(&w, &tr);

    pthread_join(th, NULL);
    CHECK(a.done && !a.intr);
    CHECK(a.obs.part_count >= 1 && a.obs.parts[0].text);
    CHECK(strstr(a.obs.parts[0].text, "file contents here") != NULL);

    agent_tool_observation_free(&a.obs);
    agent_tool_calls_free(&calls);
    fake_worker_destroy(&w);
}

static void test_tool_exec_interrupt(void) {
    agent_worker w;
    fake_worker(&w);
    agent_tool_calls calls = one_call("bash", "command", "sleep 100");

    struct te_arg a = { .w = &w, .calls = &calls };
    pthread_t th;
    pthread_create(&th, NULL, te_thread, &a);

    for (int i = 0; i < 400; i++) {
        pthread_mutex_lock(&w.mu);
        bool pending = w.tool_exec_pending;
        pthread_mutex_unlock(&w.mu);
        if (pending) break;
        short_sleep();
    }
    worker_interrupt(&w);
    pthread_join(th, NULL);

    CHECK(a.done && a.intr);
    CHECK(a.obs.part_count == 1 && (!a.obs.parts[0].text || a.obs.parts[0].text[0] == '\0'));

    agent_tool_observation_free(&a.obs);
    agent_tool_calls_free(&calls);
    fake_worker_destroy(&w);
}

struct drain_arg { agent_worker *w; char *result; bool done; };
static void *drain_thread(void *p) {
    struct drain_arg *a = p;
    a->result = worker_request_queued_user_drain(a->w);
    a->done = true;
    return NULL;
}

static void test_drain_handshake(void) {
    agent_worker w;
    fake_worker(&w);

    struct drain_arg a = { .w = &w };
    pthread_t th;
    pthread_create(&th, NULL, drain_thread, &a);

    for (int i = 0; i < 400 && !fifo_find(&w, AGENT_MSG_DRAIN_REQUEST); i++)
        short_sleep();
    CHECK(fifo_find(&w, AGENT_MSG_DRAIN_REQUEST) != NULL);

    worker_answer_queued_user_drain(&w, xstrdup("keep going"));
    pthread_join(th, NULL);

    CHECK(a.done && a.result && strcmp(a.result, "keep going") == 0);
    free(a.result);
    fake_worker_destroy(&w);
}

static void test_turn_emit_stream(void) {
    agent_worker w;
    fake_worker(&w);
    srv_turn_emit(&w, 0, AGENT_STREAM_NORMAL, "hi", 2);
    const srv_msg *m = fifo_find(&w, AGENT_MSG_STREAM);
    CHECK(m != NULL);
    if (m) {
        ap_reader r;
        ap_reader_init(&r, m->body.data, m->body.len);
        ap_stream s;
        CHECK(ap_decode_stream(&r, &s));
        CHECK(s.kind == AGENT_STREAM_NORMAL && s.text_len == 2 &&
              memcmp(s.text, "hi", 2) == 0);
    }
    fake_worker_destroy(&w);
}

/* ---- T6c: compaction boundary math + SUMMARY/SYSTEM streams ----------- */

static void test_compact_tail_boundary(void) {
    /* transcript: [sys...][user@40 ...][user@75 ...], bottom 100, sys_len 20 */
    int v[100];
    for (int i = 0; i < 100; i++) v[i] = 1;
    const int USER = 7;
    v[40] = USER;
    v[75] = USER;
    ds4_tokens t = { .v = v, .len = 100, .cap = 100 };

    /* budget 30 -> target 70; the scan back finds the user turn at 75 */
    CHECK(agent_compact_tail_boundary(&t, 100, 20, 30, USER) == 75);
    /* budget 10 -> target 90, earliest 80; no user in [80,100) -> target 90 */
    CHECK(agent_compact_tail_boundary(&t, 100, 20, 10, USER) == 90);
    /* no user marker -> plain target */
    CHECK(agent_compact_tail_boundary(&t, 100, 20, 30, -1) == 70);
    /* target clamped to sys_len */
    CHECK(agent_compact_tail_boundary(&t, 100, 60, 80, -1) == 60);
}

static void test_compact_image_boundary(void) {
    ds4_vision_span sp = {0};
    sp.token_start = 100;
    sp.embedding.token_count = 50; /* image occupies [100, 150) */

    CHECK(agent_compact_image_boundary(&sp, 1, false, 120) == 100); /* inside -> snap to start */
    CHECK(agent_compact_image_boundary(&sp, 1, false, 200) == 200); /* after -> unchanged */
    CHECK(agent_compact_image_boundary(&sp, 1, false, 40) == 40);   /* before -> unchanged */
    CHECK(agent_compact_image_boundary(&sp, 1, true, 120) == 99);   /* glm wrapper token */
    CHECK(agent_compact_image_boundary(&sp, 0, false, 120) == 120); /* no images */
}

static void test_compact_summary_and_banner_streams(void) {
    agent_worker w;
    fake_worker(&w);

    agent_publishf_system_status(&w, "COMPACTING (%s): summarizing", "tool result would exceed context");
    srv_summary_emit(&w, "user wants X; ", 14);
    srv_summary_emit(&w, "file Y edited", 13);

    /* one SYSTEM banner, then the two SUMMARY chunks coalesced */
    CHECK(fifo_count(&w) == 2);

    ap_reader r;
    ap_reader_init(&r, w.fifo_head->body.data, w.fifo_head->body.len);
    ap_stream s;
    CHECK(ap_decode_stream(&r, &s));
    CHECK(s.kind == AGENT_STREAM_SYSTEM);
    CHECK(memmem(s.text, s.text_len,
                 "COMPACTING (tool result would exceed context): summarizing",
                 57) != NULL);

    ap_reader_init(&r, w.fifo_head->next->body.data, w.fifo_head->next->body.len);
    CHECK(ap_decode_stream(&r, &s));
    CHECK(s.kind == AGENT_STREAM_SUMMARY);
    CHECK(s.text_len == 27 &&
          memcmp(s.text, "user wants X; file Y edited", 27) == 0);

    fake_worker_destroy(&w);
}

/* ---- T7: CONFIG dispatch, session list, history helpers -------------- */

static ap_map decode_map_reply(const srv_msg *m) {
    ap_reader r;
    ap_reader_init(&r, m->body.data, m->body.len);
    uint32_t tag = 0;
    CHECK(ap_get_reply_tag(&r, &tag));
    CHECK(tag == AP_REPLY_MAP);
    ap_map mp = {0};
    CHECK(ap_get_map(&r, &mp));
    return mp;
}

static void test_config_power_hints(void) {
    agent_worker w;
    fake_worker(&w);
    srv_conn_ctx c = { .w = &w };
    ap_buf b;

    /* power set 55 */
    ap_buf_init(&b);
    ap_encode_config_set_u32(&b, AGENT_CONFIG_POWER, 55);
    CHECK(srv_dispatch_frame(&c, AGENT_MSG_CONFIG, &b) == SRV_DISPATCH_CONTINUE);
    ap_buf_free(&b);
    CHECK(w.power_requested && w.requested_power == 55);
    {
        ap_map mp = decode_map_reply(w.fifo_tail);
        CHECK(ap_map_get_bool(&mp, "ok", false) == true);
        CHECK(ap_map_get_u32(&mp, "value", 0) == 55);
    }

    /* power set 0 -> rejected */
    ap_buf_init(&b);
    ap_encode_config_set_u32(&b, AGENT_CONFIG_POWER, 0);
    srv_dispatch_frame(&c, AGENT_MSG_CONFIG, &b);
    ap_buf_free(&b);
    {
        ap_map mp = decode_map_reply(w.fifo_tail);
        CHECK(ap_map_get_bool(&mp, "ok", true) == false);
    }

    /* hints set true */
    ap_buf_init(&b);
    ap_encode_config_set_bool(&b, AGENT_CONFIG_HINTS, true);
    srv_dispatch_frame(&c, AGENT_MSG_CONFIG, &b);
    ap_buf_free(&b);
    CHECK(w.hints.enabled == true);
    {
        ap_map mp = decode_map_reply(w.fifo_tail);
        CHECK(ap_map_get_bool(&mp, "ok", false) == true);
        CHECK(ap_map_get_u32(&mp, "value", 9) == 1);
    }

    /* hints get */
    ap_buf_init(&b);
    ap_encode_config_get(&b, AGENT_CONFIG_HINTS);
    srv_dispatch_frame(&c, AGENT_MSG_CONFIG, &b);
    ap_buf_free(&b);
    {
        ap_map mp = decode_map_reply(w.fifo_tail);
        CHECK(ap_map_get_u32(&mp, "value", 9) == 1);
    }

    fake_worker_destroy(&w);
}

static void write_fake_kv(const char *dir, const char *sha40,
                          const char *user_prompt, uint32_t tokens,
                          uint64_t created, uint64_t last_used) {
    char path[600];
    snprintf(path, sizeof(path), "%s/%s.kv", dir, sha40);
    char text[512];
    int tn = snprintf(text, sizeof(text),
                      "<\xef\xbd\x9c" "User\xef\xbd\x9c>%s"
                      "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>ok",
                      user_prompt);
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, 1, 4, ds4_kvstore_reason_code("agent-session"),
                            0, tokens, 0, 8192, created, last_used, 64);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)tn);
    FILE *fp = fopen(path, "wb");
    CHECK(fp != NULL);
    if (!fp) return;
    fwrite(h, 1, sizeof(h), fp);
    fwrite(tb, 1, 4, fp);
    fwrite(text, 1, (size_t)tn, fp);
    uint8_t payload[64] = {0};
    fwrite(payload, 1, sizeof(payload), fp);
    fclose(fp);
}

static void test_session_list_encode(void) {
    char dir[] = "/tmp/ds4agentsrvtestXXXXXX";
    CHECK(mkdtemp(dir) != NULL);
    write_fake_kv(dir, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                  "fix the parser bug", 1200, 1710000000, 1710000500);
    write_fake_kv(dir, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
                  "write the readme", 340, 1710000100, 1710000100);

    agent_worker w;
    fake_worker(&w);
    w.engine = NULL;              /* skips the model-id filter */
    w.cache_dir = dir;
    memcpy(w.session_sha, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", 41);

    ap_buf body;
    ap_buf_init(&body);
    server_session_list_encode(&w, &body);

    ap_reader r;
    ap_reader_init(&r, body.data, body.len);
    uint32_t tag = 0;
    CHECK(ap_get_reply_tag(&r, &tag) && tag == AP_REPLY_ARR);
    uint32_t rows = 0;
    CHECK(ap_get_arr(&r, &rows));
    CHECK(rows == 2);

    bool saw_a = false, saw_b = false;
    for (uint32_t i = 0; i < rows; i++) {
        ap_map mp = {0};
        CHECK(ap_get_map(&r, &mp));
        char sha[64] = {0};
        ap_map_get_str(&mp, "sha", sha, sizeof(sha));
        char title[128] = {0};
        ap_map_get_str(&mp, "title", title, sizeof(title));
        if (strncmp(sha, "aaaa", 4) == 0) {
            saw_a = true;
            CHECK(strcmp(title, "fix the parser bug") == 0);
            CHECK(ap_map_get_u32(&mp, "tokens", 0) == 1200);
            CHECK(ap_map_get_bool(&mp, "is_current", true) == false);
        } else if (strncmp(sha, "bbbb", 4) == 0) {
            saw_b = true;
            CHECK(ap_map_get_bool(&mp, "is_current", false) == true);
        }
    }
    CHECK(saw_a && saw_b);

    ap_buf_free(&body);
    w.cache_dir = NULL; /* not owned */
    fake_worker_destroy(&w);

    char rm[700];
    snprintf(rm, sizeof(rm), "%s/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa.kv", dir);
    unlink(rm);
    snprintf(rm, sizeof(rm), "%s/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb.kv", dir);
    unlink(rm);
    rmdir(dir);
}

static void test_history_helpers(void) {
    const char *text =
        "sys stuff <\xef\xbd\x9c" "User\xef\xbd\x9c>hello there"
        "<\xef\xbd\x9c" "Assistant\xef\xbd\x9c>hi";
    size_t len = strlen(text);
    agent_history_mark mk = AGENT_HISTORY_MARK_NONE;
    size_t mlen = 0;
    const char *m = agent_history_next_marker(text, text + len, &mk, &mlen);
    CHECK(m != NULL && mk == AGENT_HISTORY_MARK_USER);
    const char *m2 = agent_history_next_marker(m + mlen, text + len, &mk, &mlen);
    CHECK(m2 != NULL && mk == AGENT_HISTORY_MARK_ASSISTANT);

    /* tool-result unwrap */
    const char *tr = "<tool_result>payload body</tool_result>";
    const char *p = tr, *e = tr + strlen(tr);
    CHECK(agent_history_tool_result_payload(&p, &e));
    CHECK((size_t)(e - p) == strlen("payload body") &&
          memcmp(p, "payload body", 12) == 0);
    CHECK(agent_history_is_tool_user(tr, tr + strlen(tr)));

    /* latest compaction summary between the fixed markers */
    const char *comp =
        "[ds4-agent compacted earlier conversation. Durable task-state summary follows.]\n"
        "the durable summary\n"
        "[End compacted summary. Recent conversation continues verbatim below.]\n";
    const char *ss = NULL, *se = NULL;
    CHECK(agent_history_latest_compaction_summary(comp, strlen(comp), &ss, &se));
    CHECK(se > ss && memcmp(ss, "the durable summary", 18) == 0);

    /* tail_start line clamp */
    const char *lines = "l1\nl2\nl3\nl4\nl5\n";
    bool trunc = false;
    const char *st = agent_history_tail_start(lines, lines + strlen(lines),
                                              2, 0, &trunc);
    CHECK(trunc && strcmp(st, "l4\nl5\n") == 0);
}

int main(void) {
    test_parse_defaults();
    test_parse_engine_flags();
    test_parse_server_endpoint();
    test_parse_backend_names();
    test_hello_reply_fill();
    test_worker_mark_boot_failed();
    test_session_state_machine();
    test_tools_prompt_exact_only();
    test_compact_make_prompt();
    test_identity_sha_stable();
    test_stream_dsml_tool_call_chunked();
    test_stream_glm_tool_call();
    test_stream_think_is_stripped();
    test_stream_normal_coalesced();
    test_stream_periodic_flush();
    test_stream_invalid_tool_call_notice();
    test_stream_greedy_sampling_flag();
    test_srv_status_packing();
    test_srv_status_coalesce();
    test_srv_dispatch_turn();
    test_srv_dispatch_stop_interrupt();
    test_srv_publish_stream();
    test_tool_exec_handshake();
    test_tool_exec_interrupt();
    test_drain_handshake();
    test_turn_emit_stream();
    test_compact_tail_boundary();
    test_compact_image_boundary();
    test_compact_summary_and_banner_streams();
    test_config_power_hints();
    test_session_list_encode();
    test_history_helpers();

    if (failures) {
        fprintf(stderr, "%d agent server test(s) failed\n", failures);
        return 1;
    }
    puts("agent server tests passed");
    return 0;
}
