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
    server_fill_hello_reply(&r, NULL, &c, false);
    CHECK(r.proto_version == AGENT_PROTO_VERSION);
    CHECK(r.ctx_size_cli == 32768);
    CHECK(r.power_percent == 100); /* unset -> 100 */
    CHECK(r.session_parked == false);
    CHECK(strlen(r.backend_name) > 0);
    CHECK(r.vocab_size == 0 && r.model_name[0] == '\0'); /* no engine */

    c.engine.power_percent = 42;
    server_fill_hello_reply(&r, NULL, &c, true);
    CHECK(r.power_percent == 42);
    CHECK(r.session_parked == true);

    /* survives a proto round trip */
    ap_buf body;
    ap_buf_init(&body);
    ap_encode_hello_reply(&body, &r);
    ap_reader rd;
    ap_reader_init(&rd, body.data, body.len);
    ap_hello_reply got;
    char err[128] = { 0 };
    CHECK(ap_decode_hello_reply(&rd, &got, err, sizeof(err)));
    CHECK(got.power_percent == 42 && got.session_parked && got.ctx_size_cli == 32768);
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

int main(void) {
    test_parse_defaults();
    test_parse_engine_flags();
    test_parse_server_endpoint();
    test_parse_backend_names();
    test_hello_reply_fill();
    test_session_state_machine();
    test_tools_prompt_exact_only();
    test_compact_make_prompt();
    test_identity_sha_stable();
    test_stream_dsml_tool_call_chunked();
    test_stream_glm_tool_call();
    test_stream_think_is_stripped();
    test_stream_normal_coalesced();
    test_stream_invalid_tool_call_notice();
    test_stream_greedy_sampling_flag();

    if (failures) {
        fprintf(stderr, "%d agent server test(s) failed\n", failures);
        return 1;
    }
    puts("agent server tests passed");
    return 0;
}
