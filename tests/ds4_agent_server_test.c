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

    if (failures) {
        fprintf(stderr, "%d agent server test(s) failed\n", failures);
        return 1;
    }
    puts("agent server tests passed");
    return 0;
}
