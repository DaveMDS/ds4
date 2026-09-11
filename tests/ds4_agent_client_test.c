/* Layer 2 bootstrap tests for ds4-agent-client (T8 scope).
 *
 * Includes ds4_agent_client.c with its main() suppressed and drives
 * client_handshake against a mock server over a socketpair(): assert the
 * HELLO / SESSION frames the client sends, and that capabilities + the
 * authoritative ctx_size are stored. No sockets to a real server, no model.
 */

#define DS4_AGENT_CLIENT_TEST_NO_MAIN
#include "../ds4_agent_client.c"

#include <stdio.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

/* Write one framed message to fd. */
static void mock_send(int fd, uint32_t type, const ap_buf *body) {
    ap_buf f;
    ap_buf_init(&f);
    CHECK(ap_frame_encode(&f, type, body));
    CHECK(write(fd, f.data, f.len) == (ssize_t)f.len);
    ap_buf_free(&f);
}

/* Read every buffered byte from fd (non-blocking) into rx. */
static void mock_drain(int fd, ap_buf *rx) {
    char buf[8192];
    for (;;) {
        ssize_t n = recv(fd, buf, sizeof(buf), MSG_DONTWAIT);
        if (n <= 0) break;
        ap_buf_append(rx, buf, (size_t)n);
    }
}

static ap_hello_reply make_caps(bool parked) {
    ap_hello_reply r;
    memset(&r, 0, sizeof(r));
    r.proto_version = AGENT_PROTO_VERSION;
    r.engine_is_glm = true;
    r.has_vision = false;
    r.ctx_size_cli = 100000;
    snprintf(r.backend_name, sizeof(r.backend_name), "cuda");
    r.power_percent = 100;
    r.mtp_draft_tokens = 1;
    snprintf(r.model_name, sizeof(r.model_name), "deepseek-v4-flash");
    r.vocab_size = 129280;
    r.session_parked = parked;
    return r;
}

static ap_session_ready make_ready(void) {
    ap_session_ready s;
    memset(&s, 0, sizeof(s));
    s.ctx_used = 512;
    s.ctx_size = 98304;
    s.distributed_route_ready = true;
    s.state = AGENT_STATE_IDLE;
    snprintf(s.note, sizeof(s.note), "session created");
    return s;
}

static void test_handshake_new_session(void) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

    /* stage the mock server's two replies before running the client */
    {
        ap_hello_reply caps = make_caps(false);
        ap_buf b; ap_buf_init(&b);
        ap_encode_hello_reply(&b, &caps);
        mock_send(sv[1], AGENT_MSG_HELLO, &b);
        ap_buf_free(&b);

        ap_session_ready rd = make_ready();
        ap_buf_init(&b);
        ap_encode_session_ready(&b, &rd);
        mock_send(sv[1], AGENT_MSG_SESSION, &b);
        ap_buf_free(&b);
    }

    client_config cfg = {
        .server_port = 7878,
        .system = "sys text",
        .n_predict = 4096,
        .temperature = 0.7f, .temperature_set = true,
        .top_p = DS4_DEFAULT_TOP_P, .min_p = DS4_DEFAULT_MIN_P,
        .think_mode = DS4_THINK_HIGH,
        .hints_enabled = true,
    };
    snprintf(cfg.server_host, sizeof(cfg.server_host), "127.0.0.1");

    client_conn co;
    conn_init(&co, sv[0], NULL);

    client_session sess;
    char err[256] = {0};
    CHECK(client_handshake(&co, &cfg, &sess, err, sizeof(err)));
    CHECK(err[0] == '\0');
    CHECK(sess.resumed == false);
    CHECK(sess.caps.engine_is_glm == true);
    CHECK(strcmp(sess.caps.model_name, "deepseek-v4-flash") == 0);
    CHECK(sess.ready.ctx_size == 98304); /* authoritative, not the HELLO hint */
    CHECK(sess.ready.ctx_used == 512);

    /* inspect what the client sent */
    ap_buf rx; ap_buf_init(&rx);
    mock_drain(sv[1], &rx);

    uint32_t type = 0;
    ap_buf payload; ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_HELLO);
    {
        ap_reader r; ap_reader_init(&r, payload.data, payload.len);
        ap_hello h;
        CHECK(ap_decode_hello(&r, &h));
        CHECK(h.proto_version == AGENT_PROTO_VERSION);
        CHECK(strncmp(h.client_version, "ds4-agent-client/", 17) == 0);
    }

    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_SESSION);
    {
        ap_reader r; ap_reader_init(&r, payload.data, payload.len);
        ap_session_msg m;
        CHECK(ap_decode_session(&r, &m));
        CHECK(m.subcmd_known && m.subcmd == AGENT_SESSION_NEW);
        CHECK(m.new_args.temperature_set);
        CHECK(m.new_args.temperature == ap_milli_from_double(0.7));
        CHECK(m.new_args.think_mode == (uint32_t)DS4_THINK_HIGH);
        CHECK(m.new_args.n_predict == 4096);
        CHECK(m.new_args.hints_enabled == true);
        CHECK(m.new_args.sys_text_len == strlen("sys text"));
        CHECK(memcmp(m.new_args.sys_text, "sys text", 8) == 0);
    }

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    conn_free(&co); /* closes sv[0] */
    close(sv[1]);
}

static void test_handshake_resume(void) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

    {
        ap_hello_reply caps = make_caps(true); /* parked */
        ap_buf b; ap_buf_init(&b);
        ap_encode_hello_reply(&b, &caps);
        mock_send(sv[1], AGENT_MSG_HELLO, &b);
        ap_buf_free(&b);

        ap_session_ready rd = make_ready();
        ap_buf_init(&b);
        ap_encode_session_ready(&b, &rd);
        mock_send(sv[1], AGENT_MSG_SESSION, &b);
        ap_buf_free(&b);
    }

    client_config cfg = { .server_port = 7878, .think_mode = DS4_THINK_HIGH };
    snprintf(cfg.server_host, sizeof(cfg.server_host), "127.0.0.1");

    client_conn co;
    conn_init(&co, sv[0], NULL);
    client_session sess;
    char err[256] = {0};
    CHECK(client_handshake(&co, &cfg, &sess, err, sizeof(err)));
    CHECK(sess.resumed == true);

    ap_buf rx; ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    uint32_t type = 0;
    ap_buf payload; ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_HELLO);
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_SESSION);
    {
        ap_reader r; ap_reader_init(&r, payload.data, payload.len);
        ap_session_msg m;
        CHECK(ap_decode_session(&r, &m));
        CHECK(m.subcmd_known && m.subcmd == AGENT_SESSION_RESUME);
    }

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    conn_free(&co);
    close(sv[1]);
}

static void test_handshake_resume_falls_back_to_new(void) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

    {
        ap_hello_reply caps = make_caps(true); /* parked */
        ap_buf b; ap_buf_init(&b);
        ap_encode_hello_reply(&b, &caps);
        mock_send(sv[1], AGENT_MSG_HELLO, &b);
        ap_buf_free(&b);

        /* resume rejected */
        ap_buf_init(&b);
        ap_put_reply_err(&b, "no parked session to resume");
        mock_send(sv[1], AGENT_MSG_SESSION, &b);
        ap_buf_free(&b);

        /* the fallback SESSION new succeeds */
        ap_session_ready rd = make_ready();
        ap_buf_init(&b);
        ap_encode_session_ready(&b, &rd);
        mock_send(sv[1], AGENT_MSG_SESSION, &b);
        ap_buf_free(&b);
    }

    client_config cfg = { .server_port = 7878, .think_mode = DS4_THINK_HIGH };
    snprintf(cfg.server_host, sizeof(cfg.server_host), "127.0.0.1");
    client_conn co;
    conn_init(&co, sv[0], NULL);
    client_session sess;
    char err[256] = {0};
    CHECK(client_handshake(&co, &cfg, &sess, err, sizeof(err)));
    CHECK(sess.resumed == false); /* fell back to a fresh session */
    CHECK(sess.ready.ctx_size == 98304);

    ap_buf rx; ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    uint32_t type = 0;
    ap_buf payload; ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK); /* HELLO */
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK); /* resume */
    {
        ap_reader r; ap_reader_init(&r, payload.data, payload.len);
        ap_session_msg m;
        CHECK(ap_decode_session(&r, &m) && m.subcmd == AGENT_SESSION_RESUME);
    }
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK); /* new */
    {
        ap_reader r; ap_reader_init(&r, payload.data, payload.len);
        ap_session_msg m;
        CHECK(ap_decode_session(&r, &m) && m.subcmd == AGENT_SESSION_NEW);
    }

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    conn_free(&co);
    close(sv[1]);
}

static void test_handshake_version_mismatch(void) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    {
        ap_hello_reply caps = make_caps(false);
        caps.proto_version = AGENT_PROTO_VERSION + 1;
        ap_buf b; ap_buf_init(&b);
        ap_encode_hello_reply(&b, &caps);
        mock_send(sv[1], AGENT_MSG_HELLO, &b);
        ap_buf_free(&b);
    }
    client_config cfg = { .server_port = 7878, .think_mode = DS4_THINK_HIGH };
    snprintf(cfg.server_host, sizeof(cfg.server_host), "127.0.0.1");
    client_conn co;
    conn_init(&co, sv[0], NULL);
    client_session sess;
    char err[256] = {0};
    CHECK(client_handshake(&co, &cfg, &sess, err, sizeof(err)) == false);
    CHECK(strstr(err, "rebuild both binaries") != NULL);
    conn_free(&co);
    close(sv[1]);
}

static void test_parse_server_endpoint(void) {
    char *a1[] = { "ds4-agent-client", "--server", "10.0.0.5:9000" };
    client_config c1 = client_parse_options(3, a1);
    CHECK(strcmp(c1.server_host, "10.0.0.5") == 0 && c1.server_port == 9000);

    char *a2[] = { "ds4-agent-client", "--server", "somehost" };
    client_config c2 = client_parse_options(3, a2);
    CHECK(strcmp(c2.server_host, "somehost") == 0 &&
          c2.server_port == AGENT_PROTO_DEFAULT_PORT);

    char *a3[] = { "ds4-agent-client", "--think-max", "--seed", "42",
                   "--hints", "on", "-n", "1234" };
    client_config c3 = client_parse_options(8, a3);
    CHECK(c3.think_mode == DS4_THINK_MAX);
    CHECK(c3.seed == 42);
    CHECK(c3.hints_enabled == true);
    CHECK(c3.n_predict == 1234);
}

/* ---- T9: render stack + client_apply_stream_fragment ------------------ */

static char g_cap[65536];
static size_t g_cap_len;
static void cap_sink(const char *s, size_t n) {
    if (g_cap_len + n < sizeof(g_cap)) {
        memcpy(g_cap + g_cap_len, s, n);
        g_cap_len += n;
        g_cap[g_cap_len] = '\0';
    }
}

static void render_reset(agent_token_renderer *r, agent_stream_renderer *sr,
                         bool color) {
    g_cap_len = 0;
    g_cap[0] = '\0';
    g_render_sink = cap_sink;
    memset(r, 0, sizeof(*r));
    r->format_thinking = true;
    r->format_markdown = color;
    r->use_color = color;
    r->last_output_newline = true;
    client_stream_renderer_init(sr, r, 100000);
}

static void frag(agent_stream_renderer *sr, uint32_t kind, const char *t) {
    client_apply_stream_fragment(sr, kind, t, strlen(t));
}

static void test_render_normal_and_think(void) {
    agent_token_renderer r;
    agent_stream_renderer sr;
    render_reset(&r, &sr, false); /* colour off -> markdown literals pass through */

    frag(&sr, AGENT_STREAM_THINK, "planning the fix");
    frag(&sr, AGENT_STREAM_NORMAL, "Here is the answer.");
    client_stream_renderer_finish(&sr);

    CHECK(strstr(g_cap, "planning the fix") != NULL);
    CHECK(strstr(g_cap, "Here is the answer.") != NULL);
    /* no raw think tags ever */
    CHECK(strstr(g_cap, "<think>") == NULL);
}

static void test_render_system_notice(void) {
    agent_token_renderer r;
    agent_stream_renderer sr;
    render_reset(&r, &sr, false);
    frag(&sr, AGENT_STREAM_SYSTEM, "Updating system prompt cache");
    CHECK(strstr(g_cap, "\xe2\x9c\xa6 Updating system prompt cache") != NULL);
}

static void test_render_tool_read_viz(void) {
    agent_token_renderer r;
    agent_stream_renderer sr;
    render_reset(&r, &sr, false);

    frag(&sr, AGENT_STREAM_TOOL_NAME, "read");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_NAME, "path");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_VALUE, "src/");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_VALUE, "foo.c");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_NAME, "max_lines");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_VALUE, "80");
    /* a NORMAL fragment (the assistant's next text) closes the tool viz */
    frag(&sr, AGENT_STREAM_NORMAL, "done.");
    client_stream_renderer_finish(&sr);

    CHECK(strstr(g_cap, "Reading ") != NULL);
    CHECK(strstr(g_cap, "src/foo.c") != NULL);
    CHECK(strstr(g_cap, ":80") != NULL);   /* the requested max_lines */
    CHECK(strstr(g_cap, "done.") != NULL);
}

static void test_render_tool_edit_diff(void) {
    agent_token_renderer r;
    agent_stream_renderer sr;
    render_reset(&r, &sr, false);

    frag(&sr, AGENT_STREAM_TOOL_NAME, "edit");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_NAME, "path");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_VALUE, "a.c");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_NAME, "old");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_VALUE, "int x = 1;\n");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_NAME, "new");
    frag(&sr, AGENT_STREAM_TOOL_PARAM_VALUE, "int x = 2;\n");
    frag(&sr, AGENT_STREAM_NORMAL, "ok");
    client_stream_renderer_finish(&sr);

    CHECK(strstr(g_cap, "edit ") != NULL);
    CHECK(strstr(g_cap, "a.c") != NULL);
    CHECK(strstr(g_cap, "- int x = 1;") != NULL); /* diff-old prefix */
    CHECK(strstr(g_cap, "+ int x = 2;") != NULL); /* diff-new prefix */
}

static void test_build_status_text(void) {
    ap_status w = {0};
    w.state = AGENT_STATE_GENERATING;
    w.generated = 128;
    w.gen_tps = ap_centi_from_double(24.5);
    w.ctx_used = 12000;
    w.ctx_size = 100000;
    w.power_percent = 100;
    agent_status st;
    client_status_from_wire(&w, &st);

    char buf[512];
    build_status_text(&st, buf, sizeof(buf));
    CHECK(strstr(buf, "ctx 12k/100k") != NULL);
    CHECK(strstr(buf, "generation 128 tokens") != NULL);
    CHECK(strstr(buf, "24.5 t/s") != NULL);

    w.state = AGENT_STATE_IDLE;
    client_status_from_wire(&w, &st);
    build_status_text(&st, buf, sizeof(buf));
    CHECK(strstr(buf, "| idle") != NULL);

    w.state = AGENT_STATE_PREFILL;
    w.prefill_done = 50; w.prefill_total = 200;
    client_status_from_wire(&w, &st);
    build_status_text(&st, buf, sizeof(buf));
    CHECK(strstr(buf, "50/200") != NULL && strstr(buf, "25.0%") != NULL);
}

/* ---- T10: client tool execution ---------------------------------------- */

#include <stdarg.h>
#include <sys/wait.h>

static char *test_tmpdir(void) {
    static char dir[] = "/tmp/ds4agentclitestXXXXXX";
    char *d = xstrdup(dir);
    CHECK(mkdtemp(d) != NULL);
    return d;
}

static void write_file(const char *path, const char *content) {
    FILE *fp = fopen(path, "wb");
    CHECK(fp != NULL);
    if (fp) {
        fwrite(content, 1, strlen(content), fp);
        fclose(fp);
    }
}

static ap_tool_call mk_call(const char *name, int nargs, ...) {
    ap_tool_call c;
    memset(&c, 0, sizeof(c));
    c.name = name;
    c.name_len = strlen(name);
    va_list ap;
    va_start(ap, nargs);
    for (int i = 0; i < nargs; i++) {
        const char *an = va_arg(ap, const char *);
        const char *av = va_arg(ap, const char *);
        c.args[i].name = an;
        c.args[i].name_len = strlen(an);
        c.args[i].value = av;
        c.args[i].value_len = strlen(av);
        c.args[i].is_string = true;
    }
    va_end(ap);
    c.arg_count = (uint32_t)nargs;
    return c;
}

static const char *result_text(const client_tool_result *r) {
    return r->part_count ? r->parts[0] : "";
}

static void test_tool_write_read_edit(void) {
    char *dir = test_tmpdir();
    char path[600];
    snprintf(path, sizeof(path), "%s/a.c", dir);

    agent_worker w;
    client_worker_init(&w, 100000);

    ap_tool_calls calls = {0};
    calls.request_id = 1;
    calls.calls[0] = mk_call("write", 2, "path", path, "content", "int x = 1;\n");
    calls.call_count = 1;
    client_tool_result r;
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "Wrote") != NULL);
    client_tool_result_free(&r);

    calls.calls[0] = mk_call("read", 1, "path", path);
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "int x = 1;") != NULL);
    client_tool_result_free(&r);

    calls.calls[0] = mk_call("edit", 3, "path", path, "old", "int x = 1;\n",
                             "new", "int x = 2;\n");
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "Edited") != NULL);
    client_tool_result_free(&r);

    char *data = NULL;
    size_t len = 0;
    CHECK(agent_read_file_bytes(path, &data, &len, (char[64]){0}, 64) == 0);
    CHECK(strstr(data, "int x = 2;") != NULL);
    free(data);

    client_worker_free(&w);
    free(dir);
}

static void test_tool_edit_not_unique(void) {
    char *dir = test_tmpdir();
    char path[600];
    snprintf(path, sizeof(path), "%s/dup.c", dir);
    write_file(path, "foo\nfoo\n");

    agent_worker w;
    client_worker_init(&w, 100000);
    ap_tool_calls calls = {0};
    calls.request_id = 1;
    calls.calls[0] = mk_call("edit", 3, "path", path, "old", "foo\n", "new", "bar\n");
    calls.call_count = 1;
    client_tool_result r;
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "Tool error") != NULL);
    CHECK(strstr(result_text(&r), "not unique") != NULL);
    client_tool_result_free(&r);
    client_worker_free(&w);
    free(dir);
}

static void test_tool_list_and_search(void) {
    char *dir = test_tmpdir();
    char f1[600], f2[600];
    snprintf(f1, sizeof(f1), "%s/one.txt", dir);
    snprintf(f2, sizeof(f2), "%s/two.txt", dir);
    write_file(f1, "needle here\n");
    write_file(f2, "nothing\n");

    agent_worker w;
    client_worker_init(&w, 100000);

    ap_tool_calls calls = {0};
    calls.request_id = 1;
    calls.calls[0] = mk_call("list", 1, "path", dir);
    calls.call_count = 1;
    client_tool_result r;
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "one.txt") != NULL);
    CHECK(strstr(result_text(&r), "two.txt") != NULL);
    client_tool_result_free(&r);

    calls.calls[0] = mk_call("search", 2, "query", "needle", "path", dir);
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "one.txt") != NULL);
    CHECK(strstr(result_text(&r), "needle here") != NULL);
    CHECK(strstr(result_text(&r), "two.txt") == NULL);
    client_tool_result_free(&r);

    client_worker_free(&w);
    free(dir);
}

static void test_tool_bash_lifecycle(void) {
    agent_worker w;
    client_worker_init(&w, 100000);

    ap_tool_calls calls = {0};
    calls.request_id = 1;
    calls.calls[0] = mk_call("bash", 2, "command", "printf hi", "refresh_sec", "5");
    calls.call_count = 1;
    client_tool_result r;
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "status=done") != NULL);
    CHECK(strstr(result_text(&r), "exit_status=0") != NULL);
    CHECK(strstr(result_text(&r), "hi") != NULL);
    client_tool_result_free(&r);
    CHECK(w.bash_jobs == NULL); /* removed once observed done */

    /* a longer job with a short refresh: bash returns before it finishes */
    calls.calls[0] = mk_call("bash", 2, "command", "sleep 3", "refresh_sec", "1");
    client_execute_tool_calls(&w, &calls, &r);
    client_tool_result_free(&r);
    CHECK(w.bash_jobs != NULL);
    int job_id = w.bash_jobs->id;
    char job_str[16];
    snprintf(job_str, sizeof(job_str), "%d", job_id);
    calls.calls[0] = mk_call("bash_status", 1, "job", job_str);
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "status=running") != NULL);
    client_tool_result_free(&r);

    calls.calls[0] = mk_call("bash_stop", 1, "job", job_str);
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "status=done") != NULL);
    client_tool_result_free(&r);
    CHECK(w.bash_jobs == NULL);

    client_worker_free(&w);
}

static void test_tool_view_image(void) {
    char *dir = test_tmpdir();
    char path[600];
    snprintf(path, sizeof(path), "%s/pic.bin", dir);
    write_file(path, "\x89PNGfakebytes");

    agent_worker w;
    client_worker_init(&w, 100000);
    ap_tool_calls calls = {0};
    calls.request_id = 7;
    calls.calls[0] = mk_call("view_image", 1, "path", path);
    calls.call_count = 1;
    client_tool_result r;
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(r.image_count == 1);
    CHECK(r.images[0].len == strlen("\x89PNGfakebytes"));
    CHECK(memcmp(r.images[0].bytes, "\x89PNGfakebytes", r.images[0].len) == 0);
    CHECK(strstr(result_text(&r), "[tool:view_image]") == NULL); /* went to render sink, not the result */
    client_tool_result_free(&r);
    client_worker_free(&w);
    free(dir);
}

static void test_fit_context_bucket_on_ctx_size(void) {
    char *dir = test_tmpdir();
    char path[600];
    snprintf(path, sizeof(path), "%s/big.txt", dir);
    agent_buf b = {0};
    for (int i = 0; i < 400; i++) agent_buf_puts(&b, "line of text\n");
    write_file(path, b.ptr);
    free(b.ptr);

    agent_worker w;
    client_worker_init(&w, 4000); /* small ctx -> default 120 lines per read */
    ap_tool_calls calls = {0};
    calls.request_id = 1;
    calls.calls[0] = mk_call("read", 1, "path", path);
    calls.call_count = 1;
    client_tool_result r;
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "[Read truncated") != NULL);
    CHECK(w.more_valid == true); /* more picks up where this chunk left off */
    client_tool_result_free(&r);

    calls.calls[0] = mk_call("more", 0);
    client_execute_tool_calls(&w, &calls, &r);
    CHECK(strstr(result_text(&r), "line of text") != NULL);
    client_tool_result_free(&r);

    client_worker_free(&w);
    free(dir);
}

static void test_drain_reply_bash_jobs_note(void) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

    agent_worker w;
    client_worker_init(&w, 100000);
    ap_tool_calls calls = {0};
    calls.request_id = 1;
    calls.calls[0] = mk_call("bash", 2, "command", "sleep 3", "refresh_sec", "1");
    calls.call_count = 1;
    client_tool_result r;
    client_execute_tool_calls(&w, &calls, &r);
    client_tool_result_free(&r);
    CHECK(w.bash_jobs != NULL);

    client_conn co;
    conn_init(&co, sv[0], NULL);
    client_handle_drain_request(&co, &w, "continue the task");

    ap_buf rx; ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    uint32_t type = 0;
    ap_buf payload; ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_DRAIN_REPLY);
    ap_reader rd; ap_reader_init(&rd, payload.data, payload.len);
    ap_drain_reply d;
    CHECK(ap_decode_drain_reply(&rd, &d));
    CHECK(memmem(d.text, d.text_len, "Bash job update after context compaction",
                strlen("Bash job update after context compaction")) != NULL);
    CHECK(memmem(d.text, d.text_len, "continue the task",
                strlen("continue the task")) != NULL);

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    conn_free(&co);
    close(sv[1]);
    client_worker_free(&w);
}

static void test_tool_calls_over_wire(void) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    char *dir = test_tmpdir();
    char path[600];
    snprintf(path, sizeof(path), "%s/w.txt", dir);
    write_file(path, "hello\n");

    agent_worker w;
    client_worker_init(&w, 100000);

    ap_tool_calls calls = {0};
    calls.request_id = 42;
    calls.calls[0] = mk_call("read", 1, "path", path);
    calls.call_count = 1;

    client_conn co;
    conn_init(&co, sv[0], NULL);
    client_handle_tool_calls(&co, &w, &calls);

    ap_buf rx; ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    uint32_t type = 0;
    ap_buf payload; ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_TOOL_RESULT);
    ap_reader rd; ap_reader_init(&rd, payload.data, payload.len);
    ap_tool_result tr;
    CHECK(ap_decode_tool_result(&rd, &tr));
    CHECK(tr.request_id == 42);
    CHECK(tr.text_part_count >= 1);
    CHECK(memmem(tr.text_parts[0].ptr, tr.text_parts[0].len, "hello", 5) != NULL);

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    conn_free(&co);
    close(sv[1]);
    client_worker_free(&w);
    free(dir);
}

/* ---- T11a: welcome banner + footer/queue text ------------------------- */

static void test_welcome_banner(void) {
    char buf[256];
    agent_format_welcome_banner(100000, buf, sizeof(buf));
    CHECK(strstr(buf, "100k") != NULL);
    CHECK(strstr(buf, "DwarfStar") != NULL || strstr(buf, "Dwarf") != NULL);
}

static void test_footer_with_queue(void) {
    ap_status w = {0};
    w.state = AGENT_STATE_GENERATING;
    w.ctx_used = 1000;
    w.ctx_size = 100000;
    agent_status st;
    client_status_from_wire(&w, &st);

    agent_prompt_queue q = {0};
    agent_prompt_queue_push(&q, "next thing to do");
    /* build_footer_text only renders a queue preview once the buffer is more
     * than 1024 bytes past its own overhead (len - 1024 budget). */
    char buf[2048];
    build_footer_text(&st, &q, 80, buf, sizeof(buf));
    CHECK(strstr(buf, "queued:") != NULL);
    CHECK(strstr(buf, "next thing to do") != NULL);
    CHECK(strstr(buf, "generation") != NULL);
    CHECK(agent_footer_is_multiline(buf));

    /* no queue -> just the status line */
    char buf2[512];
    build_footer_text(&st, NULL, 80, buf2, sizeof(buf2));
    CHECK(!agent_footer_is_multiline(buf2));
    CHECK(strstr(buf2, "generation") != NULL);

    agent_prompt_queue_free(&q);
}

static void test_prompt_queue_take_all(void) {
    agent_prompt_queue q = {0};
    agent_prompt_queue_push(&q, "first");
    agent_prompt_queue_push(&q, "second");
    char *joined = agent_prompt_queue_take_all(&q);
    CHECK(joined != NULL);
    CHECK(strstr(joined, "first") != NULL && strstr(joined, "second") != NULL);
    CHECK(q.len == 0);
    free(joined);
}

/* ---- T11b: wire runtime -- push dispatcher + blocking RPC -------------- */

static ap_status mk_status(uint32_t state, uint32_t ctx_used, uint32_t ctx_size) {
    ap_status s = {0};
    s.state = state;
    s.ctx_used = ctx_used;
    s.ctx_size = ctx_size;
    return s;
}

/* Bring up a client_runtime over a fresh socketpair, with the render sink
 * overridden to the capture buffer (client_runtime_init points it at
 * client_editor_sink, which is only interesting once T11c has a live
 * editor). Caller must close sv[1] and free everything else. */
static void rt_setup(client_runtime *rt, client_conn *co, agent_worker *w,
                     agent_token_renderer *r, int sv[2]) {
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);
    conn_init(co, sv[0], NULL);
    client_worker_init(w, 100000);
    memset(r, 0, sizeof(*r));
    r->last_output_newline = true;
    client_runtime_init(rt, co, w, r, 100000);
    g_cap_len = 0;
    g_cap[0] = '\0';
    g_render_sink = cap_sink;
}

static void rt_teardown(client_runtime *rt, client_conn *co, agent_worker *w,
                        int sv[2]) {
    client_runtime_free(rt);
    conn_free(co); /* closes sv[0] */
    close(sv[1]);
    client_worker_free(w);
}

static void test_dispatch_push_stream_reaches_renderer(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    ap_stream s = { .kind = AGENT_STREAM_NORMAL, .text = "hello from stream",
                    .text_len = strlen("hello from stream") };
    ap_buf body;
    ap_buf_init(&body);
    ap_encode_stream(&body, &s);
    client_dispatch_push(&rt, AGENT_MSG_STREAM, &body);
    ap_buf_free(&body);

    CHECK(strstr(g_cap, "hello from stream") != NULL);

    rt_teardown(&rt, &co, &w, sv);
}

static void test_dispatch_push_status_finishes_stream_on_terminal(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    /* a busy status just records the mirror; nothing to finish yet */
    {
        ap_status st = mk_status(AGENT_STATE_GENERATING, 10, 100000);
        ap_buf body;
        ap_buf_init(&body);
        ap_encode_status(&body, &st);
        client_dispatch_push(&rt, AGENT_MSG_STATUS, &body);
        ap_buf_free(&body);
    }
    CHECK(rt.have_status);
    CHECK(rt.last_status.state == AGENT_STATE_GENERATING);

    /* a tool-name fragment opens the tool visualizer */
    {
        ap_stream s = { .kind = AGENT_STREAM_TOOL_NAME, .text = "read", .text_len = 4 };
        ap_buf body;
        ap_buf_init(&body);
        ap_encode_stream(&body, &s);
        client_dispatch_push(&rt, AGENT_MSG_STREAM, &body);
        ap_buf_free(&body);
    }
    CHECK(rt.stream.viz.active);

    /* the busy -> terminal transition closes the open visualizer */
    {
        ap_status st = mk_status(AGENT_STATE_IDLE, 10, 100000);
        ap_buf body;
        ap_buf_init(&body);
        ap_encode_status(&body, &st);
        client_dispatch_push(&rt, AGENT_MSG_STATUS, &body);
        ap_buf_free(&body);
    }
    CHECK(!rt.stream.viz.active);

    rt_teardown(&rt, &co, &w, sv);
}

static void test_dispatch_push_tool_calls_replies_over_wire(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    char *dir = test_tmpdir();
    char path[600];
    snprintf(path, sizeof(path), "%s/w.txt", dir);
    write_file(path, "hi there\n");

    ap_tool_calls calls = {0};
    calls.request_id = 7;
    calls.calls[0] = mk_call("read", 1, "path", path);
    calls.call_count = 1;
    ap_buf body;
    ap_buf_init(&body);
    ap_encode_tool_calls(&body, &calls);
    client_dispatch_push(&rt, AGENT_MSG_TOOL_CALLS, &body);
    ap_buf_free(&body);

    ap_buf rx;
    ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_TOOL_RESULT);
    ap_reader rd;
    ap_reader_init(&rd, payload.data, payload.len);
    ap_tool_result tr;
    CHECK(ap_decode_tool_result(&rd, &tr));
    CHECK(tr.request_id == 7);
    CHECK(tr.text_part_count >= 1);
    CHECK(memmem(tr.text_parts[0].ptr, tr.text_parts[0].len, "hi there", 8) != NULL);

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    rt_teardown(&rt, &co, &w, sv);
    free(dir);
}

static void test_dispatch_push_drain_request_with_no_queue(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);
    CHECK(rt.queue == NULL); /* T11c wires a real queue; empty until then */

    ap_buf empty;
    ap_buf_init(&empty);
    client_dispatch_push(&rt, AGENT_MSG_DRAIN_REQUEST, &empty);
    ap_buf_free(&empty);

    ap_buf rx;
    ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_DRAIN_REPLY);

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    rt_teardown(&rt, &co, &w, sv);
}

static void test_rpc_dispatches_pushes_before_reply(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    /* the mock server has a STREAM + STATUS tail to deliver before it gets
     * around to answering the SESSION request client_rpc is waiting on */
    {
        ap_stream s = { .kind = AGENT_STREAM_NORMAL, .text = "late tail output",
                        .text_len = strlen("late tail output") };
        ap_buf b;
        ap_buf_init(&b);
        ap_encode_stream(&b, &s);
        mock_send(sv[1], AGENT_MSG_STREAM, &b);
        ap_buf_free(&b);
    }
    {
        ap_status st = mk_status(AGENT_STATE_IDLE, 5, 100000);
        ap_buf b;
        ap_buf_init(&b);
        ap_encode_status(&b, &st);
        mock_send(sv[1], AGENT_MSG_STATUS, &b);
        ap_buf_free(&b);
    }
    {
        ap_buf b;
        ap_buf_init(&b);
        ap_put_reply_ok(&b);
        mock_send(sv[1], AGENT_MSG_SESSION, &b);
        ap_buf_free(&b);
    }

    ap_buf send_body;
    ap_buf_init(&send_body);
    ap_encode_session_simple(&send_body, AGENT_SESSION_SAVE);
    char err[256] = {0};
    ap_buf reply;
    CHECK(client_rpc(&rt, AGENT_MSG_SESSION, &send_body, AGENT_MSG_SESSION,
                     &reply, err, sizeof(err)));
    CHECK(err[0] == '\0');
    ap_buf_free(&send_body);
    ap_buf_free(&reply);

    /* the pushed frames were dispatched on the way to the reply */
    CHECK(strstr(g_cap, "late tail output") != NULL);
    CHECK(rt.have_status);
    CHECK(rt.last_status.state == AGENT_STATE_IDLE);

    rt_teardown(&rt, &co, &w, sv);
}

static void test_refresh_session_list(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    {
        ap_buf b;
        ap_buf_init(&b);
        ap_put_reply_arr_begin(&b, 2);
        {
            ap_map_writer mw;
            ap_map_begin(&mw, &b);
            ap_map_put_cstr(&mw, "sha", "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa");
            ap_map_put_cstr(&mw, "title", "first session");
            ap_map_put_u32(&mw, "tokens", 123);
            ap_map_put_i64(&mw, "created_at", 1000);
            ap_map_put_i64(&mw, "last_used", 2000);
            ap_map_put_bool(&mw, "is_current", true);
            ap_map_end(&mw);
        }
        {
            ap_map_writer mw;
            ap_map_begin(&mw, &b);
            ap_map_put_cstr(&mw, "sha", "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb");
            ap_map_put_cstr(&mw, "title", "second session");
            ap_map_put_u32(&mw, "tokens", 456);
            ap_map_put_i64(&mw, "created_at", 500);
            ap_map_put_i64(&mw, "last_used", 900);
            ap_map_put_bool(&mw, "is_current", false);
            ap_map_end(&mw);
        }
        mock_send(sv[1], AGENT_MSG_SESSION, &b);
        ap_buf_free(&b);
    }

    char err[256] = {0};
    CHECK(client_refresh_session_list(&rt, err, sizeof(err)));
    CHECK(err[0] == '\0');
    CHECK(rt.sessions.len == 2);
    CHECK(strcmp(rt.sessions.v[0].sha, "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa") == 0);
    CHECK(strcmp(rt.sessions.v[0].title, "first session") == 0);
    CHECK(rt.sessions.v[0].tokens == 123);
    CHECK(rt.sessions.v[0].is_current == true);
    CHECK(strcmp(rt.sessions.v[1].sha, "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb") == 0);
    CHECK(rt.sessions.v[1].is_current == false);

    rt_teardown(&rt, &co, &w, sv);
}

/* ---- T11c-a: command layer -- RPC wrappers + local mirrors ------------- */

static void mock_send_config_reply(int fd, bool ok, uint32_t value, const char *error) {
    ap_buf b;
    ap_buf_init(&b);
    ap_map_writer mw;
    ap_put_reply_map_begin(&b, &mw);
    ap_map_put_bool(&mw, "ok", ok);
    ap_map_put_u32(&mw, "value", value);
    ap_map_put_cstr(&mw, "error", error ? error : "");
    ap_map_end(&mw);
    mock_send(fd, AGENT_MSG_CONFIG, &b);
    ap_buf_free(&b);
}

static void mock_send_save_reply(int fd, bool ok, bool scheduled, const char *sha,
                                 uint32_t tokens, const char *error) {
    ap_buf b;
    ap_buf_init(&b);
    ap_map_writer mw;
    ap_put_reply_map_begin(&b, &mw);
    ap_map_put_bool(&mw, "ok", ok);
    ap_map_put_bool(&mw, "scheduled", scheduled);
    ap_map_put_cstr(&mw, "sha", sha ? sha : "");
    ap_map_put_u32(&mw, "tokens", tokens);
    ap_map_put_cstr(&mw, "error", error ? error : "");
    ap_map_end(&mw);
    mock_send(fd, AGENT_MSG_SESSION, &b);
    ap_buf_free(&b);
}

static void mock_send_switch_reply(int fd, bool ok, const char *sha, const char *title,
                                   uint32_t ctx_used, const char *error) {
    ap_buf b;
    ap_buf_init(&b);
    ap_map_writer mw;
    ap_put_reply_map_begin(&b, &mw);
    ap_map_put_bool(&mw, "ok", ok);
    ap_map_put_cstr(&mw, "sha", sha ? sha : "");
    ap_map_put_cstr(&mw, "title", title ? title : "");
    ap_map_put_u32(&mw, "ctx_used", ctx_used);
    ap_map_put_cstr(&mw, "error", error ? error : "");
    ap_map_end(&mw);
    mock_send(fd, AGENT_MSG_SESSION, &b);
    ap_buf_free(&b);
}

static void mock_send_del_reply(int fd, bool ok, const char *sha, uint32_t tokens,
                                const char *error) {
    ap_buf b;
    ap_buf_init(&b);
    ap_map_writer mw;
    ap_put_reply_map_begin(&b, &mw);
    ap_map_put_bool(&mw, "ok", ok);
    ap_map_put_cstr(&mw, "sha", sha ? sha : "");
    ap_map_put_u32(&mw, "tokens", tokens);
    ap_map_put_cstr(&mw, "error", error ? error : "");
    ap_map_end(&mw);
    mock_send(fd, AGENT_MSG_SESSION, &b);
    ap_buf_free(&b);
}

static void test_turn_submit_idle_gate(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    /* not idle yet (no STATUS seen) -> rejected without sending anything */
    CHECK(!client_turn_submit(&rt, "hello"));

    rt.have_status = true;
    rt.last_status.state = AGENT_STATE_GENERATING;
    CHECK(!client_turn_submit(&rt, "hello"));

    rt.last_status.state = AGENT_STATE_IDLE;
    CHECK(client_turn_submit(&rt, "hello"));
    CHECK(rt.activity_since_save);

    ap_buf rx;
    ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_TURN);
    ap_reader rd;
    ap_reader_init(&rd, payload.data, payload.len);
    ap_turn t;
    CHECK(ap_decode_turn(&rd, &t));
    CHECK(t.text_len == strlen("hello"));
    CHECK(memcmp(t.text, "hello", 5) == 0);

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    rt_teardown(&rt, &co, &w, sv);
}

static void test_worker_interrupt_sends_frame(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    client_worker_interrupt(&rt);

    ap_buf rx;
    ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_INTERRUPT);

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    rt_teardown(&rt, &co, &w, sv);
}

static void test_config_rpc_power_and_error(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    mock_send_config_reply(sv[1], true, 55, NULL);
    char err[192] = {0};
    CHECK(client_power_set(&rt, 55, err, sizeof(err)));
    CHECK(err[0] == '\0');

    mock_send_config_reply(sv[1], false, 0, "power must be 1..100");
    CHECK(!client_power_set(&rt, 200, err, sizeof(err)));
    CHECK(strstr(err, "1..100") != NULL);

    rt_teardown(&rt, &co, &w, sv);
}

static void test_config_rpc_steer_roundtrip(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    mock_send_config_reply(sv[1], true, ap_milli_from_double(12.5), NULL);
    char err[192] = {0};
    float scale = 0.0f;
    CHECK(client_steer_get(&rt, &scale, err, sizeof(err)));
    CHECK(scale > 12.4f && scale < 12.6f);

    mock_send_config_reply(sv[1], true, ap_milli_from_double(-40.0), NULL);
    CHECK(client_steer_set(&rt, -40.0f, err, sizeof(err)));

    rt_teardown(&rt, &co, &w, sv);
}

static void test_config_rpc_hints(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    mock_send_config_reply(sv[1], true, 1, NULL);
    char err[192] = {0};
    CHECK(client_hints_set(&rt, true, err, sizeof(err)));

    rt_teardown(&rt, &co, &w, sv);
}

static void test_session_save_immediate_and_scheduled(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);
    rt.activity_since_save = true;

    mock_send_save_reply(sv[1], true, false,
                         "cccccccccccccccccccccccccccccccccccccccc", 42, NULL);
    char err[192] = {0};
    bool scheduled = true;
    CHECK(client_session_save(&rt, &scheduled, err, sizeof(err)));
    CHECK(!scheduled);
    CHECK(!rt.activity_since_save);
    CHECK(strcmp(rt.current_sha, "cccccccccccccccccccccccccccccccccccccccc") == 0);

    rt.activity_since_save = true;
    mock_send_save_reply(sv[1], true, true, "", 0, NULL);
    scheduled = false;
    CHECK(client_session_save(&rt, &scheduled, err, sizeof(err)));
    CHECK(scheduled);
    CHECK(rt.activity_since_save); /* not cleared: the save hasn't run yet */

    rt_teardown(&rt, &co, &w, sv);
}

static void test_session_switch_updates_identity(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);
    rt.activity_since_save = true;

    mock_send_switch_reply(sv[1], true,
                           "dddddddddddddddddddddddddddddddddddddddd",
                           "other session", 77, NULL);
    char err[192] = {0};
    CHECK(client_session_switch(&rt, "dddd", AGENT_HISTORY_DEFAULT_TURNS, err, sizeof(err)));
    CHECK(strcmp(rt.current_sha, "dddddddddddddddddddddddddddddddddddddddd") == 0);
    CHECK(strcmp(rt.current_title, "other session") == 0);
    CHECK(!rt.activity_since_save);

    mock_send_switch_reply(sv[1], false, "", "", 0, "no such session");
    CHECK(!client_session_switch(&rt, "ffff", 0, err, sizeof(err)));
    CHECK(strstr(err, "no such session") != NULL);

    rt_teardown(&rt, &co, &w, sv);
}

static void test_session_show_history_requires_identity(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    /* no sha yet -> rejected locally, nothing sent on the wire */
    char err[192] = {0};
    CHECK(!client_session_show_history(&rt, 5, err, sizeof(err)));
    CHECK(err[0] != '\0');

    ap_buf rx;
    ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    CHECK(rx.len == 0);
    ap_buf_free(&rx);

    snprintf(rt.current_sha, sizeof(rt.current_sha), "%s",
             "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
    rt.activity_since_save = true;
    mock_send_switch_reply(sv[1], true, rt.current_sha, "same session", 9, NULL);
    CHECK(client_session_show_history(&rt, 5, err, sizeof(err)));
    /* a self-referencing history dump must not look like a discarding switch */
    CHECK(rt.activity_since_save);

    rt_teardown(&rt, &co, &w, sv);
}

static void test_session_del_and_strip(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    mock_send_del_reply(sv[1], true,
                        "ffffffffffffffffffffffffffffffffffffffff", 0, NULL);
    char sha[64] = {0};
    uint32_t tokens = 0;
    char err[192] = {0};
    CHECK(client_session_del(&rt, "ffff", false, sha, sizeof(sha), &tokens, err, sizeof(err)));
    CHECK(strcmp(sha, "ffffffffffffffffffffffffffffffffffffffff") == 0);

    mock_send_del_reply(sv[1], true,
                        "1111111111111111111111111111111111111111", 321, NULL);
    CHECK(client_session_del(&rt, "1111", true, sha, sizeof(sha), &tokens, err, sizeof(err)));
    CHECK(tokens == 321);

    rt_teardown(&rt, &co, &w, sv);
}

static void test_session_new_mid_session(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);
    snprintf(rt.current_sha, sizeof(rt.current_sha), "stale");
    rt.activity_since_save = true;

    ap_session_ready ready = make_ready();
    ap_buf b;
    ap_buf_init(&b);
    ap_encode_session_ready(&b, &ready);
    mock_send(sv[1], AGENT_MSG_SESSION, &b);
    ap_buf_free(&b);

    client_config cfg = {0};
    snprintf(cfg.server_host, sizeof(cfg.server_host), "127.0.0.1");
    char err[192] = {0};
    CHECK(client_session_new(&rt, &cfg, err, sizeof(err)));
    CHECK(rt.current_sha[0] == '\0');
    CHECK(!rt.activity_since_save);
    CHECK(rt.have_status);
    CHECK(rt.last_status.state == ready.state);
    CHECK(rt.last_status.ctx_size == ready.ctx_size);

    rt_teardown(&rt, &co, &w, sv);
}

static void test_switch_completion_from_cache(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);

    client_session_list_push(&rt.sessions, (client_session_row){
        .sha = "aa11111111111111111111111111111111111111", .title = "a" });
    client_session_list_push(&rt.sessions, (client_session_row){
        .sha = "ab22222222222222222222222222222222222222", .title = "b" });
    client_session_list_push(&rt.sessions, (client_session_row){
        .sha = "cc33333333333333333333333333333333333333", .title = "c" });

    g_completion_runtime = &rt;
    linenoiseCompletions lc = {0};
    client_switch_completion_callback("/switch a", &lc);
    CHECK(lc.len == 2);
    for (size_t i = 0; i < lc.len; i++) free(lc.cvec[i]);
    free(lc.cvec);

    linenoiseCompletions lc2 = {0};
    client_switch_completion_callback("/not-switch", &lc2);
    CHECK(lc2.len == 0);

    g_completion_runtime = NULL;
    rt_teardown(&rt, &co, &w, sv);
}

static void test_save_prompt_helpers_skip_when_clean(void) {
    client_runtime rt;
    client_conn co;
    agent_worker w;
    agent_token_renderer r;
    int sv[2];
    rt_setup(&rt, &co, &w, &r, sv);
    rt.activity_since_save = false; /* nothing to save -> no prompt, no RPC */

    CHECK(client_maybe_save_before_leaving_session(&rt));
    CHECK(client_maybe_save_before_exiting(&rt) == AGENT_EXIT_CLEAN);

    ap_buf rx;
    ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    CHECK(rx.len == 0);
    ap_buf_free(&rx);

    rt_teardown(&rt, &co, &w, sv);
}

/* ---- T11c-b: run_client_non_interactive end-to-end ---------------------
 * The one-shot path (-p/--prompt) never touches STDIN_FILENO, so it is the
 * only loop entry point this test suite can drive without a pty. Redirects
 * the real stdout to a pipe for the duration of the call to capture what
 * the render sink wrote (no live editor is active in headless mode, so
 * client_editor_sink falls back to a plain stdout write). */
static void test_non_interactive_one_shot_e2e(void) {
    int sv[2];
    CHECK(socketpair(AF_UNIX, SOCK_STREAM, 0, sv) == 0);

    int outpipe[2];
    CHECK(pipe(outpipe) == 0);
    int saved_stdout = dup(STDOUT_FILENO);
    CHECK(saved_stdout >= 0);
    CHECK(dup2(outpipe[1], STDOUT_FILENO) >= 0);
    close(outpipe[1]);

    /* Full turn scripted up front: STATUS{PREFILL} -> STREAM -> TOOL_CALLS
     * -> (the loop executes it and replies TOOL_RESULT) -> STREAM ->
     * terminal STATUS{IDLE}. */
    {
        ap_status st = mk_status(AGENT_STATE_PREFILL, 0, 100000);
        ap_buf b;
        ap_buf_init(&b);
        ap_encode_status(&b, &st);
        mock_send(sv[1], AGENT_MSG_STATUS, &b);
        ap_buf_free(&b);
    }
    {
        ap_stream s = { .kind = AGENT_STREAM_NORMAL, .text = "Sure, reading now.\n",
                        .text_len = strlen("Sure, reading now.\n") };
        ap_buf b;
        ap_buf_init(&b);
        ap_encode_stream(&b, &s);
        mock_send(sv[1], AGENT_MSG_STREAM, &b);
        ap_buf_free(&b);
    }
    char *dir = test_tmpdir();
    char path[600];
    snprintf(path, sizeof(path), "%s/w.txt", dir);
    write_file(path, "file body\n");
    {
        ap_tool_calls calls = {0};
        calls.request_id = 1;
        calls.calls[0] = mk_call("read", 1, "path", path);
        calls.call_count = 1;
        ap_buf b;
        ap_buf_init(&b);
        ap_encode_tool_calls(&b, &calls);
        mock_send(sv[1], AGENT_MSG_TOOL_CALLS, &b);
        ap_buf_free(&b);
    }
    {
        ap_stream s = { .kind = AGENT_STREAM_NORMAL, .text = "Done.\n",
                        .text_len = strlen("Done.\n") };
        ap_buf b;
        ap_buf_init(&b);
        ap_encode_stream(&b, &s);
        mock_send(sv[1], AGENT_MSG_STREAM, &b);
        ap_buf_free(&b);
    }
    {
        ap_status st = mk_status(AGENT_STATE_IDLE, 20, 100000);
        ap_buf b;
        ap_buf_init(&b);
        ap_encode_status(&b, &st);
        mock_send(sv[1], AGENT_MSG_STATUS, &b);
        ap_buf_free(&b);
    }

    client_conn co;
    conn_init(&co, sv[0], NULL);
    client_config cfg = {0};
    snprintf(cfg.server_host, sizeof(cfg.server_host), "127.0.0.1");
    cfg.prompt = "please read the file";

    client_session sess;
    memset(&sess, 0, sizeof(sess));
    sess.ready = make_ready();

    int rc = run_client_non_interactive(&co, &cfg, &sess);

    fflush(stdout);
    dup2(saved_stdout, STDOUT_FILENO);
    close(saved_stdout);

    char cap[4096] = {0};
    ssize_t n = read(outpipe[0], cap, sizeof(cap) - 1);
    close(outpipe[0]);
    if (n > 0) cap[n] = '\0';

    CHECK(rc == 0);
    CHECK(strstr(cap, "Sure, reading now.") != NULL);
    CHECK(strstr(cap, "Done.") != NULL);

    /* client sends, in order: TURN then TOOL_RESULT, nothing else. */
    ap_buf rx;
    ap_buf_init(&rx);
    mock_drain(sv[1], &rx);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    char ferr[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_TURN);
    {
        ap_reader r;
        ap_reader_init(&r, payload.data, payload.len);
        ap_turn t;
        CHECK(ap_decode_turn(&r, &t));
        CHECK(t.text_len == strlen("please read the file"));
    }
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) == AP_FRAME_OK);
    CHECK(type == AGENT_MSG_TOOL_RESULT);
    {
        ap_reader r;
        ap_reader_init(&r, payload.data, payload.len);
        ap_tool_result tr;
        CHECK(ap_decode_tool_result(&r, &tr));
        CHECK(tr.request_id == 1);
        CHECK(tr.text_part_count >= 1);
        CHECK(memmem(tr.text_parts[0].ptr, tr.text_parts[0].len, "file body", 9) != NULL);
    }
    CHECK(ap_read_frame(&rx, &type, &payload, ferr, sizeof(ferr)) != AP_FRAME_OK);

    ap_buf_free(&payload);
    ap_buf_free(&rx);
    conn_free(&co);
    close(sv[1]);
    free(dir);
}

int main(void) {
    test_handshake_new_session();
    test_handshake_resume();
    test_handshake_resume_falls_back_to_new();
    test_handshake_version_mismatch();
    test_parse_server_endpoint();
    test_render_normal_and_think();
    test_render_system_notice();
    test_render_tool_read_viz();
    test_render_tool_edit_diff();
    test_build_status_text();
    test_tool_write_read_edit();
    test_tool_edit_not_unique();
    test_tool_list_and_search();
    test_tool_bash_lifecycle();
    test_tool_view_image();
    test_fit_context_bucket_on_ctx_size();
    test_drain_reply_bash_jobs_note();
    test_tool_calls_over_wire();
    test_welcome_banner();
    test_footer_with_queue();
    test_prompt_queue_take_all();
    test_dispatch_push_stream_reaches_renderer();
    test_dispatch_push_status_finishes_stream_on_terminal();
    test_dispatch_push_tool_calls_replies_over_wire();
    test_dispatch_push_drain_request_with_no_queue();
    test_rpc_dispatches_pushes_before_reply();
    test_refresh_session_list();
    test_turn_submit_idle_gate();
    test_worker_interrupt_sends_frame();
    test_config_rpc_power_and_error();
    test_config_rpc_steer_roundtrip();
    test_config_rpc_hints();
    test_session_save_immediate_and_scheduled();
    test_session_switch_updates_identity();
    test_session_show_history_requires_identity();
    test_session_del_and_strip();
    test_session_new_mid_session();
    test_switch_completion_from_cache();
    test_save_prompt_helpers_skip_when_clean();
    test_non_interactive_one_shot_e2e();

    if (failures) {
        fprintf(stderr, "%d agent client test(s) failed\n", failures);
        return 1;
    }
    puts("agent client tests passed");
    return 0;
}
