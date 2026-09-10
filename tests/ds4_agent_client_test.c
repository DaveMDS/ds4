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

int main(void) {
    test_handshake_new_session();
    test_handshake_resume();
    test_handshake_resume_falls_back_to_new();
    test_handshake_version_mismatch();
    test_parse_server_endpoint();

    if (failures) {
        fprintf(stderr, "%d agent client test(s) failed\n", failures);
        return 1;
    }
    puts("agent client tests passed");
    return 0;
}
