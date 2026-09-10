/* ds4-agent-client: the terminal UI half of the ds4-agent client/server split.
 *
 * Connects to a ds4-agent-server over the framed TCP protocol in
 * ds4_agent_proto.h, drives the coding-agent UI locally (editor, painting,
 * tool execution) and never loads the model. ds4.h is included only for the
 * POD types the copied UI/tool code needs; no engine object is linked.
 *
 * T8 scope (AGENT-SPLIT-PLAN.md section 5): client_parse_options, --help via
 * DS4_HELP_AGENT_CLIENT, the socket connect, the HELLO handshake and the
 * session_parked branch (SESSION resume vs SESSION new). The render stack,
 * tool execution and the main loop arrive in T9-T11.
 */

#include "ds4.h"
#include "ds4_help.h"
#include "ds4_agent_proto.h"
#include "ds4_agent_utils.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

/* ========================================================================= */
/* Configuration                                                              */
/* ========================================================================= */

/* UI-side options. The sampling / think / seed / power / steer / hints / -sys
 * knobs are client flags that populate SESSION new; everything about the model,
 * backend and context lives on the server. */
typedef struct {
    char server_host[256];
    int server_port;

    const char *prompt;     /* -p / --prompt (borrowed) */
    char *prompt_owned;      /* --prompt-file contents */
    const char *system;      /* -sys / --system (borrowed) */
    char *prefix_file_text;  /* --prefix-file raw contents, forwarded verbatim */
    size_t prefix_file_len;
    const char *trace_path;
    const char *chdir_path;
    bool non_interactive;
    bool raw_prompt;

    int n_predict;
    float temperature;
    float top_p;
    float min_p;
    bool temperature_set;
    bool top_p_set;
    bool min_p_set;
    uint64_t seed;
    ds4_think_mode think_mode;
    bool power_set;
    int power;
    bool hints_enabled;
    float dir_steering_ffn;
} client_config;

/* ========================================================================= */
/* CLI                                                                        */
/* ========================================================================= */

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-agent-client: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_AGENT_CLIENT, topic);
}

static void set_host(client_config *c, const char *h, size_t n) {
    if (n >= sizeof(c->server_host)) {
        fprintf(stderr, "ds4-agent-client: host name too long\n");
        exit(2);
    }
    memcpy(c->server_host, h, n);
    c->server_host[n] = '\0';
}

/* Parse "host:port", "host", or ":port". */
static void parse_server_endpoint(client_config *c, const char *arg) {
    const char *colon = strrchr(arg, ':');
    if (colon && strchr(arg, ':') != colon && arg[0] != '[') {
        fprintf(stderr, "ds4-agent-client: --server address must be host:port "
                        "(bracket IPv6 literals)\n");
        exit(2);
    }
    if (!colon) {
        set_host(c, arg, strlen(arg));
        return;
    }
    if (colon != arg) set_host(c, arg, (size_t)(colon - arg));
    if (colon[1]) c->server_port = parse_int(colon + 1, "--server port");
}

static char *read_whole_file(const char *path, size_t *len_out) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return NULL;
    agent_input_buf b = {0};
    char chunk[65536];
    size_t n;
    while ((n = fread(chunk, 1, sizeof(chunk), fp)) > 0)
        agent_input_buf_append(&b, chunk, n);
    bool ok = !ferror(fp);
    fclose(fp);
    if (!ok) { agent_input_buf_free(&b); return NULL; }
    size_t len = b.len;
    char *out = agent_input_buf_take(&b);
    if (len_out) *len_out = len;
    return out;
}

static bool parse_hints(const char *arg, bool *enabled) {
    if (!strcmp(arg, "on")) { *enabled = true; return true; }
    if (!strcmp(arg, "off")) { *enabled = false; return true; }
    return false;
}

static client_config client_parse_options(int argc, char **argv) {
    client_config c = {
        .server_port = AGENT_PROTO_DEFAULT_PORT,
        .system = "You are a helpful coding assistant running inside ds4-agent.",
        .n_predict = 50000,
        .temperature = DS4_DEFAULT_TEMPERATURE,
        .top_p = DS4_DEFAULT_TOP_P,
        .min_p = DS4_DEFAULT_MIN_P,
        .think_mode = DS4_THINK_HIGH,
    };
    snprintf(c.server_host, sizeof(c.server_host), "%s", AGENT_PROTO_DEFAULT_HOST);

    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
            exit(0);
        } else if (!strcmp(arg, "--server")) {
            parse_server_endpoint(&c, need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            if (c.prompt) {
                fprintf(stderr, "ds4-agent-client: specify only one of -p and --prompt-file\n");
                exit(2);
            }
            c.prompt = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--prompt-file")) {
            if (c.prompt) {
                fprintf(stderr, "ds4-agent-client: specify only one of -p and --prompt-file\n");
                exit(2);
            }
            size_t n = 0;
            c.prompt_owned = read_whole_file(need_arg(&i, argc, argv, arg), &n);
            if (!c.prompt_owned) {
                fprintf(stderr, "ds4-agent-client: cannot read --prompt-file: %s\n",
                        strerror(errno));
                exit(2);
            }
            c.prompt = c.prompt_owned;
        } else if (!strcmp(arg, "--prefix-file")) {
            free(c.prefix_file_text);
            c.prefix_file_text = read_whole_file(need_arg(&i, argc, argv, arg),
                                                 &c.prefix_file_len);
            if (!c.prefix_file_text) {
                fprintf(stderr, "ds4-agent-client: cannot read --prefix-file: %s\n",
                        strerror(errno));
                exit(2);
            }
        } else if (!strcmp(arg, "--non-interactive")) {
            c.non_interactive = true;
        } else if (!strcmp(arg, "--raw") || !strcmp(arg, "--raw-prompt")) {
            c.raw_prompt = true;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--chdir")) {
            c.chdir_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
            c.temperature_set = true;
        } else if (!strcmp(arg, "--top-p")) {
            c.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.top_p_set = true;
        } else if (!strcmp(arg, "--min-p")) {
            c.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.min_p_set = true;
        } else if (!strcmp(arg, "--seed")) {
            c.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--power")) {
            int p = 0;
            if (!parse_power_percent(need_arg(&i, argc, argv, arg), &p)) {
                fprintf(stderr, "ds4-agent-client: --power must be between 1 and 100\n");
                exit(2);
            }
            c.power = p;
            c.power_set = true;
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.dir_steering_ffn =
                parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
        } else if (!strcmp(arg, "--hints")) {
            bool en = false;
            if (!parse_hints(need_arg(&i, argc, argv, arg), &en)) {
                fprintf(stderr, "ds4-agent-client: usage: --hints on|off\n");
                exit(2);
            }
            c.hints_enabled = en;
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model") ||
                   !strcmp(arg, "--backend") || !strcmp(arg, "--ctx") ||
                   !strcmp(arg, "-c") || !strcmp(arg, "--vision") ||
                   !strncmp(arg, "--gpu-", 6) || !strncmp(arg, "--mtp", 5) ||
                   !strncmp(arg, "--ssd-", 6) || !strncmp(arg, "--dspark", 8) ||
                   !strncmp(arg, "--dir-steering-", 15)) {
            fprintf(stderr, "ds4-agent-client: %s is a ds4-agent-server option; "
                            "pass it there, not to the client\n", arg);
            exit(2);
        } else {
            fprintf(stderr, "ds4-agent-client: unknown option: %s\n", arg);
            usage(stderr, NULL);
            exit(2);
        }
    }

    if (c.server_port < 1 || c.server_port > 65535) {
        fprintf(stderr, "ds4-agent-client: --server port must be between 1 and 65535\n");
        exit(2);
    }
    if (c.raw_prompt && (!c.non_interactive || !c.prompt)) {
        fprintf(stderr, "ds4-agent-client: --raw-prompt requires --non-interactive "
                        "and an initial prompt\n");
        exit(2);
    }
    return c;
}

static void client_config_free(client_config *c) {
    free(c->prompt_owned);
    free(c->prefix_file_text);
}

/* ========================================================================= */
/* Framed connection IO                                                       */
/* ========================================================================= */

typedef struct {
    int fd;
    ap_buf rx;
    ap_buf scratch;
    FILE *trace;
} client_conn;

static void conn_init(client_conn *co, int fd, FILE *trace) {
    co->fd = fd;
    ap_buf_init(&co->rx);
    ap_buf_init(&co->scratch);
    co->trace = trace;
}

static void conn_free(client_conn *co) {
    ap_buf_free(&co->rx);
    ap_buf_free(&co->scratch);
    if (co->fd >= 0) close(co->fd);
    co->fd = -1;
}

static void conn_trace(client_conn *co, const char *dir, uint32_t type, size_t len) {
    if (!co->trace) return;
    fprintf(co->trace, "%s %s len=%zu\n", dir, ap_msg_name(type), len);
    fflush(co->trace);
}

static bool conn_send(client_conn *co, uint32_t type, const ap_buf *body) {
    ap_buf_reset(&co->scratch);
    if (!ap_frame_encode(&co->scratch, type, body)) return false;
    const char *p = (const char *)co->scratch.data;
    size_t n = co->scratch.len;
    while (n) {
        ssize_t k = send(co->fd, p, n, 0);
        if (k > 0) { p += (size_t)k; n -= (size_t)k; continue; }
        if (k < 0 && errno == EINTR) continue;
        return false;
    }
    conn_trace(co, "->", type, body ? body->len : 0);
    return true;
}

typedef enum { CONN_FRAME_OK, CONN_FRAME_CLOSED, CONN_FRAME_ERROR } conn_frame_result;

static conn_frame_result conn_recv(client_conn *co, uint32_t *type,
                                   ap_buf *payload_out, char *err, size_t errlen) {
    for (;;) {
        uint32_t t = 0;
        ap_frame_status fs = ap_read_frame(&co->rx, &t, payload_out, err, errlen);
        if (fs == AP_FRAME_OK) {
            *type = t;
            conn_trace(co, "<-", t, payload_out->len);
            return CONN_FRAME_OK;
        }
        if (fs == AP_FRAME_ERROR) return CONN_FRAME_ERROR;

        char buf[16384];
        ssize_t n = recv(co->fd, buf, sizeof(buf), 0);
        if (n == 0) return CONN_FRAME_CLOSED;
        if (n < 0) {
            if (errno == EINTR) continue;
            snprintf(err, errlen, "recv: %s", strerror(errno));
            return CONN_FRAME_ERROR;
        }
        if (!ap_buf_append(&co->rx, buf, (size_t)n)) {
            snprintf(err, errlen, "receive buffer allocation failed");
            return CONN_FRAME_ERROR;
        }
    }
}

/* ========================================================================= */
/* Connect + handshake                                                        */
/* ========================================================================= */

static int client_connect(const char *host, int port, char *err, size_t errlen) {
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_NUMERICSERV;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portstr, &hints, &res);
    if (gai != 0) {
        snprintf(err, errlen, "cannot resolve %s:%d: %s", host, port,
                 gai_strerror(gai));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        snprintf(err, errlen, "cannot connect to %s:%d: %s", host, port,
                 strerror(errno));
        return -1;
    }
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    return fd;
}

static void client_fill_session_new(const client_config *cfg,
                                    ap_session_new_args *a) {
    memset(a, 0, sizeof(*a));
    a->sys_text = cfg->system;
    a->sys_text_len = cfg->system ? strlen(cfg->system) : 0;
    a->prefix_file_text = cfg->prefix_file_text;
    a->prefix_file_text_len = cfg->prefix_file_len;
    a->temperature = ap_milli_from_double(cfg->temperature);
    a->temperature_set = cfg->temperature_set;
    a->top_p = ap_milli_from_double(cfg->top_p);
    a->top_p_set = cfg->top_p_set;
    a->min_p = ap_milli_from_double(cfg->min_p);
    a->min_p_set = cfg->min_p_set;
    a->seed = (uint32_t)cfg->seed;
    a->think_mode = (uint32_t)cfg->think_mode;
    a->n_predict = cfg->n_predict > 0 ? (uint32_t)cfg->n_predict : 0;
    a->raw_prompt = cfg->raw_prompt;
    a->hints_enabled = cfg->hints_enabled;
    a->dir_steering_ffn = ap_milli_from_double(cfg->dir_steering_ffn);
    a->power_set = cfg->power_set;
    a->power = (uint32_t)cfg->power;
}

/* Session capabilities the client keeps for the life of the connection. */
typedef struct {
    ap_hello_reply caps;
    ap_session_ready ready;
    bool resumed; /* entered a parked session rather than a fresh one */
} client_session;

static bool client_send_session(client_conn *co, const client_config *cfg,
                                bool resume) {
    ap_buf body;
    ap_buf_init(&body);
    if (resume) {
        ap_encode_session_simple(&body, AGENT_SESSION_RESUME);
    } else {
        ap_session_new_args a;
        client_fill_session_new(cfg, &a);
        ap_encode_session_new(&body, &a);
    }
    bool ok = conn_send(co, AGENT_MSG_SESSION, &body);
    ap_buf_free(&body);
    return ok;
}

/* Returns false with err set on any failure. On success *sess is filled. */
static bool client_handshake(client_conn *co, const client_config *cfg,
                             client_session *sess, char *err, size_t errlen) {
    memset(sess, 0, sizeof(*sess));

    /* 1. HELLO. */
    {
        char cwd[AP_CAP_PATH];
        if (!getcwd(cwd, sizeof(cwd))) cwd[0] = '\0';
        ap_hello h;
        memset(&h, 0, sizeof(h));
        h.proto_version = AGENT_PROTO_VERSION;
        snprintf(h.client_version, sizeof(h.client_version), "ds4-agent-client/%u",
                 (unsigned)AGENT_PROTO_VERSION);
        snprintf(h.cwd, sizeof(h.cwd), "%s", cwd);
        ap_buf body;
        ap_buf_init(&body);
        ap_encode_hello(&body, &h);
        bool ok = conn_send(co, AGENT_MSG_HELLO, &body);
        ap_buf_free(&body);
        if (!ok) { snprintf(err, errlen, "failed to send HELLO"); return false; }
    }

    /* 2. HELLO reply. */
    {
        uint32_t type = 0;
        ap_buf payload;
        ap_buf_init(&payload);
        conn_frame_result fr = conn_recv(co, &type, &payload, err, errlen);
        if (fr == CONN_FRAME_CLOSED) {
            snprintf(err, errlen, "server closed the connection during HELLO");
            ap_buf_free(&payload);
            return false;
        }
        if (fr == CONN_FRAME_ERROR) { ap_buf_free(&payload); return false; }
        if (type != AGENT_MSG_HELLO) {
            snprintf(err, errlen, "expected HELLO reply, got %s", ap_msg_name(type));
            ap_buf_free(&payload);
            return false;
        }
        ap_reader r;
        ap_reader_init(&r, payload.data, payload.len);
        bool ok = ap_decode_hello_reply(&r, &sess->caps, err, errlen);
        ap_buf_free(&payload);
        if (!ok) return false; /* server ERR (version mismatch / busy) -> err set */
        if (sess->caps.proto_version != AGENT_PROTO_VERSION) {
            snprintf(err, errlen,
                     "protocol mismatch (client %u, server %u): rebuild both binaries",
                     (unsigned)AGENT_PROTO_VERSION,
                     (unsigned)sess->caps.proto_version);
            return false;
        }
    }

    /* 3. SESSION resume (parked) or SESSION new; fall back to new if resume
     *    is rejected. */
    bool want_resume = sess->caps.session_parked;
    for (int attempt = 0; attempt < 2; attempt++) {
        if (!client_send_session(co, cfg, want_resume)) {
            snprintf(err, errlen, "failed to send SESSION %s",
                     want_resume ? "resume" : "new");
            return false;
        }
        uint32_t type = 0;
        ap_buf payload;
        ap_buf_init(&payload);
        conn_frame_result fr = conn_recv(co, &type, &payload, err, errlen);
        if (fr == CONN_FRAME_CLOSED) {
            snprintf(err, errlen, "server closed the connection during SESSION setup");
            ap_buf_free(&payload);
            return false;
        }
        if (fr == CONN_FRAME_ERROR) { ap_buf_free(&payload); return false; }
        if (type != AGENT_MSG_SESSION) {
            snprintf(err, errlen, "expected SESSION reply, got %s", ap_msg_name(type));
            ap_buf_free(&payload);
            return false;
        }
        ap_reader r;
        ap_reader_init(&r, payload.data, payload.len);
        char serr[AP_CAP_ERROR] = {0};
        bool ok = ap_decode_session_ready(&r, &sess->ready, serr, sizeof(serr));
        ap_buf_free(&payload);
        if (ok) {
            sess->resumed = want_resume;
            return true;
        }
        if (want_resume && attempt == 0) {
            want_resume = false; /* parked session gone: start fresh */
            continue;
        }
        snprintf(err, errlen, "%s", serr[0] ? serr : "SESSION setup failed");
        return false;
    }
    return false; /* unreachable */
}

/* ========================================================================= */
/* main                                                                       */
/* ========================================================================= */

#ifndef DS4_AGENT_CLIENT_TEST_NO_MAIN
int main(int argc, char **argv) {
    client_config cfg = client_parse_options(argc, argv);
    signal(SIGPIPE, SIG_IGN);

    if (cfg.chdir_path && chdir(cfg.chdir_path) != 0) {
        fprintf(stderr, "ds4-agent-client: chdir %s: %s\n",
                cfg.chdir_path, strerror(errno));
        client_config_free(&cfg);
        return 1;
    }

    FILE *trace = NULL;
    if (cfg.trace_path) {
        trace = fopen(cfg.trace_path, "w");
        if (!trace)
            fprintf(stderr, "ds4-agent-client: cannot open trace %s: %s\n",
                    cfg.trace_path, strerror(errno));
    }

    char err[512] = {0};
    int fd = client_connect(cfg.server_host, cfg.server_port, err, sizeof(err));
    if (fd < 0) {
        fprintf(stderr, "ds4-agent-client: %s\n", err);
        if (trace) fclose(trace);
        client_config_free(&cfg);
        return 1;
    }

    client_conn co;
    conn_init(&co, fd, trace);

    client_session sess;
    if (!client_handshake(&co, &cfg, &sess, err, sizeof(err))) {
        fprintf(stderr, "ds4-agent-client: %s\n", err);
        conn_free(&co);
        if (trace) fclose(trace);
        client_config_free(&cfg);
        return 1;
    }

    fprintf(stderr,
            "ds4-agent-client: connected to %s:%d — %s on %s, ctx %u (%s). "
            "The UI loop lands in T9-T11.\n",
            cfg.server_host, cfg.server_port,
            sess.caps.model_name[0] ? sess.caps.model_name : "model",
            sess.caps.backend_name[0] ? sess.caps.backend_name : "?",
            sess.ready.ctx_size,
            sess.resumed ? "resumed parked session" : "new session");

    {
        ap_buf b;
        ap_buf_init(&b);
        conn_send(&co, AGENT_MSG_STOP, &b);
        ap_buf_free(&b);
    }
    conn_free(&co);
    if (trace) fclose(trace);
    client_config_free(&cfg);
    return 0;
}
#endif
