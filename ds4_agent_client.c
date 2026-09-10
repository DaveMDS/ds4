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
#include <ctype.h>
#include <errno.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
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
/* T9: client render stack.                                                   */
/*                                                                           */
/* The markdown machine, syntax highlighter and tool visualiser are copied    */
/* verbatim from ds4_agent.c (they make no ds4_* calls). The token renderer   */
/* loses its engine/worker fields and the exact/anchored highlight branch. The     */
/* server's byte-by-byte classifier is gone: client_apply_stream_fragment     */
/* drives the visualiser straight from the (kind, text) STREAM fragments.     */
/* ========================================================================= */

/* Where painted bytes go. T11 repoints this at the linenoise editor. */
static void client_render_stdout(const char *s, size_t n) {
    write_all(STDOUT_FILENO, s, n);
}
static void (*g_render_sink)(const char *, size_t) = client_render_stdout;

static int client_read_default_lines(int ctx) {
    if (ctx > 0 && ctx <= 8192) return 120;
    if (ctx > 0 && ctx <= 16384) return 240;
    return 500;
}

static bool streq_any(const char *s, const char *a, const char *b,
                      const char *c, const char *d) {
    return (a && !strcmp(s, a)) || (b && !strcmp(s, b)) ||
           (c && !strcmp(s, c)) || (d && !strcmp(s, d));
}

/* -- render / viz types (trimmed copies from ds4_agent.c) ----------- */

typedef struct agent_tail_capture {
    char *buf;
    size_t cap;
    size_t start;
    size_t len;
    size_t total;
} agent_tail_capture;

typedef enum {
    AGENT_MD_PENDING_NONE,
    AGENT_MD_PENDING_BACKTICK,
} agent_markdown_pending;

typedef struct agent_syntax agent_syntax;

typedef struct {
    bool format_thinking;
    bool format_markdown;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    bool wrote_visible_output;
    bool md_bold;
    bool md_italic;
    bool md_inline_code;
    bool md_hint;
    bool md_line_started;
    char md_hint_prefix[sizeof("> **Hint:**") - 1];
    size_t md_hint_prefix_len;
    char md_inline[4096];
    size_t md_inline_len;
    bool md_escape;
    bool md_code_block;
    bool md_fence_info;
    bool md_code_line_start;
    bool md_code_in_ml_comment;
    bool md_syntax_silent;
    bool md_syntax_has_highlight;
    agent_markdown_pending md_pending;
    size_t md_pending_len;
    const agent_syntax *md_syntax;
    char md_fence_lang[32];
    size_t md_fence_lang_len;
    const char *md_code_line_prefix;
    const char *md_code_line_prefix_color;
    char *md_code_line;
    size_t md_code_line_len;
    size_t md_code_line_cap;
    char pending[16];
    size_t pending_len;
    char utf8_pending[4];
    size_t utf8_pending_len;
    size_t utf8_pending_need;
    agent_tail_capture *capture;
} agent_token_renderer;

typedef enum {
    AGENT_TOOL_PARAM_NORMAL,
    AGENT_TOOL_PARAM_PATH,
    AGENT_TOOL_PARAM_OFFSET,
    AGENT_TOOL_PARAM_CONTENT,
    AGENT_TOOL_PARAM_DIFF_OLD,
    AGENT_TOOL_PARAM_DIFF_NEW,
    AGENT_TOOL_PARAM_BASH_COMMAND,
} agent_tool_param_kind;

typedef struct {
    bool active;
    bool tool_announced;
    bool param_active;
    bool at_line_start;
    bool last_output_newline;
    agent_tool_param_kind param_kind;
    char tool_name[64];
    char param_name[64];
    char param_end_tail[64];
    size_t param_end_len;
    bool read_style;
    bool read_prefix_rendered;
    bool read_line_rendered;
    char read_path[512];
    char read_start[32];
    char read_max[32];
    char read_whole[8];
    char tool_path[512];
    bool code_param_active;
} agent_tool_visualizer;

/* Reduced stream renderer: no parser. The current tool / param come from the
 * TOOL_NAME / TOOL_PARAM_NAME fragments. */
typedef struct {
    agent_token_renderer *renderer;
    agent_tool_visualizer viz;
    int ctx_size;        /* authoritative, from the SESSION reply */
    char cur_tool[64];
    char cur_param[64];
    bool summary_banner_shown;
} agent_stream_renderer;


/* -- tail capture --------------------------------------------------- */

static void agent_tail_capture_append(agent_tail_capture *t,
                                      const char *s, size_t n) {
    if (!t || !n) return;
    if (!t->cap) return;
    if (!t->buf) t->buf = xmalloc(t->cap);
    t->total += n;

    if (n >= t->cap) {
        memcpy(t->buf, s + n - t->cap, t->cap);
        t->start = 0;
        t->len = t->cap;
        return;
    }

    if (t->len < t->cap) {
        size_t free_tail = t->cap - t->len;
        size_t first = n < free_tail ? n : free_tail;
        size_t pos = (t->start + t->len) % t->cap;
        size_t right = t->cap - pos;
        size_t chunk = first < right ? first : right;
        memcpy(t->buf + pos, s, chunk);
        if (first > chunk) memcpy(t->buf, s + chunk, first - chunk);
        t->len += first;
        s += first;
        n -= first;
    }

    while (n) {
        size_t pos = (t->start + t->len) % t->cap;
        size_t right = t->cap - pos;
        size_t chunk = n < right ? n : right;
        memcpy(t->buf + pos, s, chunk);
        t->start = (t->start + chunk) % t->cap;
        s += chunk;
        n -= chunk;
    }
}

static char *agent_tail_capture_take(agent_tail_capture *t, size_t *len) {
    size_t n = t ? t->len : 0;
    char *out = xmalloc(n + 1);
    if (n) {
        size_t right = t->cap - t->start;
        size_t first = n < right ? n : right;
        memcpy(out, t->buf + t->start, first);
        if (n > first) memcpy(out + first, t->buf, n - first);
    }
    out[n] = '\0';
    if (len) *len = n;
    free(t->buf);
    memset(t, 0, sizeof(*t));
    return out;
}


static void renderer_write(agent_token_renderer *r, const char *s, size_t n) {
    if (r->capture) agent_tail_capture_append(r->capture, s, n);
    else g_render_sink(s, n);
}

static void renderer_set_grey(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[38;5;245m", 11);
}

static void renderer_reset_color(agent_token_renderer *r) {
    if (r->use_color) renderer_write(r, "\x1b[0m", 4);
    r->color_open = false;
}

static size_t renderer_utf8_need(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}

static bool renderer_has_text_attrs(agent_token_renderer *r) {
    return r->in_think || r->md_bold || r->md_italic ||
           r->md_inline_code || r->md_code_block;
}

static void renderer_set_text_attrs(agent_token_renderer *r) {
    if (!r->use_color) return;
    if (r->in_think) {
        renderer_set_grey(r);
        return;
    }
    if (r->md_code_block) {
        renderer_write(r, "\x1b[38;5;75m", 10);
        return;
    } else if (r->md_inline_code) {
        renderer_write(r, "\x1b[36m", 5);
    }
    if (r->md_bold) renderer_write(r, "\x1b[1m", 4);
    if (r->md_italic) renderer_write(r, "\x1b[3m", 4);
}

static void renderer_restore_text_attrs(agent_token_renderer *r) {
    if (!r->use_color || !r->color_open || !renderer_has_text_attrs(r)) return;
    renderer_set_text_attrs(r);
}

static void renderer_write_complete_char_raw(agent_token_renderer *r, const char *s, size_t n) {
    bool styled = r->use_color && renderer_has_text_attrs(r);
    if (styled && !r->color_open) {
        renderer_set_text_attrs(r);
        r->color_open = true;
    } else if (!styled && r->color_open) {
        renderer_reset_color(r);
    }
    renderer_write(r, s, n);
    if (n) r->wrote_visible_output = true;
    r->last_output_newline = n == 1 && s[0] == '\n';
}

static void renderer_flush_utf8(agent_token_renderer *r) {
    if (!r->utf8_pending_len) return;
    renderer_write_complete_char_raw(r, r->utf8_pending, r->utf8_pending_len);
    r->utf8_pending_len = 0;
    r->utf8_pending_need = 0;
}

static void renderer_write_char_raw(agent_token_renderer *r, char c) {
    unsigned char uc = (unsigned char)c;

    if (r->utf8_pending_len) {
        if ((uc & 0xc0) == 0x80 && r->utf8_pending_len < sizeof(r->utf8_pending)) {
            r->utf8_pending[r->utf8_pending_len++] = c;
            if (r->utf8_pending_len == r->utf8_pending_need) renderer_flush_utf8(r);
            return;
        }
        renderer_flush_utf8(r);
    }

    size_t need = renderer_utf8_need(uc);
    if (need == 1) {
        renderer_write_complete_char_raw(r, &c, 1);
        return;
    }
    r->utf8_pending[0] = c;
    r->utf8_pending_len = 1;
    r->utf8_pending_need = need;
}

static void renderer_write_plain_byte(agent_token_renderer *r, char c) {
    bool old_bold = r->md_bold;
    bool old_italic = r->md_italic;
    bool old_inline_code = r->md_inline_code;
    bool old_code_block = r->md_code_block;

    /* Code blocks are streamed immediately in plain text, then repainted with
     * syntax colors when a complete terminal-safe line is available.  Disable
     * markdown attributes only for this byte; renderer_write_char_raw() will
     * reset any tracked manual color once if needed. */
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    renderer_write_char_raw(r, c);
    r->md_bold = old_bold;
    r->md_italic = old_italic;
    r->md_inline_code = old_inline_code;
    r->md_code_block = old_code_block;
}

/* Poor man's code highlighter inspired by antirez/kilo: a tiny language table
 * plus one line-oriented tokenizer for comments, strings, numbers, and
 * separator-bounded keywords.  This is deliberately not a full parser; it is
 * only for making fenced Markdown code readable in the terminal. */
#define AGENT_HL_NORMAL 0
#define AGENT_HL_COMMENT 1
#define AGENT_HL_KEYWORD1 2
#define AGENT_HL_KEYWORD2 3
#define AGENT_HL_STRING 4
#define AGENT_HL_NUMBER 5

#define AGENT_SYNTAX_NUMBERS (1u<<0)
#define AGENT_SYNTAX_STRINGS (1u<<1)
#define AGENT_SYNTAX_BACKTICK_STRINGS (1u<<2)
#define AGENT_SYNTAX_CASE_INSENSITIVE (1u<<3)

struct agent_syntax {
    const char *name;
    const char *aliases;
    const char **keywords;
    const char *singleline_comments[3];
    const char *multiline_start;
    const char *multiline_end;
    unsigned flags;
};

static const char *agent_kw_generic[] = {
    "if","else","for","while","do","switch","case","default","break",
    "continue","return","try","catch","finally","throw","throws","class",
    "struct","enum","interface","trait","impl","fn","func","function",
    "def","lambda","let","var","const","static","public","private",
    "protected","import","include","from","export","package","module",
    "namespace","new","delete","async","await","yield","match","type",
    "true|","false|","null|","nil|","none|","None|","NULL|","void|",
    "int|","long|","float|","double|","char|","bool|","string|",
    "String|","usize|","isize|","u8|","u16|","u32|","u64|","i8|",
    "i16|","i32|","i64|",NULL
};

static const char *agent_kw_c[] = {
    "auto","break","case","continue","default","do","else","enum",
    "extern","for","goto","if","register","return","sizeof","static",
    "struct","switch","typedef","union","volatile","while",
    "alignas","alignof","and","and_eq","asm","bitand","bitor","class",
    "compl","constexpr","const_cast","decltype","delete","dynamic_cast",
    "explicit","export","false","friend","inline","mutable","namespace",
    "new","noexcept","not","not_eq","nullptr","operator","or","or_eq",
    "private","protected","public","reinterpret_cast","static_assert",
    "static_cast","template","this","thread_local","throw","true","try",
    "typeid","typename","virtual","xor","xor_eq",
    "NULL|","bool|","char|","const|","double|","float|","int|","long|",
    "short|","signed|","size_t|","ssize_t|","uint8_t|","uint16_t|",
    "uint32_t|","uint64_t|","unsigned|","void|",NULL
};

static const char *agent_kw_python[] = {
    "and","as","assert","async","await","break","case","class","continue",
    "def","del","elif","else","except","finally","for","from","global",
    "if","import","in","is","lambda","match","nonlocal","not","or","pass",
    "raise","return","try","while","with","yield",
    "False|","None|","True|","bool|","bytes|","dict|","float|","int|",
    "list|","object|","set|","str|","tuple|",NULL
};

static const char *agent_kw_js[] = {
    "async","await","break","case","catch","class","const","continue",
    "debugger","default","delete","do","else","export","extends",
    "finally","for","from","function","get","if","import","in",
    "instanceof","let","new","of","return","set","static","super",
    "switch","this","throw","try","typeof","var","void","while","with",
    "yield","abstract","as","declare","enum","implements","interface",
    "keyof","namespace","private","protected","public","readonly","type",
    "any|","boolean|","false|","never|","null|","number|","string|",
    "symbol|","true|","undefined|","unknown|","void|",NULL
};

static const char *agent_kw_java[] = {
    "abstract","assert","break","case","catch","class","const","continue",
    "default","do","else","enum","extends","final","finally","for","goto",
    "if","implements","import","instanceof","interface","native","new",
    "package","private","protected","public","return","static","strictfp",
    "super","switch","synchronized","this","throw","throws","transient",
    "try","volatile","while",
    "boolean|","byte|","char|","double|","false|","float|","int|","long|",
    "null|","short|","true|","void|",NULL
};

static const char *agent_kw_csharp[] = {
    "abstract","as","base","break","case","catch","checked","class","const",
    "continue","default","delegate","do","else","enum","event","explicit",
    "extern","finally","fixed","for","foreach","goto","if","implicit","in",
    "interface","internal","is","lock","namespace","new","operator","out",
    "override","params","private","protected","public","readonly","ref",
    "return","sealed","sizeof","stackalloc","static","struct","switch",
    "this","throw","try","typeof","unchecked","unsafe","using","virtual",
    "volatile","while","async","await","get","init","record","set","var",
    "bool|","byte|","char|","decimal|","double|","false|","float|","int|",
    "long|","null|","object|","sbyte|","short|","string|","true|","uint|",
    "ulong|","ushort|","void|",NULL
};

static const char *agent_kw_go[] = {
    "break","case","chan","const","continue","default","defer","else",
    "fallthrough","for","func","go","goto","if","import","interface",
    "map","package","range","return","select","struct","switch","type",
    "var","bool|","byte|","complex64|","complex128|","error|","false|",
    "float32|","float64|","int|","int8|","int16|","int32|","int64|",
    "nil|","rune|","string|","true|","uint|","uint8|","uint16|",
    "uint32|","uint64|","uintptr|",NULL
};

static const char *agent_kw_rust[] = {
    "as","async","await","break","const","continue","crate","dyn","else",
    "enum","extern","fn","for","if","impl","in","let","loop","match",
    "mod","move","mut","pub","ref","return","self","Self","static",
    "struct","super","trait","type","unsafe","use","where","while",
    "bool|","char|","false|","f32|","f64|","i8|","i16|","i32|","i64|",
    "i128|","isize|","str|","String|","true|","u8|","u16|","u32|",
    "u64|","u128|","usize|",NULL
};

static const char *agent_kw_shell[] = {
    "case","do","done","elif","else","esac","fi","for","function","if",
    "in","select","then","time","until","while","break","continue",
    "return","export","local","readonly","source","test","true|","false|",
    "echo|","printf|","cd|","pwd|","read|","set|","unset|","shift|",NULL
};

static const char *agent_kw_sql[] = {
    "add","alter","and","as","asc","between","by","case","check","column",
    "constraint","create","delete","desc","distinct","drop","else","end",
    "exists","foreign","from","group","having","in","index","insert",
    "into","is","join","key","left","like","limit","not","null","on",
    "or","order","outer","primary","references","right","select","set",
    "table","then","union","unique","update","values","view","where",
    "bigint|","boolean|","date|","decimal|","false|","int|","integer|",
    "numeric|","real|","text|","timestamp|","true|","varchar|",NULL
};

static const char *agent_kw_ruby[] = {
    "BEGIN","END","alias","and","begin","break","case","class","def",
    "defined?","do","else","elsif","end","ensure","for","if","in",
    "module","next","not","or","redo","rescue","retry","return","self",
    "super","then","undef","unless","until","when","while","yield",
    "false|","nil|","true|",NULL
};

static const char *agent_kw_php[] = {
    "abstract","and","array","as","break","callable","case","catch","class",
    "clone","const","continue","declare","default","die","do","echo","else",
    "elseif","empty","enddeclare","endfor","endforeach","endif","endswitch",
    "endwhile","eval","exit","extends","final","finally","fn","for",
    "foreach","function","global","goto","if","implements","include",
    "include_once","instanceof","insteadof","interface","isset","list",
    "match","namespace","new","or","print","private","protected","public",
    "readonly","require","require_once","return","static","switch","throw",
    "trait","try","unset","use","var","while","xor","bool|","false|",
    "float|","int|","null|","string|","true|","void|",NULL
};

static const char *agent_kw_swift[] = {
    "actor","as","associatedtype","async","await","break","case","catch",
    "class","continue","default","defer","do","else","enum","extension",
    "fallthrough","for","func","guard","if","import","in","init","inout",
    "is","let","nonisolated","operator","private","protocol","public",
    "repeat","return","self","Self","static","struct","subscript","super",
    "switch","throw","throws","try","typealias","var","where","while",
    "Any|","Bool|","Double|","false|","Float|","Int|","nil|","String|",
    "true|","Void|",NULL
};

static const char *agent_kw_kotlin[] = {
    "as","break","class","continue","do","else","false","for","fun","if",
    "in","interface","is","null","object","package","return","super",
    "this","throw","true","try","typealias","typeof","val","var","when",
    "while","actual","annotation","by","catch","companion","const",
    "constructor","crossinline","data","enum","expect","external","final",
    "finally","import","infix","init","inline","inner","internal","lateinit",
    "noinline","open","operator","out","override","private","protected",
    "public","reified","sealed","suspend","tailrec","vararg",
    "Any|","Boolean|","Byte|","Char|","Double|","Float|","Int|","Long|",
    "Short|","String|","Unit|",NULL
};

static const char *agent_kw_zig[] = {
    "addrspace","align","allowzero","and","anyframe","anytype","asm",
    "async","await","break","callconv","catch","comptime","const",
    "continue","defer","else","enum","errdefer","error","export","extern",
    "fn","for","if","inline","linksection","noalias","noinline","nosuspend",
    "opaque","or","orelse","packed","pub","resume","return","struct",
    "suspend","switch","test","threadlocal","try","union","unreachable",
    "usingnamespace","var","volatile","while",
    "bool|","false|","f32|","f64|","i32|","i64|","null|","true|","u8|",
    "u16|","u32|","u64|","usize|","void|",NULL
};

static const char *agent_kw_lua[] = {
    "and","break","do","else","elseif","end","false","for","function",
    "goto","if","in","local","nil","not","or","repeat","return","then",
    "true","until","while",NULL
};

static const char *agent_kw_html[] = {
    "a","body","button","div","doctype","form","h1","h2","h3","head",
    "html","input","label","li","link","main","meta","ol","option","p",
    "script","section","select","span","style","table","tbody","td","th",
    "thead","title","tr","ul","class|","href|","id|","name|","rel|",
    "src|","type|","value|",NULL
};

static const char *agent_kw_css[] = {
    "align-items","background","border","bottom","color","display","flex",
    "font","font-size","gap","grid","height","justify-content","left",
    "margin","max-width","min-width","padding","position","right","top",
    "transform","width","z-index","absolute|","auto|","block|","flex|",
    "grid|","hidden|","inline|","none|","relative|","solid|",NULL
};

static const agent_syntax agent_syntaxes[] = {
    {"generic", " text txt", agent_kw_generic, {"//","#",NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"c", " c h cpp c++ cc cxx hpp hxx objc objective-c", agent_kw_c, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"python", " py python py3", agent_kw_python, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"javascript", " js jsx javascript typescript ts tsx node mjs cjs", agent_kw_js, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"java", " java", agent_kw_java, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"csharp", " cs c# csharp dotnet", agent_kw_csharp, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"go", " go golang", agent_kw_go, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"rust", " rs rust", agent_kw_rust, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"shell", " sh bash zsh shell fish ksh", agent_kw_shell, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_BACKTICK_STRINGS},
    {"sql", " sql postgres mysql sqlite", agent_kw_sql, {"--",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS | AGENT_SYNTAX_CASE_INSENSITIVE},
    {"ruby", " rb ruby", agent_kw_ruby, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"php", " php", agent_kw_php, {"//","#",NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"swift", " swift", agent_kw_swift, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"kotlin", " kt kts kotlin", agent_kw_kotlin, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"zig", " zig", agent_kw_zig, {"//",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"lua", " lua", agent_kw_lua, {"--",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"html", " html htm xml svg", agent_kw_html, {NULL,NULL,NULL}, "<!--", "-->",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"css", " css scss sass", agent_kw_css, {NULL,NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"json", " json jsonc", NULL, {"//",NULL,NULL}, "/*", "*/",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"yaml", " yaml yml toml ini", NULL, {"#",NULL,NULL}, NULL, NULL,
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {"markdown", " md markdown", agent_kw_generic, {NULL,NULL,NULL}, "<!--", "-->",
        AGENT_SYNTAX_NUMBERS | AGENT_SYNTAX_STRINGS},
    {NULL, NULL, NULL, {NULL,NULL,NULL}, NULL, NULL, 0}
};

static bool agent_syntax_alias_match(const char *aliases, const char *lang) {
    if (!aliases || !lang || !lang[0]) return false;
    size_t llen = strlen(lang);
    const char *p = aliases;
    while (*p) {
        while (*p == ' ') p++;
        const char *start = p;
        while (*p && *p != ' ') p++;
        if ((size_t)(p - start) == llen && !strncasecmp(start, lang, llen))
            return true;
    }
    return false;
}

static const agent_syntax *agent_syntax_for_lang(const char *lang) {
    if (lang && lang[0]) {
        for (const agent_syntax *s = agent_syntaxes; s->name; s++) {
            if (!strcasecmp(s->name, lang) ||
                agent_syntax_alias_match(s->aliases, lang))
                return s;
        }
    }
    return &agent_syntaxes[0];
}

static const agent_syntax *agent_syntax_for_path(const char *path) {
    if (!path || !path[0]) return agent_syntax_for_lang(NULL);
    const char *base = strrchr(path, '/');
    base = base ? base + 1 : path;
    if (!strcasecmp(base, "Dockerfile")) return agent_syntax_for_lang("sh");
    if (!strcasecmp(base, "Makefile") || !strcasecmp(base, "makefile"))
        return agent_syntax_for_lang("sh");
    const char *dot = strrchr(base, '.');
    if (!dot || !dot[1]) return agent_syntax_for_lang(NULL);
    return agent_syntax_for_lang(dot + 1);
}

static bool agent_syntax_separator(char c) {
    unsigned char uc = (unsigned char)c;
    return c == '\0' || isspace(uc) || strchr(",.()+-/*=~%[]{}<>:;!&|^?", c) != NULL;
}

static const char *agent_syntax_line_comment(const agent_syntax *syn,
                                             const char *p) {
    if (!syn) return NULL;
    for (int i = 0; i < 3 && syn->singleline_comments[i]; i++) {
        const char *m = syn->singleline_comments[i];
        size_t mlen = strlen(m);
        if (mlen && !strncmp(p, m, mlen)) return m;
    }
    return NULL;
}

static int agent_syntax_color(int hl) {
    switch (hl) {
    case AGENT_HL_COMMENT: return 244;
    case AGENT_HL_KEYWORD1: return 214;
    case AGENT_HL_KEYWORD2: return 81;
    case AGENT_HL_STRING: return 150;
    case AGENT_HL_NUMBER: return 203;
    default: return 252;
    }
}

static void renderer_syntax_write(agent_token_renderer *r, int hl,
                                  const char *s, size_t n) {
    if (!n) return;
    if (hl != AGENT_HL_NORMAL) r->md_syntax_has_highlight = true;
    if (r->md_syntax_silent) return;
    if (r->use_color && hl != AGENT_HL_NORMAL) {
        char seq[32];
        snprintf(seq, sizeof(seq), "\x1b[38;5;%dm", agent_syntax_color(hl));
        renderer_write(r, seq, strlen(seq));
    }
    renderer_write(r, s, n);
    if (r->use_color && hl != AGENT_HL_NORMAL) renderer_write(r, "\x1b[0m", 4);
    r->wrote_visible_output = true;
    r->last_output_newline = false;
}

static size_t agent_syntax_keyword_len(const char *kw, bool *secondary) {
    size_t len = strlen(kw);
    *secondary = len && kw[len - 1] == '|';
    return *secondary ? len - 1 : len;
}

static bool agent_syntax_match_keyword(const agent_syntax *syn,
                                       const char *p,
                                       const char *line_end,
                                       size_t *out_len,
                                       int *out_hl) {
    if (!syn || !syn->keywords) return false;
    for (int i = 0; syn->keywords[i]; i++) {
        bool secondary = false;
        size_t klen = agent_syntax_keyword_len(syn->keywords[i], &secondary);
        if ((size_t)(line_end - p) < klen) continue;
        bool match = (syn->flags & AGENT_SYNTAX_CASE_INSENSITIVE) ?
            !strncasecmp(p, syn->keywords[i], klen) :
            !strncmp(p, syn->keywords[i], klen);
        if (!match) continue;
        if (!agent_syntax_separator(p[klen])) continue;
        *out_len = klen;
        *out_hl = secondary ? AGENT_HL_KEYWORD2 : AGENT_HL_KEYWORD1;
        return true;
    }
    return false;
}

static bool agent_syntax_number_start(const char *p, const char *line,
                                      bool prev_sep, int prev_hl) {
    unsigned char c = (unsigned char)*p;
    if (isdigit(c) && (prev_sep || prev_hl == AGENT_HL_NUMBER)) return true;
    if (*p == '.' && p > line && prev_hl == AGENT_HL_NUMBER) return true;
    return false;
}

static size_t agent_syntax_number_len(const char *p, const char *line_end) {
    const char *q = p;
    while (q < line_end) {
        unsigned char c = (unsigned char)*q;
        if (isalnum(c) || *q == '_' || *q == '.' || *q == '+' || *q == '-') q++;
        else break;
    }
    return (size_t)(q - p);
}

static void renderer_syntax_emit_line(agent_token_renderer *r,
                                      const char *line, size_t len) {
    const agent_syntax *syn = r->md_syntax ? r->md_syntax : agent_syntax_for_lang(NULL);
    const char *p = line;
    const char *end = line + len;
    bool prev_sep = true;
    int prev_hl = AGENT_HL_NORMAL;
    int in_string = 0;

    while (p < end) {
        if (r->md_code_in_ml_comment) {
            const char *mce = syn->multiline_end;
            if (mce && *mce) {
                size_t mlen = strlen(mce);
                const char *q = p;
                while (q < end && ((size_t)(end - q) < mlen ||
                       strncmp(q, mce, mlen))) q++;
                if (q < end) {
                    q += mlen;
                    renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(q - p));
                    p = q;
                    r->md_code_in_ml_comment = false;
                    prev_sep = true;
                    prev_hl = AGENT_HL_COMMENT;
                    continue;
                }
            }
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(end - p));
            return;
        }

        const char *scs = agent_syntax_line_comment(syn, p);
        if (!in_string && scs) {
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(end - p));
            return;
        }

        if (!in_string && syn->multiline_start && syn->multiline_end &&
            !strncmp(p, syn->multiline_start, strlen(syn->multiline_start))) {
            size_t mlen = strlen(syn->multiline_start);
            const char *q = p + mlen;
            size_t elen = strlen(syn->multiline_end);
            while (q < end && ((size_t)(end - q) < elen ||
                   strncmp(q, syn->multiline_end, elen))) q++;
            if (q < end) q += elen;
            else r->md_code_in_ml_comment = true;
            renderer_syntax_write(r, AGENT_HL_COMMENT, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_COMMENT;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_STRINGS) && in_string) {
            const char *q = p;
            while (q < end) {
                if (*q == '\\' && q + 1 < end) {
                    q += 2;
                    continue;
                }
                q++;
                if (q[-1] == in_string) {
                    in_string = 0;
                    break;
                }
            }
            renderer_syntax_write(r, AGENT_HL_STRING, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_STRING;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_STRINGS) &&
            (*p == '"' || *p == '\'' ||
             ((syn->flags & AGENT_SYNTAX_BACKTICK_STRINGS) && *p == '`'))) {
            int quote = *p;
            const char *q = p + 1;
            while (q < end) {
                if (*q == '\\' && q + 1 < end) {
                    q += 2;
                    continue;
                }
                q++;
                if (q[-1] == quote) {
                    break;
                }
            }
            renderer_syntax_write(r, AGENT_HL_STRING, p, (size_t)(q - p));
            p = q;
            prev_sep = false;
            prev_hl = AGENT_HL_STRING;
            continue;
        }

        if ((syn->flags & AGENT_SYNTAX_NUMBERS) &&
            agent_syntax_number_start(p, line, prev_sep, prev_hl)) {
            size_t nlen = agent_syntax_number_len(p, end);
            renderer_syntax_write(r, AGENT_HL_NUMBER, p, nlen);
            p += nlen;
            prev_sep = false;
            prev_hl = AGENT_HL_NUMBER;
            continue;
        }

        if (prev_sep) {
            size_t klen = 0;
            int khl = AGENT_HL_NORMAL;
            if (agent_syntax_match_keyword(syn, p, end, &klen, &khl)) {
                renderer_syntax_write(r, khl, p, klen);
                p += klen;
                prev_sep = false;
                prev_hl = khl;
                continue;
            }
        }

        renderer_syntax_write(r, AGENT_HL_NORMAL, p, 1);
        prev_sep = agent_syntax_separator(*p);
        prev_hl = AGENT_HL_NORMAL;
        p++;
    }
}

static void renderer_code_line_append(agent_token_renderer *r,
                                      const char *s, size_t n) {
    if (!n) return;
    if (r->md_code_line_len + n + 1 > r->md_code_line_cap) {
        size_t cap = r->md_code_line_cap ? r->md_code_line_cap * 2 : 256;
        while (cap < r->md_code_line_len + n + 1) cap *= 2;
        r->md_code_line = xrealloc(r->md_code_line, cap);
        r->md_code_line_cap = cap;
    }
    memcpy(r->md_code_line + r->md_code_line_len, s, n);
    r->md_code_line_len += n;
    r->md_code_line[r->md_code_line_len] = '\0';
}

static int renderer_terminal_cols(void) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0)
        return ws.ws_col;
    return 80;
}

static bool renderer_code_line_can_repaint(agent_token_renderer *r) {
    if (!r->use_color || r->capture || r->md_code_line_len == 0) return false;
    int cols = renderer_terminal_cols();
    size_t prefix_len = r->md_code_line_prefix ?
                        strlen(r->md_code_line_prefix) : 0;
    if (cols <= 1 || prefix_len + r->md_code_line_len >= (size_t)cols)
        return false;
    for (size_t i = 0; i < r->md_code_line_len; i++) {
        unsigned char c = (unsigned char)r->md_code_line[i];
        if (c == '\t' || c == 0x1b || c >= 0x80 || (c < 0x20 && c != '\r'))
            return false;
    }
    return true;
}

static void renderer_code_write_line_prefix(agent_token_renderer *r) {
    if (!r->md_code_line_prefix) return;
    if (r->use_color && r->md_code_line_prefix_color)
        renderer_write(r, r->md_code_line_prefix_color,
                       strlen(r->md_code_line_prefix_color));
    renderer_write(r, r->md_code_line_prefix,
                   strlen(r->md_code_line_prefix));
    if (r->use_color && r->md_code_line_prefix_color)
        renderer_write(r, "\x1b[0m", 4);
    r->color_open = false;
}

/* Run the syntax highlighter in silent mode to learn whether the already
 * streamed line would change if repainted, while preserving the multiline
 * comment state until the caller decides whether repaint is safe. */
static bool renderer_code_scan_line(agent_token_renderer *r,
                                    bool *final_ml_comment) {
    bool old_silent = r->md_syntax_silent;
    bool old_highlight = r->md_syntax_has_highlight;
    bool old_ml_comment = r->md_code_in_ml_comment;

    r->md_syntax_silent = true;
    r->md_syntax_has_highlight = false;
    renderer_syntax_emit_line(r, r->md_code_line, r->md_code_line_len);
    bool changed = r->md_syntax_has_highlight;
    *final_ml_comment = r->md_code_in_ml_comment;

    r->md_code_in_ml_comment = old_ml_comment;
    r->md_syntax_silent = old_silent;
    r->md_syntax_has_highlight = old_highlight;
    return changed;
}

/* Code is shown as soon as bytes arrive.  At end-of-line we can cheaply
 * replace only that terminal row with syntax-highlighted text, but only for
 * simple one-row ASCII lines; long, tabbed, escaped, or UTF-8 lines are left
 * as streamed and only advance the highlighter state. */
static void renderer_code_emit_buffered_line(agent_token_renderer *r,
                                             bool with_newline) {
    bool final_ml_comment = r->md_code_in_ml_comment;
    bool changed = renderer_code_scan_line(r, &final_ml_comment);
    bool repaint = changed && renderer_code_line_can_repaint(r);
    if (repaint) {
        agent_tail_capture frame = {
            .cap = 64 * (r->md_code_line_len +
                         (r->md_code_line_prefix ? strlen(r->md_code_line_prefix) : 0) + 1) + 256
        };
        agent_tail_capture *capture = r->capture;
        r->capture = &frame;
        renderer_reset_color(r);
        renderer_write(r, "\r\x1b[0K", 5);
        renderer_code_write_line_prefix(r);
        renderer_syntax_emit_line(r, r->md_code_line, r->md_code_line_len);
        r->capture = capture;
        size_t len;
        char *text = agent_tail_capture_take(&frame, &len);
        renderer_write(r, text, len);
        free(text);
    } else {
        r->md_code_in_ml_comment = final_ml_comment;
    }
    r->md_code_line_len = 0;
    if (with_newline) {
        renderer_write_plain_byte(r, '\n');
        r->wrote_visible_output = true;
        r->last_output_newline = true;
        r->md_code_line_start = true;
    }
}

static void renderer_code_byte(agent_token_renderer *r, char c) {
    if (c == '\n') {
        renderer_code_emit_buffered_line(r, true);
        return;
    }
    renderer_code_line_append(r, &c, 1);
    renderer_write_plain_byte(r, c);
    if (c != ' ' && c != '\t' && c != '\r') r->md_code_line_start = false;
}

static void renderer_code_emit_backtick_literals(agent_token_renderer *r,
                                                 size_t count) {
    for (size_t i = 0; i < count; i++) renderer_code_byte(r, '`');
}

static void renderer_code_begin(agent_token_renderer *r) {
    renderer_reset_color(r);
    r->md_code_block = true;
    r->md_inline_code = false;
    r->md_fence_info = true;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = agent_syntax_for_lang(NULL);
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_line_len = 0;
}

static void renderer_code_stream_begin(agent_token_renderer *r,
                                       const agent_syntax *syntax) {
    renderer_reset_color(r);
    r->md_code_block = true;
    r->md_inline_code = false;
    r->md_fence_info = false;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = syntax ? syntax : agent_syntax_for_lang(NULL);
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    r->md_code_line_len = 0;
}

static void renderer_code_stream_set_prefix(agent_token_renderer *r,
                                            const char *prefix,
                                            const char *color) {
    r->md_code_line_prefix = prefix;
    r->md_code_line_prefix_color = color;
}

static void renderer_code_end(agent_token_renderer *r) {
    bool only_space = true;
    for (size_t i = 0; i < r->md_code_line_len; i++) {
        if (r->md_code_line[i] != ' ' && r->md_code_line[i] != '\t' &&
            r->md_code_line[i] != '\r') {
            only_space = false;
            break;
        }
    }
    if (r->md_code_line_len && !only_space)
        renderer_code_emit_buffered_line(r, false);
    else
        r->md_code_line_len = 0;
    r->md_code_block = false;
    r->md_inline_code = false;
    r->md_fence_info = false;
    r->md_code_line_start = true;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
}

/* Tool visualization and redirected output bypass Markdown formatting. */
static void renderer_markdown_clear_pending(agent_token_renderer *r) {
    r->md_pending = AGENT_MD_PENDING_NONE;
    r->md_pending_len = 0;
}

static bool renderer_markdown_inline_pair(agent_token_renderer *r, size_t *delimiter) {
    size_t n = r->md_inline_len, open = 1, close = 0;
    char ch = r->md_inline[0];
    while (open < n && r->md_inline[open] == ch) open++;
    while (close < n && r->md_inline[n - close - 1] == ch) close++;
    *delimiter = open;
    if (open > 2 || close != open || n <= open + close) return false;
    if (ch == '*') {
        if (isspace((unsigned char)r->md_inline[open]) ||
            isspace((unsigned char)r->md_inline[n - close - 1])) return false;
        size_t escapes = 0, pos = n - close;
        while (pos && r->md_inline[--pos] == '\\') escapes++;
        if (escapes & 1) return false;
    }
    return true;
}

static void renderer_markdown_inline_flush(agent_token_renderer *r, bool format) {
    size_t n = r->md_inline_len, delimiter = 0;
    if (!n) return;
    bool paired = format && renderer_markdown_inline_pair(r, &delimiter);
    bool code = r->md_inline[0] == '`' && paired;
    r->md_inline_code = code;
    r->md_bold = paired && !code && delimiter == 2;
    r->md_italic = paired && !code && delimiter == 1;
    size_t from = paired ? delimiter : 0, to = paired ? n - delimiter : n;
    for (size_t i = from; i < to; i++) {
        if (!code && r->md_inline[i] == '\\' && i + 1 < to &&
            ispunct((unsigned char)r->md_inline[i + 1])) i++;
        renderer_write_char_raw(r, r->md_inline[i]);
    }
    r->md_inline_len = 0;
    r->md_inline_code = r->md_bold = r->md_italic = false;
    renderer_reset_color(r);
}

static void renderer_hint_flush_prefix(agent_token_renderer *r) {
    for (size_t i = 0; i < r->md_hint_prefix_len; i++)
        renderer_write_char_raw(r, r->md_hint_prefix[i]);
    r->md_hint_prefix_len = 0;
}

static void renderer_hint_end(agent_token_renderer *r) {
    renderer_hint_flush_prefix(r);
    if (r->md_hint) renderer_reset_color(r);
    r->md_hint = false;
    r->md_line_started = false;
}

static void renderer_markdown_emit_pending_literals(agent_token_renderer *r) {
    renderer_hint_flush_prefix(r);
    renderer_markdown_inline_flush(r, false);
    if (r->md_escape) {
        renderer_write_char_raw(r, '\\');
        r->md_escape = false;
    }
    size_t count = r->md_pending_len;
    renderer_markdown_clear_pending(r);
    if (r->md_code_block) renderer_code_emit_backtick_literals(r, count);
    else for (size_t i = 0; i < count; i++) renderer_write_char_raw(r, '`');
}

static void renderer_markdown_commit_backticks(agent_token_renderer *r) {
    size_t count = r->md_pending_len;
    renderer_markdown_clear_pending(r);
    if (count >= 3) {
        for (size_t i = 0; i < count; i++) renderer_write_plain_byte(r, '`');
        if (r->md_code_block) renderer_code_end(r);
        else renderer_code_begin(r);
    } else if (r->md_code_block) {
        renderer_code_emit_backtick_literals(r, count);
    }
}

/* Plain text streams immediately. An ambiguous inline span is held until its
 * closing delimiter, a newline, or 4 KiB; without a pair it is printed literally.
 * Code fences keep their existing line-at-a-time streaming highlighter. */
static void renderer_markdown_feed(agent_token_renderer *r, char c) {
    if (r->md_fence_info) {
        if (c == '\n') {
            if (r->md_code_block) {
                r->md_fence_lang[r->md_fence_lang_len] = '\0';
                r->md_syntax = agent_syntax_for_lang(r->md_fence_lang);
            }
            renderer_write_plain_byte(r, '\n');
            r->md_fence_info = false;
        } else if (r->md_code_block) {
            unsigned char uc = (unsigned char)c;
            if (r->md_fence_lang_len + 1 < sizeof(r->md_fence_lang) &&
                (isalnum(uc) || c == '_' || c == '-' || c == '+' || c == '#'))
                r->md_fence_lang[r->md_fence_lang_len++] = c;
            renderer_write_plain_byte(r, c);
        }
        return;
    }
    if (r->md_pending == AGENT_MD_PENDING_BACKTICK) {
        if (c == '`') { r->md_pending_len++; return; }
        renderer_markdown_commit_backticks(r);
        renderer_markdown_feed(r, c);
        return;
    }
    if (r->md_code_block) {
        if (c == '`' && r->md_code_line_start) {
            r->md_pending = AGENT_MD_PENDING_BACKTICK;
            r->md_pending_len = 1;
        } else renderer_code_byte(r, c);
        return;
    }
    if (r->md_inline_len) {
        size_t delimiter;
        if (c != r->md_inline[0] && renderer_markdown_inline_pair(r, &delimiter)) {
            renderer_markdown_inline_flush(r, true);
            renderer_markdown_feed(r, c);
            return;
        }
        size_t n = r->md_inline_len, run = 0;
        while (run < n && r->md_inline[run] == r->md_inline[0]) run++;
        if (run == n && c != r->md_inline[0]) {
            if (r->md_inline[0] == '`' && run >= 3) {
                r->md_inline_len = 0;
                for (size_t i = 0; i < run; i++) renderer_write_plain_byte(r, '`');
                renderer_code_begin(r);
                renderer_markdown_feed(r, c);
                return;
            }
            if (isspace((unsigned char)c) || run > 2) {
                renderer_markdown_inline_flush(r, false);
                renderer_markdown_feed(r, c);
                return;
            }
        }
        if (c == '\n' || r->md_inline_len == sizeof(r->md_inline)) {
            renderer_markdown_inline_flush(r, false);
            renderer_markdown_feed(r, c);
            return;
        }
        r->md_inline[r->md_inline_len++] = c;
        return;
    }
    if (r->md_escape) {
        r->md_escape = false;
        if (ispunct((unsigned char)c)) { renderer_write_char_raw(r, c); return; }
        renderer_write_char_raw(r, '\\');
    }
    if (c == '\\') { r->md_escape = true; return; }
    if (c == '*' || c == '`') {
        r->md_inline[0] = c;
        r->md_inline_len = 1;
        return;
    }
    renderer_write_char_raw(r, c);
}

static void renderer_markdown_finish(agent_token_renderer *r) {
    renderer_hint_flush_prefix(r);
    renderer_markdown_inline_flush(r, true);
    /* A closing code fence can be the final bytes of the assistant reply.  In
     * that case no following character arrives to force the pending backticks
     * through the normal streaming path, so commit a full fence here instead of
     * leaking the literal ``` marker to the terminal. */
    if (r->md_pending == AGENT_MD_PENDING_BACKTICK && r->md_pending_len >= 3)
        renderer_markdown_commit_backticks(r);
    else
        renderer_markdown_emit_pending_literals(r);
    if (r->md_code_block && r->md_code_line_len)
        renderer_code_emit_buffered_line(r, false);
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    r->md_fence_info = false;
    r->md_code_line_start = false;
    r->md_code_in_ml_comment = false;
    r->md_syntax = NULL;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    r->md_code_line_prefix = NULL;
    r->md_code_line_prefix_color = NULL;
    free(r->md_code_line);
    r->md_code_line = NULL;
    r->md_code_line_len = 0;
    r->md_code_line_cap = 0;
    renderer_hint_end(r);
}

static void renderer_write_char(agent_token_renderer *r, char c) {
    if (!r->format_markdown || r->in_think) {
        renderer_markdown_emit_pending_literals(r);
        renderer_hint_end(r);
        renderer_write_char_raw(r, c);
        return;
    }
    /* Recognize only the advertised hint marker at a prose line boundary.
     * Hold at most that short prefix; ordinary text still streams immediately.
     * Explicit quote lines continue the aside. No width-dependent repainting. */
    if (r->use_color && !r->md_line_started && !r->md_code_block &&
        !r->md_fence_info && !r->md_inline_len && !r->md_escape &&
        r->md_pending == AGENT_MD_PENDING_NONE) {
        const char *prefix = r->md_hint ? "> " : "> **Hint:**";
        if (c == prefix[r->md_hint_prefix_len]) {
            r->md_hint_prefix[r->md_hint_prefix_len++] = c;
            if (prefix[r->md_hint_prefix_len]) return;
            bool continuation = r->md_hint;
            r->md_hint_prefix_len = 0;
            r->md_hint = true;
            r->md_line_started = true;
            renderer_reset_color(r);
            const char rule[] = "\x1b[38;5;30m\xe2\x94\x82 \x1b[0m";
            renderer_write(r, rule, sizeof(rule) - 1);
            r->wrote_visible_output = true;
            r->last_output_newline = false;
            if (!continuation) {
                const char label[] = "\x1b[1;97;48;5;23m Hint ";
                renderer_write(r, label, sizeof(label) - 1);
                renderer_reset_color(r);
            }
            return;
        }
        if (r->md_hint) renderer_reset_color(r);
        r->md_hint = false;
        size_t n = r->md_hint_prefix_len;
        r->md_hint_prefix_len = 0;
        for (size_t i = 0; i < n; i++)
            renderer_markdown_feed(r, r->md_hint_prefix[i]);
    }
    r->md_line_started = c != '\n';
    renderer_markdown_feed(r, c);
}

/* Render assistant text while hiding <think> tags and dimming thinking text.
 * The function is also responsible for not prematurely emitting a partial
 * control tag split across model tokens. */
static void renderer_finish(agent_token_renderer *r) {
    renderer_markdown_finish(r);
    renderer_flush_utf8(r);
    renderer_reset_color(r);
    if (r->wrote_visible_output) {
        if (!r->last_output_newline) renderer_write(r, "\n", 1);
        renderer_write(r, "\n", 1);
        r->last_output_newline = true;
    }
}

static void renderer_color(agent_token_renderer *r, const char *seq) {
    renderer_markdown_emit_pending_literals(r);
    renderer_hint_end(r);
    renderer_flush_utf8(r);
    bool reset = !seq || !seq[0] || !strcmp(seq, "\x1b[0m");
    if (r->use_color && seq && seq[0]) renderer_write(r, seq, strlen(seq));
    r->color_open = r->use_color && !reset;
}

static void renderer_plain(agent_token_renderer *r, const char *s, size_t n) {
    renderer_markdown_emit_pending_literals(r);
    renderer_hint_end(r);
    renderer_flush_utf8(r);
    renderer_write(r, s, n);
    if (n) r->wrote_visible_output = true;
    if (n) r->last_output_newline = s[n - 1] == '\n';
}

/* ============================================================================
 * Streaming Tool Visualization
 * ============================================================================
 *
 * Tool calls are parsed for execution later, but they are also visualized while
 * the model is still sampling.  This state machine suppresses raw DSML and
 * prints compact, tool-specific progress such as "$ command" or
 * "Reading file 1:500...".
 */


/* -- DSML tool-call visualiser ----------------------------------- */

static agent_tool_param_kind agent_tool_param_kind_for(const char *tool, const char *param) {
    if (!tool) tool = "";
    if (!param) param = "";
    if (!strcmp(tool, "bash") && !strcmp(param, "command"))
        return AGENT_TOOL_PARAM_BASH_COMMAND;
    if (!strcmp(tool, "edit") && !strcmp(param, "old"))
        return AGENT_TOOL_PARAM_DIFF_OLD;
    if (!strcmp(tool, "edit") && !strcmp(param, "new"))
        return AGENT_TOOL_PARAM_DIFF_NEW;
    if (streq_any(param, "path", "file", "filename", NULL))
        return AGENT_TOOL_PARAM_PATH;
    if (streq_any(param, "line", "start_line", "end_line", "offset") ||
        streq_any(param, "start", "end", "count", "max_lines") ||
        streq_any(param, "timeout_sec", "refresh_sec", NULL, NULL))
        return AGENT_TOOL_PARAM_OFFSET;
    if (streq_any(param, "content", "text", NULL, NULL))
        return AGENT_TOOL_PARAM_CONTENT;
    return AGENT_TOOL_PARAM_NORMAL;
}

static const char *agent_tool_param_color(agent_tool_param_kind kind) {
    switch (kind) {
    case AGENT_TOOL_PARAM_PATH: return "\x1b[32m";
    case AGENT_TOOL_PARAM_OFFSET: return "\x1b[33m";
    case AGENT_TOOL_PARAM_CONTENT: return "\x1b[34m";
    case AGENT_TOOL_PARAM_DIFF_OLD: return "\x1b[31m";
    case AGENT_TOOL_PARAM_DIFF_NEW: return "\x1b[32m";
    case AGENT_TOOL_PARAM_BASH_COMMAND: return "\x1b[1;36m";
    default: return "\x1b[37m";
    }
}

static void agent_tool_viz_write(agent_stream_renderer *sr, const char *s, size_t n) {
    renderer_plain(sr->renderer, s, n);
    for (size_t i = 0; i < n; i++) sr->viz.last_output_newline = s[i] == '\n';
}

static void agent_tool_viz_puts(agent_stream_renderer *sr, const char *s) {
    agent_tool_viz_write(sr, s, strlen(s));
}

static void agent_tool_viz_start(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    bool line_open = !sr->renderer->last_output_newline;
    memset(v, 0, sizeof(*v));
    v->active = true;
    v->at_line_start = true;
    v->last_output_newline = true;
    if (sr->renderer->use_color) {
        (void)line_open;
        /* The raw DSML start marker may arrive after ordinary text on the
         * current row.  Clear that row only for the live terminal UI; plain
         * stdout mode must never leak cursor-control escapes into pipes. */
        agent_tool_viz_puts(sr, "\r\x1b[2K");
    } else if (line_open) {
        agent_tool_viz_puts(sr, "\n");
    }
    v->last_output_newline = true;
}

static void agent_tool_viz_line_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    agent_tool_viz_puts(sr, "🛠️ ");
    v->at_line_start = false;
}

static const char *agent_tool_viz_prefix(const char *name) {
    if (!strcmp(name, "bash")) return "$ ";
    if (!strcmp(name, "read")) return "read ";
    if (!strcmp(name, "write")) return "write ";
    if (!strcmp(name, "edit")) return "edit ";
    if (!strcmp(name, "search")) return "search ";
    if (!strcmp(name, "google_search")) return "google ";
    if (!strcmp(name, "visit_page")) return "visit ";
    return NULL;
}

static void agent_tool_viz_tool(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->tool_announced && !strcmp(v->tool_name, name)) return;
    if (v->tool_announced && !v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    snprintf(v->tool_name, sizeof(v->tool_name), "%s", name ? name : "tool");
    v->tool_announced = true;
    v->read_style = !strcmp(v->tool_name, "read");
    agent_tool_viz_line_prefix(sr);
    if (v->read_style) {
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        v->read_prefix_rendered = true;
        return;
    }
    renderer_color(sr->renderer, !strcmp(v->tool_name, "bash") ?
                                "\x1b[1;36m" : "\x1b[1;37m");
    const char *prefix = agent_tool_viz_prefix(v->tool_name);
    if (prefix) {
        agent_tool_viz_puts(sr, prefix);
    } else {
        agent_tool_viz_puts(sr, v->tool_name);
        agent_tool_viz_puts(sr, " ");
    }
    renderer_color(sr->renderer, "\x1b[0m");
}

static void agent_tool_viz_append(char *dst, size_t cap, char c) {
    size_t len = strlen(dst);
    if (len + 1 >= cap) return;
    dst[len] = c;
    dst[len + 1] = '\0';
}

static void agent_tool_viz_read_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (!strcmp(v->param_name, "path")) {
        agent_tool_viz_append(v->read_path, sizeof(v->read_path), c);
        if (v->read_prefix_rendered) agent_tool_viz_write(sr, &c, 1);
    } else if (!strcmp(v->param_name, "start_line")) {
        agent_tool_viz_append(v->read_start, sizeof(v->read_start), c);
    } else if (!strcmp(v->param_name, "max_lines")) {
        agent_tool_viz_append(v->read_max, sizeof(v->read_max), c);
    } else if (!strcmp(v->param_name, "whole")) {
        agent_tool_viz_append(v->read_whole, sizeof(v->read_whole), c);
    }
}

static void agent_tool_viz_render_read(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->read_style || v->read_line_rendered) return;

    if (!v->read_prefix_rendered) {
        agent_tool_viz_line_prefix(sr);
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, "Reading ");
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, v->read_path[0] ? v->read_path : "<unknown>");
    } else if (!v->read_path[0]) {
        renderer_color(sr->renderer, "\x1b[32m");
        agent_tool_viz_puts(sr, "<unknown>");
    }
    renderer_color(sr->renderer, "\x1b[33m");
    bool whole = agent_parse_bool_default(v->read_whole, false);
    if (whole && (!v->read_start[0] || !strcmp(v->read_start, "1"))) {
        agent_tool_viz_puts(sr, " (whole file)");
    } else if (whole) {
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start);
        agent_tool_viz_puts(sr, ":EOF");
    } else {
        char default_lines[16];
        snprintf(default_lines, sizeof(default_lines), "%d",
                 client_read_default_lines(sr->ctx_size));
        agent_tool_viz_puts(sr, " ");
        agent_tool_viz_puts(sr, v->read_start[0] ? v->read_start : "1");
        agent_tool_viz_puts(sr, ":");
        agent_tool_viz_puts(sr, v->read_max[0] ? v->read_max : default_lines);
    }
    renderer_color(sr->renderer, "\x1b[1;37m");
    agent_tool_viz_puts(sr, "...");
    renderer_color(sr->renderer, "\x1b[0m");
    agent_tool_viz_puts(sr, "\n");
    v->read_line_rendered = true;
}

static bool agent_tool_viz_param_is_code_body(agent_tool_visualizer *v) {
    if (!strcmp(v->tool_name, "write") &&
        v->param_kind == AGENT_TOOL_PARAM_CONTENT)
        return true;
    if (!strcmp(v->tool_name, "edit") &&
        (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
         v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW ||
         v->param_kind == AGENT_TOOL_PARAM_CONTENT))
        return true;
    return false;
}

static const char *agent_tool_viz_diff_prefix(agent_tool_param_kind kind,
                                              const char **color) {
    if (color) *color = NULL;
    const char *prefix = NULL;
    if (kind == AGENT_TOOL_PARAM_DIFF_OLD) {
        prefix = "- ";
        if (color) *color = "\x1b[31m";
    } else if (kind == AGENT_TOOL_PARAM_DIFF_NEW) {
        prefix = "+ ";
        if (color) *color = "\x1b[32m";
    }
    return prefix;
}

static void agent_tool_viz_code_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->at_line_start) return;
    const char *color = NULL;
    const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
    if (!prefix) return;
    renderer_color(sr->renderer, color);
    renderer_write(sr->renderer, prefix, strlen(prefix));
    renderer_color(sr->renderer, "\x1b[0m");
    sr->renderer->wrote_visible_output = true;
    sr->renderer->last_output_newline = false;
    v->last_output_newline = false;
    v->at_line_start = false;
}

static void agent_tool_viz_code_begin(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    const agent_syntax *syntax = agent_syntax_for_path(v->tool_path);
    renderer_code_stream_begin(sr->renderer, syntax);
    v->code_param_active = true;
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        const char *color = NULL;
        const char *prefix = agent_tool_viz_diff_prefix(v->param_kind, &color);
        /* Diff prefixes are terminal UI, not code.  Keep them outside the
         * syntax buffer so a later row repaint preserves their red/green color
         * while highlighting only the actual edited line. */
        renderer_code_stream_set_prefix(sr->renderer, prefix, color);
        agent_tool_viz_code_prefix(sr);
    }
}

static void agent_tool_viz_code_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->code_param_active) return;
    renderer_code_end(sr->renderer);
    v->code_param_active = false;
    v->at_line_start = true;
    v->last_output_newline = sr->renderer->last_output_newline;
}

static void agent_tool_viz_code_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    agent_tool_viz_code_prefix(sr);
    renderer_code_byte(sr->renderer, c);
    v->last_output_newline = c == '\n';
    v->at_line_start = c == '\n';
}

static void agent_tool_viz_param_begin(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && sr->cur_tool[0])
        agent_tool_viz_tool(sr, sr->cur_tool);
    snprintf(v->param_name, sizeof(v->param_name), "%s", name ? name : "");
    v->param_kind = agent_tool_param_kind_for(v->tool_name, v->param_name);
    v->param_active = true;
    v->param_end_len = 0;

    if (v->read_style) return;

    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        v->at_line_start = true;
        agent_tool_viz_code_begin(sr);
        return;
    }

    if (v->param_kind == AGENT_TOOL_PARAM_CONTENT) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        if (strcmp(v->tool_name, "write")) {
            renderer_color(sr->renderer, "\x1b[1;37m");
            agent_tool_viz_puts(sr, v->param_name);
            agent_tool_viz_puts(sr, ":\n");
        }
        v->at_line_start = true;
        if (agent_tool_viz_param_is_code_body(v)) {
            agent_tool_viz_code_begin(sr);
        } else {
            renderer_color(sr->renderer, "\x1b[34m");
        }
        return;
    }

    if (v->param_kind != AGENT_TOOL_PARAM_BASH_COMMAND) {
        if (!v->at_line_start) agent_tool_viz_puts(sr, " ");
        renderer_color(sr->renderer, "\x1b[1;37m");
        agent_tool_viz_puts(sr, v->param_name);
        agent_tool_viz_puts(sr, "=");
    } else {
        renderer_color(sr->renderer, agent_tool_param_color(AGENT_TOOL_PARAM_BASH_COMMAND));
        return;
    }
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}

static void agent_tool_viz_param_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    v->param_end_len = 0;
    if (v->code_param_active) agent_tool_viz_code_end(sr);
    if (!v->read_style) renderer_color(sr->renderer, "\x1b[0m");
    v->param_active = false;
    v->param_name[0] = '\0';
}

static void agent_tool_viz_param_raw_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->read_style) {
        agent_tool_viz_read_value_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_PATH) {
        agent_tool_viz_append(v->tool_path, sizeof(v->tool_path), c);
    }
    if (v->code_param_active) {
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_BASH_COMMAND) {
        agent_tool_viz_write(sr, &c, 1);
        v->at_line_start = c == '\n';
        return;
    }
    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        agent_tool_viz_code_begin(sr);
        agent_tool_viz_code_byte(sr, c);
        return;
    }
    agent_tool_viz_write(sr, &c, 1);
    v->at_line_start = c == '\n';
}

static void agent_tool_viz_restore_param_color(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active || !v->param_active || v->read_style) return;
    renderer_color(sr->renderer, agent_tool_param_color(v->param_kind));
}

static void agent_tool_viz_finish(agent_stream_renderer *sr, const char *status) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;
    if (v->param_active) agent_tool_viz_param_end(sr);
    if (!status || !status[0]) agent_tool_viz_render_read(sr);
    if (status && status[0]) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
        renderer_color(sr->renderer, "\x1b[90m");
        agent_tool_viz_puts(sr, status);
        renderer_color(sr->renderer, "\x1b[0m");
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, "\n");
    v->active = false;
}


/* -- (kind, text) fragment -> visualiser --------------------------- */

static void client_stream_renderer_init(agent_stream_renderer *sr,
                                        agent_token_renderer *rndr,
                                        int ctx_size) {
    memset(sr, 0, sizeof(*sr));
    sr->renderer = rndr;
    sr->ctx_size = ctx_size;
}

/* Close any open tool visualisation before ordinary text resumes. */
static void client_stream_end_tool(agent_stream_renderer *sr) {
    if (sr->viz.active) agent_tool_viz_finish(sr, NULL);
    sr->cur_tool[0] = '\0';
    sr->cur_param[0] = '\0';
}

static void client_apply_stream_fragment(agent_stream_renderer *sr,
                                         uint32_t kind,
                                         const char *text, size_t len) {
    switch (kind) {
    case AGENT_STREAM_NORMAL:
        client_stream_end_tool(sr);
        sr->summary_banner_shown = false;
        for (size_t i = 0; i < len; i++)
            renderer_write_char(sr->renderer, text[i]);
        break;

    case AGENT_STREAM_THINK:
        client_stream_end_tool(sr);
        renderer_set_grey(sr->renderer);
        renderer_write(sr->renderer, text, len);
        break;

    case AGENT_STREAM_SUMMARY:
        client_stream_end_tool(sr);
        if (!sr->summary_banner_shown) {
            const char *b = sr->renderer->use_color ?
                "\n\x1b[1;95mCompacted summary:\x1b[0m\n" : "\nCompacted summary:\n";
            renderer_write(sr->renderer, b, strlen(b));
            sr->summary_banner_shown = true;
        }
        renderer_set_grey(sr->renderer);
        renderer_write(sr->renderer, text, len);
        break;

    case AGENT_STREAM_SYSTEM: {
        client_stream_end_tool(sr);
        if (!sr->renderer->last_output_newline)
            renderer_write(sr->renderer, "\n", 1);
        if (sr->renderer->use_color)
            renderer_write(sr->renderer, "\x1b[33m\xe2\x9c\xa6 \x1b[38;5;218m",
                           strlen("\x1b[33m\xe2\x9c\xa6 \x1b[38;5;218m"));
        else
            renderer_write(sr->renderer, "\xe2\x9c\xa6 ", strlen("\xe2\x9c\xa6 "));
        renderer_write(sr->renderer, text, len);
        renderer_write(sr->renderer, sr->renderer->use_color ? "\x1b[0m\n" : "\n",
                       sr->renderer->use_color ? 5 : 1);
        sr->renderer->last_output_newline = true;
        break;
    }

    case AGENT_STREAM_TOOL_NAME:
        sr->summary_banner_shown = false;
        if (!sr->viz.active) agent_tool_viz_start(sr);
        snprintf(sr->cur_tool, sizeof(sr->cur_tool), "%.*s", (int)len, text);
        agent_tool_viz_tool(sr, sr->cur_tool);
        break;

    case AGENT_STREAM_TOOL_PARAM_NAME:
        if (sr->viz.param_active) agent_tool_viz_param_end(sr);
        snprintf(sr->cur_param, sizeof(sr->cur_param), "%.*s", (int)len, text);
        agent_tool_viz_param_begin(sr, sr->cur_param);
        break;

    case AGENT_STREAM_TOOL_PARAM_VALUE:
        for (size_t i = 0; i < len; i++)
            agent_tool_viz_param_raw_byte(sr, text[i]);
        break;

    default:
        break;
    }
}

/* Called on a terminal STATUS reached from PREFILL / GENERATING / COMPACTING. */
static void client_stream_renderer_finish(agent_stream_renderer *sr) {
    client_stream_end_tool(sr);
    renderer_finish(sr->renderer);
    sr->summary_banner_shown = false;
}

/* -- footer / status line ---------------------------------------- */

/* Mirror of ds4_agent.c's agent_worker_state, matching enum agent_state on the
 * wire (same order). */
typedef enum {
    AGENT_WORKER_IDLE,
    AGENT_WORKER_PREFILL,
    AGENT_WORKER_GENERATING,
    AGENT_WORKER_COMPACTING,
    AGENT_WORKER_DRAINING,
    AGENT_WORKER_SAVING,
    AGENT_WORKER_ERROR,
    AGENT_WORKER_STOPPED,
} agent_worker_state;

typedef struct {
    agent_worker_state state;
    int prefill_done;
    int prefill_total;
    unsigned prefill_label;
    double prefill_tps;
    int generated;
    double gen_tps;
    bool greedy_sampling;
    int ctx_used;
    int ctx_size;
    int power_percent;
    char error[256];
} agent_status;

static void client_status_from_wire(const ap_status *w, agent_status *out) {
    memset(out, 0, sizeof(*out));
    out->state = (agent_worker_state)w->state;
    out->prefill_done = (int)w->prefill_done;
    out->prefill_total = (int)w->prefill_total;
    out->prefill_label = w->prefill_label;
    out->prefill_tps = ap_centi_to_double(w->prefill_tps);
    out->generated = (int)w->generated;
    out->gen_tps = ap_centi_to_double(w->gen_tps);
    out->greedy_sampling = w->greedy_sampling;
    out->ctx_used = (int)w->ctx_used;
    out->ctx_size = (int)w->ctx_size;
    out->power_percent = (int)w->power_percent;
    size_t en = strnlen(w->error, sizeof(out->error) - 1);
    memcpy(out->error, w->error, en);
    out->error[en] = '\0';
}

#define AGENT_STATUS_STYLE_START "\x1b[48;5;238;38;5;252m"
#define AGENT_STATUS_BAR_FILL "\x1b[48;5;238;38;5;201;1m"
#define AGENT_PROGRESS_BAR_WIDTH 32
#define AGENT_PROGRESS_BAR_MAX_BYTES 256

static bool stdout_is_tty(void) {
    return isatty(STDOUT_FILENO) != 0;
}

static void agent_format_ctx_size(int ctx_size, char *buf, size_t len) {
    if (ctx_size >= 1000) {
        if (ctx_size % 1000 == 0) snprintf(buf, len, "%dk", ctx_size / 1000);
        else snprintf(buf, len, "%.1fk", (double)ctx_size / 1000.0);
    } else {
        snprintf(buf, len, "%d", ctx_size);
    }
}

static void agent_progress_append(char *buf, size_t len, size_t *pos,
                                  const char *s) {
    if (len == 0 || *pos >= len - 1) return;
    size_t avail = len - *pos;
    int n = snprintf(buf + *pos, avail, "%s", s);
    if (n <= 0) return;
    if ((size_t)n >= avail) *pos = len - 1;
    else *pos += (size_t)n;
}

static void agent_progress_bar(int done, int total, double tps,
                               char *buf, size_t len, bool color) {
    if (len == 0) return;
    if (total <= 0) total = 1;
    if (done < 0) done = 0;
    if (done > total) done = total;
    int filled = (int)(((long long)done * AGENT_PROGRESS_BAR_WIDTH) / total);
    if (filled < 0) filled = 0;
    if (filled > AGENT_PROGRESS_BAR_WIDTH) filled = AGENT_PROGRESS_BAR_WIDTH;
    if (color && filled == 0 && done < total) filled = 1;
    char rate[32] = {0};
    size_t rate_len = 0;
    if (tps > 0.0 && filled < AGENT_PROGRESS_BAR_WIDTH) {
        snprintf(rate, sizeof(rate), " %.0ft/s", tps);
        rate_len = strlen(rate);
    }
    size_t pos = 0;
    agent_progress_append(buf, len, &pos, "[");
    if (color) agent_progress_append(buf, len, &pos, AGENT_STATUS_BAR_FILL);
    for (int i = 0; i < AGENT_PROGRESS_BAR_WIDTH && pos + 1 < len; i++) {
        if (color && i == filled)
            agent_progress_append(buf, len, &pos, AGENT_STATUS_STYLE_START);
        if (i >= filled && rate_len > 0 && (size_t)(i - filled) < rate_len) {
            char ch[2] = { rate[i - filled], '\0' };
            agent_progress_append(buf, len, &pos, ch);
        } else {
            agent_progress_append(buf, len, &pos, i < filled ? "\xe2\x96\xb6" : "\xc2\xb7");
        }
    }
    if (color) agent_progress_append(buf, len, &pos, AGENT_STATUS_STYLE_START);
    agent_progress_append(buf, len, &pos, "]");
    buf[pos < len ? pos : len - 1] = '\0';
}

static void agent_power_status_suffix(const agent_status *st, char *buf, size_t len) {
    if (len == 0) return;
    if (st->power_percent > 0 && st->power_percent < 100)
        snprintf(buf, len, " | \xe2\x9a\xa1 %d%%", st->power_percent);
    else
        buf[0] = '\0';
}

static const char *agent_prefill_label(const agent_status *st) {
    static const char *labels[] = {
        "reading", "absorbing", "studying", "gathering", "crunching", "scrutinizing",
    };
    size_t n = sizeof(labels) / sizeof(labels[0]);
    return labels[(st ? st->prefill_label : 0u) % n];
}

static void build_status_text(const agent_status *st, char *buf, size_t len) {
    char used[32], total_ctx[32], power[32];
    agent_format_ctx_size(st->ctx_used, used, sizeof(used));
    agent_format_ctx_size(st->ctx_size, total_ctx, sizeof(total_ctx));
    agent_power_status_suffix(st, power, sizeof(power));

    switch (st->state) {
    case AGENT_WORKER_PREFILL: {
        int done = st->prefill_done;
        int total = st->prefill_total > 0 ? st->prefill_total : 1;
        if (done > total) done = total;
        double pct = 100.0 * (double)done / (double)total;
        char bar[AGENT_PROGRESS_BAR_MAX_BYTES];
        agent_progress_bar(done, total, st->prefill_tps, bar, sizeof(bar),
                           stdout_is_tty());
        snprintf(buf, len, "ctx %s/%s | %s %s %d/%d %.1f%%%s",
                 used, total_ctx, agent_prefill_label(st), bar,
                 done, total, pct, power);
        break;
    }
    case AGENT_WORKER_GENERATING:
        snprintf(buf, len, "ctx %s/%s | generation %d tokens%s %.1f t/s%s",
                 used, total_ctx, st->generated,
                 st->greedy_sampling ? " \xe2\x9d\x84\xef\xb8\x8f" : "",
                 st->gen_tps, power);
        break;
    case AGENT_WORKER_COMPACTING:
        snprintf(buf, len, "ctx %s/%s | COMPACTING summary %d tokens %.1f t/s%s",
                 used, total_ctx, st->generated, st->gen_tps, power);
        break;
    case AGENT_WORKER_DRAINING:
        snprintf(buf, len, "ctx %s/%s | stopping after distributed cluster drains%s",
                 used, total_ctx, power);
        break;
    case AGENT_WORKER_SAVING:
        snprintf(buf, len, "ctx %s/%s | saving session%s", used, total_ctx, power);
        break;
    case AGENT_WORKER_ERROR:
        snprintf(buf, len, "ctx %s/%s | error: %s%s", used, total_ctx,
                 st->error[0] ? st->error : "unknown error", power);
        break;
    case AGENT_WORKER_STOPPED:
        snprintf(buf, len, "ctx %s/%s | interrupted%s", used, total_ctx, power);
        break;
    default:
        snprintf(buf, len, "ctx %s/%s | idle%s", used, total_ctx, power);
        break;
    }
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
