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
#include "ds4_web.h"
#include "linenoise.h"
#include "ds4_agent_proto.h"
#include "ds4_agent_utils.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <limits.h>
#include <math.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <pthread.h>
#include <regex.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>
#ifdef __APPLE__
#include <copyfile.h>
#elif defined(__linux__)
#include <sys/xattr.h>
#endif

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


/* -- client worker context + tool-call types (no engine, no transcript) --- */

typedef struct agent_bash_job agent_bash_job;

typedef struct {
    int ctx_size;               /* authoritative, from the SESSION reply */
    pthread_mutex_t mu;         /* guards raw_mode_needs_restore */
    bool raw_mode_needs_restore;
    bool interrupt_requested;   /* T11 wires Ctrl+C into this */

    /* more_* read cursor (agent_tool_more) */
    char more_path[PATH_MAX];
    int more_next_line;
    off_t more_byte_offset;
    bool more_mid_line;
    bool more_bare;
    bool more_valid;

    agent_bash_job *bash_jobs;
    int next_bash_job_id;

    ds4_web *web;
} agent_worker;

static void client_worker_init(agent_worker *w, int ctx_size) {
    memset(w, 0, sizeof(*w));
    w->ctx_size = ctx_size;
    pthread_mutex_init(&w->mu, NULL);
}

typedef struct {
    char *name;
    char *value;
    bool is_string;
} agent_tool_arg;

typedef struct {
    char *name;
    agent_tool_arg *args;
    int argc;
    int argcap;
} agent_tool_call;

static const char *agent_tool_arg_value(const agent_tool_call *call, const char *name) {
    for (int i = 0; i < call->argc; i++) {
        if (call->args[i].name && !strcmp(call->args[i].name, name))
            return call->args[i].value ? call->args[i].value : "";
    }
    return NULL;
}

/* ========================================================================= */
/* T10: client tool execution.                                                */
/*                                                                            */
/* File tools, search, the web tools and the bash job subsystem are copied     */
/* from ds4_agent.c -- almost none of it calls ds4_*. agent_tool_observation   */
/* (which builds transcript tokens) stays server-side; the client builds a     */
/* wire-level client_tool_result (text parts + raw image bytes) instead.       */
/* agent_edit_find_old_span is exact-match only (plan 3b): old must occur     */
/* exactly once, no anchored marker syntax.                                    */
/* ========================================================================= */

#define AGENT_FILE_MAX_BYTES (16*1024*1024)
#define AGENT_WEB_HEAD_BYTES (8*1024)
#define AGENT_WEB_HEAD_LINES 100
#define AGENT_BASH_HEAD_BYTES (8*1024)
#define AGENT_BASH_HEAD_LINES 100
#define AGENT_BASH_TAIL_BYTES (32*1024)
#define AGENT_BASH_PROGRESS_TAIL_LINES 4
#define AGENT_BASH_FINAL_TAIL_LINES 20

static bool worker_should_interrupt(agent_worker *w) {
    return w->interrupt_requested;
}

/* agent_worker_note_terminal_mode_may_have_changed is defined further down,
 * alongside the bash job subsystem that uses it. */

static int agent_parse_timeout(const char *s) {
    if (!s || !s[0]) return 3600;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || v <= 0.0 || !isfinite(v)) return 3600;
    if (v < 1.0) v = 1.0;
    if (v > 24.0 * 3600.0) v = 24.0 * 3600.0;
    return (int)v;
}

static int agent_replace_file(const char *path, const char *data, size_t len,
                              const char *expected, size_t expected_len,
                              char *err, size_t errlen);

static int agent_parse_int_default(const char *s, int def, int min, int max) {
    if (!s || !s[0]) return def;
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (end == s) return def;
    while (*end == ' ' || *end == '\t' || *end == '\r' || *end == '\n') end++;
    if (*end) return def;
    if (v < min) v = min;
    if (v > max) v = max;
    return (int)v;
}

/* Worker-side output: on the client there is no separate UI thread, so this
 * writes straight through the render sink (T9). */
static void agent_publish(agent_worker *w, const char *s, size_t n) {
    (void)w;
    if (n) g_render_sink(s, n);
}

static void agent_publishf(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(stack)) { agent_publish(w, stack, (size_t)n); return; }
    char *heap = xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    agent_publish(w, heap, (size_t)n);
    free(heap);
}

/* The "✦" notice, matching the STREAM{SYSTEM} painting from T9. */
static void agent_publish_system_status(agent_worker *w, const char *msg) {
    if (!msg || !msg[0]) return;
    (void)w;
    bool color = stdout_is_tty();
    if (color) g_render_sink("\x1b[33m\xe2\x9c\xa6 \x1b[38;5;218m",
                             strlen("\x1b[33m\xe2\x9c\xa6 \x1b[38;5;218m"));
    else g_render_sink("\xe2\x9c\xa6 ", strlen("\xe2\x9c\xa6 "));
    g_render_sink(msg, strlen(msg));
    g_render_sink(color ? "\x1b[0m\n" : "\n", color ? 5 : 1);
}

static void agent_publishf_system_status(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(stack)) { agent_publish_system_status(w, stack); return; }
    char *heap = xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    agent_publish_system_status(w, heap);
    free(heap);
}

/* Local Chrome-start confirmation: a plain stdin y/n prompt for now (T11 gives
 * it the editor's raw-mode prompt). */
static int agent_web_confirm(void *privdata, const char *message,
                             char *err, size_t err_len) {
    (void)privdata;
    if (!stdout_is_tty()) {
        snprintf(err, err_len,
                 "visible Chrome browser startup requires an interactive terminal");
        return 0;
    }
    char prompt[320];
    snprintf(prompt, sizeof(prompt), "%s ",
             message ? message : "Start visible Chrome browser? (y/n)");
    g_render_sink(prompt, strlen(prompt));
    char line[16] = {0};
    if (!fgets(line, sizeof(line), stdin)) {
        snprintf(err, err_len, "no answer");
        return 0;
    }
    bool yes = line[0] == 'y' || line[0] == 'Y';
    if (!yes) snprintf(err, err_len, "user denied Chrome browser start");
    return yes ? 1 : 0;
}

static void agent_web_log(void *privdata, const char *message) {
    (void)privdata; (void)message; /* client trace lands in T11 */
}

static bool agent_web_cancel(void *privdata) {
    return worker_should_interrupt((agent_worker *)privdata);
}

static int agent_read_default_lines(agent_worker *w) {
    return client_read_default_lines(w->ctx_size);
}

typedef struct {
    size_t start;
    size_t content_end;
    size_t end;
} agent_line_span;

typedef struct {
    agent_line_span *v;
    int len;
    int cap;
} agent_line_spans;

static void agent_line_spans_free(agent_line_spans *spans) {
    free(spans->v);
    memset(spans, 0, sizeof(*spans));
}

static void agent_line_spans_push(agent_line_spans *spans, agent_line_span span) {
    if (spans->len == spans->cap) {
        spans->cap = spans->cap ? spans->cap * 2 : 128;
        spans->v = xrealloc(spans->v, (size_t)spans->cap * sizeof(spans->v[0]));
    }
    spans->v[spans->len++] = span;
}

static void agent_split_lines(const char *data, size_t len, agent_line_spans *spans) {
    size_t pos = 0;
    while (pos < len) {
        size_t start = pos;
        while (pos < len && data[pos] != '\n' && data[pos] != '\r') pos++;
        size_t content_end = pos;
        if (pos < len) {
            if (data[pos] == '\r' && pos + 1 < len && data[pos + 1] == '\n')
                pos += 2;
            else
                pos++;
        }
        agent_line_spans_push(spans, (agent_line_span){
            .start = start,
            .content_end = content_end,
            .end = pos,
        });
    }
}

static FILE *agent_open_regular_file(const char *path) {
    int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat st;
    int rc = fstat(fd, &st);
    if (rc != 0 || !S_ISREG(st.st_mode)) {
        int saved = rc != 0 ? errno : EINVAL;
        close(fd);
        errno = saved;
        return NULL;
    }
    FILE *fp = fdopen(fd, "rb");
    if (!fp) { int saved = errno; close(fd); errno = saved; }
    return fp;
}

static int agent_read_file_bytes(const char *path, char **data, size_t *len,
                                 char *err, size_t errlen) {
    FILE *fp = agent_open_regular_file(path);
    if (!fp) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return -1;
    }
    char *buf = NULL;
    size_t used = 0, cap = 0;
    char tmp[8192];
    while (true) {
        size_t n = fread(tmp, 1, sizeof(tmp), fp);
        if (n) {
            if (used + n > AGENT_FILE_MAX_BYTES) {
                fclose(fp);
                free(buf);
                snprintf(err, errlen, "file too large: %s exceeds %d bytes",
                         path, AGENT_FILE_MAX_BYTES);
                return -1;
            }
            if (used + n + 1 > cap) {
                cap = cap ? cap * 2 : 8192;
                while (cap < used + n + 1) cap *= 2;
                buf = xrealloc(buf, cap);
            }
            memcpy(buf + used, tmp, n);
            used += n;
            buf[used] = '\0';
        }
        if (n < sizeof(tmp)) {
            if (ferror(fp)) {
                snprintf(err, errlen, "read %s: %s", path, strerror(errno));
                fclose(fp);
                free(buf);
                return -1;
            }
            break;
        }
    }
    fclose(fp);
    if (!buf) buf = xstrdup("");
    *data = buf;
    *len = used;
    return 0;
}

static int agent_line_for_offset(const agent_line_spans *spans, size_t offset) {
    if (!spans || spans->len <= 0) return 1;
    for (int i = 0; i < spans->len; i++) {
        if (offset < spans->v[i].end) return i + 1;
    }
    return spans->len;
}

static bool agent_old_new_line_effect(const char *old_data, size_t old_len,
                                      const char *new_data, size_t new_len,
                                      size_t edit_offset, size_t replaced_len,
                                      int *start_line, int *end_line,
                                      int *delta) {
    agent_line_spans old_spans = {0};
    agent_line_spans new_spans = {0};
    agent_split_lines(old_data, old_len, &old_spans);
    agent_split_lines(new_data, new_len, &new_spans);
    bool ok = old_spans.len > 0;
    if (ok) {
        size_t old_last = edit_offset;
        if (replaced_len > 0) old_last = edit_offset + replaced_len - 1;
        if (old_last >= old_len) old_last = old_len ? old_len - 1 : 0;
        if (start_line) *start_line = agent_line_for_offset(&old_spans, edit_offset);
        if (end_line) *end_line = agent_line_for_offset(&old_spans, old_last);
        if (delta) *delta = new_spans.len - old_spans.len;
    }
    agent_line_spans_free(&old_spans);
    agent_line_spans_free(&new_spans);
    return ok;
}

static void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end);

static char *agent_edit_result(const char *path,
                                       int start_line, int end_line, int delta,
                                       const char *new_data, size_t new_len,
                                       const char *kind) {
    agent_buf b = {.limit = AGENT_TOOL_MAX_BYTES};
    char msg[PATH_MAX + 180];
    snprintf(msg, sizeof(msg), "Edited %s using %s\n", path, kind);
    agent_buf_puts(&b, msg);
    if (start_line > 0 && end_line >= start_line) {
        snprintf(msg, sizeof(msg),
                 "Touched old lines %d-%d; current post-edit context follows.\n",
                 start_line, end_line);
        agent_buf_puts(&b, msg);
        if (delta != 0) {
            snprintf(msg, sizeof(msg),
                     "Line shift: old lines after %d moved by %+d (old line %d is now line %d). Re-read before relying on old line numbers there.\n",
                     end_line, delta, end_line + 1, end_line + 1 + delta);
            agent_buf_puts(&b, msg);
        }
    }
    if (start_line > 0 && end_line >= start_line) {
        int new_anchor_end = end_line + delta;
        if (new_anchor_end < start_line) new_anchor_end = start_line;
        agent_edit_result_append_context(&b, path, new_data, new_len,
                                         start_line, new_anchor_end);
    }
    return agent_buf_take(&b);
}

static void agent_worker_set_more(agent_worker *w, const char *path,
                                  int next_line, bool bare) {
    if (path != w->more_path)
        snprintf(w->more_path, sizeof(w->more_path), "%s", path ? path : "");
    w->more_next_line = next_line;
    w->more_bare = bare;
    w->more_valid = path && path[0] && next_line > 0;
    w->more_byte_offset = 0;
    w->more_mid_line = false;
}

static char *agent_read_range_from(agent_worker *w, const char *path, int start_line,
                              int max_lines, bool whole_file, bool bare,
                              bool set_more, off_t offset, bool mid_line) {
    if (!path || !path[0]) return xstrdup("Tool error: read requires path\n");
    agent_buf out = {0}, body = {0};
    FILE *fp = agent_open_regular_file(path);
    if (!fp) goto failed;
    struct stat st;
    if (fstat(fileno(fp), &st) != 0) goto failed;
    if (!S_ISREG(st.st_mode)) { errno = EINVAL; goto failed; }
    if (start_line < 1) start_line = 1;
    if (max_lines <= 0) max_lines = agent_read_default_lines(w);
    int c, line = 1, lines = 0;
    if (offset > 0) {
        if (fseeko(fp, offset, SEEK_SET) != 0) goto failed;
        line = start_line;
    }
    while (line < start_line && (c = fgetc(fp)) != EOF) {
        if (c == '\r') {
            int next = fgetc(fp);
            if (next != '\n' && next != EOF) ungetc(next, fp);
        }
        if (c == '\n' || c == '\r') line++;
    }
    bool beginning = !mid_line, prefix = true;
    int last_line = 0;
    while ((whole_file || lines < max_lines) && (c = fgetc(fp)) != EOF) {
        if (body.len >= AGENT_TOOL_MAX_BYTES - 4096 &&
            ((c & 0xc0) != 0x80 || body.len >= AGENT_TOOL_MAX_BYTES - 4096 + 3)) {
            ungetc(c, fp);
            break;
        }
        if (!c) {
            agent_buf_puts(&out, "Tool error: read encountered binary data\n");
            goto done;
        }
        last_line = line;
        if (prefix && !bare) {
            char label[80];
            snprintf(label, sizeof(label), "%d%s ", line, beginning ? "" : " (continued)");
            agent_buf_puts(&body, label);
        }
        prefix = false;
        beginning = false;
        char ch = (char)c;
        if (c == '\r') {
            int next = fgetc(fp);
            if (bare) agent_buf_append(&body, &ch, 1);
            if (next == '\n') {
                ch = '\n';
            } else {
                if (next != EOF) ungetc(next, fp);
                if (bare) ch = 0;
                else ch = '\n';
            }
        }
        if (ch) agent_buf_append(&body, &ch, 1);
        if (c == '\n' || c == '\r') {
            if (line == INT_MAX) { errno = EOVERFLOW; goto failed; }
            line++;
            lines++;
            beginning = prefix = true;
        }
    }
    if (ferror(fp)) goto failed;
    off_t next_offset = ftello(fp);
    if (next_offset < 0) goto failed;
    c = fgetc(fp);
    bool more = c != EOF;
    if (ferror(fp)) goto failed;
    if (whole_file && more) {
        agent_buf_puts(&out, "Tool error: whole read exceeds the 128 KiB output limit; use read and more for chunks\n");
        goto done;
    }
    if (!bare) {
        char header[PATH_MAX + 128];
        snprintf(header, sizeof(header), "%s: lines %d-%d%s\n", path,
                 last_line ? start_line : 0, last_line, more ? " (partial read)" : " (end of file)");
        agent_buf_puts(&out, header);
    }
    agent_buf_append(&out, body.ptr, body.len);
    if (more) {
        char note[256];
        snprintf(note, sizeof(note), "\n[Read truncated. continue_offset=%d; continue_byte_offset=%lld%s. Call more to continue.]\n",
                 line, (long long)next_offset, beginning ? "" : " (within line)");
        agent_buf_puts(&out, note);
    }
    if (set_more) {
        agent_worker_set_more(w, more ? path : NULL, more ? line : 0, bare);
        w->more_byte_offset = next_offset;
        w->more_mid_line = !beginning;
    }
    free(body.ptr);
    fclose(fp);
    return agent_buf_take(&out);
failed:
    agent_buf_puts(&out, "Tool error: read failed: ");
    agent_buf_puts(&out, strerror(errno));
    agent_buf_puts(&out, "\n");
done:
    if (fp) fclose(fp);
    free(body.ptr);
    if (set_more) agent_worker_set_more(w, NULL, 0, false);
    return agent_buf_take(&out);
}

static char *agent_read_range(agent_worker *w, const char *path, int start_line,
                              int max_lines, bool whole_file, bool bare,
                              bool set_more) {
    return agent_read_range_from(w, path, start_line, max_lines, whole_file,
                                 bare, set_more, 0, false);
}

static char *agent_tool_read(agent_worker *w, const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    bool whole = agent_parse_bool_default(agent_tool_arg_value(call, "whole"), false);
    int start = agent_parse_int_default(agent_tool_arg_value(call, "start_line"),
                                        1, 1, INT_MAX);
    int count = agent_parse_int_default(agent_tool_arg_value(call, "max_lines"),
                                        agent_read_default_lines(w), 1, INT_MAX);
    bool raw = agent_parse_bool_default(agent_tool_arg_value(call, "raw"), false);
    return agent_read_range(w, path, start, count, whole, raw, true);
}

static char *agent_tool_more(agent_worker *w, const agent_tool_call *call) {
    int count = agent_parse_int_default(agent_tool_arg_value(call, "count"),
                                        agent_read_default_lines(w), 1, INT_MAX);
    if (!w->more_valid) return xstrdup("Tool error: no previous output to continue\n");
    return agent_read_range_from(w, w->more_path, w->more_next_line, count, false,
                                 w->more_bare, true, w->more_byte_offset, w->more_mid_line);
}

static char *agent_tool_write(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *path = agent_tool_arg_value(call, "path");
    const char *content = agent_tool_arg_value(call, "content");
    if (!path || !path[0]) return xstrdup("Tool error: write requires path\n");
    if (!content) return xstrdup("Tool error: write requires content\n");
    size_t len = strlen(content);
    char err[256];
    if (agent_replace_file(path, content, len, NULL, 0, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    char msg[PATH_MAX + 160];
    snprintf(msg, sizeof(msg), "Wrote %zu bytes to %s\n", len, path);
    return xstrdup(msg);
}

static char *agent_tool_list(const agent_tool_call *call) {
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    DIR *dir = opendir(path);
    if (!dir) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: opendir failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    agent_buf out = {.limit = AGENT_TOOL_MAX_BYTES};
    char hdr[PATH_MAX + 64];
    snprintf(hdr, sizeof(hdr), "%s:\n", path);
    agent_buf_puts(&out, hdr);
    struct dirent *de;
    int shown = 0;
    while ((de = readdir(dir)) != NULL && shown < 300) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", path, de->d_name);
        struct stat st;
        if (lstat(full, &st) != 0) continue;
        char type = S_ISDIR(st.st_mode) ? 'd' :
                    S_ISLNK(st.st_mode) ? 'l' :
                    S_ISREG(st.st_mode) ? '-' : '?';
        char line[PATH_MAX + 96];
        snprintf(line, sizeof(line), "%c %10lld %s%s\n", type,
                 (long long)st.st_size, de->d_name, S_ISDIR(st.st_mode) ? "/" : "");
        agent_buf_puts(&out, line);
        shown++;
    }
    if (de) agent_buf_puts(&out, "... more entries omitted ...\n");
    closedir(dir);
    return agent_buf_take(&out);
}

static bool agent_same_file_version(const struct stat *a, const struct stat *b) {
#ifdef __APPLE__
    struct timespec am = a->st_mtimespec, bm = b->st_mtimespec;
    struct timespec ac = a->st_ctimespec, bc = b->st_ctimespec;
#else
    struct timespec am = a->st_mtim, bm = b->st_mtim;
    struct timespec ac = a->st_ctim, bc = b->st_ctim;
#endif
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
           a->st_size == b->st_size && a->st_nlink == b->st_nlink &&
           am.tv_sec == bm.tv_sec && am.tv_nsec == bm.tv_nsec &&
           ac.tv_sec == bc.tv_sec && ac.tv_nsec == bc.tv_nsec;
}

#ifdef __linux__
/* Linux stores ACLs as xattrs too. A metadata copy failure must abort the
 * replacement, not silently remove access rules from the edited file. */
static int agent_copy_file_xattrs(int src, int dst) {
    if (fremovexattr(dst, "system.posix_acl_access") != 0 &&
        errno != ENODATA && errno != ENOTSUP) return -1;
    ssize_t count = flistxattr(src, NULL, 0);
    if (count < 0) return errno == ENOTSUP ? 0 : -1;
    if (!count) return 0;
    char *names = xmalloc((size_t)count);
    count = flistxattr(src, names, (size_t)count);
    int rc = -1, saved_errno;
    if (count < 0) goto done;
    for (ssize_t pos = 0; pos < count;) {
        const char *name = names + pos;
        ssize_t len = fgetxattr(src, name, NULL, 0);
        if (len < 0) goto done;
        void *value = xmalloc(len ? (size_t)len : 1);
        ssize_t readlen = fgetxattr(src, name, value, (size_t)len);
        int copied = readlen < 0 ? -1 : fsetxattr(dst, name, value, (size_t)readlen, 0);
        int saved_errno = errno;
        free(value);
        errno = saved_errno;
        if (copied != 0) goto done;
        pos += (ssize_t)strlen(name) + 1;
    }
    rc = 0;
done:
    saved_errno = errno;
    free(names);
    errno = saved_errno;
    return rc;
}
#endif

/* Commit only complete files. Follow existing symlinks, but refuse hard links:
 * an atomic rename cannot preserve their shared-inode semantics. The version
 * checks catch concurrent edits, though unrelated writers do not take a lock. */
static int agent_replace_file(const char *path, const char *data, size_t len,
                              const char *expected, size_t expected_len,
                              char *err, size_t errlen) {
    char target[PATH_MAX], temp[PATH_MAX] = "";
    struct stat before, after;
    int src = -1, fd = -1, rc = -1;
    bool exists = lstat(path, &before) == 0;
    if (!exists && errno != ENOENT) goto failed;
    if (exists) {
        if (!realpath(path, target)) goto failed;
        src = open(target, O_RDWR | O_NONBLOCK | O_NOFOLLOW);
        if (src < 0 || fstat(src, &before) != 0) goto failed;
        if (!S_ISREG(before.st_mode) || before.st_nlink != 1) {
            snprintf(err, errlen, "refusing to replace non-regular or hard-linked file: %s", path);
            goto done;
        }
        if (expected) {
            if (before.st_size < 0 || (uintmax_t)before.st_size != expected_len)
                goto changed;
            char buf[8192];
            size_t pos = 0;
            while (pos < expected_len) {
                size_t count = expected_len - pos;
                if (count > sizeof(buf)) count = sizeof(buf);
                ssize_t n = read(src, buf, count);
                if (n < 0 && errno == EINTR) continue;
                if (n < 0) goto failed;
                if (!n || memcmp(buf, expected + pos, (size_t)n)) goto changed;
                pos += (size_t)n;
            }
        }
    } else {
        if (expected) goto changed;
        if (snprintf(target, sizeof(target), "%s", path) >= (int)sizeof(target)) {
            errno = ENAMETOOLONG;
            goto failed;
        }
    }
    if (snprintf(temp, sizeof(temp), "%s.ds4-XXXXXX", target) >= (int)sizeof(temp)) {
        temp[0] = 0;
        errno = ENAMETOOLONG;
        goto failed;
    }
    fd = mkstemp(temp);
    if (fd < 0) { temp[0] = 0; goto failed; }
    if (!exists) {
        /* Let open apply the process umask without changing it in this thread. */
        if (unlink(temp) != 0) goto failed;
        close(fd);
        fd = open(temp, O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd < 0) { temp[0] = 0; goto failed; }
    }
    for (size_t pos = 0; pos < len;) {
        ssize_t n = write(fd, data + pos, len - pos);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (!n) errno = EIO; goto failed; }
        pos += (size_t)n;
    }
    if (exists) {
        if (fchown(fd, before.st_uid, before.st_gid) != 0) goto failed;
        if (fchmod(fd, before.st_mode & 07777) != 0) goto failed;
#ifdef __APPLE__
        if (fcopyfile(src, fd, NULL, COPYFILE_ACL | COPYFILE_XATTR) != 0) goto failed;
#elif defined(__linux__)
        if (agent_copy_file_xattrs(src, fd) != 0) goto failed;
#endif
    }
    if (fsync(fd) != 0) goto failed;
    if (close(fd) != 0) { fd = -1; goto failed; }
    fd = -1;
    if (exists) {
        char resolved[PATH_MAX];
        if (!realpath(path, resolved) || strcmp(resolved, target) ||
            stat(target, &after) != 0 || !agent_same_file_version(&before, &after))
            goto changed;
        if (rename(temp, target) != 0) goto failed;
    } else {
        /* Unlike rename, link will not overwrite a concurrently created file. */
        if (link(temp, target) != 0) goto failed;
        unlink(temp);
    }
    temp[0] = 0;
    rc = 0;
    goto done;
changed:
    snprintf(err, errlen, "file changed while editing; read it again: %s", path);
    goto done;
failed:
    snprintf(err, errlen, "replace %s: %s", path, strerror(errno));
done:
    if (fd >= 0) close(fd);
    if (src >= 0) close(src);
    if (temp[0]) unlink(temp);
    return rc;
}

static void agent_edit_result_append_line(agent_buf *b, const char *data,
                                          const agent_line_span *sp,
                                          int line) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%d ", line);
    agent_buf_puts(b, prefix);
    agent_buf_append(b, data + sp->start, sp->content_end - sp->start);
    agent_buf_puts(b, "\n");
}

/* Successful edits return the nearby post-edit file shape.  This spends cheap
 * prefill tokens to save expensive model retries: the model immediately sees
 * shifted line numbers, braces, semicolons, and accidental duplication. */
static void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end) {
    enum {
        CONTEXT_BEFORE = 5,
        CONTEXT_AFTER = 8,
        EDITED_CONTEXT_HEAD = 18,
        EDITED_CONTEXT_TAIL = 18
    };

    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (spans.len <= 0) {
        agent_line_spans_free(&spans);
        return;
    }

    if (anchor_start < 1) anchor_start = 1;
    if (anchor_start > spans.len) anchor_start = spans.len;
    if (anchor_end < anchor_start) anchor_end = anchor_start;
    if (anchor_end > spans.len) anchor_end = spans.len;

    int ctx_start = anchor_start - CONTEXT_BEFORE;
    if (ctx_start < 1) ctx_start = 1;
    int ctx_end = anchor_end + CONTEXT_AFTER;
    if (ctx_end > spans.len) ctx_end = spans.len;

    char hdr[PATH_MAX + 160];
    snprintf(hdr, sizeof(hdr),
             "Current file around edit: %s lines %d-%d of %d\n",
             path, ctx_start, ctx_end, spans.len);
    agent_buf_puts(b, hdr);

    int edited_lines = anchor_end - anchor_start + 1;
    if (edited_lines <= EDITED_CONTEXT_HEAD + EDITED_CONTEXT_TAIL) {
        for (int line = ctx_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    } else {
        int head_end = anchor_start + EDITED_CONTEXT_HEAD - 1;
        int tail_start = anchor_end - EDITED_CONTEXT_TAIL + 1;
        for (int line = ctx_start; line <= head_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
        snprintf(hdr, sizeof(hdr),
                 "... %d edited lines omitted ...\n",
                 tail_start - head_end - 1);
        agent_buf_puts(b, hdr);
        for (int line = tail_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    }

    agent_line_spans_free(&spans);
}

static const char *agent_memmem_simple(const char *hay, size_t hay_len,
                                       const char *needle, size_t needle_len) {
    if (!needle_len) return hay;
    if (needle_len > hay_len) return NULL;
    size_t last = hay_len - needle_len;
    for (size_t i = 0; i <= last; i++) {
        if (hay[i] == needle[0] && !memcmp(hay + i, needle, needle_len))
            return hay + i;
    }
    return NULL;
}

static bool agent_find_unique(const char *data, size_t len,
                              const char *needle, size_t needle_len,
                              const char **match, const char *label,
                              char *err, size_t err_len) {
    if (!needle || needle_len == 0) {
        snprintf(err, err_len, "%s anchor is empty", label);
        return false;
    }
    const char *first = agent_memmem_simple(data, len, needle, needle_len);
    if (!first) {
        snprintf(err, err_len, "%s anchor not found", label);
        return false;
    }
    size_t after_first = (size_t)(first - data) + 1;
    const char *second = after_first <= len ?
        agent_memmem_simple(data + after_first, len - after_first,
                            needle, needle_len) : NULL;
    if (second) {
        snprintf(err, err_len, "%s anchor is not unique", label);
        return false;
    }
    *match = first;
    return true;
}

static bool agent_edit_find_old_span(const char *data, size_t len,
                                     const char *old,
                                     const char **match, size_t *match_len,
                                     char *err, size_t err_len) {
    size_t old_len = strlen(old);
    if (!agent_find_unique(data, len, old, old_len, match, "old text",
                           err, err_len))
        return false;
    *match_len = old_len;
    return true;
}

static char *agent_apply_file_splice(const char *path,
                                     const char *data, size_t len,
                                     size_t offset, size_t remove_len,
                                     const char *insert, const char *kind) {
    char err[256];
    if (!insert) insert = "";
    size_t insert_len = strlen(insert);
    size_t out_len = offset + insert_len + (len - offset - remove_len);
    char *out = xmalloc(out_len + 1);
    memcpy(out, data, offset);
    memcpy(out + offset, insert, insert_len);
    memcpy(out + offset + insert_len, data + offset + remove_len,
           len - offset - remove_len);
    out[out_len] = '\0';

    int rc = agent_replace_file(path, out, out_len, data, len, err, sizeof(err));
    if (rc != 0) {
        free(out);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int start_line = 0, end_line = 0, delta = 0;
    agent_old_new_line_effect(data, len, out, out_len, offset, remove_len,
                              &start_line, &end_line, &delta);
    char *result = agent_edit_result(path, start_line, end_line, delta,
                                     out, out_len, kind);
    free(out);
    return result;
}

static char *agent_tool_edit(agent_worker *w, const agent_tool_call *call) {
    (void)w; /* exact-match old/new only (plan 3b); w was only needed for the
              * removed anchored-edit toggle */
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) return xstrdup("Tool error: edit requires path\n");
    const char *old = agent_tool_arg_value(call, "old");
    const char *new_text = agent_tool_arg_value(call, "new");
    if (!old || !old[0]) return xstrdup("Tool error: edit requires non-empty old text\n");
    if (!new_text) return xstrdup("Tool error: edit requires new text\n");

    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    const char *match = NULL;
    size_t match_len = 0;
    if (!agent_edit_find_old_span(data, len, old, &match, &match_len,
                                  err, sizeof(err)))
    {
        free(data);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char *result = agent_apply_file_splice(path, data, len,
                                           (size_t)(match - data), match_len,
                                           new_text, "old/new replacement");
    free(data);
    return result;
}

typedef struct {
    const char *query;
    const char *glob;
    regex_t regex;
    bool use_regex;
    bool regex_ready;
    bool case_sensitive;
    int context;
    int max_results;
    int results;
    size_t skipped;
    char first_skip[PATH_MAX + 128];
    agent_buf out;
} agent_search_ctx;

static bool agent_literal_match(const char *s, size_t n, const char *q,
                                bool case_sensitive) {
    size_t qn = strlen(q);
    if (!qn) return true;
    if (qn > n) return false;
    for (size_t i = 0; i + qn <= n; i++) {
        bool ok = true;
        for (size_t j = 0; j < qn; j++) {
            unsigned char a = (unsigned char)s[i + j];
            unsigned char b = (unsigned char)q[j];
            if (!case_sensitive) {
                a = (unsigned char)tolower(a);
                b = (unsigned char)tolower(b);
            }
            if (a != b) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

static bool agent_search_line_matches(agent_search_ctx *ctx, const char *s, size_t n) {
    if (ctx->use_regex) {
        char *line = xstrndup(s, n);
        int rc = regexec(&ctx->regex, line, 0, NULL, 0);
        free(line);
        return rc == 0;
    }
    return agent_literal_match(s, n, ctx->query, ctx->case_sensitive);
}

static void agent_search_emit_line(agent_search_ctx *ctx, const char *data,
                                   int line_no) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "  %d ", line_no);
    agent_buf_puts(&ctx->out, prefix);
    agent_buf_puts(&ctx->out, data);
    agent_buf_puts(&ctx->out, "\n");
}

static void agent_search_skip(agent_search_ctx *ctx, const char *path, const char *why) {
    ctx->skipped++;
    if (!ctx->first_skip[0])
        snprintf(ctx->first_skip, sizeof(ctx->first_skip), "%s: %s", path, why);
}

/* A file can be arbitrarily large; only the current line and a small context
 * ring are resident. Oversized or binary lines are reported, never ignored. */
static int agent_search_read_line(FILE *fp, char **text) {
    agent_buf line = {0};
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (!c || line.len == AGENT_TOOL_MAX_BYTES) {
            free(line.ptr);
            return -2;
        }
        if (c == '\n' || c == '\r') {
            if (c == '\r') {
                int next = fgetc(fp);
                if (next != '\n' && next != EOF) ungetc(next, fp);
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&line, &ch, 1);
    }
    if (ferror(fp)) { free(line.ptr); return -1; }
    if (c == EOF && !line.len) { free(line.ptr); return 0; }
    *text = agent_buf_take(&line);
    return 1;
}

/* Search one text file and emit matching lines with plain line numbers. */
static void agent_search_file(agent_search_ctx *ctx, const char *path) {
    if (ctx->results >= ctx->max_results || ctx->out.truncated) return;
    if (ctx->glob && ctx->glob[0]) {
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (fnmatch(ctx->glob, base, 0) != 0 && fnmatch(ctx->glob, path, 0) != 0)
            return;
    }
    FILE *fp = agent_open_regular_file(path);
    if (!fp) { agent_search_skip(ctx, path, strerror(errno)); return; }
    char *ring[6] = {0};
    bool printed_file = false;
    int line = 0, last_emitted = 0, after = 0;
    while ((ctx->results < ctx->max_results || after) && !ctx->out.truncated) {
        char *text = NULL;
        int rc = agent_search_read_line(fp, &text);
        if (!rc) break;
        if (rc < 0) {
            agent_search_skip(ctx, path, rc == -2 ? "binary data or line exceeds 128 KiB" : strerror(errno));
            break;
        }
        if (line == INT_MAX) {
            free(text);
            agent_search_skip(ctx, path, "line number limit reached");
            break;
        }
        line++;
        free(ring[line % 6]);
        ring[line % 6] = text;
        bool match = ctx->results < ctx->max_results &&
                     agent_search_line_matches(ctx, text, strlen(text));
        if (match && !printed_file) {
            agent_buf_puts(&ctx->out, path);
            agent_buf_puts(&ctx->out, "\n");
            printed_file = true;
        }
        if (match) {
            int from = line - ctx->context;
            if (from <= last_emitted) from = last_emitted + 1;
            for (int j = from; j <= line; j++)
                agent_search_emit_line(ctx, ring[j % 6], j);
            last_emitted = line;
            after = ctx->context;
            ctx->results++;
        } else if (after) {
            agent_search_emit_line(ctx, text, line);
            last_emitted = line;
            after--;
        }
    }
    if (printed_file) agent_buf_puts(&ctx->out, "\n");
    for (int i = 0; i < 6; i++) free(ring[i]);
    fclose(fp);
}

/* Recursively search a file or directory, avoiding .git and stopping once the
 * result cap is reached. */
static void agent_search_path(agent_search_ctx *ctx, const char *path, int depth) {
    if (ctx->results >= ctx->max_results || ctx->out.truncated) return;
    if (depth > 24) { agent_search_skip(ctx, path, "directory depth limit reached"); return; }
    struct stat st;
    if ((depth == 0 ? stat(path, &st) : lstat(path, &st)) != 0) {
        agent_search_skip(ctx, path, strerror(errno));
        return;
    }
    if (S_ISREG(st.st_mode)) {
        agent_search_file(ctx, path);
        return;
    }
    if (!S_ISDIR(st.st_mode)) {
        agent_search_skip(ctx, path, "not a regular file or directory (nested symlinks are not followed)");
        return;
    }
    DIR *dir = opendir(path);
    if (!dir) { agent_search_skip(ctx, path, strerror(errno)); return; }
    struct dirent *de;
    while (ctx->results < ctx->max_results && !ctx->out.truncated) {
        errno = 0;
        de = readdir(dir);
        if (!de) {
            if (errno) agent_search_skip(ctx, path, strerror(errno));
            break;
        }
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (!strcmp(de->d_name, ".git")) continue;
        char child[PATH_MAX];
        if (snprintf(child, sizeof(child), "%s/%s", path, de->d_name) >= (int)sizeof(child)) {
            agent_search_skip(ctx, path, "path exceeds PATH_MAX");
            continue;
        }
        agent_search_path(ctx, child, depth + 1);
    }
    closedir(dir);
}

/* Implement the search tool using either literal matching or POSIX regex. */
static char *agent_tool_search(agent_worker *w, const agent_tool_call *call) {
    (void)w;
    const char *query = agent_tool_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: search requires query\n");
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    const char *mode = agent_tool_arg_value(call, "mode");
    if (mode && strcmp(mode, "literal") && strcmp(mode, "regex"))
        return xstrdup("Tool error: search mode must be literal or regex (POSIX extended)\n");
    struct stat root;
    if (stat(path, &root) != 0 || (!S_ISDIR(root.st_mode) && !S_ISREG(root.st_mode))) {
        agent_buf error = {0};
        agent_buf_puts(&error, "Tool error: search path is missing, unreadable, or not a regular file/directory: ");
        agent_buf_puts(&error, path);
        agent_buf_puts(&error, "\n");
        return agent_buf_take(&error);
    }
    agent_search_ctx ctx = {
        .query = query,
        .glob = agent_tool_arg_value(call, "glob"),
        .use_regex = mode && !strcmp(mode, "regex"),
        .case_sensitive = agent_parse_bool_default(agent_tool_arg_value(call, "case_sensitive"), true),
        .context = agent_parse_int_default(agent_tool_arg_value(call, "context"), 0, 0, 5),
        .max_results = agent_parse_int_default(agent_tool_arg_value(call, "max_results"), 50, 1, 500),
        .out = {.limit = AGENT_TOOL_MAX_BYTES},
    };
    if (ctx.use_regex) {
        int flags = REG_EXTENDED | REG_NOSUB;
        if (!ctx.case_sensitive) flags |= REG_ICASE;
        int rc = regcomp(&ctx.regex, query, flags);
        if (rc != 0) {
            char msg[256];
            regerror(rc, &ctx.regex, msg, sizeof(msg));
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: invalid regex: ");
            agent_buf_puts(&b, msg);
            agent_buf_puts(&b, "\n");
            return agent_buf_take(&b);
        }
        ctx.regex_ready = true;
    }
    agent_search_path(&ctx, path, 0);
    if (ctx.regex_ready) regfree(&ctx.regex);
    if (!ctx.out.ptr) agent_buf_puts(&ctx.out, "No matches in searched text\n");
    else {
        char hdr[96];
        snprintf(hdr, sizeof(hdr), "%d match%s shown\n\n",
                 ctx.results, ctx.results == 1 ? "" : "es");
        size_t hdr_len = strlen(hdr);
        if (ctx.out.len + hdr_len + 1 > ctx.out.cap) {
            ctx.out.cap = ctx.out.len + hdr_len + 1;
            ctx.out.ptr = xrealloc(ctx.out.ptr, ctx.out.cap);
        }
        memmove(ctx.out.ptr + hdr_len, ctx.out.ptr, ctx.out.len + 1);
        memcpy(ctx.out.ptr, hdr, hdr_len);
        ctx.out.len += hdr_len;
    }
    ctx.out.limit = 0;
    if (ctx.skipped || ctx.results >= ctx.max_results) {
        bool truncated = ctx.out.truncated;
        ctx.out.truncated = false;
        char note[PATH_MAX + 256];
        snprintf(note, sizeof(note), "\nSearch incomplete: %zu skipped paths%s. %s\n",
                 ctx.skipped, ctx.results >= ctx.max_results ? "; match limit reached" : "", ctx.first_skip);
        agent_buf_puts(&ctx.out, note);
        ctx.out.truncated = truncated;
    }
    return agent_buf_take(&ctx.out);
}

static int agent_count_lines(const char *s) {
    if (!s || !s[0]) return 0;
    int lines = 0;
    for (const char *p = s; *p; p++) {
        if (*p == '\n') lines++;
    }
    if (s[strlen(s) - 1] != '\n') lines++;
    return lines;
}

static char *agent_string_head(const char *s, int max_lines, size_t max_bytes,
                               int *lines_read, bool *byte_limited) {
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!s) return xstrdup("");
    size_t used = 0;
    int lines = 0;
    while (s[used] && used < max_bytes && lines < max_lines) {
        if (s[used++] == '\n') lines++;
    }
    if (s[used] && used >= max_bytes && byte_limited) *byte_limited = true;
    if (used && s[used - 1] != '\n' && lines < max_lines) lines++;
    if (lines_read) *lines_read = lines;
    return xstrndup(s, used);
}

static bool agent_write_temp_text(const char *prefix, const char *text,
                                  char *path, size_t path_len,
                                  char *err, size_t err_len) {
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "/tmp/%s_XXXXXX", prefix);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        snprintf(err, err_len, "failed to create temporary file: %s", strerror(errno));
        return false;
    }
    size_t len = text ? strlen(text) : 0;
    const char *p = text ? text : "";
    size_t left = len;
    while (left) {
        ssize_t n = write(fd, p, left);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            snprintf(err, err_len, "failed to write temporary file: %s", strerror(errno));
            close(fd);
            unlink(tmpl);
            return false;
        }
        p += n;
        left -= (size_t)n;
    }
    if (close(fd) != 0) {
        snprintf(err, err_len, "failed to close temporary file: %s", strerror(errno));
        unlink(tmpl);
        return false;
    }
    snprintf(path, path_len, "%s", tmpl);
    return true;
}

static char *agent_tool_google_search(agent_worker *w, const agent_tool_call *call) {
    const char *query = agent_tool_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: google_search requires query\n");
    char err[256] = {0};
    agent_publishf_system_status(w, "Searching Google for %s...", query);
    char *md = ds4_web_google_search(w->web, query, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: google_search failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    return md;
}

static char *agent_tool_visit_page(agent_worker *w, const agent_tool_call *call) {
    const char *url = agent_tool_arg_value(call, "url");
    if (!url || !url[0]) return xstrdup("Tool error: visit_page requires url\n");
    char err[256] = {0};
    agent_publishf_system_status(w, "Opening page %s...", url);
    char *md = ds4_web_visit_page(w->web, url, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char path[PATH_MAX];
    if (!agent_write_temp_text("ds4_agent_web", md, path, sizeof(path),
                               err, sizeof(err)))
    {
        free(md);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "could not store rendered page");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int total_lines = agent_count_lines(md);
    int shown_lines = 0;
    bool byte_limited = false;
    char *head = agent_string_head(md, AGENT_WEB_HEAD_LINES, AGENT_WEB_HEAD_BYTES,
                                   &shown_lines, &byte_limited);
    bool truncated = byte_limited || shown_lines < total_lines;
    agent_buf out = {0};
    char line[PATH_MAX + 256];
    snprintf(line, sizeof(line),
             "visit_page url=%s\noutput_path=%s (%zu bytes, %d lines)\n",
             url, path, strlen(md), total_lines);
    agent_buf_puts(&out, line);
    if (truncated) {
        snprintf(line, sizeof(line), "<head -%d %s>\n",
                 AGENT_WEB_HEAD_LINES, path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</head>\n");
        agent_buf_puts(&out,
            "Use read path=<output_path> start_line=<line> max_lines=<count> raw=true to inspect more rendered Markdown.\n");
    } else {
        agent_buf_puts(&out, "<markdown>\n");
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</markdown>\n");
    }
    free(head);
    free(md);
    return agent_buf_take(&out);
}

struct agent_bash_job {
    pthread_t thread;
    pthread_mutex_t mu;
    bool thread_started;
    int id;
    pid_t pid;
    int pipe_fd;
    int tmp_fd;
    char path[PATH_MAX];
    double start_time;
    double end_time;
    double stop_deadline;
    double timeout_sec;
    size_t bytes;
    int newline_count;
    char last_byte;
    bool observed_once;
    int exit_status;
    bool running;
    bool timed_out;
    bool reaped, pipe_eof;
    int child_status;
    int output_error;
    struct agent_bash_job *next;
    agent_worker *worker;  /* back-pointer for terminal state restoration */
};

static int agent_bash_display_lines(const agent_bash_job *job) {
    if (!job || job->bytes == 0) return 0;
    return job->newline_count + (job->last_byte != '\n');
}

static void agent_bash_note_output(agent_bash_job *job, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n' && job->newline_count < INT_MAX - 1) job->newline_count++;
    }
    if (n) job->last_byte = s[n - 1];
    job->bytes += n;
}

static bool agent_bash_is_running(agent_bash_job *job) {
    pthread_mutex_lock(&job->mu);
    bool running = job->running;
    pthread_mutex_unlock(&job->mu);
    return running;
}

static void agent_bash_signal(agent_bash_job *job, int sig) {
    pthread_mutex_lock(&job->mu);
    if (job->running) {
        if (sig == SIGKILL && !job->stop_deadline) job->stop_deadline = now_sec() + 1;
        kill(-job->pid, sig);
        if (!job->reaped) kill(job->pid, sig);
    }
    pthread_mutex_unlock(&job->mu);
}

static void agent_bash_job_free(agent_bash_job *job) {
    if (!job) return;
    agent_bash_signal(job, SIGKILL);
    if (job->thread_started) pthread_join(job->thread, NULL);
    else if (job->pid > 0)
        while (waitpid(job->pid, NULL, 0) < 0 && errno == EINTR) {}
    if (job->pipe_fd >= 0) close(job->pipe_fd);
    if (job->tmp_fd >= 0) close(job->tmp_fd);
    pthread_mutex_destroy(&job->mu);
    free(job);
}

static void agent_bash_jobs_free(agent_worker *w) {
    agent_bash_job *job = w->bash_jobs;
    while (job) {
        agent_bash_job *next = job->next;
        agent_bash_job_free(job);
        job = next;
    }
    w->bash_jobs = NULL;
}

static agent_bash_job *agent_bash_find_job(agent_worker *w, int id, pid_t pid) {
    for (agent_bash_job *job = w->bash_jobs; job; job = job->next) {
        if ((id > 0 && job->id == id) || (id <= 0 && pid > 0 && job->pid == pid))
            return job;
    }
    return NULL;
}

static void agent_bash_remove_job(agent_worker *w, agent_bash_job *target) {
    agent_bash_job **link = &w->bash_jobs;
    while (*link) {
        if (*link == target) {
            *link = target->next;
            target->next = NULL;
            agent_bash_job_free(target);
            return;
        }
        link = &(*link)->next;
    }
}

static void agent_bash_drain(agent_bash_job *job) {
    if (!job || job->pipe_fd < 0) return;
    char tmp[4096];
    /* Yield even for an endless writer, so its deadline is still checked. */
    for (size_t drained = 0; drained < 256 * 1024;) {
        ssize_t n = read(job->pipe_fd, tmp, sizeof(tmp));
        if (n > 0) {
            drained += (size_t)n;
            size_t pos = 0;
            while (pos < (size_t)n) {
                ssize_t wr = write(job->tmp_fd, tmp + pos, (size_t)n - pos);
                if (wr < 0 && errno == EINTR) continue;
                if (wr <= 0) {
                    job->output_error = wr < 0 ? errno : EIO;
                    return;
                }
                agent_bash_note_output(job, tmp + pos, (size_t)wr);
                pos += (size_t)wr;
            }
            continue;
        }
        if (!n) job->pipe_eof = true;
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            job->output_error = errno;
        break;
    }
}

static void agent_worker_note_terminal_mode_may_have_changed(agent_worker *w) {
    if (!w) return;
    pthread_mutex_lock(&w->mu);
    w->raw_mode_needs_restore = true;
    pthread_mutex_unlock(&w->mu);
}

static void agent_bash_finalize(agent_bash_job *job, int status) {
    agent_bash_drain(job);
    if (job->pipe_fd >= 0) {
        close(job->pipe_fd);
        job->pipe_fd = -1;
    }
    if (job->tmp_fd >= 0) {
        close(job->tmp_fd);
        job->tmp_fd = -1;
    }
    if (WIFEXITED(status)) job->exit_status = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) job->exit_status = 128 + WTERMSIG(status);
    else job->exit_status = -1;
    job->running = false;
    job->end_time = now_sec();
    /* A child can still open /dev/tty directly and alter terminal state even
     * though its stdin is /dev/null.  Ask the UI thread to verify raw mode at
     * a safe point instead of touching linenoise from the worker path. */
    agent_worker_note_terminal_mode_may_have_changed(job->worker);
}

/* The monitor owns the descriptors and child reaping. Tool observations take
 * the job mutex; the model worker owns only the list and observation cursors. */
static void agent_bash_poll(agent_bash_job *job) {
    if (!job || !job->running) return;
    agent_bash_drain(job);

    int status = 0;
    pid_t rc = job->reaped ? 0 : waitpid(job->pid, &status, WNOHANG);
    if (rc == job->pid) {
        job->reaped = true;
        job->child_status = status;
    }
    if (rc < 0 && errno != EINTR) {
        job->exit_status = -1;
        job->running = false;
        job->end_time = now_sec();
        if (job->pipe_fd >= 0) {
            close(job->pipe_fd);
            job->pipe_fd = -1;
        }
        if (job->tmp_fd >= 0) {
            close(job->tmp_fd);
            job->tmp_fd = -1;
        }
        agent_worker_note_terminal_mode_may_have_changed(job->worker);
        return;
    }
    if (job->reaped && job->pipe_eof) {
        agent_bash_finalize(job, job->child_status);
        return;
    }
    if (!job->timed_out && (job->output_error || now_sec() - job->start_time >= job->timeout_sec)) {
        job->timed_out = !job->output_error;
        kill(-job->pid, SIGKILL);
        if (!job->reaped) {
            kill(job->pid, SIGKILL);
            while (waitpid(job->pid, &status, 0) < 0 && errno == EINTR) {}
            job->reaped = true;
            job->child_status = status;
        }
    }
    if (job->output_error || (job->timed_out && now_sec() - job->start_time > job->timeout_sec + 1) ||
        (job->stop_deadline && now_sec() >= job->stop_deadline)) {
        /* A descendant can deliberately escape the shell process group. Do
         * not let its inherited stdout prevent bounded job cleanup. */
        if (!job->output_error && !job->pipe_eof) job->output_error = EIO;
        agent_bash_finalize(job, job->child_status);
    }
}

static void *agent_bash_monitor(void *arg) {
    agent_bash_job *job = arg;
    for (;;) {
        pthread_mutex_lock(&job->mu);
        agent_bash_poll(job);
        bool running = job->running;
        int fd = job->pipe_fd;
        pthread_mutex_unlock(&job->mu);
        if (!running) break;
        struct pollfd pfd = {.fd = fd, .events = POLLIN};
        int rc = poll(&pfd, 1, 50);
        /* A closed stdout must not turn the deadline monitor into a busy loop. */
        if (rc > 0 && (pfd.revents & POLLHUP) && !(pfd.revents & POLLIN))
            usleep(50000);
    }
    return NULL;
}

/* Spawn a shell command into its own process group so bash_stop/timeout can
 * kill grandchildren created by the shell, not just the /bin/sh wrapper. */
static agent_bash_job *agent_bash_start(agent_worker *w, const char *cmd,
                                        int timeout_sec, char *err, size_t err_len) {
    char tmp_path[] = "/tmp/ds4_agent_output_XXXXXX";
    int tmpfd = mkstemp(tmp_path);
    if (tmpfd < 0) {
        snprintf(err, err_len, "failed to create temporary output file: %s", strerror(errno));
        return NULL;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        snprintf(err, err_len, "failed to create pipe: %s", strerror(errno));
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    fcntl(tmpfd, F_SETFD, FD_CLOEXEC);
    fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);
    fcntl(pipefd[1], F_SETFD, FD_CLOEXEC);
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, err_len, "failed to fork: %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    if (pid == 0) {
        setpgid(0, 0);
        close(tmpfd);
        /* The bash tool is not interactive.  Give the shell /dev/null as
         * stdin so it does not inherit the live linenoise terminal and reset
         * it from raw mode to cooked mode behind the agent's back. */
        int null_fd = open("/dev/null", O_RDONLY);
        if (null_fd >= 0) {
            if (dup2(null_fd, STDIN_FILENO) < 0)
                close(STDIN_FILENO);
            if (null_fd != STDIN_FILENO)
                close(null_fd);
        } else {
            close(STDIN_FILENO);
        }
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmd ? cmd : "", (char *)NULL);
        _exit(127);
    }

    close(pipefd[1]);
    setpgid(pid, pid);
    int old_flags;
    set_nonblock(pipefd[0], true, &old_flags);

    agent_bash_job *job = xmalloc(sizeof(*job));
    memset(job, 0, sizeof(*job));
    pthread_mutex_init(&job->mu, NULL);
    if (w->next_bash_job_id <= 0) w->next_bash_job_id = 1;
    job->id = w->next_bash_job_id++;
    job->pid = pid;
    job->pipe_fd = pipefd[0];
    job->tmp_fd = tmpfd;
    snprintf(job->path, sizeof(job->path), "%s", tmp_path);
    job->start_time = now_sec();
    job->timeout_sec = timeout_sec;
    job->exit_status = -1;
    job->running = true;
    job->worker = w;
    int thread_rc = pthread_create(&job->thread, NULL, agent_bash_monitor, job);
    if (thread_rc != 0) {
        snprintf(err, err_len, "failed to monitor shell command: %s", strerror(thread_rc));
        agent_bash_job_free(job);
        unlink(tmp_path);
        return NULL;
    }
    job->thread_started = true;
    job->next = w->bash_jobs;
    w->bash_jobs = job;
    return job;
}

/* Read the first max_lines from the output file, with a byte cap to avoid a
 * pathological single long line flooding the next model turn. */
static char *agent_bash_read_head(const agent_bash_job *job, int max_lines,
                                  size_t max_bytes, int *lines_read,
                                  bool *byte_limited) {
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf out = {0};
    int lines = 0;
    size_t available = job->bytes < max_bytes ? job->bytes : max_bytes;
    while (lines < max_lines && out.len < available) {
        int c = fgetc(fp);
        if (c == EOF) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&out, &ch, 1);
        if (ch == '\n') lines++;
    }
    if (out.len >= max_bytes && out.len < job->bytes && byte_limited) *byte_limited = true;
    fclose(fp);
    if (lines_read) *lines_read = lines + (out.len && out.ptr[out.len - 1] != '\n');
    if (!out.ptr) return xstrdup("");
    return agent_buf_take(&out);
}

/* Read the last max_lines from the full output file.  The model-visible label
 * says "tail -N <file>" so it is clear this is not the complete output. */
static char *agent_bash_read_tail_lines(const agent_bash_job *job, int max_lines) {
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf tail = {0};
    size_t bytes = job->bytes;
    size_t take = bytes < AGENT_BASH_TAIL_BYTES ? bytes : AGENT_BASH_TAIL_BYTES;
    if (fseeko(fp, (off_t)(bytes - take), SEEK_SET) != 0) {
        fclose(fp);
        return xstrdup("<failed to seek output file>\n");
    }
    char tmp[2048];
    while (take) {
        size_t want = take < sizeof(tmp) ? take : sizeof(tmp);
        size_t n = fread(tmp, 1, want, fp);
        if (n) agent_buf_append(&tail, tmp, n);
        take -= n;
        if (n < want) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
    }
    fclose(fp);
    if (!tail.ptr) return xstrdup("");

    char *start = tail.ptr;
    int newlines = tail.ptr[tail.len - 1] == '\n' ? 0 : 1;
    for (char *p = tail.ptr + tail.len; p > tail.ptr; p--) {
        if (p[-1] == '\n' && ++newlines > max_lines) {
            start = p;
            break;
        }
    }
    /* A byte-limited tail may begin inside a UTF-8 character. */
    while (((unsigned char)*start & 0xc0) == 0x80) start++;
    char *out = xstrdup(start);
    free(tail.ptr);
    return out;
}

/* The first observation uses the head; subsequent ones use a bounded tail. */
static char *agent_bash_observation(agent_bash_job *job, bool mark_observed, bool *done) {
    pthread_mutex_lock(&job->mu);
    agent_bash_job snapshot = {
        .id = job->id, .pid = job->pid,
        .start_time = job->start_time, .end_time = job->end_time,
        .timeout_sec = job->timeout_sec, .bytes = job->bytes,
        .newline_count = job->newline_count, .last_byte = job->last_byte,
        .observed_once = job->observed_once, .exit_status = job->exit_status,
        .running = job->running, .timed_out = job->timed_out,
        .output_error = job->output_error,
    };
    memcpy(snapshot.path, job->path, sizeof(snapshot.path));
    if (mark_observed) job->observed_once = true;
    pthread_mutex_unlock(&job->mu);
    /* Disk reads must not hold up the monitor's deadline checks. Only the
     * immutable observation fields above are read from this local snapshot. */
    job = &snapshot;
    if (done) *done = !snapshot.running;
    bool first_observation = !job->observed_once;
    int display_lines = agent_bash_display_lines(job);
    double elapsed = (job->running ? now_sec() : job->end_time) - job->start_time;

    agent_buf out = {0};
    char line[PATH_MAX + 256];
    if (job->running) {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=running elapsed_sec=%.1f timeout_sec=%.0f\n",
            job->id, (long)job->pid, elapsed, job->timeout_sec);
    } else {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=done elapsed_sec=%.1f timed_out=%d\n",
            job->id, (long)job->pid, elapsed, job->timed_out ? 1 : 0);
    }
    agent_buf_puts(&out, line);
    if (!job->running) {
        snprintf(line, sizeof(line), "exit_status=%d\n", job->exit_status);
        agent_buf_puts(&out, line);
    }
    if (job->output_error) {
        agent_buf_puts(&out, "Tool error: command output could not be captured completely: ");
        agent_buf_puts(&out, strerror(job->output_error));
        agent_buf_puts(&out, "\n");
    }

    if (job->bytes == 0) {
        agent_buf_puts(&out, "<output>\n</output>\n");
    } else if (first_observation) {
        int shown_lines = 0;
        bool byte_limited = false;
        char *head = agent_bash_read_head(job, AGENT_BASH_HEAD_LINES,
                                          AGENT_BASH_HEAD_BYTES,
                                          &shown_lines, &byte_limited);
        bool truncated = byte_limited || display_lines > shown_lines;
        if (!job->running && !truncated) {
            agent_buf_puts(&out, "<output>\n");
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</output>\n");
        } else {
            snprintf(line, sizeof(line),
                     "output_path=%s (%zu bytes, %d lines)\n",
                     job->path[0] ? job->path : "<unavailable>",
                     job->bytes, display_lines);
            agent_buf_puts(&out, line);
            snprintf(line, sizeof(line), "<head -%d %s>\n",
                     AGENT_BASH_HEAD_LINES, job->path);
            agent_buf_puts(&out, line);
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</head>\n");
        }
        free(head);
    } else {
        int tail_lines = job->running ? AGENT_BASH_PROGRESS_TAIL_LINES :
                                        AGENT_BASH_FINAL_TAIL_LINES;
        char *tail = agent_bash_read_tail_lines(job, tail_lines);
        snprintf(line, sizeof(line),
                 "output_path=%s (%zu bytes, %d lines)\n",
                 job->path[0] ? job->path : "<unavailable>",
                 job->bytes, display_lines);
        agent_buf_puts(&out, line);
        snprintf(line, sizeof(line), "<tail -%d %s>\n", tail_lines, job->path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, tail);
        if (tail[0] && tail[strlen(tail) - 1] != '\n') agent_buf_puts(&out, "\n");
        snprintf(line, sizeof(line), "</tail>\n");
        agent_buf_puts(&out, line);
        free(tail);
    }
    if (job->running) {
        snprintf(line, sizeof(line),
            "\nUse bash_status job=%d to get info before refresh time; use bash_stop job=%d to stop execution\n",
            job->id, job->id);
        agent_buf_puts(&out, line);
    }

    return agent_buf_take(&out);
}

/* Shell bytes are data, not instructions to our terminal. Keep only SGR color
 * sequences; OSC, cursor motion, erasure and other controls stay in the log. */
static char *agent_terminal_safe_text(const char *text, size_t len) {
    agent_buf out = {0};
    for (size_t i = 0; i < len;) {
        unsigned char c = (unsigned char)text[i++];
        if (c >= 0x80) {
            uint32_t cp;
            size_t n = linenoiseUtf8Decode(text + i - 1, len - i + 1, &cp);
            if (n > 1 && !(cp >= 0x80 && cp <= 0x9f)) {
                agent_buf_append(&out, text + i - 1, n);
                i += n - 1;
                continue;
            }
            char escaped[5];
            snprintf(escaped, sizeof(escaped), "\\x%02x", c);
            agent_buf_puts(&out, escaped);
            continue;
        }
        if (c == 0x1b) {
            size_t start = i - 1;
            if (i == len) break;
            char kind = text[i++];
            if (kind == '[') {
                bool sgr = true;
                while (i < len && (unsigned char)text[i] < 0x40) {
                    if (!isdigit((unsigned char)text[i]) && text[i] != ';' && text[i] != ':') sgr = false;
                    i++;
                }
                if (i < len) {
                    if (text[i] == 'm' && sgr && i - start < 96)
                        agent_buf_append(&out, text + start, i - start + 1);
                    i++;
                }
            } else if (kind == ']' || kind == 'P' || kind == '_' || kind == '^' || kind == 'X') {
                while (i < len) {
                    if (text[i++] == '\a' && kind == ']') break;
                    if (i >= 2 && text[i - 2] == 0x1b && text[i - 1] == '\\') break;
                }
            } else {
                while (kind >= 0x20 && kind <= 0x2f && i < len) kind = text[i++];
            }
            continue;
        }
        if ((c < 32 && c != '\n' && c != '\r' && c != '\t') || c == 127) {
            char escaped[5];
            snprintf(escaped, sizeof(escaped), "\\x%02x", c);
            agent_buf_puts(&out, escaped);
        } else {
            char ch = (char)c;
            agent_buf_append(&out, &ch, 1);
        }
    }
    return agent_buf_take(&out);
}

static void agent_bash_publish_observation(agent_worker *w, const char *obs) {
    if (!obs || !obs[0]) return;
    const char *body = NULL;
    const char *label = strstr(obs, "\n<head ");
    const char *close = NULL;
    if (label) {
        close = "</head>";
    } else {
        label = strstr(obs, "\n<tail ");
        if (label) close = "</tail>";
    }
    if (label) {
        const char *tag_end = strstr(label, ">\n");
        if (tag_end) {
            agent_publish(w, "\x1b[90m", 5);
            if (strstr(label, "\n<head ") == label)
                agent_publish(w, "[showing first output lines]\n",
                              strlen("[showing first output lines]\n"));
            else
                agent_publish(w, "[showing last output lines]\n",
                              strlen("[showing last output lines]\n"));
            agent_publish(w, "\x1b[0m", 4);
            body = tag_end + 2;
        }
    } else {
        label = strstr(obs, "\n<output>\n");
        if (label) {
            body = label + strlen("\n<output>\n");
            close = "</output>";
        }
    }
    if (!body || !body[0]) return;
    const char *end = close ? strstr(body, close) : NULL;
    size_t n = end ? (size_t)(end - body) : strlen(body);
    if (n) {
        bool failed = strstr(obs, "status=done") && !strstr(obs, "exit_status=0\n");
        if (failed) agent_publish(w, "\x1b[38;5;208m", 11);
        char *safe = agent_terminal_safe_text(body, n);
        agent_publish(w, safe, strlen(safe));
        if (safe[0] && safe[strlen(safe) - 1] != '\n') agent_publish(w, "\n", 1);
        agent_publish(w, "\x1b[0m", 4);
        free(safe);
    }
}

static void agent_bash_refresh_for(agent_worker *w, agent_bash_job *job,
                                   int refresh_sec) {
    double start = now_sec();
    while (agent_bash_is_running(job) && now_sec() - start < refresh_sec) {
        if (worker_should_interrupt(w)) break;
        usleep(20000);
    }
}

/* Common implementation for bash, bash_status, and bash_stop. */
static char *agent_bash_job_tool_result(agent_worker *w, agent_bash_job *job,
                                        bool wait, int refresh_sec,
                                        bool stop, bool remove_if_done) {
    if (stop && agent_bash_is_running(job)) {
        agent_bash_signal(job, SIGTERM);
        double start = now_sec();
        while (agent_bash_is_running(job) && now_sec() - start < 1.0) {
            usleep(20000);
        }
        agent_bash_signal(job, SIGKILL);
    }
    if (wait || stop) agent_bash_refresh_for(w, job, refresh_sec);

    bool done = false;
    char *obs = agent_bash_observation(job, true, &done);
    agent_bash_publish_observation(w, obs);
    if (remove_if_done && done) agent_bash_remove_job(w, job);
    return obs;
}

static int agent_tool_job_id(const agent_tool_call *call) {
    return agent_parse_int_default(agent_tool_arg_value(call, "job"), 0, 0, INT_MAX);
}

static pid_t agent_tool_pid(const agent_tool_call *call) {
    return (pid_t)agent_parse_int_default(agent_tool_arg_value(call, "pid"), 0, 0, INT_MAX);
}

/* Execute one parsed tool call and return the text that will be appended as
 * the tool-role result. UI visualisation already happened while streaming
 * (T9); this function is only about side effects and the model-visible
 * observation. */
static char *agent_execute_tool_call(agent_worker *w, const agent_tool_call *call) {
    agent_buf result = {0};
    if (!call->name) return xstrdup("Tool error: missing tool name\n");

    if (!strcmp(call->name, "read")) return agent_tool_read(w, call);
    if (!strcmp(call->name, "more")) return agent_tool_more(w, call);
    if (!strcmp(call->name, "write")) return agent_tool_write(w, call);
    if (!strcmp(call->name, "list")) return agent_tool_list(call);
    if (!strcmp(call->name, "edit")) return agent_tool_edit(w, call);
    if (!strcmp(call->name, "search")) return agent_tool_search(w, call);
    if (!strcmp(call->name, "google_search")) return agent_tool_google_search(w, call);
    if (!strcmp(call->name, "visit_page")) return agent_tool_visit_page(w, call);

    if (!strcmp(call->name, "bash")) {
        const char *cmd = agent_tool_arg_value(call, "command");
        if (!cmd || !cmd[0]) return xstrdup("Tool error: bash requires command\n");
        int timeout = agent_parse_timeout(agent_tool_arg_value(call, "timeout_sec"));
        int refresh = agent_parse_int_default(agent_tool_arg_value(call, "refresh_sec"),
                                              60, 1, 3600);
        char err[160] = {0};
        agent_bash_job *job = agent_bash_start(w, cmd, timeout, err, sizeof(err));
        if (!job) {
            agent_buf_puts(&result, "Tool error: bash failed to start: ");
            agent_buf_puts(&result, err[0] ? err : "unknown error");
            agent_buf_puts(&result, "\n");
            return agent_buf_take(&result);
        }
        return agent_bash_job_tool_result(w, job, true, refresh, false, true);
    }

    if (!strcmp(call->name, "bash_status") ||
        !strcmp(call->name, "bash_stop"))
    {
        int job_id = agent_tool_job_id(call);
        pid_t pid = agent_tool_pid(call);
        agent_bash_job *job = agent_bash_find_job(w, job_id, pid);
        if (!job) {
            char msg[128];
            snprintf(msg, sizeof(msg), "Tool error: bash job not found: job=%d pid=%ld\n",
                     job_id, (long)pid);
            return xstrdup(msg);
        }
        int refresh = agent_parse_int_default(agent_tool_arg_value(call, "refresh_sec"),
                                              0, 0, 3600);
        bool stop = !strcmp(call->name, "bash_stop");
        bool wait = stop || refresh > 0;
        if (stop && refresh == 0) refresh = 1;
        return agent_bash_job_tool_result(w, job, wait, refresh, stop, true);
    }

    {
        char header[256];
        snprintf(header, sizeof(header), "\n[tool:%s] unknown tool\n", call->name);
        agent_publish(w, header, strlen(header));
        agent_buf_puts(&result, "Tool error: unknown tool: ");
        agent_buf_puts(&result, call->name);
        agent_buf_puts(&result, "\n");
        return agent_buf_take(&result);
    }
}

/* Post-compaction bash-jobs reminder (Risk 7b): the server no longer carries
 * this, so the client prepends it to the DRAIN_REPLY that follows a
 * successful compaction when jobs are still live. */
static char *agent_bash_jobs_compaction_observation(agent_worker *w) {
    if (!w->bash_jobs) return NULL;
    agent_buf out = {0};
    agent_buf_puts(&out,
        "Bash job update after context compaction. Running jobs still need explicit bash_status or bash_stop if relevant.\n");
    for (agent_bash_job *job = w->bash_jobs, *next = NULL; job; job = next) {
        next = job->next;
        bool done = false;
        char *obs = agent_bash_observation(job, true, &done);
        char hdr[64];
        snprintf(hdr, sizeof(hdr), "\nJob %d:\n", job->id);
        agent_buf_puts(&out, hdr);
        agent_buf_puts(&out, obs);
        free(obs);
        if (done) agent_bash_remove_job(w, job);
    }
    return agent_buf_take(&out);
}

static void client_worker_init_web(agent_worker *w) {
    ds4_web_config cfg = {
        .home_dir = getenv("HOME"),
        .port = 9333,
        .confirm = agent_web_confirm,
        .confirm_privdata = w,
        .log = agent_web_log,
        .log_privdata = w,
        .cancel = agent_web_cancel,
        .cancel_privdata = w,
    };
    w->web = ds4_web_create(&cfg);
}

static void client_worker_free(agent_worker *w) {
    agent_bash_jobs_free(w);
    ds4_web_free(w->web);
    pthread_mutex_destroy(&w->mu);
}

/* -- wire-level tool result (text parts + raw image bytes) -------- */

typedef struct {
    uint8_t *bytes;
    size_t len;
} agent_wire_image;

typedef struct {
    char **parts;
    size_t part_count;
    size_t part_cap;
    agent_wire_image *images;
    size_t image_count;
    size_t image_cap;
} client_tool_result;

static void client_tool_result_init(client_tool_result *r) {
    memset(r, 0, sizeof(*r));
    r->parts = xmalloc(sizeof(r->parts[0]));
    r->parts[0] = xstrdup("");
    r->part_count = 1;
    r->part_cap = 1;
}

static void client_tool_result_puts(client_tool_result *r, const char *text) {
    if (!r->part_count) client_tool_result_init(r);
    if (!text || !text[0]) return;
    size_t idx = r->part_count - 1;
    size_t oldlen = strlen(r->parts[idx]);
    size_t addlen = strlen(text);
    r->parts[idx] = xrealloc(r->parts[idx], oldlen + addlen + 1);
    memcpy(r->parts[idx] + oldlen, text, addlen + 1);
}

/* Takes ownership of bytes. Starts a fresh text part, mirroring
 * agent_tool_observation_add_image's part-per-image shape. */
static void client_tool_result_add_image(client_tool_result *r,
                                         uint8_t *bytes, size_t len) {
    if (!r->part_count) client_tool_result_init(r);
    if (r->image_count == r->image_cap) {
        r->image_cap = r->image_cap ? r->image_cap * 2 : 2;
        r->images = xrealloc(r->images, r->image_cap * sizeof(r->images[0]));
    }
    r->images[r->image_count].bytes = bytes;
    r->images[r->image_count].len = len;
    r->image_count++;
    if (r->part_count == r->part_cap) {
        r->part_cap = r->part_cap ? r->part_cap * 2 : 2;
        r->parts = xrealloc(r->parts, r->part_cap * sizeof(r->parts[0]));
    }
    r->parts[r->part_count++] = xstrdup("");
}

static void client_tool_result_free(client_tool_result *r) {
    for (size_t i = 0; i < r->part_count; i++) free(r->parts[i]);
    free(r->parts);
    for (size_t i = 0; i < r->image_count; i++) free(r->images[i].bytes);
    free(r->images);
    memset(r, 0, sizeof(*r));
}

/* view_image sends the raw bytes; the server encodes them with
 * ds4_engine_vision_encode_memory (no engine on the client). */
static void client_tool_view_image(agent_worker *w, const agent_tool_call *call,
                                   client_tool_result *out) {
    const char *path = agent_tool_arg_value(call, "path");
    if (!path || !path[0]) {
        client_tool_result_puts(out, "Tool error: view_image requires path\n");
        return;
    }
    char *data = NULL;
    size_t len = 0;
    char err[256] = {0};
    if (agent_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        client_tool_result_puts(out, "Tool error: view_image failed: ");
        client_tool_result_puts(out, err[0] ? err : "unable to read image");
        client_tool_result_puts(out, "\n");
        return;
    }
    char shown[PATH_MAX + 80];
    snprintf(shown, sizeof(shown), "\n[tool:view_image] %s\n", path);
    agent_publish(w, shown, strlen(shown));
    client_tool_result_add_image(out, (uint8_t *)data, len);
    char meta[192];
    snprintf(meta, sizeof(meta), "\nImage observation attached (%s, %zu bytes).\n",
             path, len);
    client_tool_result_puts(out, meta);
}

/* Copy one TOOL_CALLS entry (borrowed from the frame payload) into an owned
 * agent_tool_call so the copied ds4_agent.c tool functions run unchanged. */
static void client_tool_call_from_wire(const ap_tool_call *src, agent_tool_call *dst,
                                       char *name_buf, size_t name_cap) {
    snprintf(name_buf, name_cap, "%.*s", (int)src->name_len, src->name);
    dst->name = name_buf;
    dst->argc = (int)src->arg_count;
    dst->argcap = dst->argc;
    dst->args = dst->argc ? xmalloc((size_t)dst->argc * sizeof(dst->args[0])) : NULL;
    for (int i = 0; i < dst->argc; i++) {
        dst->args[i].name = xstrndup(src->args[i].name, src->args[i].name_len);
        dst->args[i].value = xstrndup(src->args[i].value, src->args[i].value_len);
        dst->args[i].is_string = src->args[i].is_string;
    }
}

static void client_tool_call_free_wire(agent_tool_call *c) {
    for (int i = 0; i < c->argc; i++) {
        free(c->args[i].name);
        free(c->args[i].value);
    }
    free(c->args);
}

/* Run every call in one TOOL_CALLS request and build the TOOL_RESULT payload.
 * Mirrors ds4_agent.c's agent_execute_tool_observation. */
static void client_execute_tool_calls(agent_worker *w, const ap_tool_calls *calls,
                                      client_tool_result *out) {
    client_tool_result_init(out);
    for (uint32_t i = 0; i < calls->call_count; i++) {
        char namebuf[128];
        agent_tool_call call = {0};
        client_tool_call_from_wire(&calls->calls[i], &call, namebuf, sizeof(namebuf));

        char hdr[160];
        snprintf(hdr, sizeof(hdr), "Tool result %u (%s):\n", i + 1,
                 call.name[0] ? call.name : "unknown");
        client_tool_result_puts(out, hdr);

        if (!strcmp(call.name, "view_image")) {
            client_tool_view_image(w, &call, out);
        } else {
            char *res = agent_execute_tool_call(w, &call);
            client_tool_result_puts(out, res);
            size_t rl = strlen(res);
            if (rl && res[rl - 1] != '\n') client_tool_result_puts(out, "\n");
            free(res);
        }
        client_tool_call_free_wire(&call);
    }
    if (!calls->call_count)
        client_tool_result_puts(out, "Tool error: empty tool call block\n");
}

static void client_send_tool_result(client_conn *co, uint32_t request_id,
                                    const client_tool_result *r) {
    ap_tool_result tr = {0};
    tr.request_id = request_id;
    for (size_t i = 0; i < r->part_count && tr.text_part_count < AP_MAX_TEXT_PARTS; i++) {
        tr.text_parts[tr.text_part_count].ptr = r->parts[i];
        tr.text_parts[tr.text_part_count].len = strlen(r->parts[i]);
        tr.text_part_count++;
    }
    for (size_t i = 0; i < r->image_count && tr.image_count < AP_MAX_IMAGES; i++) {
        tr.images[tr.image_count].bytes = r->images[i].bytes;
        tr.images[tr.image_count].bytes_len = r->images[i].len;
        tr.images[tr.image_count].source = "";
        tr.images[tr.image_count].source_len = 0;
        tr.image_count++;
    }
    ap_buf b;
    ap_buf_init(&b);
    ap_encode_tool_result(&b, &tr);
    conn_send(co, AGENT_MSG_TOOL_RESULT, &b);
    ap_buf_free(&b);
}

/* Run a TOOL_CALLS request locally and reply with TOOL_RESULT. */
static void client_handle_tool_calls(client_conn *co, agent_worker *w,
                                     const ap_tool_calls *calls) {
    client_tool_result r;
    client_execute_tool_calls(w, calls, &r);
    client_send_tool_result(co, calls->request_id, &r);
    client_tool_result_free(&r);
}

static void client_send_drain_reply(client_conn *co, const char *text) {
    ap_drain_reply d = { .text = text ? text : "", .text_len = text ? strlen(text) : 0 };
    ap_buf b;
    ap_buf_init(&b);
    ap_encode_drain_reply(&b, &d);
    conn_send(co, AGENT_MSG_DRAIN_REPLY, &b);
    ap_buf_free(&b);
}

/* Reply to DRAIN_REQUEST. queued_text is whatever the prompt queue drained (T11
 * fills this in; NULL/empty until then). Per Risk 7b, a live-bash-jobs note is
 * prepended so the model still sees running jobs after a compaction. */
static void client_handle_drain_request(client_conn *co, agent_worker *w,
                                        const char *queued_text) {
    char *bash_note = agent_bash_jobs_compaction_observation(w);
    char *reply = NULL;
    if (bash_note && bash_note[0]) {
        agent_buf b = {0};
        agent_buf_puts(&b, bash_note);
        if (queued_text && queued_text[0]) {
            agent_buf_puts(&b, "\n");
            agent_buf_puts(&b, queued_text);
        }
        reply = agent_buf_take(&b);
    } else if (queued_text && queued_text[0]) {
        reply = xstrdup(queued_text);
    }
    free(bash_note);
    client_send_drain_reply(co, reply);
    free(reply);
}


/* ========================================================================= */
/* T11a: terminal editor + prompt/footer/banner primitives.                   */
/*                                                                            */
/* Copied verbatim from ds4_agent.c -- agent_editor and every editor_* live    */
/* entirely on top of linenoise + raw terminal escapes, with no knowledge of   */
/* the worker or the wire protocol. The welcome banner takes client_config     */
/* instead of agent_config; the save prompts are rewritten in T11c against    */
/* the client's session-state mirror instead of agent_worker_needs_save.       */
/* ========================================================================= */

#define AGENT_INPUT_INITIAL_BUFLEN 4096
#define AGENT_INPUT_MAX_BUFLEN (1024*1024)
#define AGENT_STATUS_STYLE_END "\x1b[0m"
#define AGENT_QUEUE_STYLE "\x1b[38;5;87;1m"
#define AGENT_STATUS_REDRAW_INTERVAL_SEC 0.20

/* linenoise.h omits this exported symbol; mirror ds4_agent.c's own forward
 * declaration for it. */
int linenoiseEditInsert(struct linenoiseState *l, const char *c, size_t clen);

static char *agent_format_user_prompt_echo(const char *text) {
    agent_buf b = {0};
    if (stdout_is_tty()) {
        agent_buf_puts(&b, "\x1b[1;91m*\x1b[1;97m ");
        agent_buf_puts(&b, text);
        agent_buf_puts(&b, "\x1b[0m\n\n");
    } else {
        agent_buf_puts(&b, "* ");
        agent_buf_puts(&b, text);
        agent_buf_puts(&b, "\n\n");
    }
    return agent_buf_take(&b);
}

static void agent_echo_user_prompt(const char *text) {
    char *msg = agent_format_user_prompt_echo(text);
    printf("%s", msg);
    fflush(stdout);
    free(msg);
}

typedef struct {
    char **v;
    size_t len;
    size_t cap;
} agent_prompt_queue;

static void agent_prompt_queue_push(agent_prompt_queue *q, const char *text) {
    if (q->len == q->cap) {
        q->cap = q->cap ? q->cap * 2 : 4;
        q->v = xrealloc(q->v, q->cap * sizeof(q->v[0]));
    }
    q->v[q->len++] = xstrdup(text ? text : "");
}

static char *agent_prompt_queue_pop(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    char *text = q->v[0];
    memmove(q->v, q->v + 1, (q->len - 1) * sizeof(q->v[0]));
    q->len--;
    return text;
}

static void agent_prompt_queue_push_front(agent_prompt_queue *q, char *text) {
    if (q->len == q->cap) {
        q->cap = q->cap ? q->cap * 2 : 4;
        q->v = xrealloc(q->v, q->cap * sizeof(q->v[0]));
    }
    memmove(q->v + 1, q->v, q->len * sizeof(q->v[0]));
    q->v[0] = text;
    q->len++;
}

static char *agent_prompt_queue_take_all(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    if (q->len == 1) return agent_prompt_queue_pop(q);

    agent_buf b = {0};
    for (size_t i = 0; i < q->len; i++) {
        char hdr[64];
        if (i) agent_buf_puts(&b, "\n\n");
        snprintf(hdr, sizeof(hdr), "Queued user message %zu:\n", i + 1);
        agent_buf_puts(&b, hdr);
        agent_buf_puts(&b, q->v[i]);
        free(q->v[i]);
    }
    q->len = 0;
    return agent_buf_take(&b);
}

static char *agent_prompt_queue_take_all_echo(agent_prompt_queue *q) {
    if (!q->len) return NULL;
    agent_buf b = {0};
    for (size_t i = 0; i < q->len; i++) {
        char *echo = agent_format_user_prompt_echo(q->v[i]);
        agent_buf_puts(&b, echo);
        free(echo);
    }
    return agent_buf_take(&b);
}

static const char *agent_prompt_queue_peek(const agent_prompt_queue *q) {
    return q->len ? q->v[0] : NULL;
}

static void agent_prompt_queue_free(agent_prompt_queue *q) {
    for (size_t i = 0; i < q->len; i++) free(q->v[i]);
    free(q->v);
    memset(q, 0, sizeof(*q));
}

static bool agent_footer_is_multiline(const char *status) {
    return status && strchr(status, '\n');
}

/* Build the editable footer.  With queued prompts, the footer becomes multiple
 * rows: a compact queue preview first, then the normal status row. */
static void build_footer_text(const agent_status *st, const agent_prompt_queue *queue,
                              int cols, char *buf, size_t len) {
    char status[512];
    build_status_text(st, status, sizeof(status));
    if (!queue || !queue->len) {
        snprintf(buf, len, "%s", status);
        return;
    }

    const char *queued = agent_prompt_queue_peek(queue);
    if (cols < 1) cols = 1;
    int max_rows = 3;
    size_t budget = (size_t)cols * (size_t)max_rows;
    const char *plain_suffix = " (ctrl+x to edit, ESC to send ASAP)";
    size_t queued_len = strlen(queued), pos = 0, cells = 8;
    size_t reserve = strlen(plain_suffix) + 4;
    size_t preview_bytes = len > 1024 ? len - 1024 : 0;
    agent_buf msg = {0};
    agent_buf_puts(&msg, "queued: ");
    while (pos < queued_len) {
        int width;
        size_t n = linenoiseNextGrapheme(queued + pos, queued_len - pos, &width);
        if (!n) break;
        bool control = (unsigned char)queued[pos] < 32 || queued[pos] == 127;
        if (control) width = 1;
        if (cells + (size_t)width + reserve > budget || msg.len + n > preview_bytes) break;
        agent_buf_append(&msg, control ? " " : queued + pos, control ? 1 : n);
        cells += (size_t)width;
        pos += n;
    }
    if (pos < queued_len) agent_buf_puts(&msg, "...");
    agent_buf_puts(&msg, plain_suffix);
    char *preview = agent_buf_take(&msg);

    agent_buf out = {0};
    pos = 0;
    size_t preview_len = strlen(preview);
    for (int row = 0; row < max_rows && pos < preview_len; row++) {
        if (row) agent_buf_puts(&out, "\n");
        if (stdout_is_tty()) agent_buf_puts(&out, AGENT_QUEUE_STYLE);
        cells = 0;
        while (pos < preview_len) {
            int width;
            size_t n = linenoiseNextGrapheme(preview + pos, preview_len - pos, &width);
            if (!n || cells + (size_t)width > (size_t)cols) break;
            agent_buf_append(&out, preview + pos, n);
            cells += (size_t)width;
            pos += n;
        }
        if (stdout_is_tty()) agent_buf_puts(&out, "\x1b[0m");
    }
    agent_buf_puts(&out, "\n");
    if (stdout_is_tty()) agent_buf_puts(&out, AGENT_STATUS_STYLE_START);
    agent_buf_puts(&out, status);
    snprintf(buf, len, "%s", out.ptr ? out.ptr : "");
    free(preview);
    free(out.ptr);
}

typedef struct {
    struct linenoiseState edit;
    char *input;
    char prompt[160];
    char status[4096];
    bool prompt_dirty, status_dirty;
    int old_stdin_flags;
    bool active;
    bool hidden;
    bool output_line_open;
    bool output_pending_wrap;
    char output_utf8[4];
    size_t output_utf8_len;
    int output_escape;
    bool output_zwj, output_regional;
    int output_glyph_width;
    bool prompt_below_output;
    int output_col;
    bool scroll_region;
    int term_rows;
    int term_cols;
    int output_bottom;
    int prompt_row;
    int reserved_rows;
    bool output_cursor_saved;
    bool output_at_scroll_boundary;
    agent_input_buf deferred_output;
    double last_prompt_redraw_time;
    char cpr_buf[32];
    size_t cpr_len;
    bool paste_open;
    bool paste_start_pending;
    char paste_tail[6];
    size_t paste_tail_len;
} agent_editor;

static void editor_queue_bytes(agent_editor *ed, const char *buf, size_t len);
static void editor_hide(agent_editor *ed);
static void editor_show(agent_editor *ed);
static bool editor_write_scroll_output_preserve_prompt(agent_editor *ed,
                                                       const char *text,
                                                       size_t len,
                                                       bool settle_boundary);

typedef enum {
    CPR_INVALID,
    CPR_PARTIAL,
    CPR_COMPLETE,
} cpr_state;

/* Classify a possible terminal cursor-position reply (ESC[row;colR).  User
 * keystrokes can arrive interleaved with these replies, so we only swallow bytes
 * when they are definitely part of a complete CPR sequence. */
static cpr_state cpr_candidate_state(const char *buf, size_t len) {
    if (len == 0) return CPR_PARTIAL;
    if ((unsigned char)buf[0] != 0x1b) return CPR_INVALID;
    if (len == 1) return CPR_PARTIAL;
    if (buf[1] != '[') return CPR_INVALID;
    if (len == 2) return CPR_PARTIAL;

    size_t p = 2;
    if (buf[p] < '0' || buf[p] > '9') return CPR_INVALID;
    while (p < len && buf[p] >= '0' && buf[p] <= '9') p++;
    if (p == len) return CPR_PARTIAL;
    if (buf[p++] != ';') return CPR_INVALID;
    if (p == len) return CPR_PARTIAL;
    if (buf[p] < '0' || buf[p] > '9') return CPR_INVALID;
    while (p < len && buf[p] >= '0' && buf[p] <= '9') p++;
    if (p == len) return CPR_PARTIAL;
    return p + 1 == len && buf[p] == 'R' ? CPR_COMPLETE : CPR_INVALID;
}

static void editor_flush_cpr_candidate(agent_editor *ed) {
    if (!ed->cpr_len) return;
    linenoiseEditQueueInput(&ed->edit, ed->cpr_buf, ed->cpr_len);
    ed->cpr_len = 0;
}

static bool agent_tail_ends_with(const char *tail, size_t tail_len,
                                 const char *seq, size_t seq_len) {
    return tail_len >= seq_len &&
           memcmp(tail + tail_len - seq_len, seq, seq_len) == 0;
}

static bool agent_tail_has_seq_prefix(const char *tail, size_t tail_len,
                                      const char *seq, size_t seq_len) {
    size_t max = tail_len < seq_len - 1 ? tail_len : seq_len - 1;
    for (size_t n = max; n > 0; n--) {
        if (memcmp(tail + tail_len - n, seq, n) == 0) return true;
    }
    return false;
}

/* Track bracketed paste markers outside linenoise.  The nonblocking event loop
 * may receive a paste in chunks; pausing linenoiseEditFeed() until ESC[201~
 * arrives prevents pasted newlines from being interpreted as Enter. */
static void editor_track_bracketed_paste(agent_editor *ed, char c) {
    static const char start[] = "\x1b[200~";
    static const char end[] = "\x1b[201~";

    if (ed->paste_tail_len == sizeof(ed->paste_tail)) {
        memmove(ed->paste_tail, ed->paste_tail + 1, sizeof(ed->paste_tail) - 1);
        ed->paste_tail_len--;
    }
    ed->paste_tail[ed->paste_tail_len++] = c;

    /* The blocking linenoise() path waits inside linenoiseEditPaste() until it
     * sees ESC[201~. In the agent the outer event loop reads stdin in
     * non-blocking chunks; if we let linenoise start parsing ESC[200~ before
     * the closing marker has arrived, pasted newlines can be interpreted as
     * Enter. Keep feeding bytes into linenoise's queue, but don't call
     * linenoiseEditFeed() while the terminal paste envelope is still open. */
    if (agent_tail_ends_with(ed->paste_tail, ed->paste_tail_len,
                             start, sizeof(start) - 1))
    {
        ed->paste_open = true;
        ed->paste_start_pending = false;
    } else if (agent_tail_ends_with(ed->paste_tail, ed->paste_tail_len,
                                    end, sizeof(end) - 1))
    {
        ed->paste_open = false;
        ed->paste_start_pending = false;
    } else {
        ed->paste_start_pending =
            !ed->paste_open &&
            agent_tail_has_seq_prefix(ed->paste_tail, ed->paste_tail_len,
                                      start, sizeof(start) - 1);
    }
}

/* Separate late CPR replies from real user input before handing bytes to
 * linenoise. */
static void editor_filter_input_byte(agent_editor *ed, char c) {
    if (ed->cpr_len || (unsigned char)c == 0x1b) {
        if (ed->cpr_len == sizeof(ed->cpr_buf)) {
            editor_flush_cpr_candidate(ed);
        }
        ed->cpr_buf[ed->cpr_len++] = c;
        cpr_state st = cpr_candidate_state(ed->cpr_buf, ed->cpr_len);
        if (st == CPR_COMPLETE) {
            ed->cpr_len = 0; /* Late terminal cursor report: discard it. */
        } else if (st == CPR_INVALID) {
            editor_flush_cpr_candidate(ed);
        }
        return;
    }
    linenoiseEditQueueInput(&ed->edit, &c, 1);
}

/* Queue raw terminal bytes into linenoise while preserving paste envelopes and
 * filtering cursor-position replies. */
static void editor_queue_bytes(agent_editor *ed, const char *buf, size_t len) {
    for (size_t i = 0; i < len; i++) {
        editor_track_bracketed_paste(ed, buf[i]);
        editor_filter_input_byte(ed, buf[i]);
    }
}

/* Drain stdin in nonblocking mode.  The outer event loop decides when queued
 * bytes are fed to linenoiseEditFeed(). */
static void editor_read_stdin(agent_editor *ed) {
    char buf[256];
    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0) {
            editor_queue_bytes(ed, buf, (size_t)n);
            continue;
        }
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}

static bool editor_take_queued_byte(agent_editor *ed, unsigned char byte) {
    struct linenoiseState *l = &ed->edit;
    for (size_t i = l->queued_input_pos; i < l->queued_input_len; i++) {
        if ((unsigned char)l->queued_input[i] != byte) continue;
        memmove(l->queued_input + i, l->queued_input + i + 1,
                l->queued_input_len - i - 1);
        l->queued_input_len--;
        if (l->queued_input_pos > l->queued_input_len)
            l->queued_input_pos = l->queued_input_len;
        return true;
    }
    return false;
}

static bool editor_take_bare_escape(agent_editor *ed) {
    if (ed->cpr_len == 1 && (unsigned char)ed->cpr_buf[0] == 0x1b) {
        ed->cpr_len = 0;
        return true;
    }
    return false;
}

static void editor_replace_input(agent_editor *ed, const char *text) {
    if (ed->hidden) editor_show(ed);
    linenoiseEditClear(&ed->edit);
    if (text && text[0]) linenoiseEditInsert(&ed->edit, text, strlen(text));
}

/* Track streamed text with the same widths as linenoise. Partial UTF-8 and
 * escape sequences remain pending across worker-output chunks. */
static void editor_note_output(agent_editor *ed, const char *text, size_t len) {
    int cols = ed->edit.cols > 0 ? (int)ed->edit.cols : 80;
    for (size_t i = 0; i < len; i++) {
        unsigned char c = (unsigned char)text[i];
        if (ed->output_escape) {
            if (ed->output_escape == 1) ed->output_escape = c == '[' ? 2 : 0;
            else if (c >= 0x40 && c <= 0x7e) ed->output_escape = 0;
            continue;
        }
        if (c == 0x1b && !ed->output_utf8_len) { ed->output_escape = 1; continue; }
        uint32_t cp = c;
        if (ed->output_utf8_len || c >= 0x80) {
            if (ed->output_utf8_len && (c & 0xc0) != 0x80) {
                ed->output_utf8_len = 0;
                ed->output_col = (ed->output_col + 1) % cols;
            }
            ed->output_utf8[ed->output_utf8_len++] = (char)c;
            size_t n = linenoiseUtf8Decode(ed->output_utf8, ed->output_utf8_len, &cp);
            if (!n) continue;
            ed->output_utf8_len = 0;
        }
        if (cp == '\n' || cp == '\r') {
            ed->output_col = 0;
            ed->output_pending_wrap = false;
            if (cp == '\n') ed->output_line_open = false;
            ed->output_zwj = ed->output_regional = false;
            ed->output_glyph_width = 0;
            continue;
        }
        if (cp == '\b') {
            if (ed->output_pending_wrap) ed->output_col = cols - 1;
            else if (ed->output_col > 0) ed->output_col--;
            ed->output_pending_wrap = false;
            continue;
        }
        if (cp == '\t') {
            int col = ed->output_pending_wrap ? cols - 1 : ed->output_col;
            ed->output_col = (col | 7) + 1;
            if (ed->output_col >= cols) ed->output_col = cols - 1;
            ed->output_pending_wrap = false;
            ed->output_line_open = true;
            continue;
        }
        int width = linenoiseCharacterWidth(cp);
        bool regional = cp >= 0x1f1e6 && cp <= 0x1f1ff;
        if (cp == 0x200d) {
            ed->output_zwj = true;
            continue;
        }
        if (cp == 0xfe0f && ed->output_glyph_width == 1) {
            width = 1;
            ed->output_glyph_width = 2;
        } else if (width) {
            if (ed->output_zwj || (regional && ed->output_regional)) {
                width = 0;
                ed->output_zwj = false;
                ed->output_regional = false;
            } else {
                ed->output_glyph_width = width;
                ed->output_regional = regional;
            }
        }
        if (width > 0) {
            if (ed->output_col + width > cols) ed->output_col = 0;
            ed->output_col = (ed->output_col + width) % cols;
            ed->output_pending_wrap = ed->output_col == 0;
            ed->output_line_open = true;
        }
    }
}

/* Normalize generated LF to CRLF for terminal output without changing the text
 * stored in the transcript. */
static void editor_write_terminal_text(const char *text, size_t len) {
    size_t start = 0;
    for (size_t i = 0; i < len; i++) {
        if (text[i] != '\n') continue;
        if (i > start) write_all(STDOUT_FILENO, text + start, i - start);
        write_all(STDOUT_FILENO, "\r\n", 2);
        start = i + 1;
    }
    if (start < len) write_all(STDOUT_FILENO, text + start, len - start);
}

/* Locate a CPR reply inside a mixed stdin buffer.  Bytes before/after the reply
 * are user input and must be queued back into linenoise. */
static bool find_cpr_reply(const char *buf, size_t len, size_t *start, size_t *end,
                           int *row, int *col) {
    for (size_t i = 0; i + 5 < len; i++) {
        if ((unsigned char)buf[i] != 0x1b || buf[i + 1] != '[') continue;
        size_t p = i + 2;
        int r = 0, c = 0;
        if (p >= len || buf[p] < '0' || buf[p] > '9') continue;
        while (p < len && buf[p] >= '0' && buf[p] <= '9') {
            r = r * 10 + (buf[p++] - '0');
        }
        if (p >= len || buf[p++] != ';') continue;
        if (p >= len || buf[p] < '0' || buf[p] > '9') continue;
        while (p < len && buf[p] >= '0' && buf[p] <= '9') {
            c = c * 10 + (buf[p++] - '0');
        }
        if (p >= len || buf[p] != 'R') continue;
        *start = i;
        *end = p + 1;
        *row = r;
        *col = c;
        return true;
    }
    return false;
}

/* Ask the terminal for the cursor column after writing model output.  Any user
 * bytes read while waiting for the CPR reply are queued back into linenoise so
 * typing during generation is not lost. */
static bool editor_query_cursor(agent_editor *ed, int *col_out) {
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return false;

    char buf[512];
    size_t len = 0, start = 0, end = 0;
    int row = 0, col = 0;
    write_all(STDOUT_FILENO, "\x1b[6n", 4);

    for (int attempt = 0; attempt < 8; attempt++) {
        struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
        int rc = poll(&pfd, 1, 5);
        if (rc < 0 && errno == EINTR) continue;
        if (rc <= 0) continue;
        for (;;) {
            ssize_t n = read(STDIN_FILENO, buf + len, sizeof(buf) - len);
            if (n > 0) {
                len += (size_t)n;
                if (find_cpr_reply(buf, len, &start, &end, &row, &col)) {
                    if (start) editor_queue_bytes(ed, buf, start);
                    if (end < len) editor_queue_bytes(ed, buf + end, len - end);
                    (void)row;
                    *col_out = col;
                    return col > 0;
                }
                if (len == sizeof(buf)) break;
                continue;
            }
            if (n < 0 && errno == EINTR) continue;
            break;
        }
    }

    if (len) editor_queue_bytes(ed, buf, len);
    return false;
}

static void editor_move_to_output_cursor(agent_editor *ed) {
    char seq[64];
    write_all(STDOUT_FILENO, "\x1b[1A", 4);
    int n = snprintf(seq, sizeof(seq), "\x1b[%dG", ed->output_col + 1);
    if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
}

static bool editor_get_terminal_size(int *rows, int *cols) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) != 0) return false;
    if (ws.ws_row < 1 || ws.ws_col < 1) return false;
    *rows = ws.ws_row;
    *cols = ws.ws_col;
    return true;
}

static void editor_csi_cursor(int row, int col) {
    char seq[64];
    int n = snprintf(seq, sizeof(seq), "\x1b[%d;%dH", row, col);
    if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
}

static void editor_save_output_cursor(agent_editor *ed) {
    if (!ed->scroll_region) return;
    write_all(STDOUT_FILENO, "\0337", 2);
    ed->output_cursor_saved = true;
}

static void editor_restore_output_cursor(agent_editor *ed) {
    if (!ed->scroll_region) return;
    if (ed->output_cursor_saved) {
        write_all(STDOUT_FILENO, "\0338", 2);
    } else {
        editor_csi_cursor(ed->output_bottom, 1);
    }
}

static void editor_move_to_prompt_row(agent_editor *ed) {
    if (!ed->scroll_region) return;
    editor_csi_cursor(ed->prompt_row, 1);
}

static void editor_move_to_prompt_cursor(agent_editor *ed) {
    if (!ed->scroll_region) return;
    if (ed->edit.screen_cursor_row > 0 && ed->edit.screen_cursor_col > 0) {
        editor_csi_cursor(ed->edit.screen_cursor_row, ed->edit.screen_cursor_col);
    } else {
        editor_move_to_prompt_row(ed);
    }
}

static void editor_clear_row(int row) {
    editor_csi_cursor(row, 1);
    write_all(STDOUT_FILENO, "\r\x1b[0K", 5);
}

static void editor_clear_prompt_region(agent_editor *ed) {
    if (!ed->scroll_region) return;
    for (int row = ed->prompt_row; row <= ed->term_rows; row++)
        editor_clear_row(row);

    /* In scroll-region mode ds4-agent owns the absolute prompt/status rows.
     * Clearing them directly is more reliable than asking linenoise to clean
     * relative to whatever cursor position the last worker/status transition
     * left behind.  Reset linenoise's render bookkeeping so the next show is a
     * pure write into the reserved rows. */
    ed->edit.oldrows = 0;
    ed->edit.oldstatusrows = 0;
    ed->edit.oldrpos = 1;
    ed->edit.oldpos = ed->edit.pos;
}

static void editor_set_scroll_margin(int bottom) {
    char seq[96];
    int n = snprintf(seq, sizeof(seq), "\x1b[1;%dr", bottom);
    if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
}

static void editor_scroll_output_up(int bottom, int lines) {
    if (lines <= 0) return;
    editor_set_scroll_margin(bottom);
    editor_csi_cursor(bottom, 1);
    for (int i = 0; i < lines; i++)
        write_all(STDOUT_FILENO, "\n", 1);
}

static bool editor_set_scroll_layout(agent_editor *ed, int reserved_rows,
                                     bool allow_shrink,
                                     bool scroll_on_grow) {
    if (!ed->scroll_region) return false;

    int rows = 0, cols = 0;
    if (!editor_get_terminal_size(&rows, &cols)) return false;
    if (rows < 8 || cols < 20) return false;
    if (reserved_rows < 2) reserved_rows = 2;
    if (reserved_rows > rows - 2) reserved_rows = rows - 2;
    if (!allow_shrink && ed->reserved_rows > 0 &&
        ed->term_rows == rows && ed->term_cols == cols &&
        reserved_rows < ed->reserved_rows)
    {
        reserved_rows = ed->reserved_rows;
    }

    int output_bottom = rows - reserved_rows;
    int prompt_row = output_bottom + 1;
    bool changed = ed->term_rows != rows ||
                   ed->term_cols != cols ||
                   ed->output_bottom != output_bottom ||
                   ed->prompt_row != prompt_row ||
                   ed->reserved_rows != reserved_rows;
    if (!changed) return true;

    /* If the prompt grows, rows that were output rows become prompt rows.  Do
     * not simply clear them: first scroll the old output region upward by the
     * number of newly reserved rows, exactly as if the model had printed more
     * lines.  If the prompt shrinks, no output is restored; the output region
     * simply grows downward and the prompt/status block remains bottom
     * anchored. */
    bool scrolled_output = false;
    if (scroll_on_grow &&
        ed->term_rows == rows && ed->term_cols == cols &&
        ed->output_bottom > 0 && output_bottom < ed->output_bottom)
    {
        editor_scroll_output_up(ed->output_bottom,
                                ed->output_bottom - output_bottom);
        scrolled_output = true;
    }

    editor_set_scroll_margin(output_bottom);

    ed->term_rows = rows;
    ed->term_cols = cols;
    ed->output_bottom = output_bottom;
    ed->prompt_row = prompt_row;
    ed->reserved_rows = reserved_rows;
    ed->output_cursor_saved = false;
    ed->output_at_scroll_boundary = scrolled_output;

    for (int row = prompt_row; row <= rows; row++)
        editor_clear_row(row);

    /* If the prompt grew while generated output was in the middle of a line,
     * the scroll above moved that partial line up with its column intact.
     * Preserve that column when saving the new output cursor; otherwise the
     * next token resumes at column 1 and overwrites the line it was extending. */
    int output_col = ed->output_line_open ? ed->output_col + 1 : 1;
    if (output_col < 1) output_col = 1;
    if (output_col > cols) output_col = cols;
    editor_csi_cursor(output_bottom, output_col);
    editor_save_output_cursor(ed);
    editor_move_to_prompt_row(ed);
    return true;
}

static int editor_linenoise_layout_changed(struct linenoiseState *l,
                                           size_t prompt_rows,
                                           size_t status_rows,
                                           void *privdata) {
    (void)l;
    agent_editor *ed = privdata;
    if (!ed || !ed->scroll_region) return 0;
    if (prompt_rows < 1) prompt_rows = 1;
    int reserved = (int)(prompt_rows + status_rows);
    if (!editor_set_scroll_layout(ed, reserved, true, true)) return 0;
    return ed->prompt_row;
}

/* Keep generated output inside a scroll region that excludes the live prompt
 * and status footer.  This lets terminals scroll model/tool output naturally
 * without rewriting the prompt on every streamed token, which is especially
 * important over SSH where full redraws are visibly expensive. */
static bool editor_configure_scroll_region(agent_editor *ed) {
    if (ed->scroll_region) return true;
    if (!isatty(STDIN_FILENO) || !isatty(STDOUT_FILENO)) return false;

    int rows = 0, cols = 0;
    if (!editor_get_terminal_size(&rows, &cols)) return false;
    if (rows < 8 || cols < 20) return false;

    ed->term_rows = 0;
    ed->term_cols = 0;
    ed->output_bottom = 0;
    ed->prompt_row = 0;
    ed->reserved_rows = 0;
    ed->output_cursor_saved = false;
    ed->output_at_scroll_boundary = false;
    ed->scroll_region = true;
    if (!editor_set_scroll_layout(ed, 2, true, false)) return false;

    /* The agent prints backend startup lines before the editor exists.  Once
     * the scroll region is installed, create an append line at the bottom of
     * that region instead of guessing that the old terminal cursor was already
     * there.  Without this first scroll, the first agent/model output can
     * overwrite the last visible startup line. */
    editor_scroll_output_up(ed->output_bottom, 1);
    ed->output_cursor_saved = false;
    editor_csi_cursor(ed->output_bottom, 1);
    editor_save_output_cursor(ed);
    editor_move_to_prompt_row(ed);
    return true;
}

static void editor_restore_terminal_layout(agent_editor *ed) {
    if (!ed->scroll_region) return;
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    write_all(STDOUT_FILENO, "\x1b[r", 3);
    editor_csi_cursor(ed->term_rows, 1);
    write_all(STDOUT_FILENO, "\r\x1b[0K\r\n", 7);
    ed->scroll_region = false;
    ed->output_cursor_saved = false;
    ed->term_rows = ed->term_cols = 0;
    ed->output_bottom = ed->prompt_row = 0;
    ed->reserved_rows = 0;
    ed->output_at_scroll_boundary = false;
}

/* Start linenoise in nonblocking mode and install the status footer. */
static int editor_start(agent_editor *ed, const char *prompt,
                        const char *status, const char *initial) {
    char *input = xmalloc(AGENT_INPUT_INITIAL_BUFLEN);
    snprintf(ed->prompt, sizeof(ed->prompt), "%s", prompt);
    snprintf(ed->status, sizeof(ed->status), "%s", status ? status : "");
    bool had_scroll_region = ed->scroll_region;
    bool use_scroll_region = editor_configure_scroll_region(ed);
    if (use_scroll_region) {
        if (had_scroll_region)
            editor_set_scroll_layout(ed, 2, true, false);
        editor_move_to_prompt_row(ed);
    }
    if (linenoiseEditStart(&ed->edit, STDIN_FILENO, STDOUT_FILENO,
                           input, AGENT_INPUT_INITIAL_BUFLEN, ed->prompt) != 0)
    {
        editor_restore_terminal_layout(ed);
        free(input);
        return -1;
    }
    bool embedded_status = agent_footer_is_multiline(ed->status);
    const char *status_start = stdout_is_tty() && !embedded_status ?
        AGENT_STATUS_STYLE_START : "";
    const char *status_end = stdout_is_tty() && ed->status[0] ?
        AGENT_STATUS_STYLE_END : "";
    linenoiseEditSetStatus(&ed->edit, ed->status,
                           status_start, status_end);
    linenoiseEditSetLayoutCallback(&ed->edit, editor_linenoise_layout_changed, ed);
    if (isatty(ed->edit.ifd) || getenv("LINENOISE_ASSUME_TTY")) {
        linenoiseHide(&ed->edit);
        linenoiseShow(&ed->edit);
    }
    ed->input = input;
    ed->edit.buflen_max = AGENT_INPUT_MAX_BUFLEN;
    ed->active = true;
    if (set_nonblock(STDIN_FILENO, true, &ed->old_stdin_flags) != 0)
        ed->old_stdin_flags = -1;
    if (initial && initial[0]) linenoiseEditInsert(&ed->edit, initial, strlen(initial));
    ed->hidden = false;
    ed->output_line_open = false;
    ed->prompt_below_output = false;
    ed->output_col = 0;
    ed->output_pending_wrap = false;
    ed->output_utf8_len = 0;
    ed->output_escape = 0;
    ed->output_zwj = ed->output_regional = false;
    ed->output_glyph_width = 0;
    ed->cpr_len = 0;
    ed->paste_open = false;
    ed->paste_start_pending = false;
    ed->paste_tail_len = 0;
    return 0;
}

/* Stop the live editor and restore stdin flags. */
static void editor_stop(agent_editor *ed) {
    if (!ed->active) return;
    if (ed->deferred_output.len)
        editor_write_scroll_output_preserve_prompt(ed, NULL, 0, true);
    /* ds4-agent treats linenoise as a live input widget, not as persistent
     * command scrollback.  Clear it before shutdown so submitting a line and
     * immediately reopening the editor does not leave the accepted
     * prompt+input duplicated above the fresh prompt. */
    if (!ed->hidden && (isatty(ed->edit.ifd) || getenv("LINENOISE_ASSUME_TTY")))
        editor_hide(ed);
    linenoiseEditStop(&ed->edit);
    if (ed->old_stdin_flags >= 0) fcntl(STDIN_FILENO, F_SETFL, ed->old_stdin_flags);
    free(ed->edit.buf);
    ed->input = NULL;
    ed->active = false;
    ed->hidden = false;
    ed->output_line_open = false;
    ed->prompt_below_output = false;
    ed->output_col = 0;
    ed->output_pending_wrap = false;
    ed->output_utf8_len = 0;
    ed->output_escape = 0;
    ed->output_zwj = ed->output_regional = false;
    ed->output_glyph_width = 0;
    ed->cpr_len = 0;
    ed->paste_open = false;
    ed->paste_start_pending = false;
    ed->paste_tail_len = 0;
}

/* Hide the live prompt before model output is written.  In scroll-region mode
 * the output cursor was saved before the prompt was drawn, so restoring it is
 * enough to append more model/tool bytes without touching the prompt rows. */
static void editor_hide(agent_editor *ed) {
    if (!ed->active || ed->hidden) return;
    if (ed->scroll_region) {
        editor_clear_prompt_region(ed);
        editor_restore_output_cursor(ed);
        ed->hidden = true;
        return;
    }
    linenoiseHide(&ed->edit);
    if (ed->prompt_below_output) {
        editor_move_to_output_cursor(ed);
        ed->prompt_below_output = false;
    }
    ed->hidden = true;
}

/* Restore the live prompt after output.  The primary path draws it in the
 * reserved bottom rows; the fallback path keeps the older one-row-below-output
 * trick for terminals where scroll regions are unavailable. */
static void editor_show(agent_editor *ed) {
    if (!ed->active || !ed->hidden) return;
    if (ed->scroll_region) {
        editor_save_output_cursor(ed);
        editor_move_to_prompt_row(ed);
        write_all(STDOUT_FILENO, "\x1b[0m", 4);
        linenoiseShow(&ed->edit);
        ed->hidden = false;
        return;
    }
    if (ed->output_line_open) {
        write_all(STDOUT_FILENO, "\r\n", 2);
        ed->prompt_below_output = true;
    } else {
        ed->prompt_below_output = false;
    }
    /* Model/tool output can leave SGR attributes active while it streams.
     * Redrawing linenoise always starts from normal attributes; tool rendering
     * re-emits its own color on the next streamed byte if it is still inside a
     * colored parameter. */
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    linenoiseShow(&ed->edit);
    ed->prompt_dirty = ed->status_dirty = false;
    ed->last_prompt_redraw_time = now_sec();
    ed->hidden = false;
}

static void editor_update_prompt(agent_editor *ed, const char *prompt) {
    if (strcmp(ed->prompt, prompt)) ed->prompt_dirty = true;
    snprintf(ed->prompt, sizeof(ed->prompt), "%s", prompt);
    ed->edit.prompt = ed->prompt;
    ed->edit.plen = strlen(ed->prompt);
}

static void editor_update_status(agent_editor *ed, const char *status) {
    if (strcmp(ed->status, status ? status : "")) ed->status_dirty = true;
    snprintf(ed->status, sizeof(ed->status), "%s", status ? status : "");
    bool embedded_status = agent_footer_is_multiline(ed->status);
    const char *status_start = stdout_is_tty() && !embedded_status ?
        AGENT_STATUS_STYLE_START : "";
    const char *status_end = stdout_is_tty() && ed->status[0] ?
        AGENT_STATUS_STYLE_END : "";
    linenoiseEditSetStatus(&ed->edit, ed->status,
                           status_start, status_end);
}

static void editor_redraw_visible_prompt(agent_editor *ed) {
    if (!ed->active || !ed->scroll_region) return;
    linenoiseBeginUpdate(STDOUT_FILENO);
    editor_clear_prompt_region(ed);
    editor_move_to_prompt_row(ed);
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    linenoiseShow(&ed->edit);
    ed->prompt_dirty = ed->status_dirty = false;
    ed->last_prompt_redraw_time = now_sec();
    linenoiseEndUpdate();
}

static bool editor_prompt_redraw_due(agent_editor *ed) {
    double now = now_sec();
    if (ed->last_prompt_redraw_time <= 0.0 ||
        now - ed->last_prompt_redraw_time >= AGENT_STATUS_REDRAW_INTERVAL_SEC)
    {
        return true;
    }
    return false;
}

static void editor_flush_prompt_status(agent_editor *ed, bool force) {
    if (!ed->active || ed->hidden) return;
    int rows, cols;
    if (ed->scroll_region && editor_get_terminal_size(&rows, &cols) &&
        (rows != ed->term_rows || cols != ed->term_cols)) {
        ed->edit.cols = cols;
        ed->prompt_dirty = true;
        force = true;
    }
    if (!ed->prompt_dirty && !ed->status_dirty) return;
    if (!force && !editor_prompt_redraw_due(ed)) return;
    linenoiseBeginUpdate(STDOUT_FILENO);
    if (ed->scroll_region) {
        if (ed->prompt_dirty || !linenoiseRefreshStatus(&ed->edit))
            editor_redraw_visible_prompt(ed);
    } else {
        editor_hide(ed);
        editor_show(ed);
    }
    ed->prompt_dirty = ed->status_dirty = false;
    ed->last_prompt_redraw_time = now_sec();
    linenoiseEndUpdate();
}

static void editor_set_prompt_status(agent_editor *ed, const char *prompt,
                                     const char *status) {
    if (strcmp(ed->prompt, prompt)) editor_update_prompt(ed, prompt);
    if (strcmp(ed->status, status ? status : "")) editor_update_status(ed, status);
    editor_flush_prompt_status(ed, false);
}

static bool editor_write_scroll_output_preserve_prompt(agent_editor *ed,
                                                       const char *text,
                                                       size_t len,
                                                       bool settle_boundary) {
    agent_input_buf_append(&ed->deferred_output, text, len);
    if (!ed->deferred_output.len) return false;

    agent_editor predicted = *ed;
    editor_note_output(&predicted, ed->deferred_output.ptr,
                       ed->deferred_output.len);
    if ((predicted.output_utf8_len || predicted.output_escape) && settle_boundary) {
        if (predicted.output_utf8_len) {
            ed->deferred_output.len -= predicted.output_utf8_len;
            agent_input_buf_append(&ed->deferred_output, "\xef\xbf\xbd", 3);
        } else {
            while (ed->deferred_output.len &&
                   ed->deferred_output.ptr[--ed->deferred_output.len] != 0x1b) {}
        }
        predicted = *ed;
        editor_note_output(&predicted, ed->deferred_output.ptr, ed->deferred_output.len);
    }
    bool at_boundary = predicted.output_pending_wrap;
    if (predicted.output_utf8_len || predicted.output_escape) return false;
    /* DECSC/DECRC does not reliably preserve pending auto-wrap.  Keep a chunk
     * that ends at the margin until the next content can trigger the wrap in
     * the same terminal write. */
    if (at_boundary && !settle_boundary) return false;

    linenoiseBeginUpdate(STDOUT_FILENO);
    editor_restore_output_cursor(ed);
    editor_write_terminal_text(ed->deferred_output.ptr,
                               ed->deferred_output.len);
    editor_note_output(ed, ed->deferred_output.ptr,
                       ed->deferred_output.len);
    if (at_boundary) {
        /* ESC 7/8 does not preserve a terminal's pending auto-wrap flag.  At
         * the end of a turn, settle it explicitly before saving the cursor. */
        write_all(STDOUT_FILENO, "\r\n", 2);
        ed->output_col = 0;
        ed->output_line_open = false;
        ed->output_pending_wrap = false;
    }
    agent_input_buf_free(&ed->deferred_output);
    editor_save_output_cursor(ed);
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    editor_move_to_prompt_cursor(ed);
    linenoiseEndUpdate();
    ed->output_at_scroll_boundary = true;
    return true;
}

#ifdef DS4_AGENT_TEST
static void test_agent_terminal_wrap_output_is_deferred(void) {
    FILE *sink = tmpfile();
    AGENT_TEST_ASSERT(sink != NULL);
    if (!sink) return;
    int saved_stdout = dup(STDOUT_FILENO);
    AGENT_TEST_ASSERT(saved_stdout >= 0);
    if (saved_stdout < 0) {
        fclose(sink);
        return;
    }
    AGENT_TEST_ASSERT(dup2(fileno(sink), STDOUT_FILENO) >= 0);

    agent_editor ed = {
        .active = true,
        .scroll_region = true,
        .output_bottom = 22,
        .prompt_row = 23,
        .output_cursor_saved = true,
    };
    ed.edit.cols = 80;
    char line[80];
    memset(line, 'x', sizeof(line));

    AGENT_TEST_ASSERT(!editor_write_scroll_output_preserve_prompt(
        &ed, line, sizeof(line), false));
    AGENT_TEST_ASSERT(ed.deferred_output.len == sizeof(line));
    AGENT_TEST_ASSERT(ed.output_col == 0);
    AGENT_TEST_ASSERT(!ed.output_line_open);

    AGENT_TEST_ASSERT(editor_write_scroll_output_preserve_prompt(
        &ed, "y", 1, false));
    AGENT_TEST_ASSERT(ed.deferred_output.len == 0);
    AGENT_TEST_ASSERT(ed.output_col == 1);
    AGENT_TEST_ASSERT(ed.output_line_open);

    ed.output_col = 0;
    ed.output_line_open = false;
    AGENT_TEST_ASSERT(editor_write_scroll_output_preserve_prompt(
        &ed, line, sizeof(line), true));
    AGENT_TEST_ASSERT(ed.deferred_output.len == 0);
    AGENT_TEST_ASSERT(ed.output_col == 0);
    AGENT_TEST_ASSERT(!ed.output_line_open);

    AGENT_TEST_ASSERT(!editor_write_scroll_output_preserve_prompt(&ed, "\xe4", 1, false));
    AGENT_TEST_ASSERT(editor_write_scroll_output_preserve_prompt(&ed, NULL, 0, true));
    AGENT_TEST_ASSERT(ed.deferred_output.len == 0 && ed.output_utf8_len == 0);
    AGENT_TEST_ASSERT(dup2(saved_stdout, STDOUT_FILENO) >= 0);
    close(saved_stdout);
    fclose(sink);
}
#endif

/* Serialize async model/tool output with linenoise.  This is the central
 * terminal contract.  In scroll-region mode the live prompt stays painted:
 * output is appended in the upper scroll area, then the cursor is returned to
 * linenoise's remembered prompt position.  The fallback path still hides and
 * redraws because it has no protected prompt rows. */
static void editor_write_async(agent_editor *ed, const char *text, size_t len,
                               const char *prompt, const char *status,
                               bool force_show) {
    if (ed->scroll_region && ed->active && !ed->hidden &&
        (len || ed->deferred_output.len)) {
        editor_flush_prompt_status(ed, false);
        bool prompt_changed = strcmp(ed->prompt, prompt) != 0;
        bool status_changed = strcmp(ed->status, status ? status : "") != 0;

        linenoiseBeginUpdate(STDOUT_FILENO);
        editor_write_scroll_output_preserve_prompt(ed, text, len, force_show);
        if (prompt_changed) editor_update_prompt(ed, prompt);
        if (status_changed) editor_update_status(ed, status);
        editor_flush_prompt_status(ed, force_show);
        linenoiseEndUpdate();
        return;
    }

    editor_hide(ed);
    if (len) {
        editor_write_terminal_text(text, len);
        if (ed->scroll_region) ed->output_at_scroll_boundary = true;
        if (!ed->scroll_region) {
            if (text[len - 1] == '\n' || text[len - 1] == '\r') {
                ed->output_col = 0;
                ed->output_line_open = false;
            } else {
                int col = 0;
                if (editor_query_cursor(ed, &col)) {
                    int cols = ed->edit.cols > 0 ? (int)ed->edit.cols : 80;
                    ed->output_col = col > 0 ? col - 1 : 0;
                    ed->output_line_open = true;
                    if (ed->output_col + 1 >= cols) {
                        write_all(STDOUT_FILENO, "\r\n", 2);
                        ed->output_col = 0;
                    }
                } else {
                    editor_note_output(ed, text, len);
                }
            }
        }
    }
    if (ed->active) {
        editor_update_prompt(ed, prompt);
        editor_update_status(ed, status);
        /* In scroll-region mode this saves the current output cursor and
         * redraws linenoise in the fixed prompt rows.  In fallback mode it may
         * put the prompt below an unfinished generated line. */
        if (force_show || len) editor_show(ed);
    }
}

/* Ctrl+C while idle is an edit-cancel key, not an exit key.  Clear the real
 * linenoise buffer so stale text cannot be submitted later, then leave a short
 * visible hint about the explicit EOF exit path. */
static void editor_cancel_input_with_hint(agent_editor *ed,
                                          const char *prompt,
                                          const char *status) {
    if (!ed->active) return;
    if (ed->hidden) editor_show(ed);
    linenoiseEditClear(&ed->edit);
    const char *msg = stdout_is_tty() ?
        "\x1b[1;33mpress Ctrl+D to exit\x1b[0m\n" :
        "press Ctrl+D to exit\n";
    editor_write_async(ed, msg, strlen(msg), prompt, status, true);
}


static void runtime_help(void) {
    puts("Commands:");
    puts("  /help        Show this help.");
    puts("  /save        Save the current session.");
    puts("  /compact     Compact the current session context now.");
    puts("  /list        List saved sessions.");
    puts("  /switch SHA  Load a saved session and show recent history.");
    puts("  /del SHA     Delete a saved session.");
    puts("  /strip SHA   Strip KV payload; /switch rebuilds it by prefill.");
    puts("  /history [N] Show N recent user turns from the current session.");
    puts("  /power N     Set GPU duty cycle percentage, 1..100.");
    puts("  /hints on|off Enable or disable brief programming hints; starts off.");
    puts("  /steer [F]   Show or set FFN steering for subsequent tokens.");
    puts("  /new         Start a fresh session from the system prompt.");
    puts("  /quit, /exit Exit.");
    puts("  Ctrl+C       Interrupt generation; clear edited text.");
    puts("  Enter        Queue text while the agent is busy.");
    puts("  Ctrl+X       Edit the first queued prompt.");
    puts("  ESC          Interrupt and send queued prompt immediately.");
    puts("  Ctrl+D       Exit from an empty prompt.");
}

/* Welcome banner: ctx_size comes from the SESSION new/resume reply, not a
 * local --ctx flag. */
static void agent_format_welcome_banner(int ctx_size, char *buf, size_t len) {
    char ctx[32];
    agent_format_ctx_size(ctx_size, ctx, sizeof(ctx));
    if (stdout_is_tty()) {
        snprintf(buf, len,
                 "\x1b[1;97mDwarf\x1b[1;94mStar\x1b[0m \xf0\x9f\x90\x8b Agent, context %s tokens\n\n",
                 ctx);
    } else {
        snprintf(buf, len, "DwarfStar Agent, context %s tokens\n\n", ctx);
    }
}

static void editor_write_welcome_banner(agent_editor *editor, int ctx_size,
                                        const char *prompt,
                                        const char *statusline) {
    char banner[256];
    agent_format_welcome_banner(ctx_size, banner, sizeof(banner));
    editor_write_async(editor, banner, strlen(banner), prompt, statusline, true);
}

typedef enum {
    AGENT_YES_NO_AUTO_NONE,
    AGENT_YES_NO_AUTO_NO,
    AGENT_YES_NO_AUTO_YES,
} agent_yes_no_auto;

typedef struct {
    int timeout_sec;
    agent_yes_no_auto timeout_answer;
} agent_yes_no_options;

static const char *agent_yes_no_auto_name(agent_yes_no_auto answer) {
    switch (answer) {
    case AGENT_YES_NO_AUTO_NO: return "no";
    case AGENT_YES_NO_AUTO_YES: return "yes";
    default: return "";
    }
}

/* Shared y/n prompt. By default it blocks forever like the historical helper;
 * callers that cannot safely stall the agent can request an automatic answer
 * after timeout_sec seconds. */
static bool agent_prompt_yes_no_ex(const char *prompt,
                                   const agent_yes_no_options *opts,
                                   bool *timed_out) {
    char buf[32];
    int timeout_sec = opts ? opts->timeout_sec : 0;
    agent_yes_no_auto auto_answer = opts ?
        opts->timeout_answer : AGENT_YES_NO_AUTO_NONE;
    bool use_timeout = timeout_sec > 0 && auto_answer != AGENT_YES_NO_AUTO_NONE;
    double deadline = use_timeout ? now_sec() + timeout_sec : 0.0;

    if (timed_out) *timed_out = false;
    for (;;) {
        printf("%s", prompt);
        if (use_timeout) {
            int rem = (int)(deadline - now_sec() + 0.999);
            if (rem < 0) rem = 0;
            printf("[auto-%s in %ds] ", agent_yes_no_auto_name(auto_answer), rem);
        }
        fflush(stdout);
        if (use_timeout) {
            double rem_sec = deadline - now_sec();
            if (rem_sec <= 0.0) {
                if (timed_out) *timed_out = true;
                printf("\n");
                return auto_answer == AGENT_YES_NO_AUTO_YES;
            }
            struct pollfd pfd = {.fd = STDIN_FILENO, .events = POLLIN};
            int timeout_ms = (int)(rem_sec * 1000.0) + 1;
            int rc;
            do {
                rc = poll(&pfd, 1, timeout_ms);
            } while (rc < 0 && errno == EINTR);
            if (rc == 0) {
                if (timed_out) *timed_out = true;
                printf("\n");
                return auto_answer == AGENT_YES_NO_AUTO_YES;
            }
            if (rc < 0) return false;
        }
        /* stdin may be in non-blocking mode (set by editor_start).
         * Temporarily switch to blocking so fgets can wait for input. */
        int saved_flags = fcntl(STDIN_FILENO, F_GETFL, 0);
        if (saved_flags >= 0 && (saved_flags & O_NONBLOCK)) {
            fcntl(STDIN_FILENO, F_SETFL, saved_flags & ~O_NONBLOCK);
        }
        bool got_line = fgets(buf, sizeof(buf), stdin) != NULL;
        if (saved_flags >= 0 && (saved_flags & O_NONBLOCK)) {
            fcntl(STDIN_FILENO, F_SETFL, saved_flags);
        }
        if (!got_line) return false;
        char *p = buf;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == 'y' || *p == 'Y') return true;
        if (*p == 'n' || *p == 'N') return false;
    }
}

static bool agent_prompt_yes_no(const char *prompt) {
    return agent_prompt_yes_no_ex(prompt, NULL, NULL);
}

static void agent_noninteractive_marker(const char *msg) {
    write_all(STDERR_FILENO, msg, strlen(msg));
    write_all(STDERR_FILENO, "\n", 1);
}

static int agent_read_stdin_available(agent_input_buf *in, bool *eof) {
    char buf[4096];
    for (;;) {
        ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
        if (n > 0) {
            agent_input_buf_append(in, buf, (size_t)n);
            continue;
        }
        if (n == 0) {
            *eof = true;
            return 0;
        }
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return 0;
        perror("ds4-agent-client: read stdin");
        return -1;
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
