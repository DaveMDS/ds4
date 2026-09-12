/* ds4_agent_client.c -- client binary for the split agent: terminal/linenoise
 * editor + tool execution, talking to ds4-agent-server over TCP.
 *
 * Mirrors the monolithic worker/UI split: a worker thread owns the socket and
 * tool execution, the UI thread owns the editor.  The client is a "dumb
 * painter": it renders tokens by kind (markdown prose, grey think, colored tool
 * names/params) and never parses DSML/GLM or builds prompts -- the server is
 * the sole authority on the transcript.
 *
 * Layer 2 mock-server harness lives under DS4_AGENT_TEST: the test acts as the
 * server over a socketpair, feeds scripted TOKEN/TURN_PAUSED/STATUS, and verifies
 * rendered output, messages sent, tool execution + fit-context, and session
 * command rendering.
 */

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <netdb.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "ds4_agent_tools.h"
#include "ds4_web.h"
#include "linenoise.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define CLIENT_DEFAULT_HOST "127.0.0.1"
#define CLIENT_DEFAULT_PORT 9334

#define CLIENT_STATUS_STYLE_START "\x1b[48;5;238;38;5;252m"
#define CLIENT_STATUS_STYLE_END "\x1b[0m"
#define CLIENT_STATUS_BAR_FILL "\x1b[48;5;238;38;5;201;1m"
#define CLIENT_PROGRESS_BAR_WIDTH 32

/* ============================================================================
 * Configuration / CLI
 * ========================================================================== */

typedef struct {
    const char *host;
    int port;
    int ctx_size;
    const char *sys_extra;
    uint64_t n_predict;
    float temperature;
    float top_p;
    float min_p;
    uint64_t think_mode;
    uint64_t seed;
    uint64_t power;
    const char *trace_path;
    bool web;
    const char *initial_prompt;
    bool non_interactive;
} client_config;

static void client_usage(FILE *fp) {
    fprintf(fp,
        "usage: ds4-agent-client [options]\n"
        "\n"
        "  --server HOST        ds4-agent-server address (default %s)\n"
        "  --port PORT          ds4-agent-server port (default %d)\n"
        "  -c, --ctx N          context size for tool fit-context (default 100000)\n"
        "  -n, --tokens N       max generated tokens (default 50000)\n"
        "  --temp F             sampling temperature\n"
        "  --top-p F            top-p sampling\n"
        "  --min-p F            min-p sampling\n"
        "  --seed N             sampling seed\n"
        "  --think | --think-max | --nothink   thinking mode\n"
        "  --power N            GPU duty cycle 1..100\n"
        "  -sys, --system TEXT  extra system prompt text\n"
        "  --web                allow browser-based tools (google_search/visit_page)\n"
        "  --trace FILE         write a token trace\n"
        "  -p, --prompt TEXT    initial prompt to send\n"
        "  --non-interactive    read prompts from stdin, no editor\n"
        "  -h, --help           show this help\n",
        CLIENT_DEFAULT_HOST, CLIENT_DEFAULT_PORT);
}

static float parse_float_range_cli(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (!s[0] || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-agent-client: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static long parse_int_cli(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-agent-client: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static uint64_t parse_u64_cli(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (!s[0] || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4-agent-client: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

static const char *need_arg_cli(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-agent-client: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static client_config client_parse_options(int argc, char **argv) {
    client_config c = {
        .host = CLIENT_DEFAULT_HOST,
        .port = CLIENT_DEFAULT_PORT,
        .ctx_size = 100000,
        .n_predict = 50000,
        .temperature = 0.0f,
        .top_p = 1.0f,
        .min_p = 0.0f,
        .think_mode = 1,
        .seed = 0,
        .power = 0,
    };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            client_usage(stdout);
            exit(0);
        } else if (!strcmp(arg, "--server")) {
            c.host = need_arg_cli(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = (int)parse_int_cli(need_arg_cli(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.ctx_size = (int)parse_int_cli(need_arg_cli(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.n_predict = (uint64_t)parse_int_cli(need_arg_cli(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.temperature = parse_float_range_cli(need_arg_cli(&i, argc, argv, arg), arg, 0.0f, 100.0f);
        } else if (!strcmp(arg, "--top-p")) {
            c.top_p = parse_float_range_cli(need_arg_cli(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--min-p")) {
            c.min_p = parse_float_range_cli(need_arg_cli(&i, argc, argv, arg), arg, 0.0f, 1.0f);
        } else if (!strcmp(arg, "--seed")) {
            c.seed = parse_u64_cli(need_arg_cli(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.think_mode = 1;
        } else if (!strcmp(arg, "--think-max")) {
            c.think_mode = 2;
        } else if (!strcmp(arg, "--nothink")) {
            c.think_mode = 0;
        } else if (!strcmp(arg, "--power")) {
            int v = (int)parse_int_cli(need_arg_cli(&i, argc, argv, arg), arg);
            if (v < 1 || v > 100) {
                fprintf(stderr, "ds4-agent-client: --power must be between 1 and 100\n");
                exit(2);
            }
            c.power = (uint64_t)v;
        } else if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.sys_extra = need_arg_cli(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--web")) {
            c.web = true;
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg_cli(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-p") || !strcmp(arg, "--prompt")) {
            c.initial_prompt = need_arg_cli(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--non-interactive")) {
            c.non_interactive = true;
        } else {
            fprintf(stderr, "ds4-agent-client: unknown option: %s\n", arg);
            exit(2);
        }
    }
    return c;
}

/* ============================================================================
 * Socket / framing
 * ========================================================================== */

static int client_connect(const client_config *cfg, char *err, size_t err_len) {
    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    struct addrinfo *res = NULL;
    char port[32];
    snprintf(port, sizeof(port), "%d", cfg->port);
    int rc = getaddrinfo(cfg->host, port, &hints, &res);
    if (rc != 0 || !res) {
        snprintf(err, err_len, "resolve %s:%d failed: %s", cfg->host, cfg->port,
                 rc == 0 ? "no address" : gai_strerror(rc));
        return -1;
    }
    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        if (connect(fd, ai->ai_addr, (socklen_t)ai->ai_addrlen) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);
    if (fd < 0) {
        snprintf(err, err_len, "connect to %s:%d failed: %s", cfg->host, cfg->port,
                 strerror(errno));
        return -1;
    }
    return fd;
}

static bool client_send_all(int fd, const void *buf, size_t len) {
    const unsigned char *p = buf;
    size_t n = 0;
    while (n < len) {
        ssize_t w = write(fd, p + n, len - n);
        if (w < 0 && errno == EINTR) continue;
        if (w <= 0) return false;
        n += (size_t)w;
    }
    return true;
}

static bool client_read_full(int fd, unsigned char *p, size_t len) {
    size_t n = 0;
    while (n < len) {
        ssize_t r = read(fd, p + n, len - n);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        n += (size_t)r;
    }
    return true;
}

static bool client_read_frame(int fd, unsigned char **buf_out, size_t *len_out) {
    unsigned char h[4];
    if (!client_read_full(fd, h, 4)) return false;
    uint32_t plen = ((uint32_t)h[0] << 24) | ((uint32_t)h[1] << 16) |
                    ((uint32_t)h[2] << 8) | (uint32_t)h[3];
    if (plen > (1u << 26)) return false;
    unsigned char *buf = xmalloc((size_t)plen + 4);
    memcpy(buf, h, 4);
    if (!client_read_full(fd, buf + 4, plen)) {
        free(buf);
        return false;
    }
    *buf_out = buf;
    *len_out = (size_t)plen + 4;
    return true;
}

/* ============================================================================
 * Dumb-painter renderer (prose markdown + think + tool kinds)
 * ========================================================================== */

typedef enum {
    CLIENT_MD_PENDING_NONE,
    CLIENT_MD_PENDING_BACKTICK,
} client_md_pending;

typedef struct {
    bool format_markdown;
    bool in_think;
    bool color_open;
    bool use_color;
    bool last_output_newline;
    bool wrote_visible_output;
    bool md_bold;
    bool md_italic;
    bool md_inline_code;
    bool md_line_started;
    char md_inline[4096];
    size_t md_inline_len;
    bool md_escape;
    bool md_code_block;
    bool md_fence_info;
    client_md_pending md_pending;
    size_t md_pending_len;
    char md_fence_lang[32];
    size_t md_fence_lang_len;
    char pending[16];
    size_t pending_len;
    char utf8_pending[4];
    size_t utf8_pending_len;
    size_t utf8_pending_need;
    agent_buf *out;
} client_renderer;

static void renderer_write(client_renderer *r, const char *s, size_t n) {
    if (r->out) agent_buf_append(r->out, s, n);
}

static void renderer_puts(client_renderer *r, const char *s) {
    renderer_write(r, s, strlen(s));
}

static void renderer_set_grey(client_renderer *r) {
    if (r->use_color) renderer_puts(r, "\x1b[38;5;245m");
}

static void renderer_reset_color(client_renderer *r) {
    if (r->use_color) renderer_puts(r, "\x1b[0m");
    r->color_open = false;
}

static size_t renderer_utf8_need(unsigned char c) {
    if (c < 0x80) return 1;
    if (c >= 0xc2 && c <= 0xdf) return 2;
    if (c >= 0xe0 && c <= 0xef) return 3;
    if (c >= 0xf0 && c <= 0xf4) return 4;
    return 1;
}

static bool renderer_has_text_attrs(client_renderer *r) {
    return r->in_think || r->md_bold || r->md_italic ||
           r->md_inline_code || r->md_code_block;
}

static void renderer_set_text_attrs(client_renderer *r) {
    if (!r->use_color) return;
    if (r->in_think) { renderer_set_grey(r); return; }
    if (r->md_code_block) { renderer_puts(r, "\x1b[38;5;75m"); return; }
    if (r->md_inline_code) renderer_puts(r, "\x1b[36m");
    if (r->md_bold) renderer_puts(r, "\x1b[1m");
    if (r->md_italic) renderer_puts(r, "\x1b[3m");
}

static void renderer_write_complete_char_raw(client_renderer *r, const char *s, size_t n) {
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

static void renderer_flush_utf8(client_renderer *r) {
    if (!r->utf8_pending_len) return;
    renderer_write_complete_char_raw(r, r->utf8_pending, r->utf8_pending_len);
    r->utf8_pending_len = 0;
    r->utf8_pending_need = 0;
}

static void renderer_write_char_raw(client_renderer *r, char c) {
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

static void renderer_write_plain_byte(client_renderer *r, char c) {
    bool old_bold = r->md_bold;
    bool old_italic = r->md_italic;
    bool old_inline_code = r->md_inline_code;
    bool old_code_block = r->md_code_block;
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

static void renderer_markdown_clear_pending(client_renderer *r) {
    r->md_pending_len = 0;
    r->md_pending = CLIENT_MD_PENDING_NONE;
}

static void renderer_code_begin(client_renderer *r) {
    r->md_code_block = true;
}

static void renderer_code_end(client_renderer *r) {
    renderer_puts(r, "\n");
    r->md_code_block = false;
}

static void renderer_code_emit_backtick_literals(client_renderer *r, size_t count) {
    for (size_t i = 0; i < count; i++) renderer_write_plain_byte(r, '`');
}

static bool renderer_markdown_inline_pair(client_renderer *r, size_t *delimiter) {
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

static void renderer_markdown_inline_flush(client_renderer *r, bool format) {
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

static void renderer_markdown_emit_pending_literals(client_renderer *r) {
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

static void renderer_markdown_commit_backticks(client_renderer *r) {
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

static void renderer_markdown_feed(client_renderer *r, char c) {
    if (r->md_fence_info) {
        if (c == '\n') {
            r->md_fence_lang[r->md_fence_lang_len] = '\0';
            renderer_write_plain_byte(r, '\n');
            r->md_fence_info = false;
        } else if (r->md_code_block) {
            if (r->md_fence_lang_len + 1 < sizeof(r->md_fence_lang) &&
                (isalnum((unsigned char)c) || c == '_' || c == '-' || c == '+' || c == '#'))
                r->md_fence_lang[r->md_fence_lang_len++] = c;
            renderer_write_plain_byte(r, c);
        }
        return;
    }
    if (r->md_pending == CLIENT_MD_PENDING_BACKTICK) {
        if (c == '`') { r->md_pending_len++; return; }
        renderer_markdown_commit_backticks(r);
        renderer_markdown_feed(r, c);
        return;
    }
    if (r->md_code_block) {
        if (c == '`' && r->md_line_started) {
            r->md_pending = CLIENT_MD_PENDING_BACKTICK;
            r->md_pending_len = 1;
        } else {
            renderer_write_plain_byte(r, c);
        }
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

static void renderer_markdown_finish(client_renderer *r) {
    renderer_markdown_inline_flush(r, true);
    if (r->md_pending == CLIENT_MD_PENDING_BACKTICK && r->md_pending_len >= 3)
        renderer_markdown_commit_backticks(r);
    else
        renderer_markdown_emit_pending_literals(r);
    r->md_bold = false;
    r->md_italic = false;
    r->md_inline_code = false;
    r->md_code_block = false;
    r->md_fence_info = false;
    r->md_line_started = false;
    r->md_fence_lang_len = 0;
    r->md_fence_lang[0] = '\0';
    renderer_reset_color(r);
}

static void renderer_write_char(client_renderer *r, char c) {
    if (!r->format_markdown || r->in_think) {
        renderer_markdown_emit_pending_literals(r);
        renderer_write_char_raw(r, c);
        return;
    }
    r->md_line_started = c != '\n';
    renderer_markdown_feed(r, c);
}

static void renderer_finish(client_renderer *r) {
    renderer_markdown_finish(r);
}

/* Paint one token by kind (server decides the kind). */
static void client_render_token(client_renderer *r, unsigned char kind, const char *text) {
    if (!text || !text[0]) return;
    switch (kind) {
    case PROTO_TOKEN_NORMAL:
        for (const char *p = text; *p; p++) renderer_write_char(r, *p);
        break;
    case PROTO_TOKEN_THINK:
        renderer_set_grey(r);
        for (const char *p = text; *p; p++) renderer_write_char_raw(r, *p);
        renderer_reset_color(r);
        break;
    case PROTO_TOKEN_TOOL_NAME:
        if (!r->last_output_newline) renderer_puts(r, "\n");
        if (r->use_color) renderer_puts(r, "\x1b[1;38;5;39m");
        renderer_puts(r, text);
        if (r->use_color) renderer_puts(r, "\x1b[0m");
        break;
    case PROTO_TOKEN_TOOL_PARAM_NAME:
        if (r->use_color) renderer_puts(r, "\x1b[38;5;229m");
        renderer_puts(r, text);
        if (r->use_color) renderer_puts(r, "\x1b[0m");
        break;
    case PROTO_TOKEN_TOOL_PARAM_VALUE:
        if (r->use_color) renderer_puts(r, "\x1b[38;5;213m");
        renderer_puts(r, text);
        if (r->use_color) renderer_puts(r, "\x1b[0m");
        break;
    }
    r->wrote_visible_output = true;
    r->last_output_newline = text[strlen(text) - 1] == '\n';
}

/* ============================================================================
 * Status mirror / footer / prompt
 * ========================================================================== */

typedef struct {
    uint64_t state;
    uint64_t ctx_used;
    uint64_t ctx_size;
    uint64_t prefill_done;
    uint64_t prefill_total;
    float prefill_tps;
    uint64_t generated;
    float gen_tps;
    bool greedy;
    uint64_t power;
    char error[256];
} client_status;

static void client_format_ctx_size(int ctx_size, char *buf, size_t len) {
    if (ctx_size >= 1000) {
        if (ctx_size % 1000 == 0) snprintf(buf, len, "%dk", ctx_size / 1000);
        else snprintf(buf, len, "%.1fk", (double)ctx_size / 1000.0);
    } else {
        snprintf(buf, len, "%d", ctx_size);
    }
}

static void client_power_status_suffix(const client_status *st, char *buf, size_t len) {
    if (st->power > 0 && st->power < 100)
        snprintf(buf, len, " | \xe2\x9a\xa1 %d%%", (int)st->power);
    else
        buf[0] = '\0';
}

static void client_progress_append(char *buf, size_t len, size_t *pos, const char *s) {
    if (len == 0 || *pos >= len - 1) return;
    size_t avail = len - *pos;
    int n = snprintf(buf + *pos, avail, "%s", s);
    if (n <= 0) return;
    if ((size_t)n >= avail) *pos = len - 1;
    else *pos += (size_t)n;
}

static void client_progress_bar(int done, int total, double tps,
                                char *buf, size_t len, bool color) {
    if (len == 0) return;
    if (total <= 0) total = 1;
    if (done < 0) done = 0;
    if (done > total) done = total;
    int filled = (int)(((long long)done * CLIENT_PROGRESS_BAR_WIDTH) / total);
    if (filled < 0) filled = 0;
    if (filled > CLIENT_PROGRESS_BAR_WIDTH) filled = CLIENT_PROGRESS_BAR_WIDTH;
    if (color && filled == 0 && done < total) filled = 1;
    char rate[32] = {0};
    size_t rate_len = 0;
    if (tps > 0.0 && filled < CLIENT_PROGRESS_BAR_WIDTH) {
        snprintf(rate, sizeof(rate), " %.0ft/s", tps);
        rate_len = strlen(rate);
    }
    size_t pos = 0;
    client_progress_append(buf, len, &pos, "[");
    if (color) client_progress_append(buf, len, &pos, CLIENT_STATUS_BAR_FILL);
    for (int i = 0; i < CLIENT_PROGRESS_BAR_WIDTH && pos + 1 < len; i++) {
        if (color && i == filled)
            client_progress_append(buf, len, &pos, CLIENT_STATUS_STYLE_START);
        if (i >= filled && rate_len > 0 && (size_t)(i - filled) < rate_len) {
            char ch[2] = {rate[i - filled], '\0'};
            client_progress_append(buf, len, &pos, ch);
        } else {
            client_progress_append(buf, len, &pos, i < filled ? "\xe2\x96\xb6" : "\xc2\xb7");
        }
    }
    if (color) client_progress_append(buf, len, &pos, CLIENT_STATUS_STYLE_START);
    client_progress_append(buf, len, &pos, "]");
    buf[pos < len ? pos : len - 1] = '\0';
}

static void client_build_prompt_text(const client_status *st, char *buf, size_t len) {
    (void)st;
    snprintf(buf, len, "ds4-agent> ");
}

static void client_build_status_text(const client_status *st, char *buf, size_t len) {
    char used[32], total_ctx[32], power[32];
    client_format_ctx_size((int)st->ctx_used, used, sizeof(used));
    client_format_ctx_size((int)st->ctx_size, total_ctx, sizeof(total_ctx));
    client_power_status_suffix(st, power, sizeof(power));
    switch (st->state) {
    case AGENT_PREFILL: {
        int done = (int)st->prefill_done;
        int total = st->prefill_total > 0 ? (int)st->prefill_total : 1;
        if (done > total) done = total;
        double pct = 100.0 * (double)done / (double)total;
        char bar[256];
        client_progress_bar(done, total, st->prefill_tps, bar, sizeof(bar),
                            isatty(STDOUT_FILENO));
        snprintf(buf, len, "ctx %s/%s | prefill %s %d/%d %.1f%%%s",
                 used, total_ctx, bar, done, total, pct, power);
        break;
    }
    case AGENT_GENERATING:
        snprintf(buf, len, "ctx %s/%s | generation %d tokens%s %.1f t/s%s",
                 used, total_ctx, (int)st->generated,
                 st->greedy ? " \xe2\x9d\x84" : "", st->gen_tps, power);
        break;
    case AGENT_COMPACTING:
        snprintf(buf, len, "ctx %s/%s | COMPACTING summary %d tokens %.1f t/s%s",
                 used, total_ctx, (int)st->generated, st->gen_tps, power);
        break;
    case AGENT_DRAINING:
        snprintf(buf, len, "ctx %s/%s | stopping after distributed cluster drains%s",
                 used, total_ctx, power);
        break;
    case AGENT_SAVING:
        snprintf(buf, len, "ctx %s/%s | saving session%s", used, total_ctx, power);
        break;
    case AGENT_ERROR:
        snprintf(buf, len, "ctx %s/%s | error: %s%s", used, total_ctx,
                 st->error[0] ? st->error : "unknown error", power);
        break;
    case AGENT_STOPPED:
        snprintf(buf, len, "ctx %s/%s | interrupted%s", used, total_ctx, power);
        break;
    default:
        snprintf(buf, len, "ctx %s/%s | idle%s", used, total_ctx, power);
        break;
    }
}

static void client_build_footer_text(const client_status *st, const char *queued,
                                     char *buf, size_t len) {
    char status[512];
    client_build_status_text(st, status, sizeof(status));
    snprintf(buf, len, "%s", status);
    if (queued && queued[0]) {
        char preview[256];
        snprintf(preview, sizeof(preview), "queued: %s (ctrl+x to edit, ESC to send ASAP)", queued);
        snprintf(buf, len, "%s\n%s", preview, status);
    }
}

/* ============================================================================
 * Prompt queue
 * ========================================================================== */

typedef struct {
    char **v;
    size_t len;
    size_t cap;
} client_prompt_queue;

static void client_prompt_queue_push(client_prompt_queue *q, const char *text) {
    if (q->len == q->cap) {
        q->cap = q->cap ? q->cap * 2 : 4;
        q->v = xrealloc(q->v, q->cap * sizeof(q->v[0]));
    }
    q->v[q->len++] = xstrdup(text ? text : "");
}

static char *client_prompt_queue_pop(client_prompt_queue *q) {
    if (!q->len) return NULL;
    char *text = q->v[0];
    memmove(q->v, q->v + 1, (q->len - 1) * sizeof(q->v[0]));
    q->len--;
    return text;
}

static const char *client_prompt_queue_peek(const client_prompt_queue *q) {
    return q->len ? q->v[0] : NULL;
}

static char *client_prompt_queue_take_all(client_prompt_queue *q) {
    if (!q->len) return NULL;
    if (q->len == 1) return client_prompt_queue_pop(q);
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

static char *client_format_user_prompt_echo(const char *text) {
    agent_buf b = {0};
    if (isatty(STDOUT_FILENO)) {
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

static void client_prompt_queue_free(client_prompt_queue *q) {
    for (size_t i = 0; i < q->len; i++) free(q->v[i]);
    free(q->v);
    memset(q, 0, sizeof(*q));
}

/* ============================================================================
 * Editor (linenoise; prompt stays above streaming output)
 * ========================================================================== */

#define CLIENT_STATUS_REDRAW_INTERVAL_SEC 0.20

static void write_all(int fd, const char *p, size_t n) {
    if (fd == STDOUT_FILENO) {
        linenoiseWrite(fd, p, n);
        return;
    }
    while (n) {
        ssize_t wr = write(fd, p, n);
        if (wr <= 0) {
            if (wr < 0 && errno == EINTR) continue;
            return;
        }
        p += (size_t)wr;
        n -= (size_t)wr;
    }
}

static bool stdout_is_tty(void) {
    return isatty(STDOUT_FILENO) != 0;
}

static bool agent_footer_is_multiline(const char *status) {
    return status && strchr(status, '\n');
}

typedef struct {
    struct linenoiseState edit;
    char *input_buf;
    size_t input_buf_len;
    char prompt[160];
    char status[4096];
    bool prompt_dirty, status_dirty;
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
    agent_buf deferred_output;
    double last_prompt_redraw_time;
    int old_stdin_flags;
} client_editor;

static void editor_clear_deferred(client_editor *ed) {
    free(ed->deferred_output.ptr);
    ed->deferred_output.ptr = NULL;
    ed->deferred_output.len = 0;
    ed->deferred_output.cap = 0;
    ed->deferred_output.limit = 0;
    ed->deferred_output.truncated = false;
}

static void editor_status_escapes(const char *status, const char **start, const char **end) {
    bool tty = stdout_is_tty();
    bool embedded = status && strchr(status, '\n');
    *start = tty && !embedded ? CLIENT_STATUS_STYLE_START : "";
    *end = tty && status && status[0] ? CLIENT_STATUS_STYLE_END : "";
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

static void editor_save_output_cursor(client_editor *ed) {
    if (!ed->scroll_region) return;
    write_all(STDOUT_FILENO, "\0337", 2);
    ed->output_cursor_saved = true;
}

static void editor_restore_output_cursor(client_editor *ed) {
    if (!ed->scroll_region) return;
    if (ed->output_cursor_saved) {
        write_all(STDOUT_FILENO, "\0338", 2);
    } else {
        editor_csi_cursor(ed->output_bottom, 1);
    }
}

static void editor_move_to_prompt_row(client_editor *ed) {
    if (!ed->scroll_region) return;
    editor_csi_cursor(ed->prompt_row, 1);
}

static void editor_move_to_prompt_cursor(client_editor *ed) {
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

static void editor_clear_prompt_region(client_editor *ed) {
    if (!ed->scroll_region) return;
    for (int row = ed->prompt_row; row <= ed->term_rows; row++)
        editor_clear_row(row);
    ed->edit.oldrows = 0;
    ed->edit.oldstatusrows = 0;
    ed->edit.oldrpos = 1;
    ed->edit.oldpos = ed->edit.pos;
}

static bool editor_set_scroll_layout(client_editor *ed, int reserved_rows,
                                     bool allow_shrink, bool scroll_on_grow) {
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
    client_editor *ed = privdata;
    if (!ed || !ed->scroll_region) return 0;
    if (prompt_rows < 1) prompt_rows = 1;
    int reserved = (int)(prompt_rows + status_rows);
    if (!editor_set_scroll_layout(ed, reserved, true, true)) return 0;
    return ed->prompt_row;
}

static bool editor_configure_scroll_region(client_editor *ed) {
    if (ed->scroll_region) return true;
    if (!isatty(STDIN_FILENO) || !stdout_is_tty()) return false;

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

    editor_scroll_output_up(ed->output_bottom, 1);
    ed->output_cursor_saved = false;
    editor_csi_cursor(ed->output_bottom, 1);
    editor_save_output_cursor(ed);
    editor_move_to_prompt_row(ed);
    return true;
}

static void editor_restore_terminal_layout(client_editor *ed) {
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

/* Track streamed text with the same widths as linenoise. Partial UTF-8 and
 * escape sequences remain pending across worker-output chunks. */
static void editor_note_output(client_editor *ed, const char *text, size_t len) {
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

static void editor_update_prompt(client_editor *ed, const char *prompt) {
    if (strcmp(ed->prompt, prompt)) ed->prompt_dirty = true;
    snprintf(ed->prompt, sizeof(ed->prompt), "%s", prompt);
    ed->edit.prompt = ed->prompt;
    ed->edit.plen = strlen(ed->prompt);
}

static void editor_update_status(client_editor *ed, const char *status) {
    if (strcmp(ed->status, status ? status : "")) ed->status_dirty = true;
    snprintf(ed->status, sizeof(ed->status), "%s", status ? status : "");
    bool embedded_status = agent_footer_is_multiline(ed->status);
    const char *status_start = stdout_is_tty() && !embedded_status ?
        CLIENT_STATUS_STYLE_START : "";
    const char *status_end = stdout_is_tty() && ed->status[0] ?
        CLIENT_STATUS_STYLE_END : "";
    linenoiseEditSetStatus(&ed->edit, ed->status, status_start, status_end);
}

static void editor_redraw_visible_prompt(client_editor *ed) {
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

static bool editor_prompt_redraw_due(client_editor *ed) {
    double now = now_sec();
    if (ed->last_prompt_redraw_time <= 0.0 ||
        now - ed->last_prompt_redraw_time >= CLIENT_STATUS_REDRAW_INTERVAL_SEC)
    {
        return true;
    }
    return false;
}

static void editor_flush_prompt_status(client_editor *ed, bool force) {
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
        const char *sstart = "", *send = "";
        editor_status_escapes(ed->status, &sstart, &send);
        linenoiseEditSetStatus(&ed->edit, ed->status, sstart, send);
    }
    ed->prompt_dirty = ed->status_dirty = false;
    ed->last_prompt_redraw_time = now_sec();
    linenoiseEndUpdate();
}

static void editor_set_prompt_status(client_editor *ed, const char *prompt,
                                     const char *status) {
    if (strcmp(ed->prompt, prompt)) editor_update_prompt(ed, prompt);
    if (strcmp(ed->status, status ? status : "")) editor_update_status(ed, status);
    editor_flush_prompt_status(ed, false);
}

static void editor_write_preserve_prompt(client_editor *ed, const char *text, size_t len) {
    if (!text || !len) return;
    if (ed->active) linenoiseEditStop(&ed->edit);
    ed->active = false;
    ed->hidden = false;
    write_all(STDOUT_FILENO, text, len);
}

static bool editor_write_scroll_output_preserve_prompt(client_editor *ed,
                                                       const char *text,
                                                       size_t len,
                                                       bool settle_boundary) {
    agent_buf_append(&ed->deferred_output, text, len);
    if (!ed->deferred_output.len) return false;

    client_editor predicted = *ed;
    editor_note_output(&predicted, ed->deferred_output.ptr,
                       ed->deferred_output.len);
    if ((predicted.output_utf8_len || predicted.output_escape) && settle_boundary) {
        if (predicted.output_utf8_len) {
            ed->deferred_output.len -= predicted.output_utf8_len;
            agent_buf_append(&ed->deferred_output, "\xef\xbf\xbd", 3);
        } else {
            while (ed->deferred_output.len &&
                   ed->deferred_output.ptr[--ed->deferred_output.len] != 0x1b) {}
        }
        predicted = *ed;
        editor_note_output(&predicted, ed->deferred_output.ptr,
                           ed->deferred_output.len);
    }
    bool at_boundary = predicted.output_pending_wrap;
    if (predicted.output_utf8_len || predicted.output_escape) return false;
    if (at_boundary && !settle_boundary) return false;

    linenoiseBeginUpdate(STDOUT_FILENO);
    editor_restore_output_cursor(ed);
    editor_write_terminal_text(ed->deferred_output.ptr,
                               ed->deferred_output.len);
    editor_note_output(ed, ed->deferred_output.ptr,
                       ed->deferred_output.len);
    if (at_boundary) {
        write_all(STDOUT_FILENO, "\r\n", 2);
        ed->output_col = 0;
        ed->output_line_open = false;
        ed->output_pending_wrap = false;
    }
    editor_clear_deferred(ed);
    editor_save_output_cursor(ed);
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    editor_move_to_prompt_cursor(ed);
    linenoiseEndUpdate();
    ed->output_at_scroll_boundary = true;
    return true;
}

static void editor_hide(client_editor *ed) {
    if (!ed->active || ed->hidden) return;
    if (ed->scroll_region) {
        editor_clear_prompt_region(ed);
        editor_restore_output_cursor(ed);
        ed->hidden = true;
        return;
    }
    linenoiseHide(&ed->edit);
    if (ed->prompt_below_output) {
        write_all(STDOUT_FILENO, "\x1b[1A", 4);
        char seq[64];
        int n = snprintf(seq, sizeof(seq), "\x1b[%dG", ed->output_col + 1);
        if (n > 0) write_all(STDOUT_FILENO, seq, (size_t)n);
        ed->prompt_below_output = false;
    }
    ed->hidden = true;
}

static void editor_show(client_editor *ed) {
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
    write_all(STDOUT_FILENO, "\x1b[0m", 4);
    linenoiseShow(&ed->edit);
    ed->prompt_dirty = ed->status_dirty = false;
    ed->last_prompt_redraw_time = now_sec();
    ed->hidden = false;
}

static void editor_write_async(client_editor *ed, const char *text, size_t len,
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

    /* Fallback (no scroll region): preserve the prompt by hiding, writing the
     * output, then redrawing linenoise. */
    if (!len) {
        editor_set_prompt_status(ed, prompt, status);
        return;
    }
    editor_write_preserve_prompt(ed, text, len);
    snprintf(ed->prompt, sizeof(ed->prompt), "%s", prompt);
    snprintf(ed->status, sizeof(ed->status), "%s", status);
    linenoiseEditStart(&ed->edit, STDIN_FILENO, STDOUT_FILENO, ed->input_buf,
                       ed->input_buf_len, prompt);
    const char *sstart = "", *send = "";
    editor_status_escapes(status, &sstart, &send);
    linenoiseEditSetStatus(&ed->edit, status, sstart, send);
    ed->active = true;
}

static void client_write_welcome_banner(client_editor *ed, int ctx_size,
                                        const char *prompt, const char *statusline) {
    char ctx[32], banner[256];
    client_format_ctx_size(ctx_size, ctx, sizeof(ctx));
    if (stdout_is_tty()) {
        snprintf(banner, sizeof(banner),
                 "\x1b[1;97mDwarf\x1b[1;94mStar\x1b[0m 🐋 Agent, context %s tokens\n\n",
                 ctx);
    } else {
        snprintf(banner, sizeof(banner), "DwarfStar Agent, context %s tokens\n\n", ctx);
    }
    editor_write_async(ed, banner, strlen(banner), prompt, statusline, true);
}

static void editor_start(client_editor *ed, const char *prompt, const char *status) {
    /* Note: do not memset the whole struct.  On a reopen (after a submitted
     * line) the scroll-region layout (term_rows/cols, output_bottom, prompt_row,
     * reserved_rows) must be preserved so the prompt stays in its reserved
     * bottom rows without tearing down / scrolling the region above.  Only the
     * input and per-prompt state is reset here. */
    ed->input_buf_len = 4096;
    ed->input_buf = xmalloc(ed->input_buf_len);
    snprintf(ed->prompt, sizeof(ed->prompt), "%s", prompt);
    snprintf(ed->status, sizeof(ed->status), "%s", status ? status : "");
    bool had_scroll_region = ed->scroll_region;
    bool use_scroll_region = editor_configure_scroll_region(ed);
    if (use_scroll_region) {
        if (had_scroll_region)
            editor_set_scroll_layout(ed, 2, true, false);
        editor_move_to_prompt_row(ed);
    }
    if (linenoiseEditStart(&ed->edit, STDIN_FILENO, STDOUT_FILENO, ed->input_buf,
                           ed->input_buf_len, ed->prompt) != 0) {
        editor_restore_terminal_layout(ed);
        return;
    }
    bool embedded_status = agent_footer_is_multiline(ed->status);
    const char *status_start = stdout_is_tty() && !embedded_status ?
        CLIENT_STATUS_STYLE_START : "";
    const char *status_end = stdout_is_tty() && ed->status[0] ?
        CLIENT_STATUS_STYLE_END : "";
    linenoiseEditSetStatus(&ed->edit, ed->status, status_start, status_end);
    linenoiseEditSetLayoutCallback(&ed->edit, editor_linenoise_layout_changed, ed);
    if (isatty(ed->edit.ifd) || getenv("LINENOISE_ASSUME_TTY")) {
        linenoiseHide(&ed->edit);
        linenoiseShow(&ed->edit);
    }
    ed->active = true;
    ed->hidden = false;
    ed->output_line_open = false;
    ed->prompt_below_output = false;
    ed->output_col = 0;
    ed->output_pending_wrap = false;
    ed->output_utf8_len = 0;
    ed->output_escape = 0;
    ed->output_zwj = ed->output_regional = false;
    ed->output_glyph_width = 0;
}

static void editor_stop(client_editor *ed) {
    if (!ed->active) {
        free(ed->input_buf);
        ed->input_buf = NULL;
        return;
    }
    if (ed->deferred_output.len)
        editor_write_scroll_output_preserve_prompt(ed, NULL, 0, true);
    if (!ed->hidden && (isatty(ed->edit.ifd) || getenv("LINENOISE_ASSUME_TTY")))
        editor_hide(ed);
    linenoiseEditStop(&ed->edit);
    /* Keep the scroll-region layout live so a reopen (after a submitted line)
     * does not tear down the region and leave a blank line in the output area.
     * The caller restores the terminal layout once at shutdown. */
    free(ed->input_buf);
    ed->input_buf = NULL;
    ed->active = false;
    ed->hidden = false;
    editor_clear_deferred(ed);
}

static void editor_read_stdin(client_editor *ed) {
    char buf[4096];
    ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
    if (n <= 0) return;
    linenoiseEditQueueInput(&ed->edit, buf, (size_t)n);
}

/* Ctrl+C arrives as byte 3 in raw mode; pull it out of the queued input. */
static bool editor_take_queued_byte(client_editor *ed, unsigned char byte) {
    size_t n = linenoiseEditQueuedInput(&ed->edit);
    if (!n) return false;
    if (ed->edit.queued_input[ed->edit.queued_input_pos] == byte) {
        ed->edit.queued_input_pos++;
        return true;
    }
    return false;
}

static void editor_cancel_input_with_hint(client_editor *ed, const char *prompt,
                                          const char *status) {
    ed->input_buf[0] = '\0';
    linenoiseEditClear(&ed->edit);
    editor_set_prompt_status(ed, prompt, status);
}

/* ============================================================================
 * Client core
 * ========================================================================== */

typedef struct {
    client_config cfg;
    int sock_fd;
    int wake_fd[2];
    pthread_t worker_thread;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    bool stop;
    bool initialized;
    client_status status;
    bool tool_result_pending;
    /* UI -> worker requests */
    char *pending_user;
    unsigned char pending_cmd_tag;
    char *pending_cmd_arg;
    bool interrupt_pending;
    /* fit-context */
    bool count_pending;
    uint64_t count_result;
    /* output publish */
    agent_buf out;
    agent_buf render_buf;
    bool wake_pending;
    /* tools */
    agent_tools_worker tools;
    ds4_web *web;
    client_renderer renderer;
    FILE *trace;
    char *session_title;
    uint64_t session_created_at;
} agent_client;

static void client_publish(agent_client *c, const char *s, size_t n) {
    if (!s || !n) return;
    pthread_mutex_lock(&c->mu);
    agent_buf_append(&c->out, s, n);
    c->wake_pending = true;
    pthread_cond_signal(&c->cond);
    pthread_mutex_unlock(&c->mu);
}

static void client_publish_puts(agent_client *c, const char *s) {
    client_publish(c, s, strlen(s));
}

static void client_status_mirror(agent_client *c, const proto_status_msg *m) {
    pthread_mutex_lock(&c->mu);
    c->status.state = m->state;
    c->status.ctx_used = m->ctx_used;
    c->status.ctx_size = m->ctx_size;
    c->status.prefill_done = m->prefill_done;
    c->status.prefill_total = m->prefill_total;
    c->status.prefill_tps = m->prefill_tps;
    c->status.generated = m->generated;
    c->status.gen_tps = m->gen_tps;
    c->status.greedy = m->greedy;
    c->status.power = m->power;
    snprintf(c->status.error, sizeof(c->status.error), "%s", m->error ? m->error : "");
    c->wake_pending = true;
    pthread_cond_signal(&c->cond);
    pthread_mutex_unlock(&c->mu);
}

static void client_trace(agent_client *c, uint64_t id, unsigned char kind, const char *text) {
    if (!c->trace) return;
    fprintf(c->trace, "token %llu kind=%u text=", (unsigned long long)id, (unsigned int)kind);
    for (const char *p = text; *p; p++) {
        if (*p == '\\' || *p == '\n') fputc('\\', c->trace);
        fputc(*p, c->trace);
    }
    fputc('\n', c->trace);
}

static void client_dispatch_s2c(agent_client *c, unsigned char *frame, size_t len);

/* Send TOKENS and wait for COUNT (fit-context math).  Other S2C messages are
 * processed while waiting. */
static bool client_count_tokens(agent_client *c, const char *text, uint64_t *count_out) {
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf = proto_encode_string(&wr, PROTO_C2S_TOKENS, text ? text : "", &out_len);
    bool ok = client_send_all(c->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
    if (!ok) return false;
    pthread_mutex_lock(&c->mu);
    c->count_pending = true;
    c->count_result = 0;
    pthread_mutex_unlock(&c->mu);
    while (true) {
        unsigned char *frame = NULL;
        size_t len = 0;
        if (!client_read_frame(c->sock_fd, &frame, &len)) return false;
        unsigned char tag = 0;
        proto_reader payload;
        if (!proto_open_frame(frame, len, &payload, &tag)) {
            free(frame);
            continue;
        }
        if (tag == PROTO_S2C_COUNT) {
            uint64_t v = 0;
            proto_reader_varint(&payload, &v);
            pthread_mutex_lock(&c->mu);
            c->count_result = v;
            c->count_pending = false;
            pthread_mutex_unlock(&c->mu);
            free(frame);
            *count_out = v;
            return true;
        }
        if (tag == PROTO_S2C_ERROR) {
            char *msg = NULL;
            proto_reader_string(&payload, &msg);
            free(frame);
            return false;
        }
        client_dispatch_s2c(c, frame, len);
        free(frame);
    }
}

static void client_trace_text(agent_client *c, const char *label, const char *text) {
    if (!c->trace) return;
    fprintf(c->trace, "%s:", label);
    for (const char *p = text; *p; p++) {
        if (*p == '\\' || *p == '\n') fputc('\\', c->trace);
        fputc(*p, c->trace);
    }
    fputc('\n', c->trace);
}

/* Execute one tool call and send its result (plus optional ATTACH_IMAGE for
 * view_image). */
static void client_execute_tool(agent_client *c, proto_tool_call *call) {
    if (!call->name) {
        char msg[] = "Tool error: missing tool name\n";
        proto_writer wr;
        size_t out_len = 0;
        unsigned char *buf = proto_encode_string(&wr, PROTO_C2S_TOOL_RESULT, msg, &out_len);
        (void)client_send_all(c->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
        return;
    }
    if (!strcmp(call->name, "view_image")) {
        const char *path = agent_tools_arg_value(call, "path");
        if (!path || !path[0]) {
            char msg[] = "Tool error: view_image requires path\n";
            proto_writer wr;
            size_t out_len = 0;
            unsigned char *buf = proto_encode_string(&wr, PROTO_C2S_TOOL_RESULT, msg, &out_len);
            (void)client_send_all(c->sock_fd, buf, out_len);
            proto_writer_free(&wr);
            free(buf);
            return;
        }
        char err[256] = {0};
        char *data = NULL;
        size_t data_len = 0;
        if (agent_tools_file_read_file_bytes(path, &data, &data_len,
                                             err, sizeof(err)) != 0) {
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: view_image failed: ");
            agent_buf_puts(&b, err[0] ? err : "unable to read image");
            agent_buf_puts(&b, "\n");
            char *msg = agent_buf_take(&b);
            proto_writer wr;
            size_t out_len = 0;
            unsigned char *buf = proto_encode_string(&wr, PROTO_C2S_TOOL_RESULT, msg, &out_len);
            (void)client_send_all(c->sock_fd, buf, out_len);
            proto_writer_free(&wr);
            free(buf);
            free(msg);
            return;
        }
        proto_attach_image_msg m = { .data = (unsigned char *)data, .data_len = data_len };
        proto_writer wr;
        size_t out_len = 0;
        unsigned char *buf = proto_encode_attach_image(&wr, &m, &out_len);
        (void)client_send_all(c->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
        free(data);
        char meta[192];
        snprintf(meta, sizeof(meta), "\n[tool:view_image] %s\n", path);
        client_publish_puts(c, meta);
        /* Send an empty TOOL_RESULT; the image rides ATTACH_IMAGE. */
        buf = proto_encode_string(&wr, PROTO_C2S_TOOL_RESULT, "", &out_len);
        (void)client_send_all(c->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
        return;
    }

    char *obs = agent_tools_dispatch(&c->tools, call);
    if (obs && obs[0]) {
        client_publish_puts(c, obs);
        /* fit-context: round-trip the observation through TOKENS->COUNT so the
         * server can compact before resuming.  Truncation is left to the server
         * (compaction), so we keep the whole observation. */
        if (c->cfg.ctx_size > 0) {
            uint64_t count = 0;
            (void)client_count_tokens(c, obs, &count);
        }
    }
    if (!obs) obs = xstrdup("");
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf = proto_encode_string(&wr, PROTO_C2S_TOOL_RESULT, obs, &out_len);
    (void)client_send_all(c->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
    free(obs);
    proto_tool_call_free(call);
}

static void client_dispatch_s2c(agent_client *c, unsigned char *frame, size_t len) {
    unsigned char tag = 0;
    proto_reader payload;
    if (!proto_open_frame(frame, len, &payload, &tag)) return;
    switch (tag) {
    case PROTO_S2C_HELLO: {
        proto_hello_msg m = {0};
        if (!proto_decode_hello(frame, len, &tag, &m)) break;
        pthread_mutex_lock(&c->mu);
        free(c->session_title);
        c->session_title = xstrdup(m.session_title ? m.session_title : "");
        c->session_created_at = m.session_created_at;
        c->initialized = true;
        pthread_mutex_unlock(&c->mu);
        client_status_mirror(c, &m.status);
        if (c->trace) client_trace_text(c, "hello", m.session_title ? m.session_title : "");
        free(m.session_title);
        free(m.status.error);
        break;
    }
    case PROTO_S2C_STATUS: {
        proto_status_msg m = {0};
        if (!proto_decode_status(frame, len, &tag, &m)) break;
        client_status_mirror(c, &m);
        free(m.error);
        break;
    }
    case PROTO_S2C_TOKEN: {
        proto_token_msg m = {0};
        if (!proto_decode_token(frame, len, &tag, &m)) break;
        if (c->trace) client_trace(c, m.id, m.kind, m.text ? m.text : "");
        client_render_token(&c->renderer, m.kind, m.text ? m.text : "");
        pthread_mutex_lock(&c->mu);
        if (c->render_buf.len) {
            agent_buf_append(&c->out, c->render_buf.ptr, c->render_buf.len);
            c->render_buf.len = 0;
            c->render_buf.ptr[0] = '\0';
        }
        c->wake_pending = true;
        pthread_cond_signal(&c->cond);
        pthread_mutex_unlock(&c->mu);
        free(m.text);
        break;
    }
    case PROTO_S2C_TURN_PAUSED:
        /* A turn ended: flush any pending markdown literals/code fences. */
        renderer_finish(&c->renderer);
        pthread_mutex_lock(&c->mu);
        if (c->render_buf.len) {
            agent_buf_append(&c->out, c->render_buf.ptr, c->render_buf.len);
            c->render_buf.len = 0;
            c->render_buf.ptr[0] = '\0';
        }
        c->tool_result_pending = true;
        c->wake_pending = true;
        pthread_cond_signal(&c->cond);
        pthread_mutex_unlock(&c->mu);
        break;
    case PROTO_S2C_TOOL_CALLS: {
        proto_tool_calls calls = {0};
        if (!proto_decode_tool_calls(frame, len, &tag, &calls)) break;
        for (int i = 0; i < calls.len; i++)
            client_execute_tool(c, &calls.v[i]);
        proto_tool_calls_free(&calls);
        break;
    }
    case PROTO_S2C_SWITCH_DONE: {
        proto_switch_done_msg m = {0};
        if (!proto_decode_switch_done(frame, len, &tag, &m)) break;
        char msg[160];
        snprintf(msg, sizeof(msg), "\nswitched to session %s\n", m.sha ? m.sha : "");
        client_publish_puts(c, msg);
        if (c->trace) client_trace_text(c, "switch_done", m.sha ? m.sha : "");
        free(m.sha);
        break;
    }
    case PROTO_S2C_COMPACT_DONE:
        client_publish_puts(c, "\ncompaction complete\n");
        break;
    case PROTO_S2C_SAVE_DONE: {
        proto_save_done_msg m = {0};
        if (!proto_decode_save_done(frame, len, &tag, &m)) break;
        char msg[160];
        snprintf(msg, sizeof(msg), "\nsaved session %s (%d tokens)\n",
                 m.sha ? m.sha : "", (int)m.tokens);
        client_publish_puts(c, msg);
        free(m.sha);
        break;
    }
    case PROTO_S2C_HISTORY: {
        char *text = NULL;
        if (!proto_decode_string(frame, len, tag, &tag, &text)) break;
        client_publish_puts(c, text ? text : "");
        if (c->trace) client_trace_text(c, "history", text ? text : "");
        free(text);
        break;
    }
    case PROTO_S2C_LIST: {
        proto_list_msg m = {0};
        if (!proto_decode_list(frame, len, &tag, &m)) break;
        agent_buf b = {0};
        agent_buf_puts(&b, "\nSaved sessions:\n");
        if (m.count == 0) agent_buf_puts(&b, "  (none)\n");
        for (size_t i = 0; i < m.count; i++) {
            char line[256];
            snprintf(line, sizeof(line), "  %.8s  %s (%d tokens)\n",
                     m.items[i].sha, m.items[i].title ? m.items[i].title : "",
                     (int)m.items[i].tokens);
            agent_buf_puts(&b, line);
        }
        char *text = agent_buf_take(&b);
        client_publish_puts(c, text);
        free(text);
        proto_list_msg_free(&m);
        break;
    }
    case PROTO_S2C_COUNT: {
        uint64_t v = 0;
        proto_reader_varint(&payload, &v);
        pthread_mutex_lock(&c->mu);
        c->count_result = v;
        c->count_pending = false;
        pthread_cond_signal(&c->cond);
        pthread_mutex_unlock(&c->mu);
        break;
    }
    case PROTO_S2C_ERROR: {
        char *msg = NULL;
        if (!proto_reader_string(&payload, &msg)) break;
        client_publish_puts(c, msg ? msg : "server error");
        if (c->trace) client_trace_text(c, "error", msg ? msg : "");
        free(msg);
        break;
    }
    }
}

/* Send one C2S message.  Called by the UI thread. */
static bool client_send_c2s(agent_client *c, unsigned char tag, const char *arg) {
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf = NULL;
    switch (tag) {
    case PROTO_C2S_NEW_SESSION: {
        proto_new_session_msg m = {0};
        m.sys_extra = (char *)(c->cfg.sys_extra ? c->cfg.sys_extra : "");
        m.n_predict = c->cfg.n_predict;
        m.temperature = c->cfg.temperature;
        m.top_p = c->cfg.top_p;
        m.min_p = c->cfg.min_p;
        m.think_mode = c->cfg.think_mode;
        m.seed = c->cfg.seed;
        m.power = c->cfg.power;
        buf = proto_encode_new_session(&wr, &m, &out_len);
        break;
    }
    case PROTO_C2S_USER:
    case PROTO_C2S_SYSTEM:
    case PROTO_C2S_TOOL_RESULT:
    case PROTO_C2S_SWITCH:
    case PROTO_C2S_DEL:
    case PROTO_C2S_STRIP:
    case PROTO_C2S_HISTORY:
        buf = proto_encode_string(&wr, tag, arg ? arg : "", &out_len);
        break;
    default:
        buf = proto_encode_empty(&wr, tag, &out_len);
        break;
    }
    bool ok = buf && client_send_all(c->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
    return ok;
}

static void client_send_initial(agent_client *c) {
    (void)client_send_c2s(c, PROTO_C2S_NEW_SESSION, NULL);
}

/* Worker thread: reads the socket, dispatches S2C, executes tools, and drains
 * UI requests (USER / session commands / interrupt). */
static void *client_worker_main(void *arg) {
    agent_client *c = arg;
    client_send_initial(c);
    while (!c->stop) {
        struct pollfd pfd[2] = {
            {.fd = c->sock_fd, .events = POLLIN},
            {.fd = c->wake_fd[0], .events = POLLIN},
        };
        int rc = poll(pfd, 2, 100);
        if (rc < 0 && errno != EINTR) break;

        /* Drain UI requests. */
        pthread_mutex_lock(&c->mu);
        bool interrupt = c->interrupt_pending;
        c->interrupt_pending = false;
        char *user = c->pending_user;
        c->pending_user = NULL;
        unsigned char cmd_tag = c->pending_cmd_tag;
        char *cmd_arg = c->pending_cmd_arg;
        c->pending_cmd_arg = NULL;
        c->pending_cmd_tag = 0;
        pthread_mutex_unlock(&c->mu);

        if (interrupt) (void)client_send_c2s(c, PROTO_C2S_INTERRUPT, NULL);
        if (cmd_tag) (void)client_send_c2s(c, cmd_tag, cmd_arg ? cmd_arg : "");
        if (user) {
            (void)client_send_c2s(c, PROTO_C2S_USER, user);
            free(user);
        }
        free(cmd_arg);

        if (rc > 0 && (pfd[0].revents & POLLIN)) {
            unsigned char *frame = NULL;
            size_t len = 0;
            if (!client_read_frame(c->sock_fd, &frame, &len)) {
                if (!c->stop) client_publish_puts(c, "\ndisconnected from server\n");
                break;
            }
            client_dispatch_s2c(c, frame, len);
            free(frame);
        }
    }
    return NULL;
}

/* ============================================================================
 * Session commands (UI side)
 * ========================================================================== */

static bool client_slash_command_with_args(const char *cmd, const char *name) {
    size_t len = strlen(name);
    return !strncmp(cmd, name, len) &&
           (cmd[len] == '\0' || isspace((unsigned char)cmd[len]));
}

static void client_request(agent_client *c, unsigned char tag, const char *arg) {
    pthread_mutex_lock(&c->mu);
    free(c->pending_cmd_arg);
    c->pending_cmd_arg = arg ? xstrdup(arg) : NULL;
    c->pending_cmd_tag = tag;
    c->wake_pending = true;
    pthread_cond_signal(&c->cond);
    pthread_mutex_unlock(&c->mu);
}

/* Echo a submitted user message into the chat output, matching the monolith's
 * agent_echo_user_prompt formatting. */
static void client_echo_user(agent_client *c, const char *text) {
    agent_buf b = {0};
    if (stdout_is_tty()) {
        agent_buf_puts(&b, "\x1b[1;91m*\x1b[1;97m ");
        agent_buf_puts(&b, text ? text : "");
        agent_buf_puts(&b, "\x1b[0m\n\n");
    } else {
        agent_buf_puts(&b, "* ");
        agent_buf_puts(&b, text ? text : "");
        agent_buf_puts(&b, "\n\n");
    }
    char *msg = agent_buf_take(&b);
    client_publish_puts(c, msg);
    free(msg);
}

static void client_submit_user(agent_client *c, const char *text) {
    pthread_mutex_lock(&c->mu);
    free(c->pending_user);
    c->pending_user = xstrdup(text ? text : "");
    c->wake_pending = true;
    pthread_cond_signal(&c->cond);
    pthread_mutex_unlock(&c->mu);
}

static void client_interrupt(agent_client *c) {
    pthread_mutex_lock(&c->mu);
    c->interrupt_pending = true;
    c->wake_pending = true;
    pthread_cond_signal(&c->cond);
    pthread_mutex_unlock(&c->mu);
}

static void client_handle_command(agent_client *c, const char *cmd) {
    if (!strcmp(cmd, "/save")) {
        client_request(c, PROTO_C2S_SAVE, NULL);
        client_publish_puts(c, "save requested\n");
    } else if (!strcmp(cmd, "/compact")) {
        client_request(c, PROTO_C2S_COMPACT, NULL);
        client_publish_puts(c, "compaction requested\n");
    } else if (!strcmp(cmd, "/list")) {
        client_request(c, PROTO_C2S_LIST, NULL);
    } else if (!strncmp(cmd, "/power", 6) &&
               (cmd[6] == '\0' || cmd[6] == ' ' || cmd[6] == '\t')) {
        char *arg = (char *)cmd + 6;
        while (*arg == ' ' || *arg == '\t') arg++;
        if (!arg[0]) {
            client_publish_puts(c, "usage: /power <1..100>\n");
        } else {
            char *end = NULL;
            long v = strtol(arg, &end, 10);
            if (end == arg || *end || v < 1 || v > 100) {
                client_publish_puts(c, "usage: /power <1..100>\n");
            } else {
                client_request(c, PROTO_C2S_POWER, NULL);
                c->cfg.power = (uint64_t)v;
            }
        }
    } else if (client_slash_command_with_args(cmd, "/switch")) {
        char *arg = (char *)cmd + strlen("/switch");
        while (*arg == ' ' || *arg == '\t') arg++;
        if (!arg[0]) client_publish_puts(c, "usage: /switch <sha-prefix>\n");
        else client_request(c, PROTO_C2S_SWITCH, arg);
    } else if (client_slash_command_with_args(cmd, "/del")) {
        char *arg = (char *)cmd + strlen("/del");
        while (*arg == ' ' || *arg == '\t') arg++;
        if (!arg[0]) client_publish_puts(c, "usage: /del <sha-prefix>\n");
        else client_request(c, PROTO_C2S_DEL, arg);
    } else if (client_slash_command_with_args(cmd, "/strip")) {
        char *arg = (char *)cmd + strlen("/strip");
        while (*arg == ' ' || *arg == '\t') arg++;
        if (!arg[0]) client_publish_puts(c, "usage: /strip <sha-prefix>\n");
        else client_request(c, PROTO_C2S_STRIP, arg);
    } else if (client_slash_command_with_args(cmd, "/history")) {
        char *arg = (char *)cmd + strlen("/history");
        while (*arg == ' ' || *arg == '\t') arg++;
        client_request(c, PROTO_C2S_HISTORY, arg);
    } else if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
        /* handled by the caller */
    } else {
        client_publish_puts(c, "unknown command\n");
    }
}

/* ============================================================================
 * Interactive main loop
 * ========================================================================== */

static void client_consume_output(agent_client *c, char **out, size_t *out_len,
                                  client_status *status) {
    pthread_mutex_lock(&c->mu);
    if (out) {
        *out = c->out.ptr ? c->out.ptr : NULL;
        *out_len = c->out.len;
        c->out.ptr = NULL;
        c->out.len = 0;
        c->out.cap = 0;
    }
    if (status) *status = c->status;
    c->wake_pending = false;
    pthread_mutex_unlock(&c->mu);
}

static bool client_worker_idle(agent_client *c) {
    pthread_mutex_lock(&c->mu);
    bool idle = c->initialized &&
        (c->status.state == AGENT_IDLE || c->status.state == AGENT_ERROR);
    pthread_mutex_unlock(&c->mu);
    return idle;
}

static void client_run_non_interactive(agent_client *c) {
    char buf[4096];
    client_status st;
    client_consume_output(c, NULL, NULL, &st);
    if (c->cfg.initial_prompt && c->cfg.initial_prompt[0])
        client_submit_user(c, c->cfg.initial_prompt);
    while (!c->stop) {
        char *out = NULL;
        size_t out_len = 0;
        client_consume_output(c, &out, &out_len, &st);
        if (out && out_len) {
            (void)write(STDOUT_FILENO, out, out_len);
            free(out);
        }
        if (st.state == AGENT_IDLE || st.state == AGENT_ERROR) {
            ssize_t n = read(STDIN_FILENO, buf, sizeof(buf));
            if (n <= 0) break;
            char *line = xstrndup(buf, (size_t)n);
            char *echo = client_format_user_prompt_echo(line);
            if (echo) {
                (void)write(STDOUT_FILENO, echo, strlen(echo));
                free(echo);
            }
            client_submit_user(c, line);
            free(line);
        } else {
            struct timespec ts;
            ts.tv_sec = 0;
            ts.tv_nsec = 50 * 1000 * 1000;
            nanosleep(&ts, NULL);
        }
    }
    char *out = NULL;
    size_t out_len = 0;
    client_consume_output(c, &out, &out_len, &st);
    if (out && out_len) (void)write(STDOUT_FILENO, out, out_len);
    free(out);
}

/* Feed one byte/line to the linenoise editor and act on the result.  Returns
 * false only when the caller should stop the loop (EOF/error or /quit). */
static bool client_feed_editor(client_editor *ed, client_prompt_queue *queue,
                               agent_client *c, bool *running) {
    errno = 0;
    char *line = linenoiseEditFeed(&ed->edit);
    if (line == linenoiseEditMore) {
        /* still editing; more queued bytes processed by the caller */
        linenoiseFree(line);
        return true;
    }
    if (!line) {
        if (errno == EAGAIN) return true;
        *running = false;
        return false;
    }
    char *cmd = line;
    while (*cmd == ' ' || *cmd == '\t' || *cmd == '\r' || *cmd == '\n') cmd++;
    char *end = cmd + strlen(cmd);
    while (end > cmd && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\r' || end[-1] == '\n')) end--;
    *end = '\0';
    if (cmd[0]) {
        if (!strcmp(cmd, "/quit") || !strcmp(cmd, "/exit")) {
            *running = false;
        } else if (cmd[0] == '/') {
            client_handle_command(c, cmd);
        } else if (client_worker_idle(c)) {
            linenoiseHistoryAdd(cmd);
            client_submit_user(c, cmd);
            client_echo_user(c, cmd);
        } else {
            /* Worker is busy: queue the prompt; drain when idle. */
            client_prompt_queue_push(queue, cmd);
        }
    }
    linenoiseFree(line);
    /* Reopen the editor after a submitted line so the prompt is redrawn empty
     * (the accepted line is not left on screen) and linenoiseEditStart
     * re-seeds the "current buffer" history slot */
    if (*running && ed->active) {
        char saved_prompt[160], saved_status[4096];
        snprintf(saved_prompt, sizeof(saved_prompt), "%s", ed->prompt);
        snprintf(saved_status, sizeof(saved_status), "%s", ed->status);
        editor_stop(ed);
        editor_start(ed, saved_prompt, saved_status);
    }
    return true;
}

static void client_run_interactive(agent_client *c) {
    linenoiseSetMultiLine(1);
    linenoiseHistorySetMaxLen(512);
    client_status st;
    client_consume_output(c, NULL, NULL, &st);
    char prompt[160], statusline[4096];
    client_build_prompt_text(&st, prompt, sizeof(prompt));
    client_build_footer_text(&st, NULL, statusline, sizeof(statusline));
    client_editor editor = {0};
    editor_start(&editor, prompt, statusline);
    client_write_welcome_banner(&editor, st.ctx_size, prompt, statusline);

    client_prompt_queue queue = {0};
    bool running = true;
    char *initial_pending = c->cfg.initial_prompt && c->cfg.initial_prompt[0] ?
                            xstrdup(c->cfg.initial_prompt) : NULL;

    while (running) {
        struct pollfd pfd[2] = {
            {.fd = STDIN_FILENO, .events = POLLIN},
            {.fd = c->wake_fd[0], .events = POLLIN},
        };
        int timeout = linenoiseEditQueuedInput(&editor.edit) > 0 ? 0 : 100;
        int rc = poll(pfd, 2, timeout);
        if (rc < 0 && errno != EINTR) break;

        if (rc > 0 && (pfd[0].revents & POLLIN)) editor_read_stdin(&editor);

        if (editor_take_queued_byte(&editor, 3)) { /* Ctrl+C */
            if (!client_worker_idle(c)) client_interrupt(c);
            else editor_cancel_input_with_hint(&editor, prompt, statusline);
        }

        char *out = NULL;
        size_t out_len = 0;
        client_consume_output(c, &out, &out_len, &st);
        client_build_prompt_text(&st, prompt, sizeof(prompt));
        client_build_footer_text(&st, client_prompt_queue_peek(&queue),
                                 statusline, sizeof(statusline));
        if (out && out_len) {
            bool force_show = st.state == AGENT_IDLE ||
                              st.state == AGENT_ERROR ||
                              st.state == AGENT_STOPPED;
            editor_write_async(&editor, out, out_len, prompt, statusline, force_show);
            free(out);
        } else {
            editor_set_prompt_status(&editor, prompt, statusline);
            editor_flush_prompt_status(&editor, st.state == AGENT_IDLE ||
                                       st.state == AGENT_ERROR ||
                                       st.state == AGENT_STOPPED);
            if (editor.hidden && (st.state == AGENT_IDLE ||
                                  st.state == AGENT_ERROR ||
                                  st.state == AGENT_STOPPED))
                editor_show(&editor);
        }

        if (initial_pending && client_worker_idle(c)) {
            client_submit_user(c, initial_pending);
            free(initial_pending);
            initial_pending = NULL;
        }

        if (queue.len && client_worker_idle(c)) {
            char *queued = client_prompt_queue_take_all(&queue);
            client_submit_user(c, queued);
            free(queued);
        }

        /* Process input without blocking the event loop on a TTY.
         * linenoiseEditFeed blocks on a blocking read(STDIN) when no bytes
         * are queued, which would stall this loop (and streamed output) until
         * the user presses a key.  Only feed it while queued bytes remain, so
         * the loop keeps rendering server output during generation.  On a
         * non-TTY stdin linenoise uses its blocking no-tty readline instead,
         * so feed it once unconditionally (a whole line arrives at once). */
        if (isatty(STDIN_FILENO)) {
            /* TTY: only feed while queued bytes remain, so this loop never
             * blocks on linenoiseEditFeed's blocking read(STDIN) and keeps
             * rendering streamed output during generation. */
            while (running && linenoiseEditQueuedInput(&editor.edit) > 0)
                if (!client_feed_editor(&editor, &queue, c, &running)) break;
        } else {
            /* Non-TTY: linenoise uses its blocking no-tty readline, which
             * returns a whole line at once; feed it once per loop pass. */
            (void)client_feed_editor(&editor, &queue, c, &running);
        }
    }
    editor_stop(&editor);
    editor_restore_terminal_layout(&editor);
    client_prompt_queue_free(&queue);
    free(initial_pending);
}

/* ============================================================================
 * main
 * ========================================================================== */

static void client_init(agent_client *c, const client_config *cfg, int sock_fd) {
    memset(c, 0, sizeof(*c));
    c->cfg = *cfg;
    c->sock_fd = sock_fd;
    c->wake_fd[0] = c->wake_fd[1] = -1;
    pthread_mutex_init(&c->mu, NULL);
    pthread_cond_init(&c->cond, NULL);
    if (pipe(c->wake_fd) != 0) {
        fprintf(stderr, "ds4-agent-client: failed to create wake pipe\n");
        exit(1);
    }
    fcntl(c->wake_fd[0], F_SETFL, O_NONBLOCK);
    fcntl(c->wake_fd[1], F_SETFL, O_NONBLOCK);
    c->tools.ctx_size = cfg->ctx_size;
    c->tools.io.ask_web = cfg->web;
    if (cfg->web) {
        ds4_web_config wcfg = {0};
        wcfg.home_dir = getenv("HOME");
        wcfg.confirm = NULL;
        wcfg.log = NULL;
        wcfg.cancel = NULL;
        c->web = ds4_web_create(&wcfg);
    }
    c->tools.web = c->web;
    c->renderer.format_markdown = true;
    c->renderer.use_color = isatty(STDOUT_FILENO) != 0;
    c->renderer.out = &c->render_buf;
    if (cfg->trace_path) {
        c->trace = fopen(cfg->trace_path, "w");
        if (!c->trace)
            fprintf(stderr, "ds4-agent-client: failed to open trace %s\n", cfg->trace_path);
    }
    c->status.state = AGENT_IDLE;
    c->status.ctx_size = cfg->ctx_size;
}

static void client_free(agent_client *c) {
    pthread_mutex_lock(&c->mu);
    c->stop = true;
    pthread_cond_signal(&c->cond);
    pthread_mutex_unlock(&c->mu);
    if (c->worker_thread) pthread_join(c->worker_thread, NULL);
    if (c->sock_fd >= 0) close(c->sock_fd);
    if (c->wake_fd[0] >= 0) close(c->wake_fd[0]);
    if (c->wake_fd[1] >= 0) close(c->wake_fd[1]);
    agent_tools_bash_jobs_free(&c->tools);
    if (c->web) ds4_web_free(c->web);
    if (c->trace) fclose(c->trace);
    free(c->out.ptr);
    free(c->render_buf.ptr);
    free(c->pending_user);
    free(c->pending_cmd_arg);
    free(c->session_title);
    pthread_mutex_destroy(&c->mu);
    pthread_cond_destroy(&c->cond);
}

static int client_run(agent_client *c) {
    if (pthread_create(&c->worker_thread, NULL, client_worker_main, c) != 0) {
        fprintf(stderr, "ds4-agent-client: failed to start worker thread\n");
        return 1;
    }
    if (c->cfg.non_interactive)
        client_run_non_interactive(c);
    else
        client_run_interactive(c);
    return 0;
}

#ifndef DS4_AGENT_TEST
int main(int argc, char **argv) {
    client_config cfg = client_parse_options(argc, argv);
    char err[256] = {0};
    int fd = client_connect(&cfg, err, sizeof(err));
    if (fd < 0) {
        fprintf(stderr, "ds4-agent-client: %s\n", err);
        return 1;
    }
    agent_client c;
    client_init(&c, &cfg, fd);
    int rc = client_run(&c);
    client_free(&c);
    return rc;
}
#endif

#ifdef DS4_AGENT_TEST
/* Layer 2 mock-server harness: the test acts as ds4-agent-server over a
 * socketpair, feeds scripted HELLO/STATUS/TOKEN/TURN_PAUSED/TOOL_CALLS/LIST,
 * and verifies the client's rendered output, messages sent, tool execution +
 * fit-context (TOKENS->COUNT), and session command rendering. */

static int client_test_failures;

static void client_test_check(bool cond, const char *expr, const char *file, int line) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    client_test_failures++;
}

#define CLIENT_TEST_CHECK(expr) client_test_check((expr), #expr, __FILE__, __LINE__)

/* Poll the client's published output until it contains needle (or timeout).
 * Returns a malloc'd buffer of everything consumed (caller frees). */
static char *client_test_wait_output(agent_client *c, const char *needle, int timeout_ms) {
    agent_buf b = {0};
    int waited = 0;
    while (waited < timeout_ms) {
        char *out = NULL;
        size_t out_len = 0;
        client_status st;
        client_consume_output(c, &out, &out_len, &st);
        if (out && out_len) {
            agent_buf_append(&b, out, out_len);
            free(out);
        }
        if (needle && strstr(b.ptr ? b.ptr : "", needle)) return agent_buf_take(&b);
        struct timespec ts = {0, 10 * 1000 * 1000};
        nanosleep(&ts, NULL);
        waited += 10;
    }
    /* final drain */
    char *out = NULL;
    size_t out_len = 0;
    client_status st;
    client_consume_output(c, &out, &out_len, &st);
    if (out && out_len) {
        agent_buf_append(&b, out, out_len);
        free(out);
    }
    return agent_buf_take(&b);
}

static void client_test_send_hello(int fd) {
    proto_writer wr;
    size_t out_len = 0;
    proto_hello_msg h = {0};
    h.session_title = xstrdup("My Session");
    h.session_created_at = 1234567890;
    h.status.state = AGENT_IDLE;
    h.status.ctx_used = 0;
    h.status.ctx_size = 8192;
    h.status.prefill_done = 0;
    h.status.prefill_total = 0;
    h.status.prefill_tps = 0.0f;
    h.status.generated = 0;
    h.status.gen_tps = 0.0f;
    h.status.greedy = false;
    h.status.power = 0;
    h.status.error = xstrdup("");
    unsigned char *f = proto_encode_hello(&wr, &h, &out_len);
    if (!client_send_all(fd, f, out_len)) { client_test_failures++; }
    proto_writer_free(&wr);
    free(f);
    free(h.session_title);
    free(h.status.error);
}

static void client_test_send_token(int fd, uint64_t id, unsigned char kind, const char *text) {
    proto_writer wr;
    size_t out_len = 0;
    proto_token_msg t = {0};
    t.id = id;
    t.kind = kind;
    t.text = xstrdup(text);
    unsigned char *f = proto_encode_token(&wr, &t, &out_len);
    if (!client_send_all(fd, f, out_len)) { client_test_failures++; }
    proto_writer_free(&wr);
    free(f);
    free(t.text);
}

static void client_test_send_status(int fd) {
    proto_writer wr;
    size_t out_len = 0;
    proto_status_msg st = {0};
    st.state = AGENT_GENERATING;
    st.ctx_used = 100;
    st.ctx_size = 8192;
    st.prefill_done = 0;
    st.prefill_total = 0;
    st.prefill_tps = 0.0f;
    st.generated = 5;
    st.gen_tps = 55.0f;
    st.greedy = false;
    st.power = 0;
    st.error = xstrdup("");
    unsigned char *f = proto_encode_status(&wr, &st, &out_len);
    if (!client_send_all(fd, f, out_len)) { client_test_failures++; }
    proto_writer_free(&wr);
    free(f);
    free(st.error);
}

static void client_test_send_empty(int fd, unsigned char tag) {
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *f = proto_encode_empty(&wr, tag, &out_len);
    if (!client_send_all(fd, f, out_len)) { client_test_failures++; }
    proto_writer_free(&wr);
    free(f);
}

static void client_test_send_tool_calls(int fd) {
    proto_writer wr;
    size_t out_len = 0;
    proto_tool_calls calls = {0};
    proto_tool_call tc = {0};
    tc.name = xstrdup("bash");
    proto_tool_call_add_arg(&tc, "command", "printf 'hello-tool'",
                            strlen("printf 'hello-tool'"), true);
    proto_tool_call_add_arg(&tc, "timeout_sec", "60", 2, false);
    proto_tool_calls_push(&calls, &tc);
    unsigned char *f = proto_encode_tool_calls(&wr, &calls, &out_len);
    if (!client_send_all(fd, f, out_len)) { client_test_failures++; }
    proto_writer_free(&wr);
    free(f);
    proto_tool_calls_free(&calls);
}

static void client_test_render_and_status(void) {
    int sv[2];
    if (socketpair(AF_UNIX, SOCK_STREAM, 0, sv) != 0) {
        fprintf(stderr, "client_test: socketpair failed\n");
        exit(1);
    }
    client_config cfg = {
        .ctx_size = 8192,
        .n_predict = 100,
        .temperature = 0.0f,
        .top_p = 1.0f,
        .min_p = 0.0f,
        .think_mode = 1,
        .seed = 0,
        .power = 0,
    };
    agent_client c;
    client_init(&c, &cfg, sv[0]);
    if (pthread_create(&c.worker_thread, NULL, client_worker_main, &c) != 0) {
        fprintf(stderr, "client_test: failed to start worker\n");
        exit(1);
    }

    /* Worker sends NEW_SESSION first. */
    unsigned char *frame = NULL;
    size_t len = 0;
    unsigned char tag = 0;
    proto_reader payload;
    CLIENT_TEST_CHECK(client_read_frame(sv[1], &frame, &len));
    CLIENT_TEST_CHECK(proto_open_frame(frame, len, &payload, &tag));
    CLIENT_TEST_CHECK(tag == PROTO_C2S_NEW_SESSION);
    proto_new_session_msg m = {0};
    CLIENT_TEST_CHECK(proto_decode_new_session(frame, len, &tag, &m));
    CLIENT_TEST_CHECK(m.n_predict == 100);
    free(m.sys_extra);
    free(frame);

    client_test_send_hello(sv[1]);
    client_test_send_token(sv[1], 1, PROTO_TOKEN_NORMAL, "Hello, ");
    client_test_send_token(sv[1], 2, PROTO_TOKEN_NORMAL, "world\n");
    client_test_send_status(sv[1]);
    client_test_send_token(sv[1], 3, PROTO_TOKEN_THINK, "secret");
    client_test_send_empty(sv[1], PROTO_S2C_TURN_PAUSED);

    char *out = client_test_wait_output(&c, "Hello, world", 5000);
    CLIENT_TEST_CHECK(out && strstr(out, "Hello, world"));
    CLIENT_TEST_CHECK(out && strstr(out, "secret"));
    free(out);

    /* STATUS mirroring: the client keeps the last server status. */
    char *o2 = NULL;
    size_t l2 = 0;
    client_status st2;
    client_consume_output(&c, &o2, &l2, &st2);
    CLIENT_TEST_CHECK(st2.state == AGENT_GENERATING);
    CLIENT_TEST_CHECK(st2.ctx_used == 100);
    CLIENT_TEST_CHECK(st2.generated == 5);
    free(o2);

    /* Tool execution + fit-context: TOKENS->COUNT round trip, then TOOL_RESULT. */
    client_test_send_tool_calls(sv[1]);
    frame = NULL;
    len = 0;
    CLIENT_TEST_CHECK(client_read_frame(sv[1], &frame, &len));
    CLIENT_TEST_CHECK(proto_open_frame(frame, len, &payload, &tag));
    CLIENT_TEST_CHECK(tag == PROTO_C2S_TOKENS);
    char *count_text = NULL;
    CLIENT_TEST_CHECK(proto_decode_string(frame, len, PROTO_C2S_TOKENS, &tag, &count_text));
    CLIENT_TEST_CHECK(count_text && strstr(count_text, "hello-tool"));
    free(count_text);
    free(frame);

    proto_writer wr;
    size_t out_len = 0;
    unsigned char *f = proto_encode_varint(&wr, PROTO_S2C_COUNT, 123, &out_len);
    CLIENT_TEST_CHECK(client_send_all(sv[1], f, out_len));
    proto_writer_free(&wr);
    free(f);

    frame = NULL;
    len = 0;
    CLIENT_TEST_CHECK(client_read_frame(sv[1], &frame, &len));
    CLIENT_TEST_CHECK(proto_open_frame(frame, len, &payload, &tag));
    CLIENT_TEST_CHECK(tag == PROTO_C2S_TOOL_RESULT);
    char *obs = NULL;
    CLIENT_TEST_CHECK(proto_decode_string(frame, len, PROTO_C2S_TOOL_RESULT, &tag, &obs));
    CLIENT_TEST_CHECK(obs && strstr(obs, "hello-tool"));
    free(obs);
    free(frame);

    out = client_test_wait_output(&c, "hello-tool", 5000);
    CLIENT_TEST_CHECK(out && strstr(out, "hello-tool"));
    free(out);

    /* Session commands: /list round trip renders the LIST response. */
    client_handle_command(&c, "/list");
    frame = NULL;
    len = 0;
    CLIENT_TEST_CHECK(client_read_frame(sv[1], &frame, &len));
    CLIENT_TEST_CHECK(proto_open_frame(frame, len, &payload, &tag));
    CLIENT_TEST_CHECK(tag == PROTO_C2S_LIST);
    free(frame);

    proto_list_item it = {0};
    strcpy(it.sha, "0123456789abcdef0123456789abcdef01234567");
    it.title = xstrdup("Title");
    it.last_used = 100;
    it.created_at = 200;
    it.tokens = 300;
    it.file_size = 400;
    it.payload_bytes = 500;
    proto_list_msg lm = {0};
    lm.count = 1;
    lm.cap = 1;
    lm.items = &it;
    f = proto_encode_list(&wr, &lm, &out_len);
    CLIENT_TEST_CHECK(client_send_all(sv[1], f, out_len));
    proto_writer_free(&wr);
    free(f);
    free(it.title);

    out = client_test_wait_output(&c, "Saved sessions", 5000);
    CLIENT_TEST_CHECK(out && strstr(out, "Saved sessions"));
    CLIENT_TEST_CHECK(out && strstr(out, "Title"));
    free(out);

    /* /power requests POWER and updates local power. */
    client_handle_command(&c, "/power 5");
    frame = NULL;
    len = 0;
    CLIENT_TEST_CHECK(client_read_frame(sv[1], &frame, &len));
    CLIENT_TEST_CHECK(proto_open_frame(frame, len, &payload, &tag));
    CLIENT_TEST_CHECK(tag == PROTO_C2S_POWER);
    free(frame);

    /* /del without an argument renders usage immediately. */
    client_handle_command(&c, "/del");
    out = client_test_wait_output(&c, "usage: /del", 2000);
    CLIENT_TEST_CHECK(out && strstr(out, "usage: /del"));
    free(out);

    /* Shutdown: stop the worker, close the mock side, then free. */
    pthread_mutex_lock(&c.mu);
    c.stop = true;
    pthread_cond_signal(&c.cond);
    pthread_mutex_unlock(&c.mu);
    close(sv[1]);
    client_free(&c);
}

int main(void) {
    client_test_render_and_status();
    if (client_test_failures) {
        fprintf(stderr, "ds4_agent_client_test: %d failure(s)\n", client_test_failures);
        return 1;
    }
    printf("ds4_agent_client_test: all ok\n");
    return 0;
}
#endif
