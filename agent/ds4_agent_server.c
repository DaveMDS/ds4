/* ds4_agent_server.c -- server-internal streaming DSML/GLM parser,
 * marker/think trackers, greedy boundaries, and render-directive producer.
 *
 * This is the Task 2 port of the corresponding logic from the monolithic
 * ds4_agent.c.  It is self-contained (no engine, no file I/O, no terminal).
 *
 * Differences from the monolithic source:
 *   - The terminal renderer (agent_token_renderer, markdown, code highlighting,
 *     colors, cursor escapes) is replaced by a render-directive sink
 *     (agent_render_sink) that tags text fragments with proto_token_kind so the
 *     client can paint them (NORMAL / THINK / TOOL_NAME / TOOL_PARAM_NAME /
 *     TOOL_PARAM_VALUE).
 *   - Terminal-only painting (prefixes "$ ", "🛠️ ", cursor clears, diff "-/+ "
 *     prefixes, syntax highlighting) is dropped.  The read-tool summary
 *     ("Reading <path> <start>:<max>...") is kept because it is a semantic
 *     projection the client should paint, and the ported tests assert it.
 *   - The mid-generation edit-old preflight
 *     (agent_preflight_edit_old / agent_stream_preflight_closed_param) is
 *     dropped: the server cannot read files (client-side).  The client checks
 *     exact-old uniqueness at edit execution time.
 *   - Parsed tool calls use the shared proto_tool_calls types so the caller can
 *     encode them straight into PROTO_S2C_TOOL_CALLS frames.
 *
 * The DSML fullwidth-bar marker bytes are written as \xEF\xBF\xBC escapes here
 * to keep this file ASCII-only; the compiler resolves them to the same bytes as
 * the monolithic source.
 *
 * Status messages ("[tool call ignored: ...]", "[invalid tool call: ...]",
 * "[tool call interrupted]") are emitted as NORMAL kind for now; the client
 * paints them as plain text.  A dedicated STATUS token kind is a possible
 * future refinement.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dirent.h>
#include <time.h>
#include <unistd.h>

#include "ds4_agent_utils.h"
#include "ds4_agent_proto.h"
#include "ds4_tool_text.h"
#include "ds4.h"
#include "ds4_kvstore.h"
#include "ds4_tp.h"
#include "ds4_distributed.h"
#include "ds4_prompt_prefix.h"

/* ============================================================================
 * Debug logging (opt-in: DS4_AGENT_DEBUG=1)
 *
 * Three line categories, colored and timestamped:
 *   <cyan>    client -> server   (frames received)
 *   <green>   server -> client   (frames sent; the per-token TOKEN stream is
 *                                skipped so generation does not flood stderr)
 *   <magenta> state / flow       (worker state transitions, dispatch decisions)
 *
 * The timestamp is on the left so a live session reads as a clear timeline.
 * ========================================================================== */

static int dbg_on;

static void dbg_init(void) {
    const char *e = getenv("DS4_AGENT_DEBUG");
    dbg_on = e && e[0] != '\0';
}

static void dbg_stamp(char *buf, size_t len) {
    struct timespec ts;
    struct tm tm;
    clock_gettime(CLOCK_REALTIME, &ts);
    localtime_r(&ts.tv_sec, &tm);
    snprintf(buf, len, "%02d:%02d:%02d.%03ld",
             tm.tm_hour, tm.tm_min, tm.tm_sec,
             (long)(ts.tv_nsec / 1000000));
}

static void dbg_log(char cat, const char *fmt, ...) {
    if (!dbg_on) return;
    char stamp[32], msg[512];
    const char *color = cat == 'C' ? "\x1b[36m" :
                        cat == 'S' ? "\x1b[32m" : "\x1b[35m";
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(msg, sizeof(msg), fmt, ap);
    va_end(ap);
    dbg_stamp(stamp, sizeof(stamp));
    fprintf(stderr, "%s %s%s\x1b[0m\n", stamp, color, msg);
}

static const char *dbg_tag_name(unsigned char tag) {
    switch (tag) {
    case PROTO_C2S_NEW_SESSION: return "NEW_SESSION";
    case PROTO_C2S_USER:        return "USER";
    case PROTO_C2S_TOOL_RESULT: return "TOOL_RESULT";
    case PROTO_C2S_SYSTEM:      return "SYSTEM";
    case PROTO_C2S_STOP_TURN:   return "STOP_TURN";
    case PROTO_C2S_INTERRUPT:   return "INTERRUPT";
    case PROTO_C2S_COMPACT:     return "COMPACT";
    case PROTO_C2S_SAVE:        return "SAVE";
    case PROTO_C2S_LIST:        return "LIST";
    case PROTO_C2S_SWITCH:      return "SWITCH";
    case PROTO_C2S_DEL:         return "DEL";
    case PROTO_C2S_STRIP:       return "STRIP";
    case PROTO_C2S_HISTORY:     return "HISTORY";
    case PROTO_C2S_TOKENS:      return "TOKENS";
    case PROTO_C2S_POWER:       return "POWER";
    case PROTO_C2S_ATTACH_IMAGE: return "ATTACH_IMAGE";
    case PROTO_S2C_HELLO:       return "HELLO";
    case PROTO_S2C_STATUS:      return "STATUS";
    case PROTO_S2C_TOKEN:       return "TOKEN";
    case PROTO_S2C_TURN_PAUSED: return "TURN_PAUSED";
    case PROTO_S2C_TOOL_CALLS:  return "TOOL_CALLS";
    case PROTO_S2C_SWITCH_DONE: return "SWITCH_DONE";
    case PROTO_S2C_COMPACT_DONE: return "COMPACT_DONE";
    case PROTO_S2C_SAVE_DONE:   return "SAVE_DONE";
    case PROTO_S2C_HISTORY:     return "HISTORY";
    case PROTO_S2C_LIST:        return "LIST";
    case PROTO_S2C_COUNT:       return "COUNT";
    case PROTO_S2C_ERROR:       return "ERROR";
    default:                    return "?";
    }
}

static const char *dbg_status_state_name(uint64_t state) {
    switch (state) {
    case AGENT_IDLE:        return "IDLE";
    case AGENT_PREFILL:     return "PREFILL";
    case AGENT_GENERATING:  return "GENERATING";
    case AGENT_COMPACTING:  return "COMPACTING";
    case AGENT_DRAINING:    return "DRAINING";
    case AGENT_SAVING:      return "SAVING";
    case AGENT_ERROR:       return "ERROR";
    case AGENT_STOPPED:     return "STOPPED";
    default:                return "?";
    }
}

static uint64_t dbg_varint_at(const unsigned char *p, size_t len, size_t *adv) {
    uint64_t v = 0;
    int shift = 0;
    size_t i = 0;
    while (i < len && i < 10) {
        unsigned char b = p[i++];
        v |= (uint64_t)(b & 0x7f) << shift;
        if (!(b & 0x80)) break;
        shift += 7;
    }
    if (adv) *adv = i;
    return v;
}
#include "ds4_gpu_args.h"

/* Fullwidth vertical bar (UTF-8 EF BF BC) that frames the DSML marker. */
#define AGENT_DSML_BAR "\xEF\xBF\xBC"

/* ============================================================================
 * Small helpers (self-contained; ported from ds4_agent.c)
 * ========================================================================== */

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static bool streq_any(const char *s, const char *a, const char *b,
                      const char *c, const char *d) {
    return (a && !strcmp(s, a)) || (b && !strcmp(s, b)) ||
           (c && !strcmp(s, c)) || (d && !strcmp(s, d));
}

/* ============================================================================
 * Render-directive sink
 *
 * The server does not paint a terminal.  Instead it records (kind, text)
 * fragments the client renders.  Adjacent fragments of the same kind are
 * coalesced to keep the stream of TOKEN messages small.
 * ========================================================================== */

typedef struct {
    unsigned char kind;
    char *text;
    size_t len;
    size_t cap;
} agent_render_fragment;

typedef struct {
    agent_render_fragment *v;
    size_t len;
    size_t cap;
    bool last_output_newline;
} agent_render_sink;

static void agent_render_sink_write(agent_render_sink *s, unsigned char kind,
                                    const char *text, size_t n) {
    if (!n) return;
    if (s->len && s->v[s->len - 1].kind == kind) {
        agent_render_fragment *f = &s->v[s->len - 1];
        if (f->len + n > f->cap) {
            size_t cap = f->cap ? f->cap * 2 : 256;
            while (cap < f->len + n) cap *= 2;
            f->text = xrealloc(f->text, cap);
            f->cap = cap;
        }
        memcpy(f->text + f->len, text, n);
        f->len += n;
    } else {
        if (s->len == s->cap) {
            s->cap = s->cap ? s->cap * 2 : 16;
            s->v = xrealloc(s->v, s->cap * sizeof(s->v[0]));
        }
        agent_render_fragment *f = &s->v[s->len++];
        f->kind = kind;
        f->text = xmalloc(n);
        memcpy(f->text, text, n);
        f->len = n;
        f->cap = n;
    }
    s->last_output_newline = text[n - 1] == '\n';
}

static void agent_render_sink_free(agent_render_sink *s) __attribute__((unused));
static void agent_render_sink_free(agent_render_sink *s) {
    for (size_t i = 0; i < s->len; i++) free(s->v[i].text);
    free(s->v);
    memset(s, 0, sizeof(*s));
}

#ifdef DS4_AGENT_TEST
/* Concatenated view of all fragments (used by tests). */
static char *agent_render_sink_concat(agent_render_sink *s) {
    size_t total = 0;
    for (size_t i = 0; i < s->len; i++) total += s->v[i].len;
    char *out = xmalloc(total + 1);
    size_t pos = 0;
    for (size_t i = 0; i < s->len; i++) {
        memcpy(out + pos, s->v[i].text, s->v[i].len);
        pos += s->v[i].len;
    }
    out[pos] = '\0';
    return out;
}
#endif /* DS4_AGENT_TEST */

/* ============================================================================
 * Tool-call types (shared proto types)
 * ========================================================================== */

static void agent_tool_call_add_arg(proto_tool_call *c, const char *name,
                                    const char *value, size_t value_len,
                                    bool is_string, const char *end_tag) {
    proto_tool_call_add_arg(c, name, value, value_len, is_string);
    if (is_string) {
        proto_tool_arg *a = &c->args[c->argc - 1];
        ds4_tool_text_unescape(a->value, end_tag);
    }
}

#ifdef DS4_AGENT_TEST
static const char *agent_tool_arg_value(const proto_tool_call *call, const char *name) {
    for (int i = 0; i < call->argc; i++) {
        if (call->args[i].name && !strcmp(call->args[i].name, name))
            return call->args[i].value ? call->args[i].value : "";
    }
    return NULL;
}
#endif /* DS4_AGENT_TEST */

/* ============================================================================
 * DSML / GLM streaming parser (ported as-is)
 * ========================================================================== */

typedef enum {
    AGENT_TOOL_SYNTAX_DSML,
    AGENT_TOOL_SYNTAX_GLM,
} agent_tool_syntax;

typedef enum {
    AGENT_DSML_SEARCH,
    AGENT_DSML_STRUCTURAL,
    AGENT_DSML_PARAM_VALUE,
    AGENT_DSML_DONE,
    AGENT_DSML_ERROR,
} agent_dsml_state;

typedef struct {
    agent_tool_syntax syntax;
    agent_dsml_state state;
    char search_tail[64];
    size_t search_len;
    char *raw;
    size_t raw_len;
    size_t raw_cap;
    size_t parse_pos;
    proto_tool_call current;
    char *param_name;
    bool param_is_string;
    size_t param_value_start;
    bool param_close_prefix;
    bool glm_after_call;
    proto_tool_calls calls;
    char error[160];
} agent_dsml_parser;

static void agent_dsml_parser_free(agent_dsml_parser *p) {
    if (!p) return;
    agent_tool_syntax syntax = p->syntax;
    free(p->raw);
    proto_tool_call_free(&p->current);
    free(p->param_name);
    proto_tool_calls_free(&p->calls);
    memset(p, 0, sizeof(*p));
    p->syntax = syntax;
}

static void agent_dsml_parser_reset(agent_dsml_parser *p) {
    agent_dsml_parser_free(p);
    p->state = AGENT_DSML_SEARCH;
}

static void agent_dsml_raw_append(agent_dsml_parser *p, const char *s, size_t n) {
    if (!n) return;
    if (p->raw_len + n + 1 > p->raw_cap) {
        size_t cap = p->raw_cap ? p->raw_cap * 2 : 512;
        while (cap < p->raw_len + n + 1) cap *= 2;
        p->raw = xrealloc(p->raw, cap);
        p->raw_cap = cap;
    }
    memcpy(p->raw + p->raw_len, s, n);
    p->raw_len += n;
    p->raw[p->raw_len] = '\0';
}

static char *agent_parse_attr(const char *tag, const char *name) {
    char pat[64];
    snprintf(pat, sizeof(pat), "%s=\"", name);
    const char *p = strstr(tag, pat);
    if (!p) return NULL;
    p += strlen(pat);
    const char *end = strchr(p, '"');
    if (!end) return NULL;
    return xstrndup(p, (size_t)(end - p));
}

static void agent_dsml_set_error(agent_dsml_parser *p, const char *msg) {
    p->state = AGENT_DSML_ERROR;
    snprintf(p->error, sizeof(p->error), "%s", msg);
}

static const char *agent_skip_ascii_space(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) p++;
    return p;
}

static void agent_trim_span(const char **p, const char **end) {
    *p = agent_skip_ascii_space(*p, *end);
    while (*end > *p) {
        char c = (*end)[-1];
        if (c != ' ' && c != '\t' && c != '\r' && c != '\n') break;
        (*end)--;
    }
}

static bool agent_dsml_open_tag_is(const char *tag, const char *name) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "%s", name);
    size_t prefix_len = strlen(prefix);
    if (strncmp(tag, prefix, prefix_len) != 0) return false;
    char c = tag[prefix_len];
    return c == '>' || c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static bool agent_dsml_close_tag_at(const char *s, const char *name, size_t *tag_len) {
    char prefix[64];
    static const char dsml_bar[] = AGENT_DSML_BAR;
    snprintf(prefix, sizeof(prefix), "</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "%s", name);
    size_t prefix_len = strlen(prefix);
    if (strncmp(s, prefix, prefix_len) != 0) return false;
    const char *p = s + prefix_len;
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (strncmp(p, dsml_bar, strlen(dsml_bar)) == 0) p += strlen(dsml_bar);
    while (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n') p++;
    if (*p != '>') return false;
    if (tag_len) *tag_len = (size_t)(p - s) + 1;
    return true;
}

/* Recognize a streamed parameter close tag prefix.  Full close detection is
 * handled by agent_dsml_close_tag_at(); this helper exists for online behavior:
 * the producer must hide partial close tags without waiting for the whole
 * parameter to finish. */
static bool agent_dsml_parameter_close_tail(const char *tail, size_t len,
                                            bool *complete) {
    static const char prefix[] = "</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter";
    static const char dsml_bar[] = AGENT_DSML_BAR;
    const size_t prefix_len = sizeof(prefix) - 1;
    const size_t bar_len = sizeof(dsml_bar) - 1;
    *complete = false;
    if (len <= prefix_len) return memcmp(prefix, tail, len) == 0;
    if (memcmp(prefix, tail, prefix_len) != 0) return false;
    size_t i = prefix_len;
    while (i < len && (tail[i] == ' ' || tail[i] == '\t' ||
                       tail[i] == '\r' || tail[i] == '\n')) i++;
    if (i < len && len - i <= bar_len) {
        if (memcmp(dsml_bar, tail + i, len - i) == 0) return true;
    }
    if (i + bar_len <= len && memcmp(tail + i, dsml_bar, bar_len) == 0)
        i += bar_len;
    for (; i < len; i++) {
        if (tail[i] == '>') {
            *complete = i == len - 1;
            return *complete;
        }
        if (tail[i] != ' ' && tail[i] != '\t' && tail[i] != '\r' && tail[i] != '\n')
            return false;
    }
    return true;
}

static bool agent_glm_arg_value_close_tail(const char *tail, size_t len,
                                           bool *complete) {
    static const char close[] = "</arg_value>";
    *complete = false;
    size_t close_len = sizeof(close) - 1;
    if (len <= close_len && memcmp(close, tail, len) == 0) {
        *complete = len == close_len;
        return true;
    }
    return false;
}

static bool agent_tool_value_close_tail(agent_tool_syntax syntax,
                                        const char *tail, size_t len,
                                        bool *complete) {
    if (syntax == AGENT_TOOL_SYNTAX_GLM)
        return agent_glm_arg_value_close_tail(tail, len, complete);
    return agent_dsml_parameter_close_tail(tail, len, complete);
}

static void agent_dsml_update_param_close_prefix(agent_dsml_parser *p) {
    p->param_close_prefix = false;
    if (p->state != AGENT_DSML_PARAM_VALUE || p->raw_len <= p->param_value_start)
        return;

    const char *value = p->raw + p->param_value_start;
    const char *end = p->raw + p->raw_len;
    const char *lt = end;
    while (lt > value) {
        lt--;
        if (*lt == '<') break;
    }
    if (lt < value || *lt != '<') return;

    size_t tail_len = (size_t)(end - lt);
    if (tail_len > 64) return;
    bool complete = false;
    p->param_close_prefix =
        agent_tool_value_close_tail(p->syntax, lt, tail_len, &complete) &&
        !complete;
}

/* Find a DSML closing tag while accepting the few harmless closing-tag variants
 * the model has been observed to emit.  Opening tags stay strict so accidental
 * prose does not become a tool call. */
static char *agent_dsml_find_close_tag(const char *s, const char *name, size_t *tag_len) {
    const char *p = s;
    while ((p = strstr(p, "</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR)) != NULL) {
        if (agent_dsml_close_tag_at(p, name, tag_len)) return (char *)p;
        p++;
    }
    return NULL;
}

static bool agent_bytes_starts_with(const char *p, const char *end,
                                    const char *prefix) {
    size_t n = strlen(prefix);
    return (size_t)(end - p) >= n && memcmp(p, prefix, n) == 0;
}

static bool agent_bytes_partial_prefix_at(const char *p, const char *end,
                                          const char *prefix) {
    return bytes_is_partial_prefix(p, (size_t)(end - p), prefix);
}

static void agent_glm_tool_parse(agent_dsml_parser *p) {
    static const char start[] = "<tool_call>";
    static const char close[] = "</tool_call>";
    static const char arg_key[] = "<arg_key>";
    static const char arg_key_close[] = "</arg_key>";
    static const char arg_value[] = "<arg_value>";
    static const char arg_value_close[] = "</arg_value>";

    if (p->raw_len < sizeof(start) - 1 ||
        memcmp(p->raw, start, sizeof(start) - 1) != 0) {
        return;
    }

    while (p->state == AGENT_DSML_STRUCTURAL ||
           p->state == AGENT_DSML_PARAM_VALUE)
    {
        const char *raw = p->raw;
        const char *end = p->raw + p->raw_len;
        if (p->state == AGENT_DSML_PARAM_VALUE) {
            const char *value_end = strstr(raw + p->param_value_start, arg_value_close);
            if (!value_end) return;
            agent_tool_call_add_arg(&p->current, p->param_name ? p->param_name : "",
                                    raw + p->param_value_start,
                                    (size_t)(value_end - (raw + p->param_value_start)),
                                    true, arg_value_close);
            free(p->param_name);
            p->param_name = NULL;
            p->param_close_prefix = false;
            p->parse_pos = (size_t)(value_end - raw) + sizeof(arg_value_close) - 1;
            p->state = AGENT_DSML_STRUCTURAL;
            continue;
        }

        while (p->parse_pos < p->raw_len &&
               (p->raw[p->parse_pos] == ' ' || p->raw[p->parse_pos] == '\t' ||
                p->raw[p->parse_pos] == '\r' || p->raw[p->parse_pos] == '\n'))
            p->parse_pos++;
        if (p->parse_pos >= p->raw_len) return;

        const char *cur = raw + p->parse_pos;
        if (p->glm_after_call) {
            if (agent_bytes_starts_with(cur, end, start)) {
                p->parse_pos += sizeof(start) - 1;
                p->glm_after_call = false;
                continue;
            }
            if (agent_bytes_partial_prefix_at(cur, end, start)) return;
            p->glm_after_call = false;
            p->state = AGENT_DSML_DONE;
            return;
        }

        if (!p->current.name) {
            const char *name_start = agent_skip_ascii_space(cur, end);
            const char *lt = memchr(name_start, '<', (size_t)(end - name_start));
            if (!lt) return;
            bool at_arg = agent_bytes_starts_with(lt, end, arg_key);
            bool at_close = agent_bytes_starts_with(lt, end, close);
            if (!at_arg && !at_close) {
                if (agent_bytes_partial_prefix_at(lt, end, arg_key) ||
                    agent_bytes_partial_prefix_at(lt, end, close))
                    return;
                agent_dsml_set_error(p, "expected <arg_key> or </tool_call> in GLM tool call");
                return;
            }
            const char *name_end = lt;
            agent_trim_span(&name_start, &name_end);
            if (name_start >= name_end) {
                agent_dsml_set_error(p, "GLM tool call without function name");
                return;
            }
            proto_tool_call_free(&p->current);
            p->current.name = xstrndup(name_start, (size_t)(name_end - name_start));
            p->parse_pos = (size_t)(lt - raw);
            cur = raw + p->parse_pos;
        }

        if (agent_bytes_starts_with(cur, end, close)) {
            p->parse_pos += sizeof(close) - 1;
            proto_tool_calls_push(&p->calls, &p->current);
            p->glm_after_call = true;
            continue;
        }
        if (agent_bytes_partial_prefix_at(cur, end, close)) return;

        if (!agent_bytes_starts_with(cur, end, arg_key)) {
            if (agent_bytes_partial_prefix_at(cur, end, arg_key)) return;
            agent_dsml_set_error(p, "expected <arg_key> in GLM tool call");
            return;
        }
        cur += sizeof(arg_key) - 1;
        const char *key_end_mut = strstr(cur, arg_key_close);
        if (!key_end_mut) return;
        const char *key_start = cur;
        const char *key_end = key_end_mut;
        agent_trim_span(&key_start, &key_end);
        if (key_start >= key_end) {
            agent_dsml_set_error(p, "empty <arg_key> in GLM tool call");
            return;
        }
        char *key = xstrndup(key_start, (size_t)(key_end - key_start));
        ds4_tool_text_unescape(key, arg_key_close);
        cur = key_end_mut + sizeof(arg_key_close) - 1;
        cur = agent_skip_ascii_space(cur, end);
        if (!agent_bytes_starts_with(cur, end, arg_value)) {
            if (agent_bytes_partial_prefix_at(cur, end, arg_value)) {
                free(key);
                return;
            }
            free(key);
            agent_dsml_set_error(p, "expected <arg_value> in GLM tool call");
            return;
        }
        cur += sizeof(arg_value) - 1;
        free(p->param_name);
        p->param_name = key;
        p->param_is_string = true;
        p->param_value_start = (size_t)(cur - raw);
        p->parse_pos = p->param_value_start;
        p->param_close_prefix = false;
        p->state = AGENT_DSML_PARAM_VALUE;
    }
}

static void agent_dsml_finish(agent_dsml_parser *p) {
    if (!p || p->state == AGENT_DSML_DONE || p->state == AGENT_DSML_ERROR)
        return;
    if (p->syntax != AGENT_TOOL_SYNTAX_GLM || !p->glm_after_call)
        return;

    while (p->parse_pos < p->raw_len &&
           (p->raw[p->parse_pos] == ' ' || p->raw[p->parse_pos] == '\t' ||
            p->raw[p->parse_pos] == '\r' || p->raw[p->parse_pos] == '\n'))
        p->parse_pos++;
    if (p->parse_pos >= p->raw_len) {
        p->glm_after_call = false;
        p->state = AGENT_DSML_DONE;
    }
}

/* Parse as much of the accumulated DSML buffer as possible.  The parser can be
 * called after every streamed byte: incomplete input leaves state unchanged
 * until enough bytes arrive, while malformed completed input switches to
 * AGENT_DSML_ERROR so the model gets a retryable tool error. */
static void agent_dsml_parse(agent_dsml_parser *p) {
    if (p->syntax == AGENT_TOOL_SYNTAX_GLM) {
        agent_glm_tool_parse(p);
        return;
    }

    while (p->state == AGENT_DSML_STRUCTURAL || p->state == AGENT_DSML_PARAM_VALUE) {
        if (p->state == AGENT_DSML_PARAM_VALUE) {
            size_t end_tag_len = 0;
            char *end = agent_dsml_find_close_tag(p->raw + p->param_value_start,
                                                  "parameter", &end_tag_len);
            if (!end) return;
            agent_tool_call_add_arg(&p->current, p->param_name ? p->param_name : "",
                                    p->raw + p->param_value_start,
                                    (size_t)(end - (p->raw + p->param_value_start)),
                                    p->param_is_string, "</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter>");
            p->param_close_prefix = false;
            free(p->param_name);
            p->param_name = NULL;
            p->parse_pos = (size_t)(end - p->raw) + end_tag_len;
            p->state = AGENT_DSML_STRUCTURAL;
            continue;
        }

        while (p->parse_pos < p->raw_len &&
               (p->raw[p->parse_pos] == ' ' || p->raw[p->parse_pos] == '\t' ||
                p->raw[p->parse_pos] == '\r' || p->raw[p->parse_pos] == '\n'))
            p->parse_pos++;
        if (p->parse_pos >= p->raw_len) return;

        size_t close_len = 0;
        if (agent_dsml_close_tag_at(p->raw + p->parse_pos, "tool_calls", &close_len)) {
            proto_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            p->state = AGENT_DSML_DONE;
            return;
        }
        if (agent_dsml_close_tag_at(p->raw + p->parse_pos, "invoke", &close_len)) {
            proto_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            continue;
        }

        char *tag_end = strchr(p->raw + p->parse_pos, '>');
        if (!tag_end) return;
        size_t tag_len = (size_t)(tag_end - (p->raw + p->parse_pos)) + 1;
        char *tag = xstrndup(p->raw + p->parse_pos, tag_len);

        if (agent_dsml_open_tag_is(tag, "invoke")) {
            proto_tool_call_free(&p->current);
            p->current.name = agent_parse_attr(tag, "name");
            if (!p->current.name) {
                free(tag);
                agent_dsml_set_error(p, "tool invoke without name");
                return;
            }
            p->parse_pos += tag_len;
        } else if (agent_dsml_open_tag_is(tag, "parameter")) {
            free(p->param_name);
            p->param_name = agent_parse_attr(tag, "name");
            char *is_string = agent_parse_attr(tag, "string");
            p->param_is_string = is_string && !strcmp(is_string, "true");
            free(is_string);
            if (!p->param_name) {
                free(tag);
                agent_dsml_set_error(p, "tool parameter without name");
                return;
            }
            p->parse_pos += tag_len;
            p->param_value_start = p->parse_pos;
            p->param_close_prefix = false;
            p->state = AGENT_DSML_PARAM_VALUE;
        } else {
            snprintf(p->error, sizeof(p->error), "unexpected DSML tag: %.*s",
                     (int)(tag_len > 80 ? 80 : tag_len), tag);
            free(tag);
            p->state = AGENT_DSML_ERROR;
            return;
        }
        free(tag);
    }
}

static void agent_dsml_start(agent_dsml_parser *p) {
    const char *start = p->syntax == AGENT_TOOL_SYNTAX_GLM ?
        "<tool_call>" : "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>";
    p->state = AGENT_DSML_STRUCTURAL;
    p->search_len = 0;
    agent_dsml_raw_append(p, start, strlen(start));
    p->parse_pos = strlen(start);
}

static void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n) {
    const char *start = p->syntax == AGENT_TOOL_SYNTAX_GLM ?
        "<tool_call>" : "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>";
    const size_t start_len = strlen(start);
    if (p->state == AGENT_DSML_DONE || p->state == AGENT_DSML_ERROR) return;

    for (size_t i = 0; i < n; i++) {
        char c = s[i];
        if (p->state == AGENT_DSML_SEARCH) {
            if (p->search_len == sizeof(p->search_tail)) {
                memmove(p->search_tail, p->search_tail + 1, --p->search_len);
            }
            p->search_tail[p->search_len++] = c;
            if (p->search_len >= start_len &&
                memcmp(p->search_tail + p->search_len - start_len, start, start_len) == 0)
                agent_dsml_start(p);
            continue;
        }

        agent_dsml_raw_append(p, &c, 1);
        agent_dsml_parse(p);
        if (p->state == AGENT_DSML_PARAM_VALUE)
            agent_dsml_update_param_close_prefix(p);
        else
            p->param_close_prefix = false;
    }
}

/* ============================================================================
 * Streaming render-directive producer (ported from agent_tool_viz_* + agent_stream_*)
 * ========================================================================== */

typedef enum {
    AGENT_TOOL_PARAM_NORMAL,
    AGENT_TOOL_PARAM_PATH,
    AGENT_TOOL_PARAM_OFFSET,
    AGENT_TOOL_PARAM_CONTENT,
    AGENT_TOOL_PARAM_DIFF_OLD,
    AGENT_TOOL_PARAM_DIFF_NEW,
    AGENT_TOOL_PARAM_BASH_COMMAND,
} agent_tool_param_kind;

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
} agent_tool_visualizer;

typedef struct {
    char tail[32];
    size_t len;
} agent_dsml_marker_detector;

typedef struct {
    agent_render_sink sink;
    agent_dsml_parser *parser;
    agent_tool_syntax syntax;
    agent_tool_visualizer viz;
    bool in_think;
    bool dsml_active;
    bool dsml_ignored;
    bool replay;
    char pending[16];
    size_t pending_len;
    char dsml_start_tail[64];
    size_t dsml_start_len;
    agent_dsml_marker_detector plain_dsml;
    agent_dsml_marker_detector think_dsml;
    bool dsml_in_think;
    bool dsml_in_think_reported;
    bool post_think_gap;
} agent_stream_renderer;

/* Write one assistant byte with the kind the client should paint it with. */
static void agent_stream_write_char(agent_stream_renderer *sr, char c) {
    unsigned char kind = sr->in_think ? PROTO_TOKEN_THINK : PROTO_TOKEN_NORMAL;
    agent_render_sink_write(&sr->sink, kind, &c, 1);
}

static void agent_tool_viz_write(agent_stream_renderer *sr, unsigned char kind,
                                 const char *s, size_t n) {
    agent_render_sink_write(&sr->sink, kind, s, n);
    for (size_t i = 0; i < n; i++) sr->viz.last_output_newline = s[i] == '\n';
}

static void agent_tool_viz_puts(agent_stream_renderer *sr, unsigned char kind,
                                const char *s) {
    agent_tool_viz_write(sr, kind, s, strlen(s));
}

static void agent_tool_viz_start(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    bool line_open = !sr->sink.last_output_newline;
    memset(v, 0, sizeof(*v));
    v->active = true;
    v->at_line_start = true;
    v->last_output_newline = true;
    if (line_open) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
    v->last_output_newline = true;
}

static void agent_tool_viz_line_prefix(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->last_output_newline) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
    v->at_line_start = false;
}

static void agent_tool_viz_tool(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->tool_announced && !strcmp(v->tool_name, name)) return;
    if (v->tool_announced && !v->last_output_newline) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
    snprintf(v->tool_name, sizeof(v->tool_name), "%s", name ? name : "tool");
    v->tool_announced = true;
    v->read_style = !strcmp(v->tool_name, "read");
    agent_tool_viz_line_prefix(sr);
    if (v->read_style) {
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_NAME, "Reading ");
        v->read_prefix_rendered = true;
        return;
    }
    agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_NAME, v->tool_name);
    agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_NAME, " ");
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
        if (v->read_prefix_rendered)
            agent_tool_viz_write(sr, PROTO_TOKEN_TOOL_PARAM_VALUE, &c, 1);
    } else if (!strcmp(v->param_name, "start_line")) {
        agent_tool_viz_append(v->read_start, sizeof(v->read_start), c);
    } else if (!strcmp(v->param_name, "max_lines")) {
        agent_tool_viz_append(v->read_max, sizeof(v->read_max), c);
    } else if (!strcmp(v->param_name, "whole")) {
        agent_tool_viz_append(v->read_whole, sizeof(v->read_whole), c);
    }
}

/* Semantic read summary ("Reading <path> <start>:<max>...") that the client
 * paints.  The path bytes are already streamed during the parameter; this only
 * adds the range suffix once the call is complete. */
static void agent_tool_viz_render_read(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->read_style || v->read_line_rendered) return;

    if (!v->read_prefix_rendered) {
        agent_tool_viz_line_prefix(sr);
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_NAME, "Reading ");
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE,
                            v->read_path[0] ? v->read_path : "<unknown>");
    } else if (!v->read_path[0]) {
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE, "<unknown>");
    }
    bool whole = v->read_whole[0] && !strcmp(v->read_whole, "true");
    if (whole && (!v->read_start[0] || !strcmp(v->read_start, "1"))) {
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE, " (whole file)");
    } else if (whole) {
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE, " ");
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE, v->read_start);
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE, ":EOF");
    } else {
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE, " ");
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE,
                            v->read_start[0] ? v->read_start : "1");
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE, ":");
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_VALUE,
                            v->read_max[0] ? v->read_max : "500");
    }
    agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_NAME, "...");
    agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
    v->read_line_rendered = true;
}

static void agent_tool_viz_param_begin(agent_stream_renderer *sr, const char *name) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && sr->parser->current.name)
        agent_tool_viz_tool(sr, sr->parser->current.name);
    snprintf(v->param_name, sizeof(v->param_name), "%s", name ? name : "");
    v->param_kind = agent_tool_param_kind_for(v->tool_name, v->param_name);
    v->param_active = true;
    v->param_end_len = 0;

    if (v->read_style) return;

    if (v->param_kind == AGENT_TOOL_PARAM_DIFF_OLD ||
        v->param_kind == AGENT_TOOL_PARAM_DIFF_NEW)
    {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
        v->at_line_start = true;
        return;
    }

    if (v->param_kind == AGENT_TOOL_PARAM_CONTENT) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
        if (strcmp(v->tool_name, "write")) {
            agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_NAME, v->param_name);
            agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_NAME, ":\n");
        }
        v->at_line_start = true;
        return;
    }

    if (v->param_kind != AGENT_TOOL_PARAM_BASH_COMMAND) {
        if (!v->at_line_start) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, " ");
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_NAME, v->param_name);
        agent_tool_viz_puts(sr, PROTO_TOKEN_TOOL_PARAM_NAME, "=");
    }
}

static void agent_tool_viz_param_end(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    v->param_end_len = 0;
    v->param_active = false;
    v->param_name[0] = '\0';
}

static void agent_tool_viz_param_raw_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;
    if (v->read_style) {
        agent_tool_viz_read_value_byte(sr, c);
        return;
    }
    agent_tool_viz_write(sr, PROTO_TOKEN_TOOL_PARAM_VALUE, &c, 1);
    v->at_line_start = c == '\n';
}

/* Stream one DSML parameter byte into the producer.  The producer must not wait
 * for the whole parameter: large write/edit contents should show progress as
 * the model emits them, while still detecting the closing parameter tag. */
static void agent_tool_viz_param_value_byte(agent_stream_renderer *sr, char c) {
    agent_tool_visualizer *v = &sr->viz;

    if (v->param_end_len || c == '<') {
        if (v->param_end_len == sizeof(v->param_end_tail)) {
            size_t keep = v->param_end_len;
            v->param_end_len = 0;
            for (size_t i = 0; i < keep; i++)
                agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
            if (c != '<') {
                agent_tool_viz_param_raw_byte(sr, c);
                return;
            }
        }
        if (v->param_end_len < sizeof(v->param_end_tail))
            v->param_end_tail[v->param_end_len++] = c;
        bool complete = false;
        if (agent_tool_value_close_tail(sr->parser->syntax,
                                        v->param_end_tail,
                                        v->param_end_len,
                                        &complete)) {
            if (complete) agent_tool_viz_param_end(sr);
            return;
        }
        size_t keep = v->param_end_len;
        v->param_end_len = 0;
        for (size_t i = 0; i < keep; i++)
            agent_tool_viz_param_raw_byte(sr, v->param_end_tail[i]);
        return;
    }
    agent_tool_viz_param_raw_byte(sr, c);
}

static void agent_tool_viz_finish(agent_stream_renderer *sr, const char *status) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;
    if (v->param_active) agent_tool_viz_param_end(sr);
    if (!status || !status[0]) agent_tool_viz_render_read(sr);
    if (status && status[0]) {
        if (!v->last_output_newline) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
        agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, status);
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
    v->active = false;
}

static void agent_tool_viz_dump_invalid_dsml(agent_stream_renderer *sr) {
    agent_tool_visualizer *v = &sr->viz;
    if (!v->active) return;

    /* The normal path hides DSML and paints a friendly semantic projection.  If
     * parsing fails, show the exact bytes we rejected so the next fix is based
     * on evidence instead of guessing from the projection. */
    if (v->param_active) {
        v->param_active = false;
        v->param_end_len = 0;
        v->param_name[0] = '\0';
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
    if (sr->parser->raw && sr->parser->raw_len) {
        agent_tool_viz_write(sr, PROTO_TOKEN_NORMAL, sr->parser->raw, sr->parser->raw_len);
    } else {
        agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "<empty DSML>");
    }
    if (!v->last_output_newline) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
}

static void agent_stream_finish_ignored_dsml(agent_stream_renderer *sr, const char *detail) {
    const char *msg =
        detail && detail[0] ? detail :
        "tool calling is not allowed inside <think></think>";
    sr->dsml_in_think = true;
    sr->dsml_in_think_reported = true;
    if (!sr->sink.last_output_newline)
        agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
    agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "[tool call ignored: ");
    agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, msg);
    agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "]\n");
    agent_dsml_parser_reset(sr->parser);
    sr->dsml_active = false;
    sr->dsml_ignored = false;
}

static void agent_stream_malformed_dsml(agent_stream_renderer *sr,
                                        const char *detail) {
    const char *msg = detail && detail[0] ? detail :
        "DSML markup outside a valid tool_calls block";
    if (sr->parser->state == AGENT_DSML_ERROR) return;
    agent_dsml_set_error(sr->parser, msg);
    if (!sr->sink.last_output_newline)
        agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
    agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "[invalid tool call: ");
    agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, msg);
    agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "]\n");
}

/* Mirror parser progress into the render-directive producer.  Parser state is
 * the source of truth; this function only decides what the client should see. */
static void agent_stream_tool_events(agent_stream_renderer *sr) {
    agent_dsml_parser *p = sr->parser;
    agent_tool_visualizer *v = &sr->viz;
    if (!v->tool_announced && p->current.name)
        agent_tool_viz_tool(sr, p->current.name);
    if (v->tool_announced && !p->current.name && !v->param_active) {
        agent_tool_viz_render_read(sr);
        if (!v->last_output_newline) agent_tool_viz_puts(sr, PROTO_TOKEN_NORMAL, "\n");
        v->read_style = false;
        v->read_prefix_rendered = false;
        v->read_line_rendered = false;
        v->read_path[0] = '\0';
        v->read_start[0] = '\0';
        v->read_max[0] = '\0';
        v->read_whole[0] = '\0';
        v->tool_announced = false;
    }
    if (!v->param_active && p->state == AGENT_DSML_PARAM_VALUE && p->param_name)
        agent_tool_viz_param_begin(sr, p->param_name);
}

static void agent_stream_feed_dsml_byte(agent_stream_renderer *sr, char c) {
    bool was_param = !sr->dsml_ignored && sr->viz.param_active;
    agent_dsml_feed(sr->parser, &c, 1);
    if (!sr->dsml_ignored) {
        agent_stream_tool_events(sr);
        if (was_param) agent_tool_viz_param_value_byte(sr, c);
        if (was_param && sr->parser->state != AGENT_DSML_PARAM_VALUE &&
            sr->viz.param_active)
        {
            agent_tool_viz_param_end(sr);
        }
    }
    if (sr->parser->state == AGENT_DSML_DONE) {
        if (sr->dsml_ignored) {
            agent_stream_finish_ignored_dsml(
                sr, "tool calling is not allowed inside <think></think>");
        } else {
            agent_tool_viz_finish(sr, NULL);
            sr->dsml_active = false;
        }
    } else if (sr->parser->state == AGENT_DSML_ERROR) {
        if (sr->dsml_ignored) {
            agent_stream_finish_ignored_dsml(
                sr, "malformed tool call inside <think></think>");
        } else {
            char status[220];
            snprintf(status, sizeof(status), "[invalid tool call: %s]\n",
                     sr->parser->error[0] ? sr->parser->error : "parse error");
            agent_tool_viz_dump_invalid_dsml(sr);
            agent_tool_viz_finish(sr, status);
            sr->dsml_active = false;
        }
    }
}

/* Start a DSML block from the streaming detector.  The detector may accept a
 * known malformed opening form for robustness, but the parser is seeded with
 * canonical bytes so all later parsing remains strict. */
static void agent_stream_start_dsml(agent_stream_renderer *sr, bool ignored) {
    sr->dsml_active = true;
    sr->dsml_ignored = ignored;
    if (ignored) sr->dsml_in_think = true;
    sr->dsml_start_len = 0;
    sr->post_think_gap = false;
    agent_dsml_start(sr->parser);
    if (!ignored) {
        agent_tool_viz_start(sr);
        agent_stream_tool_events(sr);
    }
}

static void agent_stream_note_plain_dsml_byte(agent_stream_renderer *sr, char c);

static void agent_stream_flush_start_tail(agent_stream_renderer *sr) {
    if (!sr->dsml_start_len) return;
    sr->post_think_gap = false;
    for (size_t i = 0; i < sr->dsml_start_len; i++) {
        agent_stream_write_char(sr, sr->dsml_start_tail[i]);
        agent_stream_note_plain_dsml_byte(sr, sr->dsml_start_tail[i]);
        if (sr->parser->state == AGENT_DSML_ERROR) break;
    }
    sr->dsml_start_len = 0;
}

static bool agent_stream_dsml_start_match(agent_tool_syntax syntax,
                                          const char *tail, size_t len,
                                          bool *complete,
                                          bool *implicit_invoke) {
    if (syntax == AGENT_TOOL_SYNTAX_GLM) {
        static const char glm_call[] = "<tool_call>";
        size_t form_len = sizeof(glm_call) - 1;
        *complete = false;
        *implicit_invoke = false;
        if (len <= form_len && memcmp(glm_call, tail, len) == 0) {
            *complete = len == form_len;
            return true;
        }
        return false;
    }

    static const char canonical[] = "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>";
    static const char missing_bar[] = "<DSML" AGENT_DSML_BAR "tool_calls>";
    static const char invoke[] = "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke";
    static const char invoke_missing_bar[] = "<DSML" AGENT_DSML_BAR "invoke";
    struct {
        const char *text;
        bool implicit_invoke;
    } forms[] = {
        {canonical, false},
        {missing_bar, false},
        {invoke, true},
        {invoke_missing_bar, true},
    };
    *complete = false;
    *implicit_invoke = false;
    for (size_t i = 0; i < sizeof(forms)/sizeof(forms[0]); i++) {
        size_t form_len = strlen(forms[i].text);
        if (len <= form_len && memcmp(forms[i].text, tail, len) == 0) {
            *complete = len == form_len;
            *implicit_invoke = forms[i].implicit_invoke;
            return true;
        }
    }
    return false;
}

static bool agent_tail_matches(const char *tail, size_t len,
                               const char *needle, size_t needle_len) {
    return len >= needle_len &&
           memcmp(tail + len - needle_len, needle, needle_len) == 0;
}

/* Detect DSML-looking control markers in text that is not currently owned by
 * the executable DSML parser.  This helper intentionally has no policy: inside
 * <think> the marker means "tool call attempted too early", while in normal
 * assistant output it means malformed DSML that the model should see as a tool
 * error. */
static bool agent_dsml_marker_detector_feed(agent_dsml_marker_detector *d,
                                            char c) {
    if (d->len == sizeof(d->tail)) {
        memmove(d->tail, d->tail + 1, sizeof(d->tail) - 1);
        d->len--;
    }
    d->tail[d->len++] = c;

    static const char fullwidth_marker[] = AGENT_DSML_BAR "DSML" AGENT_DSML_BAR;
    static const char ascii_marker[] = "|DSML|";
    static const char missing_open[] = "<DSML" AGENT_DSML_BAR;
    static const char missing_close[] = "</DSML" AGENT_DSML_BAR;
    return agent_tail_matches(d->tail, d->len,
                              fullwidth_marker, sizeof(fullwidth_marker) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              ascii_marker, sizeof(ascii_marker) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              missing_open, sizeof(missing_open) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              missing_close, sizeof(missing_close) - 1);
}

static void agent_stream_note_thinking_dsml_byte(agent_stream_renderer *sr,
                                                 char c) {
    if (!sr->in_think || sr->dsml_in_think) return;
    if (agent_dsml_marker_detector_feed(&sr->think_dsml, c))
        sr->dsml_in_think = true;
}

static void agent_stream_note_plain_dsml_byte(agent_stream_renderer *sr,
                                              char c) {
    if (sr->parser->state == AGENT_DSML_ERROR) return;
    if (sr->dsml_active || sr->in_think || sr->dsml_in_think) return;
    if (agent_dsml_marker_detector_feed(&sr->plain_dsml, c)) {
        agent_stream_malformed_dsml(
            sr, "DSML markup outside a valid tool_calls block");
    }
}

/* Route ordinary assistant bytes either to normal rendering or into the DSML
 * detector.  The detector must hold short prefixes because the model can split
 * the tool_calls marker across arbitrary tokens. */
static void agent_stream_normal_byte(agent_stream_renderer *sr, char c) {
    static const char canonical_invoke[] = "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke";
    const char *start = sr->syntax == AGENT_TOOL_SYNTAX_GLM ?
        "<tool_call>" : "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>";
    if (sr->parser->state == AGENT_DSML_ERROR) return;
    agent_stream_note_thinking_dsml_byte(sr, c);

    /* DeepSeek usually emits one or more blank lines after </think> before
     * either prose or a DSML tool stanza.  At that point the bytes are just a
     * visual gap between the hidden thinking phase and the real answer, and
     * printing them makes tool calls appear after odd empty lines.  We only
     * suppress whitespace in this very narrow post-thinking window; once the
     * first non-space byte arrives, normal rendering resumes. */
    if (sr->post_think_gap &&
        (c == ' ' || c == '\t' || c == '\r' || c == '\n'))
    {
        return;
    }

    if (sr->dsml_start_len || c == start[0]) {
        if (sr->dsml_start_len < sizeof(sr->dsml_start_tail))
            sr->dsml_start_tail[sr->dsml_start_len++] = c;
        bool complete = false, implicit_invoke = false;
        if (agent_stream_dsml_start_match(sr->syntax,
                                          sr->dsml_start_tail, sr->dsml_start_len,
                                          &complete, &implicit_invoke))
        {
            if (complete) {
                /* Accept the common missing-leading-bar typo
                 * "<DSML｜tool_calls>" here, but seed the parser with the
                 * canonical marker so the rest of the DSML parser stays
                 * strict and simple.  Also accept a direct invoke opener as an
                 * implicit tool_calls block; the model often knows it wants a
                 * tool but forgets the outer wrapper. */
                agent_stream_start_dsml(sr, sr->in_think);
                if (sr->syntax == AGENT_TOOL_SYNTAX_DSML && implicit_invoke) {
                    for (size_t i = 0; i < sizeof(canonical_invoke) - 1; i++)
                        agent_stream_feed_dsml_byte(sr, canonical_invoke[i]);
                }
            }
            return;
        }
        if (sr->dsml_start_len > 1 &&
            sr->dsml_start_tail[sr->dsml_start_len - 1] == start[0])
        {
            sr->post_think_gap = false;
            size_t flush = sr->dsml_start_len - 1;
            for (size_t i = 0; i < flush; i++) {
                agent_stream_write_char(sr, sr->dsml_start_tail[i]);
                agent_stream_note_plain_dsml_byte(sr, sr->dsml_start_tail[i]);
                if (sr->parser->state == AGENT_DSML_ERROR) break;
            }
            if (sr->parser->state == AGENT_DSML_ERROR) {
                sr->dsml_start_len = 0;
                return;
            }
            sr->dsml_start_tail[0] = start[0];
            sr->dsml_start_len = 1;
            return;
        }
        agent_stream_flush_start_tail(sr);
        return;
    }

    sr->post_think_gap = false;
    agent_stream_write_char(sr, c);
    agent_stream_note_plain_dsml_byte(sr, c);
}

/* This is the single streaming producer state machine for assistant output.  It
 * hides raw DSML as soon as the tool_calls marker is complete, lets the DSML
 * parser continue building executable calls, and emits render directives
 * (kind, text) from parser state changes.  The sampled transcript remains
 * unchanged: only the projection is rewritten. */
static void agent_stream_text(agent_stream_renderer *sr, const char *text, size_t len, bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = sr->pending_len + len;
    char *buf = xmalloc(total ? total : 1);
    if (sr->pending_len) memcpy(buf, sr->pending, sr->pending_len);
    if (len) memcpy(buf + sr->pending_len, text, len);
    sr->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        char *cur = buf + i;
        size_t rem = total - i;
        if (!sr->dsml_active && bytes_has_prefix(cur, rem, think_open)) {
            agent_stream_flush_start_tail(sr);
            sr->post_think_gap = false;
            sr->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (!sr->dsml_active && bytes_has_prefix(cur, rem, think_close)) {
            agent_stream_flush_start_tail(sr);
            sr->in_think = false;
            if (!sr->sink.last_output_newline)
                agent_stream_write_char(sr, '\n');
            agent_stream_write_char(sr, '\n');
            sr->sink.last_output_newline = true;
            sr->post_think_gap = true;
            i += strlen(think_close);
            continue;
        }
        if (!finish && !sr->dsml_active && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close)))
        {
            if (rem < sizeof(sr->pending)) {
                memcpy(sr->pending, cur, rem);
                sr->pending_len = rem;
            }
            break;
        }

        if (sr->dsml_active) {
            agent_stream_feed_dsml_byte(sr, cur[0]);
        } else {
            /* Tool calls are executable only after thinking has closed.  Still
             * route thinking bytes through the DSML start detector so an
             * accidental in-think tool stanza can be suppressed cleanly instead
             * of being shown as raw markup or, worse, executed. */
            agent_stream_normal_byte(sr, cur[0]);
        }
        i++;
    }
    free(buf);

    if (finish) {
        agent_stream_flush_start_tail(sr);
        sr->post_think_gap = false;
        if (sr->dsml_active) agent_dsml_finish(sr->parser);
        if (sr->dsml_active) {
            if (sr->parser->state == AGENT_DSML_DONE) {
                if (sr->dsml_ignored) {
                    agent_stream_finish_ignored_dsml(
                        sr, "tool calling is not allowed inside <think></think>");
                } else {
                    agent_tool_viz_finish(sr, NULL);
                    sr->dsml_active = false;
                }
            } else if (sr->dsml_ignored) {
                agent_stream_finish_ignored_dsml(
                    sr, "tool calling is not allowed inside <think></think>");
            } else {
                agent_tool_viz_finish(sr, "[tool call interrupted]\n");
                sr->dsml_active = false;
            }
        }
        if (sr->dsml_in_think && !sr->dsml_in_think_reported) {
            agent_stream_finish_ignored_dsml(
                sr, "tool calling is not allowed inside <think></think>");
        }
    }
}

/* ============================================================================
 * Greedy sampling boundary
 *
 * This helper is intentionally derived only from the current streaming parser
 * state.  The state object is local to one assistant round, so malformed output,
 * EOS, Ctrl+C, or the next turn cannot accidentally leave sampling greedy.
 * ========================================================================== */

static bool agent_stream_wants_greedy_sampling(const agent_stream_renderer *sr) {
    if (!sr || !sr->parser) return false;
    if (sr->parser->state == AGENT_DSML_ERROR ||
        sr->parser->state == AGENT_DSML_DONE)
        return false;

    /* A possible opening marker is being held back by the start detector.  A
     * single '<' is too common in prose/code to justify forcing argmax; after
     * the second byte, the buffered prefix still matching here is specifically
     * DSML-shaped ("<｜..." or the tolerated "<D..." typo). */
    if (sr->dsml_start_len > 1) return true;
    if (!sr->dsml_active) return false;

    if (sr->parser->state == AGENT_DSML_STRUCTURAL)
        return true;
    if (sr->parser->state != AGENT_DSML_PARAM_VALUE)
        return false;

    return sr->parser->param_close_prefix;
}

/* ============================================================================
 * ds4-agent-server: Task 3 skeleton
 *
 * Owns the engine, DS4 session, KV cache, full transcript, disk persistence,
 * compaction, sampling, DSML/GLM parsing + tool-call orchestration, status
 * push.  Tools execution and the terminal are client-side.
 *
 * Two threads:
 *   - worker thread: engine loop (turns, compaction, session ops).  It is the
 *     only writer of framed S2C messages to the socket.
 *   - reader thread: reads framed C2S messages and updates worker state under
 *     the mutex (interrupt, stop_turn, tool_result, queued user, power, ...),
 *     signalling the worker cond.  It also handles quick non-engine ops (LIST,
 *     DEL, STRIP, TOKENS).
 *
 * One session per connection; on disconnect the worker stops and the engine is
 * torn down.
 * ========================================================================== */

#define AGENT_SYSTEM_PROMPT_REMINDER_TOKENS 50000
#define AGENT_COMPACT_SOFT_PERCENT 85
#define AGENT_COMPACT_MIN_FREE_TOKENS 8192
#define AGENT_COMPACT_TAIL_DIVISOR 10
#define AGENT_COMPACT_TAIL_CAP_TOKENS 50000
#define AGENT_COMPACT_SUMMARY_MAX_TOKENS 4096
#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024
#define AGENT_STATUS_PUSH_INTERVAL 0.10

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

static const char *agent_state_name(agent_worker_state state) {
    switch (state) {
    case AGENT_WORKER_IDLE:        return "idle";
    case AGENT_WORKER_PREFILL:     return "prefill";
    case AGENT_WORKER_GENERATING:  return "generating";
    case AGENT_WORKER_COMPACTING:  return "compacting";
    case AGENT_WORKER_DRAINING:    return "draining";
    case AGENT_WORKER_SAVING:      return "saving";
    case AGENT_WORKER_ERROR:       return "error";
    case AGENT_WORKER_STOPPED:     return "stopped";
    default:                       return "unknown";
    }
}

typedef struct {
    agent_worker_state state;
    int prefill_done;
    int prefill_total;
    double prefill_tps;
    int generated;
    double gen_tps;
    bool greedy_sampling;
    int ctx_used;
    int ctx_size;
    int power_percent;
    char error[256];
} agent_status;

typedef struct {
    const char *system;
    ds4_prompt_prefix prefix;
    const char *trace_path;
    int n_predict;
    int ctx_size;
    float temperature;
    float top_p;
    float min_p;
    bool temperature_set;
    bool top_p_set;
    bool min_p_set;
    uint64_t seed;
    ds4_think_mode think_mode;
} agent_generation_options;

typedef struct {
    ds4_engine_options engine;
    agent_generation_options gen;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    const char *host;
    int port;
} agent_config;

typedef struct {
    ds4_engine *engine;
    agent_config *cfg;
    ds4_session *session;
    ds4_tokens transcript;
    ds4_vision_span *images;
    size_t image_count;
    size_t image_cap;
    char *cache_dir;
    char *sysprompt_path;
    char session_sha[41];
    char *session_title;
    uint64_t session_created_at;
    char *legacy_session_path_to_delete;
    bool user_activity;
    bool session_dirty;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cond;
    bool stop;
    bool interrupt;
    bool initialized;
    bool save_requested;
    bool compact_requested;
    bool power_requested;
    int requested_power;
    int progress_base;
    bool progress_direct;
    double progress_started_at;
    char *cmd_text;
    bool cmd_system;
    agent_status status;
    double last_status_push_at;
    int sock_fd;
    bool tool_result_pending;
    bool tool_result_answered;
    char *tool_result_text;
    unsigned char *tool_result_image;
    size_t tool_result_image_len;
    bool queued_user_pending;
    bool queued_user_answered;
    char *queued_user_text;
    bool datetime_context_injected;
    int last_system_prompt_reminder_at;
    uint64_t token_seq;
} agent_worker;

static agent_tool_syntax agent_tool_syntax_for_engine(ds4_engine *engine) {
    return ds4_engine_is_glm_dsa(engine) ? AGENT_TOOL_SYNTAX_GLM
                                         : AGENT_TOOL_SYNTAX_DSML;
}

static bool agent_tool_syntax_assistant_turn_uses_eos(agent_tool_syntax syntax) {
    return syntax != AGENT_TOOL_SYNTAX_GLM;
}

static ds4_think_mode effective_think_mode(const agent_config *cfg) {
    return ds4_think_mode_for_context(cfg->gen.think_mode, cfg->gen.ctx_size);
}

static void agent_apply_model_sampling_defaults(
        ds4_engine *engine, agent_generation_options *gen) {
    if (!engine || !gen || !ds4_engine_is_glm_dsa(engine)) return;
    if (!gen->temperature_set) gen->temperature = 1.0f;
    if (!gen->top_p_set) gen->top_p = 0.95f;
    if (!gen->min_p_set) gen->min_p = 0.0f;
}

/* ---- Prompt text (ported from ds4_agent.c; edit-upto variants dropped).
 *      The DSML markers are written as AGENT_DSML_BAR macro concatenations so
 *      this file stays ASCII-only; the compiler resolves them to the marker. */

#define AGENT_TOOL_CONTRACTS \
    "Read output is limited to 128 KiB. Use more to continue, including within an oversized line; " \
    "whole=true fails rather than returning an excerpt. raw=true omits line numbers but still reports truncation.\n" \
    "Search mode is literal (default) or regex (POSIX extended). Defaults: path=., case_sensitive=true, context=0, max_results=50. " \
    "context is 0-5, max_results is 1-500. The search skips .git and nested symlinks, and reports incomplete coverage.\n" \
    "bash timeout_sec defaults to 3600; refresh_sec defaults to 60 and waits up to that many seconds. " \
    "bash_status returns immediately unless refresh_sec is given. Jobs keep running and timeouts remain active between calls.\n" \
    "Write and edit replace complete files atomically, follow existing symlink targets, and reject hard-linked files.\n\n"

static const char agent_tools_prompt_intro[] =
    "You are a coding agent running in a local workspace. Use tools for local file and system work. "
    "Avoid printing large file contents or large code blocks as answers; create or edit files with tools, "
    "then summarize results briefly.\n\n"
    "## Tools\n\n"
    "You have access to native DSML tools. Invoke tools by writing exactly this shape:\n\n"
    "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>\n"
    "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke name=\"$TOOL_NAME\">\n"
    "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter>\n"
    "</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke>\n"
    "</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>\n\n"
    "Tool calls are not allowed inside <think></think>; finish thinking before emitting DSML.\n\n"
    "String parameters use raw text and string=\"true\". Numbers and booleans use JSON text and string=\"false\".\n\n"
    "Inside string values only, escape a literal closing parameter tag as &lt;/" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter>. "
    "To write that escaped spelling literally, use &amp;lt;/" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter>. Other HTML entities are unchanged.\n\n"
    "Read defaults to a context-sized bounded chunk, not the whole file. "
    "For first looks at large files, prefer read with explicit max_lines around 80-160; "
    "if read says more lines are available, call more with count=<lines> to read the next chunk. "
    "Use more for exact continuation; a byte-limited chunk can end within a line. "
    "If the user explicitly asks you to read a complete file into context, call read with whole=true. "
    "A whole-file read may fail if the result would not fit the current context; then explain that and use chunks.\n\n"
    AGENT_TOOL_CONTRACTS;

#define AGENT_EDIT_TARGET_RULE \
    "When editing files, state the target filename before the edit; for the edit tool, put path first."

static const char agent_tools_prompt_edit_exact[] =
    "## Editing files\n\n"
    AGENT_EDIT_TARGET_RULE "\n"
    "Use edit with path, old, and new for changes. The old text must match exactly once in the current file; "
    "otherwise edit fails for safety. Read enough of the file to provide the exact old text being replaced.\n"
    "To insert text, use edit with old set to an exact unique anchor and new set to that anchor plus the added text.\n"
    "Use read raw=true only when you need plain file text without line numbers.\n\n";

static const char agent_tools_prompt_after_edit[] =
    "For long-running bash commands, pass refresh_sec. If a bash job is still running, use "
    "bash_status to check it early or bash_stop to terminate it.\n\n"
    "Use google_search to find web pages. Use visit_page to read a known URL with a visible browser. "
    "The first web call may ask the user for permission to start Chrome.\n\n"
    "### Available Tool Schemas\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"google_search\",\n"
    "    \"description\": \"Search Google in a visible browser and return compact Markdown links.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"query\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"query\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"visit_page\",\n"
    "    \"description\": \"Open a URL in a visible browser and return rendered page Markdown.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"url\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"url\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash\",\n"
    "    \"description\": \"Run a shell command.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"command\": {\"type\": \"string\"},\n"
    "        \"timeout_sec\": {\"type\": \"integer\"},\n"
    "        \"refresh_sec\": {\"type\": \"integer\"}\n"
    "      },\n"
    "      \"required\": [\"command\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash_status\",\n"
    "    \"description\": \"Report current status and recent output for a bash job.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"job\": {\"type\": \"integer\"},\n"
    "        \"pid\": {\"type\": \"integer\"},\n"
    "        \"refresh_sec\": {\"type\": \"integer\"}\n"
    "      },\n"
    "      \"required\": [\"job\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"bash_stop\",\n"
    "    \"description\": \"Terminate a running bash job and report its final output.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"job\": {\"type\": \"integer\"},\n"
    "        \"pid\": {\"type\": \"integer\"},\n"
    "        \"refresh_sec\": {\"type\": \"integer\"}\n"
    "      },\n"
    "      \"required\": [\"job\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"read\",\n"
    "    \"description\": \"Read a text file or a range of lines.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"start_line\": {\"type\": \"integer\"},\n"
    "        \"max_lines\": {\"type\": \"integer\"},\n"
    "        \"whole\": {\"type\": \"boolean\"},\n"
    "        \"raw\": {\"type\": \"boolean\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"more\",\n"
    "    \"description\": \"Continue the previous read.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"count\": {\"type\": \"integer\"}\n"
    "      }\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"write\",\n"
    "    \"description\": \"Create or overwrite a text file.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"content\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"content\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"edit\",\n"
    "    \"description\": \"Replace exactly one old text match.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"old\": {\"type\": \"string\"},\n"
    "        \"new\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\", \"old\", \"new\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"search\",\n"
    "    \"description\": \"Search files and return compact edit-friendly matches.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"query\": {\"type\": \"string\"},\n"
    "        \"path\": {\"type\": \"string\"},\n"
    "        \"mode\": {\"type\": \"string\", \"enum\": [\"literal\", \"regex\"]},\n"
    "        \"glob\": {\"type\": \"string\"},\n"
    "        \"context\": {\"type\": \"integer\"},\n"
    "        \"max_results\": {\"type\": \"integer\"},\n"
    "        \"case_sensitive\": {\"type\": \"boolean\"}\n"
    "      },\n"
    "      \"required\": [\"query\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "{\n"
    "  \"type\": \"function\",\n"
    "  \"function\": {\n"
    "    \"name\": \"list\",\n"
    "    \"description\": \"List one directory compactly.\",\n"
    "    \"parameters\": {\n"
    "      \"type\": \"object\",\n"
    "      \"properties\": {\n"
    "        \"path\": {\"type\": \"string\"}\n"
    "      },\n"
    "      \"required\": [\"path\"]\n"
    "    }\n"
    "  }\n"
    "}\n\n"
    "# Rules\n\n"
    "- Always use strict syntax for DSML tool stanzas.\n"
    "- This system runs on local inference of a few hundred tokens/s of prefill, "
    "and a few tens of tokens/s decoding speed. Use read/search to get the "
    "exact text you need, then use edit instead of rewriting whole files.\n"
    "- Write code that is reliable and works well; always have a mental model of "
    "what is going on in complex parts of the code.\n"
    "- Work in a way that preserves the current system configuration integrity, "
    "unless explicitly asked otherwise by the user.\n";

static const char agent_vision_tool_schema[] =
    "{\"name\":\"view_image\",\"description\":\"Open a local PNG or JPEG as a visual observation.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}";

static char *agent_build_dsml_tools_prompt(bool vision) {
    const char *edit = agent_tools_prompt_edit_exact;
    size_t a = strlen(agent_tools_prompt_intro);
    size_t b = strlen(edit);
    size_t c = strlen(agent_tools_prompt_after_edit);
    const char *vision_start = "\n{\"type\":\"function\",\"function\":";
    size_t v = vision ? strlen(vision_start) + strlen(agent_vision_tool_schema) + 2 : 0;
    char *out = xmalloc(a + b + c + v + 1);
    memcpy(out, agent_tools_prompt_intro, a);
    memcpy(out + a, edit, b);
    const char *rules = strstr(agent_tools_prompt_after_edit, "\n# Rules\n");
    size_t schemas = (size_t)(rules - agent_tools_prompt_after_edit);
    memcpy(out + a + b, agent_tools_prompt_after_edit, schemas);
    if (vision) snprintf(out + a + b + schemas, v + 1, "%s%s}\n", vision_start, agent_vision_tool_schema);
    memcpy(out + a + b + schemas + v, rules, c - schemas + 1);
    return out;
}

static const char agent_glm_tools_prompt_intro[] =
    "You are a coding agent running in a local workspace. Use tools for local file and system work. "
    "Avoid printing large file contents or large code blocks as answers; create or edit files with tools, "
    "then summarize results briefly.\n\n"
    "# Tools\n\n"
    "You may call one or more functions to assist with the user query.\n\n"
    "You are provided with function signatures within <tools></tools> XML tags:\n"
    "<tools>\n";

static const char agent_glm_tools_prompt_after_schemas[] =
    "</tools>\n\n"
    AGENT_TOOL_CONTRACTS
    "Inside argument values only, escape a literal </arg_value> as &lt;/arg_value>. "
    "To write that escaped spelling literally, use &amp;lt;/arg_value>. Other HTML entities are unchanged.\n\n"
    "For a function call, output the function name and arguments within exactly this XML format:\n"
    "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
    "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>\n\n"
    "Tool calls are not allowed inside <think></think>; finish thinking before emitting <tool_call>.\n\n"
    "# Rules\n\n"
    "- Use strict native GLM tool-call syntax with <tool_call>, <arg_key>, and <arg_value> tags.\n"
    "- read path alone returns a context-sized bounded chunk, not the whole file; for first looks at large files, prefer max_lines around 80-160.\n"
    "- If read says more lines are available, call more with count=<lines> to read the next chunk.\n"
    "- Use whole=true only when the user explicitly asks for the complete file contents or when bounded chunks are insufficient for the task; add raw=true only when line numbers would corrupt the payload.\n"
    "- " AGENT_EDIT_TARGET_RULE "\n";

static const char agent_glm_tools_prompt_edit_exact[] =
    "- Use edit with exact old text and replacement new text; old must match exactly once.\n";

static const char agent_glm_tools_prompt_rules_tail[] =
    "- For long bash jobs, pass refresh_sec and then poll with bash_status or stop with bash_stop.\n"
    "- Preserve the current system configuration unless the user explicitly asks otherwise.\n";

static const char agent_glm_tool_schemas[] =
    "{\"name\":\"google_search\",\"description\":\"Search web pages.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"}},\"required\":[\"query\"]}}\n"
    "{\"name\":\"visit_page\",\"description\":\"Read a URL in browser.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"url\":{\"type\":\"string\"}},\"required\":[\"url\"]}}\n"
    "{\"name\":\"bash\",\"description\":\"Run a shell command.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"},\"timeout_sec\":{\"type\":\"integer\"},\"refresh_sec\":{\"type\":\"integer\"}},\"required\":[\"command\"]}}\n"
    "{\"name\":\"bash_status\",\"description\":\"Check a bash job.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"job\":{\"type\":\"integer\"},\"pid\":{\"type\":\"integer\"},\"refresh_sec\":{\"type\":\"integer\"}},\"required\":[\"job\"]}}\n"
    "{\"name\":\"bash_stop\",\"description\":\"Stop a bash job.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"job\":{\"type\":\"integer\"},\"pid\":{\"type\":\"integer\"},\"refresh_sec\":{\"type\":\"integer\"}},\"required\":[\"job\"]}}\n"
    "{\"name\":\"read\",\"description\":\"Read a text file/range.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"start_line\":{\"type\":\"integer\"},\"max_lines\":{\"type\":\"integer\"},\"whole\":{\"type\":\"boolean\"},\"raw\":{\"type\":\"boolean\"}},\"required\":[\"path\"]}}\n"
    "{\"name\":\"more\",\"description\":\"Continue previous read-like output.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"count\":{\"type\":\"integer\"}}}}\n"
    "{\"name\":\"write\",\"description\":\"Create or overwrite a file.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"content\":{\"type\":\"string\"}},\"required\":[\"path\",\"content\"]}}\n"
    "{\"name\":\"edit\",\"description\":\"Replace one exact old text match.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"old\":{\"type\":\"string\"},\"new\":{\"type\":\"string\"}},\"required\":[\"path\",\"old\",\"new\"]}}\n"
    "{\"name\":\"search\",\"description\":\"Search files.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"query\":{\"type\":\"string\"},\"path\":{\"type\":\"string\"},\"mode\":{\"type\":\"string\",\"enum\":[\"literal\",\"regex\"]},\"glob\":{\"type\":\"string\"},\"context\":{\"type\":\"integer\"},\"max_results\":{\"type\":\"integer\"},\"case_sensitive\":{\"type\":\"boolean\"}},\"required\":[\"query\"]}}\n"
    "{\"name\":\"list\",\"description\":\"List one directory.\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"}},\"required\":[\"path\"]}}\n";

static char *agent_build_glm_tools_prompt(bool vision) {
    size_t schemas_len = strlen(agent_glm_tool_schemas);
    const char *schemas = agent_glm_tool_schemas;
    const char *edit = agent_glm_tools_prompt_edit_exact;
    size_t a = strlen(agent_glm_tools_prompt_intro);
    size_t v = vision ? strlen(agent_vision_tool_schema) + 1 : 0;
    size_t b = schemas_len + v;
    size_t c = strlen(agent_glm_tools_prompt_after_schemas);
    size_t d = strlen(edit);
    size_t e = strlen(agent_glm_tools_prompt_rules_tail);
    char *out = xmalloc(a + b + c + d + e + 1);
    memcpy(out, agent_glm_tools_prompt_intro, a);
    memcpy(out + a, schemas, schemas_len);
    if (vision) snprintf(out + a + schemas_len, v + 1, "%s\n", agent_vision_tool_schema);
    memcpy(out + a + b, agent_glm_tools_prompt_after_schemas, c);
    memcpy(out + a + b + c, edit, d);
    memcpy(out + a + b + c + d, agent_glm_tools_prompt_rules_tail, e + 1);
    return out;
}

static char *agent_build_tools_prompt(ds4_engine *engine) {
    if (agent_tool_syntax_for_engine(engine) == AGENT_TOOL_SYNTAX_GLM)
        return agent_build_glm_tools_prompt(ds4_engine_has_vision(engine));
    return agent_build_dsml_tools_prompt(ds4_engine_has_vision(engine));
}

/* ---- System prompt building (server-side) ---- */

static void agent_append_system_prompt(ds4_engine *engine, ds4_tokens *tokens,
                                       const char *extra) {
    char *tools_prompt = agent_build_tools_prompt(engine);
    if (agent_tool_syntax_for_engine(engine) == AGENT_TOOL_SYNTAX_GLM)
        ds4_chat_append_message(engine, tokens, "system", tools_prompt);
    else
        ds4_tokenize_rendered_chat(engine, tools_prompt, tokens);
    free(tools_prompt);

    if (!extra || !extra[0]) return;
    size_t n = strlen(extra);
    char *plain = xmalloc(n + 3);
    memcpy(plain, "\n\n", 2);
    memcpy(plain + 2, extra, n + 1);
    ds4_chat_append_message(engine, tokens, "system", plain);
    free(plain);
}

static void agent_worker_build_system_tokens(agent_worker *w, ds4_tokens *out) {
    ds4_chat_begin(w->engine, out);
    ds4_think_mode think_mode = effective_think_mode(w->cfg);
    if (agent_tool_syntax_for_engine(w->engine) == AGENT_TOOL_SYNTAX_GLM) {
        const char *effort = ds4_glm_reasoning_effort_text(think_mode);
        if (effort) ds4_chat_append_message(w->engine, out, "system", effort);
    } else if (w->cfg->gen.think_mode == DS4_THINK_MAX &&
               think_mode == DS4_THINK_MAX) {
        ds4_chat_append_max_effort_prefix(w->engine, out);
    }
    agent_append_system_prompt(w->engine, out, w->cfg->gen.system);
    ds4_prompt_prefix_append(w->engine, out, &w->cfg->gen.prefix);
}

static char *agent_build_system_prompt_reminder(ds4_engine *engine) {
    char *tools = agent_build_tools_prompt(engine);
    const char *start = "\n\n[System prompt reminder follows.]\n";
    const char *end = "[End system prompt reminder.]\n\n";
    const size_t len = strlen(start) + strlen(tools) + strlen(end) + 1;
    char *out = xmalloc(len);
    snprintf(out, len, "%s%s%s", start, tools, end);
    free(tools);
    return out;
}

static void agent_worker_note_system_prompt_seen(agent_worker *w) {
    w->last_system_prompt_reminder_at = w->transcript.len;
}

static void agent_worker_maybe_append_datetime_context(agent_worker *w) {
    if (w->datetime_context_injected) return;

    time_t now = time(NULL);
    struct tm tm;
    localtime_r(&now, &tm);

    char when[128];
    if (strftime(when, sizeof(when), "%Y-%m-%d %H:%M:%S %Z", &tm) == 0)
        snprintf(when, sizeof(when), "%lld", (long long)now);

    char msg[256];
    snprintf(msg, sizeof(msg),
             "Current local date and time at session start: %s. "
             "Use this only when date or time matters.", when);
    ds4_chat_append_message(w->engine, &w->transcript, "system", msg);
    w->datetime_context_injected = true;
}

static void agent_worker_maybe_append_system_prompt_reminder(agent_worker *w) {
    if (w->last_system_prompt_reminder_at <= 0) {
        agent_worker_note_system_prompt_seen(w);
        return;
    }
    if (w->transcript.len - w->last_system_prompt_reminder_at <
        AGENT_SYSTEM_PROMPT_REMINDER_TOKENS) {
        return;
    }

    char *reminder = agent_build_system_prompt_reminder(w->engine);
    if (agent_tool_syntax_for_engine(w->engine) == AGENT_TOOL_SYNTAX_GLM) {
        ds4_chat_append_message(w->engine, &w->transcript, "system", reminder);
    } else {
        ds4_tokenize_rendered_chat(w->engine, reminder, &w->transcript);
    }
    free(reminder);

    if (w->cfg->gen.system && w->cfg->gen.system[0]) {
        ds4_chat_append_message(w->engine, &w->transcript, "system",
                                w->cfg->gen.system);
    }
    agent_worker_note_system_prompt_seen(w);
}

/* ---- Small worker helpers ---- */

static int agent_worker_effective_ctx_size(const agent_worker *w) {
    int ctx = 0;
    if (w && w->session) ctx = ds4_session_ctx(w->session);
    if (ctx <= 0 && w && w->cfg) ctx = w->cfg->gen.ctx_size;
    return ctx;
}

static void agent_worker_append_assistant_turn_end(agent_worker *w) {
    if (agent_tool_syntax_assistant_turn_uses_eos(
            agent_tool_syntax_for_engine(w->engine)))
        ds4_tokens_push(&w->transcript, ds4_token_eos(w->engine));
}

static void agent_worker_images_clear(agent_worker *w) {
    if (!w) return;
    for (size_t i = 0; i < w->image_count; i++)
        ds4_vision_embedding_free(&w->images[i].embedding);
    free(w->images);
    w->images = NULL;
    w->image_count = 0;
    w->image_cap = 0;
}

static void agent_worker_images_append(agent_worker *w,
                                       ds4_vision_span *spans,
                                       size_t count) {
    if (!count) return;
    if (w->image_count + count > w->image_cap) {
        size_t cap = w->image_cap ? w->image_cap : 4;
        while (cap < w->image_count + count) cap *= 2;
        w->images = xrealloc(w->images, cap * sizeof(w->images[0]));
        w->image_cap = cap;
    }
    memcpy(w->images + w->image_count, spans, count * sizeof(spans[0]));
    w->image_count += count;
    memset(spans, 0, count * sizeof(spans[0]));
}

static bool agent_worker_images_fit_tokens(const agent_worker *w,
                                           const ds4_tokens *tokens) {
    if (!w || !tokens) return false;
    for (size_t i = 0; i < w->image_count; i++) {
        uint64_t end = (uint64_t)w->images[i].token_start +
                       w->images[i].embedding.token_count;
        if (end > (uint64_t)tokens->len) return false;
    }
    return true;
}

static bool agent_tokens_equal(const ds4_tokens *a, const ds4_tokens *b) {
    if (!a || !b || a->len != b->len) return false;
    for (int i = 0; i < a->len; i++) {
        if (a->v[i] != b->v[i]) return false;
    }
    return true;
}

static void agent_tokens_append_range(ds4_tokens *dst, const ds4_tokens *src,
                                      int start, int end) {
    if (start < 0) start = 0;
    if (end > src->len) end = src->len;
    for (int i = start; i < end; i++) ds4_tokens_push(dst, src->v[i]);
}

/* ---- Persistence helpers (ported from ds4_agent.c) ---- */

static bool agent_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

static char *agent_default_cache_dir(void) {
    const char *home = getenv("HOME");
    if (!home || !home[0]) home = ".";
    agent_buf b = {0};
    agent_buf_puts(&b, home);
    if (b.len == 0 || b.ptr[b.len - 1] != '/') agent_buf_puts(&b, "/");
    agent_buf_puts(&b, ".ds4/kvcache");
    return agent_buf_take(&b);
}

static char *agent_kv_path_for_sha(const char *dir, const char sha[41]) {
    char name[44];
    memcpy(name, sha, 40);
    memcpy(name + 40, ".kv", 4);
    return ds4_kvstore_path_join(dir, name);
}

static void agent_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static void agent_session_identity_sha(const char *title, uint64_t created_at,
                                       char sha_out[41]) {
    size_t title_len = title ? strlen(title) : 0;
    agent_buf b = {0};
    agent_buf_append(&b, title ? title : "", title_len);
    uint8_t ts[8];
    agent_le_put64(ts, created_at);
    agent_buf_append(&b, (const char *)ts, sizeof(ts));
    ds4_kvstore_sha1_bytes_hex(b.ptr ? b.ptr : "", b.len, sha_out);
    free(b.ptr);
}

static void agent_worker_clear_session_identity(agent_worker *w) {
    w->session_sha[0] = '\0';
    free(w->session_title);
    w->session_title = NULL;
    w->session_created_at = 0;
    free(w->legacy_session_path_to_delete);
    w->legacy_session_path_to_delete = NULL;
}

typedef struct {
    bool has_title_trailer;
    bool legacy_identity;
    char *title;
    uint64_t created_at;
    char sha[41];
} agent_kv_session_meta;

static void agent_kv_session_meta_free(agent_kv_session_meta *m) {
    free(m->title);
    memset(m, 0, sizeof(*m));
}

static bool agent_fp_remaining(FILE *fp, uint64_t *out) {
    off_t pos = ftello(fp);
    if (pos < 0 || fseeko(fp, 0, SEEK_END) != 0) return false;
    off_t end = ftello(fp);
    if (end < 0 || fseeko(fp, pos, SEEK_SET) != 0) return false;
    *out = end > pos ? (uint64_t)(end - pos) : 0;
    return true;
}

static bool agent_kv_read_text(FILE *fp, uint32_t text_bytes,
                               char **text_out, char *err, size_t err_len) {
    uint64_t remaining = 0;
    if (!agent_fp_remaining(fp, &remaining) || text_bytes > remaining) {
        if (err && err_len) snprintf(err, err_len, "truncated cached text");
        return false;
    }
    char *text = xmalloc((size_t)text_bytes + 1);
    if (fread(text, 1, text_bytes, fp) != text_bytes) {
        if (err && err_len) snprintf(err, err_len, "truncated cached text");
        free(text);
        return false;
    }
    text[text_bytes] = '\0';
    *text_out = text;
    return true;
}

static bool agent_kv_write_title_trailer(FILE *fp, const char *title,
                                         char *err, size_t err_len) {
    size_t title_len = title ? strlen(title) : 0;
    if (title_len > UINT32_MAX) {
        snprintf(err, err_len, "agent session title is too large");
        return false;
    }
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)title_len);
    return fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
           fwrite(title ? title : "", 1, title_len, fp) == title_len;
}

static bool agent_kv_read_title_trailer(FILE *fp, const ds4_kvstore_entry *hdr,
                                        char **title_out,
                                        char *err, size_t err_len) {
    off_t payload_pos = ftello(fp);
    if (payload_pos < 0) {
        if (err && err_len) snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }
    if (hdr->payload_bytes > (uint64_t)LLONG_MAX ||
        fseeko(fp, (off_t)hdr->payload_bytes, SEEK_CUR) != 0)
    {
        if (err && err_len) snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }

    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) {
        if (err && err_len) snprintf(err, err_len, "missing agent session title trailer");
        fseeko(fp, payload_pos, SEEK_SET);
        return false;
    }
    uint32_t title_bytes = ds4_kvstore_le_get32(tb);
    uint64_t title_remaining = 0;
    if (!agent_fp_remaining(fp, &title_remaining) || title_bytes > title_remaining) {
        if (err && err_len) snprintf(err, err_len, "truncated agent session title trailer");
        fseeko(fp, payload_pos, SEEK_SET);
        return false;
    }
    char *title = xmalloc((size_t)title_bytes + 1);
    if (fread(title, 1, title_bytes, fp) != title_bytes) {
        if (err && err_len) snprintf(err, err_len, "truncated agent session title trailer");
        free(title);
        fseeko(fp, payload_pos, SEEK_SET);
        return false;
    }
    title[title_bytes] = '\0';
    if (fseeko(fp, payload_pos, SEEK_SET) != 0) {
        if (err && err_len) snprintf(err, err_len, "%s", strerror(errno));
        free(title);
        return false;
    }
    *title_out = title;
    return true;
}

static void agent_kv_identity_sha(const ds4_kvstore_entry *hdr,
                                  const char *text, uint32_t text_bytes,
                                  const char *title,
                                  char sha_out[41]) {
    if (hdr->ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE) {
        agent_session_identity_sha(title ? title : "", hdr->created_at, sha_out);
    } else {
        ds4_kvstore_sha1_bytes_hex(text, text_bytes, sha_out);
    }
}

static bool agent_kv_payload_requires_rebuild(const agent_worker *w,
                                              uint64_t payload_bytes) {
    if (payload_bytes == 0) return true;
    return w && w->cfg && ds4_tp_enabled(&w->cfg->engine.tp);
}

static char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes);
static int agent_worker_sync_tokens(agent_worker *w, const ds4_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len);

static bool agent_kv_load_path(agent_worker *w, const char *path,
                               const char *expected_sha,
                               const char *expected_text,
                               size_t expected_text_len,
                               ds4_tokens *loaded_tokens,
                               agent_kv_session_meta *meta_out,
                               char *err, size_t err_len) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }

    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes);
    if (!ok) snprintf(err, err_len, "invalid KV header");

    char *text = NULL;
    if (ok) ok = agent_kv_read_text(fp, text_bytes, &text, err, err_len);
    char *title = NULL;
    bool has_title = ok && (hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE);
    if (has_title)
        ok = agent_kv_read_title_trailer(fp, &hdr, &title, err, err_len);
    uint32_t expected_tokens = hdr.tokens;
    if (ok && hdr.payload_bytes != 0 &&
        hdr.model_id != (uint8_t)ds4_engine_model_id(w->engine))
    {
        snprintf(err, err_len, "KV checkpoint was written for a different model");
        ok = false;
    }
    if (ok && hdr.payload_bytes != 0 &&
        hdr.quant_bits != (uint8_t)ds4_engine_routed_quant_bits(w->engine))
    {
        snprintf(err, err_len, "KV checkpoint was written for a different quantization");
        ok = false;
    }
    if (ok && expected_text) {
        if ((size_t)text_bytes != expected_text_len ||
            memcmp(text, expected_text, expected_text_len) != 0)
        {
            snprintf(err, err_len, "cached text does not match current system prompt");
            ok = false;
        }
    }
    if (ok && expected_sha) {
        char actual_sha[41];
        agent_kv_identity_sha(&hdr, text, text_bytes, title, actual_sha);
        if (strcmp(actual_sha, expected_sha)) {
            snprintf(err, err_len, "cached session identity does not match file name");
            ok = false;
        }
    }

    char load_err[160] = {0};
    if (ok && agent_kv_payload_requires_rebuild(w, hdr.payload_bytes)) {
        ds4_tokens rebuilt = {0};
        ds4_tokenize_rendered_chat(w->engine, text, &rebuilt);
        expected_tokens = (uint32_t)rebuilt.len;
        if (agent_worker_sync_tokens(w, &rebuilt, true, err, err_len) != 0) {
            ds4_session_invalidate(w->session);
            ok = false;
        }
        ds4_tokens_free(&rebuilt);
    } else if (ok &&
               ds4_session_load_payload(w->session, fp, hdr.payload_bytes,
                                        load_err, sizeof(load_err)) != 0)
    {
        snprintf(err, err_len, "%s", load_err[0] ? load_err : "failed to load KV payload");
        ds4_session_invalidate(w->session);
        ok = false;
    }
    fclose(fp);

    if (ok) {
        const ds4_tokens *live = ds4_session_tokens(w->session);
        if (!live || live->len != (int)expected_tokens) {
            snprintf(err, err_len, "KV payload token count mismatch");
            ds4_session_invalidate(w->session);
            ok = false;
        } else if (loaded_tokens) {
            ds4_tokens_free(loaded_tokens);
            ds4_tokens_copy(loaded_tokens, live);
        }
        if (meta_out) {
            agent_kv_session_meta_free(meta_out);
            meta_out->has_title_trailer = has_title;
            meta_out->legacy_identity = !has_title;
            meta_out->created_at = hdr.created_at;
            agent_kv_identity_sha(&hdr, text, text_bytes, title, meta_out->sha);
            meta_out->title = has_title ?
                xstrdup(title) :
                agent_session_title_from_text(text, text_bytes, 0);
        }
    }
    free(title);
    free(text);
    return ok;
}

static bool agent_kv_save_path(agent_worker *w, const char *path,
                               const ds4_tokens *tokens,
                               const char *reason,
                               char sha_out[41],
                               const char *session_title,
                               uint64_t session_created_at,
                               char *err, size_t err_len) {
    const ds4_tokens *live = ds4_session_tokens(w->session);
    if (!agent_tokens_equal(live, tokens)) {
        snprintf(err, err_len, "live KV state does not match session transcript");
        return false;
    }
    const int quant_bits = ds4_engine_routed_quant_bits(w->engine);
    if (quant_bits != 2 && quant_bits != 4) {
        snprintf(err, err_len, "unsupported routed quantization for KV save");
        return false;
    }
    const int model_id = ds4_engine_model_id(w->engine);

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, tokens, &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render KV text key");
        return false;
    }
    if (text_len > UINT32_MAX) {
        snprintf(err, err_len, "rendered KV text key is too large");
        free(text);
        return false;
    }
    const bool session_identity = session_title != NULL;
    uint64_t now = (uint64_t)time(NULL);
    uint64_t created_at = session_identity && session_created_at ?
        session_created_at : now;
    char sha[41];
    if (session_identity)
        agent_session_identity_sha(session_title, created_at, sha);
    else
        ds4_kvstore_sha1_bytes_hex(text, text_len, sha);
    if (sha_out) memcpy(sha_out, sha, sizeof(sha));

    ds4_session_payload_file staged = {0};
    char save_err[160] = {0};
    if (ds4_session_stage_payload(w->session, &staged,
                                  save_err, sizeof(save_err)) != 0) {
        snprintf(err, err_len, "%s",
                 save_err[0] ? save_err : "session has no valid KV payload");
        free(text);
        return false;
    }
    uint64_t payload_bytes = staged.bytes;

    agent_buf tmpl = {0};
    agent_buf_puts(&tmpl, path);
    agent_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = agent_buf_take(&tmpl);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        ds4_session_payload_file_free(&staged);
        free(tmp);
        free(text);
        return false;
    }

    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        close(fd);
        unlink(tmp);
        ds4_session_payload_file_free(&staged);
        free(tmp);
        free(text);
        return false;
    }

    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, (uint8_t)model_id, (uint8_t)quant_bits,
                            ds4_kvstore_reason_code(reason),
                            session_identity ? DS4_KVSTORE_EXT_SESSION_TITLE : 0,
                            (uint32_t)tokens->len, 0,
                            (uint32_t)ds4_session_ctx(w->session),
                            created_at, now, payload_bytes);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)text_len);

    errno = 0;
    bool ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
              fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
              fwrite(text, 1, text_len, fp) == text_len &&
              ds4_session_write_staged_payload(&staged, fp,
                                               save_err, sizeof(save_err)) == 0 &&
              (!session_identity ||
               agent_kv_write_title_trailer(fp, session_title,
                                            save_err, sizeof(save_err))) &&
              fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    if (!ok) {
        snprintf(err, err_len, "%s",
                 saved_errno ? strerror(saved_errno) :
                 (save_err[0] ? save_err : "failed to write KV file"));
        unlink(tmp);
    }

    ds4_session_payload_file_free(&staged);
    free(tmp);
    free(text);
    return ok;
}

/* ---- Session titles / listing helpers ---- */

static char *agent_session_title_from_span(const char *p, const char *end,
                                           size_t max_bytes,
                                           const char *empty_title) {
    bool limited = max_bytes != 0;
    if (limited && max_bytes < 4) max_bytes = 4;
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;

    agent_buf b = {0};
    bool space = false;
    bool truncated = false;
    for (const char *s = p; s < end; s++) {
        unsigned char c = (unsigned char)*s;
        if (isspace(c)) {
            space = b.len != 0;
            continue;
        }
        if (space && (!limited || b.len + 4 < max_bytes)) {
            agent_buf_puts(&b, " ");
            space = false;
        }
        if (limited && b.len + 4 > max_bytes) {
            truncated = true;
            break;
        }
        agent_buf_append(&b, s, 1);
    }
    if (truncated) agent_buf_puts(&b, "...");
    if (!b.ptr || !b.len) {
        free(b.ptr);
        return xstrdup(empty_title);
    }
    return agent_buf_take(&b);
}

static char *agent_session_title_from_prompt(const char *prompt,
                                             size_t max_bytes) {
    const char *p = prompt ? prompt : "";
    return agent_session_title_from_span(p, p + strlen(p), max_bytes,
                                         "(empty user prompt)");
}

static char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes) {
    static const char user_mark[] =
        "<" AGENT_DSML_BAR "User" AGENT_DSML_BAR ">";
    static const char assistant_mark[] =
        "<" AGENT_DSML_BAR "Assistant" AGENT_DSML_BAR ">";
    const char *p = text ? strstr(text, user_mark) : NULL;
    if (!p) return xstrdup("(no user prompt)");
    p += strlen(user_mark);
    const char *end = text + text_len;
    const char *assistant = strstr(p, assistant_mark);
    const char *next_user = strstr(p, user_mark);
    if (assistant && assistant < end) end = assistant;
    if (next_user && next_user < end) end = next_user;
    return agent_session_title_from_span(p, end, max_bytes,
                                         "(empty user prompt)");
}

static char *agent_session_title_clip(const char *title, size_t max_bytes) {
    if (!title) return xstrdup("(no user prompt)");
    size_t len = strlen(title);
    if (max_bytes == 0 || len <= max_bytes) return xstrdup(title);
    if (max_bytes < 4) max_bytes = 4;
    agent_buf b = {0};
    agent_buf_append(&b, title, max_bytes - 3);
    agent_buf_puts(&b, "...");
    return agent_buf_take(&b);
}

static char *agent_session_title_from_file(const char *path, size_t max_bytes) {
    FILE *fp = fopen(path, "rb");
    if (!fp) return xstrdup("(unreadable session)");
    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    char *text = NULL;
    char *trailer_title = NULL;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              agent_kv_read_text(fp, text_bytes, &text, NULL, 0);
    if (ok && (hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE))
        ok = agent_kv_read_title_trailer(fp, &hdr, &trailer_title, NULL, 0);
    fclose(fp);
    char *title = ok ?
        (trailer_title ?
            agent_session_title_clip(trailer_title, max_bytes) :
            agent_session_title_from_text(text, text_bytes, max_bytes)) :
        xstrdup("(unreadable session)");
    free(trailer_title);
    free(text);
    return title;
}

/* ---- Socket write helper (worker is the only writer) ---- */

static bool agent_send_all(int fd, const void *buf, size_t len) {
    if (fd < 0) return true;  /* No client connected yet */

    /* Debug: log every frame sent to the client except the per-token TOKEN
     * stream (that would flood stderr during generation). */
    if (dbg_on && len >= 5 && buf) {
        unsigned char tag = ((const unsigned char *)buf)[4];
        if (tag != PROTO_S2C_TOKEN) {
            if (tag == PROTO_S2C_STATUS && len >= 6) {
                size_t adv = 0;
                uint64_t state = dbg_varint_at((const unsigned char *)buf + 5,
                                               len - 5, &adv);
                dbg_log('S', "STATUS %s -> client (len %zu)",
                        dbg_status_state_name(state), len);
            } else {
                dbg_log('S', "%s -> client (len %zu)", dbg_tag_name(tag), len);
            }
        }
    }
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

/* ============================================================================
 * Status / progress push
 *
 * The worker is the only socket writer.  Status is pushed immediately on state
 * transitions / greedy toggle / power, and throttled for prefill progress and
 * generated counts so the client footer stays live without flooding.
 * ========================================================================== */

static void agent_push_status(agent_worker *w, bool force) {
    (void)force;
    proto_status_msg m = {
        .state = (uint64_t)w->status.state,
        .ctx_used = (uint64_t)(w->status.ctx_used > 0 ? w->status.ctx_used : w->transcript.len),
        .ctx_size = (uint64_t)agent_worker_effective_ctx_size(w),
        .prefill_done = (uint64_t)w->status.prefill_done,
        .prefill_total = (uint64_t)w->status.prefill_total,
        .prefill_tps = (float)w->status.prefill_tps,
        .generated = (uint64_t)w->status.generated,
        .gen_tps = (float)w->status.gen_tps,
        .greedy = w->status.greedy_sampling,
        .power = (uint64_t)w->status.power_percent,
        .error = w->status.error,
    };
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf = proto_encode_status(&wr, &m, &out_len);
    (void)agent_send_all(w->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
}

static void agent_maybe_push_status(agent_worker *w) {
    double now = now_sec();
    if (now - w->last_status_push_at < AGENT_STATUS_PUSH_INTERVAL) return;
    w->last_status_push_at = now;
    agent_push_status(w, true);
}

static void agent_set_status(agent_worker *w, agent_worker_state state) {
    pthread_mutex_lock(&w->mu);
    agent_worker_state old = w->status.state;
    w->status.state = state;
    if (state != AGENT_WORKER_PREFILL)
        w->status.prefill_tps = 0.0;
    if (state != AGENT_WORKER_GENERATING)
        w->status.greedy_sampling = false;
    w->last_status_push_at = 0.0;
    pthread_mutex_unlock(&w->mu);
    if (old != state)
        dbg_log('T', "state %s -> %s",
                agent_state_name(old), agent_state_name(state));
    agent_push_status(w, true);
}

static void agent_set_error(agent_worker *w, const char *msg) {
    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_ERROR;
    w->status.prefill_tps = 0.0;
    w->status.greedy_sampling = false;
    snprintf(w->status.error, sizeof(w->status.error),
             "%s", msg ? msg : "unknown error");
    w->last_status_push_at = 0.0;
    pthread_mutex_unlock(&w->mu);
    dbg_log('T', "state -> error: %s", msg ? msg : "unknown error");
    agent_push_status(w, true);
}

static bool worker_should_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool interrupt = w->interrupt || w->stop;
    pthread_mutex_unlock(&w->mu);
    return interrupt;
}

static void worker_clear_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    pthread_mutex_unlock(&w->mu);
}

static bool agent_err_is_interrupted(const char *err) {
    return err && !strcmp(err, "interrupted");
}

static bool worker_cancel_session_cb(void *ud) {
    return worker_should_interrupt((agent_worker *)ud);
}

static void worker_apply_pending_power(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    int power = 0;
    bool requested = w->power_requested;
    if (requested) {
        power = w->requested_power;
        w->power_requested = false;
    }
    pthread_mutex_unlock(&w->mu);
    if (!requested) return;
    if (ds4_session_set_power(w->session, power) != 0) {
        return;
    }
    pthread_mutex_lock(&w->mu);
    w->cfg->engine.power_percent = power;
    w->status.power_percent = power;
    w->last_status_push_at = 0.0;
    pthread_mutex_unlock(&w->mu);
    agent_push_status(w, true);
}

static void worker_progress_cb(void *ud, const char *event, int current, int total) {
    agent_worker *w = ud;
    if (!w || !event) return;
    (void)total;
    worker_apply_pending_power(w);
    pthread_mutex_lock(&w->mu);
    int done = 0;
    int progress_total = w->status.prefill_total;
    if (!strcmp(event, "prefill_work")) {
        done = current;
    } else if (!strcmp(event, "prefill_chunk") ||
               !strcmp(event, "prefill_display")) {
        if (w->progress_direct) {
            pthread_mutex_unlock(&w->mu);
            return;
        }
        done = current - w->progress_base;
    } else {
        pthread_mutex_unlock(&w->mu);
        return;
    }
    if (progress_total < 0) progress_total = 0;
    if (done < 0) done = 0;
    if (done > progress_total) done = progress_total;
    w->status.prefill_total = progress_total;
    w->status.prefill_done = done;
    double elapsed = now_sec() - w->progress_started_at;
    w->status.prefill_tps =
        done > 0 && elapsed > 0.0 ? (double)done / elapsed : 0.0;
    bool push = now_sec() - w->last_status_push_at >= AGENT_STATUS_PUSH_INTERVAL;
    pthread_mutex_unlock(&w->mu);
    if (push) {
        w->last_status_push_at = now_sec();
        agent_push_status(w, true);
    }
}

/* ---- Session save / sync ---- */

static bool agent_worker_has_user_session(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool yes = w->user_activity;
    pthread_mutex_unlock(&w->mu);
    return yes;
}

static int agent_worker_sync_tokens(agent_worker *w, const ds4_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len) {
    int old_pos = ds4_session_pos(w->session);
    int common = ds4_session_common_prefix(w->session, tokens);
    int cached = common == old_pos && tokens->len >= old_pos ? common : 0;
    int suffix = tokens->len - cached;
    if (suffix < 0) suffix = tokens->len;

    if (publish_progress) {
        pthread_mutex_lock(&w->mu);
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->progress_direct = false;
        w->progress_started_at = now_sec();
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        w->last_status_push_at = 0.0;
        pthread_mutex_unlock(&w->mu);
        agent_push_status(w, true);
    }

    ds4_session_set_progress(w->session, publish_progress ? worker_progress_cb : NULL,
                             publish_progress ? w : NULL);
    ds4_session_set_display_progress(w->session,
                                     publish_progress ? worker_progress_cb : NULL,
                                     publish_progress ? w : NULL);
    ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
    int rc;
    if (w->image_count) {
        if (!agent_worker_images_fit_tokens(w, tokens)) {
            snprintf(err, err_len, "image spans do not fit the agent transcript");
            rc = 1;
        } else {
            rc = ds4_session_sync_multimodal(w->session, tokens,
                                             w->images, w->image_count,
                                             err, err_len);
        }
    } else {
        rc = ds4_session_sync(w->session, tokens, err, err_len);
    }
    ds4_session_set_cancel(w->session, NULL, NULL);
    ds4_session_set_progress(w->session, NULL, NULL);
    ds4_session_set_display_progress(w->session, NULL, NULL);
    return rc;
}

static bool agent_worker_reset_to_sysprompt(agent_worker *w, char *err, size_t err_len) {
    agent_worker_images_clear(w);
    ds4_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &sys, &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render system prompt");
        ds4_tokens_free(&sys);
        return false;
    }

    bool loaded = false;
    char load_err[160] = {0};
    if (w->sysprompt_path) {
        loaded = agent_kv_load_path(w, w->sysprompt_path, NULL,
                                    text, text_len, &w->transcript,
                                    NULL,
                                    load_err, sizeof(load_err));
    }

    if (!loaded) {
        ds4_tokens_free(&w->transcript);
        ds4_tokens_copy(&w->transcript, &sys);
        if (agent_worker_sync_tokens(w, &w->transcript, true, err, err_len) != 0) {
            free(text);
            ds4_tokens_free(&sys);
            return false;
        }
        if (w->sysprompt_path) {
            char save_err[160] = {0};
            char ignored_sha[41];
            if (!agent_kv_save_path(w, w->sysprompt_path, &w->transcript,
                                    "agent-system", ignored_sha,
                                    NULL, 0,
                                    save_err, sizeof(save_err)))
            {
                fprintf(stderr, "ds4-agent-server: failed to save system prompt KV: %s\n",
                        save_err);
            }
        }
    }

    agent_worker_note_system_prompt_seen(w);
    pthread_mutex_lock(&w->mu);
    w->user_activity = false;
    w->session_dirty = false;
    w->status.state = AGENT_WORKER_IDLE;
    w->progress_direct = false;
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.prefill_tps = 0.0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.greedy_sampling = false;
    w->status.error[0] = '\0';
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = agent_worker_effective_ctx_size(w);
    w->status.power_percent = w->cfg->engine.power_percent;
    w->last_status_push_at = 0.0;
    pthread_mutex_unlock(&w->mu);
    w->datetime_context_injected = false;
    agent_worker_clear_session_identity(w);
    free(text);
    ds4_tokens_free(&sys);
    return true;
}

static bool agent_worker_save_session_now(agent_worker *w, char sha_out[41],
                                          int *tokens_out,
                                          char *err, size_t err_len) {
    if (!agent_worker_has_user_session(w)) {
        snprintf(err, err_len, "nothing to save");
        return false;
    }
    if (w->image_count) {
        snprintf(err, err_len,
                 "sessions containing images cannot be saved yet");
        return false;
    }

    if (agent_worker_sync_tokens(w, &w->transcript, false, err, err_len) != 0)
        return false;
    if (!agent_mkdir_p(w->cache_dir)) {
        snprintf(err, err_len, "failed to create %s", w->cache_dir);
        return false;
    }

    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &w->transcript,
                                                &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render session text");
        return false;
    }
    if (!w->session_title) {
        w->session_title = agent_session_title_from_text(text, text_len, 0);
    }
    if (w->session_created_at == 0)
        w->session_created_at = (uint64_t)time(NULL);

    char sha[41];
    agent_session_identity_sha(w->session_title, w->session_created_at, sha);
    char *path = agent_kv_path_for_sha(w->cache_dir, sha);

    bool ok = agent_kv_save_path(w, path, &w->transcript,
                                 "agent-session", sha_out,
                                 w->session_title, w->session_created_at,
                                 err, err_len);
    if (ok) {
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        if (w->legacy_session_path_to_delete &&
            strcmp(w->legacy_session_path_to_delete, path) != 0)
        {
            unlink(w->legacy_session_path_to_delete);
        }
        free(w->legacy_session_path_to_delete);
        w->legacy_session_path_to_delete = NULL;
        pthread_mutex_lock(&w->mu);
        w->session_dirty = false;
        pthread_mutex_unlock(&w->mu);
        if (tokens_out) *tokens_out = w->transcript.len;
    }
    free(path);
    free(text);
    return ok;
}

/* ---- Session listing / find / delete / strip / switch ---- */

typedef struct {
    ds4_kvstore_entry entry;
    char *title;
} agent_session_list_item;

static int agent_session_list_cmp_recent(const void *a, const void *b) {
    const agent_session_list_item *sa = a, *sb = b;
    uint64_t ta = sa->entry.last_used ? sa->entry.last_used : sa->entry.created_at;
    uint64_t tb = sb->entry.last_used ? sb->entry.last_used : sb->entry.created_at;
    if (ta < tb) return 1;
    if (ta > tb) return -1;
    return strcmp(sa->entry.sha, sb->entry.sha);
}

static void agent_session_list_free(agent_session_list_item *v, int n) {
    for (int i = 0; i < n; i++) {
        ds4_kvstore_entry_free(&v[i].entry);
        free(v[i].title);
    }
    free(v);
}

static void agent_session_list_push(agent_session_list_item **v, int *len,
                                    int *cap, ds4_kvstore_entry entry,
                                    char *title) {
    if (*len == *cap) {
        *cap = *cap ? *cap * 2 : 16;
        *v = xrealloc(*v, (size_t)*cap * sizeof((*v)[0]));
    }
    (*v)[(*len)++] = (agent_session_list_item){
        .entry = entry,
        .title = title,
    };
}

/* Build the proto LIST payload from ~/.ds4/kvcache.  sysprompt.kv is
 * intentionally ignored because it is an implementation cache, not a session. */
static bool agent_worker_build_list(agent_worker *w, proto_list_msg *out,
                                    char *err, size_t err_len) {
    DIR *d = opendir(w->cache_dir);
    if (!d) {
        snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }

    agent_session_list_item *sessions = NULL;
    int sessions_len = 0, sessions_cap = 0;
    const uint8_t model_id = (uint8_t)ds4_engine_model_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        if (ds4_kvstore_read_entry_file(path, sha, &e)) {
            if (e.model_id == model_id) {
                char *title = agent_session_title_from_file(path, 160);
                agent_session_list_push(&sessions, &sessions_len, &sessions_cap,
                                        e, title);
            } else {
                ds4_kvstore_entry_free(&e);
            }
        }
        free(path);
    }
    closedir(d);

    qsort(sessions, (size_t)sessions_len, sizeof(sessions[0]),
          agent_session_list_cmp_recent);

    out->items = calloc((size_t)sessions_len, sizeof(out->items[0]));
    out->count = (size_t)sessions_len;
    out->cap = (size_t)sessions_len;
    for (int i = 0; i < sessions_len; i++) {
        ds4_kvstore_entry *e = &sessions[i].entry;
        proto_list_item *it = &out->items[i];
        strncpy(it->sha, e->sha, 40);
        it->sha[40] = '\0';
        it->title = sessions[i].title ? sessions[i].title : xstrdup("");
        sessions[i].title = NULL;
        it->last_used = e->last_used ? e->last_used : e->created_at;
        it->created_at = e->created_at;
        it->tokens = e->tokens;
        it->file_size = e->file_size;
        it->payload_bytes = e->payload_bytes;
    }
    agent_session_list_free(sessions, sessions_len);
    return true;
}

static bool agent_worker_find_session(agent_worker *w, const char *prefix,
                                      char sha_out[41], char **path_out,
                                      char *err, size_t err_len) {
    size_t plen = strlen(prefix);
    if (plen == 0 || plen > 40) {
        snprintf(err, err_len, "invalid session SHA prefix");
        return false;
    }
    for (size_t i = 0; i < plen; i++) {
        if (!isxdigit((unsigned char)prefix[i])) {
            snprintf(err, err_len, "invalid session SHA prefix");
            return false;
        }
    }

    DIR *d = opendir(w->cache_dir);
    if (!d) {
        snprintf(err, err_len, "%s", strerror(errno));
        return false;
    }
    int matches = 0;
    char match_sha[41] = {0};
    char *match_path = NULL;
    const uint8_t model_id = (uint8_t)ds4_engine_model_id(w->engine);
    struct dirent *de;
    while ((de = readdir(d)) != NULL) {
        char sha[41];
        if (!ds4_kvstore_sha_hex_name(de->d_name, sha)) continue;
        if (strncasecmp(sha, prefix, plen) != 0) continue;
        char *path = ds4_kvstore_path_join(w->cache_dir, de->d_name);
        ds4_kvstore_entry e = {0};
        bool same_model = ds4_kvstore_read_entry_file(path, sha, &e) &&
                          e.model_id == model_id;
        ds4_kvstore_entry_free(&e);
        if (!same_model) {
            free(path);
            continue;
        }
        matches++;
        if (matches == 1) {
            memcpy(match_sha, sha, sizeof(match_sha));
            match_path = path;
        } else {
            free(path);
        }
    }
    closedir(d);
    if (matches == 0) {
        snprintf(err, err_len, "no saved session matches %.40s", prefix);
        return false;
    }
    if (matches > 1) {
        snprintf(err, err_len, "session prefix %.40s is ambiguous", prefix);
        free(match_path);
        return false;
    }
    memcpy(sha_out, match_sha, 41);
    *path_out = match_path;
    return true;
}

static bool agent_worker_delete_session(agent_worker *w, const char *prefix,
                                        char sha_out[41],
                                        char *err, size_t err_len) {
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;
    if (unlink(path) != 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(path);
        return false;
    }
    if (sha_out) memcpy(sha_out, sha, 41);
    free(path);
    return true;
}

static bool agent_worker_strip_session(agent_worker *w, const char *prefix,
                                       char sha_out[41],
                                       uint32_t *tokens_out,
                                       char *err, size_t err_len) {
    if (err && err_len) err[0] = '\0';
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;

    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(path);
        return false;
    }

    ds4_kvstore_entry hdr = {0};
    uint32_t text_bytes = 0;
    char *text = NULL;
    char *title = NULL;
    bool ok = ds4_kvstore_read_header(fp, &hdr, &text_bytes) &&
              agent_kv_read_text(fp, text_bytes, &text, err, err_len);
    if (ok && (hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE))
        ok = agent_kv_read_title_trailer(fp, &hdr, &title, err, err_len);
    fclose(fp);
    if (!ok) {
        if (!err[0]) snprintf(err, err_len, "failed to read session");
        free(title);
        free(text);
        free(path);
        return false;
    }

    char actual_sha[41];
    agent_kv_identity_sha(&hdr, text, text_bytes, title, actual_sha);
    if (strcmp(actual_sha, sha)) {
        snprintf(err, err_len, "cached session identity does not match file name");
        free(title);
        free(text);
        free(path);
        return false;
    }

    ds4_tokens stripped_tokens = {0};
    ds4_tokenize_rendered_chat(w->engine, text, &stripped_tokens);
    uint32_t stripped_token_count = (uint32_t)stripped_tokens.len;
    ds4_tokens_free(&stripped_tokens);

    agent_buf tmpl = {0};
    agent_buf_puts(&tmpl, path);
    agent_buf_puts(&tmpl, ".tmp.XXXXXX");
    char *tmp = agent_buf_take(&tmpl);
    int fd = mkstemp(tmp);
    if (fd < 0) {
        snprintf(err, err_len, "%s", strerror(errno));
        free(tmp);
        free(text);
        free(path);
        return false;
    }

    fp = fdopen(fd, "wb");
    if (!fp) {
        snprintf(err, err_len, "%s", strerror(errno));
        close(fd);
        unlink(tmp);
        free(tmp);
        free(text);
        free(path);
        return false;
    }

    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    uint64_t now = (uint64_t)time(NULL);
    ds4_kvstore_fill_header(h, hdr.model_id, hdr.quant_bits, hdr.reason, hdr.ext_flags,
                            stripped_token_count, hdr.hits, hdr.ctx_size,
                            hdr.created_at, now, 0);
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, text_bytes);

    errno = 0;
    ok = fwrite(h, 1, sizeof(h), fp) == sizeof(h) &&
         fwrite(tb, 1, sizeof(tb), fp) == sizeof(tb) &&
         fwrite(text, 1, text_bytes, fp) == text_bytes &&
         (!(hdr.ext_flags & DS4_KVSTORE_EXT_SESSION_TITLE) ||
          agent_kv_write_title_trailer(fp, title, err, err_len)) &&
         fflush(fp) == 0;
    int saved_errno = errno;
    if (fclose(fp) != 0) {
        if (!saved_errno) saved_errno = errno;
        ok = false;
    }
    if (ok && rename(tmp, path) != 0) {
        saved_errno = errno;
        ok = false;
    }
    if (!ok) {
        snprintf(err, err_len, "%s",
                 saved_errno ? strerror(saved_errno) : "failed to write stripped session");
        unlink(tmp);
    } else {
        if (sha_out) memcpy(sha_out, sha, 41);
        if (tokens_out) *tokens_out = stripped_token_count;
    }

    free(tmp);
    free(title);
    free(text);
    free(path);
    return ok;
}

static bool agent_worker_switch_session(agent_worker *w, const char *prefix,
                                        int history_turns,
                                        char *err, size_t err_len) {
    (void)history_turns;
    pthread_mutex_lock(&w->mu);
    bool idle = w->initialized && w->status.state == AGENT_WORKER_IDLE && !w->cmd_text;
    pthread_mutex_unlock(&w->mu);
    if (!idle) {
        snprintf(err, err_len, "model is busy");
        return false;
    }
    char sha[41];
    char *path = NULL;
    if (!agent_worker_find_session(w, prefix, sha, &path, err, err_len))
        return false;

    ds4_kvstore_entry entry = {0};
    if (ds4_kvstore_read_entry_file(path, sha, &entry)) {
        ds4_kvstore_entry_free(&entry);
    }

    ds4_tokens loaded = {0};
    agent_kv_session_meta meta = {0};
    bool ok = agent_kv_load_path(w, path, sha, NULL, 0, &loaded, &meta,
                                 err, err_len);
    if (ok) {
        agent_worker_images_clear(w);
        ds4_tokens_free(&w->transcript);
        w->transcript = loaded;
        free(w->session_title);
        w->session_title = meta.title ? xstrdup(meta.title) : xstrdup("(no user prompt)");
        w->session_created_at = meta.created_at ? meta.created_at : (uint64_t)time(NULL);
        memcpy(w->session_sha, sha, sizeof(w->session_sha));
        free(w->legacy_session_path_to_delete);
        w->legacy_session_path_to_delete = meta.legacy_identity ? xstrdup(path) : NULL;
        w->datetime_context_injected = true;
        pthread_mutex_lock(&w->mu);
        w->user_activity = true;
        w->session_dirty = false;
        w->status.state = AGENT_WORKER_IDLE;
        w->status.ctx_used = w->transcript.len;
        w->status.ctx_size = agent_worker_effective_ctx_size(w);
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        w->status.error[0] = '\0';
        w->last_status_push_at = 0.0;
        pthread_mutex_unlock(&w->mu);
        agent_push_status(w, true);
    } else {
        ds4_tokens_free(&loaded);
    }
    agent_kv_session_meta_free(&meta);
    free(path);
    return ok;
}

/* Render the last user turns as plain text for /history (client paints it). */
static char *agent_worker_history_text(agent_worker *w, int user_turns,
                                       char *err, size_t err_len) {
    if (user_turns <= 0) user_turns = 3;
    if (user_turns > 200) user_turns = 200;
    size_t text_len = 0;
    char *text = ds4_kvstore_render_tokens_text(w->engine, &w->transcript,
                                                &text_len);
    if (!text) {
        snprintf(err, err_len, "failed to render session text");
        return NULL;
    }
    agent_buf b = {0};
    agent_buf_puts(&b, "--- session history ---\n");
    /* Send the rendered transcript of the last user turns; the client paints it
     * as plain text (simplification, approved). */
    agent_buf_append(&b, text, text_len);
    free(text);
    return agent_buf_take(&b);
}

/* ---- System status / render sink emission ---- */

static void agent_flush_render_sink(agent_worker *w, agent_render_sink *sink) {
    for (size_t i = 0; i < sink->len; i++) {
        agent_render_fragment *f = &sink->v[i];
        char *text = xstrndup(f->text, f->len);
        proto_token_msg m = { .id = ++w->token_seq, .kind = f->kind, .text = text };
        proto_writer wr;
        size_t out_len = 0;
        unsigned char *buf = proto_encode_token(&wr, &m, &out_len);
        (void)agent_send_all(w->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
        free(text);
    }
    sink->len = 0;
}

static void agent_publish_system_status(agent_worker *w, const char *msg) {
    agent_buf b = {0};
    agent_buf_puts(&b, "\n✦ ");
    agent_buf_puts(&b, msg);
    agent_buf_puts(&b, "\n");
    char *text = agent_buf_take(&b);
    proto_token_msg m = { .id = ++w->token_seq, .kind = PROTO_TOKEN_NORMAL, .text = text };
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf = proto_encode_token(&wr, &m, &out_len);
    (void)agent_send_all(w->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
    free(text);
}

static void agent_publishf(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    char *text = (size_t)n < sizeof(stack) ? xstrdup(stack) : xstrndup(stack, (size_t)n);
    proto_token_msg m = { .id = ++w->token_seq, .kind = PROTO_TOKEN_NORMAL, .text = text };
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf = proto_encode_token(&wr, &m, &out_len);
    (void)agent_send_all(w->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
    free(text);
}

/* ---- Tool observations (client-side execution; server commits the result) ---- */

typedef struct {
    char *text;
    size_t len;
    size_t cap;
} agent_tool_text_part;

typedef struct {
    agent_tool_text_part *parts;
    size_t part_count;
    size_t part_cap;
    ds4_vision_embedding *images;
    size_t image_count;
    size_t image_cap;
} agent_tool_observation;

static void agent_tool_observation_init(agent_tool_observation *obs) {
    memset(obs, 0, sizeof(*obs));
    obs->parts = xmalloc(sizeof(obs->parts[0]));
    memset(obs->parts, 0, sizeof(obs->parts[0]));
    obs->part_count = 1;
    obs->part_cap = 1;
}

static void agent_tool_observation_puts(agent_tool_observation *obs,
                                        const char *text) {
    agent_tool_text_part *part = &obs->parts[obs->part_count - 1];
    size_t n = text ? strlen(text) : 0;
    if (part->len + n + 1 > part->cap) {
        size_t cap = part->cap ? part->cap : 128;
        while (cap < part->len + n + 1) cap *= 2;
        part->text = xrealloc(part->text, cap);
        part->cap = cap;
    }
    if (n) memcpy(part->text + part->len, text, n);
    part->len += n;
    part->text[part->len] = '\0';
}

static void agent_tool_observation_add_image(agent_tool_observation *obs,
                                             ds4_vision_embedding *embedding) {
    if (obs->image_count == obs->image_cap) {
        size_t cap = obs->image_cap ? obs->image_cap * 2 : 2;
        obs->images = xrealloc(obs->images, cap * sizeof(obs->images[0]));
        obs->image_cap = cap;
    }
    obs->images[obs->image_count++] = *embedding;
    memset(embedding, 0, sizeof(*embedding));
    if (obs->part_count == obs->part_cap) {
        size_t cap = obs->part_cap ? obs->part_cap * 2 : 2;
        obs->parts = xrealloc(obs->parts, cap * sizeof(obs->parts[0]));
        obs->part_cap = cap;
    }
    memset(&obs->parts[obs->part_count++], 0, sizeof(obs->parts[0]));
}

static void agent_tool_observation_free(agent_tool_observation *obs) {
    if (!obs) return;
    for (size_t i = 0; i < obs->part_count; i++) free(obs->parts[i].text);
    for (size_t i = 0; i < obs->image_count; i++)
        ds4_vision_embedding_free(&obs->images[i]);
    free(obs->parts);
    free(obs->images);
    memset(obs, 0, sizeof(*obs));
}

static void agent_vision_spans_free(ds4_vision_span *spans, size_t count) {
    for (size_t i = 0; i < count; i++)
        ds4_vision_embedding_free(&spans[i].embedding);
    free(spans);
}

static bool agent_tool_observation_build(agent_worker *w,
                                         const agent_tool_observation *obs,
                                         ds4_tokens *tokens,
                                         ds4_vision_span **spans_out,
                                         char *err, size_t err_len) {
    const char **parts = xmalloc(obs->part_count * sizeof(parts[0]));
    for (size_t i = 0; i < obs->part_count; i++)
        parts[i] = obs->parts[i].text ? obs->parts[i].text : "";
    ds4_vision_embedding *images = NULL;
    ds4_vision_span *spans = NULL;
    if (obs->image_count) {
        images = xmalloc(obs->image_count * sizeof(images[0]));
        spans = xmalloc(obs->image_count * sizeof(spans[0]));
        memset(images, 0, obs->image_count * sizeof(images[0]));
        memset(spans, 0, obs->image_count * sizeof(spans[0]));
        const int n_embd = ds4_engine_embd_dim(w->engine);
        for (size_t i = 0; i < obs->image_count; i++) {
            const ds4_vision_embedding *src = &obs->images[i];
            if (!src->data || src->token_count == 0 || n_embd <= 0 ||
                (uint64_t)src->token_count >
                    SIZE_MAX / (uint64_t)n_embd / sizeof(float)) {
                snprintf(err, err_len, "invalid image observation embedding");
                for (size_t j = 0; j < i; j++)
                    ds4_vision_embedding_free(&images[j]);
                free(images);
                free(spans);
                free(parts);
                ds4_tokens_free(tokens);
                return false;
            }
            const size_t bytes = (size_t)src->token_count *
                                 (size_t)n_embd * sizeof(float);
            images[i] = *src;
            images[i].data = malloc(bytes);
            if (!images[i].data) {
                snprintf(err, err_len, "unable to copy image observation embedding");
                for (size_t j = 0; j <= i; j++)
                    ds4_vision_embedding_free(&images[j]);
                free(images);
                free(spans);
                free(parts);
                ds4_tokens_free(tokens);
                return false;
            }
            memcpy(images[i].data, src->data, bytes);
        }
    }
    ds4_tokens_copy(tokens, &w->transcript);
    /* GLM grounds image tokens in user turns; keep text-only observations in
     * the native tool-response role used by the rest of the agent protocol. */
    bool ok = ds4_chat_append_multimodal_message(
        w->engine, tokens, obs->image_count ? "user" : "tool",
        parts, images, obs->image_count, spans,
        err, err_len) != 0;
    free(parts);
    if (!ok) {
        for (size_t i = 0; i < obs->image_count; i++) {
            ds4_vision_embedding_free(&images[i]);
            ds4_vision_embedding_free(&spans[i].embedding);
        }
        free(images);
        free(spans);
        ds4_tokens_free(tokens);
        return false;
    }
    free(images);
    *spans_out = spans;
    return true;
}

/* -1 is a rendering/embedding error, not a request to compact the context. */
static int agent_tool_observation_fits(agent_worker *w,
                                       const agent_tool_observation *obs,
                                       int reserve_tokens,
                                       int *tokens_out,
                                       char *err, size_t err_len) {
    ds4_tokens tmp = {0};
    ds4_vision_span *spans = NULL;
    if (err_len) err[0] = '\0';
    if (!agent_tool_observation_build(w, obs, &tmp, &spans,
                                      err, err_len))
        return -1;
    int tokens = tmp.len;
    ds4_tokens_free(&tmp);
    agent_vision_spans_free(spans, obs->image_count);
    if (tokens_out) *tokens_out = tokens;
    int ctx = agent_worker_effective_ctx_size(w);
    return ctx > 0 && tokens + reserve_tokens < ctx;
}

static bool agent_tool_observation_commit(agent_worker *w,
                                          agent_tool_observation *obs,
                                          char *err, size_t err_len) {
    ds4_tokens next = {0};
    ds4_vision_span *spans = NULL;
    if (!agent_tool_observation_build(w, obs, &next, &spans, err, err_len))
        return false;
    ds4_tokens_free(&w->transcript);
    w->transcript = next;
    agent_worker_images_append(w, spans, obs->image_count);
    free(spans);
    return true;
}

static int agent_compact_reserve_tokens(agent_worker *w);

static int agent_tool_result_reserve_tokens(agent_worker *w) {
    int ctx = agent_worker_effective_ctx_size(w);
    int reserve = AGENT_TOOL_RESULT_RESERVE_TOKENS;
    if (ctx > 0) {
        int proportional = ctx / 8;
        if (proportional < 16) proportional = 16;
        if (reserve > proportional) reserve = proportional;
    }
    int compact = agent_compact_reserve_tokens(w) + 128;
    return reserve > compact ? reserve : compact;
}

/* ---- Context compaction (ported from ds4_agent.c) ---- */

static char *agent_compact_make_prompt(const char *reason) {
    agent_buf b = {0};
    agent_buf_puts(&b,
        "Internal ds4-agent context compaction request. This is not a user request.\n"
        "Write a durable task-state summary of the conversation so far. Preserve only facts that matter for continuing the work:\n"
        "- user goals, constraints, and preferences\n"
        "- files inspected or edited\n"
        "- commands run and important results\n"
        "- decisions, rejected approaches, known bugs, and pending next steps\n"
        "- reloadable bulky data with exact paths/ranges/commands when available\n\n"
        "Do not invent facts. Do not include generic narration. Do not include raw file contents unless they were essential to a conclusion.\n"
        "Do not solve unfinished tasks, calculate new results, or finish incomplete code in the summary. "
        "Record the user's requirements and what actually happened, leaving unfinished work pending.\n"
        "After the summary, stop. Do not continue the user task, do not call tools, and do not output thinking tags or DSML markup.\n"
        "Output only the compact summary.\n");
    if (reason && reason[0]) {
        agent_buf_puts(&b, "\nCompaction reason: ");
        agent_buf_puts(&b, reason);
        agent_buf_puts(&b, "\n");
    }
    return agent_buf_take(&b);
}

static int agent_compact_summary_budget(int ctx) {
    int budget = ctx / 8;
    if (budget < 256) budget = 256;
    if (budget > AGENT_COMPACT_SUMMARY_MAX_TOKENS)
        budget = AGENT_COMPACT_SUMMARY_MAX_TOKENS;
    return budget;
}

static int agent_compact_reserve_tokens(agent_worker *w) {
    char *text = agent_compact_make_prompt("context pressure before continuing the current task");
    ds4_tokens tokens = {0};
    ds4_chat_append_message(w->engine, &tokens, "user", text);
    ds4_chat_append_assistant_prefix(w->engine, &tokens, DS4_THINK_NONE);
    int reserve = tokens.len + 64 +
                  agent_compact_summary_budget(agent_worker_effective_ctx_size(w));
    free(text);
    ds4_tokens_free(&tokens);
    return reserve;
}

static int agent_special_token_id(ds4_engine *engine, const char *rendered) {
    ds4_tokens t = {0};
    ds4_tokenize_rendered_chat(engine, rendered, &t);
    int id = t.len == 1 ? t.v[0] : -1;
    ds4_tokens_free(&t);
    return id;
}

static int agent_compact_tail_boundary(const ds4_tokens *tokens, int bottom,
                                       int sys_len, int tail_budget, int user_id) {
    int target = bottom - tail_budget;
    if (target < sys_len) target = sys_len;
    if (user_id < 0) return target;

    int earliest = bottom - 2 * tail_budget;
    if (earliest < sys_len) earliest = sys_len;
    for (int i = bottom - 1; i >= earliest; i--) {
        if (tokens->v[i] == user_id) return i;
    }
    return target;
}

static int agent_compact_tail_start(agent_worker *w, int bottom, int sys_len) {
    int tail_budget = agent_worker_effective_ctx_size(w) / AGENT_COMPACT_TAIL_DIVISOR;
    if (tail_budget > AGENT_COMPACT_TAIL_CAP_TOKENS)
        tail_budget = AGENT_COMPACT_TAIL_CAP_TOKENS;
    if (tail_budget < 1) tail_budget = 1;
    int user_id = agent_special_token_id(w->engine,
        ds4_engine_is_glm_dsa(w->engine) ? "<|user|>" : "<" AGENT_DSML_BAR "User" AGENT_DSML_BAR ">");
    return agent_compact_tail_boundary(&w->transcript, bottom, sys_len, tail_budget, user_id);
}

static int agent_compact_image_boundary(const ds4_vision_span *images, size_t count,
                                        bool glm, int pos) {
    for (size_t i = 0; i < count; i++) {
        const ds4_vision_span *span = &images[i];
        int wrapper = glm ? 1 : 0;
        int start = (int)span->token_start - wrapper;
        uint64_t end = (uint64_t)span->token_start +
                       span->embedding.token_count + (unsigned)wrapper;
        if (pos > start && (uint64_t)pos < end) return start;
    }
    return pos;
}

static bool agent_worker_should_compact(agent_worker *w) {
    int ctx = agent_worker_effective_ctx_size(w);
    int used = w->transcript.len;
    if (ctx <= 0 || used <= 0) return false;
    if (used >= (ctx * AGENT_COMPACT_SOFT_PERCENT) / 100) return true;
    int free_threshold = AGENT_COMPACT_MIN_FREE_TOKENS;
    int proportional = ctx / 8;
    if (free_threshold > proportional) free_threshold = proportional;
    int reserve = agent_compact_reserve_tokens(w) + 128;
    if (free_threshold < reserve) free_threshold = reserve;
    return ctx - used <= free_threshold;
}

static bool agent_worker_compact_transcript(agent_worker *w, const char *reason,
                                            int *open_assistant,
                                            char *err, size_t err_len) {
    const int bottom = w->transcript.len;
    if (bottom <= 0) return true;

    ds4_tokens sys = {0};
    agent_worker_build_system_tokens(w, &sys);
    if (bottom <= sys.len) {
        ds4_tokens_free(&sys);
        return true;
    }

    agent_publishf(w, "COMPACTING %s: summarizing durable task state\n",
                   reason && reason[0] ? reason : "context");

    char *prompt_text = agent_compact_make_prompt(reason);
    ds4_tokens summary_suffix = {0};
    ds4_chat_append_message(w->engine, &summary_suffix, "user", prompt_text);
    free(prompt_text);
    ds4_chat_append_assistant_prefix(w->engine, &summary_suffix, DS4_THINK_NONE);
    const int ctx = agent_worker_effective_ctx_size(w);
    int summary_budget = agent_compact_summary_budget(ctx);
    int summary_bottom = bottom;
    if (summary_bottom > ctx - summary_suffix.len - summary_budget - 1)
        summary_bottom = ctx - summary_suffix.len - summary_budget - 1;
    summary_bottom = agent_compact_image_boundary(w->images, w->image_count,
                        ds4_engine_is_glm_dsa(w->engine), summary_bottom);
    if (summary_bottom <= sys.len) {
        snprintf(err, err_len, "context too small to summarize this conversation; use a larger --ctx");
        ds4_tokens_free(&summary_suffix);
        ds4_tokens_free(&sys);
        return false;
    }
    ds4_tokens prompt = {0};
    agent_tokens_append_range(&prompt, &w->transcript, 0, summary_bottom);
    agent_tokens_append_range(&prompt, &summary_suffix, 0, summary_suffix.len);
    ds4_tokens_free(&summary_suffix);
    size_t summary_images = 0;
    while (summary_images < w->image_count &&
           (uint64_t)w->images[summary_images].token_start +
             w->images[summary_images].embedding.token_count <= (uint64_t)summary_bottom)
        summary_images++;

    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_COMPACTING;
    w->progress_direct = false;
    w->progress_started_at = now_sec();
    w->status.prefill_done = 0;
    w->status.prefill_total = 0;
    w->status.prefill_tps = 0.0;
    w->status.generated = 0;
    w->status.gen_tps = 0.0;
    w->status.greedy_sampling = false;
    w->last_status_push_at = 0.0;
    pthread_mutex_unlock(&w->mu);
    agent_push_status(w, true);

    int summary_room = agent_worker_effective_ctx_size(w) - prompt.len - 1;
    if (summary_room < 256) {
        snprintf(err, err_len, "not enough context left to request compaction summary");
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        return false;
    }
    int summary_max = summary_room < summary_budget ? summary_room : summary_budget;

    ds4_session_set_progress(w->session, worker_progress_cb, w);
    ds4_session_set_display_progress(w->session, worker_progress_cb, w);
    ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
    int sync_rc;
    if (summary_images) {
        sync_rc = ds4_session_sync_multimodal(w->session, &prompt,
                                              w->images, summary_images,
                                              err, err_len);
    } else {
        sync_rc = ds4_session_sync(w->session, &prompt, err, err_len);
    }
    ds4_session_set_cancel(w->session, NULL, NULL);
    ds4_session_set_progress(w->session, NULL, NULL);
    ds4_session_set_display_progress(w->session, NULL, NULL);
    if (sync_rc == DS4_SESSION_SYNC_INTERRUPTED) {
        ds4_session_invalidate(w->session);
        snprintf(err, err_len, "interrupted");
        agent_publish_system_status(w, "Compaction interrupted; keeping the previous conversation state.");
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        worker_clear_interrupt(w);
        return false;
    }
    if (sync_rc != 0) {
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&prompt);
        ds4_tokens_free(&sys);
        return false;
    }

    agent_buf summary = {0};
    char eval_err[160] = {0};
    int dsml_id = agent_special_token_id(w->engine,
        "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR ">");
    double t0 = now_sec();
    for (int i = 0; i < summary_max; i++) {
        if (worker_should_interrupt(w)) {
            snprintf(err, err_len, "interrupted");
            ds4_session_invalidate(w->session);
            ds4_tokens_free(&prompt);
            ds4_tokens_free(&sys);
            free(summary.ptr);
            agent_publish_system_status(w, "Compaction interrupted; keeping the previous conversation state.");
            worker_clear_interrupt(w);
            return false;
        }
        int token = ds4_session_argmax(w->session);
        if (ds4_token_is_stop_for_think_mode(w->engine,
                                             token,
                                             DS4_THINK_NONE) ||
            token == dsml_id) {
            if (token == dsml_id && summary.len && summary.ptr[summary.len - 1] == '<') {
                summary.ptr[--summary.len] = '\0';
            }
            break;
        }
        if (ds4_session_eval(w->session, token, eval_err, sizeof(eval_err)) != 0) {
            snprintf(err, err_len, "%s", eval_err);
            ds4_session_invalidate(w->session);
            ds4_tokens_free(&prompt);
            ds4_tokens_free(&sys);
            free(summary.ptr);
            return false;
        }

        size_t text_len = 0;
        char *text = ds4_token_text(w->engine, token, &text_len);
        agent_buf_append(&summary, text, text_len);
        /* stream the summary token to the client for display */
        char *ntext = xstrndup(text, text_len);
        proto_token_msg m = { .id = ++w->token_seq, .kind = PROTO_TOKEN_NORMAL, .text = ntext };
        proto_writer wr;
        size_t out_len = 0;
        unsigned char *buf = proto_encode_token(&wr, &m, &out_len);
        (void)agent_send_all(w->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
        free(ntext);
        free(text);

        double dt = now_sec() - t0;
        pthread_mutex_lock(&w->mu);
        w->status.generated = i + 1;
        w->status.gen_tps = dt > 0.0 ? (double)(i + 1) / dt : 0.0;
        w->status.greedy_sampling = false;
        pthread_mutex_unlock(&w->mu);
        agent_maybe_push_status(w);
    }
    ds4_tokens_free(&prompt);

    if (!summary.ptr || !summary.ptr[0]) {
        snprintf(err, err_len, "compaction summary was empty");
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&sys);
        free(summary.ptr);
        return false;
    }

    int tail_start = agent_compact_tail_start(w, bottom, sys.len);
    if (tail_start > summary_bottom) tail_start = summary_bottom;
    tail_start = agent_compact_image_boundary(w->images, w->image_count,
                    ds4_engine_is_glm_dsa(w->engine), tail_start);
    ds4_tokens compacted = {0};
    ds4_tokens_copy(&compacted, &sys);

    agent_buf summary_msg = {0};
    agent_buf_puts(&summary_msg,
        "\n\n[ds4-agent compacted earlier conversation. Durable task-state summary follows.]\n");
    agent_buf_puts(&summary_msg, summary.ptr);
    if (summary_msg.len && summary_msg.ptr[summary_msg.len - 1] != '\n')
        agent_buf_puts(&summary_msg, "\n");
    agent_buf_puts(&summary_msg, "[End compacted summary. Recent conversation continues verbatim below.]\n\n");
    ds4_chat_append_message(w->engine, &compacted, "user", summary_msg.ptr);
    free(summary_msg.ptr);
    free(summary.ptr);

    int resumed_start = -1;
    if (open_assistant && tail_start > *open_assistant) {
        int think_start = agent_special_token_id(w->engine, "<think>");
        int think_end = agent_special_token_id(w->engine, "</think>");
        bool in_think = false;
        for (int i = *open_assistant; i < tail_start; i++) {
            if (w->transcript.v[i] == think_start) in_think = true;
            if (w->transcript.v[i] == think_end) in_think = false;
        }
        resumed_start = compacted.len;
        ds4_chat_append_assistant_prefix(w->engine, &compacted,
            in_think ? DS4_THINK_HIGH : DS4_THINK_NONE);
    } else if (tail_start < bottom &&
        w->transcript.v[tail_start] != ds4_token_user(w->engine) &&
        w->transcript.v[tail_start] != ds4_token_assistant(w->engine)) {
        ds4_chat_append_message(w->engine, &compacted, "user",
            "[Verbatim tail of the previous conversation; it may begin mid-message.]\n");
    }
    const int tail_dst_start = compacted.len;
    if (open_assistant && resumed_start < 0)
        resumed_start = tail_dst_start + *open_assistant - tail_start;
    agent_tokens_append_range(&compacted, &w->transcript, tail_start, bottom);
    if (compacted.len >= ctx - agent_compact_reserve_tokens(w) - 128) {
        snprintf(err, err_len, "compacted conversation leaves no working room; use a larger --ctx");
        ds4_tokens_free(&compacted);
        ds4_tokens_free(&sys);
        ds4_session_invalidate(w->session);
        return false;
    }

    ds4_vision_span *old_images = w->images;
    size_t old_image_count = w->image_count;
    size_t old_image_cap = w->image_cap;
    ds4_vision_span *new_images = old_image_count ?
        xmalloc(old_image_count * sizeof(new_images[0])) : NULL;
    bool *kept_images = old_image_count ?
        calloc(old_image_count, sizeof(kept_images[0])) : NULL;
    size_t new_image_count = 0;
    for (size_t i = 0; i < old_image_count; i++) {
        uint64_t image_end = (uint64_t)old_images[i].token_start +
                             old_images[i].embedding.token_count;
        if (old_images[i].token_start >= (uint32_t)tail_start &&
            image_end <= (uint64_t)bottom) {
            new_images[new_image_count] = old_images[i];
            new_images[new_image_count].token_start =
                (uint32_t)(tail_dst_start +
                           (int)old_images[i].token_start - tail_start);
            new_image_count++;
            kept_images[i] = true;
        }
    }

    agent_publishf(w, "COMPACTING rebuilding context: old=%d summary+tail=%d tail=%d\n",
                   bottom, compacted.len, bottom - tail_start);

    ds4_tokens old_transcript = {0};
    ds4_tokens_copy(&old_transcript, &w->transcript);
    ds4_tokens_free(&w->transcript);
    w->transcript = compacted;
    w->images = new_images;
    w->image_count = new_image_count;
    w->image_cap = old_image_count;
    if (agent_worker_sync_tokens(w, &w->transcript, true, err, err_len) != 0) {
        ds4_session_invalidate(w->session);
        ds4_tokens_free(&w->transcript);
        w->transcript = old_transcript;
        free(new_images);
        w->images = old_images;
        w->image_count = old_image_count;
        w->image_cap = old_image_cap;
        free(kept_images);
        ds4_tokens_free(&sys);
        return false;
    }
    for (size_t i = 0; i < old_image_count; i++) {
        if (!kept_images[i])
            ds4_vision_embedding_free(&old_images[i].embedding);
    }
    free(old_images);
    free(kept_images);
    agent_worker_note_system_prompt_seen(w);
    ds4_tokens_free(&sys);
    return true;
}

static bool agent_worker_compact(agent_worker *w, const char *reason,
                                 char *err, size_t err_len) {
    return agent_worker_compact_transcript(w, reason, NULL, err, err_len);
}

static bool agent_worker_compact_if_needed(agent_worker *w, const char *reason,
                                           char *err, size_t err_len) {
    if (!agent_worker_should_compact(w)) return true;
    return agent_worker_compact(w, reason, err, err_len);
}

/* ---- User turn helpers ---- */

static bool agent_worker_append_user(agent_worker *w, const char *text,
                                     char *err, size_t err_len) {
    ds4_tokens message = {0}, sys = {0};
    ds4_chat_append_message(w->engine, &message, "user", text);
    agent_worker_build_system_tokens(w, &sys);
    int ctx = agent_worker_effective_ctx_size(w);
    bool ok = message.len < ctx - sys.len - 4;
    ds4_tokens_free(&sys);
    if (!ok) {
        snprintf(err, err_len, "user message is too large for --ctx %d; use a larger context or a smaller message", ctx);
    } else if (message.len >= ctx - w->transcript.len - 4) {
        ok = agent_worker_compact(w, "make room for incoming user message", err, err_len);
        if (ok && message.len >= ctx - w->transcript.len - 4) {
            snprintf(err, err_len, "user message does not fit after compaction; use a larger --ctx");
            ok = false;
        }
    }
    if (ok) agent_tokens_append_range(&w->transcript, &message, 0, message.len);
    ds4_tokens_free(&message);
    return ok;
}

static int agent_worker_rewind(agent_worker *w, int pos,
                               char *err, size_t err_len) {
    ds4_session_rewind(w->session, pos);
    ds4_tokens prefix = {0};
    ds4_tokens_copy(&prefix, ds4_session_tokens(w->session));
    int rc = 0;
    if (ds4_session_common_prefix(w->session, &prefix) != prefix.len) {
        rc = agent_worker_sync_tokens(w, &prefix, false, err, err_len);
    }
    ds4_tokens_free(&prefix);
    return rc;
}

static int worker_finish_generated_token(agent_worker *w, int token, int *generated,
                                         double t0, agent_stream_renderer *stream,
                                         bool evaluate, char *err, size_t err_len) {
    ds4_tokens_push(&w->transcript, token);

    size_t text_len = 0;
    char *text = ds4_token_text(w->engine, token, &text_len);
    agent_stream_text(stream, text, text_len, false);
    agent_flush_render_sink(w, &stream->sink);
    free(text);
    (*generated)++;

    if (evaluate &&
        ds4_session_eval(w->session, token, err, err_len) != 0) {
        ds4_session_invalidate(w->session);
        return 1;
    }

    double dt = now_sec() - t0;
    pthread_mutex_lock(&w->mu);
    w->status.generated = *generated;
    w->status.gen_tps = dt > 0.0 ? (double)*generated / dt : 0.0;
    pthread_mutex_unlock(&w->mu);
    agent_maybe_push_status(w);
    return 0;
}

static int worker_sample_with_mode(agent_worker *w, const agent_config *cfg,
                                   bool greedy, uint64_t *rng) {
    return ds4_session_sample(w->session,
                              greedy ? 0.0f : cfg->gen.temperature,
                              0,
                              greedy ? 1.0f : cfg->gen.top_p,
                              greedy ? 0.0f : cfg->gen.min_p,
                              rng);
}

static void worker_set_greedy_sampling(agent_worker *w, bool greedy) {
    pthread_mutex_lock(&w->mu);
    if (w->status.greedy_sampling != greedy) {
        w->status.greedy_sampling = greedy;
        w->last_status_push_at = 0.0;
        pthread_mutex_unlock(&w->mu);
        dbg_log('S', "greedy %s", greedy ? "ON" : "OFF");
        agent_push_status(w, true);
    } else {
        pthread_mutex_unlock(&w->mu);
    }
}

#ifndef DS4_AGENT_TEST

/* ---- Syntax reminders (malformed tool-call observations) ---- */

static const char agent_dsml_syntax_reminder[] =
    "DSML syntax reminder:\n"
    "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>\n"
    "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke name=\"$TOOL_NAME\">\n"
    "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter>\n"
    "</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke>\n"
    "</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>\n";

static const char agent_glm_syntax_reminder[] =
    "GLM tool-call syntax reminder:\n"
    "<tool_call>$TOOL_NAME<arg_key>$PARAMETER_NAME</arg_key>"
    "<arg_value>$PARAMETER_VALUE</arg_value></tool_call>\n";

/* ---- CLI (engine options minus UI/tool opts + --host/--port) ---- */

static int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-agent-server: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static ds4_backend parse_backend(const char *s) {
    if (!strcmp(s, "metal")) return DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
    if (!strcmp(s, "rocm")) return DS4_BACKEND_CUDA;
#else
    if (!strcmp(s, "cuda")) return DS4_BACKEND_CUDA;
#endif
    if (!strcmp(s, "cpu")) return DS4_BACKEND_CPU;
    fprintf(stderr, "ds4-agent-server: invalid backend: %s\n", s);
#ifdef DS4_ROCM_BUILD
    fprintf(stderr, "ds4-agent-server: valid backends are: metal, rocm, cpu\n");
#else
    fprintf(stderr, "ds4-agent-server: valid backends are: metal, cuda, cpu\n");
#endif
    exit(2);
}

static ds4_backend default_backend(void) {
#ifdef DS4_NO_GPU
    return DS4_BACKEND_CPU;
#elif defined(__APPLE__)
    return DS4_BACKEND_METAL;
#else
    return DS4_BACKEND_CUDA;
#endif
}

static void usage(FILE *fp) {
    fprintf(fp,
        "usage: ds4-agent-server [options]\n"
        "  --host <addr>     listen address (default 127.0.0.1)\n"
        "  --port <port>     listen port (default 9334)\n"
        "  -m, --model FILE  model path (default ds4flash.gguf)\n"
        "  -c, --ctx N       context size (default 100000)\n"
        "  -n, --tokens N    max generated tokens per turn (default 50000)\n"
        "  --temp F, --top-p F, --min-p F   sampling parameters\n"
        "  --seed N          sampling seed\n"
        "  --think, --think-max, --nothink  thinking mode\n"
        "  --backend B       metal/cuda/cpu backend\n"
        "  --power N         power limit percent (1-100)\n"
        "  --vision FILE     vision model path\n"
        "  --mtp, --mtp-model FILE, --mtp-draft N, --mtp-margin F\n"
        "  --gpu-vram ARG, --gpu-devices ARG   CUDA GPU config\n"
        "  --ssd-streaming, --prefill-chunk N, --warm-weights\n"
        "  --dir-steering-file FILE, --dir-steering-ffn F, --dir-steering-attn F\n"
        "  -t, --threads N   CPU threads\n"
        "  --quality, --cpu, --metal\n");
}

static agent_config parse_options(int argc, char **argv) {
    agent_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .gen = {
            .system = "You are a helpful coding assistant running inside ds4-agent-server.",
            .n_predict = 50000,
            .ctx_size = 100000,
            .temperature = DS4_DEFAULT_TEMPERATURE,
            .top_p = DS4_DEFAULT_TOP_P,
            .min_p = DS4_DEFAULT_MIN_P,
            .think_mode = DS4_THINK_HIGH,
        },
        .host = "127.0.0.1",
        .port = 9334,
    };

    bool steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            usage(stdout);
            exit(0);
        }
        char dist_parse_err[256] = {0};
        ds4_dist_cli_parse_result dist_parse =
            ds4_dist_parse_cli_arg(arg, &i, argc, argv, &c.engine.distributed,
                                   dist_parse_err, sizeof(dist_parse_err));
        if (dist_parse == DS4_DIST_CLI_ERROR) {
            fprintf(stderr, "ds4-agent-server: %s\n",
                    dist_parse_err[0] ? dist_parse_err : "invalid distributed option");
            exit(2);
        }
        if (dist_parse == DS4_DIST_CLI_MATCHED) continue;

        char tp_parse_err[256] = {0};
        ds4_tp_cli_parse_result tp_parse =
            ds4_tp_parse_cli_arg(arg, &i, argc, argv, &c.engine.tp,
                                 tp_parse_err, sizeof(tp_parse_err));
        if (tp_parse == DS4_TP_CLI_ERROR) {
            fprintf(stderr, "ds4-agent-server: %s\n",
                    tp_parse_err[0] ? tp_parse_err : "invalid tensor-parallel option");
            exit(2);
        }
        if (tp_parse == DS4_TP_CLI_MATCHED) continue;

        if (!strcmp(arg, "-sys") || !strcmp(arg, "--system")) {
            c.gen.system = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "-m") || !strcmp(arg, "--model")) {
            c.engine.model_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--vision")) {
            c.engine.vision_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp")) {
            c.engine.glm_mtp = true;
        } else if (!strcmp(arg, "--mtp-model")) {
            c.engine.mtp_path = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--mtp-draft")) {
            c.engine.mtp_draft_tokens = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--mtp-margin")) {
            c.engine.mtp_margin = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1000.0f);
        } else if (!strcmp(arg, "--mtp-timing")) {
            c.engine.glm_mtp = true;
            c.engine.glm_mtp_timing = true;
        } else if (!strcmp(arg, "--dspark")) {
            c.engine.dspark = true;
        } else if (!strcmp(arg, "--dspark-confidence")) {
            c.engine.dspark = true;
            c.engine.dspark_confidence_threshold =
                parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.engine.dspark_confidence_threshold_set = true;
        } else if (!strcmp(arg, "--dspark-strict")) {
            c.engine.dspark = true;
            c.engine.dspark_strict = true;
        } else if (!strcmp(arg, "--mtp-exact-sampling")) {
            c.engine.dspark_exact_sampling = true;
        } else if (!strcmp(arg, "-c") || !strcmp(arg, "--ctx")) {
            c.gen.ctx_size = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "-n") || !strcmp(arg, "--tokens")) {
            c.gen.n_predict = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--temp")) {
            c.gen.temperature = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 100.0f);
            c.gen.temperature_set = true;
        } else if (!strcmp(arg, "--top-p")) {
            c.gen.top_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.gen.top_p_set = true;
        } else if (!strcmp(arg, "--min-p")) {
            c.gen.min_p = parse_float_range(need_arg(&i, argc, argv, arg), arg, 0.0f, 1.0f);
            c.gen.min_p_set = true;
        } else if (!strcmp(arg, "--seed")) {
            c.gen.seed = parse_u64(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--think")) {
            c.gen.think_mode = DS4_THINK_HIGH;
        } else if (!strcmp(arg, "--think-max")) {
            c.gen.think_mode = DS4_THINK_MAX;
        } else if (!strcmp(arg, "--nothink")) {
            c.gen.think_mode = DS4_THINK_NONE;
        } else if (!strcmp(arg, "--backend")) {
            c.engine.backend = parse_backend(need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--metal")) {
            c.engine.backend = DS4_BACKEND_METAL;
#ifdef DS4_ROCM_BUILD
        } else if (!strcmp(arg, "--rocm")) {
            c.engine.backend = DS4_BACKEND_CUDA;
#else
        } else if (!strcmp(arg, "--cuda")) {
            c.engine.backend = DS4_BACKEND_CUDA;
#endif
        } else if (!strcmp(arg, "--gpu-vram")) {
            c.gpu_vram_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--gpu-devices")) {
            c.gpu_devices_arg = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--cuda-tensor-parallel")) {
            c.engine.cuda_tensor_parallel = true;
        } else if (!strcmp(arg, "--cpu")) {
            c.engine.backend = DS4_BACKEND_CPU;
        } else if (!strcmp(arg, "-t") || !strcmp(arg, "--threads")) {
            c.engine.n_threads = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--quality")) {
            c.engine.quality = true;
        } else if (!strcmp(arg, "--ssd-streaming")) {
            c.engine.ssd_streaming = true;
        } else if (!strcmp(arg, "--ssd-streaming-cold")) {
            c.engine.ssd_streaming_cold = true;
        } else if (!strcmp(arg, "--ssd-streaming-cache-experts")) {
            uint32_t experts = 0;
            uint64_t bytes = 0;
            if (!ds4_parse_streaming_cache_experts_arg(
                    need_arg(&i, argc, argv, arg), &experts, &bytes)) {
                fprintf(stderr,
                        "ds4-agent-server: --ssd-streaming-cache-experts must be a positive count or <number>GB\n");
                exit(2);
            }
            c.engine.ssd_streaming_cache_experts = experts;
            c.engine.ssd_streaming_cache_bytes = bytes;
        } else if (!strcmp(arg, "--ssd-streaming-full-layers")) {
            int v = parse_nonnegative_int(need_arg(&i, argc, argv, arg), arg);
            c.engine.ssd_streaming_full_layers = (uint32_t)v;
            c.engine.ssd_streaming_full_layers_set = true;
        } else if (!strcmp(arg, "--ssd-streaming-preload-experts")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-agent-server: --ssd-streaming-preload-experts must be positive\n");
                exit(2);
            }
            c.engine.ssd_streaming_preload_experts = (uint32_t)v;
        } else if (!strcmp(arg, "--simulate-used-memory")) {
            if (!ds4_parse_gib_arg(need_arg(&i, argc, argv, arg),
                                   &c.engine.simulate_used_memory_bytes)) {
                fprintf(stderr,
                        "ds4-agent-server: --simulate-used-memory must be a positive GiB value, e.g. 64GB\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--prefill-chunk")) {
            int v = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (v <= 0) {
                fprintf(stderr, "ds4-agent-server: --prefill-chunk must be positive\n");
                exit(2);
            }
            c.engine.prefill_chunk = (uint32_t)v;
        } else if (!strcmp(arg, "--power")) {
            c.engine.power_percent = parse_int(need_arg(&i, argc, argv, arg), arg);
            if (c.engine.power_percent < 1 || c.engine.power_percent > 100) {
                fprintf(stderr, "ds4-agent-server: --power must be between 1 and 100\n");
                exit(2);
            }
        } else if (!strcmp(arg, "--warm-weights")) {
            c.engine.warm_weights = true;
        } else if (!strcmp(arg, "--dir-steering-file")) {
            c.engine.directional_steering_file = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--dir-steering-ffn")) {
            c.engine.directional_steering_ffn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn = parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else if (!strcmp(arg, "--host")) {
            c.host = need_arg(&i, argc, argv, arg);
        } else if (!strcmp(arg, "--port")) {
            c.port = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else {
            fprintf(stderr, "ds4-agent-server: unknown option: %s\n", arg);
            usage(stderr);
            exit(2);
        }
    }

    if (c.engine.directional_steering_file && !steering_scale_set)
        c.engine.directional_steering_ffn = 1.0f;
    char tp_err[256];
    if (!ds4_tp_adopt_distributed_options(&c.engine.tp, &c.engine.distributed,
                                          tp_err, sizeof(tp_err))) {
        fprintf(stderr, "ds4-agent-server: %s\n", tp_err);
        exit(2);
    }
    char dist_err[256];
    if (ds4_dist_prepare_engine_options(&c.engine.distributed, &c.engine,
                                        dist_err, sizeof(dist_err)) != 0) {
        fprintf(stderr, "ds4-agent-server: %s\n", dist_err);
        exit(2);
    }
    if (!ds4_tp_validate_engine_options(&c.engine, tp_err, sizeof(tp_err))) {
        fprintf(stderr, "ds4-agent-server: %s\n", tp_err);
        exit(2);
    }
    if (c.engine.distributed.role == DS4_DISTRIBUTED_WORKER ||
        c.engine.tp.role == DS4_TP_WORKER) {
        fprintf(stderr, "ds4-agent-server: --role worker is a serving mode; start workers with ./ds4\n");
        exit(2);
    }
    return c;
}

/* ---- Worker request / answer helpers ---- */

static void worker_request_save(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->save_requested = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

static void worker_request_compact(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->compact_requested = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

static void worker_request_power(agent_worker *w, int power) {
    pthread_mutex_lock(&w->mu);
    w->requested_power = power;
    w->power_requested = true;
    w->status.power_percent = power;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

static bool worker_take_save_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->save_requested;
    w->save_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}

static bool worker_take_compact_requested(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->compact_requested;
    w->compact_requested = false;
    pthread_mutex_unlock(&w->mu);
    return requested;
}

static char *worker_request_queued_user_drain(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->queued_user_pending = true;
    w->queued_user_answered = false;
    free(w->queued_user_text);
    w->queued_user_text = NULL;
    pthread_cond_signal(&w->cond);
    while (!w->stop && !w->queued_user_answered)
        pthread_cond_wait(&w->cond, &w->mu);
    char *text = w->queued_user_text;
    w->queued_user_text = NULL;
    w->queued_user_pending = false;
    w->queued_user_answered = false;
    pthread_mutex_unlock(&w->mu);
    return text;
}

static void worker_answer_queued_user_drain(agent_worker *w, char *text) {
    pthread_mutex_lock(&w->mu);
    free(w->queued_user_text);
    w->queued_user_text = text;
    w->queued_user_answered = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

static char *worker_request_tool_result(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->tool_result_pending = true;
    w->tool_result_answered = false;
    free(w->tool_result_text);
    w->tool_result_text = NULL;
    pthread_cond_signal(&w->cond);
    while (!w->stop && !w->tool_result_answered)
        pthread_cond_wait(&w->cond, &w->mu);
    char *text = w->tool_result_text;
    w->tool_result_text = NULL;
    w->tool_result_pending = false;
    w->tool_result_answered = false;
    pthread_mutex_unlock(&w->mu);
    return text;
}

static void worker_answer_tool_result(agent_worker *w, char *text) {
    pthread_mutex_lock(&w->mu);
    free(w->tool_result_text);
    w->tool_result_text = text;
    w->tool_result_answered = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

/* ---- Tool execution over the wire ----
 * The client executes the parsed calls.  The server sends TOOL_CALLS then
 * TURN_PAUSED and blocks until TOOL_RESULT (and optional ATTACH_IMAGE) come
 * back; then it builds the tool-role observation and commits it. */

static agent_tool_observation worker_run_tool_calls(agent_worker *w,
                                                    const proto_tool_calls *calls,
                                                    bool *interrupted_out) {
    *interrupted_out = false;
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf = proto_encode_tool_calls(&wr, calls, &out_len);
    (void)agent_send_all(w->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
    buf = proto_encode_empty(&wr, PROTO_S2C_TURN_PAUSED, &out_len);
    (void)agent_send_all(w->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);

    char *text = worker_request_tool_result(w);
    if (!text && worker_should_interrupt(w)) {
        *interrupted_out = true;
        agent_publish_system_status(w, "Stopped by user");
        worker_clear_interrupt(w);
        return (agent_tool_observation){0};
    }

    agent_tool_observation obs;
    agent_tool_observation_init(&obs);
    for (int i = 0; i < calls->len; i++) {
        char hdr[128];
        snprintf(hdr, sizeof(hdr), "Tool result %d (%s):\n", i + 1,
                 calls->v[i].name ? calls->v[i].name : "unknown");
        agent_tool_observation_puts(&obs, hdr);
    }
    if (text) {
        agent_tool_observation_puts(&obs, text);
        if (text[0] && text[strlen(text) - 1] != '\n')
            agent_tool_observation_puts(&obs, "\n");
    } else {
        agent_tool_observation_puts(&obs, "Tool error: no tool result received\n");
    }
    free(text);

    unsigned char *img = w->tool_result_image;
    size_t img_len = w->tool_result_image_len;
    w->tool_result_image = NULL;
    w->tool_result_image_len = 0;
    if (img && img_len) {
        ds4_vision_embedding emb = {0};
        char verr[256] = {0};
        if (ds4_engine_vision_encode_memory(w->engine, img, img_len, &emb,
                                            verr, sizeof(verr)) != 0) {
            uint32_t wd = emb.width, ht = emb.height;
            uint32_t tc = emb.token_count;
            agent_tool_observation_add_image(&obs, &emb);
            char meta[192];
            snprintf(meta, sizeof(meta),
                     "\nImage observation attached (%ux%u, %u visual tokens).\n",
                     wd, ht, tc);
            agent_tool_observation_puts(&obs, meta);
        } else {
            agent_tool_observation_puts(&obs, "Tool error: view_image failed: ");
            agent_tool_observation_puts(&obs, verr[0] ? verr : "unable to decode image");
            agent_tool_observation_puts(&obs, "\n");
        }
        free(img);
    }
    if (calls->len == 0)
        agent_tool_observation_puts(&obs, "Tool error: empty tool call block\n");
    return obs;
}

static bool agent_stream_compaction_needs_lookahead(const agent_stream_renderer *sr) {
    return !sr->dsml_active && sr->parser->state == AGENT_DSML_SEARCH &&
           (sr->pending_len || sr->dsml_start_len);
}

/* ---- Turn loop (ported from ds4_agent.c worker_run_turn; tool execution is
 * client-side, so parsed calls travel on TOOL_CALLS/TURN_PAUSED/TOOL_RESULT). */

static int worker_run_turn(agent_worker *w, const char *user_text) {
    dbg_log('T', "worker_run_turn start (user len %zu)",
            user_text ? strlen(user_text) : 0);
    agent_config *cfg = w->cfg;
    ds4_think_mode think_mode = effective_think_mode(cfg);
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    w->status.error[0] = '\0';
    pthread_mutex_unlock(&w->mu);

    char compact_err[160] = {0};
    if (!agent_worker_compact_if_needed(w, "soft limit before user turn",
                                        compact_err, sizeof(compact_err)))
    {
        if (agent_err_is_interrupted(compact_err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
        return 1;
    }
    agent_worker_maybe_append_datetime_context(w);
    if (!agent_worker_append_user(w, user_text, compact_err, sizeof(compact_err))) {
        if (agent_err_is_interrupted(compact_err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        agent_set_error(w, compact_err);
        return 1;
    }
    if (!w->session_title) {
        w->session_title = agent_session_title_from_prompt(user_text, 0);
        w->session_created_at = (uint64_t)time(NULL);
        agent_session_identity_sha(w->session_title, w->session_created_at,
                                   w->session_sha);
    }

    uint64_t rng = cfg->gen.seed ? cfg->gen.seed :
        ((uint64_t)time(NULL) ^ ((uint64_t)getpid() << 32) ^ (uint64_t)clock());
    pthread_mutex_lock(&w->mu);
    w->user_activity = true;
    w->session_dirty = true;
    pthread_mutex_unlock(&w->mu);

    int carried_generation = 0;
    bool resume_assistant = false, resume_in_think = false;
    int resume_start = 0;
    for (;;) {
        if (!resume_assistant &&
            !agent_worker_compact_if_needed(w, "soft limit before assistant continuation",
                                            compact_err, sizeof(compact_err)))
        {
            if (agent_err_is_interrupted(compact_err)) {
                worker_clear_interrupt(w);
                agent_set_status(w, AGENT_WORKER_IDLE);
                return 0;
            }
            agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
            return 1;
        }
        if (!resume_assistant)
            agent_worker_maybe_append_system_prompt_reminder(w);
        const int assistant_start = resume_assistant ? resume_start : w->transcript.len;
        if (!resume_assistant)
            ds4_chat_append_assistant_prefix(w->engine, &w->transcript, think_mode);

        const ds4_tokens *prompt_for_sync = &w->transcript;
        int old_pos = ds4_session_pos(w->session);
        int common = ds4_session_common_prefix(w->session, &w->transcript);
        int cached = common == old_pos && w->transcript.len >= old_pos ? common : 0;
        int suffix = prompt_for_sync->len - cached;

        pthread_mutex_lock(&w->mu);
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->progress_direct = false;
        w->progress_started_at = now_sec();
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        w->status.greedy_sampling = false;
        pthread_mutex_unlock(&w->mu);
        agent_push_status(w, true);

        char err[160];
        ds4_session_set_progress(w->session, worker_progress_cb, w);
        ds4_session_set_display_progress(w->session, worker_progress_cb, w);
        ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
        int sync_rc = w->image_count ?
            ds4_session_sync_multimodal(w->session, prompt_for_sync,
                                        w->images, w->image_count,
                                        err, sizeof(err)) :
            ds4_session_sync(w->session, prompt_for_sync, err, sizeof(err));
        ds4_session_set_cancel(w->session, NULL, NULL);
        ds4_session_set_progress(w->session, NULL, NULL);
        ds4_session_set_display_progress(w->session, NULL, NULL);
        if (sync_rc == DS4_SESSION_SYNC_INTERRUPTED) {
            agent_publish_system_status(
                w, "Model reading interrupted; the model may only be aware of the prefix processed so far.");
            agent_worker_append_assistant_turn_end(w);
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (sync_rc != 0) {
            agent_set_error(w, err);
            return 1;
        }

        int max_tokens = cfg->gen.n_predict - carried_generation;
        int room = ds4_session_ctx(w->session) - ds4_session_pos(w->session);
        int generation_room = room - agent_compact_reserve_tokens(w);
        bool context_limited = generation_room < max_tokens;
        if (context_limited) max_tokens = generation_room;
        if (max_tokens <= 0 && cfg->gen.n_predict > 0) {
            agent_set_error(w, "context has no generation room after compaction; use a larger --ctx");
            return 1;
        }

        bool in_think = resume_assistant ? resume_in_think : ds4_think_mode_enabled(think_mode);
        resume_assistant = false;
        agent_tool_syntax tool_syntax = agent_tool_syntax_for_engine(w->engine);
        agent_dsml_parser dsml = {
            .syntax = tool_syntax,
            .state = AGENT_DSML_SEARCH,
        };
        agent_stream_renderer stream = {
            .parser = &dsml,
            .syntax = tool_syntax,
            .in_think = in_think,
        };
        bool got_tool = false;
        bool malformed_tool = false;
        bool model_stopped = false;
        int generated = 0;
        int compaction_lookahead = 0;
        double t0 = now_sec();
        pthread_mutex_lock(&w->mu);
        w->status.state = AGENT_WORKER_GENERATING;
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        pthread_mutex_unlock(&w->mu);
        agent_push_status(w, true);

        bool status_greedy_sampling = false;
        while (generated < max_tokens && !worker_should_interrupt(w)) {
            worker_apply_pending_power(w);
            bool greedy_sampling = agent_stream_wants_greedy_sampling(&stream);
            if (greedy_sampling != status_greedy_sampling) {
                worker_set_greedy_sampling(w, greedy_sampling);
                status_greedy_sampling = greedy_sampling;
            }
            int token = worker_sample_with_mode(w, cfg, greedy_sampling, &rng);
            if (ds4_token_is_stop_for_think_mode(w->engine, token, think_mode)) {
                model_stopped = true;
                break;
            }

            int toks[17];
            int ntok = 0;
            const int block_start = ds4_session_pos(w->session);
            if (ds4_engine_mtp_draft_tokens(w->engine) > 1 &&
                getenv("DS4_MTP_SPEC_DISABLE") == NULL) {
                ntok = ds4_session_eval_speculative(
                    w->session, token, max_tokens - generated,
                    ds4_token_eos(w->engine),
                    greedy_sampling ? 0.0f : cfg->gen.temperature, 0,
                    greedy_sampling ? 1.0f : cfg->gen.top_p,
                    greedy_sampling ? 0.0f : cfg->gen.min_p,
                    &rng, toks, (int)(sizeof(toks) / sizeof(toks[0])),
                    err, sizeof(err));
                if (ntok < 0) {
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, err);
                    return 1;
                }
            } else {
                if (ds4_session_eval(w->session, token, err, sizeof(err)) != 0) {
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, err);
                    return 1;
                }
                toks[0] = token;
                ntok = 1;
            }
            if (ntok == 0) {
                agent_dsml_parser_free(&dsml);
                agent_set_error(w, "decode returned no tokens");
                return 1;
            }

            bool stop_block = false;
            bool restart_sampling = false;
            for (int ti = 0; ti < ntok && generated < max_tokens; ti++) {
                token = toks[ti];
                if (ds4_token_is_stop_for_think_mode(w->engine, token, think_mode)) {
                    model_stopped = true;
                    ds4_session_rewind(w->session, block_start + ti);
                    stop_block = true;
                    break;
                }
                if (worker_finish_generated_token(w, token, &generated, t0,
                                                  &stream, false,
                                                  err, sizeof(err)) != 0) {
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, err);
                    return 1;
                }
                const bool next_greedy =
                    agent_stream_wants_greedy_sampling(&stream);
                if (next_greedy != status_greedy_sampling) {
                    worker_set_greedy_sampling(w, next_greedy);
                    status_greedy_sampling = next_greedy;
                }
                if (dsml.state == AGENT_DSML_DONE) {
                    got_tool = true;
                    stop_block = true;
                } else if (dsml.state == AGENT_DSML_ERROR ||
                           stream.dsml_in_think) {
                    malformed_tool = true;
                    stop_block = true;
                }
                if (stop_block) {
                    if (ti + 1 < ntok) {
                        ds4_session_rewind(w->session, block_start + ti + 1);
                    }
                    break;
                }
                if (ti + 1 < ntok && ds4_engine_mtp_exact_sampling(w->engine) &&
                    cfg->gen.temperature > 0.0f && next_greedy != greedy_sampling) {
                    if (agent_worker_rewind(w, block_start + ti, err, sizeof(err)) != 0 ||
                        ds4_session_eval(w->session, token, err, sizeof(err)) != 0) {
                        agent_dsml_parser_free(&dsml);
                        agent_set_error(w, err);
                        return 1;
                    }
                    restart_sampling = true;
                    break;
                }
            }
            if (stop_block) break;
            if (restart_sampling) continue;
            if (context_limited && generated == max_tokens &&
                compaction_lookahead < 32 &&
                agent_stream_compaction_needs_lookahead(&stream)) {
                max_tokens++;
                compaction_lookahead++;
                if (max_tokens == cfg->gen.n_predict - carried_generation)
                    context_limited = false;
            }
        }

        bool interrupted = worker_should_interrupt(w);
        const bool partial_tool = stream.dsml_active || stream.dsml_start_len > 0 ||
                                  dsml.state != AGENT_DSML_SEARCH;
        agent_stream_text(&stream, NULL, 0, true);
        agent_flush_render_sink(w, &stream.sink);
        worker_set_greedy_sampling(w, false);
        if (!got_tool && dsml.state == AGENT_DSML_DONE && dsml.calls.len > 0)
            got_tool = true;
        if (interrupted) {
            agent_worker_append_assistant_turn_end(w);
            agent_dsml_parser_free(&dsml);
            agent_publish_system_status(w, "Stopped by user");
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }
        if (context_limited && generated == max_tokens && !model_stopped &&
            !got_tool && !malformed_tool) {
            carried_generation += generated;
            if (partial_tool) {
                w->transcript.len = assistant_start;
                ds4_session_invalidate(w->session);
            }
            agent_dsml_parser_free(&dsml);
            int open_assistant = assistant_start;
            if (!agent_worker_compact_transcript(w, "generation reached compaction reserve",
                                                 partial_tool ? NULL : &open_assistant,
                                                 compact_err, sizeof(compact_err))) {
                if (agent_err_is_interrupted(compact_err)) {
                    worker_clear_interrupt(w);
                    agent_set_status(w, AGENT_WORKER_IDLE);
                    return 0;
                }
                agent_set_error(w, compact_err);
                return 1;
            }
            if (partial_tool) {
                ds4_chat_append_message(w->engine, &w->transcript, "user",
                    "The last tool call was interrupted by context compaction and was NOT executed. "
                    "Continue the user's task, respecting its constraints. If a tool is needed, "
                    "retry with smaller arguments or smaller edits.\n");
            } else {
                resume_assistant = true;
                resume_start = open_assistant;
                resume_in_think = stream.in_think;
            }
            continue;
        }
        if (stream.dsml_in_think) {
            got_tool = false;
            malformed_tool = true;
            snprintf(dsml.error, sizeof(dsml.error),
                     "tool calling is not allowed inside <think></think>");
        } else if (!malformed_tool && dsml.state == AGENT_DSML_ERROR) {
            malformed_tool = true;
        } else if (!got_tool && !malformed_tool && !interrupted &&
                   (dsml.state == AGENT_DSML_STRUCTURAL ||
                    dsml.state == AGENT_DSML_PARAM_VALUE))
        {
            malformed_tool = true;
            snprintf(dsml.error, sizeof(dsml.error),
                     tool_syntax == AGENT_TOOL_SYNTAX_GLM ?
                     "incomplete GLM tool call" :
                     "incomplete DSML tool call");
        }

        agent_worker_append_assistant_turn_end(w);

        if (!got_tool && !malformed_tool) {
            agent_dsml_parser_free(&dsml);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return 0;
        }

        agent_tool_observation observation;
        agent_tool_observation_init(&observation);
        if (malformed_tool) {
            agent_tool_observation_puts(
                &observation, tool_syntax == AGENT_TOOL_SYNTAX_GLM ?
                "Tool error: invalid GLM tool call: " :
                "Tool error: invalid DSML tool call: ");
            agent_tool_observation_puts(
                &observation, dsml.error[0] ? dsml.error : "parse error");
            agent_tool_observation_puts(&observation, "\n");
            agent_tool_observation_puts(
                &observation, tool_syntax == AGENT_TOOL_SYNTAX_GLM ?
                agent_glm_syntax_reminder : agent_dsml_syntax_reminder);
        } else {
            agent_tool_observation_free(&observation);
            bool tool_interrupted = false;
            observation = worker_run_tool_calls(w, &dsml.calls, &tool_interrupted);
            if (tool_interrupted) {
                agent_dsml_parser_free(&dsml);
                agent_set_status(w, AGENT_WORKER_IDLE);
                return 0;
            }
        }
        int projected_tokens = 0;
        int result_reserve = agent_tool_result_reserve_tokens(w);
        char append_err[160] = {0};
        int fits = agent_tool_observation_fits(w, &observation, result_reserve,
                                               &projected_tokens, append_err, sizeof(append_err));
        if (fits < 0) goto observation_error;
        if (!fits) {
            if (!agent_worker_compact(w, "tool result would exceed context",
                                      compact_err, sizeof(compact_err)))
            {
                agent_tool_observation_free(&observation);
                agent_dsml_parser_free(&dsml);
                if (agent_err_is_interrupted(compact_err)) {
                    worker_clear_interrupt(w);
                    agent_set_status(w, AGENT_WORKER_IDLE);
                    return 0;
                }
                agent_set_error(w, compact_err[0] ? compact_err : "context compaction failed");
                return 1;
            }
            fits = agent_tool_observation_fits(w, &observation, result_reserve,
                                               &projected_tokens, append_err, sizeof(append_err));
            if (fits < 0) goto observation_error;
            if (!fits) {
                agent_tool_observation_free(&observation);
                agent_tool_observation_init(&observation);
                char msg[256];
                snprintf(msg, sizeof(msg),
                         "Tool error: tool result still does not fit after context compaction "
                         "(projected_prompt=%d tokens, ctx=%d, reserve=%d). "
                         "Retry with a smaller read/search/bash output.\n",
                         projected_tokens, agent_worker_effective_ctx_size(w),
                         result_reserve);
                agent_tool_observation_puts(&observation, msg);
                fits = agent_tool_observation_fits(w, &observation, 16, NULL,
                                                   append_err, sizeof(append_err));
                if (fits < 0) goto observation_error;
                if (!fits) {
                    agent_tool_observation_free(&observation);
                    agent_dsml_parser_free(&dsml);
                    agent_set_error(w, "context full after compaction");
                    return 1;
                }
            }
        }
        if (!agent_tool_observation_commit(w, &observation,
                                           append_err, sizeof(append_err)))
            goto observation_error;
        agent_tool_observation_free(&observation);
        agent_dsml_parser_free(&dsml);
        carried_generation = 0;

        char *queued_user = worker_request_queued_user_drain(w);
        if (queued_user && queued_user[0]) {
            if (!agent_worker_append_user(w, queued_user, compact_err, sizeof(compact_err))) {
                free(queued_user);
                if (agent_err_is_interrupted(compact_err)) {
                    worker_clear_interrupt(w);
                    agent_set_status(w, AGENT_WORKER_IDLE);
                    return 0;
                }
                agent_set_error(w, compact_err);
                return 1;
            }
            pthread_mutex_lock(&w->mu);
            w->user_activity = true;
            w->session_dirty = true;
            pthread_mutex_unlock(&w->mu);
        }
        free(queued_user);
        continue;

observation_error:
        agent_tool_observation_free(&observation);
        agent_dsml_parser_free(&dsml);
        agent_set_error(w, append_err[0] ? append_err : "unable to append tool observation");
        return 1;
    }
}

/* ---- Deferred session ops (worker thread) ---- */

static void worker_run_deferred_save(agent_worker *w) {
    if (!worker_take_save_requested(w)) return;
    agent_set_status(w, AGENT_WORKER_SAVING);
    char err[160] = {0};
    char sha[41];
    int tokens = 0;
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf;
    if (agent_worker_save_session_now(w, sha, &tokens, err, sizeof(err))) {
        proto_save_done_msg m = { .sha = sha, .tokens = (uint64_t)tokens };
        buf = proto_encode_save_done(&wr, &m, &out_len);
    } else {
        char msg[160];
        snprintf(msg, sizeof(msg), "%s", err[0] ? err : "save failed");
        buf = proto_encode_string(&wr, PROTO_S2C_ERROR, msg, &out_len);
    }
    (void)agent_send_all(w->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
    agent_set_status(w, AGENT_WORKER_IDLE);
}

static void worker_run_deferred_compact(agent_worker *w) {
    if (!worker_take_compact_requested(w)) return;
    if (!agent_worker_has_user_session(w)) {
        proto_writer wr;
        size_t out_len = 0;
        unsigned char *buf = proto_encode_string(&wr, PROTO_S2C_ERROR,
            "compact skipped: nothing to compact", &out_len);
        (void)agent_send_all(w->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
        return;
    }
    int before = w->transcript.len;
    char err[160] = {0};
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf;
    if (agent_worker_compact(w, "user requested compaction", err, sizeof(err))) {
        if (w->transcript.len != before) {
            pthread_mutex_lock(&w->mu);
            w->session_dirty = true;
            pthread_mutex_unlock(&w->mu);
        }
        buf = proto_encode_empty(&wr, PROTO_S2C_COMPACT_DONE, &out_len);
        (void)agent_send_all(w->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
        agent_set_status(w, AGENT_WORKER_IDLE);
    } else {
        if (agent_err_is_interrupted(err)) {
            worker_clear_interrupt(w);
            agent_set_status(w, AGENT_WORKER_IDLE);
            return;
        }
        char msg[160];
        snprintf(msg, sizeof(msg), "%s", err[0] ? err : "context compaction failed");
        buf = proto_encode_string(&wr, PROTO_S2C_ERROR, msg, &out_len);
        (void)agent_send_all(w->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
    }
}

static void worker_append_system_message(agent_worker *w, const char *text) {
    ds4_chat_append_message(w->engine, &w->transcript, "system", text);
    char err[160] = {0};
    if (agent_worker_sync_tokens(w, &w->transcript, true, err, sizeof(err)) != 0) {
        if (agent_err_is_interrupted(err)) {
            worker_clear_interrupt(w);
        }
    }
}

/* ---- Worker thread entry ---- */

static void *worker_main(void *arg) {
    agent_worker *w = arg;
    char init_err[160] = {0};
    if (!agent_worker_reset_to_sysprompt(w, init_err, sizeof(init_err))) {
        agent_set_error(w, init_err[0] ? init_err : "failed to initialize system prompt");
    }
    pthread_mutex_lock(&w->mu);
    w->initialized = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);

    while (true) {
        pthread_mutex_lock(&w->mu);
        while (!w->stop && !w->cmd_text && !w->save_requested &&
               !w->compact_requested && !w->power_requested)
            pthread_cond_wait(&w->cond, &w->mu);
        if (w->stop) {
            pthread_mutex_unlock(&w->mu);
            break;
        }
        if (w->power_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_apply_pending_power(w);
            continue;
        }
        if (!w->cmd_text && w->save_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_save(w);
            continue;
        }
        if (!w->cmd_text && w->compact_requested) {
            pthread_mutex_unlock(&w->mu);
            worker_run_deferred_compact(w);
            continue;
        }
        char *cmd = w->cmd_text;
        bool cmd_system = w->cmd_system;
        w->cmd_text = NULL;
        w->cmd_system = false;
        pthread_mutex_unlock(&w->mu);
        dbg_log('T', "worker_main got cmd (len %zu)",
                cmd ? strlen(cmd) : 0);

        if (cmd_system) {
            worker_append_system_message(w, cmd ? cmd : "");
        } else {
            worker_run_turn(w, cmd ? cmd : "");
        }
        free(cmd);
        worker_apply_pending_power(w);
        worker_run_deferred_compact(w);
        worker_run_deferred_save(w);
    }

    agent_set_status(w, AGENT_WORKER_STOPPED);
    return NULL;
}

/* ---- Worker init ---- */

static int agent_worker_init(agent_worker *w, ds4_engine *engine, agent_config *cfg) {
    memset(w, 0, sizeof(*w));
    w->engine = engine;
    w->cfg = cfg;
    w->sock_fd = -1;
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cond, NULL);
    w->status.state = AGENT_WORKER_IDLE;
    if (ds4_session_create(&w->session, engine, cfg->gen.ctx_size) != 0) {
        fprintf(stderr, "ds4-agent-server: session backend is required\n");
        return -1;
    }
    w->cache_dir = agent_default_cache_dir();
    if (!agent_mkdir_p(w->cache_dir)) {
        fprintf(stderr, "ds4-agent-server: failed to create %s: %s\n",
                w->cache_dir, strerror(errno));
        return -1;
    }
    w->sysprompt_path = ds4_kvstore_path_join(w->cache_dir, "sysprompt.kv");
    if (pthread_create(&w->thread, NULL, worker_main, w) != 0) return -1;
    return 0;
}

/* ---- Socket helpers ---- */

static bool server_read_full(int fd, void *buf, size_t n) {
    unsigned char *p = buf;
    size_t got = 0;
    while (got < n) {
        ssize_t r = read(fd, p + got, n - got);
        if (r < 0 && errno == EINTR) continue;
        if (r <= 0) return false;
        got += (size_t)r;
    }
    return true;
}

static bool server_read_frame(int fd, unsigned char **buf_out, size_t *len_out) {
    unsigned char h[4];
    if (!server_read_full(fd, h, 4)) return false;
    uint32_t plen = ((uint32_t)h[0] << 24) | ((uint32_t)h[1] << 16) |
                    ((uint32_t)h[2] << 8) | (uint32_t)h[3];
    if (plen > (1u << 26)) return false; /* 64 MiB sanity bound */
    unsigned char *buf = xmalloc((size_t)plen + 4);
    memcpy(buf, h, 4);
    if (!server_read_full(fd, buf + 4, plen)) {
        free(buf);
        return false;
    }
    *buf_out = buf;
    *len_out = (size_t)plen + 4;
    return true;
}

static void server_send_error(agent_worker *w, const char *msg) {
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf = proto_encode_string(&wr, PROTO_S2C_ERROR, msg, &out_len);
    (void)agent_send_all(w->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
}

static void server_send_hello(agent_worker *w) {
    proto_hello_msg m = {0};
    m.session_title = w->session_title ? w->session_title : "";
    m.session_created_at = w->session_created_at;
    m.status.state = (uint64_t)w->status.state;
    m.status.ctx_used = (uint64_t)w->status.ctx_used;
    m.status.ctx_size = (uint64_t)agent_worker_effective_ctx_size(w);
    m.status.prefill_done = (uint64_t)w->status.prefill_done;
    m.status.prefill_total = (uint64_t)w->status.prefill_total;
    m.status.prefill_tps = (float)w->status.prefill_tps;
    m.status.generated = (uint64_t)w->status.generated;
    m.status.gen_tps = (float)w->status.gen_tps;
    m.status.greedy = w->status.greedy_sampling;
    m.status.power = (uint64_t)w->status.power_percent;
    m.status.error = w->status.error;
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *buf = proto_encode_hello(&wr, &m, &out_len);
    (void)agent_send_all(w->sock_fd, buf, out_len);
    proto_writer_free(&wr);
    free(buf);
}

/* ---- NEW_SESSION ---- */

static void server_new_session(agent_worker *w, const proto_new_session_msg *m,
                               char *err, size_t err_len) {
    w->cfg->gen.n_predict = (int)m->n_predict;
    w->cfg->gen.temperature = m->temperature;
    w->cfg->gen.top_p = m->top_p;
    w->cfg->gen.min_p = m->min_p;
    w->cfg->gen.temperature_set = true;
    w->cfg->gen.top_p_set = true;
    w->cfg->gen.min_p_set = true;
    w->cfg->gen.think_mode = (ds4_think_mode)m->think_mode;
    w->cfg->gen.seed = m->seed;
    w->cfg->engine.power_percent = (int)m->power;
    if (m->sys_extra && m->sys_extra[0])
        w->cfg->gen.system = m->sys_extra;
    if (!agent_worker_reset_to_sysprompt(w, err, err_len))
        return;
    server_send_hello(w);
}

/* ---- Dispatch loop ---- */

static void server_dispatch(agent_worker *w, unsigned char *frame, size_t len) {
    unsigned char tag = 0;
    proto_reader payload;
    if (!proto_open_frame(frame, len, &payload, &tag)) {
        server_send_error(w, "malformed frame");
        return;
    }
    char err[160] = {0};
    switch (tag) {
    case PROTO_C2S_NEW_SESSION: {
        proto_new_session_msg m = {0};
        if (!proto_decode_new_session(frame, len, &tag, &m)) {
            server_send_error(w, "malformed NEW_SESSION");
            break;
        }
        server_new_session(w, &m, err, sizeof(err));
        if (err[0]) server_send_error(w, err);
        free(m.sys_extra);
        break;
    }
    case PROTO_C2S_USER: {
        dbg_log('C', "client -> USER (len %zu)", len);
        char *text = NULL;
        if (!proto_decode_string(frame, len, tag, &tag, &text)) {
            server_send_error(w, "malformed USER");
            break;
        }
        pthread_mutex_lock(&w->mu);
        bool paused = w->tool_result_pending;
        if (paused) w->tool_result_pending = false;
        pthread_mutex_unlock(&w->mu);
        if (paused) {
            /* paused turn: queue the user message for the next tool round */
            worker_answer_queued_user_drain(w, text);
            break;
        }
        pthread_mutex_lock(&w->mu);
        bool idle = w->initialized && w->status.state == AGENT_WORKER_IDLE && !w->cmd_text;
        dbg_log('T', "USER dispatch idle=%d init=%d state=%d",
                (int)idle, (int)w->initialized, (int)w->status.state);
        if (idle) {
            free(w->cmd_text);
            w->cmd_text = text;
            w->cmd_system = false;
            pthread_cond_signal(&w->cond);
            pthread_mutex_unlock(&w->mu);
            dbg_log('T', "queued USER to worker thread");
        } else {
            pthread_mutex_unlock(&w->mu);
            server_send_error(w, "model is busy; wait for the turn to finish");
            dbg_log('T', "USER dropped (model busy)");
            free(text);
        }
        break;
    }
    case PROTO_C2S_TOOL_RESULT: {
        char *text = NULL;
        if (!proto_decode_string(frame, len, tag, &tag, &text)) {
            server_send_error(w, "malformed TOOL_RESULT");
            break;
        }
        pthread_mutex_lock(&w->mu);
        bool waiting = w->tool_result_pending;
        pthread_mutex_unlock(&w->mu);
        if (waiting) {
            worker_answer_tool_result(w, text);
        } else {
            server_send_error(w, "no tool result pending");
            free(text);
        }
        break;
    }
    case PROTO_C2S_ATTACH_IMAGE: {
        proto_attach_image_msg m = {0};
        if (!proto_decode_attach_image(frame, len, &tag, &m)) {
            server_send_error(w, "malformed ATTACH_IMAGE");
            break;
        }
        pthread_mutex_lock(&w->mu);
        free(w->tool_result_image);
        w->tool_result_image = m.data;
        w->tool_result_image_len = m.data_len;
        pthread_mutex_unlock(&w->mu);
        break;
    }
    case PROTO_C2S_SYSTEM: {
        char *text = NULL;
        if (!proto_decode_string(frame, len, tag, &tag, &text)) {
            server_send_error(w, "malformed SYSTEM");
            break;
        }
        pthread_mutex_lock(&w->mu);
        free(w->cmd_text);
        w->cmd_text = text;
        w->cmd_system = true;
        pthread_cond_signal(&w->cond);
        pthread_mutex_unlock(&w->mu);
        break;
    }
    case PROTO_C2S_STOP_TURN:
    case PROTO_C2S_INTERRUPT:
        pthread_mutex_lock(&w->mu);
        w->interrupt = true;
        pthread_cond_signal(&w->cond);
        pthread_mutex_unlock(&w->mu);
        break;
    case PROTO_C2S_COMPACT:
        worker_request_compact(w);
        break;
    case PROTO_C2S_SAVE:
        worker_request_save(w);
        break;
    case PROTO_C2S_LIST: {
        proto_list_msg out = {0};
        if (!agent_worker_build_list(w, &out, err, sizeof(err))) {
            server_send_error(w, err[0] ? err : "list failed");
            break;
        }
        proto_writer wr;
        size_t out_len = 0;
        unsigned char *buf = proto_encode_list(&wr, &out, &out_len);
        (void)agent_send_all(w->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
        proto_list_msg_free(&out);
        break;
    }
    case PROTO_C2S_SWITCH: {
        char *prefix = NULL;
        if (!proto_decode_string(frame, len, tag, &tag, &prefix)) {
            server_send_error(w, "malformed SWITCH");
            break;
        }
        if (!agent_worker_switch_session(w, prefix, 3, err, sizeof(err))) {
            server_send_error(w, err[0] ? err : "switch failed");
        } else {
            proto_switch_done_msg m = { .sha = w->session_sha };
            proto_writer wr;
            size_t out_len = 0;
            unsigned char *buf = proto_encode_switch_done(&wr, &m, &out_len);
            (void)agent_send_all(w->sock_fd, buf, out_len);
            proto_writer_free(&wr);
            free(buf);
        }
        free(prefix);
        break;
    }
    case PROTO_C2S_DEL: {
        char *prefix = NULL;
        if (!proto_decode_string(frame, len, tag, &tag, &prefix)) {
            server_send_error(w, "malformed DEL");
            break;
        }
        char sha[41] = {0};
        if (!agent_worker_delete_session(w, prefix, sha, err, sizeof(err))) {
            server_send_error(w, err[0] ? err : "delete failed");
        } else {
            char msg[160];
            snprintf(msg, sizeof(msg), "deleted session %.8s", sha);
            proto_writer wr;
            size_t out_len = 0;
            unsigned char *buf = proto_encode_string(&wr, PROTO_S2C_HISTORY, msg, &out_len);
            (void)agent_send_all(w->sock_fd, buf, out_len);
            proto_writer_free(&wr);
            free(buf);
        }
        free(prefix);
        break;
    }
    case PROTO_C2S_STRIP: {
        char *prefix = NULL;
        if (!proto_decode_string(frame, len, tag, &tag, &prefix)) {
            server_send_error(w, "malformed STRIP");
            break;
        }
        char sha[41] = {0};
        uint32_t tokens = 0;
        if (!agent_worker_strip_session(w, prefix, sha, &tokens, err, sizeof(err))) {
            server_send_error(w, err[0] ? err : "strip failed");
        } else {
            char msg[160];
            snprintf(msg, sizeof(msg), "stripped session %.8s (%u tokens)", sha, tokens);
            proto_writer wr;
            size_t out_len = 0;
            unsigned char *buf = proto_encode_string(&wr, PROTO_S2C_HISTORY, msg, &out_len);
            (void)agent_send_all(w->sock_fd, buf, out_len);
            proto_writer_free(&wr);
            free(buf);
        }
        free(prefix);
        break;
    }
    case PROTO_C2S_HISTORY: {
        char *prefix = NULL;
        if (!proto_decode_string(frame, len, tag, &tag, &prefix)) {
            server_send_error(w, "malformed HISTORY");
            break;
        }
        int turns = prefix && prefix[0] ?
            (int)strtol(prefix, NULL, 10) : 3;
        if (turns <= 0) turns = 3;
        if (turns > 200) turns = 200;
        char *text = agent_worker_history_text(w, turns, err, sizeof(err));
        if (!text) {
            server_send_error(w, err[0] ? err : "history failed");
        } else {
            proto_writer wr;
            size_t out_len = 0;
            unsigned char *buf = proto_encode_string(&wr, PROTO_S2C_HISTORY, text, &out_len);
            (void)agent_send_all(w->sock_fd, buf, out_len);
            proto_writer_free(&wr);
            free(buf);
            free(text);
        }
        free(prefix);
        break;
    }
    case PROTO_C2S_TOKENS: {
        char *text = NULL;
        if (!proto_decode_string(frame, len, tag, &tag, &text)) {
            server_send_error(w, "malformed TOKENS");
            break;
        }
        ds4_tokens toks = {0};
        ds4_tokenize_text(w->engine, text ? text : "", &toks);
        proto_writer wr;
        size_t out_len = 0;
        unsigned char *buf = proto_encode_varint(&wr, PROTO_S2C_COUNT,
                                                 (uint64_t)toks.len, &out_len);
        (void)agent_send_all(w->sock_fd, buf, out_len);
        proto_writer_free(&wr);
        free(buf);
        ds4_tokens_free(&toks);
        free(text);
        break;
    }
    case PROTO_C2S_POWER: {
        uint64_t v = 0;
        if (!proto_decode_varint(frame, len, tag, &tag, &v)) {
            server_send_error(w, "malformed POWER");
            break;
        }
        worker_request_power(w, (int)v);
        break;
    }
    default:
        server_send_error(w, "unknown message tag");
        break;
    }
}

/* ---- Server main ---- */

static int server_accept(const agent_config *cfg, agent_worker *w) {
    int listen_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd < 0) return -1;
    int one = 1;
    setsockopt(listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)cfg->port);
    if (inet_pton(AF_INET, cfg->host, &addr.sin_addr) != 1 ||
        bind(listen_fd, (struct sockaddr *)&addr, sizeof(addr)) != 0 ||
        listen(listen_fd, 1) != 0) {
        fprintf(stderr, "ds4-agent-server: cannot listen on %s:%d: %s\n",
                cfg->host, cfg->port, strerror(errno));
        close(listen_fd);
        return -1;
    }
    fprintf(stderr, "ds4-agent-server: listening on %s:%d\n", cfg->host, cfg->port);
    int sock = accept(listen_fd, NULL, NULL);
    close(listen_fd);
    if (sock < 0) return -1;
    w->sock_fd = sock;
    dbg_log('T', "client connected");
    /* Wait for the worker thread's startup system-prompt prefill to finish
     * before dispatching NEW_SESSION on the main thread.  Otherwise two
     * threads drive GPU prefill concurrently on the same engine/session,
     * which corrupts the ROCm command stream (memory aperture violation
     * -> abort): the worker's startup reset races the NEW_SESSION reset. */
    pthread_mutex_lock(&w->mu);
    while (!w->initialized)
        pthread_cond_wait(&w->cond, &w->mu);
    pthread_mutex_unlock(&w->mu);
    return 0;
}

static int server_run(agent_worker *w) {
    for (;;) {
        unsigned char *frame = NULL;
        size_t len = 0;
        if (!server_read_frame(w->sock_fd, &frame, &len)) {
            dbg_log('T', "client disconnected");
            break;
        }
        dbg_log('C', "client -> %s (len %zu)",
                len > 4 ? dbg_tag_name(frame[4]) : "?", len);
        server_dispatch(w, frame, len);
        free(frame);
    }
    pthread_mutex_lock(&w->mu);
    w->stop = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
    return 0;
}

int main(int argc, char **argv) {
    dbg_init();
    agent_config cfg = parse_options(argc, argv);
    cfg.engine.context_size = cfg.gen.ctx_size;
    cfg.engine.placement_ctx_hint = cfg.gen.ctx_size;
    if (cfg.gpu_vram_arg || cfg.gpu_devices_arg) {
        cfg.engine.backend = cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "0")
            ? DS4_BACKEND_CPU
            : DS4_BACKEND_CUDA;
    }
    ds4_engine *engine = NULL;
    if (cfg.gpu_vram_arg || cfg.gpu_devices_arg) {
        ds4_gpu_config gpu_cfg = {0};
        bool skip_cuda = false;
        char gpu_err[256];
        if (parse_gpu_vram_arg(cfg.gpu_vram_arg, cfg.gpu_devices_arg,
                               &gpu_cfg, &skip_cuda,
                               gpu_err, sizeof(gpu_err)) != 0) {
            fprintf(stderr, "ds4-agent-server: %s\n", gpu_err);
            return 2;
        }
        if (skip_cuda) {
            cfg.engine.backend = DS4_BACKEND_CPU;
            if (ds4_engine_open(&engine, &cfg.engine) != 0) return 1;
        } else {
            const bool was_auto =
                (cfg.gpu_vram_arg && !strcmp(cfg.gpu_vram_arg, "auto")) ||
                (!cfg.gpu_vram_arg && cfg.gpu_devices_arg);
            char layout[256];
            if (format_gpu_layout_line(&gpu_cfg, was_auto, layout, sizeof(layout)) > 0) {
                fprintf(stdout, "%s\n", layout);
                fflush(stdout);
            }
            cfg.engine.backend = DS4_BACKEND_CUDA;
            if (ds4_engine_create_with_gpu_config(&engine, &cfg.engine, &gpu_cfg) != 0) return 1;
        }
    } else if (ds4_engine_open(&engine, &cfg.engine) != 0) {
        return 1;
    }
    ds4_tp *tp_leader = NULL;
    if (cfg.engine.tp.role == DS4_TP_LEADER) {
        char tp_err[256] = "";
        ds4_tp_identity tp_id = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = (uint32_t)cfg.gen.ctx_size,
        };
        ds4_engine_tp_gate_schedule(engine,
                                    &tp_id.gate_slot_start,
                                    &tp_id.gate_slot_step,
                                    &tp_id.gates_per_token,
                                    tp_id.gate_slot_mask);
        if (!ds4_tp_create(&tp_leader, &cfg.engine.tp, &tp_id,
                           tp_err, sizeof(tp_err)) ||
            !ds4_engine_tp_bind(engine, tp_leader, tp_err, sizeof(tp_err))) {
            fprintf(stderr, "ds4-agent-server: %s\n", tp_err);
            ds4_tp_free(tp_leader);
            ds4_engine_close(engine);
            return 1;
        }
    }
    agent_apply_model_sampling_defaults(engine, &cfg.gen);

    agent_worker worker;
    if (agent_worker_init(&worker, engine, &cfg) != 0) {
        if (tp_leader) ds4_tp_send_stop(tp_leader);
        ds4_engine_close(engine);
        ds4_tp_free(tp_leader);
        return 1;
    }
    if (server_accept(&cfg, &worker) != 0) {
        pthread_mutex_lock(&worker.mu);
        worker.stop = true;
        pthread_cond_signal(&worker.cond);
        pthread_mutex_unlock(&worker.mu);
        pthread_join(worker.thread, NULL);
        if (tp_leader) ds4_tp_send_stop(tp_leader);
        ds4_engine_close(engine);
        ds4_tp_free(tp_leader);
        return 1;
    }
    int rc = server_run(&worker);
    pthread_join(worker.thread, NULL);
    close(worker.sock_fd);
    if (tp_leader) ds4_tp_send_stop(tp_leader);
    ds4_engine_close(engine);
    ds4_tp_free(tp_leader);
    return rc;
}

#endif /* !DS4_AGENT_TEST */

/* ============================================================================
 * Tests (ported from ds4_agent.c, DS4_AGENT_TEST only)
 * ========================================================================== */

#ifdef DS4_AGENT_TEST

static int agent_test_failures;

static void agent_test_assert(bool cond, const char *expr,
                              const char *file, int line) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    agent_test_failures++;
}

#define AGENT_TEST_ASSERT(expr) \
    agent_test_assert((expr), #expr, __FILE__, __LINE__)

static char *agent_test_stream_capture(agent_tool_syntax syntax,
                                       const char **chunks,
                                       size_t chunk_count,
                                       agent_dsml_parser *p,
                                       bool *dsml_in_think) {
    *p = (agent_dsml_parser){
        .syntax = syntax,
        .state = AGENT_DSML_SEARCH,
    };
    agent_stream_renderer stream = {
        .parser = p,
        .syntax = syntax,
    };

    for (size_t i = 0; i < chunk_count; i++)
        agent_stream_text(&stream, chunks[i], strlen(chunks[i]), false);
    agent_stream_text(&stream, NULL, 0, true);
    if (dsml_in_think) *dsml_in_think = stream.dsml_in_think;
    char *out = agent_render_sink_concat(&stream.sink);
    agent_render_sink_free(&stream.sink);
    return out;
}

static void test_agent_glm_tool_parser_single_arg(void) {
    const char *text =
        "prose before <tool_call>list"
        "<arg_key>path</arg_key><arg_value>.</arg_value>"
        "</tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, text, strlen(text));
    agent_dsml_finish(&p);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "list"));
    AGENT_TEST_ASSERT(p.calls.v[0].argc == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].args[0].name &&
                      !strcmp(p.calls.v[0].args[0].name, "path"));
    AGENT_TEST_ASSERT(p.calls.v[0].args[0].value &&
                      !strcmp(p.calls.v[0].args[0].value, "."));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_chunked_multi_arg(void) {
    const char *a = "<tool_call>bash<arg_key>command</arg_key>";
    const char *b = "<arg_value>printf hi</arg_value>";
    const char *c =
        "<arg_key>refresh_sec</arg_key><arg_value>1</arg_value></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, a, strlen(a));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_STRUCTURAL);
    agent_dsml_feed(&p, b, strlen(b));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_STRUCTURAL);
    agent_dsml_feed(&p, c, strlen(c));
    agent_dsml_finish(&p);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "bash"));
    AGENT_TEST_ASSERT(p.calls.v[0].argc == 2);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"),
                              "printf hi"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "refresh_sec"),
                              "1"));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_streams_param_state(void) {
    const char *a =
        "<tool_call>bash<arg_key>command</arg_key><arg_value>printf hi";
    const char *b = "</arg";
    const char *c =
        "_value><arg_key>refresh_sec</arg_key><arg_value>1</arg_value></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, a, strlen(a));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(p.current.name && !strcmp(p.current.name, "bash"));
    AGENT_TEST_ASSERT(p.param_name && !strcmp(p.param_name, "command"));
    AGENT_TEST_ASSERT(!p.param_close_prefix);

    agent_dsml_feed(&p, b, strlen(b));
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(p.param_close_prefix);

    agent_dsml_feed(&p, c, strlen(c));
    agent_dsml_finish(&p);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"),
                              "printf hi"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "refresh_sec"),
                              "1"));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_multiple_adjacent_calls(void) {
    const char *text =
        "<tool_call>list<arg_key>path</arg_key><arg_value>.</arg_value></tool_call>\n"
        "<tool_call>bash<arg_key>command</arg_key><arg_value>pwd</arg_value></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, text, strlen(text));
    agent_dsml_finish(&p);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 2);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "list"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"), "."));
    AGENT_TEST_ASSERT(p.calls.v[1].name && !strcmp(p.calls.v[1].name, "bash"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[1], "command"), "pwd"));

    agent_dsml_parser_free(&p);
}

static void test_agent_glm_stream_tool_call_chunked(void) {
    const char *chunks[] = {
        "intro ",
        "<to",
        "ol_call>bash<arg_key>command</arg_key><arg_value>printf hi</arg",
        "_value><arg_key>refresh_sec</arg_key><arg_value>1</arg_value></tool_call>",
    };
    agent_dsml_parser p;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_GLM,
                                          chunks,
                                          sizeof(chunks)/sizeof(chunks[0]),
                                          &p,
                                          NULL);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "bash"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "command"),
                              "printf hi"));
    AGENT_TEST_ASSERT(strstr(out, "intro ") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "printf hi") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "<tool_call>") == NULL);
    AGENT_TEST_ASSERT(strstr(out, "<arg_key>") == NULL);
    AGENT_TEST_ASSERT(strstr(out, "</arg_value>") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_tool_argument_literal_markup(void) {
    const char *glm[] = {
        "<tool_call>write<arg_key>content</arg_key><arg_value><p>&amp; &lt;</p> </tool_call> </think> ",
        "&lt;/arg_value> &amp;lt;/arg_value></arg_value></tool_call>",
    };
    const char *deepseek[] = {
        "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls><" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke name=\"write\"><" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter name=\"content\" string=\"true\"><p>&amp; &lt;</p> </tool_call> </think> ",
        "&lt;/" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter> &amp;lt;/" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter></" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter></" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke></" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>",
    };
    const char *expected[] = {
        "<p>&amp; &lt;</p> </tool_call> </think> </" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter> &lt;/" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter>",
        "<p>&amp; &lt;</p> </tool_call> </think> </arg_value> &lt;/arg_value>",
    };
    for (int is_glm = 0; is_glm <= 1; is_glm++) {
        agent_dsml_parser p;
        char *out = agent_test_stream_capture(
            is_glm ? AGENT_TOOL_SYNTAX_GLM : AGENT_TOOL_SYNTAX_DSML,
            is_glm ? glm : deepseek, 2, &p, NULL);
        AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
        AGENT_TEST_ASSERT(p.calls.len == 1);
        if (p.calls.len)
            AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "content"), expected[is_glm]));
        free(out);
        agent_dsml_parser_free(&p);
    }
}

static void test_agent_glm_stream_ignores_tool_inside_think(void) {
    const char *chunks[] = {
        "<think>plan <tool",
        "_call>bash<arg_key>command</arg_key><arg_value>printf hi</arg_value></tool_call>",
        "</think>done",
    };
    agent_dsml_parser p;
    bool dsml_in_think = false;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_GLM,
                                          chunks,
                                          sizeof(chunks)/sizeof(chunks[0]),
                                          &p,
                                          &dsml_in_think);

    AGENT_TEST_ASSERT(dsml_in_think);
    AGENT_TEST_ASSERT(p.calls.len == 0);
    AGENT_TEST_ASSERT(strstr(out, "[tool call ignored:") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "<tool_call>") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_glm_stream_greedy_sampling_boundaries(void) {
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };
    agent_stream_renderer stream = {
        .parser = &p,
        .syntax = AGENT_TOOL_SYNTAX_GLM,
    };

    agent_stream_text(&stream, "<tool", strlen("<tool"), false);
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));

    const char *body =
        "_call>bash<arg_key>command</arg_key><arg_value>printf hi";
    agent_stream_text(&stream, body, strlen(body), false);
    AGENT_TEST_ASSERT(p.state == AGENT_DSML_PARAM_VALUE);
    AGENT_TEST_ASSERT(!agent_stream_wants_greedy_sampling(&stream));

    agent_stream_text(&stream, "</arg", strlen("</arg"), false);
    AGENT_TEST_ASSERT(p.param_close_prefix);
    AGENT_TEST_ASSERT(agent_stream_wants_greedy_sampling(&stream));

    const char *close = "_value></tool_call>";
    agent_stream_text(&stream, close, strlen(close), false);
    agent_stream_text(&stream, NULL, 0, true);
    char *out = agent_render_sink_concat(&stream.sink);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(strstr(out, "</arg_value>") == NULL);

    free(out);
    agent_render_sink_free(&stream.sink);
    agent_dsml_parser_free(&p);
}

static void test_agent_dsml_stream_tool_call_chunked(void) {
    const char *chunks[] = {
        "<" AGENT_DSML_BAR "D",
        "SML" AGENT_DSML_BAR "tool_calls><" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke name=\"read\">"
        "<" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter name=\"path\" string=\"true\">ds4_agent.c</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "parameter>"
        "</" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "invoke></" AGENT_DSML_BAR "DSML" AGENT_DSML_BAR "tool_calls>",
    };
    agent_dsml_parser p;
    char *out = agent_test_stream_capture(AGENT_TOOL_SYNTAX_DSML,
                                          chunks,
                                          sizeof(chunks)/sizeof(chunks[0]),
                                          &p,
                                          NULL);

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_DONE);
    AGENT_TEST_ASSERT(p.calls.len == 1);
    AGENT_TEST_ASSERT(p.calls.v[0].name && !strcmp(p.calls.v[0].name, "read"));
    AGENT_TEST_ASSERT(!strcmp(agent_tool_arg_value(&p.calls.v[0], "path"),
                              "ds4_agent.c"));
    AGENT_TEST_ASSERT(strstr(out, "Reading ") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "ds4_agent.c") != NULL);
    AGENT_TEST_ASSERT(strstr(out, "DSML") == NULL);

    free(out);
    agent_dsml_parser_free(&p);
}

static void test_agent_glm_tool_parser_rejects_missing_value(void) {
    const char *text = "<tool_call>list<arg_key>path</arg_key></tool_call>";
    agent_dsml_parser p = {
        .syntax = AGENT_TOOL_SYNTAX_GLM,
        .state = AGENT_DSML_SEARCH,
    };

    agent_dsml_feed(&p, text, strlen(text));

    AGENT_TEST_ASSERT(p.state == AGENT_DSML_ERROR);
    AGENT_TEST_ASSERT(strstr(p.error, "expected <arg_value>") != NULL);

    agent_dsml_parser_free(&p);
}

/* ---- Layer 3: server control-logic helpers (pure / file / lock) ---- */

static void test_agent_session_identity_sha(void) {
    char a[41], b[41], c[41];
    agent_session_identity_sha("hello", 12345, a);
    agent_session_identity_sha("hello", 12345, b);
    agent_session_identity_sha("hello", 54321, c);
    AGENT_TEST_ASSERT(!strcmp(a, b));          /* deterministic */
    AGENT_TEST_ASSERT(strcmp(a, c));           /* created_at matters */

    /* independent: sha1(title ++ le64(created_at)) */
    const char *title = "hello";
    uint8_t ts[8];
    agent_le_put64(ts, 12345);
    agent_buf buf = {0};
    agent_buf_append(&buf, title, strlen(title));
    agent_buf_append(&buf, (const char *)ts, sizeof(ts));
    char expect[41];
    ds4_kvstore_sha1_bytes_hex(buf.ptr ? buf.ptr : "", buf.len, expect);
    AGENT_TEST_ASSERT(!strcmp(a, expect));
    free(buf.ptr);

    agent_session_identity_sha("", 0, a);      /* empty title is valid */
    AGENT_TEST_ASSERT(a[0] != '\0');
}

static void test_agent_session_title_from_text(void) {
#define T_USER "<" AGENT_DSML_BAR "User" AGENT_DSML_BAR ">"
#define T_ASSISTANT "<" AGENT_DSML_BAR "Assistant" AGENT_DSML_BAR ">"
    char *t = agent_session_title_from_text(
        "prefix " T_USER " what is the capital of france? " T_ASSISTANT " paris",
        strlen("prefix " T_USER " what is the capital of france? " T_ASSISTANT " paris"), 0);
    AGENT_TEST_ASSERT(t && !strcmp(t, "what is the capital of france?"));
    free(t);

    /* no user marker -> "(no user prompt)" */
    t = agent_session_title_from_text("just prose", strlen("just prose"), 0);
    AGENT_TEST_ASSERT(t && !strcmp(t, "(no user prompt)"));
    free(t);

    /* user marker with no assistant -> span ends at next user marker */
    t = agent_session_title_from_text("a " T_USER " hello " T_USER " world",
                                      strlen("a " T_USER " hello " T_USER " world"), 0);
    AGENT_TEST_ASSERT(t && !strcmp(t, "hello"));
    free(t);

    /* empty user prompt -> "(empty user prompt)" */
    t = agent_session_title_from_text(T_USER T_ASSISTANT,
                                      strlen(T_USER T_ASSISTANT), 0);
    AGENT_TEST_ASSERT(t && !strcmp(t, "(empty user prompt)"));
    free(t);
#undef T_USER
#undef T_ASSISTANT
}

static void test_agent_session_title_clip(void) {
    char *t = agent_session_title_clip("short", 100);
    AGENT_TEST_ASSERT(t && !strcmp(t, "short"));
    free(t);

    t = agent_session_title_clip("0123456789", 7);
    AGENT_TEST_ASSERT(t && !strcmp(t, "0123..."));
    free(t);

    t = agent_session_title_clip(NULL, 0);
    AGENT_TEST_ASSERT(t && !strcmp(t, "(no user prompt)"));
    free(t);
}

static void test_agent_session_title_from_file(void) {
    char dir[] = "/tmp/ds4_agent_l3_XXXXXX";
    AGENT_TEST_ASSERT(mkdtemp(dir) != NULL);
    char path[512];
    snprintf(path, sizeof(path), "%s/session.kv", dir);
    FILE *fp = fopen(path, "wb");
    AGENT_TEST_ASSERT(fp != NULL);

    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    ds4_kvstore_fill_header(h, 1, 4, DS4_KVSTORE_REASON_AGENT_SESSION,
                            DS4_KVSTORE_EXT_SESSION_TITLE,
                            100, 0, 8192, 12345, 12345, 0);
    AGENT_TEST_ASSERT(fwrite(h, 1, sizeof(h), fp) == sizeof(h));

    static const char text[] =
        "pre <" AGENT_DSML_BAR "User" AGENT_DSML_BAR "> from the transcript <" AGENT_DSML_BAR "Assistant" AGENT_DSML_BAR "> ok";
    uint8_t tb[4];
    ds4_kvstore_le_put32(tb, (uint32_t)(sizeof(text) - 1));
    AGENT_TEST_ASSERT(fwrite(tb, 1, 4, fp) == 4);
    AGENT_TEST_ASSERT(fwrite(text, 1, sizeof(text) - 1, fp) == sizeof(text) - 1);

    /* title trailer */
    static const char title[] = "from the transcript";
    ds4_kvstore_le_put32(tb, (uint32_t)(sizeof(title) - 1));
    AGENT_TEST_ASSERT(fwrite(tb, 1, 4, fp) == 4);
    AGENT_TEST_ASSERT(fwrite(title, 1, sizeof(title) - 1, fp) == sizeof(title) - 1);
    AGENT_TEST_ASSERT(fclose(fp) == 0);

    char *got = agent_session_title_from_file(path, 0);
    AGENT_TEST_ASSERT(got && !strcmp(got, "from the transcript"));
    free(got);

    /* clipped variant */
    got = agent_session_title_from_file(path, 8);
    AGENT_TEST_ASSERT(got && !strcmp(got, "from ..."));
    free(got);

    unlink(path);
    rmdir(dir);
}

static void test_agent_kv_title_trailer_round_trip(void) {
    char dir[] = "/tmp/ds4_agent_l3_XXXXXX";
    AGENT_TEST_ASSERT(mkdtemp(dir) != NULL);
    char path[512];
    snprintf(path, sizeof(path), "%s/trailer.kv", dir);

    FILE *fp = fopen(path, "wb");
    AGENT_TEST_ASSERT(fp != NULL);
    char err[128];
    AGENT_TEST_ASSERT(agent_kv_write_title_trailer(fp, "hello world", err, sizeof(err)));
    AGENT_TEST_ASSERT(fclose(fp) == 0);

    fp = fopen(path, "rb");
    AGENT_TEST_ASSERT(fp != NULL);
    ds4_kvstore_entry hdr = {0};
    char *title = NULL;
    AGENT_TEST_ASSERT(agent_kv_read_title_trailer(fp, &hdr, &title, err, sizeof(err)));
    AGENT_TEST_ASSERT(title && !strcmp(title, "hello world"));
    free(title);
    AGENT_TEST_ASSERT(fclose(fp) == 0);

    unlink(path);
    rmdir(dir);
}

static void test_agent_kv_identity_sha(void) {
    ds4_kvstore_entry hdr = {0};
    char sha[41], expect[41];

    /* session-title flag -> identity from title + created_at */
    hdr.ext_flags = DS4_KVSTORE_EXT_SESSION_TITLE;
    hdr.created_at = 777;
    agent_kv_identity_sha(&hdr, "text", 4, "mytitle", sha);
    agent_session_identity_sha("mytitle", 777, expect);
    AGENT_TEST_ASSERT(!strcmp(sha, expect));

    /* no flag -> identity from raw text bytes, independent of title */
    hdr.ext_flags = 0;
    agent_kv_identity_sha(&hdr, "text", 4, "mytitle", sha);
    AGENT_TEST_ASSERT(strcmp(sha, expect));
    ds4_kvstore_sha1_bytes_hex("text", 4, expect);
    AGENT_TEST_ASSERT(!strcmp(sha, expect));
}

static void test_agent_compact_make_prompt(void) {
    char *p = agent_compact_make_prompt("out of memory");
    AGENT_TEST_ASSERT(p != NULL);
    AGENT_TEST_ASSERT(strstr(p, "durable task-state summary") != NULL);
    AGENT_TEST_ASSERT(strstr(p, "Compaction reason: out of memory") != NULL);
    free(p);

    p = agent_compact_make_prompt("");
    AGENT_TEST_ASSERT(p != NULL);
    AGENT_TEST_ASSERT(strstr(p, "Compaction reason:") == NULL);
    free(p);
}

static void test_agent_compact_summary_budget(void) {
    AGENT_TEST_ASSERT(agent_compact_summary_budget(100) == 256);       /* clamp min */
    AGENT_TEST_ASSERT(agent_compact_summary_budget(8192) == 1024);     /* ctx / 8 */
    AGENT_TEST_ASSERT(agent_compact_summary_budget(1 << 30) ==
                      AGENT_COMPACT_SUMMARY_MAX_TOKENS);               /* clamp max */
    AGENT_TEST_ASSERT(agent_compact_summary_budget(4096) == 512);
}

static void test_agent_session_list_sort(void) {
    agent_session_list_item a = {0}, b = {0}, c = {0};
    a.entry.last_used = 100; a.entry.created_at = 50;
    b.entry.last_used = 200; b.entry.created_at = 60;
    /* recent-first ordering */
    AGENT_TEST_ASSERT(agent_session_list_cmp_recent(&a, &b) > 0);
    AGENT_TEST_ASSERT(agent_session_list_cmp_recent(&b, &a) < 0);

    /* equal last_used -> tiebreak by sha */
    c.entry.last_used = 100; c.entry.created_at = 90;
    strcpy(a.entry.sha, "aaaa");
    strcpy(c.entry.sha, "zzzz");
    AGENT_TEST_ASSERT(agent_session_list_cmp_recent(&a, &c) < 0);
    AGENT_TEST_ASSERT(agent_session_list_cmp_recent(&c, &a) > 0);
}

static void test_agent_worker_state_helpers(void) {
    agent_worker w = {0};
    pthread_mutex_init(&w.mu, NULL);

    w.user_activity = false;
    AGENT_TEST_ASSERT(!agent_worker_has_user_session(&w));
    w.user_activity = true;
    AGENT_TEST_ASSERT(agent_worker_has_user_session(&w));

    w.session_title = xstrdup("x");
    w.session_sha[0] = '1'; w.session_sha[1] = '\0';
    w.session_created_at = 9;
    w.legacy_session_path_to_delete = xstrdup("/tmp/x");
    agent_worker_clear_session_identity(&w);
    AGENT_TEST_ASSERT(w.session_sha[0] == '\0');
    AGENT_TEST_ASSERT(w.session_title == NULL);
    AGENT_TEST_ASSERT(w.session_created_at == 0);
    AGENT_TEST_ASSERT(w.legacy_session_path_to_delete == NULL);

    pthread_mutex_destroy(&w.mu);
}

static void test_agent_proto_frame_round_trip(void) {
    proto_writer wr;
    size_t out_len = 0;
    unsigned char *f = proto_encode_varint(&wr, PROTO_C2S_POWER, 42, &out_len);
    proto_reader payload;
    unsigned char tag = 0;
    AGENT_TEST_ASSERT(proto_open_frame(f, out_len, &payload, &tag));
    AGENT_TEST_ASSERT(tag == PROTO_C2S_POWER);
    uint64_t v = 0;
    AGENT_TEST_ASSERT(proto_reader_varint(&payload, &v));
    AGENT_TEST_ASSERT(v == 42);
    free(f);

    /* truncated / length-mismatched frame rejected */
    unsigned char bad[] = {0, 0, 0, 10, PROTO_C2S_POWER};
    AGENT_TEST_ASSERT(!proto_open_frame(bad, sizeof(bad), &payload, &tag));

    /* a 1-byte payload (just the tag) is a valid frame */
    unsigned char ok[] = {0, 0, 0, 1, PROTO_C2S_POWER};
    AGENT_TEST_ASSERT(proto_open_frame(ok, sizeof(ok), &payload, &tag));
    AGENT_TEST_ASSERT(tag == PROTO_C2S_POWER);
}

static void agent_test_run_all(void) {
    test_agent_glm_tool_parser_single_arg();
    test_agent_glm_tool_parser_chunked_multi_arg();
    test_agent_glm_tool_parser_streams_param_state();
    test_agent_glm_tool_parser_multiple_adjacent_calls();
    test_agent_glm_stream_tool_call_chunked();
    test_agent_tool_argument_literal_markup();
    test_agent_glm_stream_ignores_tool_inside_think();
    test_agent_glm_stream_greedy_sampling_boundaries();
    test_agent_dsml_stream_tool_call_chunked();
    test_agent_glm_tool_parser_rejects_missing_value();

    /* Layer 3 */
    test_agent_session_identity_sha();
    test_agent_session_title_from_text();
    test_agent_session_title_clip();
    test_agent_session_title_from_file();
    test_agent_kv_title_trailer_round_trip();
    test_agent_kv_identity_sha();
    test_agent_compact_make_prompt();
    test_agent_compact_summary_budget();
    test_agent_session_list_sort();
    test_agent_worker_state_helpers();
    test_agent_proto_frame_round_trip();
}

#endif /* DS4_AGENT_TEST */
