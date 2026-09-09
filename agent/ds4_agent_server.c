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

#include "ds4_agent_utils.h"
#include "ds4_agent_proto.h"
#include "ds4_tool_text.h"

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
}

#endif /* DS4_AGENT_TEST */
