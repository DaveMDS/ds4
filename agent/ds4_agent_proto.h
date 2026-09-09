/* ds4_agent_proto.h -- shared client/server binary wire protocol.
 *
 * Only symbols BOTH binaries need live here
 *
 * The server-side DSML/GLM streaming parser, marker/think trackers, and the
 * render-directive producer are server-internal and live in ds4_agent_server.c.
 * The client is a dumb painter and does not parse markup.
 *
 * It is self-contained (no engine, no file I/O).  Everything is `static inline`
 * so including the header in several translation units cannot create link
 * conflicts.
 */


#ifndef DS4_AGENT_PROTO_H
#define DS4_AGENT_PROTO_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>

#include "ds4_agent_utils.h"



/* ============================================================================
 * Binary wire protocol
 *
 * Framing: 4-byte big-endian payload length + payload.
 * Payload: 1-byte tag + fields.
 * Fields: varint (unsigned LEB128), string (varint len + bytes), f32
 * (4-byte little-endian), bool (1 byte), u64 (varint).
 * ========================================================================== */

#define PROTO_VERSION 1

/* Client -> Server tags */
#define PROTO_C2S_NEW_SESSION  0x01
#define PROTO_C2S_USER         0x02
#define PROTO_C2S_TOOL_RESULT  0x03
#define PROTO_C2S_SYSTEM       0x04
#define PROTO_C2S_STOP_TURN    0x05
#define PROTO_C2S_INTERRUPT    0x06
#define PROTO_C2S_COMPACT      0x07
#define PROTO_C2S_SAVE         0x08
#define PROTO_C2S_LIST         0x09
#define PROTO_C2S_SWITCH       0x0A
#define PROTO_C2S_DEL          0x0B
#define PROTO_C2S_STRIP        0x0C
#define PROTO_C2S_HISTORY      0x0D
#define PROTO_C2S_TOKENS       0x0E
#define PROTO_C2S_POWER        0x0F
#define PROTO_C2S_ATTACH_IMAGE 0x10

/* Server -> Client tags */
#define PROTO_S2C_HELLO        0x81
#define PROTO_S2C_STATUS       0x82
#define PROTO_S2C_TOKEN        0x83
#define PROTO_S2C_TURN_PAUSED  0x84
#define PROTO_S2C_TOOL_CALLS   0x8C
#define PROTO_S2C_SWITCH_DONE  0x85
#define PROTO_S2C_COMPACT_DONE 0x86
#define PROTO_S2C_SAVE_DONE    0x87
#define PROTO_S2C_HISTORY      0x88
#define PROTO_S2C_LIST         0x89
#define PROTO_S2C_COUNT        0x8A
#define PROTO_S2C_ERROR        0x8B

/* ---- Writer ---- */

typedef struct {
    unsigned char *buf;
    size_t len;
    size_t cap;
} proto_writer;

static inline void proto_writer_init(proto_writer *w) {
    w->buf = NULL;
    w->len = 0;
    w->cap = 0;
}

static inline void proto_writer_free(proto_writer *w) {
    free(w->buf);
}

static inline void proto_writer_reserve(proto_writer *w, size_t n) {
    if (w->len + n > w->cap) {
        size_t cap = w->cap ? w->cap * 2 : 128;
        while (cap < w->len + n) cap *= 2;
        w->buf = xrealloc(w->buf, cap);
        w->cap = cap;
    }
}

static inline void proto_writer_byte(proto_writer *w, unsigned char b) {
    proto_writer_reserve(w, 1);
    w->buf[w->len++] = b;
}

static inline void proto_writer_varint(proto_writer *w, uint64_t v) {
    do {
        unsigned char b = v & 0x7f;
        v >>= 7;
        if (v) b |= 0x80;
        proto_writer_byte(w, b);
    } while (v);
}

static inline void proto_writer_string(proto_writer *w, const char *s) {
    size_t n = s ? strlen(s) : 0;
    proto_writer_varint(w, n);
    if (n) {
        proto_writer_reserve(w, n);
        memcpy(w->buf + w->len, s, n);
        w->len += n;
    }
}

static inline void proto_writer_f32(proto_writer *w, float f) {
    union { uint32_t u; float f; } u;
    u.f = f;
    proto_writer_byte(w, u.u & 0xff);
    proto_writer_byte(w, (u.u >> 8) & 0xff);
    proto_writer_byte(w, (u.u >> 16) & 0xff);
    proto_writer_byte(w, (u.u >> 24) & 0xff);
}

static inline void proto_writer_bool(proto_writer *w, bool b) {
    proto_writer_byte(w, b ? 1 : 0);
}

/* Finish a message: prepend the 4-byte big-endian payload length and return the
 * framed wire buffer (caller frees). */
static inline unsigned char *proto_frame(proto_writer *w, size_t *out_len) {
    unsigned char *out = xmalloc(w->len + 4);
    out[0] = (unsigned char)((w->len >> 24) & 0xff);
    out[1] = (unsigned char)((w->len >> 16) & 0xff);
    out[2] = (unsigned char)((w->len >> 8) & 0xff);
    out[3] = (unsigned char)(w->len & 0xff);
    if (w->len) memcpy(out + 4, w->buf, w->len);
    if (out_len) *out_len = w->len + 4;
    return out;
}

/* ---- Reader ---- */

typedef struct {
    const unsigned char *p;
    const unsigned char *end;
    bool ok;
} proto_reader;

static inline void proto_reader_init(proto_reader *r, const void *data, size_t len) {
    r->p = (const unsigned char *)data;
    r->end = r->p + len;
    r->ok = true;
}

static inline bool proto_reader_byte(proto_reader *r, unsigned char *b) {
    if (!r->ok || r->p >= r->end) { r->ok = false; return false; }
    *b = *r->p++;
    return true;
}

static inline bool proto_reader_varint(proto_reader *r, uint64_t *v) {
    uint64_t res = 0;
    int shift = 0;
    for (int i = 0; i < 10; i++) {
        unsigned char b;
        if (!proto_reader_byte(r, &b)) return false;
        res |= (uint64_t)(b & 0x7f) << shift;
        if (!(b & 0x80)) { *v = res; return true; }
        shift += 7;
    }
    r->ok = false;
    return false;
}

static inline bool proto_reader_string(proto_reader *r, char **out) {
    uint64_t n;
    if (!proto_reader_varint(r, &n)) return false;
    if (n > (uint64_t)(r->end - r->p)) { r->ok = false; return false; }
    char *s = xmalloc(n + 1);
    memcpy(s, r->p, n);
    s[n] = '\0';
    r->p += n;
    *out = s;
    return true;
}

static inline bool proto_reader_f32(proto_reader *r, float *f) {
    unsigned char b[4];
    for (int i = 0; i < 4; i++) {
        if (!proto_reader_byte(r, &b[i])) return false;
    }
    union { uint32_t u; float f; } u;
    u.u = (uint32_t)b[0] | ((uint32_t)b[1] << 8) |
          ((uint32_t)b[2] << 16) | ((uint32_t)b[3] << 24);
    *f = u.f;
    return true;
}

static inline bool proto_reader_bool(proto_reader *r, bool *b) {
    unsigned char x;
    if (!proto_reader_byte(r, &x)) return false;
    *b = x != 0;
    return true;
}

/* Open a received framed message: verify the 4-byte length, expose the payload
 * reader (which includes the tag byte) and read the tag.  Returns false on
 * malformed length or truncation. */
static inline bool proto_open_frame(const void *data, size_t len,
                                    proto_reader *payload, unsigned char *tag) {
    if (len < 5) return false; /* 4-byte length + tag byte */
    const unsigned char *p = (const unsigned char *)data;
    uint32_t plen = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
                    ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    if (plen + 4 != len) return false;
    proto_reader_init(payload, p + 4, plen);
    if (!proto_reader_byte(payload, tag)) return false;
    return true;
}

/* ---- Generic messages: empty / string / varint ---- */

static inline unsigned char *proto_encode_empty(proto_writer *w, unsigned char tag,
                                                size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, tag);
    return proto_frame(w, out_len);
}

static inline bool proto_decode_empty(const void *data, size_t len,
                                      unsigned char tag, unsigned char *got_tag) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    return *got_tag == tag && r.p == r.end;
}

static inline unsigned char *proto_encode_string(proto_writer *w, unsigned char tag,
                                                 const char *s, size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, tag);
    proto_writer_string(w, s);
    return proto_frame(w, out_len);
}

static inline bool proto_decode_string(const void *data, size_t len,
                                       unsigned char tag, unsigned char *got_tag,
                                       char **s) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != tag) return false;
    return proto_reader_string(&r, s);
}

static inline unsigned char *proto_encode_varint(proto_writer *w, unsigned char tag,
                                                 uint64_t v, size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, tag);
    proto_writer_varint(w, v);
    return proto_frame(w, out_len);
}

static inline bool proto_decode_varint(const void *data, size_t len,
                                       unsigned char tag, unsigned char *got_tag,
                                       uint64_t *v) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != tag) return false;
    return proto_reader_varint(&r, v);
}

/* ---- NEW_SESSION ---- */

typedef struct {
    char *sys_extra; /* user -sys text (plain), malloc'd by decoder */
    uint64_t n_predict;
    float temperature;
    float top_p;
    float min_p;
    uint64_t think_mode;
    uint64_t seed;
    uint64_t power;
} proto_new_session_msg;

static inline unsigned char *proto_encode_new_session(proto_writer *w,
                                                      const proto_new_session_msg *m,
                                                      size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, PROTO_C2S_NEW_SESSION);
    proto_writer_string(w, m->sys_extra);
    proto_writer_varint(w, m->n_predict);
    proto_writer_f32(w, m->temperature);
    proto_writer_f32(w, m->top_p);
    proto_writer_f32(w, m->min_p);
    proto_writer_varint(w, m->think_mode);
    proto_writer_varint(w, m->seed);
    proto_writer_varint(w, m->power);
    return proto_frame(w, out_len);
}

static inline bool proto_decode_new_session(const void *data, size_t len,
                                            unsigned char *got_tag,
                                            proto_new_session_msg *m) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != PROTO_C2S_NEW_SESSION) return false;
    if (!proto_reader_string(&r, &m->sys_extra)) return false;
    if (!proto_reader_varint(&r, &m->n_predict)) return false;
    if (!proto_reader_f32(&r, &m->temperature)) return false;
    if (!proto_reader_f32(&r, &m->top_p)) return false;
    if (!proto_reader_f32(&r, &m->min_p)) return false;
    if (!proto_reader_varint(&r, &m->think_mode)) return false;
    if (!proto_reader_varint(&r, &m->seed)) return false;
    if (!proto_reader_varint(&r, &m->power)) return false;
    return true;
}


/* ---- ATTACH_IMAGE (C2S) ---- */

typedef struct {
    unsigned char *data;
    size_t data_len;
} proto_attach_image_msg;

static inline unsigned char *proto_encode_attach_image(
        proto_writer *w, const proto_attach_image_msg *m, size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, PROTO_C2S_ATTACH_IMAGE);
    proto_writer_varint(w, (uint64_t)m->data_len);
    if (m->data_len) {
        proto_writer_reserve(w, m->data_len);
        memcpy(w->buf + w->len, m->data, m->data_len);
        w->len += m->data_len;
    }
    return proto_frame(w, out_len);
}

static inline bool proto_decode_attach_image(const void *data, size_t len,
                                             unsigned char *got_tag,
                                             proto_attach_image_msg *m) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != PROTO_C2S_ATTACH_IMAGE) return false;
    m->data = NULL;
    m->data_len = 0;
    uint64_t n = 0;
    if (!proto_reader_varint(&r, &n)) return false;
    if (n > (uint64_t)(r.end - r.p)) { r.ok = false; return false; }
    if (n) {
        m->data = xmalloc((size_t)n);
        memcpy(m->data, r.p, (size_t)n);
        r.p += n;
        m->data_len = (size_t)n;
    }
    return true;
}


/* ---- TOKEN ---- */

typedef enum {
    PROTO_TOKEN_NORMAL,          /* prose: client applies markdown */
    PROTO_TOKEN_THINK,           /* hidden thinking: client renders grey */
    PROTO_TOKEN_TOOL_NAME,       /* tool name: client paints the tool line */
    PROTO_TOKEN_TOOL_PARAM_NAME, /* parameter label (name): client paints name= */
    PROTO_TOKEN_TOOL_PARAM_VALUE /* parameter value: client colours per kind */
} proto_token_kind;

typedef struct {
    uint64_t id;
    unsigned char kind;  /* type of token, used by client to select renderer */
    char *text;          /* malloc'd by decoder, caller frees */
} proto_token_msg;

static inline unsigned char *proto_encode_token(proto_writer *w,
                                                const proto_token_msg *m,
                                                size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, PROTO_S2C_TOKEN);
    proto_writer_varint(w, m->id);
    proto_writer_varint(w, m->kind);
    proto_writer_string(w, m->text);
    return proto_frame(w, out_len);
}

static inline bool proto_decode_token(const void *data, size_t len,
                                      unsigned char *got_tag,
                                      proto_token_msg *m) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != PROTO_S2C_TOKEN) return false;
    if (!proto_reader_varint(&r, &m->id)) return false;
    uint64_t kind = 0;
    if (!proto_reader_varint(&r, &kind)) return false;
    m->kind = (unsigned char)kind;
    if (!proto_reader_string(&r, &m->text)) return false;
    return true;
}

/* ---- TOOLS ---- */

typedef struct {
    char *name;
    char *value;
    bool is_string;
} proto_tool_arg;

typedef struct {
    char *name;
    proto_tool_arg *args;
    int argc;
    int argcap;
} proto_tool_call;

typedef struct {
    proto_tool_call *v;
    int len;
    int cap;
} proto_tool_calls;


static inline void proto_tool_call_free(proto_tool_call *c) {
    if (!c) return;
    free(c->name);
    for (int i = 0; i < c->argc; i++) {
        free(c->args[i].name);
        free(c->args[i].value);
    }
    free(c->args);
    memset(c, 0, sizeof(*c));
}

static inline void proto_tool_calls_free(proto_tool_calls *calls) {
    if (!calls) return;
    for (int i = 0; i < calls->len; i++) proto_tool_call_free(&calls->v[i]);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}

static inline void proto_tool_call_add_arg(proto_tool_call *c, const char *name,
                                           const char *value, size_t value_len,
                                           bool is_string) {
    if (c->argc == c->argcap) {
        c->argcap = c->argcap ? c->argcap * 2 : 4;
        c->args = xrealloc(c->args,
                           (size_t)c->argcap * sizeof(c->args[0]));
    }
    c->args[c->argc++] = (proto_tool_arg){
        .name = xstrdup(name),
        .value = xstrndup(value, value_len),
        .is_string = is_string,
    };
}

static inline void proto_tool_calls_push(proto_tool_calls *calls,
                                         proto_tool_call *call) {
    if (!call->name) return;
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 2;
        calls->v = xrealloc(calls->v,
                            (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = *call;
    memset(call, 0, sizeof(*call));
}

/* Serialize a parsed proto_tool_calls list (name + args). Sent by the server
 * when a complete tool_calls block has been parsed; the client executes these
 * calls without parsing the raw stream itself. */
static inline unsigned char *proto_encode_tool_calls(
        proto_writer *w, const proto_tool_calls *calls, size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, PROTO_S2C_TOOL_CALLS);
    proto_writer_varint(w, (uint64_t)calls->len);
    for (int i = 0; i < calls->len; i++) {
        const proto_tool_call *c = &calls->v[i];
        proto_writer_string(w, c->name ? c->name : "");
        proto_writer_varint(w, (uint64_t)c->argc);
        for (int j = 0; j < c->argc; j++) {
            const proto_tool_arg *a = &c->args[j];
            proto_writer_string(w, a->name ? a->name : "");
            proto_writer_string(w, a->value ? a->value : "");
            proto_writer_byte(w, a->is_string ? 1 : 0);
        }
    }
    return proto_frame(w, out_len);
}

static inline bool proto_decode_tool_calls(const void *data, size_t len,
                                           unsigned char *got_tag,
                                           proto_tool_calls *calls) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != PROTO_S2C_TOOL_CALLS) return false;
    uint64_t count = 0;
    if (!proto_reader_varint(&r, &count)) return false;
    if (count > 1024) return false;
    for (uint64_t i = 0; i < count; i++) {
        char *name = NULL;
        if (!proto_reader_string(&r, &name)) { proto_tool_calls_free(calls); return false; }
        uint64_t argc = 0;
        if (!proto_reader_varint(&r, &argc)) {
            free(name);
            proto_tool_calls_free(calls);
            return false;
        }
        proto_tool_call c;
        memset(&c, 0, sizeof(c));
        c.name = name;
        if (argc > 4096) {
            proto_tool_call_free(&c);
            proto_tool_calls_free(calls);
            return false;
        }
        for (uint64_t j = 0; j < argc; j++) {
            char *aname = NULL, *aval = NULL;
            unsigned char isstr = 0;
            if (!proto_reader_string(&r, &aname) ||
                !proto_reader_string(&r, &aval) ||
                !proto_reader_byte(&r, &isstr)) {
                free(aname); free(aval);
                proto_tool_call_free(&c);
                proto_tool_calls_free(calls);
                return false;
            }
            proto_tool_call_add_arg(&c, aname ? aname : "", aval ? aval : "", aval ? strlen(aval) : 0, isstr != 0);
            free(aname); free(aval);
        }
        proto_tool_calls_push(calls, &c);
    }
    return true;
}

/* ---- STATUS ---- */

/* Session/turn state, carried in proto_status_msg.state (PROTO_S2C_STATUS).
 * The server is authoritative: it decides the state, the client only mirrors it. */
typedef enum {
    AGENT_IDLE,
    AGENT_PREFILL,
    AGENT_GENERATING,
    AGENT_COMPACTING,
    AGENT_DRAINING,
    AGENT_SAVING,
    AGENT_ERROR,
    AGENT_STOPPED,
} proto_agent_state;

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
    char *error; /* malloc'd by decoder, caller frees; may be "" */
} proto_status_msg;

static inline void proto_status_write_fields(proto_writer *w,
                                             const proto_status_msg *m) {
    proto_writer_varint(w, m->state);
    proto_writer_varint(w, m->ctx_used);
    proto_writer_varint(w, m->ctx_size);
    proto_writer_varint(w, m->prefill_done);
    proto_writer_varint(w, m->prefill_total);
    proto_writer_f32(w, m->prefill_tps);
    proto_writer_varint(w, m->generated);
    proto_writer_f32(w, m->gen_tps);
    proto_writer_bool(w, m->greedy);
    proto_writer_varint(w, m->power);
    proto_writer_string(w, m->error ? m->error : "");
}

static inline bool proto_status_read_fields(proto_reader *r, proto_status_msg *m) {
    if (!proto_reader_varint(r, &m->state)) return false;
    if (!proto_reader_varint(r, &m->ctx_used)) return false;
    if (!proto_reader_varint(r, &m->ctx_size)) return false;
    if (!proto_reader_varint(r, &m->prefill_done)) return false;
    if (!proto_reader_varint(r, &m->prefill_total)) return false;
    if (!proto_reader_f32(r, &m->prefill_tps)) return false;
    if (!proto_reader_varint(r, &m->generated)) return false;
    if (!proto_reader_f32(r, &m->gen_tps)) return false;
    if (!proto_reader_bool(r, &m->greedy)) return false;
    if (!proto_reader_varint(r, &m->power)) return false;
    if (!proto_reader_string(r, &m->error)) return false;
    return true;
}

static inline unsigned char *proto_encode_status(proto_writer *w,
                                                 const proto_status_msg *m,
                                                 size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, PROTO_S2C_STATUS);
    proto_status_write_fields(w, m);
    return proto_frame(w, out_len);
}

static inline bool proto_decode_status(const void *data, size_t len,
                                       unsigned char *got_tag,
                                       proto_status_msg *m) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != PROTO_S2C_STATUS) return false;
    return proto_status_read_fields(&r, m);
}

/* ---- HELLO ---- */

typedef struct {
    char *session_title; /* malloc'd by decoder, caller frees */
    uint64_t session_created_at;
    proto_status_msg status;
} proto_hello_msg;

static inline unsigned char *proto_encode_hello(proto_writer *w,
                                                const proto_hello_msg *m,
                                                size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, PROTO_S2C_HELLO);
    proto_writer_string(w, m->session_title ? m->session_title : "");
    proto_writer_varint(w, m->session_created_at);
    proto_status_write_fields(w, &m->status);
    return proto_frame(w, out_len);
}

static inline bool proto_decode_hello(const void *data, size_t len,
                                      unsigned char *got_tag,
                                      proto_hello_msg *m) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != PROTO_S2C_HELLO) return false;
        if (!proto_reader_string(&r, &m->session_title)) return false;
    if (!proto_reader_varint(&r, &m->session_created_at)) return false;
    if (!proto_status_read_fields(&r, &m->status)) return false;
    return true;
}

/* ---- SAVE_DONE ---- */

typedef struct {
    char *sha; /* malloc'd by decoder, caller frees */
    uint64_t tokens;
} proto_save_done_msg;

static inline unsigned char *proto_encode_save_done(proto_writer *w,
                                                    const proto_save_done_msg *m,
                                                    size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, PROTO_S2C_SAVE_DONE);
    proto_writer_string(w, m->sha);
    proto_writer_varint(w, m->tokens);
    return proto_frame(w, out_len);
}

static inline bool proto_decode_save_done(const void *data, size_t len,
                                          unsigned char *got_tag,
                                          proto_save_done_msg *m) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != PROTO_S2C_SAVE_DONE) return false;
    if (!proto_reader_string(&r, &m->sha)) return false;
    if (!proto_reader_varint(&r, &m->tokens)) return false;
    return true;
}

/* ---- LIST ---- */

typedef struct {
    char sha[41];            /* 40-hex session id + NUL */
    char *title;             /* malloc'd by decoder, caller frees */
    uint64_t last_used;
    uint64_t created_at;
    uint64_t tokens;
    uint64_t file_size;
    uint64_t payload_bytes;  /* 0 => stripped */
} proto_list_item;

typedef struct {
    proto_list_item *items;
    size_t count;
    size_t cap;
} proto_list_msg;

static inline void proto_list_msg_free(proto_list_msg *m) {
    if (m->items) {
        for (size_t i = 0; i < m->count; i++) free(m->items[i].title);
        free(m->items);
    }
    m->items = NULL;
    m->count = 0;
    m->cap = 0;
}

static inline unsigned char *proto_encode_list(proto_writer *w,
                                               const proto_list_msg *m,
                                               size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, PROTO_S2C_LIST);
    proto_writer_varint(w, (uint64_t)m->count);
    for (size_t i = 0; i < m->count; i++) {
        const proto_list_item *it = &m->items[i];
        proto_writer_string(w, it->sha);
        proto_writer_string(w, it->title ? it->title : "");
        proto_writer_varint(w, it->last_used);
        proto_writer_varint(w, it->created_at);
        proto_writer_varint(w, it->tokens);
        proto_writer_varint(w, it->file_size);
        proto_writer_varint(w, it->payload_bytes);
    }
    return proto_frame(w, out_len);
}

static inline bool proto_decode_list(const void *data, size_t len,
                                     unsigned char *got_tag,
                                     proto_list_msg *m) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != PROTO_S2C_LIST) return false;
    uint64_t count;
    if (!proto_reader_varint(&r, &count)) return false;
    m->items = NULL;
    m->count = 0;
    m->cap = 0;
    if (count == 0) return true;
    if (count > 4096) return false; /* sanity bound */
    m->items = calloc((size_t)count, sizeof(m->items[0]));
    if (!m->items) return false;
    for (uint64_t i = 0; i < count; i++) {
        proto_list_item *it = &m->items[i];
        char *s = NULL;
        if (!proto_reader_string(&r, &s)) goto fail;
        strncpy(it->sha, s, 40);
        it->sha[40] = '\0';
        free(s);
        if (!proto_reader_string(&r, &it->title)) goto fail;
        if (!proto_reader_varint(&r, &it->last_used)) goto fail;
        if (!proto_reader_varint(&r, &it->created_at)) goto fail;
        if (!proto_reader_varint(&r, &it->tokens)) goto fail;
        if (!proto_reader_varint(&r, &it->file_size)) goto fail;
        if (!proto_reader_varint(&r, &it->payload_bytes)) goto fail;
        m->count++;
    }
    m->count = (size_t)count;
    return true;
fail:
    proto_list_msg_free(m);
    return false;
}

/* ---- SWITCH_DONE ---- */

typedef struct {
    char *sha; /* malloc'd by decoder, caller frees */
} proto_switch_done_msg;

static inline unsigned char *proto_encode_switch_done(proto_writer *w,
                                                      const proto_switch_done_msg *m,
                                                      size_t *out_len) {
    proto_writer_init(w);
    proto_writer_byte(w, PROTO_S2C_SWITCH_DONE);
    proto_writer_string(w, m->sha ? m->sha : "");
    return proto_frame(w, out_len);
}

static inline bool proto_decode_switch_done(const void *data, size_t len,
                                            unsigned char *got_tag,
                                            proto_switch_done_msg *m) {
    proto_reader r;
    if (!proto_open_frame(data, len, &r, got_tag)) return false;
    if (*got_tag != PROTO_S2C_SWITCH_DONE) return false;
    return proto_reader_string(&r, &m->sha);
}

#endif /* DS4_AGENT_PROTO_H */
