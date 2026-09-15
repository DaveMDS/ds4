/* Wire protocol for the ds4-agent client/server split. See ds4_agent_proto.h.
 *
 * Everything here works on plain byte buffers. Big-endian integers are
 * assembled and read byte by byte, so the encoding does not depend on the host
 * byte order and needs no networking header.
 */

#include "ds4_agent_proto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ========================================================================= */
/* Growable write buffer                                                      */
/* ========================================================================= */

void ap_buf_init(ap_buf *b) {
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
    b->error = false;
}

void ap_buf_free(ap_buf *b) {
    free(b->data);
    b->data = NULL;
    b->len = 0;
    b->cap = 0;
    b->error = false;
}

void ap_buf_reset(ap_buf *b) {
    b->len = 0;
    /* keep the allocation and the error flag: a reset buffer that failed to
     * grow once is still unusable until re-init */
}

static bool ap_buf_grow(ap_buf *b, size_t need) {
    if (b->error) return false;
    if (need <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 64;
    while (cap < need) {
        if (cap > (SIZE_MAX / 2)) {
            b->error = true;
            return false;
        }
        cap *= 2;
    }
    void *p = realloc(b->data, cap);
    if (!p) {
        b->error = true;
        return false;
    }
    b->data = p;
    b->cap = cap;
    return true;
}

bool ap_buf_append(ap_buf *b, const void *data, size_t n) {
    if (b->error) return false;
    if (n == 0) return true;
    if (!ap_buf_grow(b, b->len + n)) return false;
    memcpy(b->data + b->len, data, n);
    b->len += n;
    return true;
}

bool ap_buf_patch_u32(ap_buf *b, size_t pos, uint32_t v) {
    if (b->error) return false;
    if (pos > b->len || b->len - pos < 4) {
        b->error = true;
        return false;
    }
    b->data[pos + 0] = (uint8_t)(v >> 24);
    b->data[pos + 1] = (uint8_t)(v >> 16);
    b->data[pos + 2] = (uint8_t)(v >> 8);
    b->data[pos + 3] = (uint8_t)(v);
    return true;
}

void ap_put_u8(ap_buf *b, uint8_t v) {
    ap_buf_append(b, &v, 1);
}

void ap_put_u16(ap_buf *b, uint16_t v) {
    uint8_t t[2] = { (uint8_t)(v >> 8), (uint8_t)v };
    ap_buf_append(b, t, sizeof(t));
}

void ap_put_u32(ap_buf *b, uint32_t v) {
    uint8_t t[4] = {
        (uint8_t)(v >> 24), (uint8_t)(v >> 16), (uint8_t)(v >> 8), (uint8_t)v
    };
    ap_buf_append(b, t, sizeof(t));
}

void ap_put_bool(ap_buf *b, bool v) {
    uint8_t t = v ? 1u : 0u;
    ap_buf_append(b, &t, 1);
}

void ap_put_strn(ap_buf *b, const char *s, size_t n) {
    if (n > 0xFFFFFFFFu) {
        b->error = true;
        return;
    }
    ap_put_u32(b, (uint32_t)n);
    if (n) ap_buf_append(b, s, n);
}

void ap_put_cstr(ap_buf *b, const char *s) {
    ap_put_strn(b, s ? s : "", s ? strlen(s) : 0);
}

void ap_put_blob(ap_buf *b, const void *data, size_t n) {
    ap_put_strn(b, (const char *)data, n);
}

/* ========================================================================= */
/* Bounds-checked read cursor                                                 */
/* ========================================================================= */

void ap_reader_init(ap_reader *r, const void *data, size_t len) {
    r->data = (const uint8_t *)data;
    r->len = len;
    r->pos = 0;
    r->error = false;
}

bool ap_reader_ok(const ap_reader *r) {
    return !r->error;
}

bool ap_reader_at_end(const ap_reader *r) {
    return !r->error && r->pos == r->len;
}

size_t ap_reader_remaining(const ap_reader *r) {
    return r->error ? 0 : (r->len - r->pos);
}

static bool ap_need(ap_reader *r, size_t n) {
    if (r->error) return false;
    if (n > r->len - r->pos) {
        r->error = true;
        return false;
    }
    return true;
}

bool ap_get_u8(ap_reader *r, uint8_t *out) {
    if (!ap_need(r, 1)) return false;
    *out = r->data[r->pos++];
    return true;
}

bool ap_get_u16(ap_reader *r, uint16_t *out) {
    if (!ap_need(r, 2)) return false;
    *out = (uint16_t)((uint16_t)r->data[r->pos] << 8 | r->data[r->pos + 1]);
    r->pos += 2;
    return true;
}

bool ap_get_u32(ap_reader *r, uint32_t *out) {
    if (!ap_need(r, 4)) return false;
    *out = (uint32_t)r->data[r->pos] << 24 |
           (uint32_t)r->data[r->pos + 1] << 16 |
           (uint32_t)r->data[r->pos + 2] << 8 |
           (uint32_t)r->data[r->pos + 3];
    r->pos += 4;
    return true;
}

bool ap_get_bool(ap_reader *r, bool *out) {
    if (!ap_need(r, 1)) return false;
    uint8_t v = r->data[r->pos++];
    if (v > 1) {
        r->error = true;
        return false;
    }
    *out = (v == 1);
    return true;
}

bool ap_get_strn(ap_reader *r, const char **out, size_t *out_len) {
    uint32_t n;
    if (!ap_get_u32(r, &n)) return false;
    if (!ap_need(r, n)) return false;
    *out = (const char *)(r->data + r->pos);
    *out_len = n;
    r->pos += n;
    return true;
}

bool ap_get_blob(ap_reader *r, const uint8_t **out, size_t *out_len) {
    return ap_get_strn(r, (const char **)out, out_len);
}

bool ap_get_str_cap(ap_reader *r, char *dst, size_t dstsz) {
    if (r->error) return false;
    if (dstsz == 0) {
        r->error = true;
        return false;
    }
    uint32_t n;
    if (!ap_get_u32(r, &n)) return false;
    if (n > dstsz - 1) {
        r->error = true;
        return false;
    }
    if (!ap_need(r, n)) return false;
    if (n) memcpy(dst, r->data + r->pos, n);
    dst[n] = '\0';
    r->pos += n;
    return true;
}

bool ap_skip_bytes(ap_reader *r, size_t n) {
    if (!ap_need(r, n)) return false;
    r->pos += n;
    return true;
}

/* ========================================================================= */
/* Fixed-point helpers                                                        */
/* ========================================================================= */

static uint32_t ap_scaled_from_double(double v, double scale) {
    if (!(v > 0.0)) return 0; /* also catches NaN */
    double s = v * scale + 0.5;
    if (s >= 4294967295.0) return 0xFFFFFFFFu;
    return (uint32_t)s;
}

uint32_t ap_milli_from_double(double v) { return ap_scaled_from_double(v, 1000.0); }
double   ap_milli_to_double(uint32_t v) { return (double)v / 1000.0; }
uint32_t ap_centi_from_double(double v) { return ap_scaled_from_double(v, 100.0); }
double   ap_centi_to_double(uint32_t v) { return (double)v / 100.0; }

/* ========================================================================= */
/* Frame IO                                                                   */
/* ========================================================================= */

bool ap_write_frame(ap_buf *out, uint32_t type, const void *payload, size_t payload_len) {
    if (payload_len > AGENT_PROTO_MAX_FRAME) {
        out->error = true;
        return false;
    }
    ap_put_u32(out, AGENT_PROTO_MAGIC);
    ap_put_u32(out, type);
    ap_put_u32(out, (uint32_t)payload_len);
    if (payload_len) ap_buf_append(out, payload, payload_len);
    return !out->error;
}

bool ap_frame_encode(ap_buf *out, uint32_t type, const ap_buf *body) {
    if (body && body->error) {
        out->error = true;
        return false;
    }
    return ap_write_frame(out, type,
                          body ? body->data : NULL,
                          body ? body->len : 0);
}

static uint32_t ap_bswap32(uint32_t v) {
    return (v >> 24) | ((v >> 8) & 0x0000FF00u) |
           ((v << 8) & 0x00FF0000u) | (v << 24);
}

ap_frame_status ap_read_frame(ap_buf *rx, uint32_t *type, ap_buf *payload_out,
                              char *err, size_t errlen) {
    if (rx->error) {
        if (err && errlen) snprintf(err, errlen, "receive buffer allocation failed");
        return AP_FRAME_ERROR;
    }
    if (rx->len < AGENT_PROTO_HEADER_LEN) return AP_FRAME_NEED_MORE;

    const uint8_t *h = rx->data;
    uint32_t magic = (uint32_t)h[0] << 24 | (uint32_t)h[1] << 16 |
                     (uint32_t)h[2] << 8 | (uint32_t)h[3];
    uint32_t type_v = (uint32_t)h[4] << 24 | (uint32_t)h[5] << 16 |
                      (uint32_t)h[6] << 8 | (uint32_t)h[7];
    uint32_t plen = (uint32_t)h[8] << 24 | (uint32_t)h[9] << 16 |
                    (uint32_t)h[10] << 8 | (uint32_t)h[11];

    if (magic != AGENT_PROTO_MAGIC) {
        if (err && errlen) {
            if (ap_bswap32(magic) == AGENT_PROTO_MAGIC)
                snprintf(err, errlen,
                         "byte-swapped frame magic 0x%08x (peer endianness or wrong program)",
                         magic);
            else
                snprintf(err, errlen, "bad frame magic 0x%08x", magic);
        }
        return AP_FRAME_ERROR;
    }
    if (plen > AGENT_PROTO_MAX_FRAME) {
        if (err && errlen)
            snprintf(err, errlen, "frame too large: %u bytes (max %u)",
                     plen, (unsigned)AGENT_PROTO_MAX_FRAME);
        return AP_FRAME_ERROR;
    }

    size_t total = (size_t)AGENT_PROTO_HEADER_LEN + plen;
    if (rx->len < total) return AP_FRAME_NEED_MORE;

    ap_buf_reset(payload_out);
    if (!ap_buf_append(payload_out, rx->data + AGENT_PROTO_HEADER_LEN, plen)) {
        if (err && errlen) snprintf(err, errlen, "payload allocation failed");
        return AP_FRAME_ERROR;
    }

    size_t remain = rx->len - total;
    if (remain) memmove(rx->data, rx->data + total, remain);
    rx->len = remain;

    *type = type_v;
    return AP_FRAME_OK;
}

const char *ap_msg_name(uint32_t type) {
    switch (type) {
        case AGENT_MSG_HELLO:         return "HELLO";
        case AGENT_MSG_SESSION:       return "SESSION";
        case AGENT_MSG_CONFIG:        return "CONFIG";
        case AGENT_MSG_TURN:          return "TURN";
        case AGENT_MSG_INTERRUPT:     return "INTERRUPT";
        case AGENT_MSG_STOP:          return "STOP";
        case AGENT_MSG_TOOL_RESULT:   return "TOOL_RESULT";
        case AGENT_MSG_DRAIN_REPLY:   return "DRAIN_REPLY";
        case AGENT_MSG_STREAM:        return "STREAM";
        case AGENT_MSG_STATUS:        return "STATUS";
        case AGENT_MSG_TOOL_CALLS:    return "TOOL_CALLS";
        case AGENT_MSG_DRAIN_REQUEST: return "DRAIN_REQUEST";
        default:                      return "?";
    }
}

/* ========================================================================= */
/* Generic reply codec                                                        */
/* ========================================================================= */

void ap_put_reply_ok(ap_buf *b) {
    ap_put_u32(b, AP_REPLY_OK);
}

void ap_put_reply_err(ap_buf *b, const char *message) {
    ap_put_u32(b, AP_REPLY_ERR);
    ap_put_cstr(b, message);
}

void ap_put_reply_int(ap_buf *b, uint32_t v) {
    ap_put_u32(b, AP_REPLY_INT);
    ap_put_u32(b, v);
}

void ap_put_reply_str(ap_buf *b, const char *s, size_t n) {
    ap_put_u32(b, AP_REPLY_STR);
    ap_put_strn(b, s, n);
}

void ap_map_begin(ap_map_writer *w, ap_buf *b) {
    w->b = b;
    w->n = 0;
    w->count_pos = b->len;
    ap_put_u32(b, 0); /* placeholder, backpatched by ap_map_end */
}

void ap_map_put_strn(ap_map_writer *w, const char *key, const char *val, size_t val_len) {
    ap_put_cstr(w->b, key);
    ap_put_strn(w->b, val, val_len);
    w->n++;
}

void ap_map_put_cstr(ap_map_writer *w, const char *key, const char *val) {
    ap_map_put_strn(w, key, val ? val : "", val ? strlen(val) : 0);
}

void ap_map_put_u32(ap_map_writer *w, const char *key, uint32_t val) {
    char t[16];
    int n = snprintf(t, sizeof(t), "%u", val);
    if (n < 0) n = 0;
    ap_map_put_strn(w, key, t, (size_t)n);
}

void ap_map_put_i64(ap_map_writer *w, const char *key, long long val) {
    char t[24];
    int n = snprintf(t, sizeof(t), "%lld", val);
    if (n < 0) n = 0;
    ap_map_put_strn(w, key, t, (size_t)n);
}

void ap_map_put_bool(ap_map_writer *w, const char *key, bool val) {
    ap_map_put_strn(w, key, val ? "1" : "0", 1);
}

void ap_map_end(ap_map_writer *w) {
    ap_buf_patch_u32(w->b, w->count_pos, w->n);
}

void ap_put_reply_map_begin(ap_buf *b, ap_map_writer *w) {
    ap_put_u32(b, AP_REPLY_MAP);
    ap_map_begin(w, b);
}

void ap_put_reply_arr_begin(ap_buf *b, uint32_t row_count) {
    ap_put_u32(b, AP_REPLY_ARR);
    ap_put_u32(b, row_count);
}

bool ap_get_reply_tag(ap_reader *r, uint32_t *tag) {
    return ap_get_u32(r, tag);
}

bool ap_get_reply_err(ap_reader *r, char *dst, size_t dstsz) {
    return ap_get_str_cap(r, dst, dstsz);
}

bool ap_get_reply_int(ap_reader *r, uint32_t *out) {
    return ap_get_u32(r, out);
}

bool ap_get_reply_str(ap_reader *r, const char **out, size_t *out_len) {
    return ap_get_strn(r, out, out_len);
}

bool ap_get_map(ap_reader *r, ap_map *out) {
    uint32_t n;
    if (!ap_get_u32(r, &n)) return false;
    if (n > AP_MAP_MAX_ENTRIES) {
        r->error = true;
        return false;
    }
    out->n = n;
    for (uint32_t i = 0; i < n; i++) {
        if (!ap_get_strn(r, &out->e[i].key, &out->e[i].key_len)) return false;
        if (!ap_get_strn(r, &out->e[i].val, &out->e[i].val_len)) return false;
    }
    return true;
}

bool ap_get_arr(ap_reader *r, uint32_t *row_count) {
    uint32_t n;
    if (!ap_get_u32(r, &n)) return false;
    if (n > AP_ARR_MAX_ROWS) {
        r->error = true;
        return false;
    }
    *row_count = n;
    return true;
}

const char *ap_map_find(const ap_map *m, const char *key, size_t *val_len) {
    size_t klen = strlen(key);
    for (uint32_t i = 0; i < m->n; i++) {
        if (m->e[i].key_len == klen && memcmp(m->e[i].key, key, klen) == 0) {
            if (val_len) *val_len = m->e[i].val_len;
            return m->e[i].val;
        }
    }
    return NULL;
}

bool ap_map_get_str(const ap_map *m, const char *key, char *dst, size_t dstsz) {
    if (dstsz == 0) return false;
    size_t vlen = 0;
    const char *v = ap_map_find(m, key, &vlen);
    if (!v) return false;
    if (vlen > dstsz - 1) vlen = dstsz - 1;
    if (vlen) memcpy(dst, v, vlen);
    dst[vlen] = '\0';
    return true;
}

uint32_t ap_map_get_u32(const ap_map *m, const char *key, uint32_t dflt) {
    char t[24];
    if (!ap_map_get_str(m, key, t, sizeof(t))) return dflt;
    if (t[0] == '\0') return dflt;
    char *end = NULL;
    unsigned long v = strtoul(t, &end, 10);
    if (end == t || *end != '\0') return dflt;
    return (uint32_t)v;
}

long long ap_map_get_i64(const ap_map *m, const char *key, long long dflt) {
    char t[24];
    if (!ap_map_get_str(m, key, t, sizeof(t))) return dflt;
    if (t[0] == '\0') return dflt;
    char *end = NULL;
    long long v = strtoll(t, &end, 10);
    if (end == t || *end != '\0') return dflt;
    return v;
}

bool ap_map_get_bool(const ap_map *m, const char *key, bool dflt) {
    char t[8];
    if (!ap_map_get_str(m, key, t, sizeof(t))) return dflt;
    if (strcmp(t, "1") == 0) return true;
    if (strcmp(t, "0") == 0) return false;
    return dflt;
}

/* Shared tail for the two MAP-shaped replies (HELLO reply, SESSION ready). */
static bool ap_decode_map_reply(ap_reader *r, ap_map *out, char *err, size_t errlen) {
    uint32_t tag;
    if (!ap_get_reply_tag(r, &tag)) {
        if (err && errlen) snprintf(err, errlen, "truncated reply");
        return false;
    }
    if (tag == AP_REPLY_ERR) {
        char msg[AP_CAP_ERROR];
        if (!ap_get_reply_err(r, msg, sizeof(msg))) {
            if (err && errlen) snprintf(err, errlen, "truncated ERR reply");
        } else if (err && errlen) {
            snprintf(err, errlen, "%s", msg);
        }
        return false;
    }
    if (tag != AP_REPLY_MAP) {
        if (err && errlen) snprintf(err, errlen, "unexpected reply tag %u", tag);
        return false;
    }
    if (!ap_get_map(r, out)) {
        if (err && errlen) snprintf(err, errlen, "malformed MAP reply");
        return false;
    }
    return true;
}

/* ========================================================================= */
/* Typed messages                                                             */
/* ========================================================================= */

void ap_encode_hello(ap_buf *body, const ap_hello *h) {
    ap_put_u32(body, h->proto_version);
    ap_put_cstr(body, h->client_version);
    ap_put_cstr(body, h->cwd);
}

bool ap_decode_hello(ap_reader *r, ap_hello *out) {
    memset(out, 0, sizeof(*out));
    if (!ap_get_u32(r, &out->proto_version)) return false;
    if (!ap_get_str_cap(r, out->client_version, sizeof(out->client_version))) return false;
    if (!ap_get_str_cap(r, out->cwd, sizeof(out->cwd))) return false;
    return true;
}

void ap_encode_hello_reply(ap_buf *body, const ap_hello_reply *h) {
    ap_map_writer w;
    ap_put_reply_map_begin(body, &w);
    ap_map_put_u32(&w, "proto_version", h->proto_version);
    ap_map_put_bool(&w, "engine_is_glm", h->engine_is_glm);
    ap_map_put_bool(&w, "has_vision", h->has_vision);
    ap_map_put_u32(&w, "ctx_size_cli", h->ctx_size_cli);
    ap_map_put_cstr(&w, "backend_name", h->backend_name);
    ap_map_put_u32(&w, "power_percent", h->power_percent);
    ap_map_put_u32(&w, "mtp_draft_tokens", h->mtp_draft_tokens);
    ap_map_put_cstr(&w, "model_name", h->model_name);
    ap_map_put_u32(&w, "vocab_size", h->vocab_size);
    ap_map_put_bool(&w, "session_parked", h->session_parked);
    ap_map_put_bool(&w, "model_loading", h->model_loading);
    ap_map_end(&w);
}

bool ap_decode_hello_reply(ap_reader *r, ap_hello_reply *out, char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    ap_map m;
    if (!ap_decode_map_reply(r, &m, err, errlen)) return false;
    out->proto_version = ap_map_get_u32(&m, "proto_version", 0);
    out->engine_is_glm = ap_map_get_bool(&m, "engine_is_glm", false);
    out->has_vision = ap_map_get_bool(&m, "has_vision", false);
    out->ctx_size_cli = ap_map_get_u32(&m, "ctx_size_cli", 0);
    ap_map_get_str(&m, "backend_name", out->backend_name, sizeof(out->backend_name));
    out->power_percent = ap_map_get_u32(&m, "power_percent", 0);
    out->mtp_draft_tokens = ap_map_get_u32(&m, "mtp_draft_tokens", 0);
    ap_map_get_str(&m, "model_name", out->model_name, sizeof(out->model_name));
    out->vocab_size = ap_map_get_u32(&m, "vocab_size", 0);
    out->session_parked = ap_map_get_bool(&m, "session_parked", false);
    out->model_loading = ap_map_get_bool(&m, "model_loading", false);
    return true;
}

bool ap_session_subcmd_valid(uint32_t subcmd) {
    return subcmd <= AGENT_SESSION_COMPACT;
}

void ap_encode_session_simple(ap_buf *body, uint32_t subcmd) {
    ap_put_u32(body, subcmd);
}

void ap_encode_session_new(ap_buf *body, const ap_session_new_args *a) {
    ap_put_u32(body, AGENT_SESSION_NEW);
    ap_put_strn(body, a->sys_text, a->sys_text_len);
    ap_put_strn(body, a->prefix_file_text, a->prefix_file_text_len);
    ap_put_u32(body, a->temperature);
    ap_put_bool(body, a->temperature_set);
    ap_put_u32(body, a->top_p);
    ap_put_bool(body, a->top_p_set);
    ap_put_u32(body, a->min_p);
    ap_put_bool(body, a->min_p_set);
    ap_put_u32(body, a->seed);
    ap_put_u32(body, a->think_mode);
    ap_put_u32(body, a->n_predict);
    ap_put_bool(body, a->raw_prompt);
    ap_put_bool(body, a->hints_enabled);
    ap_put_u32(body, a->dir_steering_ffn);
    ap_put_bool(body, a->power_set);
    ap_put_u32(body, a->power);
}

void ap_encode_session_switch(ap_buf *body, const ap_session_switch_args *a) {
    ap_put_u32(body, AGENT_SESSION_SWITCH);
    ap_put_cstr(body, a->sha_prefix);
    ap_put_u32(body, a->history_turns);
}

void ap_encode_session_del(ap_buf *body, const ap_session_del_args *a) {
    ap_put_u32(body, AGENT_SESSION_DEL);
    ap_put_cstr(body, a->sha_prefix);
    ap_put_bool(body, a->strip);
}

bool ap_decode_session(ap_reader *r, ap_session_msg *out) {
    memset(out, 0, sizeof(*out));
    if (!ap_get_u32(r, &out->subcmd)) return false;
    out->subcmd_known = ap_session_subcmd_valid(out->subcmd);
    if (!out->subcmd_known) return true; /* server replies ERR; no body to parse */

    switch (out->subcmd) {
        case AGENT_SESSION_NEW: {
            ap_session_new_args *a = &out->new_args;
            if (!ap_get_strn(r, &a->sys_text, &a->sys_text_len)) return false;
            if (!ap_get_strn(r, &a->prefix_file_text, &a->prefix_file_text_len)) return false;
            if (!ap_get_u32(r, &a->temperature)) return false;
            if (!ap_get_bool(r, &a->temperature_set)) return false;
            if (!ap_get_u32(r, &a->top_p)) return false;
            if (!ap_get_bool(r, &a->top_p_set)) return false;
            if (!ap_get_u32(r, &a->min_p)) return false;
            if (!ap_get_bool(r, &a->min_p_set)) return false;
            if (!ap_get_u32(r, &a->seed)) return false;
            if (!ap_get_u32(r, &a->think_mode)) return false;
            if (!ap_get_u32(r, &a->n_predict)) return false;
            if (!ap_get_bool(r, &a->raw_prompt)) return false;
            if (!ap_get_bool(r, &a->hints_enabled)) return false;
            if (!ap_get_u32(r, &a->dir_steering_ffn)) return false;
            if (!ap_get_bool(r, &a->power_set)) return false;
            if (!ap_get_u32(r, &a->power)) return false;
            return true;
        }
        case AGENT_SESSION_SWITCH: {
            ap_session_switch_args *a = &out->switch_args;
            if (!ap_get_str_cap(r, a->sha_prefix, sizeof(a->sha_prefix))) return false;
            if (!ap_get_u32(r, &a->history_turns)) return false;
            return true;
        }
        case AGENT_SESSION_DEL: {
            ap_session_del_args *a = &out->del_args;
            if (!ap_get_str_cap(r, a->sha_prefix, sizeof(a->sha_prefix))) return false;
            if (!ap_get_bool(r, &a->strip)) return false;
            return true;
        }
        default:
            /* resume / save / list / compact: no further body */
            return true;
    }
}

void ap_encode_session_ready(ap_buf *body, const ap_session_ready *s) {
    ap_map_writer w;
    ap_put_reply_map_begin(body, &w);
    ap_map_put_u32(&w, "ctx_used", s->ctx_used);
    ap_map_put_u32(&w, "ctx_size", s->ctx_size);
    ap_map_put_bool(&w, "distributed_route_ready", s->distributed_route_ready);
    ap_map_put_u32(&w, "state", s->state);
    ap_map_put_cstr(&w, "note", s->note);
    ap_map_end(&w);
}

bool ap_decode_session_ready(ap_reader *r, ap_session_ready *out, char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    ap_map m;
    if (!ap_decode_map_reply(r, &m, err, errlen)) return false;
    out->ctx_used = ap_map_get_u32(&m, "ctx_used", 0);
    out->ctx_size = ap_map_get_u32(&m, "ctx_size", 0);
    out->distributed_route_ready = ap_map_get_bool(&m, "distributed_route_ready", false);
    out->state = ap_map_get_u32(&m, "state", AGENT_STATE_IDLE);
    ap_map_get_str(&m, "note", out->note, sizeof(out->note));
    return true;
}

void ap_encode_config_get(ap_buf *body, uint32_t key) {
    ap_put_u32(body, AGENT_CONFIG_GET);
    ap_put_u32(body, key);
}

void ap_encode_config_set_u32(ap_buf *body, uint32_t key, uint32_t val) {
    ap_put_u32(body, AGENT_CONFIG_SET);
    ap_put_u32(body, key);
    ap_put_u32(body, val);
}

void ap_encode_config_set_bool(ap_buf *body, uint32_t key, bool val) {
    ap_put_u32(body, AGENT_CONFIG_SET);
    ap_put_u32(body, key);
    ap_put_bool(body, val);
}

bool ap_decode_config(ap_reader *r, ap_config_msg *out) {
    memset(out, 0, sizeof(*out));
    if (!ap_get_u32(r, &out->op)) return false;
    if (!ap_get_u32(r, &out->key)) return false;
    out->known = (out->op <= AGENT_CONFIG_SET) && (out->key <= AGENT_CONFIG_HINTS);
    if (!out->known) return true; /* server replies ERR */
    if (out->op == AGENT_CONFIG_SET) {
        if (out->key == AGENT_CONFIG_HINTS) {
            if (!ap_get_bool(r, &out->boolval)) return false;
        } else {
            if (!ap_get_u32(r, &out->u32val)) return false;
        }
    }
    return true;
}

static void ap_encode_images(ap_buf *body, const ap_wire_image *images, uint32_t count) {
    ap_put_u32(body, count);
    for (uint32_t i = 0; i < count; i++) {
        ap_put_blob(body, images[i].bytes, images[i].bytes_len);
        ap_put_strn(body, images[i].source, images[i].source_len);
    }
}

static bool ap_decode_images(ap_reader *r, ap_wire_image *images, uint32_t *count) {
    uint32_t n;
    if (!ap_get_u32(r, &n)) return false;
    if (n > AP_MAX_IMAGES) {
        r->error = true;
        return false;
    }
    for (uint32_t i = 0; i < n; i++) {
        if (!ap_get_blob(r, &images[i].bytes, &images[i].bytes_len)) return false;
        if (!ap_get_strn(r, &images[i].source, &images[i].source_len)) return false;
    }
    *count = n;
    return true;
}

void ap_encode_turn(ap_buf *body, const ap_turn *t) {
    ap_put_strn(body, t->text, t->text_len);
    ap_encode_images(body, t->images, t->image_count);
}

bool ap_decode_turn(ap_reader *r, ap_turn *out) {
    memset(out, 0, sizeof(*out));
    if (!ap_get_strn(r, &out->text, &out->text_len)) return false;
    if (!ap_decode_images(r, out->images, &out->image_count)) return false;
    return true;
}

void ap_encode_stream(ap_buf *body, const ap_stream *s) {
    ap_put_u32(body, s->stream_id);
    ap_put_u32(body, s->kind);
    ap_put_strn(body, s->text, s->text_len);
}

bool ap_decode_stream(ap_reader *r, ap_stream *out) {
    memset(out, 0, sizeof(*out));
    if (!ap_get_u32(r, &out->stream_id)) return false;
    if (!ap_get_u32(r, &out->kind)) return false;
    if (out->kind >= AGENT_STREAM_KIND_COUNT) {
        r->error = true;
        return false;
    }
    if (!ap_get_strn(r, &out->text, &out->text_len)) return false;
    return true;
}

void ap_encode_status(ap_buf *body, const ap_status *s) {
    ap_put_u32(body, s->state);
    ap_put_u32(body, s->prefill_done);
    ap_put_u32(body, s->prefill_total);
    ap_put_u32(body, s->prefill_label);
    ap_put_u32(body, s->prefill_tps);
    ap_put_u32(body, s->generated);
    ap_put_u32(body, s->gen_tps);
    ap_put_bool(body, s->greedy_sampling);
    ap_put_u32(body, s->ctx_used);
    ap_put_u32(body, s->ctx_size);
    ap_put_u32(body, s->power_percent);
    ap_put_cstr(body, s->error);
}

bool ap_decode_status(ap_reader *r, ap_status *out) {
    memset(out, 0, sizeof(*out));
    if (!ap_get_u32(r, &out->state)) return false;
    if (out->state >= AGENT_STATE_COUNT) {
        r->error = true;
        return false;
    }
    if (!ap_get_u32(r, &out->prefill_done)) return false;
    if (!ap_get_u32(r, &out->prefill_total)) return false;
    if (!ap_get_u32(r, &out->prefill_label)) return false;
    if (!ap_get_u32(r, &out->prefill_tps)) return false;
    if (!ap_get_u32(r, &out->generated)) return false;
    if (!ap_get_u32(r, &out->gen_tps)) return false;
    if (!ap_get_bool(r, &out->greedy_sampling)) return false;
    if (!ap_get_u32(r, &out->ctx_used)) return false;
    if (!ap_get_u32(r, &out->ctx_size)) return false;
    if (!ap_get_u32(r, &out->power_percent)) return false;
    if (!ap_get_str_cap(r, out->error, sizeof(out->error))) return false;
    return true;
}

void ap_encode_tool_calls(ap_buf *body, const ap_tool_calls *t) {
    ap_put_u32(body, t->request_id);
    ap_put_u32(body, t->call_count);
    for (uint32_t i = 0; i < t->call_count; i++) {
        const ap_tool_call *c = &t->calls[i];
        ap_put_strn(body, c->name, c->name_len);
        ap_put_u32(body, c->arg_count);
        for (uint32_t j = 0; j < c->arg_count; j++) {
            const ap_tool_arg *a = &c->args[j];
            ap_put_strn(body, a->name, a->name_len);
            ap_put_strn(body, a->value, a->value_len);
            ap_put_bool(body, a->is_string);
        }
    }
}

bool ap_decode_tool_calls(ap_reader *r, ap_tool_calls *out) {
    memset(out, 0, sizeof(*out));
    if (!ap_get_u32(r, &out->request_id)) return false;
    if (!ap_get_u32(r, &out->call_count)) return false;
    if (out->call_count > AP_MAX_TOOL_CALLS) {
        r->error = true;
        return false;
    }
    for (uint32_t i = 0; i < out->call_count; i++) {
        ap_tool_call *c = &out->calls[i];
        if (!ap_get_strn(r, &c->name, &c->name_len)) return false;
        if (!ap_get_u32(r, &c->arg_count)) return false;
        if (c->arg_count > AP_MAX_TOOL_ARGS) {
            r->error = true;
            return false;
        }
        for (uint32_t j = 0; j < c->arg_count; j++) {
            ap_tool_arg *a = &c->args[j];
            if (!ap_get_strn(r, &a->name, &a->name_len)) return false;
            if (!ap_get_strn(r, &a->value, &a->value_len)) return false;
            if (!ap_get_bool(r, &a->is_string)) return false;
        }
    }
    return true;
}

void ap_encode_tool_result(ap_buf *body, const ap_tool_result *t) {
    ap_put_u32(body, t->request_id);
    ap_put_u32(body, t->text_part_count);
    for (uint32_t i = 0; i < t->text_part_count; i++)
        ap_put_strn(body, t->text_parts[i].ptr, t->text_parts[i].len);
    ap_encode_images(body, t->images, t->image_count);
}

bool ap_decode_tool_result(ap_reader *r, ap_tool_result *out) {
    memset(out, 0, sizeof(*out));
    if (!ap_get_u32(r, &out->request_id)) return false;
    if (!ap_get_u32(r, &out->text_part_count)) return false;
    if (out->text_part_count > AP_MAX_TEXT_PARTS) {
        r->error = true;
        return false;
    }
    for (uint32_t i = 0; i < out->text_part_count; i++) {
        if (!ap_get_strn(r, &out->text_parts[i].ptr, &out->text_parts[i].len)) return false;
    }
    if (!ap_decode_images(r, out->images, &out->image_count)) return false;
    return true;
}

void ap_encode_drain_reply(ap_buf *body, const ap_drain_reply *d) {
    ap_put_strn(body, d->text, d->text_len);
}

bool ap_decode_drain_reply(ap_reader *r, ap_drain_reply *out) {
    memset(out, 0, sizeof(*out));
    return ap_get_strn(r, &out->text, &out->text_len);
}
