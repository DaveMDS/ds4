/* Layer 1 tests for the ds4-agent wire protocol (ds4_agent_proto.[ch]).
 *
 * Pure: links only ds4_agent_proto.o (and, from T2, ds4_agent_utils.o). No
 * engine, no sockets, no model. Covers the framing and codec edge cases listed
 * in AGENT-SPLIT-PLAN.md section 1.
 */

#include "../ds4_agent_proto.h"
#include "../ds4_agent_utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int failures;

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        failures++; \
    } \
} while (0)

/* ------------------------------------------------------------------------- */
/* helpers                                                                    */
/* ------------------------------------------------------------------------- */

/* Feed a whole serialized frame at once and pull exactly one frame back. */
static bool pump_one(const ap_buf *frame, uint32_t *type, ap_buf *payload) {
    ap_buf rx;
    ap_buf_init(&rx);
    ap_buf_append(&rx, frame->data, frame->len);
    char err[256] = {0};
    ap_frame_status st = ap_read_frame(&rx, type, payload, err, sizeof(err));
    ap_buf_free(&rx);
    return st == AP_FRAME_OK;
}

static bool slice_eq(const char *ptr, size_t len, const char *s) {
    return len == strlen(s) && memcmp(ptr, s, len) == 0;
}

/* ------------------------------------------------------------------------- */
/* primitives                                                                 */
/* ------------------------------------------------------------------------- */

static void test_primitive_roundtrip(void) {
    ap_buf b;
    ap_buf_init(&b);
    ap_put_u8(&b, 0x7F);
    ap_put_u16(&b, 0xBEEF);
    ap_put_u32(&b, 0xDEADBEEFu);
    ap_put_bool(&b, true);
    ap_put_bool(&b, false);
    ap_put_cstr(&b, "hello");
    ap_put_strn(&b, "", 0);
    const unsigned char raw[] = { 0, 1, 2, 253, 254, 255 };
    ap_put_blob(&b, raw, sizeof(raw));
    CHECK(!b.error);

    ap_reader r;
    ap_reader_init(&r, b.data, b.len);
    uint8_t u8 = 0;
    uint16_t u16 = 0;
    uint32_t u32 = 0;
    bool bt = false, bf = true;
    CHECK(ap_get_u8(&r, &u8) && u8 == 0x7F);
    CHECK(ap_get_u16(&r, &u16) && u16 == 0xBEEF);
    CHECK(ap_get_u32(&r, &u32) && u32 == 0xDEADBEEFu);
    CHECK(ap_get_bool(&r, &bt) && bt == true);
    CHECK(ap_get_bool(&r, &bf) && bf == false);
    const char *s = NULL;
    size_t slen = 999;
    CHECK(ap_get_strn(&r, &s, &slen) && slice_eq(s, slen, "hello"));
    CHECK(ap_get_strn(&r, &s, &slen) && slen == 0);
    const uint8_t *blob = NULL;
    size_t blen = 0;
    CHECK(ap_get_blob(&r, &blob, &blen) && blen == sizeof(raw) &&
          memcmp(blob, raw, sizeof(raw)) == 0);
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&b);
}

static void test_fixed_width_truncation(void) {
    /* u32 with only three bytes available -> reject, no OOB read. */
    unsigned char three[3] = { 1, 2, 3 };
    ap_reader r;
    ap_reader_init(&r, three, sizeof(three));
    uint32_t v = 0;
    CHECK(!ap_get_u32(&r, &v));
    CHECK(!ap_reader_ok(&r));
    /* sticky: a following smaller read also fails */
    uint8_t b = 0;
    CHECK(!ap_get_u8(&r, &b));

    /* u16 straddling the end */
    ap_reader_init(&r, three, 1);
    uint16_t h = 0;
    CHECK(!ap_get_u16(&r, &h));
}

static void test_bool_reject(void) {
    unsigned char bad[1] = { 2 };
    ap_reader r;
    ap_reader_init(&r, bad, 1);
    bool v = false;
    CHECK(!ap_get_bool(&r, &v));
    CHECK(!ap_reader_ok(&r));

    unsigned char ff[1] = { 0xFF };
    ap_reader_init(&r, ff, 1);
    CHECK(!ap_get_bool(&r, &v));
}

static void test_string_length_past_end(void) {
    /* claim 10 bytes, provide 2 */
    unsigned char buf[6] = { 0, 0, 0, 10, 'a', 'b' };
    ap_reader r;
    ap_reader_init(&r, buf, sizeof(buf));
    const char *s = NULL;
    size_t slen = 0;
    CHECK(!ap_get_strn(&r, &s, &slen));
    CHECK(!ap_reader_ok(&r));

    /* str_cap rejects an over-cap length without touching the destination */
    unsigned char big[4] = { 0, 0, 1, 0 }; /* len = 256 */
    ap_reader_init(&r, big, sizeof(big));
    char dst[16];
    memset(dst, 'x', sizeof(dst));
    CHECK(!ap_get_str_cap(&r, dst, sizeof(dst)));
    CHECK(!ap_reader_ok(&r));
}

static void test_str_cap_exact_fit(void) {
    ap_buf b;
    ap_buf_init(&b);
    ap_put_cstr(&b, "abcdef"); /* 6 chars */
    ap_reader r;
    ap_reader_init(&r, b.data, b.len);
    char dst[7]; /* exactly fits 6 + NUL */
    CHECK(ap_get_str_cap(&r, dst, sizeof(dst)));
    CHECK(strcmp(dst, "abcdef") == 0);
    CHECK(ap_reader_at_end(&r));

    ap_reader_init(&r, b.data, b.len);
    char tight[6]; /* one short */
    CHECK(!ap_get_str_cap(&r, tight, sizeof(tight)));
    ap_buf_free(&b);
}

static void test_fixed_point(void) {
    CHECK(ap_milli_from_double(0.7) == 700);
    CHECK(ap_milli_from_double(0.0) == 0);      /* explicit 0.0 is a valid milli value */
    CHECK(ap_milli_from_double(1.0) == 1000);
    CHECK(ap_milli_from_double(-1.0) == 0);     /* clamped */
    CHECK(ap_centi_from_double(12.70) == 1270);
    CHECK(ap_centi_from_double(0.0) == 0);

    CHECK(ap_milli_to_double(700) > 0.6999 && ap_milli_to_double(700) < 0.7001);
    CHECK(ap_centi_to_double(1270) > 12.699 && ap_centi_to_double(1270) < 12.701);

    /* round trip through the wire representation */
    for (int i = 0; i < 4000; i += 37) {
        double d = ap_milli_to_double((uint32_t)i);
        CHECK(ap_milli_from_double(d) == (uint32_t)i);
    }
}

/* ------------------------------------------------------------------------- */
/* framing                                                                    */
/* ------------------------------------------------------------------------- */

static void test_frame_roundtrip_and_byte_order(void) {
    ap_buf body;
    ap_buf_init(&body);
    ap_put_u32(&body, 0x01020304u);
    ap_buf frame;
    ap_buf_init(&frame);
    CHECK(ap_frame_encode(&frame, AGENT_MSG_STATUS, &body));

    /* header is big-endian regardless of host: magic 44 53 34 41 */
    CHECK(frame.len == AGENT_PROTO_HEADER_LEN + 4);
    CHECK(frame.data[0] == 0x44 && frame.data[1] == 0x53 &&
          frame.data[2] == 0x34 && frame.data[3] == 0x41);
    CHECK(frame.data[7] == AGENT_MSG_STATUS); /* type low byte */
    CHECK(frame.data[8] == 0 && frame.data[9] == 0 &&
          frame.data[10] == 0 && frame.data[11] == 4); /* len big-endian */

    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    CHECK(pump_one(&frame, &type, &payload));
    CHECK(type == AGENT_MSG_STATUS);
    CHECK(payload.len == 4);
    ap_reader r;
    ap_reader_init(&r, payload.data, payload.len);
    uint32_t v = 0;
    CHECK(ap_get_u32(&r, &v) && v == 0x01020304u);

    ap_buf_free(&body);
    ap_buf_free(&frame);
    ap_buf_free(&payload);
}

static void test_frame_empty_payload(void) {
    ap_buf frame;
    ap_buf_init(&frame);
    CHECK(ap_write_frame(&frame, AGENT_MSG_INTERRUPT, NULL, 0));
    CHECK(frame.len == AGENT_PROTO_HEADER_LEN);

    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    CHECK(pump_one(&frame, &type, &payload));
    CHECK(type == AGENT_MSG_INTERRUPT);
    CHECK(payload.len == 0);
    ap_buf_free(&frame);
    ap_buf_free(&payload);
}

static void test_frame_bad_magic(void) {
    ap_buf frame;
    ap_buf_init(&frame);
    ap_write_frame(&frame, AGENT_MSG_HELLO, "xyz", 3);

    /* corrupt to an unrelated magic */
    ap_buf bad = frame;
    unsigned char *copy = malloc(frame.len);
    memcpy(copy, frame.data, frame.len);
    bad.data = copy;
    copy[0] = 0xAA;

    ap_buf rx;
    ap_buf_init(&rx);
    ap_buf_append(&rx, bad.data, bad.len);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    char err[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, err, sizeof(err)) == AP_FRAME_ERROR);
    CHECK(strstr(err, "magic") != NULL);
    ap_buf_free(&rx);

    /* byte-swapped magic: write the magic value little-endian */
    copy[0] = 0x41; copy[1] = 0x34; copy[2] = 0x53; copy[3] = 0x44;
    ap_buf_init(&rx);
    ap_buf_append(&rx, copy, bad.len);
    err[0] = '\0';
    CHECK(ap_read_frame(&rx, &type, &payload, err, sizeof(err)) == AP_FRAME_ERROR);
    CHECK(strstr(err, "byte-swapped") != NULL);
    ap_buf_free(&rx);

    free(copy);
    ap_buf_free(&frame);
    ap_buf_free(&payload);
}

static void test_frame_oversize_rejected_before_bytes(void) {
    /* header only, declaring a payload one byte past the cap: must be ERROR,
     * not NEED_MORE -- the cap is checked before waiting for / allocating the
     * body. */
    unsigned char hdr[AGENT_PROTO_HEADER_LEN];
    uint32_t over = AGENT_PROTO_MAX_FRAME + 1u;
    hdr[0] = 0x44; hdr[1] = 0x53; hdr[2] = 0x34; hdr[3] = 0x41;
    hdr[4] = 0; hdr[5] = 0; hdr[6] = 0; hdr[7] = AGENT_MSG_TURN;
    hdr[8] = (unsigned char)(over >> 24); hdr[9] = (unsigned char)(over >> 16);
    hdr[10] = (unsigned char)(over >> 8); hdr[11] = (unsigned char)over;

    ap_buf rx;
    ap_buf_init(&rx);
    ap_buf_append(&rx, hdr, sizeof(hdr));
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    char err[128] = {0};
    CHECK(ap_read_frame(&rx, &type, &payload, err, sizeof(err)) == AP_FRAME_ERROR);
    CHECK(strstr(err, "too large") != NULL);
    ap_buf_free(&rx);

    /* exactly at the cap is allowed through the gate (just needs more bytes) */
    uint32_t atcap = AGENT_PROTO_MAX_FRAME;
    hdr[8] = (unsigned char)(atcap >> 24); hdr[9] = (unsigned char)(atcap >> 16);
    hdr[10] = (unsigned char)(atcap >> 8); hdr[11] = (unsigned char)atcap;
    ap_buf_init(&rx);
    ap_buf_append(&rx, hdr, sizeof(hdr));
    CHECK(ap_read_frame(&rx, &type, &payload, err, sizeof(err)) == AP_FRAME_NEED_MORE);
    ap_buf_free(&rx);
    ap_buf_free(&payload);

    /* ap_write_frame refuses an oversize payload up front */
    ap_buf f;
    ap_buf_init(&f);
    CHECK(!ap_write_frame(&f, AGENT_MSG_TURN, NULL, (size_t)AGENT_PROTO_MAX_FRAME + 1));
    CHECK(f.error);
    ap_buf_free(&f);
}

static void test_frame_reassembly_byte_by_byte(void) {
    ap_buf body;
    ap_buf_init(&body);
    ap_put_cstr(&body, "reassemble me");
    ap_buf frame;
    ap_buf_init(&frame);
    ap_frame_encode(&frame, AGENT_MSG_DRAIN_REPLY, &body);

    ap_buf rx;
    ap_buf_init(&rx);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    char err[128] = {0};

    for (size_t i = 0; i < frame.len; i++) {
        ap_buf_append(&rx, &frame.data[i], 1);
        ap_frame_status st = ap_read_frame(&rx, &type, &payload, err, sizeof(err));
        if (i + 1 < frame.len) {
            CHECK(st == AP_FRAME_NEED_MORE);
        } else {
            CHECK(st == AP_FRAME_OK);
        }
    }
    CHECK(type == AGENT_MSG_DRAIN_REPLY);
    ap_reader r;
    ap_reader_init(&r, payload.data, payload.len);
    ap_drain_reply d;
    CHECK(ap_decode_drain_reply(&r, &d));
    CHECK(slice_eq(d.text, d.text_len, "reassemble me"));
    CHECK(ap_reader_at_end(&r));

    ap_buf_free(&body);
    ap_buf_free(&frame);
    ap_buf_free(&rx);
    ap_buf_free(&payload);
}

static void test_frame_multiple_in_one_read(void) {
    ap_buf f1, f2, f3;
    ap_buf_init(&f1);
    ap_buf_init(&f2);
    ap_buf_init(&f3);
    ap_write_frame(&f1, AGENT_MSG_INTERRUPT, NULL, 0);
    ap_write_frame(&f2, AGENT_MSG_STOP, NULL, 0);
    ap_write_frame(&f3, AGENT_MSG_DRAIN_REQUEST, "tail", 4);

    ap_buf rx;
    ap_buf_init(&rx);
    ap_buf_append(&rx, f1.data, f1.len);
    ap_buf_append(&rx, f2.data, f2.len);
    ap_buf_append(&rx, f3.data, f3.len);

    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    char err[128] = {0};

    CHECK(ap_read_frame(&rx, &type, &payload, err, sizeof(err)) == AP_FRAME_OK &&
          type == AGENT_MSG_INTERRUPT && payload.len == 0);
    CHECK(ap_read_frame(&rx, &type, &payload, err, sizeof(err)) == AP_FRAME_OK &&
          type == AGENT_MSG_STOP);
    CHECK(ap_read_frame(&rx, &type, &payload, err, sizeof(err)) == AP_FRAME_OK &&
          type == AGENT_MSG_DRAIN_REQUEST && payload.len == 4);
    CHECK(ap_read_frame(&rx, &type, &payload, err, sizeof(err)) == AP_FRAME_NEED_MORE);
    CHECK(rx.len == 0);

    ap_buf_free(&f1);
    ap_buf_free(&f2);
    ap_buf_free(&f3);
    ap_buf_free(&rx);
    ap_buf_free(&payload);
}

/* ------------------------------------------------------------------------- */
/* generic reply codec                                                        */
/* ------------------------------------------------------------------------- */

static void test_generic_replies(void) {
    ap_buf b;
    uint32_t tag = 0;
    ap_reader r;

    /* OK */
    ap_buf_init(&b);
    ap_put_reply_ok(&b);
    ap_reader_init(&r, b.data, b.len);
    CHECK(ap_get_reply_tag(&r, &tag) && tag == AP_REPLY_OK);
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&b);

    /* ERR */
    ap_buf_init(&b);
    ap_put_reply_err(&b, "busy");
    ap_reader_init(&r, b.data, b.len);
    CHECK(ap_get_reply_tag(&r, &tag) && tag == AP_REPLY_ERR);
    char msg[64];
    CHECK(ap_get_reply_err(&r, msg, sizeof(msg)) && strcmp(msg, "busy") == 0);
    ap_buf_free(&b);

    /* INT */
    ap_buf_init(&b);
    ap_put_reply_int(&b, 4096);
    ap_reader_init(&r, b.data, b.len);
    CHECK(ap_get_reply_tag(&r, &tag) && tag == AP_REPLY_INT);
    uint32_t iv = 0;
    CHECK(ap_get_reply_int(&r, &iv) && iv == 4096);
    ap_buf_free(&b);

    /* STR */
    ap_buf_init(&b);
    ap_put_reply_str(&b, "abc", 3);
    ap_reader_init(&r, b.data, b.len);
    CHECK(ap_get_reply_tag(&r, &tag) && tag == AP_REPLY_STR);
    const char *sp = NULL;
    size_t sl = 0;
    CHECK(ap_get_reply_str(&r, &sp, &sl) && slice_eq(sp, sl, "abc"));
    ap_buf_free(&b);

    /* MAP with mixed value types */
    ap_buf_init(&b);
    ap_map_writer w;
    ap_put_reply_map_begin(&b, &w);
    ap_map_put_cstr(&w, "sha", "deadbeef");
    ap_map_put_u32(&w, "tokens", 12345);
    ap_map_put_bool(&w, "is_current", true);
    ap_map_put_i64(&w, "created_at", 1700000000LL);
    ap_map_end(&w);
    ap_reader_init(&r, b.data, b.len);
    CHECK(ap_get_reply_tag(&r, &tag) && tag == AP_REPLY_MAP);
    ap_map m;
    CHECK(ap_get_map(&r, &m) && m.n == 4);
    char sha[32];
    CHECK(ap_map_get_str(&m, "sha", sha, sizeof(sha)) && strcmp(sha, "deadbeef") == 0);
    CHECK(ap_map_get_u32(&m, "tokens", 0) == 12345);
    CHECK(ap_map_get_bool(&m, "is_current", false) == true);
    CHECK(ap_map_get_i64(&m, "created_at", 0) == 1700000000LL);
    CHECK(ap_map_get_u32(&m, "absent", 77) == 77);
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&b);

    /* empty MAP */
    ap_buf_init(&b);
    ap_put_reply_map_begin(&b, &w);
    ap_map_end(&w);
    ap_reader_init(&r, b.data, b.len);
    CHECK(ap_get_reply_tag(&r, &tag) && tag == AP_REPLY_MAP);
    CHECK(ap_get_map(&r, &m) && m.n == 0);
    ap_buf_free(&b);

    /* ARR of two MAP rows + empty ARR */
    ap_buf_init(&b);
    ap_put_reply_arr_begin(&b, 2);
    for (int i = 0; i < 2; i++) {
        ap_map_begin(&w, &b);
        ap_map_put_u32(&w, "i", (uint32_t)i);
        ap_map_end(&w);
    }
    ap_reader_init(&r, b.data, b.len);
    CHECK(ap_get_reply_tag(&r, &tag) && tag == AP_REPLY_ARR);
    uint32_t rows = 0;
    CHECK(ap_get_arr(&r, &rows) && rows == 2);
    for (uint32_t i = 0; i < rows; i++) {
        CHECK(ap_get_map(&r, &m) && ap_map_get_u32(&m, "i", 999) == i);
    }
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&b);

    ap_buf_init(&b);
    ap_put_reply_arr_begin(&b, 0);
    ap_reader_init(&r, b.data, b.len);
    CHECK(ap_get_reply_tag(&r, &tag) && tag == AP_REPLY_ARR);
    CHECK(ap_get_arr(&r, &rows) && rows == 0);
    ap_buf_free(&b);
}

static void test_map_count_past_end(void) {
    /* MAP claiming one entry, no bytes follow */
    unsigned char buf[4] = { 0, 0, 0, 1 };
    ap_reader r;
    ap_reader_init(&r, buf, sizeof(buf));
    ap_map m;
    CHECK(!ap_get_map(&r, &m));
    CHECK(!ap_reader_ok(&r));

    /* MAP with an absurd entry count -> rejected before scanning */
    unsigned char big[4] = { 0xFF, 0xFF, 0xFF, 0xFF };
    ap_reader_init(&r, big, sizeof(big));
    CHECK(!ap_get_map(&r, &m));

    /* ARR with an absurd row count -> rejected */
    ap_reader_init(&r, big, sizeof(big));
    uint32_t rows = 0;
    CHECK(!ap_get_arr(&r, &rows));
}

/* ------------------------------------------------------------------------- */
/* typed messages                                                             */
/* ------------------------------------------------------------------------- */

static void decode_body(uint32_t want_type, const ap_buf *body, uint32_t *got_type,
                        ap_buf *payload) {
    ap_buf frame;
    ap_buf_init(&frame);
    CHECK(ap_frame_encode(&frame, want_type, body));
    CHECK(pump_one(&frame, got_type, payload));
    CHECK(*got_type == want_type);
    ap_buf_free(&frame);
}

static void test_hello_roundtrip(void) {
    ap_hello h = {0};
    h.proto_version = AGENT_PROTO_VERSION;
    snprintf(h.client_version, sizeof(h.client_version), "ds4-agent-client/1");
    snprintf(h.cwd, sizeof(h.cwd), "/home/dave/ds4");

    ap_buf body;
    ap_buf_init(&body);
    ap_encode_hello(&body, &h);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    decode_body(AGENT_MSG_HELLO, &body, &type, &payload);

    ap_reader r;
    ap_reader_init(&r, payload.data, payload.len);
    ap_hello got;
    CHECK(ap_decode_hello(&r, &got));
    CHECK(got.proto_version == AGENT_PROTO_VERSION);
    CHECK(strcmp(got.client_version, "ds4-agent-client/1") == 0);
    CHECK(strcmp(got.cwd, "/home/dave/ds4") == 0);
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&body);
    ap_buf_free(&payload);
}

static void test_hello_reply_and_err_path(void) {
    ap_hello_reply hr = {0};
    hr.proto_version = AGENT_PROTO_VERSION;
    hr.engine_is_glm = true;
    hr.has_vision = false;
    hr.ctx_size_cli = 32768;
    snprintf(hr.backend_name, sizeof(hr.backend_name), "rocm");
    hr.power_percent = 80;
    hr.mtp_draft_tokens = 3;
    snprintf(hr.model_name, sizeof(hr.model_name), "DeepSeek-V4-Flash");
    hr.vocab_size = 129280;
    hr.session_parked = true;

    ap_buf body;
    ap_buf_init(&body);
    ap_encode_hello_reply(&body, &hr);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    decode_body(AGENT_MSG_HELLO, &body, &type, &payload);

    ap_reader r;
    ap_reader_init(&r, payload.data, payload.len);
    ap_hello_reply got;
    char err[128] = {0};
    CHECK(ap_decode_hello_reply(&r, &got, err, sizeof(err)));
    CHECK(got.proto_version == AGENT_PROTO_VERSION);
    CHECK(got.engine_is_glm && !got.has_vision);
    CHECK(got.ctx_size_cli == 32768);
    CHECK(strcmp(got.backend_name, "rocm") == 0);
    CHECK(got.power_percent == 80 && got.mtp_draft_tokens == 3);
    CHECK(strcmp(got.model_name, "DeepSeek-V4-Flash") == 0);
    CHECK(got.vocab_size == 129280 && got.session_parked);
    ap_buf_free(&body);
    ap_buf_free(&payload);

    /* an ERR reply in place of the MAP is surfaced through the err buffer */
    ap_buf_init(&body);
    ap_put_reply_err(&body, "proto_version mismatch: rebuild both binaries");
    ap_reader_init(&r, body.data, body.len);
    err[0] = '\0';
    CHECK(!ap_decode_hello_reply(&r, &got, err, sizeof(err)));
    CHECK(strstr(err, "mismatch") != NULL);
    ap_buf_free(&body);
}

static void test_session_new_roundtrip(void) {
    ap_session_new_args a = {0};
    const char *sys = "You are a coding agent.";
    a.sys_text = sys;
    a.sys_text_len = strlen(sys);
    a.prefix_file_text = "";
    a.prefix_file_text_len = 0;
    a.temperature = ap_milli_from_double(0.7);
    a.temperature_set = true;
    a.top_p = ap_milli_from_double(0.95);
    a.top_p_set = true;
    a.min_p = 0;
    a.min_p_set = false;
    a.seed = 123456789u;
    a.think_mode = 2;
    a.n_predict = 4096;
    a.raw_prompt = false;
    a.hints_enabled = true;
    a.dir_steering_ffn = ap_milli_from_double(0.15);
    a.power_set = true;
    a.power = 90;

    ap_buf body;
    ap_buf_init(&body);
    ap_encode_session_new(&body, &a);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    decode_body(AGENT_MSG_SESSION, &body, &type, &payload);

    ap_reader r;
    ap_reader_init(&r, payload.data, payload.len);
    ap_session_msg m;
    CHECK(ap_decode_session(&r, &m));
    CHECK(m.subcmd == AGENT_SESSION_NEW && m.subcmd_known);
    CHECK(slice_eq(m.new_args.sys_text, m.new_args.sys_text_len, sys));
    CHECK(m.new_args.prefix_file_text_len == 0);
    CHECK(m.new_args.temperature == 700 && m.new_args.temperature_set);
    CHECK(m.new_args.top_p == 950 && m.new_args.top_p_set);
    CHECK(m.new_args.min_p == 0 && !m.new_args.min_p_set);
    CHECK(m.new_args.seed == 123456789u);
    CHECK(m.new_args.think_mode == 2 && m.new_args.n_predict == 4096);
    CHECK(!m.new_args.raw_prompt && m.new_args.hints_enabled);
    CHECK(m.new_args.dir_steering_ffn == 150);
    CHECK(m.new_args.power_set && m.new_args.power == 90);
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&body);
    ap_buf_free(&payload);
}

static void test_session_subcmds_and_range(void) {
    ap_buf body;
    ap_reader r;
    ap_session_msg m;

    /* simple verbs carry only the selector */
    const uint32_t simple[] = {
        AGENT_SESSION_RESUME, AGENT_SESSION_SAVE, AGENT_SESSION_LIST, AGENT_SESSION_COMPACT
    };
    for (size_t i = 0; i < sizeof(simple) / sizeof(simple[0]); i++) {
        ap_buf_init(&body);
        ap_encode_session_simple(&body, simple[i]);
        ap_reader_init(&r, body.data, body.len);
        CHECK(ap_decode_session(&r, &m));
        CHECK(m.subcmd == simple[i] && m.subcmd_known);
        CHECK(ap_reader_at_end(&r));
        ap_buf_free(&body);
    }

    /* switch */
    ap_session_switch_args sw = {0};
    snprintf(sw.sha_prefix, sizeof(sw.sha_prefix), "a1b2c3");
    sw.history_turns = 5;
    ap_buf_init(&body);
    ap_encode_session_switch(&body, &sw);
    ap_reader_init(&r, body.data, body.len);
    CHECK(ap_decode_session(&r, &m));
    CHECK(m.subcmd == AGENT_SESSION_SWITCH);
    CHECK(strcmp(m.switch_args.sha_prefix, "a1b2c3") == 0 && m.switch_args.history_turns == 5);
    ap_buf_free(&body);

    /* del + strip */
    ap_session_del_args del = {0};
    snprintf(del.sha_prefix, sizeof(del.sha_prefix), "ffff");
    del.strip = true;
    ap_buf_init(&body);
    ap_encode_session_del(&body, &del);
    ap_reader_init(&r, body.data, body.len);
    CHECK(ap_decode_session(&r, &m));
    CHECK(m.subcmd == AGENT_SESSION_DEL && m.del_args.strip &&
          strcmp(m.del_args.sha_prefix, "ffff") == 0);
    ap_buf_free(&body);

    /* out-of-range selector: decodes, flagged not known, no crash */
    ap_buf_init(&body);
    ap_put_u32(&body, 99);
    ap_reader_init(&r, body.data, body.len);
    CHECK(ap_decode_session(&r, &m));
    CHECK(m.subcmd == 99 && !m.subcmd_known);
    CHECK(!ap_session_subcmd_valid(99));
    ap_buf_free(&body);

    /* truncated NEW body -> hard decode failure */
    ap_buf_init(&body);
    ap_put_u32(&body, AGENT_SESSION_NEW);
    ap_put_cstr(&body, "sys"); /* nothing after */
    ap_reader_init(&r, body.data, body.len);
    CHECK(!ap_decode_session(&r, &m));
    ap_buf_free(&body);
}

static void test_config_roundtrip_and_range(void) {
    ap_buf body;
    ap_reader r;
    ap_config_msg c;

    ap_buf_init(&body);
    ap_encode_config_get(&body, AGENT_CONFIG_POWER);
    ap_reader_init(&r, body.data, body.len);
    CHECK(ap_decode_config(&r, &c));
    CHECK(c.op == AGENT_CONFIG_GET && c.key == AGENT_CONFIG_POWER && c.known);
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&body);

    ap_buf_init(&body);
    ap_encode_config_set_u32(&body, AGENT_CONFIG_STEER, 250);
    ap_reader_init(&r, body.data, body.len);
    CHECK(ap_decode_config(&r, &c));
    CHECK(c.op == AGENT_CONFIG_SET && c.key == AGENT_CONFIG_STEER && c.known && c.u32val == 250);
    ap_buf_free(&body);

    ap_buf_init(&body);
    ap_encode_config_set_bool(&body, AGENT_CONFIG_HINTS, true);
    ap_reader_init(&r, body.data, body.len);
    CHECK(ap_decode_config(&r, &c));
    CHECK(c.op == AGENT_CONFIG_SET && c.key == AGENT_CONFIG_HINTS && c.known && c.boolval);
    ap_buf_free(&body);

    /* op / key out of range: decoded, flagged not known */
    ap_buf_init(&body);
    ap_put_u32(&body, 7); /* bad op */
    ap_put_u32(&body, 0);
    ap_reader_init(&r, body.data, body.len);
    CHECK(ap_decode_config(&r, &c));
    CHECK(!c.known);
    ap_buf_free(&body);

    ap_buf_init(&body);
    ap_put_u32(&body, AGENT_CONFIG_SET);
    ap_put_u32(&body, 9); /* bad key */
    ap_reader_init(&r, body.data, body.len);
    CHECK(ap_decode_config(&r, &c));
    CHECK(!c.known);
    ap_buf_free(&body);
}

static void test_turn_with_inline_image(void) {
    /* ~1 MiB blob: exercises a large inline payload well under the 32 MiB cap */
    size_t n = 1u << 20;
    unsigned char *img = malloc(n);
    for (size_t i = 0; i < n; i++) img[i] = (unsigned char)(i * 31u + 7u);

    ap_turn t = {0};
    const char *text = "look at test.png";
    t.text = text;
    t.text_len = strlen(text);
    t.image_count = 1;
    t.images[0].bytes = img;
    t.images[0].bytes_len = n;
    t.images[0].source = "test.png";
    t.images[0].source_len = 8;

    ap_buf body;
    ap_buf_init(&body);
    ap_encode_turn(&body, &t);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    decode_body(AGENT_MSG_TURN, &body, &type, &payload);
    CHECK(payload.len < AGENT_PROTO_MAX_FRAME);

    ap_reader r;
    ap_reader_init(&r, payload.data, payload.len);
    ap_turn got;
    CHECK(ap_decode_turn(&r, &got));
    CHECK(slice_eq(got.text, got.text_len, text));
    CHECK(got.image_count == 1);
    CHECK(got.images[0].bytes_len == n && memcmp(got.images[0].bytes, img, n) == 0);
    CHECK(slice_eq(got.images[0].source, got.images[0].source_len, "test.png"));
    CHECK(ap_reader_at_end(&r));

    /* empty text, no images */
    ap_turn empty = {0};
    empty.text = "";
    ap_buf_reset(&body);
    ap_encode_turn(&body, &empty);
    ap_reader_init(&r, body.data, body.len);
    CHECK(ap_decode_turn(&r, &got) && got.text_len == 0 && got.image_count == 0);

    /* too many images -> rejected */
    ap_buf_reset(&body);
    ap_put_strn(&body, "", 0);
    ap_put_u32(&body, AP_MAX_IMAGES + 1);
    ap_reader_init(&r, body.data, body.len);
    CHECK(!ap_decode_turn(&r, &got));

    free(img);
    ap_buf_free(&body);
    ap_buf_free(&payload);
}

static void test_stream_roundtrip_and_kind_range(void) {
    for (uint32_t k = 0; k < AGENT_STREAM_KIND_COUNT; k++) {
        ap_stream s = {0};
        s.stream_id = 0xFFFFFFFEu; /* large id */
        s.kind = k;
        s.text = "fragment";
        s.text_len = 8;
        ap_buf body;
        ap_buf_init(&body);
        ap_encode_stream(&body, &s);
        ap_reader r;
        ap_reader_init(&r, body.data, body.len);
        ap_stream got;
        CHECK(ap_decode_stream(&r, &got));
        CHECK(got.stream_id == 0xFFFFFFFEu && got.kind == k);
        CHECK(slice_eq(got.text, got.text_len, "fragment"));
        CHECK(ap_reader_at_end(&r));
        ap_buf_free(&body);
    }

    /* unknown kind -> rejected */
    ap_buf body;
    ap_buf_init(&body);
    ap_put_u32(&body, 1);
    ap_put_u32(&body, 42);
    ap_put_strn(&body, "x", 1);
    ap_reader r;
    ap_reader_init(&r, body.data, body.len);
    ap_stream got;
    CHECK(!ap_decode_stream(&r, &got));
    ap_buf_free(&body);
}

static void test_status_roundtrip(void) {
    ap_status s = {0};
    s.state = AGENT_STATE_GENERATING;
    s.prefill_done = 120;
    s.prefill_total = 200;
    s.prefill_label = 3;
    s.prefill_tps = ap_centi_from_double(45.5);
    s.generated = 64;
    s.gen_tps = ap_centi_from_double(12.7);
    s.greedy_sampling = true;
    s.ctx_used = 8192;
    s.ctx_size = 32768;
    s.power_percent = 80;
    snprintf(s.error, sizeof(s.error), "%s", "");

    ap_buf body;
    ap_buf_init(&body);
    ap_encode_status(&body, &s);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    decode_body(AGENT_MSG_STATUS, &body, &type, &payload);

    ap_reader r;
    ap_reader_init(&r, payload.data, payload.len);
    ap_status got;
    CHECK(ap_decode_status(&r, &got));
    CHECK(got.state == AGENT_STATE_GENERATING);
    CHECK(got.prefill_done == 120 && got.prefill_total == 200 && got.prefill_label == 3);
    CHECK(got.prefill_tps == 4550);
    CHECK(got.gen_tps == 1270);
    CHECK(ap_centi_to_double(got.gen_tps) > 12.69 && ap_centi_to_double(got.gen_tps) < 12.71);
    CHECK(got.generated == 64 && got.greedy_sampling);
    CHECK(got.ctx_used == 8192 && got.ctx_size == 32768 && got.power_percent == 80);
    CHECK(got.error[0] == '\0');
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&body);
    ap_buf_free(&payload);

    /* bad state -> rejected */
    ap_buf_init(&body);
    ap_put_u32(&body, AGENT_STATE_COUNT + 3);
    ap_reader_init(&r, body.data, body.len);
    CHECK(!ap_decode_status(&r, &got));
    ap_buf_free(&body);
}

static void test_tool_calls_roundtrip(void) {
    ap_tool_calls t = {0};
    t.request_id = 0xABCDEF01u;
    t.call_count = 1;
    t.calls[0].name = "edit";
    t.calls[0].name_len = 4;
    t.calls[0].arg_count = 3;
    t.calls[0].args[0] = (ap_tool_arg){ "path", 4, "src/foo.c", 9, true };
    t.calls[0].args[1] = (ap_tool_arg){ "old", 3, "int x = 1;", 10, true };
    t.calls[0].args[2] = (ap_tool_arg){ "new", 3, "int x = 2;", 10, true };

    ap_buf body;
    ap_buf_init(&body);
    ap_encode_tool_calls(&body, &t);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    decode_body(AGENT_MSG_TOOL_CALLS, &body, &type, &payload);

    ap_reader r;
    ap_reader_init(&r, payload.data, payload.len);
    ap_tool_calls got;
    CHECK(ap_decode_tool_calls(&r, &got));
    CHECK(got.request_id == 0xABCDEF01u && got.call_count == 1);
    CHECK(slice_eq(got.calls[0].name, got.calls[0].name_len, "edit"));
    CHECK(got.calls[0].arg_count == 3);
    CHECK(slice_eq(got.calls[0].args[0].name, got.calls[0].args[0].name_len, "path"));
    CHECK(slice_eq(got.calls[0].args[0].value, got.calls[0].args[0].value_len, "src/foo.c"));
    CHECK(got.calls[0].args[2].is_string);
    CHECK(slice_eq(got.calls[0].args[2].value, got.calls[0].args[2].value_len, "int x = 2;"));
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&body);
    ap_buf_free(&payload);

    /* too many calls -> rejected */
    ap_buf_init(&body);
    ap_put_u32(&body, 1);
    ap_put_u32(&body, AP_MAX_TOOL_CALLS + 1);
    ap_reader_init(&r, body.data, body.len);
    CHECK(!ap_decode_tool_calls(&r, &got));
    ap_buf_free(&body);
}

static void test_tool_result_roundtrip(void) {
    ap_tool_result t = {0};
    t.request_id = 7;
    t.text_part_count = 2;
    t.text_parts[0].ptr = "line one\n";
    t.text_parts[0].len = 9;
    t.text_parts[1].ptr = "line two\n";
    t.text_parts[1].len = 9;
    unsigned char png[] = { 0x89, 'P', 'N', 'G', 1, 2, 3, 4 };
    t.image_count = 1;
    t.images[0].bytes = png;
    t.images[0].bytes_len = sizeof(png);
    t.images[0].source = "shot.png";
    t.images[0].source_len = 8;

    ap_buf body;
    ap_buf_init(&body);
    ap_encode_tool_result(&body, &t);
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    decode_body(AGENT_MSG_TOOL_RESULT, &body, &type, &payload);

    ap_reader r;
    ap_reader_init(&r, payload.data, payload.len);
    ap_tool_result got;
    CHECK(ap_decode_tool_result(&r, &got));
    CHECK(got.request_id == 7 && got.text_part_count == 2);
    CHECK(slice_eq(got.text_parts[0].ptr, got.text_parts[0].len, "line one\n"));
    CHECK(slice_eq(got.text_parts[1].ptr, got.text_parts[1].len, "line two\n"));
    CHECK(got.image_count == 1 && got.images[0].bytes_len == sizeof(png));
    CHECK(memcmp(got.images[0].bytes, png, sizeof(png)) == 0);
    CHECK(slice_eq(got.images[0].source, got.images[0].source_len, "shot.png"));
    CHECK(ap_reader_at_end(&r));
    ap_buf_free(&body);
    ap_buf_free(&payload);
}

static void test_session_ready_roundtrip(void) {
    ap_session_ready s = {0};
    s.ctx_used = 1024;
    s.ctx_size = 32000;
    s.distributed_route_ready = true;
    s.state = AGENT_STATE_IDLE;
    snprintf(s.note, sizeof(s.note), "resumed parked session");

    ap_buf body;
    ap_buf_init(&body);
    ap_encode_session_ready(&body, &s);
    ap_reader r;
    ap_reader_init(&r, body.data, body.len);
    ap_session_ready got;
    char err[128] = {0};
    CHECK(ap_decode_session_ready(&r, &got, err, sizeof(err)));
    CHECK(got.ctx_used == 1024 && got.ctx_size == 32000);
    CHECK(got.distributed_route_ready && got.state == AGENT_STATE_IDLE);
    CHECK(strcmp(got.note, "resumed parked session") == 0);
    ap_buf_free(&body);

    ap_buf_init(&body);
    ap_put_reply_err(&body, "sysprompt build failed");
    ap_reader_init(&r, body.data, body.len);
    err[0] = '\0';
    CHECK(!ap_decode_session_ready(&r, &got, err, sizeof(err)));
    CHECK(strstr(err, "sysprompt") != NULL);
    ap_buf_free(&body);
}

static void test_msg_names(void) {
    CHECK(strcmp(ap_msg_name(AGENT_MSG_HELLO), "HELLO") == 0);
    CHECK(strcmp(ap_msg_name(AGENT_MSG_TOOL_CALLS), "TOOL_CALLS") == 0);
    CHECK(strcmp(ap_msg_name(999), "?") == 0);
}

/* ------------------------------------------------------------------------- */
/* shared helpers (ds4_agent_utils)                                           */
/* ------------------------------------------------------------------------- */

static void test_utils_agent_buf_growth(void) {
    agent_buf b = {0};
    for (int i = 0; i < 5000; i++) agent_buf_puts(&b, "0123456789");
    CHECK(b.len == 50000);
    CHECK(!b.truncated);
    CHECK(b.ptr[b.len] == '\0');
    char *taken = agent_buf_take(&b);
    CHECK(strlen(taken) == 50000);
    CHECK(b.ptr == NULL && b.len == 0 && b.cap == 0);
    free(taken);
}

static void test_utils_agent_buf_truncation(void) {
    agent_buf b = {0};
    b.limit = 16;
    agent_buf_puts(&b, "abcdefghij");        /* 10, fits */
    agent_buf_puts(&b, "klmnopqrstuvwxyz");  /* crosses the 16-byte limit */
    CHECK(b.truncated);
    CHECK(b.len == 16);
    char *taken = agent_buf_take(&b);
    CHECK(strncmp(taken, "abcdefghijklmnop", 16) == 0);
    CHECK(strstr(taken, "[Output truncated") != NULL);
    free(taken);
}

static void test_utils_input_buf(void) {
    agent_input_buf b = {0};
    agent_input_buf_append(&b, "hello ", 6);
    agent_input_buf_append(&b, "world", 5);
    CHECK(b.len == 11 && strcmp(b.ptr, "hello world") == 0);
    char *taken = agent_input_buf_take(&b);
    CHECK(strcmp(taken, "hello world") == 0);
    CHECK(b.ptr == NULL);
    free(taken);

    /* take on an untouched buffer yields an owned empty string */
    agent_input_buf empty = {0};
    char *e = agent_input_buf_take(&empty);
    CHECK(e && e[0] == '\0');
    free(e);
    agent_input_buf_free(&empty);
}

static void test_utils_parsers(void) {
    CHECK(parse_int("42", "--x") == 42);
    CHECK(parse_nonnegative_int("0", "--x") == 0);
    CHECK(parse_u64("1", "--seed") == 1ull);
    CHECK(parse_u64("18446744073709551615", "--seed") == UINT64_MAX);

    float f = -1.0f;
    CHECK(parse_float_range("0.7", "--temp", 0.0f, 2.0f) > 0.69f);
    CHECK(parse_float_range("0.7", "--temp", 0.0f, 2.0f) < 0.71f);

    int p = 0;
    CHECK(parse_power_percent("50", &p) && p == 50);
    CHECK(!parse_power_percent("0", &p));
    CHECK(!parse_power_percent("101", &p));
    CHECK(!parse_power_percent("abc", &p));

    CHECK(parse_steering_level("2.5", &f) && f > 2.49f && f < 2.51f);
    CHECK(parse_steering_level("-3", &f) && f < -2.99f);
    CHECK(!parse_steering_level("200", &f));
    CHECK(!parse_steering_level("nan", &f));
    CHECK(!parse_steering_level("", &f));

    CHECK(agent_parse_bool_default("YES", false) == true);
    CHECK(agent_parse_bool_default("0", true) == false);
    CHECK(agent_parse_bool_default("maybe", true) == true);
    CHECK(agent_parse_bool_default(NULL, true) == true);
    CHECK(agent_parse_bool_default("", false) == false);
}

static void test_utils_mkdir_p(void) {
    char tmpl[] = "/tmp/ds4_agent_utils_XXXXXX";
    char *base = mkdtemp(tmpl);
    CHECK(base != NULL);
    if (!base) return;

    char nested[512];
    snprintf(nested, sizeof(nested), "%s/a/b/c/d", base);
    CHECK(agent_mkdir_p(nested));
    struct stat st;
    CHECK(stat(nested, &st) == 0 && S_ISDIR(st.st_mode));
    /* idempotent */
    CHECK(agent_mkdir_p(nested));
    CHECK(!agent_mkdir_p(""));

    /* cleanup */
    char cmd[600];
    snprintf(cmd, sizeof(cmd), "rm -rf '%s'", base);
    CHECK(system(cmd) == 0);
}

static void test_utils_misc(void) {
    char *s = xstrndup("abcdef", 3);
    CHECK(strcmp(s, "abc") == 0);
    free(s);

    double t0 = now_sec();
    double t1 = now_sec();
    CHECK(t1 >= t0);
}

int main(void) {
    test_primitive_roundtrip();
    test_fixed_width_truncation();
    test_bool_reject();
    test_string_length_past_end();
    test_str_cap_exact_fit();
    test_fixed_point();

    test_frame_roundtrip_and_byte_order();
    test_frame_empty_payload();
    test_frame_bad_magic();
    test_frame_oversize_rejected_before_bytes();
    test_frame_reassembly_byte_by_byte();
    test_frame_multiple_in_one_read();

    test_generic_replies();
    test_map_count_past_end();

    test_hello_roundtrip();
    test_hello_reply_and_err_path();
    test_session_new_roundtrip();
    test_session_subcmds_and_range();
    test_config_roundtrip_and_range();
    test_turn_with_inline_image();
    test_stream_roundtrip_and_kind_range();
    test_status_roundtrip();
    test_tool_calls_roundtrip();
    test_tool_result_roundtrip();
    test_session_ready_roundtrip();
    test_msg_names();

    test_utils_agent_buf_growth();
    test_utils_agent_buf_truncation();
    test_utils_input_buf();
    test_utils_parsers();
    test_utils_mkdir_p();
    test_utils_misc();

    if (failures) {
        fprintf(stderr, "%d agent proto test(s) failed\n", failures);
        return 1;
    }
    puts("agent proto tests passed");
    return 0;
}
