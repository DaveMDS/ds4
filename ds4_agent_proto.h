#ifndef DS4_AGENT_PROTO_H
#define DS4_AGENT_PROTO_H

/* Wire protocol for the ds4-agent client/server split.
 *
 * Pure C99, libc only: this header and ds4_agent_proto.c never include ds4.h
 * or any networking header. All framing and codec work is done on plain byte
 * buffers so both binaries and the protocol test link it without the engine.
 *
 * Framing: every frame is a 12-byte header {u32 magic, u32 type, u32 len}
 * followed by len payload bytes, ALL big-endian (assembled byte by byte, so
 * the wire layout is identical on any host and compatible with the htonl()
 * framing used by ds4_distributed). magic = 0x44533441 ("DS4A"). A wrong or
 * byte-swapped magic means a wrong peer / endianness and the connection is
 * closed. len is rejected before allocation when it exceeds
 * AGENT_PROTO_MAX_FRAME (32 MiB): images ride inline, there is no chunk
 * sub-protocol.
 *
 * Payload primitives (all integers big-endian):
 *   u8 / u16 / u32  fixed width
 *   bool            1 byte, rejected if not 0 or 1
 *   string          u32 len + len UTF-8 bytes, no NUL
 *   blob            u32 len + len octets
 *   array           u32 count + count elements
 * No varint. No float: scalar knobs travel as fixed-point u32 (milli-units,
 * value x1000; centi-units, value x100 for the display-only tps fields).
 *
 * Message bodies are positional: a fixed ordered sequence of primitives with a
 * hand-written encode/decode pair, no field tags, no optional fields. HELLO
 * negotiates proto_version; a mismatch is a hard error (rebuild both binaries).
 *
 * Borrowing: decoders keep large/variable fields (stream text, tool-call
 * argument values, image bytes, generic-map values) as pointers INTO the
 * decoded payload buffer. That buffer must outlive every use of the decoded
 * struct. Short, bounded fields (versions, shas, paths, error text, tool and
 * parameter names) are copied into fixed arrays and a value over the documented
 * cap is rejected.
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define AGENT_PROTO_MAGIC      0x44533441u /* "DS4A" */
#define AGENT_PROTO_VERSION    1u
#define AGENT_PROTO_HEADER_LEN 12u
#define AGENT_PROTO_MAX_FRAME  (32u * 1024u * 1024u)

/* Default TCP endpoint (see the CLI split in AGENT-SPLIT-PLAN.md). */
#define AGENT_PROTO_DEFAULT_PORT 7878
#define AGENT_PROTO_DEFAULT_HOST "127.0.0.1"

/* Bounded string caps (decode rejects anything longer). */
#define AP_CAP_VERSION 64
#define AP_CAP_NAME    128
#define AP_CAP_SHA     64
#define AP_CAP_TITLE   256
#define AP_CAP_NOTE    256
#define AP_CAP_ERROR   512
#define AP_CAP_PATH    4096
#define AP_CAP_MODEL   128

/* Fixed collection caps (decode rejects anything larger). */
#define AP_MAP_MAX_ENTRIES  32
#define AP_ARR_MAX_ROWS     65536
#define AP_MAX_IMAGES       16
#define AP_MAX_TOOL_CALLS   8
#define AP_MAX_TOOL_ARGS    24
#define AP_MAX_TEXT_PARTS   8

/* Frame types. Flat: one value per message, 8 client verbs + 4 server pushes.
 * SESSION and CONFIG carry a sub-command selector in their body rather than
 * spending a frame type on each. */
enum agent_msg {
    AGENT_MSG_INVALID = 0,
    /* client -> server */
    AGENT_MSG_HELLO = 1,
    AGENT_MSG_SESSION = 2,
    AGENT_MSG_CONFIG = 3,
    AGENT_MSG_TURN = 4,
    AGENT_MSG_INTERRUPT = 5,
    AGENT_MSG_STOP = 6,
    AGENT_MSG_TOOL_RESULT = 7,
    AGENT_MSG_DRAIN_REPLY = 8,
    /* server -> client */
    AGENT_MSG_STREAM = 9,
    AGENT_MSG_STATUS = 10,
    AGENT_MSG_TOOL_CALLS = 11,
    AGENT_MSG_DRAIN_REQUEST = 12,
};

/* SESSION body sub-command (first u32 of the body). */
enum agent_session_subcmd {
    AGENT_SESSION_NEW = 0,
    AGENT_SESSION_RESUME = 1,
    AGENT_SESSION_SAVE = 2,
    AGENT_SESSION_SWITCH = 3,
    AGENT_SESSION_LIST = 4,
    AGENT_SESSION_DEL = 5,
    AGENT_SESSION_COMPACT = 6,
};

/* CONFIG body: u32 op, u32 key, then the value on a set. */
enum agent_config_op  { AGENT_CONFIG_GET = 0, AGENT_CONFIG_SET = 1 };
enum agent_config_key { AGENT_CONFIG_POWER = 0, AGENT_CONFIG_STEER = 1, AGENT_CONFIG_HINTS = 2 };

/* Result of the server parser's byte-by-byte classification of the token
 * stream. A semantic tag: the protocol never names a colour, the client maps
 * kind -> paint. Adjacent same-kind fragments are coalesced by the server. */
enum agent_stream_kind {
    AGENT_STREAM_NORMAL = 0,           /* visible assistant text + [tool call ...] notices */
    AGENT_STREAM_THINK = 1,            /* inside <think>...</think>, tags already stripped */
    AGENT_STREAM_SUMMARY = 2,          /* compaction summary tokens */
    AGENT_STREAM_SYSTEM = 3,           /* the "*" system notices */
    AGENT_STREAM_TOOL_NAME = 4,        /* a just-recognised tool name */
    AGENT_STREAM_TOOL_PARAM_NAME = 5,  /* a tool-call parameter name */
    AGENT_STREAM_TOOL_PARAM_VALUE = 6, /* a chunk of the current parameter value */
    AGENT_STREAM_KIND_COUNT
};

/* Mirror of agent_worker_state in ds4_agent.c. */
enum agent_state {
    AGENT_STATE_IDLE = 0,
    AGENT_STATE_PREFILL = 1,
    AGENT_STATE_GENERATING = 2,
    AGENT_STATE_COMPACTING = 3,
    AGENT_STATE_DRAINING = 4,
    AGENT_STATE_SAVING = 5,
    AGENT_STATE_ERROR = 6,
    AGENT_STATE_STOPPED = 7,
    AGENT_STATE_COUNT
};

/* Generic reply tag: every SESSION / CONFIG reply is one of these. */
enum ap_reply_tag {
    AP_REPLY_OK = 0,   /* no body */
    AP_REPLY_ERR = 1,  /* string message */
    AP_REPLY_INT = 2,  /* u32 */
    AP_REPLY_STR = 3,  /* string */
    AP_REPLY_MAP = 4,  /* u32 n + n*(string key, string value) */
    AP_REPLY_ARR = 5,  /* u32 n + n*MAP */
};

/* ========================================================================= */
/* Growable write buffer                                                      */
/* ========================================================================= */

typedef struct {
    uint8_t *data;
    size_t len;
    size_t cap;
    bool error; /* sticky: once set, every further put is a no-op */
} ap_buf;

void ap_buf_init(ap_buf *b);
void ap_buf_free(ap_buf *b);
void ap_buf_reset(ap_buf *b); /* len = 0, keep the allocation */
bool ap_buf_append(ap_buf *b, const void *data, size_t n);
bool ap_buf_patch_u32(ap_buf *b, size_t pos, uint32_t v); /* backpatch a big-endian u32 */

void ap_put_u8(ap_buf *b, uint8_t v);
void ap_put_u16(ap_buf *b, uint16_t v);
void ap_put_u32(ap_buf *b, uint32_t v);
void ap_put_bool(ap_buf *b, bool v);
void ap_put_strn(ap_buf *b, const char *s, size_t n);
void ap_put_cstr(ap_buf *b, const char *s); /* strlen(s), NULL -> empty */
void ap_put_blob(ap_buf *b, const void *data, size_t n);

/* ========================================================================= */
/* Bounds-checked read cursor                                                 */
/* ========================================================================= */

typedef struct {
    const uint8_t *data;
    size_t len;
    size_t pos;
    bool error; /* sticky: once a get runs past the end, every further get fails */
} ap_reader;

void ap_reader_init(ap_reader *r, const void *data, size_t len);
bool ap_reader_ok(const ap_reader *r);       /* !error */
bool ap_reader_at_end(const ap_reader *r);    /* !error && pos == len */
size_t ap_reader_remaining(const ap_reader *r);

bool ap_get_u8(ap_reader *r, uint8_t *out);
bool ap_get_u16(ap_reader *r, uint16_t *out);
bool ap_get_u32(ap_reader *r, uint32_t *out);
bool ap_get_bool(ap_reader *r, bool *out);
/* Borrowing: *out points into r->data. */
bool ap_get_strn(ap_reader *r, const char **out, size_t *out_len);
bool ap_get_blob(ap_reader *r, const uint8_t **out, size_t *out_len);
/* Copying: rejects (sets r->error) when the source is longer than dstsz - 1. */
bool ap_get_str_cap(ap_reader *r, char *dst, size_t dstsz);
/* Skip a string/blob without materialising it. */
bool ap_skip_bytes(ap_reader *r, size_t n);

/* ========================================================================= */
/* Fixed-point helpers (no float on the wire)                                 */
/* ========================================================================= */

uint32_t ap_milli_from_double(double v); /* v * 1000, rounded, clamped to >= 0 */
double   ap_milli_to_double(uint32_t v); /* v / 1000.0 */
uint32_t ap_centi_from_double(double v); /* v * 100, rounded, clamped to >= 0 */
double   ap_centi_to_double(uint32_t v); /* v / 100.0 */

/* ========================================================================= */
/* Frame IO                                                                   */
/* ========================================================================= */

/* Append a complete frame (12-byte BE header + payload) to out.
 * Fails (returns false) when payload_len > AGENT_PROTO_MAX_FRAME. */
bool ap_write_frame(ap_buf *out, uint32_t type, const void *payload, size_t payload_len);
/* Same, taking the payload from an ap_buf (body may be NULL for empty). */
bool ap_frame_encode(ap_buf *out, uint32_t type, const ap_buf *body);

typedef enum {
    AP_FRAME_NEED_MORE = 0,
    AP_FRAME_OK = 1,
    AP_FRAME_ERROR = -1,
} ap_frame_status;

/* Pull one frame off the front of rx (a buffer fed by raw ap_buf_append of
 * received bytes). On AP_FRAME_OK *type is set and payload_out is reset and
 * filled with exactly the payload bytes; the consumed bytes are removed from
 * rx. On AP_FRAME_NEED_MORE nothing is consumed. On AP_FRAME_ERROR err holds a
 * message and the caller must close the connection. */
ap_frame_status ap_read_frame(ap_buf *rx, uint32_t *type, ap_buf *payload_out,
                              char *err, size_t errlen);

const char *ap_msg_name(uint32_t type); /* for traces; "?" on unknown */

/* ========================================================================= */
/* Generic reply codec (OK / ERR / INT / STR / MAP / ARR)                     */
/* ========================================================================= */

void ap_put_reply_ok(ap_buf *b);
void ap_put_reply_err(ap_buf *b, const char *message);
void ap_put_reply_int(ap_buf *b, uint32_t v);
void ap_put_reply_str(ap_buf *b, const char *s, size_t n);

/* MAP writer: begin, put entries in any order, end backpatches the count.
 * Usable mid-buffer, so it also builds the rows of an ARR reply. */
typedef struct {
    ap_buf *b;
    size_t count_pos;
    uint32_t n;
} ap_map_writer;

void ap_map_begin(ap_map_writer *w, ap_buf *b);
void ap_map_put_strn(ap_map_writer *w, const char *key, const char *val, size_t val_len);
void ap_map_put_cstr(ap_map_writer *w, const char *key, const char *val);
void ap_map_put_u32(ap_map_writer *w, const char *key, uint32_t val);
void ap_map_put_i64(ap_map_writer *w, const char *key, long long val);
void ap_map_put_bool(ap_map_writer *w, const char *key, bool val);
void ap_map_end(ap_map_writer *w);

void ap_put_reply_map_begin(ap_buf *b, ap_map_writer *w); /* tag + ap_map_begin */
void ap_put_reply_arr_begin(ap_buf *b, uint32_t row_count); /* tag + u32 count */

/* Decoded MAP: borrowed key/value slices into the payload buffer. */
typedef struct {
    uint32_t n;
    struct {
        const char *key;
        size_t key_len;
        const char *val;
        size_t val_len;
    } e[AP_MAP_MAX_ENTRIES];
} ap_map;

bool ap_get_reply_tag(ap_reader *r, uint32_t *tag);
bool ap_get_reply_err(ap_reader *r, char *dst, size_t dstsz);
bool ap_get_reply_int(ap_reader *r, uint32_t *out);
bool ap_get_reply_str(ap_reader *r, const char **out, size_t *out_len);
bool ap_get_map(ap_reader *r, ap_map *out);
bool ap_get_arr(ap_reader *r, uint32_t *row_count); /* then loop ap_get_map */

const char *ap_map_find(const ap_map *m, const char *key, size_t *val_len);
bool     ap_map_get_str(const ap_map *m, const char *key, char *dst, size_t dstsz);
uint32_t ap_map_get_u32(const ap_map *m, const char *key, uint32_t dflt);
long long ap_map_get_i64(const ap_map *m, const char *key, long long dflt);
bool     ap_map_get_bool(const ap_map *m, const char *key, bool dflt);

/* ========================================================================= */
/* Typed messages                                                             */
/* ========================================================================= */

/* HELLO (client -> server). */
typedef struct {
    uint32_t proto_version;
    char client_version[AP_CAP_VERSION];
    char cwd[AP_CAP_PATH];
} ap_hello;

void ap_encode_hello(ap_buf *body, const ap_hello *h);
bool ap_decode_hello(ap_reader *r, ap_hello *out);

/* HELLO reply (server -> client): travels as a MAP. */
typedef struct {
    uint32_t proto_version;
    bool engine_is_glm;
    bool has_vision;
    uint32_t ctx_size_cli;
    char backend_name[AP_CAP_NAME];
    uint32_t power_percent;
    uint32_t mtp_draft_tokens;
    char model_name[AP_CAP_MODEL];
    uint32_t vocab_size;
    bool session_parked;
} ap_hello_reply;

void ap_encode_hello_reply(ap_buf *body, const ap_hello_reply *h);
/* false when the reply was ERR (message copied to err) or malformed. */
bool ap_decode_hello_reply(ap_reader *r, ap_hello_reply *out, char *err, size_t errlen);

/* SESSION new args. seed is u32 on the wire (the 64-bit engine seed is
 * truncated, matching the plan). Sampler knobs are milli-units. */
typedef struct {
    const char *sys_text;
    size_t sys_text_len;
    const char *prefix_file_text;
    size_t prefix_file_text_len;
    uint32_t temperature;
    bool temperature_set;
    uint32_t top_p;
    bool top_p_set;
    uint32_t min_p;
    bool min_p_set;
    uint32_t seed;
    uint32_t think_mode;
    uint32_t n_predict;
    bool raw_prompt;
    bool hints_enabled;
    uint32_t dir_steering_ffn;
    bool power_set;
    uint32_t power;
} ap_session_new_args;

typedef struct {
    char sha_prefix[AP_CAP_SHA];
    uint32_t history_turns;
} ap_session_switch_args;

typedef struct {
    char sha_prefix[AP_CAP_SHA];
    bool strip;
} ap_session_del_args;

typedef struct {
    uint32_t subcmd;
    bool subcmd_known;
    ap_session_new_args new_args;      /* valid when subcmd == AGENT_SESSION_NEW */
    ap_session_switch_args switch_args; /* valid when subcmd == AGENT_SESSION_SWITCH */
    ap_session_del_args del_args;      /* valid when subcmd == AGENT_SESSION_DEL */
} ap_session_msg;

void ap_encode_session_new(ap_buf *body, const ap_session_new_args *a);
void ap_encode_session_switch(ap_buf *body, const ap_session_switch_args *a);
void ap_encode_session_del(ap_buf *body, const ap_session_del_args *a);
void ap_encode_session_simple(ap_buf *body, uint32_t subcmd); /* resume/save/list/compact */
bool ap_decode_session(ap_reader *r, ap_session_msg *out);
bool ap_session_subcmd_valid(uint32_t subcmd);

/* SESSION new / resume reply: travels as a MAP. ctx_size is authoritative. */
typedef struct {
    uint32_t ctx_used;
    uint32_t ctx_size;
    bool distributed_route_ready;
    uint32_t state;
    char note[AP_CAP_NOTE];
} ap_session_ready;

void ap_encode_session_ready(ap_buf *body, const ap_session_ready *s);
bool ap_decode_session_ready(ap_reader *r, ap_session_ready *out, char *err, size_t errlen);

/* CONFIG (client -> server). */
typedef struct {
    uint32_t op;  /* enum agent_config_op */
    uint32_t key; /* enum agent_config_key */
    bool known;   /* op and key both in range */
    uint32_t u32val; /* on set of power/steer */
    bool boolval;    /* on set of hints */
} ap_config_msg;

void ap_encode_config_get(ap_buf *body, uint32_t key);
void ap_encode_config_set_u32(ap_buf *body, uint32_t key, uint32_t val);
void ap_encode_config_set_bool(ap_buf *body, uint32_t key, bool val);
bool ap_decode_config(ap_reader *r, ap_config_msg *out);

/* A wire image: borrowed bytes + borrowed source path/hint. */
typedef struct {
    const uint8_t *bytes;
    size_t bytes_len;
    const char *source;
    size_t source_len;
} ap_wire_image;

/* TURN (client -> server). */
typedef struct {
    const char *text;
    size_t text_len;
    ap_wire_image images[AP_MAX_IMAGES];
    uint32_t image_count;
} ap_turn;

void ap_encode_turn(ap_buf *body, const ap_turn *t);
bool ap_decode_turn(ap_reader *r, ap_turn *out);

/* STREAM (server -> client). */
typedef struct {
    uint32_t stream_id;
    uint32_t kind; /* enum agent_stream_kind */
    const char *text;
    size_t text_len;
} ap_stream;

void ap_encode_stream(ap_buf *body, const ap_stream *s);
bool ap_decode_stream(ap_reader *r, ap_stream *out);

/* STATUS (server -> client): mirror of agent_status. tps fields centi-units. */
typedef struct {
    uint32_t state; /* enum agent_state */
    uint32_t prefill_done;
    uint32_t prefill_total;
    uint32_t prefill_label;
    uint32_t prefill_tps;
    uint32_t generated;
    uint32_t gen_tps;
    bool greedy_sampling;
    uint32_t ctx_used;
    uint32_t ctx_size;
    uint32_t power_percent;
    char error[AP_CAP_ERROR];
} ap_status;

void ap_encode_status(ap_buf *body, const ap_status *s);
bool ap_decode_status(ap_reader *r, ap_status *out);

/* TOOL_CALLS (server -> client). */
typedef struct {
    const char *name;
    size_t name_len;
    const char *value;
    size_t value_len;
    bool is_string;
} ap_tool_arg;

typedef struct {
    const char *name;
    size_t name_len;
    ap_tool_arg args[AP_MAX_TOOL_ARGS];
    uint32_t arg_count;
} ap_tool_call;

typedef struct {
    uint32_t request_id;
    ap_tool_call calls[AP_MAX_TOOL_CALLS];
    uint32_t call_count;
} ap_tool_calls;

void ap_encode_tool_calls(ap_buf *body, const ap_tool_calls *t);
bool ap_decode_tool_calls(ap_reader *r, ap_tool_calls *out);

/* TOOL_RESULT (client -> server). */
typedef struct {
    uint32_t request_id;
    struct {
        const char *ptr;
        size_t len;
    } text_parts[AP_MAX_TEXT_PARTS];
    uint32_t text_part_count;
    ap_wire_image images[AP_MAX_IMAGES];
    uint32_t image_count;
} ap_tool_result;

void ap_encode_tool_result(ap_buf *body, const ap_tool_result *t);
bool ap_decode_tool_result(ap_reader *r, ap_tool_result *out);

/* DRAIN_REPLY (client -> server). */
typedef struct {
    const char *text;
    size_t text_len;
} ap_drain_reply;

void ap_encode_drain_reply(ap_buf *body, const ap_drain_reply *d);
bool ap_decode_drain_reply(ap_reader *r, ap_drain_reply *out);

#endif /* DS4_AGENT_PROTO_H */
