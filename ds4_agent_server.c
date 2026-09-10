/* ds4-agent-server: the inference half of the ds4-agent client/server split.
 *
 * Loads the model, owns the engine / session / transcript, and serves one
 * ds4-agent-client at a time over the framed TCP protocol in ds4_agent_proto.h.
 * ds4_agent.c (the monolith) is untouched; code here is copied from it, minus
 * `static`, with the pure helpers taken from ds4_agent_utils.h.
 *
 * T3 scope (AGENT-SPLIT-PLAN.md section 5): CLI + --help, engine bootstrap,
 * the TCP listener with a single active connection, the HELLO handshake, and
 * the {none, active, parked} session state machine with SESSION new / resume
 * and park-on-disconnect. The worker, streaming parser, turn loop, compaction
 * and session persistence arrive in T4-T7; the stubs here reply ERR.
 */

#include "ds4.h"
#include "ds4_ssd.h"
#include "ds4_distributed.h"
#include "ds4_tp.h"
#include "ds4_help.h"
#include "ds4_gpu_args.h"
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
#include <sys/stat.h>
#include <unistd.h>

/* ========================================================================= */
/* Configuration                                                              */
/* ========================================================================= */

/* Generation knobs. On the server these are NOT parsed from the CLI: the
 * client sends them in SESSION new. The fields are kept so the worker (T4) can
 * be copied from the monolith with minimal edits, and are rewritten on every
 * SESSION new. */
typedef struct {
    int n_predict;
    int ctx_size; /* the one exception: set by --ctx, server-only */
    float temperature;
    float top_p;
    float min_p;
    bool temperature_set;
    bool top_p_set;
    bool min_p_set;
    uint64_t seed;
    ds4_think_mode think_mode;
} server_generation_options;

typedef struct {
    ds4_engine_options engine;
    server_generation_options gen;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    const char *trace_path;
    char host[256];
    int port;
} server_config;

/* ========================================================================= */
/* CLI                                                                        */
/* ========================================================================= */

static const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-agent-server: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
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

static void usage(FILE *fp, const char *topic) {
    ds4_help_print(fp, DS4_HELP_AGENT_SERVER, topic);
}

static void set_host(server_config *c, const char *h, size_t n) {
    if (n >= sizeof(c->host)) {
        fprintf(stderr, "ds4-agent-server: host name too long\n");
        exit(2);
    }
    memcpy(c->host, h, n);
    c->host[n] = '\0';
}

/* Parse a "host:port", "host", or ":port" endpoint into cfg->host / cfg->port. */
static void parse_endpoint(server_config *c, const char *arg) {
    const char *colon = strrchr(arg, ':');
    /* Reject a bracketless IPv6 literal (multiple colons, no brackets). */
    if (colon && strchr(arg, ':') != colon && arg[0] != '[') {
        fprintf(stderr, "ds4-agent-server: --server address must be host:port "
                        "(bracket IPv6 literals)\n");
        exit(2);
    }
    if (!colon) {
        set_host(c, arg, strlen(arg));
        return;
    }
    if (colon != arg) set_host(c, arg, (size_t)(colon - arg));
    if (colon[1]) c->port = parse_int(colon + 1, "--server port");
}

static server_config server_parse_options(int argc, char **argv) {
    server_config c = {
        .engine = {
            .model_path = "ds4flash.gguf",
            .backend = default_backend(),
            .mtp_draft_tokens = 1,
            .mtp_margin = 3.0f,
        },
        .gen = {
            .n_predict = 50000,
            .ctx_size = 100000,
            .temperature = DS4_DEFAULT_TEMPERATURE,
            .top_p = DS4_DEFAULT_TOP_P,
            .min_p = DS4_DEFAULT_MIN_P,
            .think_mode = DS4_THINK_HIGH,
        },
        .port = AGENT_PROTO_DEFAULT_PORT,
    };
    set_host(&c, AGENT_PROTO_DEFAULT_HOST, strlen(AGENT_PROTO_DEFAULT_HOST));

    bool steering_scale_set = false;
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "-h") || !strcmp(arg, "--help")) {
            const char *topic = (i + 1 < argc && argv[i + 1][0] != '-') ?
                argv[i + 1] : NULL;
            usage(stdout, topic);
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

        if (!strcmp(arg, "--host")) {
            const char *h = need_arg(&i, argc, argv, arg);
            set_host(&c, h, strlen(h));
        } else if (!strcmp(arg, "--port")) {
            c.port = parse_int(need_arg(&i, argc, argv, arg), arg);
        } else if (!strcmp(arg, "--server")) {
            parse_endpoint(&c, need_arg(&i, argc, argv, arg));
        } else if (!strcmp(arg, "--trace")) {
            c.trace_path = need_arg(&i, argc, argv, arg);
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
                fprintf(stderr, "ds4-agent-server: --ssd-streaming-cache-experts must be "
                                "a positive count or <number>GB\n");
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
                fprintf(stderr, "ds4-agent-server: --simulate-used-memory must be a "
                                "positive GiB value, e.g. 64GB\n");
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
            c.engine.directional_steering_ffn =
                parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else if (!strcmp(arg, "--dir-steering-attn")) {
            c.engine.directional_steering_attn =
                parse_float_range(need_arg(&i, argc, argv, arg), arg, -100.0f, 100.0f);
            steering_scale_set = true;
        } else {
            fprintf(stderr, "ds4-agent-server: unknown option: %s\n", arg);
            usage(stderr, NULL);
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
        fprintf(stderr, "ds4-agent-server: --role worker is a serving mode; "
                        "start workers with ./ds4\n");
        exit(2);
    }
    if (c.port < 1 || c.port > 65535) {
        fprintf(stderr, "ds4-agent-server: --port must be between 1 and 65535\n");
        exit(2);
    }
    return c;
}

/* ========================================================================= */
/* Engine bootstrap (copy of the monolith main body, without --chdir)         */
/* ========================================================================= */

static int server_open_engine(server_config *cfg, ds4_engine **engine_out,
                              ds4_tp **tp_out) {
    *engine_out = NULL;
    *tp_out = NULL;

    cfg->engine.context_size = cfg->gen.ctx_size;
    cfg->engine.placement_ctx_hint = cfg->gen.ctx_size;
    if (cfg->gpu_vram_arg || cfg->gpu_devices_arg) {
        cfg->engine.backend = cfg->gpu_vram_arg && !strcmp(cfg->gpu_vram_arg, "0")
            ? DS4_BACKEND_CPU
            : DS4_BACKEND_CUDA;
    }

    ds4_engine *engine = NULL;
    if (cfg->gpu_vram_arg || cfg->gpu_devices_arg) {
        ds4_gpu_config gpu_cfg = {0};
        bool skip_cuda = false;
        char gpu_err[256];
        if (parse_gpu_vram_arg(cfg->gpu_vram_arg, cfg->gpu_devices_arg,
                               &gpu_cfg, &skip_cuda, gpu_err, sizeof(gpu_err)) != 0) {
            fprintf(stderr, "ds4-agent-server: %s\n", gpu_err);
            return 2;
        }
        if (skip_cuda) {
            cfg->engine.backend = DS4_BACKEND_CPU;
            if (ds4_engine_open(&engine, &cfg->engine) != 0) return 1;
        } else {
            const bool was_auto =
                (cfg->gpu_vram_arg && !strcmp(cfg->gpu_vram_arg, "auto")) ||
                (!cfg->gpu_vram_arg && cfg->gpu_devices_arg);
            char layout[256];
            if (format_gpu_layout_line(&gpu_cfg, was_auto, layout, sizeof(layout)) > 0) {
                fprintf(stdout, "%s\n", layout);
                fflush(stdout);
            }
            cfg->engine.backend = DS4_BACKEND_CUDA;
            if (ds4_engine_create_with_gpu_config(&engine, &cfg->engine, &gpu_cfg) != 0)
                return 1;
        }
    } else if (ds4_engine_open(&engine, &cfg->engine) != 0) {
        return 1;
    }

    ds4_tp *tp_leader = NULL;
    if (cfg->engine.tp.role == DS4_TP_LEADER) {
        char tp_err[256] = "";
        ds4_tp_identity tp_id = {
            .gguf_bytes = ds4_engine_model_bytes(engine),
            .model_id = (uint32_t)ds4_engine_model_id(engine),
            .n_layer = (uint32_t)ds4_engine_layer_count(engine),
            .n_embd = (uint32_t)ds4_engine_embd_dim(engine),
            .n_vocab = (uint32_t)ds4_engine_vocab_size(engine),
            .quant_bits = (uint32_t)ds4_engine_routed_quant_bits(engine),
            .ctx_size = (uint32_t)cfg->gen.ctx_size,
        };
        ds4_engine_tp_gate_schedule(engine, &tp_id.gate_slot_start,
                                    &tp_id.gate_slot_step, &tp_id.gates_per_token,
                                    tp_id.gate_slot_mask);
        if (!ds4_tp_create(&tp_leader, &cfg->engine.tp, &tp_id, tp_err, sizeof(tp_err)) ||
            !ds4_engine_tp_bind(engine, tp_leader, tp_err, sizeof(tp_err))) {
            fprintf(stderr, "ds4-agent-server: %s\n", tp_err);
            ds4_tp_free(tp_leader);
            ds4_engine_close(engine);
            return 1;
        }
    }

    *engine_out = engine;
    *tp_out = tp_leader;
    return 0;
}

/* ========================================================================= */
/* Capabilities (HELLO reply)                                                 */
/* ========================================================================= */

/* engine may be NULL (unit tests): the engine-derived fields stay zeroed. */
static void server_fill_hello_reply(ap_hello_reply *r, ds4_engine *engine,
                                    const server_config *cfg, bool parked) {
    memset(r, 0, sizeof(*r));
    r->proto_version = AGENT_PROTO_VERSION;
    r->session_parked = parked;
    if (cfg) {
        r->ctx_size_cli = (uint32_t)cfg->gen.ctx_size;
        r->power_percent = cfg->engine.power_percent > 0
            ? (uint32_t)cfg->engine.power_percent : 100u;
        const char *bn = ds4_backend_name(cfg->engine.backend);
        snprintf(r->backend_name, sizeof(r->backend_name), "%s", bn ? bn : "");
    } else {
        r->power_percent = 100u;
    }
    if (engine) {
        r->engine_is_glm = ds4_engine_is_glm_dsa(engine);
        r->has_vision = ds4_engine_has_vision(engine);
        r->vocab_size = (uint32_t)ds4_engine_vocab_size(engine);
        r->mtp_draft_tokens = (uint32_t)ds4_engine_mtp_draft_tokens(engine);
        const char *mn = ds4_engine_model_name(engine);
        snprintf(r->model_name, sizeof(r->model_name), "%s", mn ? mn : "");
    }
}

/* ========================================================================= */
/* Session state machine: {none} -> {active} <-> {parked}                     */
/* ========================================================================= */

enum server_session_state {
    SERVER_SESSION_NONE = 0,
    SERVER_SESSION_ACTIVE,
    SERVER_SESSION_PARKED,
};

typedef struct {
    enum server_session_state state;
    bool turn_in_progress; /* set by the worker (T6) once a turn is running */
    ap_session_ready ready; /* snapshot returned by SESSION new / resume */
    /* T4: ds4_session *session, ds4_tokens transcript, image span array,
     * cache_dir, sysprompt_path, session_sha, ... */
} server_session;

static void server_session_clear(server_session *s) {
    /* T4: ds4_session_free, free transcript / images / metadata here. */
    memset(s, 0, sizeof(*s));
}

/* Any state -> ACTIVE. A parked session is discarded. */
static void server_session_begin_new(server_session *s, int ctx_size) {
    server_session_clear(s);
    s->state = SERVER_SESSION_ACTIVE;
    s->ready.ctx_used = 0;
    s->ready.ctx_size = (uint32_t)(ctx_size > 0 ? ctx_size : 0);
    s->ready.distributed_route_ready = true;
    s->ready.state = AGENT_STATE_IDLE;
    snprintf(s->ready.note, sizeof(s->ready.note), "session created");
}

/* PARKED -> ACTIVE. Returns false (no transition) from any other state. */
static bool server_session_resume(server_session *s) {
    if (s->state != SERVER_SESSION_PARKED) return false;
    s->state = SERVER_SESSION_ACTIVE;
    if (s->turn_in_progress) {
        /* A turn was cut off by the disconnect: the worker (T6) appends the
         * assistant-turn-end marker so the transcript is left at a well-formed
         * IDLE boundary. */
        s->turn_in_progress = false;
    }
    s->ready.state = AGENT_STATE_IDLE;
    snprintf(s->ready.note, sizeof(s->ready.note), "resumed parked session");
    return true;
}

/* ACTIVE -> PARKED, on socket loss. The engine / session / transcript stay in
 * RAM; only the socket is gone. */
static void server_session_park(server_session *s) {
    if (s->state != SERVER_SESSION_ACTIVE) return;
    s->state = SERVER_SESSION_PARKED;
    /* turn_in_progress is left set; server_session_resume cleans it up. */
}

/* Any state -> NONE, on SIGINT / shutdown / a clean STOP. */
static void server_session_shutdown(server_session *s) {
    server_session_clear(s);
}

/* ========================================================================= */
/* Framed connection IO (synchronous; the reader/writer thread split is T6)   */
/* ========================================================================= */

typedef struct {
    int fd;
    ap_buf rx;       /* received bytes awaiting a full frame */
    ap_buf scratch;  /* frame assembly for sends */
    FILE *trace;
} server_conn;

static void conn_init(server_conn *co, int fd, FILE *trace) {
    co->fd = fd;
    ap_buf_init(&co->rx);
    ap_buf_init(&co->scratch);
    co->trace = trace;
}

static void conn_free(server_conn *co) {
    ap_buf_free(&co->rx);
    ap_buf_free(&co->scratch);
    if (co->fd >= 0) close(co->fd);
    co->fd = -1;
}

static void conn_trace(server_conn *co, const char *dir, uint32_t type, size_t len) {
    if (!co->trace) return;
    fprintf(co->trace, "%s %s len=%zu\n", dir, ap_msg_name(type), len);
    fflush(co->trace);
}

static bool conn_send(server_conn *co, uint32_t type, const ap_buf *body) {
    ap_buf_reset(&co->scratch);
    if (!ap_frame_encode(&co->scratch, type, body)) return false;
    write_all(co->fd, (const char *)co->scratch.data, co->scratch.len);
    conn_trace(co, "->", type, body ? body->len : 0);
    return true;
}

static bool conn_send_reply_err(server_conn *co, uint32_t type, const char *msg) {
    ap_buf b;
    ap_buf_init(&b);
    ap_put_reply_err(&b, msg);
    bool ok = conn_send(co, type, &b);
    ap_buf_free(&b);
    return ok;
}

typedef enum { CONN_FRAME_OK, CONN_FRAME_CLOSED, CONN_FRAME_ERROR } conn_frame_result;

/* Block until one frame is available. payload_out is filled with the body. */
static conn_frame_result conn_recv(server_conn *co, uint32_t *type, ap_buf *payload_out,
                                   char *err, size_t errlen) {
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
/* Connection handler                                                         */
/* ========================================================================= */

/* Outcome of serving one connection, so run_server knows how to treat the
 * session afterwards. */
typedef enum {
    SERVE_DISCONNECT, /* socket lost / protocol error: park an active session */
    SERVE_STOP,       /* client sent STOP: free the session, keep listening */
} serve_result;

static serve_result serve_connection(server_conn *co, ds4_engine *engine,
                                     const server_config *cfg, server_session *sess,
                                     bool busy) {
    char err[256] = {0};
    uint32_t type = 0;
    ap_buf payload;
    ap_buf_init(&payload);
    serve_result result = SERVE_DISCONNECT;

    /* 1. HELLO handshake. */
    conn_frame_result fr = conn_recv(co, &type, &payload, err, sizeof(err));
    if (fr != CONN_FRAME_OK) goto done;
    if (type != AGENT_MSG_HELLO) {
        conn_send_reply_err(co, AGENT_MSG_HELLO, "expected HELLO first");
        goto done;
    }
    {
        ap_reader r;
        ap_reader_init(&r, payload.data, payload.len);
        ap_hello hello;
        if (!ap_decode_hello(&r, &hello)) {
            conn_send_reply_err(co, AGENT_MSG_HELLO, "malformed HELLO");
            goto done;
        }
        if (hello.proto_version != AGENT_PROTO_VERSION) {
            char m[128];
            snprintf(m, sizeof(m),
                     "proto_version mismatch (server %u, client %u): rebuild both binaries",
                     (unsigned)AGENT_PROTO_VERSION, (unsigned)hello.proto_version);
            conn_send_reply_err(co, AGENT_MSG_HELLO, m);
            goto done;
        }
        if (busy) {
            conn_send_reply_err(co, AGENT_MSG_HELLO, "busy");
            goto done;
        }
        ap_hello_reply reply;
        server_fill_hello_reply(&reply, engine, cfg,
                                sess->state == SERVER_SESSION_PARKED);
        ap_buf body;
        ap_buf_init(&body);
        ap_encode_hello_reply(&body, &reply);
        conn_send(co, AGENT_MSG_HELLO, &body);
        ap_buf_free(&body);
    }

    /* 2. Request loop. T4-T7 fill in TURN / CONFIG / TOOL_RESULT / the rest of
     * the SESSION sub-commands; for now anything unimplemented replies ERR. */
    for (;;) {
        fr = conn_recv(co, &type, &payload, err, sizeof(err));
        if (fr == CONN_FRAME_CLOSED) { result = SERVE_DISCONNECT; goto done; }
        if (fr == CONN_FRAME_ERROR) {
            if (co->trace) { fprintf(co->trace, "!! %s\n", err); fflush(co->trace); }
            result = SERVE_DISCONNECT;
            goto done;
        }

        if (type == AGENT_MSG_STOP) { result = SERVE_STOP; goto done; }

        if (type == AGENT_MSG_SESSION) {
            ap_reader r;
            ap_reader_init(&r, payload.data, payload.len);
            ap_session_msg m;
            if (!ap_decode_session(&r, &m)) {
                conn_send_reply_err(co, AGENT_MSG_SESSION, "malformed SESSION");
                result = SERVE_DISCONNECT;
                goto done;
            }
            if (!m.subcmd_known) {
                conn_send_reply_err(co, AGENT_MSG_SESSION, "unknown SESSION sub-command");
                continue;
            }
            switch (m.subcmd) {
                case AGENT_SESSION_NEW: {
                    /* T4 builds the real ds4_session, sysprompt and identity;
                     * here only the state machine moves. */
                    server_session_begin_new(sess, cfg->gen.ctx_size);
                    ap_buf body;
                    ap_buf_init(&body);
                    ap_encode_session_ready(&body, &sess->ready);
                    conn_send(co, AGENT_MSG_SESSION, &body);
                    ap_buf_free(&body);
                    break;
                }
                case AGENT_SESSION_RESUME: {
                    if (!server_session_resume(sess)) {
                        conn_send_reply_err(co, AGENT_MSG_SESSION, "no parked session to resume");
                        break;
                    }
                    ap_buf body;
                    ap_buf_init(&body);
                    ap_encode_session_ready(&body, &sess->ready);
                    conn_send(co, AGENT_MSG_SESSION, &body);
                    ap_buf_free(&body);
                    break;
                }
                default:
                    conn_send_reply_err(co, AGENT_MSG_SESSION, "not implemented yet");
                    break;
            }
            continue;
        }

        if (co->trace) {
            fprintf(co->trace, "?? unhandled %s (T4-T7)\n", ap_msg_name(type));
            fflush(co->trace);
        }
    }

done:
    ap_buf_free(&payload);
    return result;
}

/* ========================================================================= */
/* Listener                                                                   */
/* ========================================================================= */

static volatile sig_atomic_t g_shutdown = 0;
static int g_listen_fd = -1;

static void server_sigint_handler(int sig) {
    (void)sig;
    g_shutdown = 1;
    if (g_listen_fd >= 0) close(g_listen_fd);
    g_listen_fd = -1;
}

static int server_listen(const char *host, int port) {
    char portstr[16];
    snprintf(portstr, sizeof(portstr), "%d", port);

    struct addrinfo hints;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_flags = AI_PASSIVE | AI_NUMERICSERV;

    struct addrinfo *res = NULL;
    int gai = getaddrinfo(host, portstr, &hints, &res);
    if (gai != 0) {
        fprintf(stderr, "ds4-agent-server: cannot resolve %s:%d: %s\n",
                host, port, gai_strerror(gai));
        return -1;
    }

    int fd = -1;
    for (struct addrinfo *ai = res; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd < 0) continue;
        int one = 1;
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
        if (bind(fd, ai->ai_addr, ai->ai_addrlen) == 0 && listen(fd, 1) == 0) break;
        close(fd);
        fd = -1;
    }
    freeaddrinfo(res);

    if (fd < 0) {
        fprintf(stderr, "ds4-agent-server: cannot bind %s:%d: %s\n",
                host, port, strerror(errno));
        return -1;
    }
    return fd;
}

static int run_server(ds4_engine *engine, const server_config *cfg) {
    FILE *trace = NULL;
    if (cfg->trace_path) {
        trace = fopen(cfg->trace_path, "w");
        if (!trace)
            fprintf(stderr, "ds4-agent-server: cannot open trace %s: %s\n",
                    cfg->trace_path, strerror(errno));
    }

    g_listen_fd = server_listen(cfg->host, cfg->port);
    if (g_listen_fd < 0) {
        if (trace) fclose(trace);
        return 1;
    }
    fprintf(stderr, "ds4-agent-server: listening on %s:%d\n", cfg->host, cfg->port);

    server_session sess;
    server_session_clear(&sess);
    bool serving = false;
    int rc = 0;

    while (!g_shutdown) {
        int cfd = accept(g_listen_fd, NULL, NULL);
        if (cfd < 0) {
            if (errno == EINTR) continue;
            if (g_shutdown) break;
            fprintf(stderr, "ds4-agent-server: accept: %s\n", strerror(errno));
            rc = 1;
            break;
        }

        int one = 1;
        setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));

        server_conn co;
        conn_init(&co, cfd, trace);
        /* A second connection while one is active is told "busy" and dropped;
         * the first keeps its session. */
        serve_result sr = serve_connection(&co, engine, cfg, &sess, serving);
        conn_free(&co);

        if (serving) continue; /* the rejected extra connection: nothing changed */

        if (sr == SERVE_STOP) {
            server_session_shutdown(&sess);
        } else {
            server_session_park(&sess);
        }
    }

    server_session_shutdown(&sess);
    if (g_listen_fd >= 0) { close(g_listen_fd); g_listen_fd = -1; }
    if (trace) fclose(trace);
    return rc;
}

/* ========================================================================= */
/* main                                                                       */
/* ========================================================================= */

#ifndef DS4_AGENT_SERVER_TEST_NO_MAIN
int main(int argc, char **argv) {
    server_config cfg = server_parse_options(argc, argv);

    ds4_engine *engine = NULL;
    ds4_tp *tp_leader = NULL;
    int rc = server_open_engine(&cfg, &engine, &tp_leader);
    if (rc != 0) return rc;

    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sigemptyset(&sa.sa_mask);
    sa.sa_handler = server_sigint_handler;
    struct sigaction old_int;
    bool sigint_installed = sigaction(SIGINT, &sa, &old_int) == 0;
    signal(SIGPIPE, SIG_IGN);

    rc = run_server(engine, &cfg);

    if (sigint_installed) sigaction(SIGINT, &old_int, NULL);
    if (tp_leader) ds4_tp_send_stop(tp_leader);
    ds4_engine_close(engine);
    ds4_tp_free(tp_leader);
    return rc;
}
#endif
