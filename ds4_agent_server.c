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
#include "ds4_kvstore.h"
#include "ds4_prompt_prefix.h"
#include "ds4_tool_text.h"
#include "ds4_agent_proto.h"
#include "ds4_agent_utils.h"

#include <arpa/inet.h>
#include <ctype.h>
#include <errno.h>
#include <limits.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <time.h>
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
    /* Populated from SESSION new, rewritten on every /new. The client sends the
     * -sys text and the --prefix-file contents; the server builds all the
     * system tokens from them. */
    char *system;             /* owned */
    ds4_prompt_prefix prefix; /* parsed from SESSION new's prefix_file_text */
} server_generation_options;

typedef struct {
    ds4_engine_options engine;
    server_generation_options gen;
    const char *gpu_vram_arg;
    const char *gpu_devices_arg;
    const char *trace_path;
    char host[256];
    int port;
    bool non_interactive; /* always false on the server; kept so the copied
                           * worker code compiles unchanged */
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
/* Worker: agent_worker, system/tool prompt, sysprompt.kv, lifecycle.         */
/*                                                                           */
/* Copied almost verbatim from ds4_agent.c (everything is static in one TU,   */
/* so no rename is needed). The client-only pieces are left out: web, bash,   */
/* the more-cursor, linenoise and raw-prompt. Per AGENT-SPLIT-PLAN.md section */
/* 3b the tools-prompt builders drop the exact/anchored edit toggle and       */
/* always emit the exact-match strings. worker_run_turn and the deferred      */
/* save / compact handlers are T5-T7 stubs; turn execution, the DSML/GLM      */
/* parser and compaction land in later tasks.                                 */
/* ========================================================================= */

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

typedef struct {
    bool enabled;
    bool instructed;
    enum {
        AGENT_HINTS_UNSET,
        AGENT_HINTS_OFF,
        AGENT_HINTS_ON,
        AGENT_HINTS_UNKNOWN,
    } applied;
} agent_hints;

typedef enum {
    AGENT_TOOL_SYNTAX_DSML,
    AGENT_TOOL_SYNTAX_GLM,
} agent_tool_syntax;

/* Trimmed copy of ds4_agent.c's agent_worker: no web, bash, more-cursor,
 * linenoise or raw-mode fields (those live on the client). */
typedef struct {
    ds4_engine *engine;
    server_config *cfg;
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
    int wake_fd[2];
    FILE *trace;
    bool wake_pending;
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
    agent_status status;
    char *out;
    size_t out_len;
    size_t out_cap;
    bool queued_user_drain_pending;
    bool queued_user_drain_answered;
    char *queued_user_drain_text;
    bool datetime_context_injected;
    agent_hints hints;
    int last_system_prompt_reminder_at;
} agent_worker;

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

static bool worker_is_idle(agent_worker *w);
static void worker_apply_pending_power(agent_worker *w);
static unsigned agent_next_prefill_label(void);
static void agent_trace(agent_worker *w, const char *fmt, ...);
static void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len);
static void agent_publish_system_status(agent_worker *w, const char *msg);
static int agent_worker_sync_tokens(agent_worker *w, const ds4_tokens *tokens,
                                    bool publish_progress,
                                    char *err, size_t err_len);
static char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes);
static void *worker_main(void *arg);

/* -- syntax helpers, effective ctx / think mode ------------------------- */

static agent_tool_syntax agent_tool_syntax_for_engine(ds4_engine *engine) {
    return ds4_engine_is_glm_dsa(engine) ? AGENT_TOOL_SYNTAX_GLM
                                         : AGENT_TOOL_SYNTAX_DSML;
}

static bool agent_tool_syntax_assistant_turn_uses_eos(agent_tool_syntax syntax) {
    return syntax != AGENT_TOOL_SYNTAX_GLM;
}

static void agent_worker_append_assistant_turn_end(agent_worker *w) {
    if (agent_tool_syntax_assistant_turn_uses_eos(
            agent_tool_syntax_for_engine(w->engine)))
        ds4_tokens_push(&w->transcript, ds4_token_eos(w->engine));
}

static int agent_worker_effective_ctx_size(const agent_worker *w) {
    int ctx = 0;
    if (w && w->session) ctx = ds4_session_ctx(w->session);
    if (ctx <= 0 && w && w->cfg) ctx = w->cfg->gen.ctx_size;
    return ctx;
}

static ds4_think_mode effective_think_mode(const server_config *cfg) {
    return ds4_think_mode_for_context(cfg->gen.think_mode, cfg->gen.ctx_size);
}

/* -- vision span helpers --------------------------------------------- */

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

/* -- system / tool prompt (exact-match edit only, plan 3b) --------- */

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
    "<｜DSML｜tool_calls>\n"
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n"
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n"
    "</｜DSML｜invoke>\n"
    "</｜DSML｜tool_calls>\n\n"
    "Tool calls are not allowed inside <think></think>; finish thinking before emitting DSML.\n\n"
    "String parameters use raw text and string=\"true\". Numbers and booleans use JSON text and string=\"false\".\n\n"
    "Inside string values only, escape a literal closing parameter tag as &lt;/｜DSML｜parameter>. "
    "To write that escaped spelling literally, use &amp;lt;/｜DSML｜parameter>. Other HTML entities are unchanged.\n\n"
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
    "}\n"
    "\n"
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

static void agent_worker_note_system_prompt_seen(agent_worker *w) {
    w->last_system_prompt_reminder_at = w->transcript.len;
}

/* -- worker output queue + status + trace -------------------------- */

static void agent_wake_locked(agent_worker *w) {
    if (w->wake_pending) return;
    w->wake_pending = true;
    char c = 'x';
    ssize_t wr = write(w->wake_fd[1], &c, 1);
    (void)wr;
}

/* Queue rendered output for the UI thread.  The worker never writes directly
 * to the terminal, which keeps linenoise redraws serialized in one place. */
static void agent_publish(agent_worker *w, const char *s, size_t n) {
    if (!n) return;
    pthread_mutex_lock(&w->mu);
    if (w->out_len + n + 1 > w->out_cap) {
        size_t cap = w->out_cap ? w->out_cap * 2 : 4096;
        while (cap < w->out_len + n + 1) cap *= 2;
        char *p = realloc(w->out, cap);
        if (!p) {
            pthread_mutex_unlock(&w->mu);
            return;
        }
        w->out = p;
        w->out_cap = cap;
    }
    memcpy(w->out + w->out_len, s, n);
    w->out_len += n;
    w->out[w->out_len] = '\0';
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void agent_publishf(agent_worker *w, const char *fmt, ...) {
    char stack[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(stack, sizeof(stack), fmt, ap);
    va_end(ap);
    if (n <= 0) return;
    if ((size_t)n < sizeof(stack)) {
        agent_publish(w, stack, (size_t)n);
        return;
    }

    char *heap = xmalloc((size_t)n + 1);
    va_start(ap, fmt);
    vsnprintf(heap, (size_t)n + 1, fmt, ap);
    va_end(ap);
    agent_publish(w, heap, (size_t)n);
    free(heap);
}

static bool worker_is_idle(agent_worker *w);

static void agent_set_status(agent_worker *w, agent_worker_state state) {
    pthread_mutex_lock(&w->mu);
    w->status.state = state;
    if (state != AGENT_WORKER_PREFILL)
        w->status.prefill_tps = 0.0;
    if (state != AGENT_WORKER_GENERATING)
        w->status.greedy_sampling = false;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void agent_set_error(agent_worker *w, const char *msg) {
    pthread_mutex_lock(&w->mu);
    w->status.state = AGENT_WORKER_ERROR;
    w->status.prefill_tps = 0.0;
    w->status.greedy_sampling = false;
    snprintf(w->status.error, sizeof(w->status.error), "%s", msg ? msg : "unknown error");
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static void agent_trace_time(FILE *fp) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    struct tm tm;
    localtime_r(&ts.tv_sec, &tm);
    fprintf(fp, "%04d-%02d-%02d %02d:%02d:%02d.%03ld",
            tm.tm_year + 1900, tm.tm_mon + 1, tm.tm_mday,
            tm.tm_hour, tm.tm_min, tm.tm_sec, ts.tv_nsec / 1000000);
}

static void agent_trace(agent_worker *w, const char *fmt, ...) {
    if (!w || !w->trace) return;
    pthread_mutex_lock(&w->mu);
    agent_trace_time(w->trace);
    fputs(" ", w->trace);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(w->trace, fmt, ap);
    va_end(ap);
    fputc('\n', w->trace);
    fflush(w->trace);
    pthread_mutex_unlock(&w->mu);
}

static void agent_trace_escaped(FILE *fp, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
        case '\\': fputs("\\\\", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        case '"': fputs("\\\"", fp); break;
        default:
            if (c < 32 || c == 127) fprintf(fp, "\\x%02x", c);
            else fputc(c, fp);
            break;
        }
    }
}

static void agent_trace_token(agent_worker *w, int token, const char *text,
                              size_t text_len, int index) {
    if (!w || !w->trace) return;
    pthread_mutex_lock(&w->mu);
    agent_trace_time(w->trace);
    fprintf(w->trace, " token index=%d id=%d bytes=%zu text=\"",
            index, token, text_len);
    agent_trace_escaped(w->trace, text ? text : "", text_len);
    fputs("\" hex=", w->trace);
    for (size_t i = 0; i < text_len; i++)
        fprintf(w->trace, "%02x", (unsigned char)text[i]);
    fputc('\n', w->trace);
    fflush(w->trace);
    pthread_mutex_unlock(&w->mu);
}

static void agent_trace_tokens(agent_worker *w, const char *label,
                               const ds4_tokens *tokens, int start) {
    if (!w || !w->trace || !tokens) return;
    if (start < 0) start = 0;
    if (start > tokens->len) start = tokens->len;
    agent_trace(w, "tokens label=%s start=%d len=%d", label ? label : "",
                start, tokens->len);
    for (int i = start; i < tokens->len; i++) {
        size_t text_len = 0;
        char *text = ds4_token_text(w->engine, tokens->v[i], &text_len);
        agent_trace_token(w, tokens->v[i], text, text_len, i);
        free(text);
    }
}

static void agent_trace_text(agent_worker *w, const char *label,
                             const char *text, size_t len) {
    if (!w || !w->trace) return;
    pthread_mutex_lock(&w->mu);
    agent_trace_time(w->trace);
    fprintf(w->trace, " %s=\"", label ? label : "text");
    agent_trace_escaped(w->trace, text ? text : "", len);
    fputs("\"\n", w->trace);
    fflush(w->trace);
    pthread_mutex_unlock(&w->mu);
}

/* -- identity SHA, cache dir, token compare ----------------------- */

static bool agent_tokens_equal(const ds4_tokens *a, const ds4_tokens *b) {
    if (!a || !b || a->len != b->len) return false;
    for (int i = 0; i < a->len; i++) {
        if (a->v[i] != b->v[i]) return false;
    }
    return true;
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

/* Agent session IDs are intentionally independent from the rendered transcript:
 * once a session has a title and creation time, resaving it keeps the same file
 * name while the transcript and KV payload evolve. */
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

/* -- KV persistence (sysprompt.kv + saved sessions) -------------- */

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
    /* text_bytes is read from the on-disk header; cap it against the bytes
     * actually left in the file before allocating, so a 1-byte file declaring
     * text_bytes = 0xFFFFFFFF can't request ~4 GiB. */
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

/* Read the optional agent title trailer without disturbing the payload cursor.
 * The caller is positioned just after rendered text, which is also the payload
 * start expected by ds4_session_load_payload(). */
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
    /* Same cap as agent_kv_read_text: reject a title length larger than the
     * bytes left in the file before allocating. */
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
        /* A saved payload contains only the leader's graph state. Rebuild from
         * rendered text under TP so session_sync mirrors the same token prefix
         * to the worker before either rank resumes decoding. */
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

/* Save the current live KV under the rendered transcript identity.  The caller
 * decides the policy: fixed sysprompt path or SHA-named session path. */
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

/* -- session title extraction ----------------------------------- */

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

/* Extract a human-readable title from the first user turn stored in the
 * rendered transcript.  max_bytes==0 means "full normalized title"; callers
 * that render to the terminal pass an explicit display budget. */
static char *agent_session_title_from_text(const char *text, size_t text_len,
                                           size_t max_bytes) {
    static const char user_mark[] = "<｜User｜>";
    static const char assistant_mark[] = "<｜Assistant｜>";
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

/* -- progress + interrupt callbacks ---------------------------- */

static void worker_progress_cb(void *ud, const char *event, int current, int total) {
    agent_worker *w = ud;
    if (!w || !event) return;
    (void)total;
    worker_apply_pending_power(w);
    pthread_mutex_lock(&w->mu);
    int done = 0;
    int progress_total = w->status.prefill_total;
    if (!strcmp(event, "prefill_work")) {
        /* GLM reports intra-chunk work relative to the suffix being synced.
         * It is display progress only: checkpoint progress still arrives as
         * absolute prefill_chunk positions after each completed chunk. */
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
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static bool worker_should_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool interrupt = w->interrupt || w->stop;
    pthread_mutex_unlock(&w->mu);
    return interrupt;
}

/* Ctrl+C is a latched request consumed by the worker.  Once an interrupted
 * operation has reached a stable append-only boundary and is about to publish
 * IDLE, the request must be acknowledged; otherwise the editor can observe an
 * idle worker with a stale interrupt still pending. */
static void worker_clear_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = false;
    pthread_mutex_unlock(&w->mu);
}

static bool agent_err_is_interrupted(const char *err) {
    return err && !strcmp(err, "interrupted");
}

static bool worker_cancel_session_cb(void *ud) {
    return worker_should_interrupt(ud);
}

/* -- system-token build, sync, reset to sysprompt ------------- */

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

static void agent_publish_system_status(agent_worker *w, const char *msg) {
    if (w->cfg->non_interactive) return;
    if (isatty(STDOUT_FILENO)) {
        static const char marker[] = "\x1b[33m✦ \x1b[38;5;218m";
        agent_publish(w, marker, sizeof(marker) - 1);
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\x1b[0m\n", strlen("\x1b[0m\n"));
    } else {
        agent_publish(w, "✦ ", strlen("✦ "));
        agent_publish(w, msg, strlen(msg));
        agent_publish(w, "\n", 1);
    }
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
        unsigned prefill_label = w->status.state == AGENT_WORKER_PREFILL ?
            w->status.prefill_label : agent_next_prefill_label();
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_base = cached;
        w->progress_direct = false;
        w->progress_started_at = now_sec();
        w->status.prefill_done = 0;
        w->status.prefill_total = suffix;
        w->status.prefill_label = prefill_label;
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        agent_wake_locked(w);
        pthread_mutex_unlock(&w->mu);
    }

    ds4_session_set_progress(w->session, publish_progress ? worker_progress_cb : NULL,
                             publish_progress ? w : NULL);
    ds4_session_set_display_progress(w->session,
                                     publish_progress ? worker_progress_cb : NULL,
                                     publish_progress ? w : NULL);
    ds4_session_set_cancel(w->session, worker_cancel_session_cb, w);
    double t_sync0 = now_sec();
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
    double t_sync1 = now_sec();
    ds4_session_set_cancel(w->session, NULL, NULL);
    ds4_session_set_progress(w->session, NULL, NULL);
    ds4_session_set_display_progress(w->session, NULL, NULL);
    agent_trace(w, "prefill sync done prompt=%d cached=%d suffix=%d rc=%d %.3f ms",
                tokens->len, cached, suffix, rc, (t_sync1 - t_sync0) * 1000.0);
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
        if (loaded) {
            agent_trace(w, "sysprompt kv hit file=%s tokens=%d",
                        w->sysprompt_path, w->transcript.len);
        }
    }

    if (!loaded) {
        if (w->sysprompt_path)
            agent_publish_system_status(w, "Updating system prompt cache...");
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
                if (w->cfg->non_interactive) {
                    fprintf(stderr, "ds4-agent: failed to save system prompt KV: %s\n",
                            save_err);
                } else {
                    agent_buf b = {0};
                    agent_buf_puts(&b, "\nds4-agent: failed to save system prompt KV: ");
                    agent_buf_puts(&b, save_err);
                    agent_buf_puts(&b, "\n");
                    char *msg = agent_buf_take(&b);
                    agent_publish(w, msg, strlen(msg));
                    free(msg);
                }
            } else {
                agent_trace(w, "sysprompt kv stored file=%s tokens=%d",
                            w->sysprompt_path, w->transcript.len);
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
    if (w->initialized) w->hints = (agent_hints){0};
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
    w->datetime_context_injected = false;
    agent_worker_clear_session_identity(w);
    free(text);
    ds4_tokens_free(&sys);
    return true;
}

static bool agent_worker_should_stop(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool stop = w->stop;
    pthread_mutex_unlock(&w->mu);
    return stop;
}

static bool agent_worker_wait_distributed_route(agent_worker *w, char *err, size_t err_len) {
    if (!w || !w->cfg ||
        w->cfg->engine.distributed.role != DS4_DISTRIBUTED_COORDINATOR)
        return true;

    char last[160] = {0};
    unsigned ticks = 0;
    const struct timespec delay = {0, 250000000L};
    for (;;) {
        int ready = ds4_session_distributed_route_ready(w->session, err, err_len);
        if (ready > 0) {
            if (ticks != 0) {
                if (w->cfg->non_interactive)
                    fprintf(stderr, "ds4-agent: distributed route ready\n");
                else
                    agent_publish_system_status(w, "Distributed route ready.");
            }
            if (err_len) err[0] = '\0';
            return true;
        }
        if (ready < 0) return false;

        const char *why = err && err[0] ? err : "route incomplete";
        if (strcmp(last, why) != 0 || (ticks % 20u) == 0) {
            if (w->cfg->non_interactive) {
                fprintf(stderr, "ds4-agent: waiting for distributed route: %s\n", why);
            } else {
                char msg[224];
                snprintf(msg, sizeof(msg), "Waiting for distributed route: %s", why);
                agent_publish_system_status(w, msg);
            }
            snprintf(last, sizeof(last), "%s", why);
        }
        if (agent_worker_should_stop(w)) {
            snprintf(err, err_len, "agent stopped while waiting for distributed route");
            return false;
        }
        nanosleep(&delay, NULL);
        ticks++;
    }
}

/* -- worker lifecycle: submit / interrupt / stop / status ----- */

static void worker_request_power(agent_worker *w, int power) {
    pthread_mutex_lock(&w->mu);
    w->requested_power = power;
    w->power_requested = true;
    w->status.power_percent = power;
    pthread_cond_signal(&w->cond);
    agent_wake_locked(w);
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

static bool worker_take_power_requested(agent_worker *w, int *power) {
    pthread_mutex_lock(&w->mu);
    bool requested = w->power_requested;
    if (requested) {
        if (power) *power = w->requested_power;
        w->power_requested = false;
    }
    pthread_mutex_unlock(&w->mu);
    return requested;
}

static void worker_apply_pending_power(agent_worker *w) {
    int power = 0;
    if (!worker_take_power_requested(w, &power)) return;
    if (ds4_session_set_power(w->session, power) != 0) {
        agent_publishf(w, "\npower change failed\n");
        return;
    }
    pthread_mutex_lock(&w->mu);
    w->cfg->engine.power_percent = power;
    w->status.power_percent = power;
    agent_wake_locked(w);
    pthread_mutex_unlock(&w->mu);
}

static bool worker_submit(agent_worker *w, const char *text) {
    pthread_mutex_lock(&w->mu);
    bool ok = w->initialized && w->status.state == AGENT_WORKER_IDLE && !w->cmd_text;
    if (ok) {
        w->cmd_text = xstrdup(text);
        /* A submitted turn is no longer idle, even if the worker thread has
         * not yet reached its real prefill accounting.  Non-interactive mode
         * depends on this to avoid exiting in the small handoff window between
         * accepting stdin and starting generation. */
        w->status.state = AGENT_WORKER_PREFILL;
        w->progress_direct = false;
        w->status.prefill_done = 0;
        w->status.prefill_total = 0;
        w->status.prefill_label = agent_next_prefill_label();
        w->status.prefill_tps = 0.0;
        w->status.generated = 0;
        w->status.gen_tps = 0.0;
        w->status.greedy_sampling = false;
        pthread_cond_signal(&w->cond);
    }
    pthread_mutex_unlock(&w->mu);
    return ok;
}

static int worker_status_power_locked(agent_worker *w) {
    if (w->power_requested) return w->requested_power;
    int power = w->cfg->engine.power_percent;
    return power > 0 ? power : 100;
}

/* Request interruption at the next model/tool polling point. */
static void worker_interrupt(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->interrupt = true;
    if (w->cfg &&
        w->cfg->engine.distributed.role == DS4_DISTRIBUTED_COORDINATOR &&
        (w->status.state == AGENT_WORKER_PREFILL ||
         w->status.state == AGENT_WORKER_GENERATING ||
         w->status.state == AGENT_WORKER_COMPACTING))
    {
        w->status.state = AGENT_WORKER_DRAINING;
        w->status.prefill_tps = 0.0;
        w->status.greedy_sampling = false;
        agent_wake_locked(w);
    }
    pthread_mutex_unlock(&w->mu);
}

/* Stop the worker thread. */
static void worker_stop(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    w->stop = true;
    pthread_cond_signal(&w->cond);
    pthread_mutex_unlock(&w->mu);
}

/* The UI thread consumes output in batches.  Taking ownership of w->out under
 * the mutex keeps terminal writes outside the lock while preserving order. */
static void worker_consume(agent_worker *w, char **out, size_t *out_len, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    if (out) {
        *out = w->out;
        *out_len = w->out_len;
        w->out = NULL;
        w->out_len = 0;
        w->out_cap = 0;
    }
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = agent_worker_effective_ctx_size(w);
    w->status.power_percent = worker_status_power_locked(w);
    if (status) *status = w->status;
    w->wake_pending = false;
    pthread_mutex_unlock(&w->mu);
}

static void worker_get_status(agent_worker *w, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = agent_worker_effective_ctx_size(w);
    w->status.power_percent = worker_status_power_locked(w);
    *status = w->status;
    pthread_mutex_unlock(&w->mu);
}

static bool worker_is_idle(agent_worker *w) {
    pthread_mutex_lock(&w->mu);
    bool idle = w->initialized &&
        (w->status.state == AGENT_WORKER_IDLE ||
         w->status.state == AGENT_WORKER_ERROR);
    pthread_mutex_unlock(&w->mu);
    return idle;
}

static bool worker_is_initialized(agent_worker *w, agent_status *status) {
    pthread_mutex_lock(&w->mu);
    w->status.ctx_used = w->transcript.len;
    w->status.ctx_size = agent_worker_effective_ctx_size(w);
    w->status.power_percent = worker_status_power_locked(w);
    if (status) *status = w->status;
    bool initialized = w->initialized;
    pthread_mutex_unlock(&w->mu);
    return initialized;
}

/* -- T5/T6/T7 stubs: turn loop, deferred save + compaction ---- */

static void worker_run_turn(agent_worker *w, const char *text) {
    (void)text;
    /* T5 adds the DSML/GLM parser + STREAM fragment emitter; T6 the turn
     * loop with tool suspend/resume and mid-turn compaction. Until then a
     * submitted turn just returns to IDLE. */
    agent_trace(w, "worker_run_turn: turn execution not wired yet (T5/T6)");
    agent_set_status(w, AGENT_WORKER_IDLE);
}

static void worker_run_deferred_save(agent_worker *w) {
    if (!worker_take_save_requested(w)) return;
    agent_trace(w, "deferred save: session persistence lands in T7");
    agent_set_status(w, AGENT_WORKER_IDLE);
}

static void worker_run_deferred_compact(agent_worker *w) {
    if (!worker_take_compact_requested(w)) return;
    agent_trace(w, "deferred compact: compaction lands in T6");
}

/* -- compaction summary prompt (pure text; the rest is T6) ---- */

#define AGENT_COMPACT_SUMMARY_MAX_TOKENS 4096

static void agent_tokens_append_range(ds4_tokens *dst, const ds4_tokens *src,
                                      int start, int end) {
    if (start < 0) start = 0;
    if (end > src->len) end = src->len;
    for (int i = start; i < end; i++) ds4_tokens_push(dst, src->v[i]);
}

/* Build the private prompt used to ask the model for durable state.  The prompt
 * explicitly forbids tool calls because the result is consumed internally, not
 * delivered as an assistant turn. */
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

/* -- worker thread entry ------------------------------------- */

static unsigned agent_next_prefill_label(void) {
    static unsigned next;
    return next++;
}

static void *worker_main(void *arg) {
    agent_worker *w = arg;
    agent_trace(w, "agent worker start ctx=%d backend=%s model=%s trace=%s",
                agent_worker_effective_ctx_size(w),
                ds4_backend_name(w->cfg->engine.backend),
                w->cfg->engine.model_path ? w->cfg->engine.model_path : "",
                w->cfg->trace_path ? w->cfg->trace_path : "");
    char init_err[160] = {0};
    bool init_ok = agent_worker_wait_distributed_route(w, init_err, sizeof(init_err));
    if (init_ok)
        init_ok = agent_worker_reset_to_sysprompt(w, init_err, sizeof(init_err));
    if (!init_ok)
        agent_set_error(w, init_err[0] ? init_err : "failed to initialize system prompt");
    agent_trace_tokens(w, "initial_system_prompt", &w->transcript, 0);
    pthread_mutex_lock(&w->mu);
    w->initialized = true;
    agent_wake_locked(w);
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
        w->cmd_text = NULL;
        pthread_mutex_unlock(&w->mu);

        worker_run_turn(w, cmd);
        free(cmd);
        worker_apply_pending_power(w);
        worker_run_deferred_compact(w);
        worker_run_deferred_save(w);
    }

    agent_set_status(w, AGENT_WORKER_STOPPED);
    return NULL;
}

static int agent_worker_init(agent_worker *w, ds4_engine *engine,
                             server_config *cfg, FILE *trace) {
    memset(w, 0, sizeof(*w));
    w->engine = engine;
    w->cfg = cfg;
    w->trace = trace;
    w->wake_fd[0] = -1;
    w->wake_fd[1] = -1;
    pthread_mutex_init(&w->mu, NULL);
    pthread_cond_init(&w->cond, NULL);
    w->status.state = AGENT_WORKER_IDLE;
    if (pipe(w->wake_fd) != 0) return -1;
    int old_flags;
    set_nonblock(w->wake_fd[0], true, &old_flags);
    set_nonblock(w->wake_fd[1], true, &old_flags);
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

static void agent_worker_free(agent_worker *w) {
    worker_stop(w);
    if (w->thread) pthread_join(w->thread, NULL);
    ds4_session_free(w->session);
    ds4_tokens_free(&w->transcript);
    agent_worker_images_clear(w);
    free(w->cache_dir);
    free(w->sysprompt_path);
    free(w->session_title);
    free(w->legacy_session_path_to_delete);
    free(w->queued_user_drain_text);
    if (w->wake_fd[0] >= 0) close(w->wake_fd[0]);
    if (w->wake_fd[1] >= 0) close(w->wake_fd[1]);
    free(w->cmd_text);
    free(w->out);
    pthread_cond_destroy(&w->cond);
    pthread_mutex_destroy(&w->mu);
}


/* ========================================================================= */
/* Streaming classifier: DSML/GLM parser + (kind, text) fragment emitter.      */
/*                                                                            */
/* The parser cluster (agent_dsml_* / agent_glm_tool_parse / agent_tool_call*)*/
/* is copied verbatim from ds4_agent.c. The painting layer is NOT: instead of */
/* driving agent_token_renderer / agent_tool_viz_*, the control flow feeds     */
/* srv_emit(kind, bytes), coalescing adjacent same-kind fragments. The client */
/* (T9) rebuilds the terminal projection from the (kind, text) stream. See    */
/* AGENT-SPLIT-PLAN.md "Rendering split". TOOL_CALLS emission and the turn     */
/* loop are T6.                                                                */
/* ========================================================================= */

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

typedef struct {
    agent_tool_call *v;
    int len;
    int cap;
} agent_tool_calls;

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
    agent_tool_call current;
    char *param_name;
    bool param_is_string;
    size_t param_value_start;
    bool param_close_prefix;
    bool glm_after_call;
    agent_tool_calls calls;
    char error[160];
} agent_dsml_parser;

typedef struct {
    char tail[32];
    size_t len;
} agent_dsml_marker_detector;

static bool bytes_has_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n >= plen && memcmp(p, prefix, plen) == 0;
}

static bool bytes_is_partial_prefix(const char *p, size_t n, const char *prefix) {
    size_t plen = strlen(prefix);
    return n < plen && memcmp(prefix, p, n) == 0;
}

static void agent_tool_call_free(agent_tool_call *c) {
    if (!c) return;
    free(c->name);
    for (int i = 0; i < c->argc; i++) {
        free(c->args[i].name);
        free(c->args[i].value);
    }
    free(c->args);
    memset(c, 0, sizeof(*c));
}

static void agent_tool_calls_free(agent_tool_calls *calls) {
    if (!calls) return;
    for (int i = 0; i < calls->len; i++) agent_tool_call_free(&calls->v[i]);
    free(calls->v);
    memset(calls, 0, sizeof(*calls));
}

static void agent_tool_call_add_arg(agent_tool_call *c, const char *name,
                                    const char *value, size_t value_len,
                                    bool is_string, const char *end_tag) {
    if (c->argc == c->argcap) {
        c->argcap = c->argcap ? c->argcap * 2 : 4;
        c->args = xrealloc(c->args, (size_t)c->argcap * sizeof(c->args[0]));
    }
    c->args[c->argc++] = (agent_tool_arg){
        .name = xstrdup(name),
        .value = xstrndup(value, value_len),
        .is_string = is_string,
    };
    if (is_string) ds4_tool_text_unescape(c->args[c->argc - 1].value, end_tag);
}

static void agent_tool_calls_push(agent_tool_calls *calls, agent_tool_call *call) {
    if (!call->name) return;
    if (calls->len == calls->cap) {
        calls->cap = calls->cap ? calls->cap * 2 : 2;
        calls->v = xrealloc(calls->v, (size_t)calls->cap * sizeof(calls->v[0]));
    }
    calls->v[calls->len++] = *call;
    memset(call, 0, sizeof(*call));
}

static const char *agent_tool_arg_value(const agent_tool_call *call, const char *name) {
    for (int i = 0; i < call->argc; i++) {
        if (call->args[i].name && !strcmp(call->args[i].name, name))
            return call->args[i].value ? call->args[i].value : "";
    }
    return NULL;
}

static void agent_dsml_parser_free(agent_dsml_parser *p) {
    if (!p) return;
    agent_tool_syntax syntax = p->syntax;
    free(p->raw);
    agent_tool_call_free(&p->current);
    free(p->param_name);
    agent_tool_calls_free(&p->calls);
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
    snprintf(prefix, sizeof(prefix), "<｜DSML｜%s", name);
    size_t prefix_len = strlen(prefix);
    if (strncmp(tag, prefix, prefix_len) != 0) return false;
    char c = tag[prefix_len];
    return c == '>' || c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

static bool agent_dsml_close_tag_at(const char *s, const char *name, size_t *tag_len) {
    char prefix[64];
    static const char dsml_bar[] = "｜";
    snprintf(prefix, sizeof(prefix), "</｜DSML｜%s", name);
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
 * terminal rendering must hide partial close tags without waiting for the whole
 * parameter to finish. */
static bool agent_dsml_parameter_close_tail(const char *tail, size_t len,
                                            bool *complete) {
    static const char prefix[] = "</｜DSML｜parameter";
    static const char dsml_bar[] = "｜";
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
    while ((p = strstr(p, "</｜DSML｜")) != NULL) {
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
            agent_tool_call_free(&p->current);
            p->current.name = xstrndup(name_start, (size_t)(name_end - name_start));
            p->parse_pos = (size_t)(lt - raw);
            cur = raw + p->parse_pos;
        }

        if (agent_bytes_starts_with(cur, end, close)) {
            p->parse_pos += sizeof(close) - 1;
            agent_tool_calls_push(&p->calls, &p->current);
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
                                    p->param_is_string, "</｜DSML｜parameter>");
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
            agent_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            p->state = AGENT_DSML_DONE;
            return;
        }
        if (agent_dsml_close_tag_at(p->raw + p->parse_pos, "invoke", &close_len)) {
            agent_tool_calls_push(&p->calls, &p->current);
            p->parse_pos += close_len;
            continue;
        }

        char *tag_end = strchr(p->raw + p->parse_pos, '>');
        if (!tag_end) return;
        size_t tag_len = (size_t)(tag_end - (p->raw + p->parse_pos)) + 1;
        char *tag = xstrndup(p->raw + p->parse_pos, tag_len);

        if (agent_dsml_open_tag_is(tag, "invoke")) {
            agent_tool_call_free(&p->current);
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
        "<tool_call>" : "<｜DSML｜tool_calls>";
    p->state = AGENT_DSML_STRUCTURAL;
    p->search_len = 0;
    agent_dsml_raw_append(p, start, strlen(start));
    p->parse_pos = strlen(start);
}

static void agent_dsml_feed(agent_dsml_parser *p, const char *s, size_t n) {
    const char *start = p->syntax == AGENT_TOOL_SYNTAX_GLM ?
        "<tool_call>" : "<｜DSML｜tool_calls>";
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

    static const char canonical[] = "<｜DSML｜tool_calls>";
    static const char missing_bar[] = "<DSML｜tool_calls>";
    static const char invoke[] = "<｜DSML｜invoke";
    static const char invoke_missing_bar[] = "<DSML｜invoke";
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

    static const char fullwidth_marker[] = "｜DSML｜";
    static const char ascii_marker[] = "|DSML|";
    static const char missing_open[] = "<DSML｜";
    static const char missing_close[] = "</DSML｜";
    return agent_tail_matches(d->tail, d->len,
                              fullwidth_marker, sizeof(fullwidth_marker) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              ascii_marker, sizeof(ascii_marker) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              missing_open, sizeof(missing_open) - 1) ||
           agent_tail_matches(d->tail, d->len,
                              missing_close, sizeof(missing_close) - 1);
}

/* -- (kind, text) fragment emitter with per-kind coalescing --------- */

/* Sink for a coalesced fragment: kind is enum agent_stream_kind. In the server
 * this feeds the STREAM writer (T6); the Layer 3 test captures it directly. */
typedef void (*srv_emit_fn)(void *ud, uint32_t stream_id, uint32_t kind,
                            const char *text, size_t len);

typedef struct {
    agent_dsml_parser *parser;
    agent_tool_syntax syntax;
    agent_worker *worker;   /* for agent_trace; NULL in unit tests */
    srv_emit_fn emit;
    void *emit_ud;

    /* coalescing */
    int cur_kind;           /* -1 when nothing buffered */
    char *cbuf;
    size_t clen;
    size_t ccap;
    uint32_t stream_id;
    bool at_newline;        /* last emitted byte was '\n' (for notice framing) */

    /* <think>/</think> stripping (consumed here, never sent) */
    bool in_think;
    char pending[16];
    size_t pending_len;

    /* DSML/GLM start-marker detection in the visible stream */
    bool dsml_active;
    bool dsml_ignored;      /* stanza opened inside <think> */
    char dsml_start_tail[64];
    size_t dsml_start_len;
    bool post_think_gap;

    /* stray-marker detectors (malformed DSML outside a valid block) */
    agent_dsml_marker_detector plain_dsml;
    agent_dsml_marker_detector think_dsml;
    bool dsml_in_think;
    bool dsml_in_think_reported;

    /* tool-call fragment tracking (mirrors parser progress) */
    bool tool_announced;
    char tool_name[64];
    bool param_active;
    char param_name[64];
    char param_end_tail[64];
    size_t param_end_len;
} srv_stream;

static void srv_stream_flush(srv_stream *s) {
    if (s->cur_kind < 0 || s->clen == 0) {
        s->cur_kind = -1;
        s->clen = 0;
        return;
    }
    if (s->emit)
        s->emit(s->emit_ud, ++s->stream_id, (uint32_t)s->cur_kind,
                s->cbuf, s->clen);
    s->clen = 0;
    s->cur_kind = -1;
}

static void srv_emit(srv_stream *s, int kind, const char *text, size_t len) {
    if (!len) return;
    if (s->cur_kind != -1 && s->cur_kind != kind) srv_stream_flush(s);
    s->cur_kind = kind;
    if (s->clen + len > s->ccap) {
        size_t cap = s->ccap ? s->ccap * 2 : 256;
        while (cap < s->clen + len) cap *= 2;
        s->cbuf = xrealloc(s->cbuf, cap);
        s->ccap = cap;
    }
    memcpy(s->cbuf + s->clen, text, len);
    s->clen += len;
    for (size_t i = 0; i < len; i++) s->at_newline = text[i] == '\n';
}

static void srv_emit_cstr(srv_stream *s, int kind, const char *str) {
    srv_emit(s, kind, str, strlen(str));
}

/* Emit a bracketed notice ("[invalid tool call: ...]") on its own line, as
 * NORMAL text (the client colours it). */
static void srv_emit_notice(srv_stream *s, const char *body) {
    if (!s->at_newline) srv_emit(s, AGENT_STREAM_NORMAL, "\n", 1);
    srv_emit_cstr(s, AGENT_STREAM_NORMAL, body);
}

/* -- tool-call structure -> TOOL_NAME / TOOL_PARAM_NAME fragments --- */

static void srv_stream_tool_events(srv_stream *s) {
    agent_dsml_parser *p = s->parser;
    if (!s->tool_announced && p->current.name) {
        snprintf(s->tool_name, sizeof(s->tool_name), "%s", p->current.name);
        s->tool_announced = true;
        srv_emit_cstr(s, AGENT_STREAM_TOOL_NAME, p->current.name);
    }
    if (s->tool_announced && !p->current.name && !s->param_active) {
        /* the current call was pushed to p->calls; ready for the next one */
        s->tool_announced = false;
        s->tool_name[0] = '\0';
    }
    if (!s->param_active && p->state == AGENT_DSML_PARAM_VALUE && p->param_name) {
        snprintf(s->param_name, sizeof(s->param_name), "%s", p->param_name);
        s->param_active = true;
        s->param_end_len = 0;
        srv_emit_cstr(s, AGENT_STREAM_TOOL_PARAM_NAME, p->param_name);
    }
}

static void srv_stream_param_end(srv_stream *s) {
    s->param_active = false;
    s->param_name[0] = '\0';
    s->param_end_len = 0;
}

/* One parameter-value byte. Mirrors agent_tool_viz_param_value_byte: hold back
 * a possible closing tag ("</｜DSML｜parameter>" / "</arg_value>") so its
 * bytes never leak into a TOOL_PARAM_VALUE fragment. */
static void srv_stream_param_value_byte(srv_stream *s, char c) {
    if (s->param_end_len || c == '<') {
        if (s->param_end_len == sizeof(s->param_end_tail)) {
            size_t keep = s->param_end_len;
            s->param_end_len = 0;
            srv_emit(s, AGENT_STREAM_TOOL_PARAM_VALUE, s->param_end_tail, keep);
            if (c != '<') {
                srv_emit(s, AGENT_STREAM_TOOL_PARAM_VALUE, &c, 1);
                return;
            }
        }
        if (s->param_end_len < sizeof(s->param_end_tail))
            s->param_end_tail[s->param_end_len++] = c;
        bool complete = false;
        if (agent_tool_value_close_tail(s->parser->syntax, s->param_end_tail,
                                        s->param_end_len, &complete)) {
            if (complete) srv_stream_param_end(s);
            return;
        }
        size_t keep = s->param_end_len;
        s->param_end_len = 0;
        srv_emit(s, AGENT_STREAM_TOOL_PARAM_VALUE, s->param_end_tail, keep);
        return;
    }
    srv_emit(s, AGENT_STREAM_TOOL_PARAM_VALUE, &c, 1);
}

/* -- notices -------------------------------------------------------- */

static void srv_stream_finish_ignored_dsml(srv_stream *s, const char *detail) {
    const char *msg = detail && detail[0] ? detail :
        "tool calling is not allowed inside <think></think>";
    s->dsml_in_think = true;
    s->dsml_in_think_reported = true;
    agent_trace(s->worker, "dsml ignored inside thinking: %s", msg);
    char line[320];
    snprintf(line, sizeof(line), "[tool call ignored: %s]\n", msg);
    srv_emit_notice(s, line);
    agent_dsml_parser_reset(s->parser);
    s->dsml_active = false;
    s->dsml_ignored = false;
    srv_stream_param_end(s);
    s->tool_announced = false;
    s->tool_name[0] = '\0';
}

static void srv_stream_malformed_dsml(srv_stream *s, const char *detail) {
    const char *msg = detail && detail[0] ? detail :
        "DSML markup outside a valid tool_calls block";
    if (s->parser->state == AGENT_DSML_ERROR) return;
    agent_dsml_set_error(s->parser, msg);
    agent_trace(s->worker, "malformed dsml in assistant output: %s", msg);
    char line[320];
    snprintf(line, sizeof(line), "[invalid tool call: %s]\n", msg);
    srv_emit_notice(s, line);
}

/* -- byte routing ------------------------------------------------- */

static void srv_stream_feed_dsml_byte(srv_stream *s, char c) {
    bool was_param = !s->dsml_ignored && s->param_active;
    agent_dsml_feed(s->parser, &c, 1);
    if (!s->dsml_ignored) {
        srv_stream_tool_events(s);
        if (was_param) srv_stream_param_value_byte(s, c);
        if (was_param && s->parser->state != AGENT_DSML_PARAM_VALUE &&
            s->param_active)
            srv_stream_param_end(s);
    }
    if (s->parser->state == AGENT_DSML_DONE) {
        if (s->dsml_ignored) {
            srv_stream_finish_ignored_dsml(
                s, "tool calling is not allowed inside <think></think>");
        } else {
            agent_trace(s->worker, "dsml done calls=%d", s->parser->calls.len);
            s->dsml_active = false;
        }
    } else if (s->parser->state == AGENT_DSML_ERROR) {
        if (s->dsml_ignored) {
            srv_stream_finish_ignored_dsml(
                s, "malformed tool call inside <think></think>");
        } else {
            agent_trace(s->worker, "dsml error %s",
                        s->parser->error[0] ? s->parser->error : "parse error");
            if (s->parser->raw && s->parser->raw_len) {
                if (!s->at_newline) srv_emit(s, AGENT_STREAM_NORMAL, "\n", 1);
                srv_emit(s, AGENT_STREAM_NORMAL, s->parser->raw,
                         s->parser->raw_len);
            }
            char line[320];
            snprintf(line, sizeof(line), "[invalid tool call: %s]\n",
                     s->parser->error[0] ? s->parser->error : "parse error");
            srv_emit_notice(s, line);
            s->dsml_active = false;
        }
    }
}

static void srv_stream_start_dsml(srv_stream *s, bool ignored) {
    s->dsml_active = true;
    s->dsml_ignored = ignored;
    if (ignored) s->dsml_in_think = true;
    s->dsml_start_len = 0;
    s->post_think_gap = false;
    agent_trace(s->worker, "%s tool start detected%s",
                s->syntax == AGENT_TOOL_SYNTAX_GLM ? "glm" : "dsml",
                ignored ? " inside thinking" : "");
    agent_dsml_start(s->parser);
    if (!ignored) srv_stream_tool_events(s);
}

static void srv_stream_note_plain_dsml_byte(srv_stream *s, char c);

static void srv_stream_flush_start_tail(srv_stream *s) {
    if (!s->dsml_start_len) return;
    s->post_think_gap = false;
    for (size_t i = 0; i < s->dsml_start_len; i++) {
        char c = s->dsml_start_tail[i];
        srv_emit(s, s->in_think ? AGENT_STREAM_THINK : AGENT_STREAM_NORMAL, &c, 1);
        srv_stream_note_plain_dsml_byte(s, c);
        if (s->parser->state == AGENT_DSML_ERROR) break;
    }
    s->dsml_start_len = 0;
}

static void srv_stream_note_thinking_dsml_byte(srv_stream *s, char c) {
    if (!s->in_think || s->dsml_in_think) return;
    if (agent_dsml_marker_detector_feed(&s->think_dsml, c))
        s->dsml_in_think = true;
}

static void srv_stream_note_plain_dsml_byte(srv_stream *s, char c) {
    if (s->parser->state == AGENT_DSML_ERROR) return;
    if (s->dsml_active || s->in_think || s->dsml_in_think) return;
    if (agent_dsml_marker_detector_feed(&s->plain_dsml, c))
        srv_stream_malformed_dsml(
            s, "DSML markup outside a valid tool_calls block");
}

static void srv_stream_normal_byte(srv_stream *s, char c) {
    static const char canonical_invoke[] = "<｜DSML｜invoke";
    const char *start = s->syntax == AGENT_TOOL_SYNTAX_GLM ?
        "<tool_call>" : "<｜DSML｜tool_calls>";
    if (s->parser->state == AGENT_DSML_ERROR) return;
    srv_stream_note_thinking_dsml_byte(s, c);

    if (s->post_think_gap &&
        (c == ' ' || c == '\t' || c == '\r' || c == '\n'))
        return;

    if (s->dsml_start_len || c == start[0]) {
        if (s->dsml_start_len < sizeof(s->dsml_start_tail))
            s->dsml_start_tail[s->dsml_start_len++] = c;
        bool complete = false, implicit_invoke = false;
        if (agent_stream_dsml_start_match(s->syntax, s->dsml_start_tail,
                                          s->dsml_start_len, &complete,
                                          &implicit_invoke)) {
            if (complete) {
                srv_stream_start_dsml(s, s->in_think);
                if (s->syntax == AGENT_TOOL_SYNTAX_DSML && implicit_invoke) {
                    for (size_t i = 0; i < sizeof(canonical_invoke) - 1; i++)
                        srv_stream_feed_dsml_byte(s, canonical_invoke[i]);
                }
            }
            return;
        }
        if (s->dsml_start_len > 1 &&
            s->dsml_start_tail[s->dsml_start_len - 1] == start[0]) {
            s->post_think_gap = false;
            size_t flush = s->dsml_start_len - 1;
            for (size_t i = 0; i < flush; i++) {
                char b = s->dsml_start_tail[i];
                srv_emit(s, s->in_think ? AGENT_STREAM_THINK : AGENT_STREAM_NORMAL,
                         &b, 1);
                srv_stream_note_plain_dsml_byte(s, b);
                if (s->parser->state == AGENT_DSML_ERROR) break;
            }
            if (s->parser->state == AGENT_DSML_ERROR) {
                s->dsml_start_len = 0;
                return;
            }
            s->dsml_start_tail[0] = start[0];
            s->dsml_start_len = 1;
            return;
        }
        srv_stream_flush_start_tail(s);
        return;
    }

    s->post_think_gap = false;
    srv_emit(s, s->in_think ? AGENT_STREAM_THINK : AGENT_STREAM_NORMAL, &c, 1);
    srv_stream_note_plain_dsml_byte(s, c);
}

/* Top-level: classify one chunk of sampled assistant text into fragments. The
 * transcript is unchanged; only the projection is rewritten. */
static void srv_stream_text(srv_stream *s, const char *text, size_t len,
                            bool finish) {
    const char *think_open = "<think>";
    const char *think_close = "</think>";
    size_t total = s->pending_len + len;
    char *buf = xmalloc(total ? total : 1);
    if (s->pending_len) memcpy(buf, s->pending, s->pending_len);
    if (len) memcpy(buf + s->pending_len, text, len);
    s->pending_len = 0;

    size_t i = 0;
    while (i < total) {
        char *cur = buf + i;
        size_t rem = total - i;
        if (!s->dsml_active && bytes_has_prefix(cur, rem, think_open)) {
            srv_stream_flush_start_tail(s);
            s->post_think_gap = false;
            s->in_think = true;
            i += strlen(think_open);
            continue;
        }
        if (!s->dsml_active && bytes_has_prefix(cur, rem, think_close)) {
            srv_stream_flush_start_tail(s);
            s->in_think = false;
            if (!s->at_newline) srv_emit(s, AGENT_STREAM_NORMAL, "\n", 1);
            srv_emit(s, AGENT_STREAM_NORMAL, "\n", 1);
            s->post_think_gap = true;
            i += strlen(think_close);
            continue;
        }
        if (!finish && !s->dsml_active && cur[0] == '<' &&
            (bytes_is_partial_prefix(cur, rem, think_open) ||
             bytes_is_partial_prefix(cur, rem, think_close))) {
            if (rem < sizeof(s->pending)) {
                memcpy(s->pending, cur, rem);
                s->pending_len = rem;
            }
            break;
        }

        if (s->dsml_active)
            srv_stream_feed_dsml_byte(s, cur[0]);
        else
            srv_stream_normal_byte(s, cur[0]);
        i++;
    }
    free(buf);

    if (finish) {
        srv_stream_flush_start_tail(s);
        s->post_think_gap = false;
        if (s->dsml_active) agent_dsml_finish(s->parser);
        if (s->dsml_active) {
            if (s->parser->state == AGENT_DSML_DONE) {
                if (s->dsml_ignored)
                    srv_stream_finish_ignored_dsml(
                        s, "tool calling is not allowed inside <think></think>");
                else {
                    agent_trace(s->worker, "dsml done calls=%d",
                                s->parser->calls.len);
                    s->dsml_active = false;
                }
            } else if (s->dsml_ignored) {
                srv_stream_finish_ignored_dsml(
                    s, "tool calling is not allowed inside <think></think>");
            } else {
                srv_emit_notice(s, "[tool call interrupted]\n");
                s->dsml_active = false;
            }
        }
        if (s->dsml_in_think && !s->dsml_in_think_reported)
            srv_stream_finish_ignored_dsml(
                s, "tool calling is not allowed inside <think></think>");
        srv_stream_flush(s);
    }
}

static void srv_stream_init(srv_stream *s, agent_dsml_parser *parser,
                            agent_tool_syntax syntax, agent_worker *worker,
                            srv_emit_fn emit, void *emit_ud) {
    memset(s, 0, sizeof(*s));
    s->parser = parser;
    s->syntax = syntax;
    s->worker = worker;
    s->emit = emit;
    s->emit_ud = emit_ud;
    s->cur_kind = -1;
    s->at_newline = true;
}

static void srv_stream_free(srv_stream *s) {
    free(s->cbuf);
    s->cbuf = NULL;
    s->clen = s->ccap = 0;
}

/* Greedy-sampling decision: stays server-side, so there is no per-token
 * round-trip. Copied from ds4_agent.c, retargeted to srv_stream. */
static bool agent_stream_wants_greedy_sampling(const srv_stream *s) {
    if (!s || !s->parser) return false;
    if (s->parser->state == AGENT_DSML_ERROR ||
        s->parser->state == AGENT_DSML_DONE)
        return false;
    if (s->dsml_start_len > 1) return true;
    if (!s->dsml_active) return false;
    if (s->parser->state == AGENT_DSML_STRUCTURAL) return true;
    if (s->parser->state != AGENT_DSML_PARAM_VALUE) return false;
    return s->parser->param_close_prefix;
}

static bool agent_stream_compaction_needs_lookahead(const srv_stream *s) {
    return !s->dsml_active && s->parser->state == AGENT_DSML_SEARCH &&
           (s->pending_len || s->dsml_start_len);
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

/* Apply a SESSION new: fold the client's sampler / think / seed / -sys /
 * prefix-file / hints / power knobs into cfg->gen, rebuild the transcript at
 * the system+tools prompt (sysprompt.kv fast path), and fill the authoritative
 * SESSION new reply. Returns false (with err set) on a build failure or a busy
 * worker; the caller then replies ERR. */
static bool server_apply_session_new(agent_worker *w, server_config *cfg,
                                     const ap_session_new_args *a,
                                     ap_session_ready *ready,
                                     char *err, size_t errlen) {
    /* The worker builds (or loads) sysprompt.kv on startup; that first prefill
     * can take a while. Wait for it rather than bouncing the first SESSION new. */
    for (int i = 0; !worker_is_initialized(w, NULL) && i < 6000; i++) {
        struct timespec d = {0, 50000000L}; /* 50 ms, up to ~5 min total */
        nanosleep(&d, NULL);
    }
    if (!worker_is_initialized(w, NULL)) {
        snprintf(err, errlen, "worker did not finish starting up");
        return false;
    }
    if (!worker_is_idle(w)) {
        snprintf(err, errlen, "busy");
        return false;
    }

    if (a->temperature_set) {
        cfg->gen.temperature = (float)ap_milli_to_double(a->temperature);
        cfg->gen.temperature_set = true;
    }
    if (a->top_p_set) {
        cfg->gen.top_p = (float)ap_milli_to_double(a->top_p);
        cfg->gen.top_p_set = true;
    }
    if (a->min_p_set) {
        cfg->gen.min_p = (float)ap_milli_to_double(a->min_p);
        cfg->gen.min_p_set = true;
    }
    cfg->gen.seed = a->seed;
    if (a->n_predict > 0) cfg->gen.n_predict = (int)a->n_predict;
    if (a->think_mode <= (uint32_t)DS4_THINK_MAX)
        cfg->gen.think_mode = (ds4_think_mode)a->think_mode;
    if (a->power_set && a->power >= 1 && a->power <= 100)
        cfg->engine.power_percent = (int)a->power;

    free(cfg->gen.system);
    cfg->gen.system = (a->sys_text && a->sys_text_len)
        ? xstrndup(a->sys_text, a->sys_text_len) : NULL;

    ds4_prompt_prefix_free(&cfg->gen.prefix);
    memset(&cfg->gen.prefix, 0, sizeof(cfg->gen.prefix));
    if (a->prefix_file_text && a->prefix_file_text_len) {
        char perr[160] = {0};
        if (ds4_prompt_prefix_parse(&cfg->gen.prefix, a->prefix_file_text,
                                    a->prefix_file_text_len,
                                    perr, sizeof(perr)) != 0) {
            snprintf(err, errlen, "prefix file: %s",
                     perr[0] ? perr : "parse failed");
            return false;
        }
    }

    pthread_mutex_lock(&w->mu);
    w->hints.enabled = a->hints_enabled;
    w->hints.instructed = false;
    w->hints.applied = AGENT_HINTS_UNSET;
    pthread_mutex_unlock(&w->mu);

    if (!agent_worker_reset_to_sysprompt(w, err, errlen))
        return false;

    ready->ctx_used = (uint32_t)w->transcript.len;
    ready->ctx_size = (uint32_t)agent_worker_effective_ctx_size(w);
    ready->distributed_route_ready = true;
    ready->state = AGENT_STATE_IDLE;
    snprintf(ready->note, sizeof(ready->note), "session created");
    return true;
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

static serve_result serve_connection(server_conn *co, agent_worker *w,
                                     server_config *cfg, server_session *sess,
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
        server_fill_hello_reply(&reply, w->engine, cfg,
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
                    server_session_begin_new(sess, cfg->gen.ctx_size);
                    char nerr[256] = {0};
                    if (!server_apply_session_new(w, cfg, &m.new_args,
                                                  &sess->ready, nerr, sizeof(nerr))) {
                        conn_send_reply_err(co, AGENT_MSG_SESSION,
                                            nerr[0] ? nerr : "failed to start session");
                        break;
                    }
                    ap_buf body;
                    ap_buf_init(&body);
                    ap_encode_session_ready(&body, &sess->ready);
                    conn_send(co, AGENT_MSG_SESSION, &body);
                    ap_buf_free(&body);
                    break;
                }
                case AGENT_SESSION_RESUME: {
                    bool had_turn = sess->turn_in_progress;
                    if (!server_session_resume(sess)) {
                        conn_send_reply_err(co, AGENT_MSG_SESSION, "no parked session to resume");
                        break;
                    }
                    /* A turn cut off by the disconnect: close it at a
                     * well-formed IDLE boundary before the client re-enters. */
                    if (had_turn) {
                        worker_clear_interrupt(w);
                        agent_worker_append_assistant_turn_end(w);
                    }
                    sess->ready.ctx_used = (uint32_t)w->transcript.len;
                    sess->ready.ctx_size = (uint32_t)agent_worker_effective_ctx_size(w);
                    sess->ready.distributed_route_ready = true;
                    sess->ready.state = AGENT_STATE_IDLE;
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

static int run_server(ds4_engine *engine, server_config *cfg) {
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

    /* One worker for the life of the process: it owns the engine session, the
     * transcript and sysprompt.kv, and survives client disconnects (the parked
     * session). It builds sysprompt.kv on startup, like the monolith. */
    agent_worker worker;
    if (agent_worker_init(&worker, engine, cfg, trace) != 0) {
        fprintf(stderr, "ds4-agent-server: failed to start worker\n");
        agent_worker_free(&worker);
        if (g_listen_fd >= 0) { close(g_listen_fd); g_listen_fd = -1; }
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
        serve_result sr = serve_connection(&co, &worker, cfg, &sess, serving);
        conn_free(&co);

        if (serving) continue; /* the rejected extra connection: nothing changed */

        if (sr == SERVE_STOP) {
            server_session_shutdown(&sess);
        } else {
            server_session_park(&sess);
        }
    }

    server_session_shutdown(&sess);
    agent_worker_free(&worker);
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
    free(cfg.gen.system);
    ds4_prompt_prefix_free(&cfg.gen.prefix);
    return rc;
}
#endif
