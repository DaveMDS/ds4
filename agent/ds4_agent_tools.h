/* ds4_agent_tools.h -- shared client-side tool framework.
 *
 * Tool structs (proto_tool_call from proto.h), tool-argument parsing,
 * fit-context (TOKENS -> COUNT math), the client worker struct, and dispatch
 * declarations live here; implementations are one file per tool group:
 *   ds4_agent_tools_file.c  (read/more/write/list/edit/search)
 *   ds4_agent_tools_bash.c  (bash/bash_status/bash_stop)
 *   ds4_agent_tools_web.c   (google_search/visit_page + approval)
 * view_image is a special case: it lives on the client (reads file bytes) but
 * the image attach is a generic server service (ATTACH_IMAGE), not a client
 * tool, so it is not part of dispatch.
 *
 * Pure helpers are `static inline` so test harnesses can include this header
 * without link conflicts.  The per-group functions are external (one object
 * per group) and are declared here.
 */


#ifndef DS4_AGENT_TOOLS_H
#define DS4_AGENT_TOOLS_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <limits.h>
#include <sys/types.h>

#include "ds4_agent_utils.h"
#include "ds4_agent_proto.h"


/* ============================================================================
 * Constants (shared by the tool groups)
 * ========================================================================== */

#define AGENT_TOOL_MAX_BYTES            (128*1024)
#define AGENT_FILE_MAX_BYTES            (16*1024*1024)
#define AGENT_READ_DEFAULT_LINES_SMALL  120
#define AGENT_READ_DEFAULT_LINES_MEDIUM 240
#define AGENT_READ_DEFAULT_LINES_LARGE  500
#define AGENT_READ_SMALL_CONTEXT_MAX    8192
#define AGENT_READ_MEDIUM_CONTEXT_MAX   16384
#define AGENT_TOOL_RESULT_RESERVE_TOKENS 1024
#define AGENT_EDIT_UPTO_MIN_PREFIX_BYTES 64
#define AGENT_EDIT_UPTO_MIN_PREFIX_LINES 2

#define AGENT_WEB_HEAD_BYTES (8*1024)
#define AGENT_WEB_HEAD_LINES 100

#define AGENT_BASH_HEAD_LINES          100
#define AGENT_BASH_HEAD_BYTES           (8*1024)
#define AGENT_BASH_PROGRESS_TAIL_LINES  4
#define AGENT_BASH_FINAL_TAIL_LINES     20
#define AGENT_BASH_TAIL_BYTES           (32*1024)

/* ============================================================================
 * Tool-argument parsing (pure)
 * ========================================================================== */

static inline int agent_tools_parse_timeout(const char *s) {
    if (!s || !s[0]) return 3600;
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s || v <= 0.0 || !isfinite(v)) return 3600;
    if (v < 1.0) v = 1.0;
    if (v > 24.0 * 3600.0) v = 24.0 * 3600.0;
    return (int)v;
}

static inline int agent_tools_parse_int_default(const char *s, int def, int min, int max) {
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

static inline bool agent_tools_parse_bool_default(const char *s, bool def) {
    if (!s || !s[0]) return def;
    if (!strcasecmp(s, "true") || !strcasecmp(s, "yes") || !strcmp(s, "1"))
        return true;
    if (!strcasecmp(s, "false") || !strcasecmp(s, "no") || !strcmp(s, "0"))
        return false;
    return def;
}

/* Look up a tool argument by name.  Returns "" for a present-but-empty value
 * and NULL when the argument is absent. */
static inline const char *agent_tools_arg_value(const proto_tool_call *call,
                                                const char *name) {
    if (!call) return NULL;
    for (int i = 0; i < call->argc; i++) {
        if (call->args[i].name && !strcmp(call->args[i].name, name))
            return call->args[i].value ? call->args[i].value : "";
    }
    return NULL;
}

/* ============================================================================
 * Fit-context (TOKENS -> COUNT math)
 *
 * The client sends a TOKENS (C2S, string) message and the server replies COUNT
 * (S2C, varint) with the token count of that text.  These helpers decide, from
 * a token count and the CLI context size, whether a tool observation fits.
 * ========================================================================== */

static inline int agent_tools_reserve_tokens(int ctx) {
    int reserve = AGENT_TOOL_RESULT_RESERVE_TOKENS;
    if (ctx > 0) {
        int proportional = ctx / 8;
        if (proportional < 16) proportional = 16;
        if (reserve > proportional) reserve = proportional;
    }
    return reserve;
}

static inline bool agent_tools_fit_tokens(int ctx, int tokens, int reserve) {
    return ctx > 0 && tokens + reserve < ctx;
}

/* ============================================================================
 * Client worker
 * ========================================================================== */

typedef struct agent_tools_bash_job agent_tools_bash_job;
typedef struct ds4_web ds4_web;

/* UI callbacks.  The tools framework itself is UI-free: these are optional and
 * the client (ds4_agent_client.c) wires them to its UI thread. */
typedef void (*agent_tools_publish_fn)(void *ud, const char *text, size_t len);
typedef void (*agent_tools_status_fn)(void *ud, const char *msg);

typedef struct {
    void *ud;
    agent_tools_publish_fn publish;   /* raw text to the UI (bash output) */
    agent_tools_status_fn status;     /* one-line status ("Searching Google...") */
    bool ask_web;                     /* require user approval for web tools */
} agent_tools_io;

typedef struct {
    agent_tools_io io;
    int ctx_size;                     /* CLI context size (--ctx) */
    bool edit_upto;                   /* allow edit old=... [upto] anchors */
    ds4_web *web;                     /* owned by the client, NULL in tests */
    bool interrupt;                   /* set by INTERRUPT to stop a bash wait */

    /* read/more continuation state */
    char more_path[PATH_MAX];
    int more_next_line;
    off_t more_byte_offset;
    bool more_mid_line;
    bool more_bare;
    bool more_valid;

    /* bash jobs */
    agent_tools_bash_job *bash_jobs;
    int next_bash_job_id;
    bool raw_mode_needs_restore;
} agent_tools_worker;

static inline bool agent_tools_should_interrupt(const agent_tools_worker *w) {
    return w && w->interrupt;
}

/* ============================================================================
 * Dispatch and per-group declarations
 * ========================================================================== */

char *agent_tools_dispatch(agent_tools_worker *w, const proto_tool_call *call);

/* file group: ds4_agent_tools_file.c */
char *agent_tools_file_read(agent_tools_worker *w, const proto_tool_call *call);
char *agent_tools_file_more(agent_tools_worker *w, const proto_tool_call *call);
char *agent_tools_file_write(agent_tools_worker *w, const proto_tool_call *call);
char *agent_tools_file_list(const proto_tool_call *call);
char *agent_tools_file_edit(agent_tools_worker *w, const proto_tool_call *call);
char *agent_tools_file_search(agent_tools_worker *w, const proto_tool_call *call);

/* test-facing file helpers (also used internally by the file tools) */
int agent_tools_file_replace_file(const char *path, const char *data, size_t len,
                                  const char *expected, size_t expected_len,
                                  char *err, size_t errlen);
int agent_tools_file_read_file_bytes(const char *path, char **data, size_t *len,
                                     char *err, size_t errlen);
bool agent_tools_file_edit_find_old_span(const char *data, size_t len,
                                         const char *old, bool allow_upto,
                                         const char **match,
                                         size_t *match_len, bool *anchored,
                                         char *err, size_t err_len);
char *agent_tools_file_read_range(agent_tools_worker *w, const char *path,
                                  int start_line, int max_lines, bool whole_file,
                                  bool bare, bool set_more);

/* bash group: ds4_agent_tools_bash.c */
char *agent_tools_bash_run(agent_tools_worker *w, const proto_tool_call *call);
char *agent_tools_bash_status(agent_tools_worker *w, const proto_tool_call *call);
char *agent_tools_bash_stop(agent_tools_worker *w, const proto_tool_call *call);
void agent_tools_bash_jobs_free(agent_tools_worker *w);

/* test-facing bash internals (also used internally by the bash tools) */
agent_tools_bash_job *agent_tools_bash_start(agent_tools_worker *w, const char *cmd,
                                             int timeout_sec, char *err, size_t err_len);
char *agent_tools_bash_observation(agent_tools_bash_job *job, bool mark_observed,
                                   bool *done);
void agent_tools_bash_remove_job(agent_tools_worker *w, agent_tools_bash_job *target);
void agent_tools_bash_signal(agent_tools_bash_job *job, int sig);
bool agent_tools_bash_is_running(agent_tools_bash_job *job);
const char *agent_tools_bash_job_path(const agent_tools_bash_job *job);
size_t agent_tools_bash_job_bytes(const agent_tools_bash_job *job);
int agent_tools_bash_job_id(const agent_tools_bash_job *job);

/* web group: ds4_agent_tools_web.c */
char *agent_tools_web_google_search(agent_tools_worker *w, const proto_tool_call *call);
char *agent_tools_web_visit_page(agent_tools_worker *w, const proto_tool_call *call);

#endif /* DS4_AGENT_TOOLS_H */
