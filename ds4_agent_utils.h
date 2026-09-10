#ifndef DS4_AGENT_UTILS_H
#define DS4_AGENT_UTILS_H

/* Pure helpers shared by ds4-agent-server and ds4-agent-client.
 *
 * Copied from ds4_agent.c (which keeps its own static copies, untouched) so
 * the two split binaries do not each duplicate them. Libc only: no ds4.h, no
 * engine, no linenoise. See AGENT-SPLIT-PLAN.md section 3.
 *
 * The parse_* helpers keep the monolith's behaviour: on a bad value they print
 * "ds4-agent: invalid value for <opt>: <val>" to stderr and exit(2).
 */

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Allocation wrappers that abort the process on failure. */
void *xmalloc(size_t n);
void *xrealloc(void *ptr, size_t n);
char *xstrdup(const char *s);
char *xstrndup(const char *s, size_t n);

/* Byte-capped growable text buffer used to assemble tool output. When the
 * append would cross b->limit the buffer stops growing, trims back to a UTF-8
 * boundary, and sets b->truncated; agent_buf_take then appends a truncation
 * notice. A zero limit means unbounded. */
#define AGENT_TOOL_MAX_BYTES (128 * 1024)

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
    size_t limit;
    bool truncated;
} agent_buf;

void  agent_buf_append(agent_buf *b, const char *s, size_t n);
void  agent_buf_puts(agent_buf *b, const char *s);
char *agent_buf_take(agent_buf *b); /* returns an owned string, resets b */

/* Unbounded growable input buffer (NUL-terminated). */
typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} agent_input_buf;

void  agent_input_buf_append(agent_input_buf *b, const char *s, size_t n);
char *agent_input_buf_take(agent_input_buf *b); /* owned string, resets b */
void  agent_input_buf_free(agent_input_buf *b);

/* write(2) loop, retrying on EINTR. Unlike the monolith copy this never routes
 * stdout through linenoise; terminal painting is a client concern. */
void write_all(int fd, const char *p, size_t n);

/* Monotonic clock in seconds. */
double now_sec(void);

/* CLI value parsers. All exit(2) with a message on a malformed value. */
int      parse_int(const char *s, const char *opt);             /* > 0 */
int      parse_nonnegative_int(const char *s, const char *opt); /* >= 0 */
uint64_t parse_u64(const char *s, const char *opt);             /* > 0 */
float    parse_float_range(const char *s, const char *opt, float min, float max);

/* Soft parsers: return false on a bad value, leave *out untouched. */
bool parse_power_percent(const char *arg, int *out);   /* 1..100 */
bool parse_steering_level(const char *arg, float *out); /* -100..100, finite */

/* Lenient boolean: true/yes/1 and false/no/0 (case-insensitive), else def. */
bool agent_parse_bool_default(const char *s, bool def);

/* mkdir -p with 0700 components. Returns false on the first real failure. */
bool agent_mkdir_p(const char *path);

/* Toggle O_NONBLOCK on fd; stores the previous flags in *old_flags when given.
 * Returns -1 on an fcntl error. */
int set_nonblock(int fd, bool on, int *old_flags);

/* Read a self-pipe / eventfd dry, ignoring EINTR. */
void drain_wake_fd(int fd);

#endif /* DS4_AGENT_UTILS_H */
