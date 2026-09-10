/* ds4_agent_utils.h -- shared client/server basic utils.
 *
 * It is self-contained (no engine, no file I/O).  Everything is `static inline`
 * so including the header in several translation units cannot create link conflicts.
 */


#ifndef DS4_AGENT_UTILS_H
#define DS4_AGENT_UTILS_H

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <math.h>


/* ============================================================================
 * Minimal allocation helpers (self-contained)
 * ========================================================================== */

static inline void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) { perror("ds4-agent: malloc"); exit(1); }
    return p;
}

static inline void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) { perror("ds4-agent: realloc"); exit(1); }
    return p;
}

static inline char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

static inline char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

/* ============================================================================
 * Shared CLI helpers (both binaries)
 * ========================================================================== */

static inline double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static inline const char *need_arg(int *i, int argc, char **argv, const char *opt) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "ds4-agent: missing value for %s\n", opt);
        exit(2);
    }
    return argv[++(*i)];
}

static inline int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

static inline uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

static inline float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

/* ============================================================================
 * agent_buf -- dynamic string buffer (server prompt/title/persistence building,
 * client tool output)
 * ========================================================================== */

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
    size_t limit;
    bool truncated;
} agent_buf;

static inline void agent_buf_append(agent_buf *b, const char *s, size_t n) {
    if (!n || b->truncated) return;
    const size_t max = b->limit ? b->limit : SIZE_MAX - 1;
    if (n > max - b->len) {
        n = max > b->len ? max - b->len : 0;
        while (n && ((unsigned char)s[n] & 0xc0) == 0x80) n--;
        b->truncated = true;
    }
    if (!n) return;
    if (b->len + n + 1 > b->cap) {
        size_t cap = b->cap ? b->cap * 2 : 4096;
        while (cap < b->len + n + 1) cap *= 2;
        b->ptr = xrealloc(b->ptr, cap);
        b->cap = cap;
    }
    memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
}

static inline void agent_buf_puts(agent_buf *b, const char *s) {
    agent_buf_append(b, s, strlen(s));
}

static inline char *agent_buf_take(agent_buf *b) {
    if (b->truncated) {
        b->truncated = false;
        b->limit = 0;
        agent_buf_puts(b, "\n[Output truncated at the tool byte limit. Narrow the request.]\n");
    }
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

#endif /* DS4_AGENT_UTILS_H */
