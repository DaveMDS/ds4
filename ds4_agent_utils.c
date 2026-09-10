/* Pure helpers shared by ds4-agent-server and ds4-agent-client.
 * Copied from ds4_agent.c minus `static` (and minus the linenoise branch in
 * write_all). See ds4_agent_utils.h. */

#include "ds4_agent_utils.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

void *xmalloc(size_t n) {
    void *p = malloc(n ? n : 1);
    if (!p) {
        perror("ds4-agent: malloc");
        exit(1);
    }
    return p;
}

char *xstrdup(const char *s) {
    if (!s) s = "";
    size_t n = strlen(s);
    char *p = xmalloc(n + 1);
    memcpy(p, s, n + 1);
    return p;
}

char *xstrndup(const char *s, size_t n) {
    char *p = xmalloc(n + 1);
    memcpy(p, s, n);
    p[n] = '\0';
    return p;
}

void *xrealloc(void *ptr, size_t n) {
    void *p = realloc(ptr, n ? n : 1);
    if (!p) {
        perror("ds4-agent: realloc");
        exit(1);
    }
    return p;
}

void agent_buf_append(agent_buf *b, const char *s, size_t n) {
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

void agent_buf_puts(agent_buf *b, const char *s) {
    agent_buf_append(b, s, strlen(s));
}

char *agent_buf_take(agent_buf *b) {
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

void agent_input_buf_append(agent_input_buf *b, const char *s, size_t n) {
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

char *agent_input_buf_take(agent_input_buf *b) {
    if (!b->ptr) return xstrdup("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

void agent_input_buf_free(agent_input_buf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

void write_all(int fd, const char *p, size_t n) {
    while (n) {
        ssize_t wr = write(fd, p, n);
        if (wr <= 0) {
            if (wr < 0 && errno == EINTR) continue;
            return;
        }
        p += wr;
        n -= (size_t)wr;
    }
}

int parse_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v <= 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

int parse_nonnegative_int(const char *s, const char *opt) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v < 0 || v > INT32_MAX) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (int)v;
}

bool parse_power_percent(const char *arg, int *out) {
    char *end = NULL;
    long v = strtol(arg, &end, 10);
    if (!arg[0] || *end != '\0' || v < 1 || v > 100) return false;
    *out = (int)v;
    return true;
}

bool parse_steering_level(const char *arg, float *out) {
    char *end = NULL;
    errno = 0;
    float v = strtof(arg, &end);
    if (!arg[0] || *end != '\0' || errno == ERANGE || !isfinite(v) ||
        v < -100.0f || v > 100.0f) {
        return false;
    }
    *out = v;
    return true;
}

uint64_t parse_u64(const char *s, const char *opt) {
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (s[0] == '\0' || *end != '\0' || v == 0) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return (uint64_t)v;
}

float parse_float_range(const char *s, const char *opt, float min, float max) {
    char *end = NULL;
    float v = strtof(s, &end);
    if (s[0] == '\0' || *end != '\0' || !isfinite(v) || v < min || v > max) {
        fprintf(stderr, "ds4-agent: invalid value for %s: %s\n", opt, s);
        exit(2);
    }
    return v;
}

double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

bool agent_parse_bool_default(const char *s, bool def) {
    if (!s || !s[0]) return def;
    if (!strcasecmp(s, "true") || !strcasecmp(s, "yes") || !strcmp(s, "1"))
        return true;
    if (!strcasecmp(s, "false") || !strcasecmp(s, "no") || !strcmp(s, "0"))
        return false;
    return def;
}

bool agent_mkdir_p(const char *path) {
    if (!path || !path[0]) return false;
    char *tmp = xstrdup(path);
    for (char *p = tmp + 1; *p; p++) {
        if (*p != '/') continue;
        *p = '\0';
        if (mkdir(tmp, 0700) != 0 && errno != EEXIST) {
            free(tmp);
            return false;
        }
        *p = '/';
    }
    bool ok = mkdir(tmp, 0700) == 0 || errno == EEXIST;
    free(tmp);
    return ok;
}

int set_nonblock(int fd, bool on, int *old_flags) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (old_flags) *old_flags = flags;
    int next = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(fd, F_SETFL, next);
}

void drain_wake_fd(int fd) {
    char buf[128];
    for (;;) {
        ssize_t n = read(fd, buf, sizeof(buf));
        if (n > 0) continue;
        if (n < 0 && errno == EINTR) continue;
        break;
    }
}
