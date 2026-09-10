/* ds4_agent_tools_web.c -- client-side web tools: google_search, visit_page.
 * Ported from the monolithic ds4_agent.c (agent_tool_google_search ->
 * agent_tools_web_google_search, agent_tool_visit_page ->
 * agent_tools_web_visit_page) plus the small helper trio agent_count_lines /
 * agent_string_head / agent_write_temp_text.
 *
 * The browser itself (ds4_web.c) is owned by the client; the tools framework
 * only calls it and formats observations.  Status lines go through
 * w->io.status (optional), not a direct UI.
 *
 * Build/link note: this is one object; include "ds4_agent_tools.h" from the
 * client and the test harness, and link ds4_web.o.
 */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "ds4_agent_tools.h"
#include "ds4_web.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

static int agent_tools_web_count_lines(const char *s) {
    if (!s || !s[0]) return 0;
    int lines = 0;
    for (const char *p = s; *p; p++) {
        if (*p == '\n') lines++;
    }
    if (s[strlen(s) - 1] != '\n') lines++;
    return lines;
}

static char *agent_tools_web_string_head(const char *s, int max_lines, size_t max_bytes,
                                         int *lines_read, bool *byte_limited) {
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!s) return xstrdup("");
    size_t used = 0;
    int lines = 0;
    while (s[used] && used < max_bytes && lines < max_lines) {
        if (s[used++] == '\n') lines++;
    }
    if (s[used] && used >= max_bytes && byte_limited) *byte_limited = true;
    if (used && s[used - 1] != '\n' && lines < max_lines) lines++;
    if (lines_read) *lines_read = lines;
    return xstrndup(s, used);
}

static bool agent_tools_web_write_temp_text(const char *prefix, const char *text,
                                            char *path, size_t path_len,
                                            char *err, size_t err_len) {
    char tmpl[PATH_MAX];
    snprintf(tmpl, sizeof(tmpl), "/tmp/%s_XXXXXX", prefix);
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        snprintf(err, err_len, "failed to create temporary file: %s", strerror(errno));
        return false;
    }
    size_t len = text ? strlen(text) : 0;
    const char *p = text ? text : "";
    size_t left = len;
    while (left) {
        ssize_t n = write(fd, p, left);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) {
            snprintf(err, err_len, "failed to write temporary file: %s", strerror(errno));
            close(fd);
            unlink(tmpl);
            return false;
        }
        p += n;
        left -= (size_t)n;
    }
    if (close(fd) != 0) {
        snprintf(err, err_len, "failed to close temporary file: %s", strerror(errno));
        unlink(tmpl);
        return false;
    }
    snprintf(path, path_len, "%s", tmpl);
    return true;
}

static void agent_tools_web_status(agent_tools_worker *w, const char *fmt,
                                   const char *arg) {
    if (!w->io.status) return;
    char msg[256];
    snprintf(msg, sizeof(msg), fmt, arg);
    w->io.status(w->io.ud, msg);
}

char *agent_tools_web_google_search(agent_tools_worker *w, const proto_tool_call *call) {
    const char *query = agent_tools_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: google_search requires query\n");
    if (!w->web)
        return xstrdup("Tool error: google_search requires web browser support (--web)\n");
    char err[256] = {0};
    agent_tools_web_status(w, "Searching Google for %s...", query);
    char *md = ds4_web_google_search(w->web, query, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: google_search failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    return md;
}

char *agent_tools_web_visit_page(agent_tools_worker *w, const proto_tool_call *call) {
    const char *url = agent_tools_arg_value(call, "url");
    if (!url || !url[0]) return xstrdup("Tool error: visit_page requires url\n");
    if (!w->web)
        return xstrdup("Tool error: visit_page requires web browser support (--web)\n");
    char err[256] = {0};
    agent_tools_web_status(w, "Opening page %s...", url);
    char *md = ds4_web_visit_page(w->web, url, err, sizeof(err));
    if (!md) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "unknown error");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char path[PATH_MAX];
    if (!agent_tools_web_write_temp_text("ds4_agent_web", md, path, sizeof(path),
                                         err, sizeof(err)))
    {
        free(md);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: visit_page failed: ");
        agent_buf_puts(&b, err[0] ? err : "could not store rendered page");
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int total_lines = agent_tools_web_count_lines(md);
    int shown_lines = 0;
    bool byte_limited = false;
    char *head = agent_tools_web_string_head(md, AGENT_WEB_HEAD_LINES, AGENT_WEB_HEAD_BYTES,
                                             &shown_lines, &byte_limited);
    bool truncated = byte_limited || shown_lines < total_lines;
    agent_buf out = {0};
    char line[PATH_MAX + 256];
    snprintf(line, sizeof(line),
             "visit_page url=%s\noutput_path=%s (%zu bytes, %d lines)\n",
             url, path, strlen(md), total_lines);
    agent_buf_puts(&out, line);
    if (truncated) {
        snprintf(line, sizeof(line), "<head -%d %s>\n",
                 AGENT_WEB_HEAD_LINES, path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</head>\n");
        agent_buf_puts(&out,
            "Use read path=<output_path> start_line=<line> max_lines=<count> raw=true to inspect more rendered Markdown.\n");
    } else {
        agent_buf_puts(&out, "<markdown>\n");
        agent_buf_puts(&out, head);
        if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
        agent_buf_puts(&out, "</markdown>\n");
    }
    free(head);
    free(md);
    return agent_buf_take(&out);
}
