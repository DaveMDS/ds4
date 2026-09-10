/* ds4_agent_tools_file.c -- client-side file tools: read/more/write/list/
 * edit/search.  Ported from the monolithic ds4_agent.c (agent_tool_read ->
 * agent_tools_file_read, ...).  Pure file I/O, no engine, no UI: the whole
 * file is unit-testable.
 *
 * Build/link note: this is one object; include "ds4_agent_tools.h" from the
 * client and the test harness.
 */

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/xattr.h>
#include <errno.h>
#include <fcntl.h>
#include <fnmatch.h>
#include <regex.h>
#include <ctype.h>
#include <unistd.h>

#include "ds4_agent_tools.h"

/* ============================================================================
 * Line-span helpers
 * ========================================================================== */

typedef struct {
    size_t start;
    size_t content_end;
    size_t end;
} agent_line_span;

typedef struct {
    agent_line_span *v;
    int len;
    int cap;
} agent_line_spans;

static void agent_line_spans_free(agent_line_spans *spans) {
    free(spans->v);
    memset(spans, 0, sizeof(*spans));
}

static void agent_line_spans_push(agent_line_spans *spans, agent_line_span span) {
    if (spans->len == spans->cap) {
        spans->cap = spans->cap ? spans->cap * 2 : 128;
        spans->v = xrealloc(spans->v, (size_t)spans->cap * sizeof(spans->v[0]));
    }
    spans->v[spans->len++] = span;
}

/* Split a text buffer into line spans.  content_end excludes CR/LF so callers
 * can print or compare line content without newline spelling differences. */
static void agent_split_lines(const char *data, size_t len, agent_line_spans *spans) {
    size_t pos = 0;
    while (pos < len) {
        size_t start = pos;
        while (pos < len && data[pos] != '\n' && data[pos] != '\r') pos++;
        size_t content_end = pos;
        if (pos < len) {
            if (data[pos] == '\r' && pos + 1 < len && data[pos + 1] == '\n')
                pos += 2;
            else
                pos++;
        }
        agent_line_spans_push(spans, (agent_line_span){
            .start = start,
            .content_end = content_end,
            .end = pos,
        });
    }
}

/* ============================================================================
 * Open / read helpers
 * ========================================================================== */

static FILE *agent_open_regular_file(const char *path) {
    int fd = open(path, O_RDONLY | O_NONBLOCK | O_CLOEXEC);
    if (fd < 0) return NULL;
    struct stat st;
    int rc = fstat(fd, &st);
    if (rc != 0 || !S_ISREG(st.st_mode)) {
        int saved = rc != 0 ? errno : EINVAL;
        close(fd);
        errno = saved;
        return NULL;
    }
    FILE *fp = fdopen(fd, "rb");
    if (!fp) { int saved = errno; close(fd); errno = saved; }
    return fp;
}

int agent_tools_file_read_file_bytes(const char *path, char **data, size_t *len,
                                     char *err, size_t errlen) {
    FILE *fp = agent_open_regular_file(path);
    if (!fp) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return -1;
    }
    char *buf = NULL;
    size_t used = 0, cap = 0;
    char tmp[8192];
    while (true) {
        size_t n = fread(tmp, 1, sizeof(tmp), fp);
        if (n) {
            if (used + n > AGENT_FILE_MAX_BYTES) {
                fclose(fp);
                free(buf);
                snprintf(err, errlen, "file too large: %s exceeds %d bytes",
                         path, AGENT_FILE_MAX_BYTES);
                return -1;
            }
            if (used + n + 1 > cap) {
                cap = cap ? cap * 2 : 8192;
                while (cap < used + n + 1) cap *= 2;
                buf = xrealloc(buf, cap);
            }
            memcpy(buf + used, tmp, n);
            used += n;
            buf[used] = '\0';
        }
        if (n < sizeof(tmp)) {
            if (ferror(fp)) {
                snprintf(err, errlen, "read %s: %s", path, strerror(errno));
                fclose(fp);
                free(buf);
                return -1;
            }
            break;
        }
    }
    fclose(fp);
    if (!buf) buf = xstrdup("");
    *data = buf;
    *len = used;
    return 0;
}

static int agent_line_for_offset(const agent_line_spans *spans, size_t offset) {
    if (!spans || spans->len <= 0) return 1;
    for (int i = 0; i < spans->len; i++) {
        if (offset < spans->v[i].end) return i + 1;
    }
    return spans->len;
}

static bool agent_old_new_line_effect(const char *old_data, size_t old_len,
                                      const char *new_data, size_t new_len,
                                      size_t edit_offset, size_t replaced_len,
                                      int *start_line, int *end_line,
                                      int *delta) {
    agent_line_spans old_spans = {0};
    agent_line_spans new_spans = {0};
    agent_split_lines(old_data, old_len, &old_spans);
    agent_split_lines(new_data, new_len, &new_spans);
    bool ok = old_spans.len > 0;
    if (ok) {
        size_t old_last = edit_offset;
        if (replaced_len > 0) old_last = edit_offset + replaced_len - 1;
        if (old_last >= old_len) old_last = old_len ? old_len - 1 : 0;
        if (start_line) *start_line = agent_line_for_offset(&old_spans, edit_offset);
        if (end_line) *end_line = agent_line_for_offset(&old_spans, old_last);
        if (delta) *delta = new_spans.len - old_spans.len;
    }
    agent_line_spans_free(&old_spans);
    agent_line_spans_free(&new_spans);
    return ok;
}

/* ============================================================================
 * Atomic file replacement (write/edit core)
 * ========================================================================== */

static void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end);

static char *agent_edit_result(const char *path,
                               int start_line, int end_line, int delta,
                               const char *new_data, size_t new_len,
                               const char *kind) {
    agent_buf b = {.limit = AGENT_TOOL_MAX_BYTES};
    char msg[PATH_MAX + 180];
    snprintf(msg, sizeof(msg), "Edited %s using %s\n", path, kind);
    agent_buf_puts(&b, msg);
    if (start_line > 0 && end_line >= start_line) {
        snprintf(msg, sizeof(msg),
                 "Touched old lines %d-%d; current post-edit context follows.\n",
                 start_line, end_line);
        agent_buf_puts(&b, msg);
        if (delta != 0) {
            snprintf(msg, sizeof(msg),
                     "Line shift: old lines after %d moved by %+d (old line %d is now line %d). Re-read before relying on old line numbers there.\n",
                     end_line, delta, end_line + 1, end_line + 1 + delta);
            agent_buf_puts(&b, msg);
        }
    }
    if (start_line > 0 && end_line >= start_line) {
        int new_anchor_end = end_line + delta;
        if (new_anchor_end < start_line) new_anchor_end = start_line;
        agent_edit_result_append_context(&b, path, new_data, new_len,
                                         start_line, new_anchor_end);
    }
    return agent_buf_take(&b);
}

static bool agent_same_file_version(const struct stat *a, const struct stat *b) {
#ifdef __APPLE__
    struct timespec am = a->st_mtimespec, bm = b->st_mtimespec;
    struct timespec ac = a->st_ctimespec, bc = b->st_ctimespec;
#else
    struct timespec am = a->st_mtim, bm = b->st_mtim;
    struct timespec ac = a->st_ctim, bc = b->st_ctim;
#endif
    return a->st_dev == b->st_dev && a->st_ino == b->st_ino &&
           a->st_size == b->st_size && a->st_nlink == b->st_nlink &&
           am.tv_sec == bm.tv_sec && am.tv_nsec == bm.tv_nsec &&
           ac.tv_sec == bc.tv_sec && ac.tv_nsec == bc.tv_nsec;
}

#ifdef __linux__
/* Linux stores ACLs as xattrs too. A metadata copy failure must abort the
 * replacement, not silently remove access rules from the edited file. */
static int agent_copy_file_xattrs(int src, int dst) {
    if (fremovexattr(dst, "system.posix_acl_access") != 0 &&
        errno != ENODATA && errno != ENOTSUP) return -1;
    ssize_t count = flistxattr(src, NULL, 0);
    if (count < 0) return errno == ENOTSUP ? 0 : -1;
    if (!count) return 0;
    char *names = xmalloc((size_t)count);
    count = flistxattr(src, names, (size_t)count);
    int rc = -1, saved_errno;
    if (count < 0) goto done;
    for (ssize_t pos = 0; pos < count;) {
        const char *name = names + pos;
        ssize_t len = fgetxattr(src, name, NULL, 0);
        if (len < 0) goto done;
        void *value = xmalloc(len ? (size_t)len : 1);
        ssize_t readlen = fgetxattr(src, name, value, (size_t)len);
        int copied = readlen < 0 ? -1 : fsetxattr(dst, name, value, (size_t)readlen, 0);
        int saved_errno = errno;
        free(value);
        errno = saved_errno;
        if (copied != 0) goto done;
        pos += (ssize_t)strlen(name) + 1;
    }
    rc = 0;
done:
    saved_errno = errno;
    free(names);
    errno = saved_errno;
    return rc;
}
#endif

/* Commit only complete files. Follow existing symlinks, but refuse hard links:
 * an atomic rename cannot preserve their shared-inode semantics. The version
 * checks catch concurrent edits, though unrelated writers do not take a lock. */
int agent_tools_file_replace_file(const char *path, const char *data, size_t len,
                                  const char *expected, size_t expected_len,
                                  char *err, size_t errlen) {
    char target[PATH_MAX], temp[PATH_MAX] = "";
    struct stat before, after;
    int src = -1, fd = -1, rc = -1;
    bool exists = lstat(path, &before) == 0;
    if (!exists && errno != ENOENT) goto failed;
    if (exists) {
        if (!realpath(path, target)) goto failed;
        src = open(target, O_RDWR | O_NONBLOCK | O_NOFOLLOW);
        if (src < 0 || fstat(src, &before) != 0) goto failed;
        if (!S_ISREG(before.st_mode) || before.st_nlink != 1) {
            snprintf(err, errlen, "refusing to replace non-regular or hard-linked file: %s", path);
            goto done;
        }
        if (expected) {
            if (before.st_size < 0 || (uintmax_t)before.st_size != expected_len)
                goto changed;
            char buf[8192];
            size_t pos = 0;
            while (pos < expected_len) {
                size_t count = expected_len - pos;
                if (count > sizeof(buf)) count = sizeof(buf);
                ssize_t n = read(src, buf, count);
                if (n < 0 && errno == EINTR) continue;
                if (n < 0) goto failed;
                if (!n || memcmp(buf, expected + pos, (size_t)n)) goto changed;
                pos += (size_t)n;
            }
        }
    } else {
        if (expected) goto changed;
        if (snprintf(target, sizeof(target), "%s", path) >= (int)sizeof(target)) {
            errno = ENAMETOOLONG;
            goto failed;
        }
    }
    if (snprintf(temp, sizeof(temp), "%s.ds4-XXXXXX", target) >= (int)sizeof(temp)) {
        temp[0] = 0;
        errno = ENAMETOOLONG;
        goto failed;
    }
    fd = mkstemp(temp);
    if (fd < 0) { temp[0] = 0; goto failed; }
    if (!exists) {
        /* Let open apply the process umask without changing it in this thread. */
        if (unlink(temp) != 0) goto failed;
        close(fd);
        fd = open(temp, O_WRONLY | O_CREAT | O_EXCL, 0666);
        if (fd < 0) { temp[0] = 0; goto failed; }
    }
    for (size_t pos = 0; pos < len;) {
        ssize_t n = write(fd, data + pos, len - pos);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) { if (!n) errno = EIO; goto failed; }
        pos += (size_t)n;
    }
    if (exists) {
        if (fchown(fd, before.st_uid, before.st_gid) != 0) goto failed;
        if (fchmod(fd, before.st_mode & 07777) != 0) goto failed;
#ifdef __APPLE__
        if (fcopyfile(src, fd, NULL, COPYFILE_ACL | COPYFILE_XATTR) != 0) goto failed;
#elif defined(__linux__)
        if (agent_copy_file_xattrs(src, fd) != 0) goto failed;
#endif
    }
    if (fsync(fd) != 0) goto failed;
    if (close(fd) != 0) { fd = -1; goto failed; }
    fd = -1;
    if (exists) {
        char resolved[PATH_MAX];
        if (!realpath(path, resolved) || strcmp(resolved, target) ||
            stat(target, &after) != 0 || !agent_same_file_version(&before, &after))
            goto changed;
        if (rename(temp, target) != 0) goto failed;
    } else {
        /* Unlike rename, link will not overwrite a concurrently created file. */
        if (link(temp, target) != 0) goto failed;
        unlink(temp);
    }
    temp[0] = 0;
    rc = 0;
    goto done;
changed:
    snprintf(err, errlen, "file changed while editing; read it again: %s", path);
    goto done;
failed:
    snprintf(err, errlen, "replace %s: %s", path, strerror(errno));
done:
    if (fd >= 0) close(fd);
    if (src >= 0) close(src);
    if (temp[0]) unlink(temp);
    return rc;
}

static void agent_edit_result_append_line(agent_buf *b, const char *data,
                                          const agent_line_span *sp,
                                          int line) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%d ", line);
    agent_buf_puts(b, prefix);
    agent_buf_append(b, data + sp->start, sp->content_end - sp->start);
    agent_buf_puts(b, "\n");
}

/* Successful edits return the nearby post-edit file shape.  This spends cheap
 * prefill tokens to save expensive model retries: the model immediately sees
 * shifted line numbers, braces, semicolons, and accidental duplication. */
static void agent_edit_result_append_context(agent_buf *b,
                                             const char *path,
                                             const char *data, size_t len,
                                             int anchor_start,
                                             int anchor_end) {
    enum {
        CONTEXT_BEFORE = 5,
        CONTEXT_AFTER = 8,
        EDITED_CONTEXT_HEAD = 18,
        EDITED_CONTEXT_TAIL = 18
    };

    agent_line_spans spans = {0};
    agent_split_lines(data, len, &spans);
    if (spans.len <= 0) {
        agent_line_spans_free(&spans);
        return;
    }

    if (anchor_start < 1) anchor_start = 1;
    if (anchor_start > spans.len) anchor_start = spans.len;
    if (anchor_end < anchor_start) anchor_end = anchor_start;
    if (anchor_end > spans.len) anchor_end = spans.len;

    int ctx_start = anchor_start - CONTEXT_BEFORE;
    if (ctx_start < 1) ctx_start = 1;
    int ctx_end = anchor_end + CONTEXT_AFTER;
    if (ctx_end > spans.len) ctx_end = spans.len;

    char hdr[PATH_MAX + 160];
    snprintf(hdr, sizeof(hdr),
             "Current file around edit: %s lines %d-%d of %d\n",
             path, ctx_start, ctx_end, spans.len);
    agent_buf_puts(b, hdr);

    int edited_lines = anchor_end - anchor_start + 1;
    if (edited_lines <= EDITED_CONTEXT_HEAD + EDITED_CONTEXT_TAIL) {
        for (int line = ctx_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    } else {
        int head_end = anchor_start + EDITED_CONTEXT_HEAD - 1;
        int tail_start = anchor_end - EDITED_CONTEXT_TAIL + 1;
        for (int line = ctx_start; line <= head_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
        snprintf(hdr, sizeof(hdr),
                 "... %d edited lines omitted ...\n",
                 tail_start - head_end - 1);
        agent_buf_puts(b, hdr);
        for (int line = tail_start; line <= ctx_end; line++)
            agent_edit_result_append_line(b, data, &spans.v[line - 1], line);
    }

    agent_line_spans_free(&spans);
}

/* ============================================================================
 * Old-text anchor matching (edit)
 * ========================================================================== */

static const char *agent_memmem_simple(const char *hay, size_t hay_len,
                                       const char *needle, size_t needle_len) {
    if (!needle_len) return hay;
    if (needle_len > hay_len) return NULL;
    size_t last = hay_len - needle_len;
    for (size_t i = 0; i <= last; i++) {
        if (hay[i] == needle[0] && !memcmp(hay + i, needle, needle_len))
            return hay + i;
    }
    return NULL;
}

static bool agent_find_unique(const char *data, size_t len,
                              const char *needle, size_t needle_len,
                              const char **match, const char *label,
                              char *err, size_t err_len) {
    if (!needle || needle_len == 0) {
        snprintf(err, err_len, "%s anchor is empty", label);
        return false;
    }
    const char *first = agent_memmem_simple(data, len, needle, needle_len);
    if (!first) {
        snprintf(err, err_len, "%s anchor not found", label);
        return false;
    }
    size_t after_first = (size_t)(first - data) + 1;
    const char *second = after_first <= len ?
        agent_memmem_simple(data + after_first, len - after_first,
                            needle, needle_len) : NULL;
    if (second) {
        snprintf(err, err_len, "%s anchor is not unique", label);
        return false;
    }
    *match = first;
    return true;
}

/* Find an anchor only in the suffix after start.
 *
 * Anchored edits use "head [upto] tail": the head fixes the edit start, and
 * the tail should delimit the first unique end point after that start. A tail
 * may legitimately appear earlier in the file, so checking global uniqueness
 * would reject valid edits.
 */
static bool agent_find_unique_after(const char *data, size_t len,
                                    const char *start,
                                    const char *needle, size_t needle_len,
                                    const char **match, const char *label,
                                    char *err, size_t err_len) {
    if (!needle || needle_len == 0) {
        snprintf(err, err_len, "%s anchor is empty", label);
        return false;
    }
    if (start < data || start > data + len) {
        snprintf(err, err_len, "%s search starts outside file", label);
        return false;
    }
    size_t off = (size_t)(start - data);
    const char *first = agent_memmem_simple(data + off, len - off,
                                            needle, needle_len);
    if (!first) {
        snprintf(err, err_len, "%s anchor not found after old head", label);
        return false;
    }
    size_t after_first = (size_t)(first - data) + 1;
    const char *second = after_first <= len ?
        agent_memmem_simple(data + after_first, len - after_first,
                            needle, needle_len) : NULL;
    if (second) {
        snprintf(err, err_len, "%s anchor is not unique after old head", label);
        return false;
    }
    *match = first;
    return true;
}

static bool agent_span_has_nonspace(const char *s, size_t len) {
    for (size_t i = 0; i < len; i++) {
        if (!isspace((unsigned char)s[i])) return true;
    }
    return false;
}

/* The client checks exact-old uniqueness at edit execution time; the
 * mid-generation edit-old preflight (server-side forcer) is dropped.  The
 * [upto] marker is still honored here when the CLI enables edit_upto. */
bool agent_tools_file_edit_find_old_span(const char *data, size_t len,
                                         const char *old, bool allow_upto,
                                         const char **match,
                                         size_t *match_len, bool *anchored,
                                         char *err, size_t err_len) {
    static const char marker[] = "[upto]";
    size_t old_len = strlen(old);
    const char *upto = strstr(old, marker);
    if (!allow_upto || !upto) {
        *anchored = false;
        if (!agent_find_unique(data, len, old, old_len, match, "old text",
                               err, err_len))
            return false;
        *match_len = old_len;
        return true;
    }
    if (strstr(upto + strlen(marker), marker)) {
        snprintf(err, err_len, "old text contains more than one [upto] marker");
        return false;
    }
    size_t head_len = (size_t)(upto - old);
    const char *tail = upto + strlen(marker);
    size_t tail_len = old_len - head_len - strlen(marker);
    /* Strip leading newline/CR from tail before searching.  The head already
     * includes the newline at its end, so the extra \n that follows [upto] in
     * the old text (whether injected by the forcer or written by the model)
     * must not be part of the tail needle -- the file after the head has no
     * duplicate newline. */
    while (tail_len > 0 && (*tail == '\n' || *tail == '\r')) {
        tail++;
        tail_len--;
    }
    if (!agent_span_has_nonspace(tail, tail_len)) {
        snprintf(err, err_len,
                 "old text after [upto] must include a unique tail anchor");
        return false;
    }
    const char *head_pos = NULL;
    const char *tail_pos = NULL;
    if (!agent_find_unique(data, len, old, head_len, &head_pos, "old head",
                           err, err_len))
        return false;
    if (!agent_find_unique_after(data, len, head_pos + head_len,
                                 tail, tail_len, &tail_pos, "old tail",
                                 err, err_len))
        return false;
    *anchored = true;
    *match = head_pos;
    *match_len = (size_t)(tail_pos - head_pos) + tail_len;
    return true;
}

/* ============================================================================
 * Read / more
 * ========================================================================== */

static void agent_worker_set_more(agent_tools_worker *w, const char *path,
                                  int next_line, bool bare) {
    if (path != w->more_path)
        snprintf(w->more_path, sizeof(w->more_path), "%s", path ? path : "");
    w->more_next_line = next_line;
    w->more_bare = bare;
    w->more_valid = path && path[0] && next_line > 0;
    w->more_byte_offset = 0;
    w->more_mid_line = false;
}

static int agent_read_default_lines(agent_tools_worker *w) {
    int ctx = w->ctx_size;
    if (ctx > 0 && ctx <= AGENT_READ_SMALL_CONTEXT_MAX)
        return AGENT_READ_DEFAULT_LINES_SMALL;
    if (ctx > 0 && ctx <= AGENT_READ_MEDIUM_CONTEXT_MAX)
        return AGENT_READ_DEFAULT_LINES_MEDIUM;
    return AGENT_READ_DEFAULT_LINES_LARGE;
}

/* Read file text for the model.  Normal mode shows plain line numbers.  Raw
 * mode is reserved for cases where line decoration would corrupt the payload
 * being inspected. */
static char *agent_read_range_from(agent_tools_worker *w, const char *path, int start_line,
                                   int max_lines, bool whole_file, bool bare,
                                   bool set_more, off_t offset, bool mid_line) {
    if (!path || !path[0]) return xstrdup("Tool error: read requires path\n");
    agent_buf out = {0}, body = {0};
    FILE *fp = agent_open_regular_file(path);
    if (!fp) goto failed;
    struct stat st;
    if (fstat(fileno(fp), &st) != 0) goto failed;
    if (!S_ISREG(st.st_mode)) { errno = EINVAL; goto failed; }
    if (start_line < 1) start_line = 1;
    if (max_lines <= 0) max_lines = agent_read_default_lines(w);
    int c, line = 1, lines = 0;
    if (offset > 0) {
        if (fseeko(fp, offset, SEEK_SET) != 0) goto failed;
        line = start_line;
    }
    while (line < start_line && (c = fgetc(fp)) != EOF) {
        if (c == '\r') {
            int next = fgetc(fp);
            if (next != '\n' && next != EOF) ungetc(next, fp);
        }
        if (c == '\n' || c == '\r') line++;
    }
    bool beginning = !mid_line, prefix = true;
    int last_line = 0;
    while ((whole_file || lines < max_lines) && (c = fgetc(fp)) != EOF) {
        if (body.len >= AGENT_TOOL_MAX_BYTES - 4096 &&
            ((c & 0xc0) != 0x80 || body.len >= AGENT_TOOL_MAX_BYTES - 4096 + 3)) {
            ungetc(c, fp);
            break;
        }
        if (!c) {
            agent_buf_puts(&out, "Tool error: read encountered binary data\n");
            goto done;
        }
        last_line = line;
        if (prefix && !bare) {
            char label[80];
            snprintf(label, sizeof(label), "%d%s ", line, beginning ? "" : " (continued)");
            agent_buf_puts(&body, label);
        }
        prefix = false;
        beginning = false;
        char ch = (char)c;
        if (c == '\r') {
            int next = fgetc(fp);
            if (bare) agent_buf_append(&body, &ch, 1);
            if (next == '\n') {
                ch = '\n';
            } else {
                if (next != EOF) ungetc(next, fp);
                if (bare) ch = 0;
                else ch = '\n';
            }
        }
        if (ch) agent_buf_append(&body, &ch, 1);
        if (c == '\n' || c == '\r') {
            if (line == INT_MAX) { errno = EOVERFLOW; goto failed; }
            line++;
            lines++;
            beginning = prefix = true;
        }
    }
    if (ferror(fp)) goto failed;
    off_t next_offset = ftello(fp);
    if (next_offset < 0) goto failed;
    c = fgetc(fp);
    bool more = c != EOF;
    if (ferror(fp)) goto failed;
    if (whole_file && more) {
        agent_buf_puts(&out, "Tool error: whole read exceeds the 128 KiB output limit; use read and more for chunks\n");
        goto done;
    }
    if (!bare) {
        char header[PATH_MAX + 128];
        snprintf(header, sizeof(header), "%s: lines %d-%d%s\n", path,
                 last_line ? start_line : 0, last_line, more ? " (partial read)" : " (end of file)");
        agent_buf_puts(&out, header);
    }
    agent_buf_append(&out, body.ptr, body.len);
    if (more) {
        char note[256];
        snprintf(note, sizeof(note), "\n[Read truncated. continue_offset=%d; continue_byte_offset=%lld%s. Call more to continue.]\n",
                 line, (long long)next_offset, beginning ? "" : " (within line)");
        agent_buf_puts(&out, note);
    }
    if (set_more) {
        agent_worker_set_more(w, more ? path : NULL, more ? line : 0, bare);
        w->more_byte_offset = next_offset;
        w->more_mid_line = !beginning;
    }
    free(body.ptr);
    fclose(fp);
    return agent_buf_take(&out);
failed:
    agent_buf_puts(&out, "Tool error: read failed: ");
    agent_buf_puts(&out, strerror(errno));
    agent_buf_puts(&out, "\n");
done:
    if (fp) fclose(fp);
    free(body.ptr);
    if (set_more) agent_worker_set_more(w, NULL, 0, false);
    return agent_buf_take(&out);
}

char *agent_tools_file_read_range(agent_tools_worker *w, const char *path, int start_line,
                                  int max_lines, bool whole_file, bool bare,
                                  bool set_more) {
    return agent_read_range_from(w, path, start_line, max_lines, whole_file,
                                 bare, set_more, 0, false);
}

char *agent_tools_file_read(agent_tools_worker *w, const proto_tool_call *call) {
    const char *path = agent_tools_arg_value(call, "path");
    bool whole = agent_tools_parse_bool_default(agent_tools_arg_value(call, "whole"), false);
    int start = agent_tools_parse_int_default(agent_tools_arg_value(call, "start_line"),
                                              1, 1, INT_MAX);
    int count = agent_tools_parse_int_default(agent_tools_arg_value(call, "max_lines"),
                                              agent_read_default_lines(w), 1, INT_MAX);
    bool raw = agent_tools_parse_bool_default(agent_tools_arg_value(call, "raw"), false);
    return agent_tools_file_read_range(w, path, start, count, whole, raw, true);
}

char *agent_tools_file_more(agent_tools_worker *w, const proto_tool_call *call) {
    int count = agent_tools_parse_int_default(agent_tools_arg_value(call, "count"),
                                              agent_read_default_lines(w), 1, INT_MAX);
    if (!w->more_valid) return xstrdup("Tool error: no previous output to continue\n");
    return agent_read_range_from(w, w->more_path, w->more_next_line, count, false,
                                 w->more_bare, true, w->more_byte_offset, w->more_mid_line);
}

char *agent_tools_file_write(agent_tools_worker *w, const proto_tool_call *call) {
    (void)w;
    const char *path = agent_tools_arg_value(call, "path");
    const char *content = agent_tools_arg_value(call, "content");
    if (!path || !path[0]) return xstrdup("Tool error: write requires path\n");
    if (!content) return xstrdup("Tool error: write requires content\n");
    size_t len = strlen(content);
    char err[256];
    if (agent_tools_file_replace_file(path, content, len, NULL, 0, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    char msg[PATH_MAX + 160];
    snprintf(msg, sizeof(msg), "Wrote %zu bytes to %s\n", len, path);
    return xstrdup(msg);
}

char *agent_tools_file_list(const proto_tool_call *call) {
    const char *path = agent_tools_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    DIR *dir = opendir(path);
    if (!dir) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: opendir failed: ");
        agent_buf_puts(&b, strerror(errno));
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }
    agent_buf out = {.limit = AGENT_TOOL_MAX_BYTES};
    char hdr[PATH_MAX + 64];
    snprintf(hdr, sizeof(hdr), "%s:\n", path);
    agent_buf_puts(&out, hdr);
    struct dirent *de;
    int shown = 0;
    while ((de = readdir(dir)) != NULL && shown < 300) {
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        char full[PATH_MAX];
        snprintf(full, sizeof(full), "%s/%s", path, de->d_name);
        struct stat st;
        if (lstat(full, &st) != 0) continue;
        char type = S_ISDIR(st.st_mode) ? 'd' :
                    S_ISLNK(st.st_mode) ? 'l' :
                    S_ISREG(st.st_mode) ? '-' : '?';
        char line[PATH_MAX + 96];
        snprintf(line, sizeof(line), "%c %10lld %s%s\n", type,
                 (long long)st.st_size, de->d_name, S_ISDIR(st.st_mode) ? "/" : "");
        agent_buf_puts(&out, line);
        shown++;
    }
    if (de) agent_buf_puts(&out, "... more entries omitted ...\n");
    closedir(dir);
    return agent_buf_take(&out);
}

/* ============================================================================
 * Edit
 * ========================================================================== */

static char *agent_apply_file_splice(const char *path,
                                     const char *data, size_t len,
                                     size_t offset, size_t remove_len,
                                     const char *insert, const char *kind) {
    char err[256];
    if (!insert) insert = "";
    size_t insert_len = strlen(insert);
    size_t out_len = offset + insert_len + (len - offset - remove_len);
    char *out = xmalloc(out_len + 1);
    memcpy(out, data, offset);
    memcpy(out + offset, insert, insert_len);
    memcpy(out + offset + insert_len, data + offset + remove_len,
           len - offset - remove_len);
    out[out_len] = '\0';

    int rc = agent_tools_file_replace_file(path, out, out_len, data, len, err, sizeof(err));
    if (rc != 0) {
        free(out);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    int start_line = 0, end_line = 0, delta = 0;
    agent_old_new_line_effect(data, len, out, out_len, offset, remove_len,
                              &start_line, &end_line, &delta);
    char *result = agent_edit_result(path, start_line, end_line, delta,
                                     out, out_len, kind);
    free(out);
    return result;
}

char *agent_tools_file_edit(agent_tools_worker *w, const proto_tool_call *call) {
    const char *path = agent_tools_arg_value(call, "path");
    if (!path || !path[0]) return xstrdup("Tool error: edit requires path\n");
    const char *old = agent_tools_arg_value(call, "old");
    const char *new_text = agent_tools_arg_value(call, "new");
    if (!old || !old[0]) return xstrdup("Tool error: edit requires non-empty old text\n");
    if (!new_text) return xstrdup("Tool error: edit requires new text\n");

    char err[256];
    char *data = NULL;
    size_t len = 0;
    if (agent_tools_file_read_file_bytes(path, &data, &len, err, sizeof(err)) != 0) {
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    bool allow_upto = w && w->edit_upto;
    if (!agent_tools_file_edit_find_old_span(data, len, old, allow_upto,
                                  &match, &match_len,
                                  &anchored, err, sizeof(err)))
    {
        free(data);
        agent_buf b = {0};
        agent_buf_puts(&b, "Tool error: ");
        agent_buf_puts(&b, err);
        agent_buf_puts(&b, "\n");
        return agent_buf_take(&b);
    }

    char *result = agent_apply_file_splice(path, data, len,
                                           (size_t)(match - data), match_len,
                                           new_text,
                                           anchored ? "anchored old/new replacement"
                                                    : "old/new replacement");
    free(data);
    return result;
}

/* ============================================================================
 * Search
 * ========================================================================== */

typedef struct {
    const char *query;
    const char *glob;
    regex_t regex;
    bool use_regex;
    bool regex_ready;
    bool case_sensitive;
    int context;
    int max_results;
    int results;
    size_t skipped;
    char first_skip[PATH_MAX + 128];
    agent_buf out;
} agent_search_ctx;

static bool agent_literal_match(const char *s, size_t n, const char *q,
                                bool case_sensitive) {
    size_t qn = strlen(q);
    if (!qn) return true;
    if (qn > n) return false;
    for (size_t i = 0; i + qn <= n; i++) {
        bool ok = true;
        for (size_t j = 0; j < qn; j++) {
            unsigned char a = (unsigned char)s[i + j];
            unsigned char b = (unsigned char)q[j];
            if (!case_sensitive) {
                a = (unsigned char)tolower(a);
                b = (unsigned char)tolower(b);
            }
            if (a != b) {
                ok = false;
                break;
            }
        }
        if (ok) return true;
    }
    return false;
}

static bool agent_search_line_matches(agent_search_ctx *ctx, const char *s, size_t n) {
    if (ctx->use_regex) {
        char *line = xstrndup(s, n);
        int rc = regexec(&ctx->regex, line, 0, NULL, 0);
        free(line);
        return rc == 0;
    }
    return agent_literal_match(s, n, ctx->query, ctx->case_sensitive);
}

static void agent_search_emit_line(agent_search_ctx *ctx, const char *data,
                                   int line_no) {
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "  %d ", line_no);
    agent_buf_puts(&ctx->out, prefix);
    agent_buf_puts(&ctx->out, data);
    agent_buf_puts(&ctx->out, "\n");
}

static void agent_search_skip(agent_search_ctx *ctx, const char *path, const char *why) {
    ctx->skipped++;
    if (!ctx->first_skip[0])
        snprintf(ctx->first_skip, sizeof(ctx->first_skip), "%s: %s", path, why);
}

/* A file can be arbitrarily large; only the current line and a small context
 * ring are resident. Oversized or binary lines are reported, never ignored. */
static int agent_search_read_line(FILE *fp, char **text) {
    agent_buf line = {0};
    int c;
    while ((c = fgetc(fp)) != EOF) {
        if (!c || line.len == AGENT_TOOL_MAX_BYTES) {
            free(line.ptr);
            return -2;
        }
        if (c == '\n' || c == '\r') {
            if (c == '\r') {
                int next = fgetc(fp);
                if (next != '\n' && next != EOF) ungetc(next, fp);
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&line, &ch, 1);
    }
    if (ferror(fp)) { free(line.ptr); return -1; }
    if (c == EOF && !line.len) { free(line.ptr); return 0; }
    *text = agent_buf_take(&line);
    return 1;
}

/* Search one text file and emit matching lines with plain line numbers. */
static void agent_search_file(agent_search_ctx *ctx, const char *path) {
    if (ctx->results >= ctx->max_results || ctx->out.truncated) return;
    if (ctx->glob && ctx->glob[0]) {
        const char *base = strrchr(path, '/');
        base = base ? base + 1 : path;
        if (fnmatch(ctx->glob, base, 0) != 0 && fnmatch(ctx->glob, path, 0) != 0)
            return;
    }
    FILE *fp = agent_open_regular_file(path);
    if (!fp) { agent_search_skip(ctx, path, strerror(errno)); return; }
    char *ring[6] = {0};
    bool printed_file = false;
    int line = 0, last_emitted = 0, after = 0;
    while ((ctx->results < ctx->max_results || after) && !ctx->out.truncated) {
        char *text = NULL;
        int rc = agent_search_read_line(fp, &text);
        if (!rc) break;
        if (rc < 0) {
            agent_search_skip(ctx, path, rc == -2 ? "binary data or line exceeds 128 KiB" : strerror(errno));
            break;
        }
        if (line == INT_MAX) {
            free(text);
            agent_search_skip(ctx, path, "line number limit reached");
            break;
        }
        line++;
        free(ring[line % 6]);
        ring[line % 6] = text;
        bool match = ctx->results < ctx->max_results &&
                     agent_search_line_matches(ctx, text, strlen(text));
        if (match && !printed_file) {
            agent_buf_puts(&ctx->out, path);
            agent_buf_puts(&ctx->out, "\n");
            printed_file = true;
        }
        if (match) {
            int from = line - ctx->context;
            if (from <= last_emitted) from = last_emitted + 1;
            for (int j = from; j <= line; j++)
                agent_search_emit_line(ctx, ring[j % 6], j);
            last_emitted = line;
            after = ctx->context;
            ctx->results++;
        } else if (after) {
            agent_search_emit_line(ctx, text, line);
            last_emitted = line;
            after--;
        }
    }
    if (printed_file) agent_buf_puts(&ctx->out, "\n");
    for (int i = 0; i < 6; i++) free(ring[i]);
    fclose(fp);
}

/* Recursively search a file or directory, avoiding .git and stopping once the
 * result cap is reached. */
static void agent_search_path(agent_search_ctx *ctx, const char *path, int depth) {
    if (ctx->results >= ctx->max_results || ctx->out.truncated) return;
    if (depth > 24) { agent_search_skip(ctx, path, "directory depth limit reached"); return; }
    struct stat st;
    if ((depth == 0 ? stat(path, &st) : lstat(path, &st)) != 0) {
        agent_search_skip(ctx, path, strerror(errno));
        return;
    }
    if (S_ISREG(st.st_mode)) {
        agent_search_file(ctx, path);
        return;
    }
    if (!S_ISDIR(st.st_mode)) {
        agent_search_skip(ctx, path, "not a regular file or directory (nested symlinks are not followed)");
        return;
    }
    DIR *dir = opendir(path);
    if (!dir) { agent_search_skip(ctx, path, strerror(errno)); return; }
    struct dirent *de;
    while (ctx->results < ctx->max_results && !ctx->out.truncated) {
        errno = 0;
        de = readdir(dir);
        if (!de) {
            if (errno) agent_search_skip(ctx, path, strerror(errno));
            break;
        }
        if (!strcmp(de->d_name, ".") || !strcmp(de->d_name, "..")) continue;
        if (!strcmp(de->d_name, ".git")) continue;
        char child[PATH_MAX];
        if (snprintf(child, sizeof(child), "%s/%s", path, de->d_name) >= (int)sizeof(child)) {
            agent_search_skip(ctx, path, "path exceeds PATH_MAX");
            continue;
        }
        agent_search_path(ctx, child, depth + 1);
    }
    closedir(dir);
}

char *agent_tools_file_search(agent_tools_worker *w, const proto_tool_call *call) {
    (void)w;
    const char *query = agent_tools_arg_value(call, "query");
    if (!query || !query[0]) return xstrdup("Tool error: search requires query\n");
    const char *path = agent_tools_arg_value(call, "path");
    if (!path || !path[0]) path = ".";
    const char *mode = agent_tools_arg_value(call, "mode");
    if (mode && strcmp(mode, "literal") && strcmp(mode, "regex"))
        return xstrdup("Tool error: search mode must be literal or regex (POSIX extended)\n");
    struct stat root;
    if (stat(path, &root) != 0 || (!S_ISDIR(root.st_mode) && !S_ISREG(root.st_mode))) {
        agent_buf error = {0};
        agent_buf_puts(&error, "Tool error: search path is missing, unreadable, or not a regular file/directory: ");
        agent_buf_puts(&error, path);
        agent_buf_puts(&error, "\n");
        return agent_buf_take(&error);
    }
    agent_search_ctx ctx = {
        .query = query,
        .glob = agent_tools_arg_value(call, "glob"),
        .use_regex = mode && !strcmp(mode, "regex"),
        .case_sensitive = agent_tools_parse_bool_default(agent_tools_arg_value(call, "case_sensitive"), true),
        .context = agent_tools_parse_int_default(agent_tools_arg_value(call, "context"), 0, 0, 5),
        .max_results = agent_tools_parse_int_default(agent_tools_arg_value(call, "max_results"), 50, 1, 500),
        .out = {.limit = AGENT_TOOL_MAX_BYTES},
    };
    if (ctx.use_regex) {
        int flags = REG_EXTENDED | REG_NOSUB;
        if (!ctx.case_sensitive) flags |= REG_ICASE;
        int rc = regcomp(&ctx.regex, query, flags);
        if (rc != 0) {
            char msg[256];
            regerror(rc, &ctx.regex, msg, sizeof(msg));
            agent_buf b = {0};
            agent_buf_puts(&b, "Tool error: invalid regex: ");
            agent_buf_puts(&b, msg);
            agent_buf_puts(&b, "\n");
            return agent_buf_take(&b);
        }
        ctx.regex_ready = true;
    }
    agent_search_path(&ctx, path, 0);
    if (ctx.regex_ready) regfree(&ctx.regex);
    if (!ctx.out.ptr) agent_buf_puts(&ctx.out, "No matches in searched text\n");
    else {
        char hdr[96];
        snprintf(hdr, sizeof(hdr), "%d match%s shown\n\n",
                 ctx.results, ctx.results == 1 ? "" : "es");
        size_t hdr_len = strlen(hdr);
        if (ctx.out.len + hdr_len + 1 > ctx.out.cap) {
            ctx.out.cap = ctx.out.len + hdr_len + 1;
            ctx.out.ptr = xrealloc(ctx.out.ptr, ctx.out.cap);
        }
        memmove(ctx.out.ptr + hdr_len, ctx.out.ptr, ctx.out.len + 1);
        memcpy(ctx.out.ptr, hdr, hdr_len);
        ctx.out.len += hdr_len;
    }
    ctx.out.limit = 0;
    if (ctx.skipped || ctx.results >= ctx.max_results) {
        bool truncated = ctx.out.truncated;
        ctx.out.truncated = false;
        char note[PATH_MAX + 256];
        snprintf(note, sizeof(note), "\nSearch incomplete: %zu skipped paths%s. %s\n",
                 ctx.skipped, ctx.results >= ctx.max_results ? "; match limit reached" : "", ctx.first_skip);
        agent_buf_puts(&ctx.out, note);
        ctx.out.truncated = truncated;
    }
    return agent_buf_take(&ctx.out);
}
