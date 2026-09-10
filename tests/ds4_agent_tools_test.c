/*
 * Unit test for the client-side tool framework (Task 4): file tools
 * (read/more/write/list/edit/search), asynchronous bash jobs, and the edit
 * [upto] anchor logic.  Ported from tests/ds4_agent_test.c, which included
 * the monolith source; this harness links the compiled per-group tool objects
 * instead (they are external), so it must NOT include the .c files.
 *
 * Build:
 *   cc -O2 -Wall -Wextra -std=c99 -D_GNU_SOURCE -I. -pthread \
 *     -o tests/ds4_agent_tools_test \
 *     tests/ds4_agent_tools_test.c \
 *     agent/ds4_agent_tools_file.c agent/ds4_agent_tools_bash.c \
 *     agent/ds4_agent_tools_web.c agent/ds4_agent_tools_dispatch.c ds4_web.o
 * Run: ./tests/ds4_agent_tools_test
 */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#if defined(__APPLE__) || defined(__linux__)
#include <sys/xattr.h>
#endif

#include "../agent/ds4_agent_tools.h"

static int agent_test_failures;

static void agent_test_assert(bool cond, const char *expr,
                              const char *file, int line) {
    if (cond) return;
    fprintf(stderr, "%s:%d: assertion failed: %s\n", file, line, expr);
    agent_test_failures++;
}

#define AGENT_TEST_ASSERT(expr) \
    agent_test_assert((expr), #expr, __FILE__, __LINE__)

static void test_tool_arg(proto_tool_call *call, const char *name, const char *value) {
    proto_tool_call_add_arg(call, name, value, strlen(value), true);
}

static int test_write_file(const char *path, const char *data, size_t len,
                           char *err, size_t errlen) {
    return agent_tools_file_replace_file(path, data, len, NULL, 0, err, errlen);
}

static void test_atomic_file_tools(void) {
    char dir[] = "/tmp/ds4-agent-files-XXXXXX";
    AGENT_TEST_ASSERT(mkdtemp(dir) != NULL);
    char path[PATH_MAX], linkpath[PATH_MAX], err[256];
    snprintf(path, sizeof(path), "%s/file", dir);
    snprintf(linkpath, sizeof(linkpath), "%s/link", dir);
    char original[4096];
    memset(original, 'x', sizeof(original));
    AGENT_TEST_ASSERT(test_write_file(path, original, sizeof(original), err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(chmod(path, 0751) == 0);
#ifdef __APPLE__
    AGENT_TEST_ASSERT(setxattr(path, "com.ds4.agent-test", "keep", 4, 0, 0) == 0);
#elif defined(__linux__)
    AGENT_TEST_ASSERT(setxattr(path, "user.ds4-agent-test", "keep", 4, 0) == 0);
#endif

    pid_t child = fork();
    AGENT_TEST_ASSERT(child >= 0);
    if (child == 0) {
        signal(SIGXFSZ, SIG_IGN);
        struct rlimit limit = {64, 64};
        if (setrlimit(RLIMIT_FSIZE, &limit)) _exit(2);
        int rc = agent_tools_file_replace_file(path, original, sizeof(original),
                                               original, sizeof(original),
                                               err, sizeof(err));
        _exit(rc == -1 ? 0 : 3);
    }
    int status = 0;
    if (child > 0) waitpid(child, &status, 0);
    AGENT_TEST_ASSERT(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    char *data = NULL;
    size_t len = 0;
    AGENT_TEST_ASSERT(agent_tools_file_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(len == sizeof(original) && !memcmp(data, original, len));
    free(data);
    AGENT_TEST_ASSERT(agent_tools_file_replace_file(path, "bad", 3, "stale", 5, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "changed") != NULL);

    AGENT_TEST_ASSERT(symlink("file", linkpath) == 0);
    AGENT_TEST_ASSERT(test_write_file(linkpath, "new", 3, err, sizeof(err)) == 0);
    struct stat st;
    AGENT_TEST_ASSERT(lstat(linkpath, &st) == 0 && S_ISLNK(st.st_mode));
    AGENT_TEST_ASSERT(stat(path, &st) == 0 && (st.st_mode & 0777) == 0751);
    AGENT_TEST_ASSERT(st.st_uid == getuid());
#ifdef __APPLE__
    char attribute[16];
    AGENT_TEST_ASSERT(getxattr(path, "com.ds4.agent-test", attribute, sizeof(attribute), 0, 0) == 4);
    AGENT_TEST_ASSERT(!memcmp(attribute, "keep", 4));
#elif defined(__linux__)
    char attribute[16];
    AGENT_TEST_ASSERT(getxattr(path, "user.ds4-agent-test", attribute, sizeof(attribute)) == 4);
    AGENT_TEST_ASSERT(!memcmp(attribute, "keep", 4));
#endif
    AGENT_TEST_ASSERT(agent_tools_file_read_file_bytes(path, &data, &len, err, sizeof(err)) == 0);
    AGENT_TEST_ASSERT(len == 3 && !memcmp(data, "new", 3));
    free(data);
    unlink(linkpath);
    AGENT_TEST_ASSERT(link(path, linkpath) == 0);
    AGENT_TEST_ASSERT(test_write_file(path, "bad", 3, err, sizeof(err)) == -1);
    AGENT_TEST_ASSERT(strstr(err, "hard-linked") != NULL);
    unlink(linkpath);
    unlink(path);
    /* Failed replacements must not leave temporary files behind. */
    AGENT_TEST_ASSERT(rmdir(dir) == 0);

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = true;
    AGENT_TEST_ASSERT(agent_tools_file_edit_find_old_span("literal [upto] text", 19,
                         "[upto]", false, &match, &match_len, &anchored, err, sizeof(err)));
    AGENT_TEST_ASSERT(!anchored && match_len == 6 && !memcmp(match, "[upto]", 6));
}

static void test_streaming_file_tools(void) {
    char path[] = "/tmp/ds4-agent-read-XXXXXX";
    int fd = mkstemp(path);
    AGENT_TEST_ASSERT(fd >= 0);
    FILE *fp = fdopen(fd, "wb");
    /* Larger than the old whole-file cap, with matches at both ends. */
    fputs("needle first\r\n", fp);
    for (int i = 0; i < 1024 * 1024; i++) fputs("0123456789abcdef\n", fp);
    fputs("needle last\r", fp);
    fclose(fp);
    agent_tools_worker w = {0};
    char *text = agent_tools_file_read_range(&w, path, 1, 1, false, false, true);
    AGENT_TEST_ASSERT(strstr(text, "needle first") && w.more_valid);
    AGENT_TEST_ASSERT(w.more_next_line == 2 && w.more_byte_offset == 14);
    free(text);
    proto_tool_call more = {0};
    test_tool_arg(&more, "count", "1");
    text = agent_tools_file_more(&w, &more);
    AGENT_TEST_ASSERT(strstr(text, "2 0123456789abcdef"));
    free(text);
    proto_tool_call_free(&more);

    proto_tool_call call = {0};
    test_tool_arg(&call, "path", path);
    test_tool_arg(&call, "query", "needle");
    char linkpath[PATH_MAX];
    snprintf(linkpath, sizeof(linkpath), "%s-link", path);
    AGENT_TEST_ASSERT(symlink(path, linkpath) == 0);
    proto_tool_call linked = {0};
    test_tool_arg(&linked, "path", linkpath);
    test_tool_arg(&linked, "query", "needle");
    text = agent_tools_file_search(&w, &linked);
    AGENT_TEST_ASSERT(strstr(text, "2 matches") && strstr(text, "needle last"));
    free(text);
    proto_tool_call_free(&linked);
    unlink(linkpath);
    text = agent_tools_file_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "2 matches") && strstr(text, "needle last"));
    free(text);
    test_tool_arg(&call, "mode", "regexp");
    text = agent_tools_file_search(&w, &call);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:"));
    free(text);
    proto_tool_call_free(&call);

    fp = fopen(path, "wb");
    for (int i = 0; i < 256 * 1024; i++) fputc('x', fp);
    fclose(fp);
    size_t total = 0;
    text = agent_tools_file_read_range(&w, path, 1, 1, false, true, true);
    do {
        size_t n = strspn(text, "x");
        total += n;
        AGENT_TEST_ASSERT(n > 0 && n < AGENT_TOOL_MAX_BYTES);
        if (w.more_valid) AGENT_TEST_ASSERT(strstr(text, "Read truncated") != NULL);
        free(text);
        if (!w.more_valid) break;
        text = agent_tools_file_more(&w, &more);
    } while (total < 512 * 1024);
    AGENT_TEST_ASSERT(total == 256 * 1024 && !w.more_valid);
    text = agent_tools_file_read_range(&w, path, 1, INT_MAX, true, true, true);
    AGENT_TEST_ASSERT(strstr(text, "Tool error: whole read") && !w.more_valid);
    free(text);
    fp = fopen(path, "wb");
    for (int i = 0; i < 256 * 1024; i++) fputc(0x80, fp);
    fclose(fp);
    text = agent_tools_file_read_range(&w, path, 1, 1, false, true, true);
    AGENT_TEST_ASSERT(strlen(text) < AGENT_TOOL_MAX_BYTES && w.more_valid);
    free(text);
    unlink(path);
    AGENT_TEST_ASSERT(mkfifo(path, 0600) == 0);
    double started = now_sec();
    text = agent_tools_file_read_range(&w, path, 1, 1, false, false, true);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:") && now_sec() - started < 0.5);
    free(text);
    unlink(path);
    proto_tool_call call2 = {0};
    test_tool_arg(&call2, "path", path);
    test_tool_arg(&call2, "query", "needle");
    text = agent_tools_file_search(&w, &call2);
    AGENT_TEST_ASSERT(strstr(text, "Tool error:"));
    free(text);
    proto_tool_call_free(&call2);

    agent_buf b = {0};
    char *large = xmalloc(200000);
    memset(large, 'q', 200000);
    agent_buf_append(&b, large, 200000);
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(strlen(text) == 200000);
    free(text);
    b.limit = 100;
    agent_buf_append(&b, large, 200000);
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(strstr(text, "Output truncated") != NULL);
    free(large);
    free(text);
    b.limit = 3;
    agent_buf_puts(&b, "a\xe4\xb8\xad" "b");
    text = agent_buf_take(&b);
    AGENT_TEST_ASSERT(!strncmp(text, "a\n[Output truncated", 19));
    free(text);
}

static void test_background_jobs(void) {
    agent_tools_worker w = {0};
    char err[256], marker[] = "/tmp/ds4-agent-deadline-XXXXXX";
    int fd = mkstemp(marker);
    close(fd);
    unlink(marker);
    char cmd[PATH_MAX + 128];
    snprintf(cmd, sizeof(cmd), "sleep 2; printf late > %s", marker);
    agent_tools_bash_job *job = agent_tools_bash_start(&w, cmd, 1, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    /* Deliberately no status polling: this stands in for model generation. */
    usleep(2400000);
    AGENT_TEST_ASSERT(access(marker, F_OK) != 0);
    bool finished = false;
    char *obs = agent_tools_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "timed_out=1"));
    free(obs);
    unlink(agent_tools_bash_job_path(job));
    agent_tools_bash_remove_job(&w, job);

    job = agent_tools_bash_start(&w, "(sleep 0.1; printf descendant-output) &", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    usleep(350000);
    obs = agent_tools_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "descendant-output"));
    free(obs);
    unlink(agent_tools_bash_job_path(job));
    agent_tools_bash_remove_job(&w, job);

    job = agent_tools_bash_start(&w, "head -c 2097152 /dev/zero | tr '\\000' x", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    usleep(900000);
    obs = agent_tools_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "exit_status=0"));
    AGENT_TEST_ASSERT(agent_tools_bash_job_bytes(job) == 2097152);
    free(obs);
    obs = agent_tools_bash_observation(job, true, &finished);
    AGENT_TEST_ASSERT(strlen(obs) < AGENT_BASH_TAIL_BYTES + 2048);
    free(obs);
    unlink(agent_tools_bash_job_path(job));
    agent_tools_bash_remove_job(&w, job);

    job = agent_tools_bash_start(&w, "sleep 0.3; printf completed", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job != NULL);
    if (!job) goto done;
    char output_path[PATH_MAX];
    snprintf(output_path, sizeof(output_path), "%s", agent_tools_bash_job_path(job));
    proto_tool_call call = {0};
    call.name = xstrdup("bash_status");
    char id[32];
    snprintf(id, sizeof(id), "%d", agent_tools_bash_job_id(job));
    test_tool_arg(&call, "job", id);
    test_tool_arg(&call, "refresh_sec", "1");
    double start = now_sec();
    obs = agent_tools_dispatch(&w, &call);
    AGENT_TEST_ASSERT(now_sec() - start >= 0.2);
    AGENT_TEST_ASSERT(strstr(obs, "status=done") && strstr(obs, "completed"));
    AGENT_TEST_ASSERT(w.bash_jobs == NULL);
    free(obs);
    proto_tool_call_free(&call);
    unlink(output_path);

    /* Two monitors must make progress together, and stopping one must neither
     * block shutdown nor kill the other job's process group. */
    job = agent_tools_bash_start(&w, "sleep 30", 60, err, sizeof(err));
    agent_tools_bash_job *other = agent_tools_bash_start(&w, "sleep 0.2; printf independent", 5, err, sizeof(err));
    AGENT_TEST_ASSERT(job && other);
    if (!job || !other) goto done;
    start = now_sec();
    agent_tools_bash_signal(job, SIGKILL);
    unlink(agent_tools_bash_job_path(job));
    agent_tools_bash_remove_job(&w, job);
    AGENT_TEST_ASSERT(now_sec() - start < 2);
    while (agent_tools_bash_is_running(other) && now_sec() - start < 3) usleep(10000);
    obs = agent_tools_bash_observation(other, true, &finished);
    AGENT_TEST_ASSERT(finished && strstr(obs, "exit_status=0") && strstr(obs, "independent"));
    free(obs);
    unlink(agent_tools_bash_job_path(other));
    agent_tools_bash_remove_job(&w, other);
done:
    agent_tools_bash_jobs_free(&w);
    unlink(marker);
}

static void test_agent_edit_upto_tail_newline_is_not_part_of_anchor(void) {
    const char *data =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "clean:\n"
        "\trm -f bc\n";
    const char *old =
        "CFLAGS = -Wall -Wextra -g\n"
        "LDFLAGS =\n"
        "\n"
        "all: bc\n"
        "\n"
        "bc: main.c\n"
        "\t$(CC) $(CFLAGS) -o bc main.c $(LDFLAGS)\n"
        "\n"
        "[upto]\n"
        "clean:\n";

    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};
    AGENT_TEST_ASSERT(!agent_tools_file_edit_find_old_span(data, strlen(data), old, false,
                                                           &match, &match_len, &anchored,
                                                           err, sizeof(err)));
    AGENT_TEST_ASSERT(strstr(err, "not found") != NULL);
    err[0] = '\0';
    AGENT_TEST_ASSERT(agent_tools_file_edit_find_old_span(data, strlen(data), old, true,
                                                          &match, &match_len, &anchored,
                                                          err, sizeof(err)));
    AGENT_TEST_ASSERT(anchored);
    AGENT_TEST_ASSERT(match == data);
    AGENT_TEST_ASSERT(match_len == strlen(data) - strlen("\trm -f bc\n"));
}

static void test_agent_edit_upto_requires_tail_after_newline_strip(void) {
    const char *data = "head\nbody\ntail\n";
    const char *old = "head\n[upto]\n";
    const char *match = NULL;
    size_t match_len = 0;
    bool anchored = false;
    char err[128] = {0};

    AGENT_TEST_ASSERT(!agent_tools_file_edit_find_old_span(data, strlen(data), old, true,
                                                           &match, &match_len, &anchored,
                                                           err, sizeof(err)));
    AGENT_TEST_ASSERT(strstr(err, "must include a unique tail anchor") != NULL);
}

int main(void) {
    test_atomic_file_tools();
    test_streaming_file_tools();
    test_background_jobs();
    test_agent_edit_upto_tail_newline_is_not_part_of_anchor();
    test_agent_edit_upto_requires_tail_after_newline_strip();
    if (agent_test_failures) {
        fprintf(stderr, "ds4_agent_tools_test: %d assertion(s) failed\n", agent_test_failures);
        return 1;
    }
    printf("ds4_agent_tools_test: all ok\n");
    return 0;
}
