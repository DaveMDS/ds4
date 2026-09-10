/* ds4_agent_tools_bash.c -- client-side asynchronous bash jobs: bash,
 * bash_status, bash_stop.  Ported from the monolithic ds4_agent.c
 * (agent_bash_* -> agent_tools_bash_*).  Pure process/pipe/file I/O, no engine,
 * no UI: the whole file is unit-testable.
 *
 * UI note: the monolith's agent_bash_publish_observation streamed bash output
 * to the terminal.  This port is UI-free; the observation text is returned by
 * dispatch and the client wires it to its UI.  agent_terminal_safe_text is
 * dropped with it.
 *
 * Build/link note: this is one object; include "ds4_agent_tools.h" from the
 * client and the test harness.
 */

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "ds4_agent_tools.h"

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* ============================================================================
 * Job structure
 * ========================================================================== */

struct agent_tools_bash_job {
    pthread_t thread;
    pthread_mutex_t mu;
    bool thread_started;
    int id;
    pid_t pid;
    int pipe_fd;
    int tmp_fd;
    char path[PATH_MAX];
    double start_time;
    double end_time;
    double stop_deadline;
    double timeout_sec;
    size_t bytes;
    int newline_count;
    char last_byte;
    bool observed_once;
    int exit_status;
    bool running;
    bool timed_out;
    bool reaped, pipe_eof;
    int child_status;
    int output_error;
    struct agent_tools_bash_job *next;
    agent_tools_worker *worker;  /* back-pointer for terminal state restoration */
};

static int set_nonblock(int fd, bool on, int *old_flags) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags < 0) return -1;
    if (old_flags) *old_flags = flags;
    int next = on ? (flags | O_NONBLOCK) : (flags & ~O_NONBLOCK);
    return fcntl(fd, F_SETFL, next);
}

static int agent_tools_bash_display_lines(const agent_tools_bash_job *job) {
    if (!job || job->bytes == 0) return 0;
    return job->newline_count + (job->last_byte != '\n');
}

static void agent_tools_bash_note_output(agent_tools_bash_job *job, const char *s, size_t n) {
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '\n' && job->newline_count < INT_MAX - 1) job->newline_count++;
    }
    if (n) job->last_byte = s[n - 1];
    job->bytes += n;
}

bool agent_tools_bash_is_running(agent_tools_bash_job *job) {
    pthread_mutex_lock(&job->mu);
    bool running = job->running;
    pthread_mutex_unlock(&job->mu);
    return running;
}

void agent_tools_bash_signal(agent_tools_bash_job *job, int sig) {
    pthread_mutex_lock(&job->mu);
    if (job->running) {
        if (sig == SIGKILL && !job->stop_deadline) job->stop_deadline = now_sec() + 1;
        kill(-job->pid, sig);
        if (!job->reaped) kill(job->pid, sig);
    }
    pthread_mutex_unlock(&job->mu);
}

static void agent_tools_bash_job_free(agent_tools_bash_job *job) {
    if (!job) return;
    agent_tools_bash_signal(job, SIGKILL);
    if (job->thread_started) pthread_join(job->thread, NULL);
    else if (job->pid > 0)
        while (waitpid(job->pid, NULL, 0) < 0 && errno == EINTR) {}
    if (job->pipe_fd >= 0) close(job->pipe_fd);
    if (job->tmp_fd >= 0) close(job->tmp_fd);
    pthread_mutex_destroy(&job->mu);
    free(job);
}

void agent_tools_bash_jobs_free(agent_tools_worker *w) {
    agent_tools_bash_job *job = w->bash_jobs;
    while (job) {
        agent_tools_bash_job *next = job->next;
        agent_tools_bash_job_free(job);
        job = next;
    }
    w->bash_jobs = NULL;
}

static agent_tools_bash_job *agent_tools_bash_find_job(agent_tools_worker *w,
                                                       int id, pid_t pid) {
    for (agent_tools_bash_job *job = w->bash_jobs; job; job = job->next) {
        if ((id > 0 && job->id == id) || (id <= 0 && pid > 0 && job->pid == pid))
            return job;
    }
    return NULL;
}

void agent_tools_bash_remove_job(agent_tools_worker *w, agent_tools_bash_job *target) {
    agent_tools_bash_job **link = &w->bash_jobs;
    while (*link) {
        if (*link == target) {
            *link = target->next;
            target->next = NULL;
            agent_tools_bash_job_free(target);
            return;
        }
        link = &(*link)->next;
    }
}

static void agent_tools_bash_drain(agent_tools_bash_job *job) {
    if (!job || job->pipe_fd < 0) return;
    char tmp[4096];
    /* Yield even for an endless writer, so its deadline is still checked. */
    for (size_t drained = 0; drained < 256 * 1024;) {
        ssize_t n = read(job->pipe_fd, tmp, sizeof(tmp));
        if (n > 0) {
            drained += (size_t)n;
            size_t pos = 0;
            while (pos < (size_t)n) {
                ssize_t wr = write(job->tmp_fd, tmp + pos, (size_t)n - pos);
                if (wr < 0 && errno == EINTR) continue;
                if (wr <= 0) {
                    job->output_error = wr < 0 ? errno : EIO;
                    return;
                }
                agent_tools_bash_note_output(job, tmp + pos, (size_t)wr);
                pos += (size_t)wr;
            }
            continue;
        }
        if (!n) job->pipe_eof = true;
        if (n < 0 && errno == EINTR) continue;
        if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK)
            job->output_error = errno;
        break;
    }
}

static void agent_tools_note_terminal_mode_may_have_changed(agent_tools_worker *w) {
    if (!w) return;
    w->raw_mode_needs_restore = true;
}

static void agent_tools_bash_finalize(agent_tools_bash_job *job, int status) {
    agent_tools_bash_drain(job);
    if (job->pipe_fd >= 0) {
        close(job->pipe_fd);
        job->pipe_fd = -1;
    }
    if (job->tmp_fd >= 0) {
        close(job->tmp_fd);
        job->tmp_fd = -1;
    }
    if (WIFEXITED(status)) job->exit_status = WEXITSTATUS(status);
    else if (WIFSIGNALED(status)) job->exit_status = 128 + WTERMSIG(status);
    else job->exit_status = -1;
    job->running = false;
    job->end_time = now_sec();
    /* A child can still open /dev/tty directly and alter terminal state even
     * though its stdin is /dev/null.  Ask the client to verify raw mode at a
     * safe point instead of touching linenoise from the worker path. */
    agent_tools_note_terminal_mode_may_have_changed(job->worker);
}

/* The monitor owns the descriptors and child reaping. Tool observations take
 * the job mutex; the model worker owns only the list and observation cursors. */
static void agent_tools_bash_poll(agent_tools_bash_job *job) {
    if (!job || !job->running) return;
    agent_tools_bash_drain(job);

    int status = 0;
    pid_t rc = job->reaped ? 0 : waitpid(job->pid, &status, WNOHANG);
    if (rc == job->pid) {
        job->reaped = true;
        job->child_status = status;
    }
    if (rc < 0 && errno != EINTR) {
        job->exit_status = -1;
        job->running = false;
        job->end_time = now_sec();
        if (job->pipe_fd >= 0) {
            close(job->pipe_fd);
            job->pipe_fd = -1;
        }
        if (job->tmp_fd >= 0) {
            close(job->tmp_fd);
            job->tmp_fd = -1;
        }
        agent_tools_note_terminal_mode_may_have_changed(job->worker);
        return;
    }
    if (job->reaped && job->pipe_eof) {
        agent_tools_bash_finalize(job, job->child_status);
        return;
    }
    if (!job->timed_out && (job->output_error || now_sec() - job->start_time >= job->timeout_sec)) {
        job->timed_out = !job->output_error;
        kill(-job->pid, SIGKILL);
        if (!job->reaped) {
            kill(job->pid, SIGKILL);
            while (waitpid(job->pid, &status, 0) < 0 && errno == EINTR) {}
            job->reaped = true;
            job->child_status = status;
        }
    }
    if (job->output_error || (job->timed_out && now_sec() - job->start_time > job->timeout_sec + 1) ||
        (job->stop_deadline && now_sec() >= job->stop_deadline)) {
        /* A descendant can deliberately escape the shell process group. Do
         * not let its inherited stdout prevent bounded job cleanup. */
        if (!job->output_error && !job->pipe_eof) job->output_error = EIO;
        agent_tools_bash_finalize(job, job->child_status);
    }
}

static void *agent_tools_bash_monitor(void *arg) {
    agent_tools_bash_job *job = arg;
    for (;;) {
        pthread_mutex_lock(&job->mu);
        agent_tools_bash_poll(job);
        bool running = job->running;
        int fd = job->pipe_fd;
        pthread_mutex_unlock(&job->mu);
        if (!running) break;
        struct pollfd pfd = {.fd = fd, .events = POLLIN};
        int rc = poll(&pfd, 1, 50);
        /* A closed stdout must not turn the deadline monitor into a busy loop. */
        if (rc > 0 && (pfd.revents & POLLHUP) && !(pfd.revents & POLLIN))
            usleep(50000);
    }
    return NULL;
}

/* Spawn a shell command into its own process group so bash_stop/timeout can
 * kill grandchildren created by the shell, not just the /bin/sh wrapper. */
agent_tools_bash_job *agent_tools_bash_start(agent_tools_worker *w,
                                             const char *cmd,
                                             int timeout_sec,
                                             char *err, size_t err_len) {
    char tmp_path[] = "/tmp/ds4_agent_output_XXXXXX";
    int tmpfd = mkstemp(tmp_path);
    if (tmpfd < 0) {
        snprintf(err, err_len, "failed to create temporary output file: %s", strerror(errno));
        return NULL;
    }

    int pipefd[2];
    if (pipe(pipefd) != 0) {
        snprintf(err, err_len, "failed to create pipe: %s", strerror(errno));
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    fcntl(tmpfd, F_SETFD, FD_CLOEXEC);
    fcntl(pipefd[0], F_SETFD, FD_CLOEXEC);
    fcntl(pipefd[1], F_SETFD, FD_CLOEXEC);
    pid_t pid = fork();
    if (pid < 0) {
        snprintf(err, err_len, "failed to fork: %s", strerror(errno));
        close(pipefd[0]);
        close(pipefd[1]);
        close(tmpfd);
        unlink(tmp_path);
        return NULL;
    }
    if (pid == 0) {
        setpgid(0, 0);
        close(tmpfd);
        /* The bash tool is not interactive.  Give the shell /dev/null as
         * stdin so it does not inherit the live linenoise terminal and reset
         * it from raw mode to cooked mode behind the agent's back. */
        int null_fd = open("/dev/null", O_RDONLY);
        if (null_fd >= 0) {
            if (dup2(null_fd, STDIN_FILENO) < 0)
                close(STDIN_FILENO);
            if (null_fd != STDIN_FILENO)
                close(null_fd);
        } else {
            close(STDIN_FILENO);
        }
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        dup2(pipefd[1], STDERR_FILENO);
        close(pipefd[1]);
        execl("/bin/sh", "sh", "-c", cmd ? cmd : "", (char *)NULL);
        _exit(127);
    }

    close(pipefd[1]);
    setpgid(pid, pid);
    int old_flags;
    set_nonblock(pipefd[0], true, &old_flags);

    agent_tools_bash_job *job = xmalloc(sizeof(*job));
    memset(job, 0, sizeof(*job));
    pthread_mutex_init(&job->mu, NULL);
    if (w->next_bash_job_id <= 0) w->next_bash_job_id = 1;
    job->id = w->next_bash_job_id++;
    job->pid = pid;
    job->pipe_fd = pipefd[0];
    job->tmp_fd = tmpfd;
    snprintf(job->path, sizeof(job->path), "%s", tmp_path);
    job->start_time = now_sec();
    job->timeout_sec = timeout_sec;
    job->exit_status = -1;
    job->running = true;
    job->worker = w;
    int thread_rc = pthread_create(&job->thread, NULL, agent_tools_bash_monitor, job);
    if (thread_rc != 0) {
        snprintf(err, err_len, "failed to monitor shell command: %s", strerror(thread_rc));
        agent_tools_bash_job_free(job);
        unlink(tmp_path);
        return NULL;
    }
    job->thread_started = true;
    job->next = w->bash_jobs;
    w->bash_jobs = job;
    return job;
}

/* Read the first max_lines from the output file, with a byte cap to avoid a
 * pathological single long line flooding the next model turn. */
static char *agent_tools_bash_read_head(const agent_tools_bash_job *job,
                                        int max_lines, size_t max_bytes,
                                        int *lines_read, bool *byte_limited) {
    if (lines_read) *lines_read = 0;
    if (byte_limited) *byte_limited = false;
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf out = {0};
    int lines = 0;
    size_t available = job->bytes < max_bytes ? job->bytes : max_bytes;
    while (lines < max_lines && out.len < available) {
        int c = fgetc(fp);
        if (c == EOF) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
        char ch = (char)c;
        agent_buf_append(&out, &ch, 1);
        if (ch == '\n') lines++;
    }
    if (out.len >= max_bytes && out.len < job->bytes && byte_limited) *byte_limited = true;
    fclose(fp);
    if (lines_read) *lines_read = lines + (out.len && out.ptr[out.len - 1] != '\n');
    if (!out.ptr) return xstrdup("");
    return agent_buf_take(&out);
}

/* Read the last max_lines from the full output file.  The model-visible label
 * says "tail -N <file>" so it is clear this is not the complete output. */
static char *agent_tools_bash_read_tail_lines(const agent_tools_bash_job *job, int max_lines) {
    if (!job || !job->path[0] || job->bytes == 0) return xstrdup("");
    FILE *fp = fopen(job->path, "rb");
    if (!fp) return xstrdup("<failed to reopen output file>\n");

    agent_buf tail = {0};
    size_t bytes = job->bytes;
    size_t take = bytes < AGENT_BASH_TAIL_BYTES ? bytes : AGENT_BASH_TAIL_BYTES;
    if (fseeko(fp, (off_t)(bytes - take), SEEK_SET) != 0) {
        fclose(fp);
        return xstrdup("<failed to seek output file>\n");
    }
    char tmp[2048];
    while (take) {
        size_t want = take < sizeof(tmp) ? take : sizeof(tmp);
        size_t n = fread(tmp, 1, want, fp);
        if (n) agent_buf_append(&tail, tmp, n);
        take -= n;
        if (n < want) {
            if (ferror(fp) && errno == EINTR) {
                clearerr(fp);
                continue;
            }
            break;
        }
    }
    fclose(fp);
    if (!tail.ptr) return xstrdup("");

    char *start = tail.ptr;
    int newlines = tail.ptr[tail.len - 1] == '\n' ? 0 : 1;
    for (char *p = tail.ptr + tail.len; p > tail.ptr; p--) {
        if (p[-1] == '\n' && ++newlines > max_lines) {
            start = p;
            break;
        }
    }
    /* A byte-limited tail may begin inside a UTF-8 character. */
    while (((unsigned char)*start & 0xc0) == 0x80) start++;
    char *out = xstrdup(start);
    free(tail.ptr);
    return out;
}

/* The first observation uses the head; subsequent ones use a bounded tail. */
char *agent_tools_bash_observation(agent_tools_bash_job *job, bool mark_observed,
                                   bool *done) {
    pthread_mutex_lock(&job->mu);
    agent_tools_bash_job snapshot = {
        .id = job->id, .pid = job->pid,
        .start_time = job->start_time, .end_time = job->end_time,
        .timeout_sec = job->timeout_sec, .bytes = job->bytes,
        .newline_count = job->newline_count, .last_byte = job->last_byte,
        .observed_once = job->observed_once, .exit_status = job->exit_status,
        .running = job->running, .timed_out = job->timed_out,
        .output_error = job->output_error,
    };
    memcpy(snapshot.path, job->path, sizeof(snapshot.path));
    if (mark_observed) job->observed_once = true;
    pthread_mutex_unlock(&job->mu);
    /* Disk reads must not hold up the monitor's deadline checks. Only the
     * immutable observation fields above are read from this local snapshot. */
    job = &snapshot;
    if (done) *done = !snapshot.running;
    bool first_observation = !job->observed_once;
    int display_lines = agent_tools_bash_display_lines(job);
    double elapsed = (job->running ? now_sec() : job->end_time) - job->start_time;

    agent_buf out = {0};
    char line[PATH_MAX + 256];
    if (job->running) {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=running elapsed_sec=%.1f timeout_sec=%.0f\n",
            job->id, (long)job->pid, elapsed, job->timeout_sec);
    } else {
        snprintf(line, sizeof(line),
            "bash job=%d pid=%ld status=done elapsed_sec=%.1f timed_out=%d\n",
            job->id, (long)job->pid, elapsed, job->timed_out ? 1 : 0);
    }
    agent_buf_puts(&out, line);
    if (!job->running) {
        snprintf(line, sizeof(line), "exit_status=%d\n", job->exit_status);
        agent_buf_puts(&out, line);
    }
    if (job->output_error) {
        agent_buf_puts(&out, "Tool error: command output could not be captured completely: ");
        agent_buf_puts(&out, strerror(job->output_error));
        agent_buf_puts(&out, "\n");
    }

    if (job->bytes == 0) {
        agent_buf_puts(&out, "<output>\n</output>\n");
    } else if (first_observation) {
        int shown_lines = 0;
        bool byte_limited = false;
        char *head = agent_tools_bash_read_head(job, AGENT_BASH_HEAD_LINES,
                                                AGENT_BASH_HEAD_BYTES,
                                                &shown_lines, &byte_limited);
        bool truncated = byte_limited || display_lines > shown_lines;
        if (!job->running && !truncated) {
            agent_buf_puts(&out, "<output>\n");
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</output>\n");
        } else {
            snprintf(line, sizeof(line),
                     "output_path=%s (%zu bytes, %d lines)\n",
                     job->path[0] ? job->path : "<unavailable>",
                     job->bytes, display_lines);
            agent_buf_puts(&out, line);
            snprintf(line, sizeof(line), "<head -%d %s>\n",
                     AGENT_BASH_HEAD_LINES, job->path);
            agent_buf_puts(&out, line);
            agent_buf_puts(&out, head);
            if (head[0] && head[strlen(head) - 1] != '\n') agent_buf_puts(&out, "\n");
            agent_buf_puts(&out, "</head>\n");
        }
        free(head);
    } else {
        int tail_lines = job->running ? AGENT_BASH_PROGRESS_TAIL_LINES :
                                        AGENT_BASH_FINAL_TAIL_LINES;
        char *tail = agent_tools_bash_read_tail_lines(job, tail_lines);
        snprintf(line, sizeof(line),
                 "output_path=%s (%zu bytes, %d lines)\n",
                 job->path[0] ? job->path : "<unavailable>",
                 job->bytes, display_lines);
        agent_buf_puts(&out, line);
        snprintf(line, sizeof(line), "<tail -%d %s>\n", tail_lines, job->path);
        agent_buf_puts(&out, line);
        agent_buf_puts(&out, tail);
        if (tail[0] && tail[strlen(tail) - 1] != '\n') agent_buf_puts(&out, "\n");
        snprintf(line, sizeof(line), "</tail>\n");
        agent_buf_puts(&out, line);
        free(tail);
    }
    if (job->running) {
        snprintf(line, sizeof(line),
            "\nUse bash_status job=%d to get info before refresh time; use bash_stop job=%d to stop execution\n",
            job->id, job->id);
        agent_buf_puts(&out, line);
    }

    return agent_buf_take(&out);
}

static void agent_tools_bash_refresh_for(agent_tools_worker *w,
                                         agent_tools_bash_job *job,
                                         int refresh_sec) {
    double start = now_sec();
    while (agent_tools_bash_is_running(job) && now_sec() - start < refresh_sec) {
        if (agent_tools_should_interrupt(w)) break;
        usleep(20000);
    }
}

/* Common implementation for bash, bash_status, and bash_stop. */
static char *agent_tools_bash_job_tool_result(agent_tools_worker *w,
                                              agent_tools_bash_job *job,
                                              bool wait, int refresh_sec,
                                              bool stop, bool remove_if_done) {
    if (stop && agent_tools_bash_is_running(job)) {
        agent_tools_bash_signal(job, SIGTERM);
        double start = now_sec();
        while (agent_tools_bash_is_running(job) && now_sec() - start < 1.0) {
            usleep(20000);
        }
        agent_tools_bash_signal(job, SIGKILL);
    }
    if (wait || stop) agent_tools_bash_refresh_for(w, job, refresh_sec);

    bool done = false;
    char *obs = agent_tools_bash_observation(job, true, &done);
    if (remove_if_done && done) agent_tools_bash_remove_job(w, job);
    return obs;
}

static int agent_tools_bash_arg_job_id(const proto_tool_call *call) {
    return agent_tools_parse_int_default(agent_tools_arg_value(call, "job"), 0, 0, INT_MAX);
}

static pid_t agent_tools_bash_arg_pid(const proto_tool_call *call) {
    return (pid_t)agent_tools_parse_int_default(agent_tools_arg_value(call, "pid"), 0, 0, INT_MAX);
}

/* ============================================================================
 * Public entry points
 * ========================================================================== */

char *agent_tools_bash_run(agent_tools_worker *w, const proto_tool_call *call) {
    const char *cmd = agent_tools_arg_value(call, "command");
    if (!cmd || !cmd[0]) return xstrdup("Tool error: bash requires command\n");
    int timeout = agent_tools_parse_timeout(agent_tools_arg_value(call, "timeout_sec"));
    int refresh = agent_tools_parse_int_default(agent_tools_arg_value(call, "refresh_sec"),
                                                60, 1, 3600);
    char err[160] = {0};
    agent_tools_bash_job *job = agent_tools_bash_start(w, cmd, timeout, err, sizeof(err));
    if (!job) {
        agent_buf result = {0};
        agent_buf_puts(&result, "Tool error: bash failed to start: ");
        agent_buf_puts(&result, err[0] ? err : "unknown error");
        agent_buf_puts(&result, "\n");
        return agent_buf_take(&result);
    }
    return agent_tools_bash_job_tool_result(w, job, true, refresh, false, true);
}

char *agent_tools_bash_status(agent_tools_worker *w, const proto_tool_call *call) {
    int job_id = agent_tools_bash_arg_job_id(call);
    pid_t pid = agent_tools_bash_arg_pid(call);
    agent_tools_bash_job *job = agent_tools_bash_find_job(w, job_id, pid);
    if (!job) {
        char msg[128];
        snprintf(msg, sizeof(msg), "Tool error: bash job not found: job=%d pid=%ld\n",
                 job_id, (long)pid);
        return xstrdup(msg);
    }
    int refresh = agent_tools_parse_int_default(agent_tools_arg_value(call, "refresh_sec"),
                                                0, 0, 3600);
    bool wait = refresh > 0;
    return agent_tools_bash_job_tool_result(w, job, wait, refresh, false, true);
}

char *agent_tools_bash_stop(agent_tools_worker *w, const proto_tool_call *call) {
    int job_id = agent_tools_bash_arg_job_id(call);
    pid_t pid = agent_tools_bash_arg_pid(call);
    agent_tools_bash_job *job = agent_tools_bash_find_job(w, job_id, pid);
    if (!job) {
        char msg[128];
        snprintf(msg, sizeof(msg), "Tool error: bash job not found: job=%d pid=%ld\n",
                 job_id, (long)pid);
        return xstrdup(msg);
    }
    int refresh = agent_tools_parse_int_default(agent_tools_arg_value(call, "refresh_sec"),
                                                0, 0, 3600);
    if (refresh == 0) refresh = 1;
    return agent_tools_bash_job_tool_result(w, job, true, refresh, true, true);
}

/* ---- Job field accessors (test-facing) ---- */

const char *agent_tools_bash_job_path(const agent_tools_bash_job *job) {
    return job ? job->path : "";
}

size_t agent_tools_bash_job_bytes(const agent_tools_bash_job *job) {
    return job ? job->bytes : 0;
}

int agent_tools_bash_job_id(const agent_tools_bash_job *job) {
    return job ? job->id : 0;
}

