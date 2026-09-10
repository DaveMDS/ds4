/* ds4_agent_tools_dispatch.c -- tool dispatch.  Ported from the monolithic
 * agent_execute_tool_call (ds4_agent.c ~9033); each branch delegates to the
 * per-group implementation.  UI-free: the monolith's "[tool:%s]" headers and
 * publish calls are dropped, only the model-visible observation text is
 * returned.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ds4_agent_tools.h"

char *agent_tools_dispatch(agent_tools_worker *w, const proto_tool_call *call) {
    if (!call->name) return xstrdup("Tool error: missing tool name\n");

    if (!strcmp(call->name, "read")) return agent_tools_file_read(w, call);
    if (!strcmp(call->name, "more")) return agent_tools_file_more(w, call);
    if (!strcmp(call->name, "write")) return agent_tools_file_write(w, call);
    if (!strcmp(call->name, "list")) return agent_tools_file_list(call);
    if (!strcmp(call->name, "edit")) return agent_tools_file_edit(w, call);
    if (!strcmp(call->name, "search")) return agent_tools_file_search(w, call);
    if (!strcmp(call->name, "google_search")) return agent_tools_web_google_search(w, call);
    if (!strcmp(call->name, "visit_page")) return agent_tools_web_visit_page(w, call);

    if (!strcmp(call->name, "bash")) return agent_tools_bash_run(w, call);
    if (!strcmp(call->name, "bash_status")) return agent_tools_bash_status(w, call);
    if (!strcmp(call->name, "bash_stop")) return agent_tools_bash_stop(w, call);

    {
        agent_buf result = {0};
        agent_buf_puts(&result, "Tool error: unknown tool: ");
        agent_buf_puts(&result, call->name);
        agent_buf_puts(&result, "\n");
        return agent_buf_take(&result);
    }
}
