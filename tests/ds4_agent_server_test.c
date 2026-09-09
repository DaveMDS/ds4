/*
 * Unit test for the server-internal streaming parser / render-directive
 * producer (Task 2).
 * Build:  cc -O2 -Wall -Wextra -std=c99 -D_GNU_SOURCE -I. \
 *         -o tests/ds4_agent_server_test tests/ds4_agent_server_test.c
 * Run:    ./tests/ds4_agent_server_test
 */

#define DS4_AGENT_TEST
#include "../agent/ds4_agent_server.c"

int main(void) {
    agent_test_run_all();
    if (agent_test_failures) {
        fprintf(stderr, "ds4_agent_server_test: %d failures\n", agent_test_failures);
        return 1;
    }
    printf("ds4_agent_server_test: all ok\n");
    return 0;
}
