/*
 * Unit test for the shared client/server wire protocol (Layer 1).
 * Build:  cc -O2 -Wall -Wextra -std=c99 -I. -o tests/ds4_agent_proto_test tests/ds4_agent_proto_test.c
 * Run:    ./tests/ds4_agent_proto_test
 */

#define DS4_AGENT_TEST
#include "../agent/ds4_agent_proto.h"

int main(void) {
    proto_test_run_all();
    return 0;
}
