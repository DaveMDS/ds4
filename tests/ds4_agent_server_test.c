/*
 * Unit test for the server-internal streaming parser / render-directive
 * producer (Task 2).
 * Build:  cc -O2 -Wall -Wextra -std=c99 -D_GNU_SOURCE -I. \
 *         -o tests/ds4_agent_server_test tests/ds4_agent_server_test.c
 * Run:    ./tests/ds4_agent_server_test
 */

#define DS4_AGENT_TEST
#include "../agent/ds4_agent_server.c"

/* ---- Pure kvstore primitives used by the Layer 3 tests.
 * The real ds4_kvstore.c pulls in the whole engine, so the engine-free
 * server test provides faithful self-contained stubs for the handful of
 * pure byte/sha primitives these control-logic tests exercise. */

#define KV_CACHE_MAGIC0 'K'
#define KV_CACHE_MAGIC1 'V'
#define KV_CACHE_MAGIC2 'C'
#define KV_CACHE_VERSION 1u
#define KV_CACHE_PAYLOAD_ABI 2u

static void kv_test_le_put64(uint8_t *p, uint64_t v) {
    for (int i = 0; i < 8; i++) p[i] = (uint8_t)(v >> (8 * i));
}

static uint64_t kv_test_le_get64(const uint8_t *p) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; i--) v = (v << 8) | p[i];
    return v;
}

void ds4_kvstore_le_put32(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)v;
    p[1] = (uint8_t)(v >> 8);
    p[2] = (uint8_t)(v >> 16);
    p[3] = (uint8_t)(v >> 24);
}

uint32_t ds4_kvstore_le_get32(const uint8_t *p) {
    return (uint32_t)p[0] |
           ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) |
           ((uint32_t)p[3] << 24);
}

void ds4_kvstore_fill_header(uint8_t h[DS4_KVSTORE_FIXED_HEADER],
                             uint8_t model_id, uint8_t quant_bits,
                             uint8_t reason, uint8_t ext_flags,
                             uint32_t tokens, uint32_t hits, uint32_t ctx_size,
                             uint64_t created_at, uint64_t last_used,
                             uint64_t payload_bytes) {
    memset(h, 0, DS4_KVSTORE_FIXED_HEADER);
    h[0] = KV_CACHE_MAGIC0;
    h[1] = KV_CACHE_MAGIC1;
    h[2] = KV_CACHE_MAGIC2;
    h[3] = KV_CACHE_VERSION;
    h[4] = quant_bits;
    h[5] = reason;
    h[6] = ext_flags;
    h[7] = model_id;
    ds4_kvstore_le_put32(h + 8, tokens);
    ds4_kvstore_le_put32(h + 12, hits);
    ds4_kvstore_le_put32(h + 16, ctx_size);
    h[20] = KV_CACHE_PAYLOAD_ABI;
    kv_test_le_put64(h + 24, created_at);
    kv_test_le_put64(h + 32, last_used);
    kv_test_le_put64(h + 40, payload_bytes);
}

bool ds4_kvstore_read_header(FILE *fp, ds4_kvstore_entry *e,
                             uint32_t *text_bytes) {
    uint8_t h[DS4_KVSTORE_FIXED_HEADER];
    if (fread(h, 1, sizeof(h), fp) != sizeof(h)) return false;
    if (h[0] != KV_CACHE_MAGIC0 || h[1] != KV_CACHE_MAGIC1 ||
        h[2] != KV_CACHE_MAGIC2 || h[3] != KV_CACHE_VERSION) return false;
    if (h[20] != KV_CACHE_PAYLOAD_ABI) return false;
    e->quant_bits = h[4];
    e->reason = h[5] <= DS4_KVSTORE_REASON_AGENT_SESSION ? h[5] :
                DS4_KVSTORE_REASON_UNKNOWN;
    e->ext_flags = h[6];
    e->model_id = h[7];
    e->tokens = ds4_kvstore_le_get32(h + 8);
    e->hits = ds4_kvstore_le_get32(h + 12);
    e->ctx_size = ds4_kvstore_le_get32(h + 16);
    e->created_at = kv_test_le_get64(h + 24);
    e->last_used = kv_test_le_get64(h + 32);
    e->payload_bytes = kv_test_le_get64(h + 40);
    uint8_t tb[4];
    if (fread(tb, 1, sizeof(tb), fp) != sizeof(tb)) return false;
    *text_bytes = ds4_kvstore_le_get32(tb);
    e->text_bytes = *text_bytes;
    return e->tokens != 0 && (e->quant_bits == 2 || e->quant_bits == 4);
}

typedef struct {
    uint32_t h[5];
    uint64_t bytes;
    uint8_t block[64];
    size_t used;
} kv_test_sha1_ctx;

static uint32_t kv_test_rol32(uint32_t v, int n) {
    return (v << n) | (v >> (32 - n));
}

static void kv_test_sha1_transform(kv_test_sha1_ctx *c,
                                   const uint8_t block[64]) {
    uint32_t w[80];
    for (int i = 0; i < 16; i++) {
        w[i] = ((uint32_t)block[i * 4] << 24) |
               ((uint32_t)block[i * 4 + 1] << 16) |
               ((uint32_t)block[i * 4 + 2] << 8) |
               (uint32_t)block[i * 4 + 3];
    }
    for (int i = 16; i < 80; i++)
        w[i] = kv_test_rol32(w[i - 3] ^ w[i - 8] ^ w[i - 14] ^ w[i - 16], 1);
    uint32_t a = c->h[0], b = c->h[1], d = c->h[3], e = c->h[4];
    uint32_t cc = c->h[2];
    for (int i = 0; i < 80; i++) {
        uint32_t f, k;
        if (i < 20) {
            f = (b & cc) | ((~b) & d);
            k = 0x5a827999u;
        } else if (i < 40) {
            f = b ^ cc ^ d;
            k = 0x6ed9eba1u;
        } else if (i < 60) {
            f = (b & cc) | (b & d) | (cc & d);
            k = 0x8f1bbcdcu;
        } else {
            f = b ^ cc ^ d;
            k = 0xca62c1d6u;
        }
        uint32_t tmp = kv_test_rol32(a, 5) + f + e + k + w[i];
        e = d;
        d = cc;
        cc = kv_test_rol32(b, 30);
        b = a;
        a = tmp;
    }
    c->h[0] += a;
    c->h[1] += b;
    c->h[2] += cc;
    c->h[3] += d;
    c->h[4] += e;
}

static void kv_test_sha1_init(kv_test_sha1_ctx *c) {
    c->h[0] = 0x67452301u;
    c->h[1] = 0xefcdab89u;
    c->h[2] = 0x98badcfeu;
    c->h[3] = 0x10325476u;
    c->h[4] = 0xc3d2e1f0u;
    c->bytes = 0;
    c->used = 0;
}

static void kv_test_sha1_update(kv_test_sha1_ctx *c, const void *ptr, size_t len) {
    const uint8_t *p = ptr;
    c->bytes += len;
    while (len != 0) {
        size_t n = 64 - c->used;
        if (n > len) n = len;
        memcpy(c->block + c->used, p, n);
        c->used += n;
        p += n;
        len -= n;
        if (c->used == 64) {
            kv_test_sha1_transform(c, c->block);
            c->used = 0;
        }
    }
}

static void kv_test_sha1_final(kv_test_sha1_ctx *c, uint8_t out[20]) {
    uint64_t bits = c->bytes * 8;
    uint8_t one = 0x80;
    uint8_t zero = 0;
    kv_test_sha1_update(c, &one, 1);
    while (c->used != 56) kv_test_sha1_update(c, &zero, 1);
    uint8_t len[8];
    for (int i = 0; i < 8; i++) len[7 - i] = (uint8_t)(bits >> (8 * i));
    kv_test_sha1_update(c, len, sizeof(len));
    for (int i = 0; i < 5; i++) {
        out[i * 4] = (uint8_t)(c->h[i] >> 24);
        out[i * 4 + 1] = (uint8_t)(c->h[i] >> 16);
        out[i * 4 + 2] = (uint8_t)(c->h[i] >> 8);
        out[i * 4 + 3] = (uint8_t)c->h[i];
    }
}

void ds4_kvstore_sha1_bytes_hex(const void *ptr, size_t len, char out[41]) {
    kv_test_sha1_ctx c;
    kv_test_sha1_init(&c);
    kv_test_sha1_update(&c, ptr, len);
    uint8_t digest[20];
    kv_test_sha1_final(&c, digest);
    static const char hex[] = "0123456789abcdef";
    for (int i = 0; i < 20; i++) {
        out[i * 2] = hex[digest[i] >> 4];
        out[i * 2 + 1] = hex[digest[i] & 15];
    }
    out[40] = '\0';
}

int main(void) {
    agent_test_run_all();
    if (agent_test_failures) {
        fprintf(stderr, "ds4_agent_server_test: %d failures\n", agent_test_failures);
        return 1;
    }
    printf("ds4_agent_server_test: all ok\n");
    return 0;
}
