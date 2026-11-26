// mixcheck SRC TYPE NCOLS NROWS SRC_OFF MIX MIX_OFF: cosine similarity between
// sampled rows of a Q2 tensor (IQ2_XXS 16 / Q2_K 10) and its MXFP4 version.
// Real DeepSeek MXFP4 weights should correlate well with their Q2 quantization;
// a wrong range or corrupt download gives ~0.
#define main mixsyn_main
#include "mixsyn.c"
#undef main

static float mx_value(uint8_t c) { return (c & 8 ? -1.0f : 1.0f) * mx_values[c & 7]; }

static void deq_mxfp4(const block_mxfp4 *x, float *y, int nb) {
    for (int i = 0; i < nb; i++, y += 32) {
        uint32_t bits = x[i].e == 0 ? 0x00400000u : (uint32_t)x[i].e << 23;
        float d;
        memcpy(&d, &bits, 4);
        for (int j = 0; j < 16; j++) {
            y[j] = d * mx_value(x[i].qs[j] & 15);
            y[j + 16] = d * mx_value(x[i].qs[j] >> 4);
        }
    }
}

static const uint8_t *map_range(const char *path, long off, long bytes) {
    const long page = getpagesize(), base = off / page * page;
    int fd = open(path, O_RDONLY);
    if (fd < 0) { perror(path); exit(1); }
    const uint8_t *m = mmap(NULL, bytes + (off - base), PROT_READ, MAP_PRIVATE, fd, base);
    if (m == MAP_FAILED) { perror("mmap"); exit(1); }
    return m + (off - base);
}

int main(int argc, char **argv) {
    if (argc != 8) { fprintf(stderr, "usage: mixcheck SRC TYPE NCOLS NROWS SRC_OFF MIX MIX_OFF\n"); return 2; }
    const int ty = atoi(argv[2]), nc = atoi(argv[3]);
    const long nr = atol(argv[4]);
    const long irow = (long)nc / 256 * (ty == 16 ? 66 : 84), orow = (long)nc / 32 * 17;
    const uint8_t *q = map_range(argv[1], atol(argv[5]), irow * nr);
    const uint8_t *m = map_range(argv[6], atol(argv[7]), orow * nr);
    float *a = malloc(sizeof(float) * nc), *b = malloc(sizeof(float) * nc);
    const long step = nr / 4096 > 0 ? nr / 4096 : 1;
    double sum = 0.0, mn = 1.0;
    long n = 0, bad = 0;
    for (long r = 0; r < nr; r += step, n++) {
        if (ty == 16) deq_iq2_xxs((const block_iq2_xxs *)(q + r * irow), a, nc / 256);
        else deq_q2_K((const block_q2_K *)(q + r * irow), a, nc / 256);
        deq_mxfp4((const block_mxfp4 *)(m + r * orow), b, nc / 32);
        double ab = 0, aa = 0, bb = 0;
        for (int j = 0; j < nc; j++) { ab += (double)a[j] * b[j]; aa += (double)a[j] * a[j]; bb += (double)b[j] * b[j]; }
        const double c = aa > 0 && bb > 0 ? ab / sqrt(aa * bb) : 0.0;
        sum += c;
        if (c < mn) mn = c;
        if (c < 0.5) bad++;
    }
    printf("rows %ld cos mean %.4f min %.4f below0.5 %ld\n", n, sum / n, mn, bad);
    return 0;
}
