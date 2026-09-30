/* Time one mdx_forward pass per kernel type.
 *
 *     ./bench_forward [models/kara.bin] [T=256] [--ref]
 *
 * Prints wall-clock seconds and GFLOP/s per kernel type (FLOPs counted from the
 * model's hyperparameters, 2 per multiply-add). --ref uses the naive kernels.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../mdx.h"

int main(int argc, char **argv) {
    const char *path = "models/kara.bin";
    int T = 256, ref = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--ref"))
            ref = 1;
        else if (strstr(argv[i], ".bin"))
            path = argv[i];
        else
            T = atoi(argv[i]);
    }

    MdxModel m;
    if (mdx_load(&m, path) != 0) return 1;
    const MdxConfig *c = &m.config;
    MdxState s;
    if (mdx_state_init(&s, c, T) != 0) return 1;
    s.reference = ref;

    /* FLOPs per kernel type */
    double fl[MDX_PROF_N] = {0};
    double g = c->growth, F = c->dim_f, Tt = T, bn = c->bn_factor;
    fl[MDX_PROF_CONV1X1] = 2 * (c->dim_c * g * F * Tt) * 2; /* first + final */
    double ch = g, f = F, t = Tt;
    for (uint32_t lvl = 0; lvl <= c->n_scales; lvl++) {
        int blocks = lvl < c->n_scales ? 2 : 1; /* encoder + decoder block at each level, 1 bottleneck */
        fl[MDX_PROF_CONV3X3] += blocks * c->n_tfc * 2 * ch * ch * 9 * t * f;
        fl[MDX_PROF_TDF] += blocks * 2 * (2 * ch * t * f * (f / bn));
        if (lvl < c->n_scales) { /* down: ch -> ch+g at half size; up: back */
            fl[MDX_PROF_DOWN] += 2 * ch * (ch + g) * 4 * (t / 2) * (f / 2);
            fl[MDX_PROF_UP] += 2 * (ch + g) * ch * 4 * (t / 2) * (f / 2);
        }
        ch += g, f /= 2, t /= 2;
    }

    size_t n_io = (size_t)c->dim_c * c->dim_f * T;
    float *in = malloc(n_io * sizeof(float)), *out = malloc(n_io * sizeof(float));
    unsigned long long st = 12345;
    for (size_t i = 0; i < n_io; i++) {
        st = st * 6364136223846793005ull + 1442695040888963407ull;
        in[i] = (float)((st >> 40) / 16777216.0 - 0.5);
    }

    mdx_forward(&m, &s, in, out);

    double total = 0, total_fl = 0;
    for (int k = 0; k < MDX_PROF_N; k++) total += s.prof[k], total_fl += fl[k];
    printf("%s kernels, T=%d\n", ref ? "reference" : "fast", T);
    printf("%-14s %9s %7s %10s %9s\n", "kernel", "seconds", "share", "GFLOP", "GFLOP/s");
    for (int k = 0; k < MDX_PROF_N; k++)
        printf("%-14s %9.3f %6.1f%% %10.1f %9.2f\n", mdx_prof_names[k], s.prof[k], 100 * s.prof[k] / total,
               fl[k] / 1e9, s.prof[k] > 0 ? fl[k] / 1e9 / s.prof[k] : 0.0);
    printf("%-14s %9.3f %6.1f%% %10.1f %9.2f\n", "total", total, 100.0, total_fl / 1e9, total_fl / 1e9 / total);

    free(in), free(out);
    mdx_state_free(&s);
    mdx_free(&m);
    return 0;
}
