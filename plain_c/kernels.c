/* Fast versions of the convolution and matmul kernels.
 *
 * Same results as the *_ref kernels in mdx.c (up to float summation order),
 * reordered so the innermost loops run over contiguous memory, which lets the
 * compiler vectorise them, and parallelised with OpenMP when built with
 * -fopenmp (the pragmas are ignored otherwise). Formulas: see mdx.h.
 */
#include "mdx.h"

#include <string.h>

/* ------------------------------------------------------------------------- */
/* conv3x3, stride 1, zero pad 1
 *
 * Work is split into tiles of OB output channels x one output row x VW
 * consecutive columns. A tile's OB*VW accumulators stay in registers while we
 * loop over all input channels and the 9 taps; for each tap the inner loop is
 *     acc[o][v] += W[o][i][ky][kx] * x[i][h+ky-1][w0+v+kx-1],   v = 0..VW-1
 * i.e. a scalar weight times VW contiguous inputs - one vector FMA per o.
 */
#ifndef OB
#define OB 16 /* output channels per tile */
#endif
#ifndef VW
#define VW 64 /* columns per tile */
#endif

/* nob <= OB. Inlined with nob = OB as a constant for full tiles, so the o-loops
 * have a fixed trip count and the accumulators can live in registers. */
static inline void conv3x3_tile(float *y, const float *x, const float *W, const float *b, int cin, int H, int Wd,
                                int o0, int nob, int h, int w0) {
    float acc[OB][VW];
    for (int o = 0; o < nob; o++)
        for (int v = 0; v < VW; v++) acc[o][v] = b[o0 + o];

    long hw = (long)H * Wd;
    int interior = w0 > 0 && w0 + VW < Wd; /* all VW+2 input columns in range */
    for (int i = 0; i < cin; i++) {
        for (int ky = 0; ky < 3; ky++) {
            int hh = h + ky - 1;
            if (hh < 0 || hh >= H) continue; /* zero padding: contributes nothing */
            const float *row = x + i * hw + (long)hh * Wd;
            float edge[VW + 2];
            const float *xs; /* xs[j] = input column w0 - 1 + j, j = 0..VW+1 */
            if (interior) {
                xs = row + w0 - 1;
            } else {
                for (int j = 0; j < VW + 2; j++) {
                    int c = w0 - 1 + j;
                    edge[j] = c >= 0 && c < Wd ? row[c] : 0.0f;
                }
                xs = edge;
            }
            const float *wk = W + ((long)o0 * cin + i) * 9 + ky * 3; /* W[o0][i][ky][0] */
            for (int kx = 0; kx < 3; kx++)
                for (int o = 0; o < nob; o++) {
                    float wv = wk[(long)o * cin * 9 + kx];
                    for (int v = 0; v < VW; v++) acc[o][v] += wv * xs[kx + v];
                }
        }
    }
    for (int o = 0; o < nob; o++) memcpy(y + (o0 + o) * hw + (long)h * Wd + w0, acc[o], VW * sizeof(float));
}

void mdx_conv3x3(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd) {
    if (Wd % VW) { /* tiles need whole VW-column blocks; every layer of the model has Wd % 32 == 0 */
        mdx_conv3x3_ref(y, x, W, b, cin, cout, H, Wd);
        return;
    }
    int n_ob = (cout + OB - 1) / OB;
#pragma omp parallel for collapse(2) schedule(static)
    for (int ob = 0; ob < n_ob; ob++) {
        for (int h = 0; h < H; h++) {
            int o0 = ob * OB;
            for (int w0 = 0; w0 < Wd; w0 += VW) {
                if (o0 + OB <= cout)
                    conv3x3_tile(y, x, W, b, cin, H, Wd, o0, OB, h, w0);
                else
                    conv3x3_tile(y, x, W, b, cin, H, Wd, o0, cout - o0, h, w0);
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* conv1x1: for each output channel, y[o] = b[o] + sum_i W[o][i] * x[i] as
 * whole-plane axpys (contiguous). */
void mdx_conv1x1(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd) {
    long hw = (long)H * Wd;
#pragma omp parallel for schedule(static)
    for (int o = 0; o < cout; o++) {
        float *yo = y + o * hw;
        for (long p = 0; p < hw; p++) yo[p] = b[o];
        for (int i = 0; i < cin; i++) {
            float wv = W[(long)o * cin + i];
            const float *xi = x + i * hw;
            for (long p = 0; p < hw; p++) yo[p] += wv * xi[p];
        }
    }
}

/* ------------------------------------------------------------------------- */
/* conv2x2 stride 2: per output row, accumulate over (i, ky, kx) with the
 * stride-2 input columns 2w+kx. */
void mdx_conv2x2_s2(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd) {
    int Ho = H / 2, Wo = Wd / 2;
    long hw = (long)H * Wd;
#pragma omp parallel for collapse(2) schedule(static)
    for (int o = 0; o < cout; o++) {
        for (int h = 0; h < Ho; h++) {
            float *yr = y + ((long)o * Ho + h) * Wo;
            for (int w = 0; w < Wo; w++) yr[w] = b[o];
            for (int i = 0; i < cin; i++) {
                const float *wk = W + ((long)o * cin + i) * 4;
                const float *r0 = x + i * hw + (long)(2 * h) * Wd, *r1 = r0 + Wd;
                for (int w = 0; w < Wo; w++)
                    yr[w] += wk[0] * r0[2 * w] + wk[1] * r0[2 * w + 1] + wk[2] * r1[2 * w] + wk[3] * r1[2 * w + 1];
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* convT2x2 stride 2: output row 2h+ky gets, for every input channel i,
 * W[i][o][ky][0] * x[i][h][w] at column 2w and W[i][o][ky][1] * x[i][h][w] at 2w+1. */
void mdx_convT2x2_s2(float *y, const float *x, const float *W, const float *b, int cin, int cout, int H, int Wd) {
    int Ho = 2 * H, Wo = 2 * Wd;
    long hw = (long)H * Wd;
#pragma omp parallel for collapse(2) schedule(static)
    for (int o = 0; o < cout; o++) {
        for (int yy = 0; yy < Ho; yy++) {
            int h = yy / 2, ky = yy % 2;
            float *yr = y + ((long)o * Ho + yy) * Wo;
            for (int c = 0; c < Wo; c++) yr[c] = b[o];
            for (int i = 0; i < cin; i++) {
                const float *wk = W + (((long)i * cout + o) * 2 + ky) * 2; /* W[i][o][ky][.] */
                const float *xr = x + i * hw + (long)h * Wd;
                for (int w = 0; w < Wd; w++) {
                    yr[2 * w] += wk[0] * xr[w];
                    yr[2 * w + 1] += wk[1] * xr[w];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------------- */
/* matmul over the last axis: y[r][:] = sum_f x[r][f] * W[f][:].
 * RB rows at a time so each W row (contiguous, fout floats) is loaded once per
 * RB output rows; the RB output rows stay in L1. */
#ifndef RB
#define RB 4
#endif
void mdx_matmul_lastdim(float *y, const float *x, const float *W, int rows, int fin, int fout) {
    int n_rb = (rows + RB - 1) / RB;
#pragma omp parallel for schedule(static)
    for (int rb = 0; rb < n_rb; rb++) {
        int r0 = rb * RB, nr = rows - r0 < RB ? rows - r0 : RB;
        float *yr = y + (long)r0 * fout;
        memset(yr, 0, (size_t)nr * fout * sizeof(float));
        for (int f = 0; f < fin; f++) {
            const float *wf = W + (long)f * fout;
            for (int rr = 0; rr < nr; rr++) {
                float xv = x[(long)(r0 + rr) * fin + f];
                float *yrr = yr + (long)rr * fout;
                for (int j = 0; j < fout; j++) yrr[j] += xv * wf[j];
            }
        }
    }
}
