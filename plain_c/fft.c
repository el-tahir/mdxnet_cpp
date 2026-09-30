#include "fft.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

int fft_init(FFTPlan *p, int n) {
    memset(p, 0, sizeof(*p));
    if (n < 1 || (n & (n - 1))) return -1; /* power of two only */
    int bits = 0;
    while ((1 << bits) < n) bits++;

    p->n = n;
    p->cos_t = malloc((size_t)(n / 2 + 1) * sizeof(float));
    p->sin_t = malloc((size_t)(n / 2 + 1) * sizeof(float));
    p->rev = malloc((size_t)n * sizeof(int));
    if (!p->cos_t || !p->sin_t || !p->rev) {
        fft_free(p);
        return -1;
    }
    for (int k = 0; k < n / 2; k++) { /* computed in double, stored as float */
        p->cos_t[k] = (float)cos(2.0 * M_PI * k / n);
        p->sin_t[k] = (float)sin(2.0 * M_PI * k / n);
    }
    for (int i = 0; i < n; i++) { /* reverse the low `bits` bits of i */
        int r = 0;
        for (int b = 0; b < bits; b++)
            if (i & (1 << b)) r |= 1 << (bits - 1 - b);
        p->rev[i] = r;
    }
    return 0;
}

void fft_free(FFTPlan *p) {
    free(p->cos_t);
    free(p->sin_t);
    free(p->rev);
    memset(p, 0, sizeof(*p));
}

/* Iterative Cooley-Tukey, decimation in time.
 * sign = -1 forward, +1 inverse: twiddle w = e^{sign * 2 pi i k / len}. */
static void fft_run(const FFTPlan *p, Complex *x, int sign) {
    int n = p->n;

    /* 1. reorder input into bit-reversed index order */
    for (int i = 0; i < n; i++) {
        int j = p->rev[i];
        if (i < j) {
            Complex t = x[i];
            x[i] = x[j];
            x[j] = t;
        }
    }

    /* 2. log2(n) stages. At each stage, pairs of length-half DFTs are combined
     * into length-len DFTs with the butterfly
     *     a = x[k],  b = w^k * x[k + half]
     *     x[k] = a + b,  x[k + half] = a - b
     * where w^k = e^{sign 2 pi i k / len} = table entry k * (n / len). */
    for (int len = 2; len <= n; len <<= 1) {
        int half = len / 2, step = n / len;
        for (int start = 0; start < n; start += len) {
            for (int k = 0; k < half; k++) {
                float wr = p->cos_t[k * step];
                float wi = sign * p->sin_t[k * step];
                Complex *a = &x[start + k], *b = &x[start + k + half];
                float br = b->re * wr - b->im * wi;
                float bi = b->re * wi + b->im * wr;
                b->re = a->re - br;
                b->im = a->im - bi;
                a->re += br;
                a->im += bi;
            }
        }
    }
}

void fft_forward(const FFTPlan *p, Complex *x) { fft_run(p, x, -1); }

void fft_inverse(const FFTPlan *p, Complex *x) { fft_run(p, x, +1); }
