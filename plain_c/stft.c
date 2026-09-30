#include "stft.h"

#include <math.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void stft_hann(float *w, int n) {
    /* DSPCore::create_hann_window writes 2.0f * M_PI * n / n_fft: M_PI is a double,
     * so the whole expression is evaluated in double and rounded to float once */
    for (int i = 0; i < n; i++) w[i] = (float)(0.5 * (1.0 - cos(2.0 * M_PI * i / n)));
}

void stft_reflect_pad(float *y, const float *x, long n, int pad) {
    for (int i = 0; i < pad; i++) y[i] = x[pad - 1 - i];
    memcpy(y + pad, x, (size_t)n * sizeof(float));
    for (int i = 0; i < pad; i++) y[pad + n + i] = x[n - 1 - i];
}

void stft_frame(const FFTPlan *p, const float *w, const float *x, Complex *X) {
    for (int i = 0; i < p->n; i++) {
        X[i].re = w[i] * x[i];
        X[i].im = 0.0f;
    }
    fft_forward(p, X);
}

void istft_frame(const FFTPlan *p, const float *w, const Complex *X, Complex *scratch, float *y) {
    memcpy(scratch, X, (size_t)p->n * sizeof(Complex));
    fft_inverse(p, scratch);
    for (int i = 0; i < p->n; i++) y[i] = w[i] * (scratch[i].re / p->n);
}

void pack_chunk(float *t, const Complex *L, const Complex *R, int n_valid, int T, int F, int n_fft) {
    long plane = (long)F * T; /* one of the 4 channels */
    memset(t, 0, 4 * (size_t)plane * sizeof(float));
    for (int fr = 0; fr < n_valid; fr++) {
        const Complex *l = L + (long)fr * n_fft, *r = R + (long)fr * n_fft;
        for (int f = 3; f < F; f++) { /* bins 0, 1, 2 stay zero */
            long i = (long)f * T + fr;
            t[0 * plane + i] = l[f].re;
            t[1 * plane + i] = l[f].im;
            t[2 * plane + i] = r[f].re;
            t[3 * plane + i] = r[f].im;
        }
    }
}

void unpack_chunk(Complex *L, Complex *R, const float *t, int n_valid, int T, int F, int n_fft) {
    long plane = (long)F * T;
    int nyq = n_fft / 2;
    for (int fr = 0; fr < n_valid; fr++) {
        Complex *l = L + (long)fr * n_fft, *r = R + (long)fr * n_fft;
        for (int f = 0; f < F; f++) {
            long i = (long)f * T + fr;
            l[f].re = t[0 * plane + i];
            l[f].im = t[1 * plane + i];
            r[f].re = t[2 * plane + i];
            r[f].im = t[3 * plane + i];
        }
        for (int f = F; f <= nyq; f++) { /* bins the model doesn't produce, incl. Nyquist */
            l[f].re = l[f].im = 0.0f;
            r[f].re = r[f].im = 0.0f;
        }
        l[0].im = 0.0f; /* DC of a real signal is real */
        r[0].im = 0.0f;
        for (int k = 1; k < nyq; k++) { /* upper half = conjugate mirror of the lower half */
            l[n_fft - k].re = l[k].re;
            l[n_fft - k].im = -l[k].im;
            r[n_fft - k].re = r[k].re;
            r[n_fft - k].im = -r[k].im;
        }
    }
}
