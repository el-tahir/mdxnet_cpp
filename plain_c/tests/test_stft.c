/* M6 check: stft.c.
 *
 * 1. Hann window: periodic, peak 1, and sum of w^2 over the 4 overlapping frames
 *    at hop n/4 is 1.5 (the constant the pipeline divides by).
 * 2. Reflect pad matches DSPCore::pad_audio on a small example.
 * 3. Analysis + synthesis with no model in between, framed exactly like
 *    src/main.cpp: pad 2048, frames every 1024 from offset 0, ISTFT, overlap-add,
 *    crop, / 1.5. The result must equal x[p] * S(p) / 1.5, where S(p) is the sum
 *    of w^2 over the frames covering p. S = 1.5 in the interior, so the interior
 *    reproduces the input; near both ends fewer frames overlap and the C++
 *    pipeline attenuates the signal - we reproduce that too.
 * 4. pack_chunk / unpack_chunk: tensor layout, zeroed bins/frames, and that
 *    unpack(pack(STFT)) gives back the STFT except the bins the pipeline drops.
 *
 *     ./test_stft
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../stft.h"

#define N_FFT 4096
#define HOP 1024
#define PAD (N_FFT / 2)
#define F 2048
#define T 256

static int n_failed;

static void result(int ok, const char *fmt, double v) {
    if (!ok) n_failed++;
    printf("%-4s ", ok ? "ok" : "FAIL");
    printf(fmt, v);
    printf("\n");
}

static unsigned long long rng_state = 88172645463325252ull;
static float frand(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((rng_state >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0);
}

static double snr_db(const float *got, const double *want, long n) {
    double sig = 0, err = 0;
    for (long i = 0; i < n; i++) {
        sig += want[i] * want[i];
        err += (got[i] - want[i]) * (got[i] - want[i]);
    }
    return err > 0 ? 10 * log10(sig / err) : INFINITY;
}

int main(void) {
    FFTPlan plan;
    if (fft_init(&plan, N_FFT) != 0) return 1;
    float w[N_FFT];
    stft_hann(w, N_FFT);

    /* 1. window */
    {
        double worst = 0;
        for (int i = 1; i < N_FFT; i++) worst = fmax(worst, fabs(w[i] - w[N_FFT - i]));
        result(w[0] == 0.0f && w[N_FFT / 2] == 1.0f && worst < 1e-6, "hann: w[0]=0, w[n/2]=1, periodic symmetry (max asym %.1e)", worst);
        double dev = 0;
        for (int i = 0; i < HOP; i++) {
            double s = 0;
            for (int j = 0; j < N_FFT / HOP; j++) s += (double)w[i + j * HOP] * w[i + j * HOP];
            dev = fmax(dev, fabs(s - 1.5));
        }
        result(dev < 1e-6, "hann: sum of w^2 at hop n/4 = 1.5 (max dev %.1e)", dev);
    }

    /* 2. reflect pad */
    {
        float x[5] = {1, 2, 3, 4, 5}, y[5 + 2 * 3];
        float want[11] = {3, 2, 1, 1, 2, 3, 4, 5, 5, 4, 3};
        stft_reflect_pad(y, x, 5, 3);
        result(!memcmp(y, want, sizeof(want)), "reflect pad matches DSPCore::pad_audio%.0s", 0);
    }

    /* 3. analysis + synthesis round trip */
    long n = 3 * 44100 + 123; /* not a multiple of the hop */
    float *x = malloc(n * sizeof(float));
    for (long i = 0; i < n; i++) x[i] = 0.5f * frand();

    long plen = n + 2 * PAD;
    float *padded = malloc(plen * sizeof(float));
    stft_reflect_pad(padded, x, n, PAD);

    long n_frames = 0;
    for (long off = 0; off + N_FFT <= plen; off += HOP) n_frames++;
    Complex *frames = malloc((size_t)n_frames * N_FFT * sizeof(Complex));
    for (long fr = 0; fr < n_frames; fr++) stft_frame(&plan, w, padded + fr * HOP, frames + fr * N_FFT);

    float *recon = calloc(plen, sizeof(float)), *y = malloc(N_FFT * sizeof(float));
    Complex *scratch = malloc(N_FFT * sizeof(Complex));
    double *S = calloc(plen, sizeof(double));
    for (long fr = 0; fr < n_frames; fr++) {
        istft_frame(&plan, w, frames + fr * N_FFT, scratch, y);
        for (int i = 0; i < N_FFT; i++) {
            recon[fr * HOP + i] += y[i];
            S[fr * HOP + i] += (double)w[i] * w[i];
        }
    }
    float *out = malloc(n * sizeof(float));
    double *want = malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) {
        out[i] = recon[PAD + i] / 1.5f;
        want[i] = x[i] * S[PAD + i] / 1.5;
    }
    result(snr_db(out, want, n) > 90, "round trip == x * S(p) / 1.5 everywhere: SNR %.1f dB", snr_db(out, want, n));

    long edge = N_FFT; /* the interior has full overlap, so it reproduces x itself */
    double *xd = malloc(n * sizeof(double));
    for (long i = 0; i < n; i++) xd[i] = x[i];
    double interior = snr_db(out + edge, xd + edge, n - 2 * edge);
    result(interior > 90, "round trip == x in the interior: SNR %.1f dB", interior);
    double gmin = 1;
    for (long i = 0; i < n; i++) gmin = fmin(gmin, S[PAD + i] / 1.5);
    printf("     (edge gain inherited from the C++ framing: min %.3f in the first/last ~1024 samples)\n", gmin);

    /* 4. pack / unpack */
    {
        int n_valid = 100;
        const Complex *L = frames, *R = frames + 7 * N_FFT; /* any two frame sequences */
        float *t = malloc(4 * (size_t)F * T * sizeof(float));
        for (long i = 0; i < 4L * F * T; i++) t[i] = NAN; /* pack must overwrite everything */
        pack_chunk(t, L, R, n_valid, T, F, N_FFT);

        int layout_ok = 1;
        for (int c = 0; c < 4; c++)
            for (int f = 0; f < F; f++)
                for (int fr = 0; fr < T; fr++) {
                    float v = t[((long)c * F + f) * T + fr];
                    const Complex *src = (c < 2 ? L : R) + (long)fr * N_FFT + f;
                    float expect = (fr >= n_valid || f < 3) ? 0.0f : (c % 2 == 0 ? src->re : src->im);
                    if (v != expect) layout_ok = 0;
                }
        result(layout_ok, "pack: t[c][f][t] layout, bins 0-2 and frames >= n_valid zero%.0s", 0);

        Complex *L2 = malloc((size_t)n_valid * N_FFT * sizeof(Complex));
        Complex *R2 = malloc((size_t)n_valid * N_FFT * sizeof(Complex));
        unpack_chunk(L2, R2, t, n_valid, T, F, N_FFT);
        /* expected: the original spectrum (conjugate symmetric, it came from a real
         * signal) with the bins the pipeline drops set to zero: 0,1,2 on pack, the
         * Nyquist bin on unpack, and their mirrors n-1, n-2 */
        double maxdiff = 0, maxref = 0;
        for (int fr = 0; fr < n_valid; fr++)
            for (int side = 0; side < 2; side++) {
                const Complex *orig = (side ? R : L) + (long)fr * N_FFT;
                const Complex *got = (side ? R2 : L2) + (long)fr * N_FFT;
                for (int k = 0; k < N_FFT; k++) {
                    int dropped = k <= 2 || k >= N_FFT - 2 || k == N_FFT / 2;
                    double er = dropped ? 0 : orig[k].re, ei = dropped ? 0 : orig[k].im;
                    maxdiff = fmax(maxdiff, hypot(got[k].re - er, got[k].im - ei));
                    maxref = fmax(maxref, hypot(er, ei));
                }
            }
        result(maxdiff / maxref < 1e-5, "unpack(pack(X)) == X minus dropped bins: rel err %.2e", maxdiff / maxref);
        free(t), free(L2), free(R2);
    }

    free(x), free(padded), free(frames), free(recon), free(y), free(scratch), free(S), free(out), free(want), free(xd);
    fft_free(&plan);
    printf("%s\n", n_failed ? "FAILED" : "OK");
    return n_failed ? 1 : 0;
}
