/* M6 check: fft.c against a naive O(n^2) DFT computed in double.
 *
 *     ./test_fft
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "../fft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int n_failed;

static unsigned long long rng_state = 88172645463325252ull;
static float frand(void) { /* xorshift64, uniform in [-1, 1) */
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return (float)((rng_state >> 11) * (1.0 / 9007199254740992.0) * 2.0 - 1.0);
}

/* X[k] = sum_t x[t] e^{sign 2 pi i k t / n}, in double */
static void dft(const Complex *x, double *re, double *im, int n, int sign) {
    for (int k = 0; k < n; k++) {
        double sr = 0, si = 0;
        for (int t = 0; t < n; t++) {
            double a = sign * 2.0 * M_PI * (double)((long)k * t % n) / n; /* reduce k*t mod n for accuracy */
            sr += x[t].re * cos(a) - x[t].im * sin(a);
            si += x[t].re * sin(a) + x[t].im * cos(a);
        }
        re[k] = sr;
        im[k] = si;
    }
}

static void check(const char *what, int n, const Complex *got, const double *re, const double *im, double tol) {
    double maxdiff = 0, maxref = 0;
    for (int k = 0; k < n; k++) {
        double d = hypot(got[k].re - re[k], got[k].im - im[k]);
        double r = hypot(re[k], im[k]);
        if (d > maxdiff) maxdiff = d;
        if (r > maxref) maxref = r;
    }
    double rel = maxref > 0 ? maxdiff / maxref : maxdiff;
    int ok = rel < tol;
    if (!ok) n_failed++;
    printf("%-4s %-22s n=%-5d rel err %.2e\n", ok ? "ok" : "FAIL", what, n, rel);
}

int main(void) {
    for (int n = 1; n <= 4096; n *= 2) {
        FFTPlan p;
        if (fft_init(&p, n) != 0) {
            printf("FAIL fft_init(%d)\n", n);
            return 1;
        }
        Complex *x = malloc(n * sizeof(Complex)), *y = malloc(n * sizeof(Complex));
        double *re = malloc(n * sizeof(double)), *im = malloc(n * sizeof(double));

        for (int i = 0; i < n; i++) x[i].re = frand(), x[i].im = frand();

        for (int i = 0; i < n; i++) y[i] = x[i];
        fft_forward(&p, y);
        dft(x, re, im, n, -1);
        check("forward vs DFT", n, y, re, im, 1e-5);

        for (int i = 0; i < n; i++) y[i] = x[i];
        fft_inverse(&p, y);
        dft(x, re, im, n, +1);
        check("inverse vs DFT", n, y, re, im, 1e-5);

        /* inverse(forward(x)) = n * x */
        for (int i = 0; i < n; i++) y[i] = x[i];
        fft_forward(&p, y);
        fft_inverse(&p, y);
        for (int i = 0; i < n; i++) re[i] = (double)n * x[i].re, im[i] = (double)n * x[i].im;
        check("inverse(forward(x))", n, y, re, im, 1e-5);

        free(x), free(y), free(re), free(im);
        fft_free(&p);
    }

    /* non powers of two are rejected */
    int bad[] = {0, -4, 3, 6, 4095};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        FFTPlan p;
        if (fft_init(&p, bad[i]) == 0) {
            printf("FAIL fft_init(%d) accepted\n", bad[i]);
            n_failed++;
            fft_free(&p);
        }
    }

    printf("%s\n", n_failed ? "FAILED" : "OK");
    return n_failed ? 1 : 0;
}
