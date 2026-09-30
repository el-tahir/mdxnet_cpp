/* Complex FFT, radix-2, in place. Replaces kiss_fft for the plain C port. */
#ifndef FFT_H
#define FFT_H

typedef struct {
    float re, im;
} Complex;

typedef struct {
    int n;        /* transform size, a power of two */
    float *cos_t; /* twiddles: cos(2*pi*k/n), k < n/2 */
    float *sin_t; /*           sin(2*pi*k/n) */
    int *rev;     /* bit-reversal permutation of 0..n-1 */
} FFTPlan;

/* Returns 0 on success, -1 if n is not a power of two >= 1 or out of memory. */
int fft_init(FFTPlan *p, int n);
void fft_free(FFTPlan *p);

/* In place.
 *   forward: X[k] = sum_t x[t] e^{-2 pi i k t / n}
 *   inverse: x[t] = sum_k X[k] e^{+2 pi i k t / n}    (no 1/n, same as kiss_fft) */
void fft_forward(const FFTPlan *p, Complex *x);
void fft_inverse(const FFTPlan *p, Complex *x);

#endif
