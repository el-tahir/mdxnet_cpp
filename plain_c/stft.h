/* STFT / ISTFT and the spectrogram <-> model tensor mapping.
 * Mirrors src/DSPCore.cpp and src/utils.cpp of the C++ reference exactly. */
#ifndef STFT_H
#define STFT_H

#include "fft.h"

/* periodic Hann: w[i] = 0.5 * (1 - cos(2 pi i / n)) */
void stft_hann(float *w, int n);

/* y has n + 2*pad samples. Mirrored at the edges *including* the edge sample
 * (y[pad-1] = x[0], y[pad+n] = x[n-1]), as DSPCore::pad_audio does.
 * Requires n >= pad. */
void stft_reflect_pad(float *y, const float *x, long n, int pad);

/* X[0..n) = FFT(w * x[0..n)) */
void stft_frame(const FFTPlan *p, const float *w, const float *x, Complex *X);

/* y[i] = w[i] * Re(IFFT(X))[i] / n. scratch holds n Complex; X is not modified. */
void istft_frame(const FFTPlan *p, const float *w, const Complex *X, Complex *scratch, float *y);

/* Frames -> model input tensor t[4][F][T] = (L.re, L.im, R.re, R.im)[freq][frame].
 * L, R: n_valid frames of n_fft bins each, stored back to back.
 * Frames n_valid..T-1 are zero (short last chunk). Bins 0..2 are forced to 0. */
void pack_chunk(float *t, const Complex *L, const Complex *R, int n_valid, int T, int F, int n_fft);

/* Model output tensor -> n_valid full frames of n_fft bins:
 *   bins 0..F-1 from the tensor, DC imag = 0, bins F..n_fft/2 = 0,
 *   X[n_fft-k] = conj(X[k]) for k = 1..n_fft/2-1 (spectrum of a real signal). */
void unpack_chunk(Complex *L, Complex *R, const float *t, int n_valid, int T, int F, int n_fft);

#endif
