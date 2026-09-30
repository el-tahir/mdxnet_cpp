/* separator: vocal removal with UVR_MDXNET_KARA_2, in plain C.
 *
 *     ./separator input.(wav|mp3|flac|...) output.wav [models/kara.bin]
 *
 * Step for step the same pipeline as src/main.cpp (the C++/ONNX Runtime
 * reference), see PLAN.md section 5:
 *
 *   ffmpeg (once, if available) -> 44.1 kHz stereo 16-bit WAV
 *   read WAV, split L/R
 *   reflect-pad 2048, Hann-windowed 4096-point STFT every 1024 samples
 *   chunks of 256 frames -> [4][2048][256] tensor -> MDX-Net -> back to frames
 *   ISTFT, overlap-add, crop the padding, / 1.5
 *   noise gate (-40 dB), write float32 WAV
 *
 * One difference in *how*, not *what*: the C++ keeps the STFT of the whole song
 * in memory; here each chunk goes STFT -> model -> ISTFT -> overlap-add before
 * the next one starts. Frames are independent and are added in the same order,
 * so the output is the same.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "fft.h"
#include "mdx.h"
#include "stft.h"
#include "wav.h"

#define N_FFT 4096
#define HOP 1024 /* 75% overlap */
#define PAD (N_FFT / 2)

/* ------------------------------------------------------------------------- */
/* ffmpeg: the only external program, run once before anything else */

/* Appends s to cmd wrapped in single quotes, safe for any path. */
static void append_quoted(char *cmd, size_t cap, const char *s) {
    size_t n = strlen(cmd);
    if (n + 1 < cap) cmd[n++] = '\'';
    for (; *s && n + 5 < cap; s++) {
        if (*s == '\'') { /* ' -> '\'' */
            memcpy(cmd + n, "'\\''", 4);
            n += 4;
        } else {
            cmd[n++] = *s;
        }
    }
    if (n + 1 < cap) cmd[n++] = '\'';
    cmd[n] = '\0';
}

/* Converts input to 16-bit 44.1 kHz stereo WAV in a temp file. On success
 * returns 1 and writes the temp path to out; on failure returns 0 and the
 * caller uses the original file (same fallback as the C++). */
static int preprocess_input(const char *input, char *out, size_t out_cap) {
    srand((unsigned)time(NULL));
    snprintf(out, out_cap, "temp_input_%d.wav", rand());

    /* -map_metadata -1 -fflags +bitexact: no extra chunks, so the WAV header
     * is exactly 44 bytes, which is all wav_read understands */
    char cmd[8192] = "ffmpeg -y -i ";
    append_quoted(cmd, sizeof(cmd), input);
    strncat(cmd, " -map_metadata -1 -fflags +bitexact -acodec pcm_s16le -ar 44100 -ac 2 ",
            sizeof(cmd) - strlen(cmd) - 1);
    append_quoted(cmd, sizeof(cmd), out);
    strncat(cmd, " -loglevel error", sizeof(cmd) - strlen(cmd) - 1);

    printf("preprocessing input with ffmpeg...\n");
    if (system(cmd) != 0) {
        fprintf(stderr, "ffmpeg preprocessing failed or ffmpeg not found. trying original file...\n");
        remove(out);
        return 0;
    }
    return 1;
}

/* ------------------------------------------------------------------------- */

/* Zeroes each stereo frame whose RMS over the surrounding +-window interleaved
 * samples is below threshold_db. In place, and later windows see earlier
 * zeroed samples, exactly as apply_noise_gate in the C++. */
static void noise_gate(float *st, long n_values, float threshold_db, int window) {
    if (n_values < 2) return;
    float threshold = powf(10.0f, threshold_db / 20.0f);
    for (long i = 0; i < n_values; i += 2) {
        float sum_sq = 0.0f;
        int count = 0;
        long start = i - window > 0 ? i - window : 0;
        long end = i + window < n_values ? i + window : n_values;
        for (long j = start; j < end; j++) {
            sum_sq += st[j] * st[j];
            count++;
        }
        float rms = sqrtf(sum_sq / count);
        if (rms < threshold) {
            st[i] = 0.0f;
            if (i + 1 < n_values) st[i + 1] = 0.0f;
        }
    }
}

static int run_separation(const char *input_path, const char *output_path, const char *model_path) {
    int rc = -1;
    WavHeader header;
    float *stereo = NULL, *left = NULL, *right = NULL, *lpad = NULL, *rpad = NULL, *lrec = NULL, *rrec = NULL;
    float *tensor = NULL, *processed = NULL, *window = NULL, *frame_out = NULL;
    Complex *lfr = NULL, *rfr = NULL, *scratch = NULL;
    FFTPlan plan = {0};
    MdxModel model = {0};
    MdxState state = {0};
    long n;

    /* read, split channels */
    printf("loading %s...\n", input_path);
    if (wav_read(input_path, &header, &stereo, &n) != 0) return -1;
    if (n < PAD) { /* reflect padding needs at least PAD samples (the C++ reads out of bounds here) */
        fprintf(stderr, "input too short: %ld samples, need at least %d\n", n, PAD);
        goto done;
    }
    left = malloc(n * sizeof(float));
    right = malloc(n * sizeof(float));
    if (!left || !right) goto oom;
    for (long i = 0; i < n; i++) left[i] = stereo[2 * i], right[i] = stereo[2 * i + 1];

    /* model */
    if (mdx_load(&model, model_path) != 0) goto done;
    const MdxConfig *cfg = &model.config;
    int T = (int)cfg->dim_t, F = (int)cfg->dim_f, C = (int)cfg->dim_c;
    if (C != 4 || F > N_FFT / 2) {
        fprintf(stderr, "model shape (dim_c=%d, dim_f=%d) does not fit this pipeline\n", C, F);
        goto done;
    }
    printf("model loaded successfully: %s\n", model_path);
    if (mdx_state_init(&state, cfg, T) != 0) goto done;

    /* analysis setup */
    if (fft_init(&plan, N_FFT) != 0) goto oom;
    window = malloc(N_FFT * sizeof(float));
    long plen = n + 2 * PAD;
    lpad = malloc(plen * sizeof(float));
    rpad = malloc(plen * sizeof(float));
    lrec = calloc(plen, sizeof(float));
    rrec = calloc(plen, sizeof(float));
    lfr = malloc((size_t)T * N_FFT * sizeof(Complex));
    rfr = malloc((size_t)T * N_FFT * sizeof(Complex));
    scratch = malloc(N_FFT * sizeof(Complex));
    frame_out = malloc(N_FFT * sizeof(float));
    tensor = malloc((size_t)C * F * T * sizeof(float));
    processed = malloc((size_t)C * F * T * sizeof(float));
    if (!window || !lpad || !rpad || !lrec || !rrec || !lfr || !rfr || !scratch || !frame_out || !tensor || !processed)
        goto oom;
    stft_hann(window, N_FFT);
    stft_reflect_pad(lpad, left, n, PAD);
    stft_reflect_pad(rpad, right, n, PAD);

    long n_frames = 0;
    for (long off = 0; off + N_FFT <= plen; off += HOP) n_frames++;
    long n_chunks = (n_frames + T - 1) / T;
    printf("running inference on %ld frames (%ld chunks of %d)...\n", n_frames, n_chunks, T);

    for (long c = 0; c < n_chunks; c++) {
        long first = c * T;
        int valid = (int)(n_frames - first < T ? n_frames - first : T);
        clock_t t0 = clock();

        /* STFT of this chunk's frames */
        for (int k = 0; k < valid; k++) {
            long off = (first + k) * HOP;
            stft_frame(&plan, window, lpad + off, lfr + (long)k * N_FFT);
            stft_frame(&plan, window, rpad + off, rfr + (long)k * N_FFT);
        }

        /* model */
        pack_chunk(tensor, lfr, rfr, valid, T, F, N_FFT);
        mdx_forward(&model, &state, tensor, processed);
        unpack_chunk(lfr, rfr, processed, valid, T, F, N_FFT);

        /* ISTFT + overlap-add, frames in order */
        for (int k = 0; k < valid; k++) {
            long off = (first + k) * HOP;
            istft_frame(&plan, window, lfr + (long)k * N_FFT, scratch, frame_out);
            for (int i = 0; i < N_FFT; i++) lrec[off + i] += frame_out[i];
            istft_frame(&plan, window, rfr + (long)k * N_FFT, scratch, frame_out);
            for (int i = 0; i < N_FFT; i++) rrec[off + i] += frame_out[i];
        }
        printf("  chunk %ld/%ld (%d frames): %.1fs\n", c + 1, n_chunks, valid, (double)(clock() - t0) / CLOCKS_PER_SEC);
        fflush(stdout);
    }

    /* crop the padding, interleave, COLA normalisation: with a periodic Hann
     * window applied in both STFT and ISTFT at 75% overlap, sum w^2 = 1.5 */
    for (long i = 0; i < n; i++) {
        stereo[2 * i] = lrec[PAD + i];
        stereo[2 * i + 1] = rrec[PAD + i];
    }
    for (long i = 0; i < 2 * n; i++) stereo[i] /= 1.5f;

    noise_gate(stereo, 2 * n, -40.0f, 2048);

    if (wav_write(output_path, &header, stereo, n) != 0) goto done;
    rc = 0;
    goto done;

oom:
    fprintf(stderr, "out of memory\n");
done:
    free(stereo), free(left), free(right), free(lpad), free(rpad), free(lrec), free(rrec);
    free(tensor), free(processed), free(window), free(frame_out), free(lfr), free(rfr), free(scratch);
    fft_free(&plan);
    mdx_state_free(&state);
    mdx_free(&model);
    return rc;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("usage: ./separator <input> <output.wav> [model.bin]\n");
        return 1;
    }
    const char *model_path = argc > 3 ? argv[3] : "models/kara.bin";

    char temp[256];
    int needs_cleanup = preprocess_input(argv[1], temp, sizeof(temp));
    const char *input = needs_cleanup ? temp : argv[1];

    int rc = run_separation(input, argv[2], model_path);
    if (needs_cleanup) remove(temp);
    if (rc != 0) {
        fprintf(stderr, "error: separation failed\n");
        return 1;
    }
    printf("done! saved to %s\n", argv[2]);
    return 0;
}
