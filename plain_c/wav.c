#include "wav.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* little-endian field access into the raw header */
static uint16_t get16(const unsigned char *p) { return (uint16_t)(p[0] | p[1] << 8); }
static uint32_t get32(const unsigned char *p) {
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static void put16(unsigned char *p, uint16_t v) {
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
}
static void put32(unsigned char *p, uint32_t v) {
    for (int i = 0; i < 4; i++) p[i] = (unsigned char)(v >> (8 * i));
}

/* header byte offsets */
enum { OFF_CHUNK_SIZE = 4, OFF_AUDIO_FORMAT = 20, OFF_CHANNELS = 22, OFF_SAMPLE_RATE = 24, OFF_BYTE_RATE = 28,
       OFF_BLOCK_ALIGN = 32, OFF_BITS = 34, OFF_DATA_SIZE = 40 };

int wav_read(const char *path, WavHeader *h, float **stereo, long *n) {
    *stereo = NULL;
    *n = 0;
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "failed to open file: %s\n", path);
        return -1;
    }
    if (fread(h->raw, 1, WAV_HEADER_SIZE, f) != WAV_HEADER_SIZE) {
        fprintf(stderr, "%s: too short for a WAV header\n", path);
        fclose(f);
        return -1;
    }
    h->audio_format = get16(h->raw + OFF_AUDIO_FORMAT);
    h->num_channels = get16(h->raw + OFF_CHANNELS);
    h->sample_rate = get32(h->raw + OFF_SAMPLE_RATE);
    h->bits_per_sample = get16(h->raw + OFF_BITS);
    h->data_size = get32(h->raw + OFF_DATA_SIZE);

    if (memcmp(h->raw, "RIFF", 4) || memcmp(h->raw + 8, "WAVE", 4) || memcmp(h->raw + 36, "data", 4)) {
        fprintf(stderr, "%s: not a plain 44-byte-header WAV\n", path);
        fclose(f);
        return -1;
    }
    if (h->sample_rate != 44100) {
        fprintf(stderr, "unsupported sample rate: %u, expected 44100\n", h->sample_rate);
        fclose(f);
        return -1;
    }
    if (!(h->audio_format == 1 && h->bits_per_sample == 16) && !(h->audio_format == 3 && h->bits_per_sample == 32)) {
        fprintf(stderr, "unsupported audio format %u / %u bit (expected 16-bit PCM or 32-bit float)\n",
                h->audio_format, h->bits_per_sample);
        fclose(f);
        return -1;
    }
    if (h->num_channels != 1 && h->num_channels != 2) {
        fprintf(stderr, "unsupported channel count: %u\n", h->num_channels);
        fclose(f);
        return -1;
    }

    long n_samples = (long)(h->data_size / (h->bits_per_sample / 8)); /* all channels */
    float *buf = calloc((size_t)n_samples + 1, sizeof(float));
    if (!buf) {
        fclose(f);
        return -1;
    }
    long got;
    if (h->audio_format == 1) {
        int16_t *tmp = calloc((size_t)n_samples + 1, sizeof(int16_t));
        if (!tmp) {
            free(buf);
            fclose(f);
            return -1;
        }
        got = (long)fread(tmp, sizeof(int16_t), (size_t)n_samples, f);
        for (long i = 0; i < n_samples; i++) buf[i] = tmp[i] / 32768.0f; /* short samples stay 0, like the C++ */
        free(tmp);
    } else {
        got = (long)fread(buf, sizeof(float), (size_t)n_samples, f);
    }
    fclose(f);
    if (got != n_samples) fprintf(stderr, "warning: %s: data chunk truncated (%ld of %ld samples)\n", path, got, n_samples);

    if (h->num_channels == 1) { /* duplicate mono to stereo */
        float *st = malloc(2 * (size_t)n_samples * sizeof(float) + 1);
        if (!st) {
            free(buf);
            return -1;
        }
        for (long i = 0; i < n_samples; i++) st[2 * i] = st[2 * i + 1] = buf[i];
        free(buf);
        *stereo = st;
        *n = n_samples;
    } else {
        /* like the C++, an odd trailing sample (malformed file) is ignored */
        *stereo = buf;
        *n = n_samples / 2;
    }
    return 0;
}

int wav_write(const char *path, const WavHeader *h, const float *stereo, long n) {
    unsigned char raw[WAV_HEADER_SIZE];
    memcpy(raw, h->raw, sizeof(raw));
    uint32_t data_size = (uint32_t)(2 * n * sizeof(float));
    put16(raw + OFF_CHANNELS, 2);
    put16(raw + OFF_BITS, 32);
    put16(raw + OFF_AUDIO_FORMAT, 3);
    put32(raw + OFF_BYTE_RATE, h->sample_rate * 2 * 4);
    put32(raw + OFF_DATA_SIZE, data_size);
    put16(raw + OFF_BLOCK_ALIGN, 2 * 4);
    put32(raw + OFF_CHUNK_SIZE, 36 + data_size);

    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "could not open file for saving: %s\n", path);
        return -1;
    }
    int ok = fwrite(raw, 1, sizeof(raw), f) == sizeof(raw) &&
             fwrite(stereo, sizeof(float), 2 * (size_t)n, f) == 2 * (size_t)n;
    ok = (fclose(f) == 0) && ok;
    if (!ok) fprintf(stderr, "error writing %s\n", path);
    return ok ? 0 : -1;
}
