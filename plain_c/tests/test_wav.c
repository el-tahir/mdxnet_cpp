/* M7 check: wav.c reads and writes the way include/WAVHeader.h does.
 *
 *     ./test_wav
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../wav.h"

static int n_failed;
static const char *TMP = "_test_wav_tmp.wav";
static const char *TMP2 = "_test_wav_tmp2.wav";

static void result(int ok, const char *what) {
    if (!ok) n_failed++;
    printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
}

static void put16(unsigned char *p, unsigned v) { p[0] = v & 255, p[1] = (v >> 8) & 255; }
static void put32(unsigned char *p, unsigned long v) {
    for (int i = 0; i < 4; i++) p[i] = (v >> (8 * i)) & 255;
}
static unsigned get16(const unsigned char *p) { return p[0] | p[1] << 8; }
static unsigned long get32(const unsigned char *p) {
    return p[0] | (unsigned long)p[1] << 8 | (unsigned long)p[2] << 16 | (unsigned long)p[3] << 24;
}

/* canonical 44-byte header followed by raw sample bytes */
static void write_raw(const char *path, unsigned fmt, unsigned ch, unsigned long rate, unsigned bits, const void *data,
                      unsigned long bytes) {
    unsigned char h[44];
    memcpy(h, "RIFF", 4), put32(h + 4, 36 + bytes), memcpy(h + 8, "WAVE", 4);
    memcpy(h + 12, "fmt ", 4), put32(h + 16, 16), put16(h + 20, fmt), put16(h + 22, ch), put32(h + 24, rate);
    put32(h + 28, rate * ch * bits / 8), put16(h + 32, ch * bits / 8), put16(h + 34, bits);
    memcpy(h + 36, "data", 4), put32(h + 40, bytes);
    FILE *f = fopen(path, "wb");
    fwrite(h, 1, 44, f);
    fwrite(data, 1, bytes, f);
    fclose(f);
}

int main(void) {
    WavHeader h;
    float *st;
    long n;

    /* PCM16 mono: scaled by 1/32768 and duplicated to both channels */
    int16_t mono[5] = {-32768, -1, 0, 1, 32767};
    write_raw(TMP, 1, 1, 44100, 16, mono, sizeof(mono));
    int ok = wav_read(TMP, &h, &st, &n) == 0 && n == 5;
    for (int i = 0; ok && i < 5; i++) ok = st[2 * i] == mono[i] / 32768.0f && st[2 * i + 1] == mono[i] / 32768.0f;
    result(ok, "pcm16 mono: /32768, duplicated to L and R");
    if (ok) free(st);

    /* PCM16 stereo: interleaving kept */
    int16_t ster[6] = {100, -200, 300, -400, 500, -600};
    write_raw(TMP, 1, 2, 44100, 16, ster, sizeof(ster));
    ok = wav_read(TMP, &h, &st, &n) == 0 && n == 3;
    for (int i = 0; ok && i < 6; i++) ok = st[i] == ster[i] / 32768.0f;
    result(ok, "pcm16 stereo: interleaved L/R kept");

    /* write: float32 stereo, header = input header with format fields patched */
    if (ok) {
        ok = wav_write(TMP2, &h, st, n) == 0;
        free(st);
        FILE *f = fopen(TMP2, "rb");
        unsigned char o[44];
        float data[6];
        ok = ok && f && fread(o, 1, 44, f) == 44 && fread(data, 4, 6, f) == 6 && fgetc(f) == EOF;
        if (f) fclose(f);
        ok = ok && !memcmp(o, "RIFF", 4) && get32(o + 4) == 36 + 24 && !memcmp(o + 8, "WAVEfmt ", 8) &&
             get32(o + 16) == 16 && get16(o + 20) == 3 && get16(o + 22) == 2 && get32(o + 24) == 44100 &&
             get32(o + 28) == 44100 * 8 && get16(o + 32) == 8 && get16(o + 34) == 32 && !memcmp(o + 36, "data", 4) &&
             get32(o + 40) == 24;
        for (int i = 0; ok && i < 6; i++) ok = data[i] == ster[i] / 32768.0f;
        result(ok, "write: float32 stereo, header fields patched like the C++");
    }

    /* float32 read of what we wrote: exact */
    ok = wav_read(TMP2, &h, &st, &n) == 0 && n == 3;
    for (int i = 0; ok && i < 6; i++) ok = st[i] == ster[i] / 32768.0f;
    result(ok, "float32 stereo read back exactly");
    if (ok) free(st);

    /* rejected inputs */
    fprintf(stderr, "  (expected errors follow)\n");
    write_raw(TMP, 1, 2, 48000, 16, ster, sizeof(ster));
    result(wav_read(TMP, &h, &st, &n) != 0, "48 kHz rejected");
    unsigned char b24[9] = {0};
    write_raw(TMP, 1, 1, 44100, 24, b24, sizeof(b24));
    result(wav_read(TMP, &h, &st, &n) != 0, "24-bit PCM rejected");
    result(wav_read("_does_not_exist.wav", &h, &st, &n) != 0, "missing file rejected");

    remove(TMP);
    remove(TMP2);
    printf("%s\n", n_failed ? "FAILED" : "OK");
    return n_failed ? 1 : 0;
}
