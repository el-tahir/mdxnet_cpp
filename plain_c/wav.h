/* WAV read/write, mirroring include/WAVHeader.h of the C++ reference. */
#ifndef WAV_H
#define WAV_H

#include <stdint.h>

#define WAV_HEADER_SIZE 44

/* The canonical 44-byte header: RIFF chunk, 16-byte "fmt " chunk, "data" chunk.
 * Like the C++ reader, no other chunks are expected (the input comes from
 * ffmpeg with -fflags +bitexact -map_metadata -1). */
typedef struct {
    unsigned char raw[WAV_HEADER_SIZE]; /* as read; wav_write patches the format fields */
    uint16_t audio_format;              /* 1 = PCM, 3 = IEEE float */
    uint16_t num_channels;
    uint32_t sample_rate;
    uint16_t bits_per_sample;
    uint32_t data_size;                 /* bytes of sample data */
} WavHeader;

/* Reads a 44.1 kHz PCM16 or float32 WAV, mono or stereo. Returns interleaved
 * stereo float samples (mono is duplicated to both channels) in *stereo
 * (caller frees) and the number of samples per channel in *n. PCM16 is scaled
 * by 1/32768. Returns 0 on success, -1 on error (message on stderr). */
int wav_read(const char *path, WavHeader *h, float **stereo, long *n);

/* Writes interleaved stereo float32 using h's header bytes with the format
 * fields rewritten (float, 2 channels, 32 bit, sizes), as the C++ does. */
int wav_write(const char *path, const WavHeader *h, const float *stereo, long n);

#endif
