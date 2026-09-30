/* Compare two float32 stereo WAVs (e.g. the C++ and plain C separator outputs).
 *
 *     ./compare_wav reference.wav test.wav [min_snr_db]
 *
 * Prints whether the headers are byte-identical, the SNR of test against
 * reference, the max absolute difference, and how many samples are exactly
 * zero in one file but not the other (noise gate decisions). Exits 1 if the
 * SNR is below min_snr_db (default 60).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../wav.h"

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s reference.wav test.wav [min_snr_db]\n", argv[0]);
        return 2;
    }
    double min_snr = argc > 3 ? atof(argv[3]) : 60.0;
    WavHeader ha, hb;
    float *a, *b;
    long na, nb;
    if (wav_read(argv[1], &ha, &a, &na) != 0 || wav_read(argv[2], &hb, &b, &nb) != 0) return 2;

    printf("headers:        %s\n", memcmp(ha.raw, hb.raw, WAV_HEADER_SIZE) ? "DIFFER" : "identical");
    printf("samples/chan:   %ld vs %ld\n", na, nb);
    if (na != nb) {
        printf("FAIL length differs\n");
        return 1;
    }

    double sig = 0, err = 0, maxdiff = 0, peak = 0;
    long gate_mismatch = 0, identical = 0;
    for (long i = 0; i < 2 * na; i++) {
        double d = (double)b[i] - a[i];
        sig += (double)a[i] * a[i];
        err += d * d;
        if (fabs(d) > maxdiff) maxdiff = fabs(d);
        if (fabs(a[i]) > peak) peak = fabs(a[i]);
        if ((a[i] == 0.0f) != (b[i] == 0.0f)) gate_mismatch++;
        if (a[i] == b[i]) identical++;
    }
    double snr = err > 0 ? 10 * log10(sig / err) : INFINITY;
    printf("SNR:            %.1f dB\n", snr);
    printf("max |diff|:     %.3g (reference peak %.3g)\n", maxdiff, peak);
    printf("bit-identical:  %ld of %ld values (%.2f%%)\n", identical, 2 * na, 100.0 * identical / (2 * na));
    printf("zero in only one file: %ld values\n", gate_mismatch);

    int ok = snr >= min_snr;
    printf("%s (threshold %.0f dB)\n", ok ? "OK" : "FAIL", min_snr);
    free(a);
    free(b);
    return ok ? 0 : 1;
}
