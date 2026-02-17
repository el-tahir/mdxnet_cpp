#include "utils.h"
#include <vector>
#include <utility>
#include <algorithm>

std::vector<float> stft_to_tensor(const std::vector<std::vector<kiss_fft_cpx>>& left_stft, const std::vector<std::vector<kiss_fft_cpx>>& right_stft) {
    // 256 frames -> each 4096 bins

    // since shape is [4, 2048, 256], index for (channel, freq, time) is
    // index = (channel * 2048 * 256) + (freq * 256) + time

    std::vector<float> output;
    output.resize(4 * 2048 * 256, 0.0f);  // Zero-initialize for partial batches

    int stride = 2048 * 256;

    size_t num_valid_frames = std::min(left_stft.size(), (size_t)256);

    for (size_t t = 0; t < num_valid_frames; t++) {
        // start from f=3 to zero out first 3 bins
        for (size_t f = 3; f < 2048; f++) {
            output[0 * stride + f * 256 + t] = left_stft[t][f].r;
            output[1 * stride + f * 256 + t] = left_stft[t][f].i;
            output[2 * stride + f * 256 + t] = right_stft[t][f].r;
            output[3 * stride + f * 256 + t] = right_stft[t][f].i;
        }
    }

    return output;


}

std::pair<std::vector<std::vector<kiss_fft_cpx>>, std::vector<std::vector<kiss_fft_cpx>>> tensor_to_stft(const std::vector<float>& model_output) {


    std::vector<std::vector<kiss_fft_cpx>> left_stft(256, std::vector<kiss_fft_cpx>(4096));
    std::vector<std::vector<kiss_fft_cpx>> right_stft(256, std::vector<kiss_fft_cpx>(4096));

    for (size_t t = 0; t < 256; t++) {
        for (size_t f = 0; f < 4096; f++) {
            left_stft[t][f].r = 0.0f;
            left_stft[t][f].i = 0.0f;
            right_stft[t][f].r = 0.0f;
            right_stft[t][f].i = 0.0f;
        }
    }

    int stride = 2048 * 256;

    // fill lower half (bins 0 to 2047 - the first 2048 bins)
    for (size_t t = 0; t < 256; t++) {
        for (size_t f = 0; f < 2048; f++) {
            left_stft[t][f].r = model_output[0 * stride + f * 256 + t];
            left_stft[t][f].i = model_output[1 * stride + f * 256 + t];

            right_stft[t][f].r = model_output[2 * stride + f * 256 + t];
            right_stft[t][f].i = model_output[3 * stride + f * 256 + t];
        }
    }

    // reconstruct upper half using conjugate symmetry: X[N - k] = conj(X[k])
    // for real input signals: real part same, imaginary part negated
    for (size_t t = 0; t < 256; t++) {
        // DC bin (f=0): imaginary part should be 0 for real signals
        left_stft[t][0].i = 0.0f;
        right_stft[t][0].i = 0.0f;

        // Nyquist bin (f=2048 for n_fft=4096): imaginary part should be 0 for real signals
        // the model doesn't output the Nyquist bin directly, so we set it to 0
        left_stft[t][2048].r = 0.0f;
        left_stft[t][2048].i = 0.0f;
        right_stft[t][2048].r = 0.0f;
        right_stft[t][2048].i = 0.0f;

        // mirror bins 1 to 2047 to 4095 to 2049
        for (size_t f = 1; f < 2048; f++) {
            size_t mirror_f = 4096 - f;

            left_stft[t][mirror_f].r = left_stft[t][f].r;
            left_stft[t][mirror_f].i = -left_stft[t][f].i;

            right_stft[t][mirror_f].r = right_stft[t][f].r;
            right_stft[t][mirror_f].i = -right_stft[t][f].i;
        }
    }

    return {left_stft, right_stft};

}
