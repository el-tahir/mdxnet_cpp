#include <vector>
#include <utility>
#include "kiss_fft.h"

std::vector<float> stft_to_tensor(const std::vector<std::vector<kiss_fft_cpx>>& left_stft, const std::vector<std::vector<kiss_fft_cpx>>& right_stft) {
    // 256 frames -> each 4096 bins

    // since shape is [4, 2048, 256], index for (channel, freq, time) is 
    // index = (channel * 2048 * 256) + (freq * 256) + time

    std::vector<float> output;
    output.resize(4 * 2048 * 256);

    int stride = 2048 * 256;

    for (size_t f = 0; f < 2048; f++) {
        for (size_t t = 0; t < 256; t++) {
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

    int stride = 2048 * 256;


    // fill lower half
    for (size_t f = 0; f < 2048; f++) {
        for (size_t t = 0; t < 256; t++) {


            left_stft[t][f].r = model_output[0 * stride + f * 256 + t];
            left_stft[t][f].i = model_output[1 * stride + f * 256 + t];

            right_stft[t][f].r = model_output[2 * stride + f * 256 + t];
            right_stft[t][f].i = model_output[3 * stride + f * 256 + t];

        }
    }

    // reconstruct upper half
    // conjugate symmetry X[N - k] = conj(X[k])
    // real part same, imaginary part negated

    for (size_t t = 0; t < 256; t++) {

        left_stft[t][0].i = 0.0f;
        right_stft[t][0].i = 0.0f;

        left_stft[t][2048].i = 0.0f;
        right_stft[t][2048].i = 0.0f;

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