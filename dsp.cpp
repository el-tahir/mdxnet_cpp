#include "kiss_fft.h"
#include <cstdint>
#include <iostream>
#include <vector>

class DSPCore {
    private:

    kiss_fft_cfg forward;
    kiss_fft_cfg inverse;

    uint32_t n_fft; //frame size
    uint32_t hop_length;

    std::vector<float> window; //hann window

    void create_hann_window();

    public:

    DSPCore(uint32_t n_fft, uint32_t hop_length);
    ~DSPCore();

    std::vector<kiss_fft_cpx> stft(const std::vector<float>& frame);
    std::vector<float> istft(const std::vector<kiss_fft_cpx>& frame);
};

DSPCore::DSPCore(uint32_t n_fft, uint32_t hop_length)
:n_fft(n_fft), hop_length(hop_length) {

    forward = kiss_fft_alloc(n_fft, 0, nullptr, nullptr);

    inverse = kiss_fft_alloc(n_fft, 1, nullptr, nullptr);

    create_hann_window();

}

DSPCore::~DSPCore() {
    kiss_fft_free(forward);
    kiss_fft_free(inverse);

}

void DSPCore::create_hann_window() {
    window.resize(n_fft);

    for (uint32_t n = 0; n < n_fft; n++) {
        window[n] = 0.5f * (1.0f - std::cos(2.0f * M_PI * n / (n_fft - 1)));
    }
}


std::vector<kiss_fft_cpx> DSPCore::stft(const std::vector<float>& frame) {

    if (frame.size() != n_fft) {
        throw std::runtime_error("input size != n_fft");
    }

    std::vector<kiss_fft_cpx> windowed;
    windowed.resize(n_fft);

    for (uint32_t i = 0; i < n_fft; i++) {
        windowed[i].r = window[i] * frame[i];
        windowed[i].i = 0;
    }

    std::vector<kiss_fft_cpx> output;
    output.resize(n_fft);

    kiss_fft(forward, windowed.data(), output.data());

    return output;
}

std::vector<float> DSPCore::istft(const std::vector<kiss_fft_cpx>& frame) {
    
    if (frame.size() != n_fft) {
        throw std::runtime_error("input size != n_fft");
    }

    std::vector<kiss_fft_cpx> output(n_fft);

    kiss_fft(inverse, frame.data(), output.data());

    std::vector<float> result(n_fft);

    for (uint32_t i = 0; i < n_fft; i++) {
        result[i] = (output[i].r / n_fft);
    }

    return result;


}