#include "DSPCore.h"

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

std::vector<float> DSPCore::pad_audio(const std::vector<float>& audio) {

    uint32_t pad_length = n_fft / 2;

    std::vector<float> padded(audio.size() + 2 * pad_length);

    for (uint32_t i = 0; i < pad_length; i++) {
        padded[i] = audio[pad_length - 1 - i];
    }

    for (int i = 0; i < audio.size(); i++) {
        padded[pad_length + i] = audio[i];
    }

    for (int i = 0; i < pad_length; i++) {
        padded[pad_length + audio.size() + i] = audio[audio.size() - 1 - i];
    }

    return padded;
}

std::vector<float> DSPCore::process(const std::vector<float>& audio) {
    std::vector<float> padded = pad_audio(audio);

    std::vector<float> reconstruction(padded.size(), 0.0f);

    for (uint32_t offset = 0; offset + n_fft <= padded.size(); offset += hop_length ) {

        std::vector<float> frame(n_fft);
        for (uint32_t i = 0; i < n_fft; i++) {
            frame[i] = padded[offset + i];
        }

       std::vector<kiss_fft_cpx> freq_bins = stft(frame);

       std::vector<float> processed_frame = istft(freq_bins);

        for (uint32_t i = 0; i < n_fft; i++) {
            reconstruction[offset + i] += processed_frame[i];
        }

    }

    uint32_t pad_length = n_fft / 2;

    std::vector<float> result(audio.size());

    for (uint32_t i = 0; i < audio.size(); i++) {
        result[i] = reconstruction[pad_length + i];
    }

    return result;

}