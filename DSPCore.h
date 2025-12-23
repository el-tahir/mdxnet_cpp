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

    std::vector<float> pad_audio(const std::vector<float>& audio);

    std::vector<float> process(const std::vector<float>& audio);

};