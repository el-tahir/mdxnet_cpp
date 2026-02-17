#pragma once
#include <vector>
#include <utility>
#include "kiss_fft.h"

// convert seperate left/right STFT frames into the interleaved tensor format expected by MDX-net
std::vector<float> stft_to_tensor(const std::vector<std::vector<kiss_fft_cpx>>& left_stft, const std::vector<std::vector<kiss_fft_cpx>>& right_stft);

// convert the interlearved tensor output back into separate STFT frames
std::pair<std::vector<std::vector<kiss_fft_cpx>>, std::vector<std::vector<kiss_fft_cpx>>> tensor_to_stft(const std::vector<float>& model_output);
