#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include "DSPCore.h"
#include "WAVHeader.h"


int main() {
    std::string input_file = "piano.wav";
    std::string output_file = "test.wav";

    uint32_t n_fft = 4096;
    uint32_t hop_length = 2048;

    try {
        std::vector<float> stereo_buffer;

        WAVHeader header = read_wav(input_file, stereo_buffer);

        std::cout << "loaded " << input_file << ": " 
                  << header.sample_rate << "Hz, "
                  << header.num_channels << " channels, "
                  << stereo_buffer.size() << " samples." << std::endl;

        DSPCore dsp(n_fft, hop_length);

        std::vector<float> left_channel, right_channel;
        left_channel.reserve(stereo_buffer.size() / 2);
        right_channel.reserve(stereo_buffer.size() / 2);

        for (size_t i = 0; i < stereo_buffer.size(); i += 2) {
            left_channel.push_back(stereo_buffer[i]);
            right_channel.push_back(stereo_buffer[i + 1]);
        }

        std::cout << "processing left channel..." << std::endl;
        std::vector<float> left_out = dsp.process(left_channel);
        std::cout << "processing right channel..." << std::endl;
        std::vector<float> right_out = dsp.process(right_channel);

        std::vector<float> final_output;
        final_output.reserve(left_out.size() * 2);

        size_t min_size = std::min(left_out.size(), right_out.size());

        for (size_t i = 0; i < min_size; i++) {
            final_output.push_back(left_out[i]);
            final_output.push_back(right_out[i]);
        }

        write_wav(header, output_file, final_output);
        std::cout << "success! saved to " << output_file << std::endl;
        

    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}