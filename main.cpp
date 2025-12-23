#include <iostream>
#include <vector>
#include <fstream>
#include <cstdint>
#include "DSPCore.h"


#pragma pack(push, 1)

struct WAVHeader {
    // RIFF chunk
    char chunk_id[4]; // "RIFF"
    uint32_t chunk_size; //file size - 8
    char format[4]; // "WAVE"
    // fmt sub-chunk
    char subchunk1_id[4]; // "fmt "
    uint32_t subchunk1_size; // 16 for PCM
    uint16_t audio_format; // 1 for PCM, 3 for float
    uint16_t num_channels; // 1 for mono, 2 for stereo
    uint32_t sample_rate; // e.g. 44100
    uint32_t byte_rate; // sample_rate * num_channels * bits_per_sample/8
    uint16_t block_align; // num_channels * bits_per_sample/8
    uint16_t bits_per_sample; // 16 or 32
    // data sub_chunk
    char subchunk2_id[4]; // "data"
    uint32_t subchunk2_size; // num_samples * num_channels * bits_per_sample/8
};
#pragma pack(pop)

WAVHeader read_wav(const std::string& full_path, std::vector<float>& stereo_buffer) {
    WAVHeader header;
    std::ifstream wav_file(full_path, std::ios::binary);
    if (!wav_file) throw std::runtime_error("failed to open file: " + full_path);
    wav_file.read(reinterpret_cast<char*> (&header), sizeof(WAVHeader));
    if (header.sample_rate != 44100) throw std::runtime_error("unsupported sample rate: " + std::to_string(header.sample_rate) + "expected 44100");
    if (header.audio_format != 1 && header.audio_format != 3 ) throw std::runtime_error("unsupported audio format: " + std::to_string(header.audio_format) + " (expected PCM or flaot)");
    uint32_t num_samples = header.subchunk2_size /  (header.bits_per_sample / 8 );
    std::vector<float> buffer(num_samples);
    if (header.audio_format == 1) {
        std::vector<int16_t> temp_buffer(num_samples);
        wav_file.read(reinterpret_cast<char*> (temp_buffer.data()), header.subchunk2_size);
        for (size_t i = 0; i < num_samples; i++) {
            buffer[i] = temp_buffer[i] / 32768.0f;
        }
    } else if (header.audio_format == 3) {
        // just read directly
        wav_file.read(reinterpret_cast<char*>(buffer.data()), header.subchunk2_size);
    }
    if (header.num_channels == 1) {
        stereo_buffer.resize(num_samples * 2);
        for (size_t i = 0; i < num_samples; i++) {
            stereo_buffer[i * 2] = buffer[i];
            stereo_buffer[i * 2 + 1] = buffer[i];
        }
    }
    else {
        stereo_buffer = buffer;
    }
    header.num_channels = 2;
    header.bits_per_sample = 32;
    header.audio_format = 3;
    header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.subchunk2_size = stereo_buffer.size() * sizeof(float);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.chunk_size = 36 + header.subchunk2_size;
    return header;
}

void write_wav(WAVHeader& header, const std::string& filename, std::vector<float>& buffer) {
    std::ofstream output(filename, std::ios::binary);
    if (!output) throw std::runtime_error("could not open file for saving");
    output.write(reinterpret_cast<char*>(&header), sizeof(header));
    output.write(reinterpret_cast<char*> (buffer.data()), buffer.size() * sizeof(float));
}

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