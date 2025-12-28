#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include "DSPCore.h"
#include "kiss_fft.h"
#include "ModelHandler.h"
#include "WAVHeader.h"
#include "utils.cpp"


void run_seperation(const std::string& input_path, const std::string& output_path, const std::string& model_path) {
    // setup
    std::cout << "loading " << input_path << "..." << std::endl;
    std::vector<float> stereo_buffer;
    WAVHeader header = read_wav(input_path, stereo_buffer);
std::cout << "DEBUG: Stereo Buffer Size: " << stereo_buffer.size() << std::endl; // <--- Add this

    // split channels

    std::vector<float> left_audio, right_audio;
    for (size_t i = 0; i < stereo_buffer.size(); i += 2) {
        left_audio.push_back(stereo_buffer[i]);
        right_audio.push_back(stereo_buffer[i + 1]);
    }

    uint32_t n_fft = 4096; uint32_t hop_length = 1024;

    DSPCore dsp(n_fft, hop_length);
    ModelHandler model;
    model.load_model(model_path);

    // analysis

    std::vector<float> left_padded = dsp.pad_audio(left_audio);
    std::vector<float> right_padded = dsp.pad_audio(right_audio);


    std::vector<std::vector<kiss_fft_cpx>> all_left_frames, all_right_frames;


    for (size_t offset = 0; offset + n_fft <= left_padded.size(); offset += hop_length) {
        std::vector<float> left_frame(n_fft), right_frame(n_fft);

        for (uint32_t i = 0; i < n_fft; i++) {
            left_frame[i] = left_padded[offset + i];
            right_frame[i] = right_padded[offset + i];
        }

        all_left_frames.push_back(dsp.stft(left_frame));
        all_right_frames.push_back(dsp.stft(right_frame));

    }
    
    std::vector<std::vector<kiss_fft_cpx>> processed_left, processed_right;

    int batch_size = 256;
    int num_frames = all_left_frames.size();

    std::cout << "runnning inference on " << num_frames << " frames..." << std::endl;

    std::vector<int64_t> input_shape = {1, 4, 2048, 256};

    for (int i = 0; i < num_frames; i += batch_size) {

        std::vector<std::vector<kiss_fft_cpx>> left_batch(batch_size, std::vector<kiss_fft_cpx>(n_fft)), right_batch(batch_size, std::vector<kiss_fft_cpx>(n_fft));

        int frames_remaining = num_frames - i;
        int actual_batch = std::min(batch_size, frames_remaining);

        for (int j = 0; j < actual_batch; j++) {
            left_batch[j] = all_left_frames[i + j];
            right_batch[j] = all_right_frames[i + j];
        }

        std::vector<float> tensor = stft_to_tensor(left_batch, right_batch);

        std::vector<float> processed = model.run_inference(tensor, input_shape);

        auto output = tensor_to_stft(processed);

        for (int k = 0; k < actual_batch; k++) {
            processed_left.push_back(output.first[k]);
            processed_right.push_back(output.second[k]);
        }

    }

    std::vector<float> left_reconstructed(left_padded.size(), 0.0f);
    std::vector<float> right_reconstructed(right_padded.size(), 0.0f);

    uint32_t pad_length = n_fft / 2;
    for (size_t frame_idx = 0; frame_idx < processed_left.size(); frame_idx++) {
        std::vector<float> left_time = dsp.istft(processed_left[frame_idx]);
        std::vector<float> right_time = dsp.istft(processed_right[frame_idx]);

        size_t offset = frame_idx * hop_length;

        for (uint32_t n = 0; n < n_fft; n++) {
            left_reconstructed[offset + n] += left_time[n];
            right_reconstructed[offset + n] += right_time[n];
        }

    }

    std::vector<float> left_final(left_audio.size());
    std::vector<float> right_final(right_audio.size());

    for (size_t i = 0; i < left_audio.size(); i++) {
        left_final[i] = left_reconstructed[pad_length + i];
        right_final[i] = right_reconstructed[pad_length + i];
    }

    std::vector<float> stereo_output;
    stereo_output.reserve(left_final.size() * 2);

    for (size_t i = 0; i < left_final.size(); i++) {
        stereo_output.push_back(left_final[i]);
        stereo_output.push_back(right_final[i]);
    }

    // normalization for COLA (Constant Overlap-Add):
    // with window applied in both stft and istft (symmetric), we have w^2(n) at each position.
    // For 75% overlap (hop_length = n_fft/4) with periodic hann window:
    // sum of squared windows at each position = 1.5
    // So we divide by 1.5 to normalize

    for (float& sample : stereo_output) {
        sample /= 1.5f;
    }


    write_wav(header, output_path, stereo_output);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "usage: ./seperator <input.wav> <output.wav>" << std::endl;
        return 1;
    }

    std::string input_file = argv[1];
    std::string output_file = argv[2];
    std::string model_file = "UVR_MDXNET_KARA_2.onnx";

    try {
        run_seperation(input_file, output_file, model_file);
        std::cout << "done! saved to " << output_file << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}