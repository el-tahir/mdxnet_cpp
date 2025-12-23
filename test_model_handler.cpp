#include <iostream>
#include <vector>
#include <numeric>
#include "ModelHandler.h"


int main() {

    std::string model_path = "UVR_MDXNET_KARA_2.onnx";
    ModelHandler handler;

    try {

        handler.load_model(model_path);

        // MDX-NET expects [1, 2, Dim, Dim]
        // dummy data
        std::vector<int64_t> input_shape = {1, 4, 2048, 256};

        // {batch = 1, channels = 4, times = 2048, freq = 256}

        size_t total_elements = 1 * 4 * 2048 * 256;

        std::vector<float> input_data(total_elements);

        std::iota(input_data.begin(), input_data.end(), 0.0f);

        std::cout << "running inference on dummy data..." << std::endl;

        std::vector<float> output = handler.run_inference(input_data, input_shape);


        std::cout << "success!" << std::endl;
        std::cout << "output vector size: " << output.size() << std::endl;
        
        std::cout << "first 5 values: ";
        for (int i = 0; i < 5 && i < output.size(); i++) std::cout << output[i] << " ";
        std::cout << std::endl;


    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}