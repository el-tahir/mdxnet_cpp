#include "ModelHandler.h"
#include <iostream>

void ModelHandler::load_model(const std::string& model_path) {
    if (model != nullptr) {
        delete model;
    }

    try {
        model = new Ort::Session(env, model_path.c_str(), config);

        std::cout << "model loaded successfully: " << model_path << std::endl;

    } catch(const Ort::Exception& e) {
        std::cerr << "failed to load model: " << e.what() << std::endl;
    }


}

std::vector<float> ModelHandler::run_inference(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape) {

    if (model == nullptr) {
        throw std::runtime_error("model not loaded! call load_model() first");
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info,
        const_cast<float*>(input_data.data()), input_data.size(), input_shape.data(), input_shape.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;

    Ort::AllocatedStringPtr input_name_ptr = model->GetInputNameAllocated(0, allocator);

    Ort::AllocatedStringPtr output_name_ptr = model->GetOutputNameAllocated(0, allocator);

    const char* input_names[] = {input_name_ptr.get()};
    
    const char* output_names[] = { output_name_ptr.get()};


    auto output_tensors = model->Run(
        Ort::RunOptions{nullptr}, 
        input_names, &input_tensor, 1, 
        output_names, 1
    );

    float* float_arr = output_tensors[0].GetTensorMutableData<float>();

    size_t output_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

    return std::vector<float>(float_arr, float_arr + output_size);

}