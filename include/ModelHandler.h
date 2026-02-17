#pragma once

#include <onnxruntime_cxx_api.h>
#include <memory>

class ModelHandler {

    private:
        Ort::Env env;
        std::unique_ptr<Ort::Session> model;
        Ort::SessionOptions config;
        Ort::AllocatorWithDefaultOptions allocator;

    public:
        ModelHandler(): env(ORT_LOGGING_LEVEL_WARNING, "ModelHandler"){
            config.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }

        void load_model(const std::string& model_path);

        std::vector<float> run_inference(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape);

};
