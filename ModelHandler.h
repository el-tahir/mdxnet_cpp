#include <onnxruntime_cxx_api.h>

class ModelHandler {

    private:
        Ort::Env env;
        Ort::Session* model;
        Ort::SessionOptions config;
        Ort::AllocatorWithDefaultOptions allocator;

    public:
        ModelHandler(): env(ORT_LOGGING_LEVEL_WARNING, "ModelHandler"), model(nullptr) {
            config.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }

        ~ModelHandler() {
            if (model != nullptr) {
                delete model;
            }
        }

        void load_model(const std::string& model_path);

        std::vector<float> run_inference(const std::vector<float>& input_data, const std::vector<int64_t>& input_shape);

};