#include <iostream>

#include <argparse/argparse.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

#include "multi_infer.hpp"
#include "one_decode_multi_infer.hpp"
#include "sync_infer.hpp"

std::unique_ptr<argparse::ArgumentParser> parseArg(int argc, char *const *argv) {
    using namespace std;

    auto program = std::make_unique<argparse::ArgumentParser>("run_infer");
    program->add_argument("-t", "--time")
            .help("time in seconds for benchmark")
            .default_value(60)
            .scan<'i', int>();
    program->add_argument("-rm", "--run_mode")
            .help("run mode: sync, async or multi")
            .default_value(string{"sync"})
            .choices("sync", "multi", "one_decode_multi_infer");
    program->add_argument("-n", "--n_stream")
            .help("number of infer_one_stream stream")
            .default_value(24)
            .scan<'i', int>();
    program->add_argument("-m", "--model")
            .help("model used for inference")
            .default_value(string{"resnet_50"});
    program->add_argument("-mt", "--model_type")
            .help("model type: fp32, fp16 or int8")
            .default_value(string{"int8"})
            .choices("fp32", "fp16", "int8");
    program->add_argument("-d", "--device")
            .help("device used for inference")
            .default_value(string{"CPU"});
    program->add_argument("-io", "--inference_only")
            .help("inference only, no video decode, use random input.")
            .default_value(false)
            .implicit_value(true);
    program->add_argument("-op", "--openvino_preprocess")
            .help("preprocess image by openvino ppp.")
            .default_value(false)
            .implicit_value(true);
    program->parse_args(argc, argv);
    return program;
}

ov::Core config_core(const std::string &run_mode) {
    ov::Core core;
    if (run_mode == "async" or run_mode == "multi" or run_mode == "one_decode_multi_infer") {
        core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        core.set_property("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    }
    return core;
}

std::shared_ptr<ov::Model> config_model(const ov::Core &core, const std::string &model_path, bool ov_preprocess) {
    auto model = core.read_model(model_path);

    if (ov_preprocess) {
        ov::preprocess::PrePostProcessor ppp(model);
        ov::preprocess::InputInfo &input = ppp.input();
        input.tensor()
                .set_element_type(ov::element::u8)
                .set_spatial_dynamic_shape()
                .set_layout("NHWC")
                .set_color_format(ov::preprocess::ColorFormat::BGR);
        input.model().set_layout("NCHW");
        input.preprocess()
                .convert_element_type(ov::element::f32)
                .convert_color(ov::preprocess::ColorFormat::RGB)
                .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)
                .mean({0.485 * 255, 0.456 * 255, 0.406 * 255})
                .scale({0.229 * 255, 0.224 * 255, 0.225 * 255});
        std::stringstream buffer;
        buffer << "Dump preprocessor: " << ppp << std::endl;
        spdlog::info(buffer.str());
        model = ppp.build();
    }
    return model;
}

int main(int argc, char *argv[]) {
    using namespace std;

    auto program = parseArg(argc, argv);

    // Reading command line args
    int sec = program->get<int>("time");
    string run_mode = program->get<string>("run_mode");
    int n_stream = program->get<int>("n_stream");
    string device = program->get<string>("device");
    bool inference_only = program->get<bool>("inference_only");
    bool ov_preprocess = program->get<bool>("openvino_preprocess");
    string model_name = program->get<string>("model");
    string model_type = program->get<string>("model_type");
    string model_path = fmt::format("output/model/{}/{}/model.xml", model_name, model_type);

    ov::Core core = config_core(run_mode);
    auto model = config_model(core, model_path, ov_preprocess);
    auto compared_model = core.compile_model(model, device);

    if (run_mode == "sync") {
        sync_infer(compared_model, sec, inference_only, ov_preprocess);
    } else if (run_mode == "multi") {
        multi_infer(compared_model, n_stream, sec, inference_only, ov_preprocess);
    } else if (run_mode == "one_decode_multi_infer") {
        one_decode_multi_infer(compared_model, n_stream, sec, inference_only, ov_preprocess);
    }
}
