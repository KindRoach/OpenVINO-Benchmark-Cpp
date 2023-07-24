#include <argparse/argparse.hpp>
#include <chrono>
#include <future>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <set>
#include <spdlog/spdlog.h>


using namespace cv;
using namespace std;

ov::Tensor infer_one_frame(ov::InferRequest infer_request, Mat frame) {
    ov::element::Type input_type = ov::element::u8;
    auto shape = frame.size();
    auto height = static_cast<unsigned long>(shape.height);
    auto width = static_cast<unsigned long>(shape.width);
    ov::Shape input_shape = {1, height, width, 3};
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, frame.data);
    infer_request.set_input_tensor(input_tensor);
    infer_request.start_async();
    infer_request.wait();
    return infer_request.get_output_tensor();
}

int infer(ov::CompiledModel model, int sec, vector<ov::Tensor> &outputs) {
    VideoCapture cap("output/video.mp4");

    if (!cap.isOpened()) {
        spdlog::error("Error opening video stream or file");
        exit(1);
    }

    Mat frame;
    int frame_count = 0;
    auto finish = chrono::system_clock::now() + chrono::seconds(sec);
    ov::InferRequest infer_request = model.create_infer_request();
    while (chrono::system_clock::now() < finish) {
        bool success = cap.read(frame);
        if (success) {
            outputs.push_back(infer_one_frame(infer_request, frame));
        } else {
            cap.set(CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        frame_count++;
    }

    cap.release();

    return frame_count;
}

void sync_infer(ov::CompiledModel model, int sec) {
    spdlog::info("sync inference in {} seconds...", sec);
    auto start = chrono::system_clock::now();
    vector<ov::Tensor> outputs;
    auto frames = infer(model, sec, outputs);
    auto end = chrono::system_clock::now();
    auto diff = chrono::duration_cast<chrono::seconds>(end - start).count();
    double fps = static_cast<double>(frames) / static_cast<double>(diff);
    cout << "fps: " << fps << endl;
}

void multi_infer(ov::CompiledModel model, int n_stream, int sec) {
    spdlog::info("async encoding with {} threads in {} seconds...", n_stream, sec);
    vector<future<int>> futures;
    vector<vector<ov::Tensor>> outputs(n_stream);

    auto start = chrono::system_clock::now();

    for (int i = 0; i < n_stream; ++i) {
        futures.push_back(async(infer, model, sec, std::ref(outputs[i])));
    }

    int total_frames = 0;
    for (auto &result: futures) {
        int frames = result.get();
        if (frames > 0) {
            total_frames += frames;
        }
    }

    auto end = chrono::system_clock::now();
    auto diff = chrono::duration_cast<chrono::seconds>(end - start).count();
    double fps = static_cast<double>(total_frames) / static_cast<double>(diff);

    cout << "fps: " << fps << endl;
}

argparse::ArgumentParser parseArg(int argc, char *const *argv) {
    argparse::ArgumentParser program("sync_decode");
    program.add_argument("-t", "--time")
            .help("time in seconds for benchmark")
            .default_value(60)
            .scan<'i', int>();
    program.add_argument("-rm", "--run_mode")
            .help("run mode: sync, async or multi")
            .default_value(string{"sync"})
            .action([](const string &value) {
                static const set<string> choices = {"sync", "async", "multi"};
                if (choices.contains(value)) {
                    return value;
                }
                spdlog::error("illegal input: {}", value);
                exit(1);
            });
    program.add_argument("-n", "--n_stream")
            .help("number of infer stream")
            .default_value(16)
            .scan<'i', int>();
    program.add_argument("-m", "--model")
            .help("model used for inference")
            .default_value(string{"resnet_50"});
    program.add_argument("-mt", "--model_type")
            .help("model type: fp32, fp16 or int8")
            .default_value(string{"int8"})
            .action([](const string &value) {
                static const set<string> choices = {"fp32", "fp16", "int8"};
                if (choices.contains(value)) {
                    return value;
                }
                spdlog::error("illegal input for model type: {}", value);
                exit(1);
            });
    program.add_argument("-d", "--device")
            .help("device used for inference")
            .default_value(string{"CPU"});
    program.add_argument("-io", "--inference_only")
            .help("inference only, no video decode, use random input.")
            .default_value(false)
            .implicit_value(true);
    program.parse_args(argc, argv);
    return program;
}

int main(int argc, char *argv[]) {
    argparse::ArgumentParser program = parseArg(argc, argv);

    int sec = program.get<int>("time");
    string run_mode = program.get<string>("run_mode");
    int n_stream = program.get<int>("n_stream");
    string device = program.get<string>("device");

    string model_name = program.get<string>("model");
    string model_type = program.get<string>("model_type");
    string model_path = fmt::format("output/model/{}/{}/model.xml", model_name, model_type);

    ov::Core core;
    if (run_mode == "async" or run_mode == "multi") {
        core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
        core.set_property("GPU", ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));
    }
    auto model = core.read_model(model_path);

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
            .mean({0.485, 0.456, 0.406})
            .scale({0.229, 0.224, 0.225});
    std::stringstream buffer;
    buffer << "Dump preprocessor: " << ppp << std::endl;
    spdlog::info(buffer.str());
    model = ppp.build();

    auto compared_model = core.compile_model(model, device);


    if (run_mode == "sync") {
        sync_infer(compared_model, sec);
    } else if (run_mode == "multi") {
        multi_infer(compared_model, n_stream, sec);
    }
}