#include "sync_infer.hpp"
#include "argparse/argparse.hpp"
#include "common.hpp"
#include "openvino/openvino.hpp"
#include "spdlog/spdlog.h"
#include <chrono>
#include <iostream>

std::vector<ov::Tensor> sync_infer(ov::CompiledModel model, int sec, bool inference_only, bool ov_preprocess) {
    using namespace std;

    spdlog::info("sync inference in {} seconds...", sec);
    auto start = chrono::system_clock::now();

    vector<ov::Tensor> outputs;
    auto frames = infer_one_stream(outputs, model, sec, inference_only, ov_preprocess);

    auto end = chrono::system_clock::now();
    auto diff = chrono::duration_cast<chrono::seconds>(end - start).count();
    double fps = static_cast<double>(frames) / static_cast<double>(diff);
    cout << "fps: " << fps << endl;

    return outputs;
}