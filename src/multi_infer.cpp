#include "multi_infer.hpp"
#include "argparse/argparse.hpp"
#include "common.hpp"
#include "openvino/openvino.hpp"
#include "spdlog/spdlog.h"
#include <chrono>
#include <future>
#include <iostream>

std::vector<std::vector<ov::Tensor>> multi_infer(ov::CompiledModel model, int n_stream, int sec, bool inference_only, bool ov_preprocess) {
    using namespace std;

    spdlog::info("async inference with {} threads in {} seconds...", n_stream, sec);
    vector<future<int>> futures;
    vector<vector<ov::Tensor>> outputs(n_stream);

    auto start = chrono::system_clock::now();

    for (int i = 0; i < n_stream; ++i) {
        futures.push_back(async(infer_one_stream, ref(outputs[i]), model, sec, inference_only, ov_preprocess));
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

    return outputs;
}
