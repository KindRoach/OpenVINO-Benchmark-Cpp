#include "one_decode_multi_infer.hpp"
#include "ConcurrentQueue.hpp"
#include "argparse/argparse.hpp"
#include "common.hpp"
#include "openvino/openvino.hpp"
#include "spdlog/spdlog.h"
#include <chrono>
#include <future>
#include <iostream>

void decode_and_submit_infer(concurrentQueue &queue, ov::CompiledModel model, int sec, bool inference_only, bool ov_preprocess) {
    cv::VideoCapture cap("output/video.mp4");

    if (!cap.isOpened()) {
        spdlog::error("Error opening video stream or file");
        exit(1);
    }

    auto input_shape = model.input().get_shape();
    auto input_type = model.input().get_element_type();
    thread_local auto infer_request = model.create_infer_request();

    cv::Mat frame;
    int frame_count = 0;
    auto finish = std::chrono::system_clock::now() + std::chrono::seconds(sec);
    while (std::chrono::system_clock::now() < finish) {
        bool success = cap.read(frame);
        if (success) {
            queue.push(async(infer_one_frame, model, frame, input_type, input_shape, ov_preprocess));
        } else {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        frame_count++;
    }

    cap.release();
    queue.push(std::nullopt);
}


std::vector<ov::Tensor> gather_infer_result(concurrentQueue &queue) {
    std::vector<ov::Tensor> result;
    auto item = queue.pop();
    while (item.has_value()) {
        result.push_back(item.value().get());
        item = queue.pop();
    }
    return result;
}

std::vector<ov::Tensor> one_decode_multi_infer(ov::CompiledModel model, int n_stream, int sec, bool inference_only, bool ov_preprocess) {
    using namespace std;

    spdlog::info("async inference with {} threads in {} seconds...", n_stream, sec);
    auto start = chrono::system_clock::now();

    auto queue = ConcurrentQueue<optional<shared_future<ov::Tensor>>>(n_stream);
    thread t(decode_and_submit_infer, ref(queue), model, sec, inference_only, ov_preprocess);
    auto outputs = gather_infer_result(queue);
    t.join();

    auto end = chrono::system_clock::now();
    auto diff = chrono::duration_cast<chrono::seconds>(end - start).count();
    double fps = static_cast<double>(outputs.size()) / static_cast<double>(diff);
    cout << "fps: " << fps << endl;

    return outputs;
}
