#include "common.hpp"
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <spdlog/spdlog.h>

int infer_one_stream(std::vector<ov::Tensor> &outputs, ov::CompiledModel model, int sec, bool inference_only, bool ov_preprocess) {
    cv::VideoCapture cap("output/video.mp4");

    if (!cap.isOpened()) {
        spdlog::error("Error opening video stream or file");
        exit(1);
    }

    auto input_shape = model.input().get_shape();
    auto input_type = model.input().get_element_type();
    auto infer_request = model.create_infer_request();

    cv::Mat frame;
    int frame_count = 0;
    auto finish = std::chrono::system_clock::now() + std::chrono::seconds(sec);
    while (std::chrono::system_clock::now() < finish) {
        bool success = cap.read(frame);
        if (success) {
            const auto &model_output = infer_one_frame(model, frame, input_type, input_shape, ov_preprocess);
            outputs.push_back(model_output);
        } else {
            cap.set(cv::CAP_PROP_POS_FRAMES, 0);
            continue;
        }
        frame_count++;
    }

    cap.release();

    return frame_count;
}

cv::Mat preprocess_frame(cv::Mat frame, ov::Shape input_shape) {
    int width = static_cast<int>(input_shape[2]);
    int height = static_cast<int>(input_shape[3]);

    resize(frame, frame, cv::Size(width, height));
    cvtColor(frame, frame, cv::COLOR_BGR2RGB);
    frame.convertTo(frame, CV_32F, 1.0 / 255, 0);

    cv::Mat mean = cv::Mat(frame.size(), CV_32FC3, cv::Scalar(0.485, 0.456, 0.406));
    cv::Mat std = cv::Mat(frame.size(), CV_32FC3, cv::Scalar(0.229, 0.224, 0.225));
    subtract(frame, mean, frame);
    divide(frame, std, frame);

    return frame;
}

thread_local bool request_created = false;
thread_local ov::InferRequest infer_request;
ov::Tensor infer_one_frame(ov::CompiledModel model, cv::Mat frame, ov::element::Type input_type, ov::Shape input_shape, bool ov_preprocess) {
    if (!ov_preprocess) frame = preprocess_frame(frame, input_shape);
    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, frame.data);
    if (!request_created) infer_request = model.create_infer_request();
    infer_request.set_input_tensor(input_tensor);
    infer_request.start_async();
    infer_request.wait();
    return infer_request.get_output_tensor();
}
