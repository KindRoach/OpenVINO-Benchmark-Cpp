#pragma once

#include "opencv2/opencv.hpp"
#include "openvino/openvino.hpp"
#include "spdlog/spdlog.h"

int infer_one_stream(std::vector<ov::Tensor> &outputs, ov::CompiledModel model, int sec, bool inference_only, bool ov_preprocess);

cv::Mat preprocess_frame(cv::Mat frame, ov::Shape input_shape);

ov::Tensor infer_one_frame(ov::CompiledModel model, cv::Mat frame, ov::element::Type input_type, ov::Shape input_shape, bool ov_preprocess);
