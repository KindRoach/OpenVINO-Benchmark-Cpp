#pragma once

#include "ConcurrentQueue.hpp"
#include "openvino/openvino.hpp"
#include <future>
#include <iostream>
#include <optional>

typedef ConcurrentQueue<std::optional<std::shared_future<ov::Tensor>>> concurrentQueue;
std::vector<ov::Tensor> one_decode_multi_infer(ov::CompiledModel model, int n_stream, int sec, bool inference_only, bool ov_preprocess);