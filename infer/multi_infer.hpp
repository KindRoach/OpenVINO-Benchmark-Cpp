#pragma once

#include <openvino/openvino.hpp>

std::vector<std::vector<ov::Tensor>> multi_infer(ov::CompiledModel model, int n_stream, int sec, bool inference_only, bool ov_preprocess);
