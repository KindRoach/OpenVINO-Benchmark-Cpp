#pragma once

#include <openvino/openvino.hpp>

std::vector<ov::Tensor> one_decode_multi_infer(ov::CompiledModel model, int n_stream, int sec, bool inference_only, bool ov_preprocess);