#pragma once

#include "openvino/openvino.hpp"

std::vector<ov::Tensor> sync_infer(ov::CompiledModel model, int sec, bool inference_only, bool ov_preprocess);
