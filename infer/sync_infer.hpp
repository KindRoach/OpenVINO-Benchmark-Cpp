#pragma once

#include <openvino/openvino.hpp>


void sync_infer(ov::CompiledModel model, int sec, bool inference_only, bool ov_preprocess);
