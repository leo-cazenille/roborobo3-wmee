#include "WMEE/include/mlp.h"
#include <utility>

TorchMLPImpl::TorchMLPImpl(int64_t input_dim, int64_t output_dim, int64_t h_dim) :
        fc1(input_dim, h_dim),
        fc2(h_dim, output_dim) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
}

torch::Tensor TorchMLPImpl::forward(torch::Tensor x) {
    auto h = torch::relu(fc1->forward(x));
    auto res = torch::sigmoid(fc2->forward(h));
    return res;
}
