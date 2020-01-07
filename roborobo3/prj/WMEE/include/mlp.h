#pragma once

#include <torch/torch.h>
#include <utility>


class TorchMLPImpl : public torch::nn::Module {
    public:
        TorchMLPImpl(int64_t input_dim, int64_t output_dim, int64_t h_dim);
        torch::Tensor forward(torch::Tensor x);
    private:

        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
        //torch::nn::Linear fc3;
        //torch::nn::Linear fc4;
        //torch::nn::Linear fc5;
};

TORCH_MODULE(TorchMLP);
