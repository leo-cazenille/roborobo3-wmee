#pragma once

#include <torch/torch.h>
#include <utility>

struct AEOutput {
    torch::Tensor reconstruction;
    torch::Tensor z;
};


class AEImpl : public torch::nn::Module {
    public:
        AEImpl(int64_t image_size, int64_t h_dim, int64_t z_dim);
        torch::Tensor decode(torch::Tensor z);
        AEOutput forward(torch::Tensor x);
    private:
        torch::Tensor encode(torch::Tensor x);
        //std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x);
        //torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor log_var);

        torch::nn::Linear fc1;
        torch::nn::Linear fc2;
        torch::nn::Linear fc3;
        torch::nn::Linear fc4;
        //torch::nn::Linear fc5;
};

TORCH_MODULE(AE);
