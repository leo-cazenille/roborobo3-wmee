#pragma once

#include <torch/torch.h>
#include <utility>
#include <memory>

struct MDNLSTMOutput {
    torch::Tensor pi;
    torch::Tensor sigma;
    torch::Tensor mu;
};


class MDNLSTMImpl : public torch::nn::Module {
    public:
        MDNLSTMImpl(int64_t sequence_dim, int64_t h_dim, int64_t z_dim, int64_t actions_dim, int64_t nb_layers, int64_t nb_gaussians, int64_t hidden_dim, double temperature, double learning_rate);
        MDNLSTMOutput forward(torch::Tensor x);

        std::shared_ptr<torch::optim::Optimizer> optim = nullptr; // XXX

        torch::Tensor& get_hidden() { return _hidden; }
        void reset_hidden();

    private:
        //torch::Tensor encode(torch::Tensor x);
        //std::pair<torch::Tensor, torch::Tensor> encode(torch::Tensor x);
        //torch::Tensor reparameterize(torch::Tensor mu, torch::Tensor log_var);

        int64_t _sequence_dim;
        int64_t _hidden_units;
        int64_t _z_dim;
        int64_t _nb_layers;
        int64_t _nb_gaussians;
        double _temperature;

        torch::nn::Linear fc1;
        torch::nn::LSTM lstm;
        torch::nn::Linear z_pi;
        torch::nn::Linear z_sigma;
        torch::nn::Linear z_mu;

        torch::Tensor _hidden;
        //torch::Tensor _cell;
};

TORCH_MODULE(MDNLSTM);
