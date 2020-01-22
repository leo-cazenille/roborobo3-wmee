#include "WMEE/include/lstm.h"
#include <utility>
#include <memory>
#include <cmath>


MDNLSTMImpl::MDNLSTMImpl(int64_t sequence_dim, int64_t hidden_units, int64_t z_dim, int64_t actions_dim, int64_t nb_layers, int64_t nb_gaussians, int64_t hidden_dim, double temperature, double learning_rate) :
        _sequence_dim(sequence_dim),
        _hidden_units(hidden_units),
        _z_dim(z_dim),
        _nb_layers(nb_layers),
        _nb_gaussians(nb_gaussians),
        _temperature(temperature),
        //fc1(z_dim + 1, hidden_dim),
        fc1(z_dim + actions_dim, hidden_dim),
        //lstm(hidden_dim, hidden_units, nb_layers),
        lstm(nullptr),
        z_pi(hidden_units, nb_gaussians * z_dim),
        z_sigma(hidden_units, nb_gaussians * z_dim),
        z_mu(hidden_units, nb_gaussians * z_dim) {

    torch::nn::LSTMOptions lstm_options(hidden_dim, hidden_units);
    lstm_options.layers(nb_layers);
    lstm = torch::nn::LSTM(std::make_shared<torch::nn::LSTMImpl>(lstm_options));

    register_module("fc1", fc1);
    register_module("lstm", lstm);
    register_module("z_pi", z_pi);
    register_module("z_sigma", z_sigma);
    register_module("z_mu", z_mu);
    //optim.reset(new torch::optim::SGD(this->parameters(), 1e-3)); // XXX
    optim.reset(new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate))); // XXX

    reset_hidden();
}

//torch::Tensor AEImpl::encode(torch::Tensor x) {
////    auto h = torch::relu(fc1->forward(x));
////    return torch::relu(fc2->forward(h));
//    auto h = torch::tanh(fc1->forward(x));
//    return torch::tanh(fc2->forward(h));
//}

MDNLSTMOutput MDNLSTMImpl::forward(torch::Tensor x) {
    lstm->flatten_parameters();
    auto sequence = x.size(0);

    auto x2 = torch::relu(fc1->forward(x));
    //std::cout << "DEBUGforward: x2:" << x2.size(0) << "," << x2.size(1) << std::endl;
    //std::cout << "DEBUGforward: _hidden:" << _hidden.size(0) << "," << _hidden.size(1) << "," << _hidden.size(2) << std::endl;
    auto res_lstm = lstm->forward(x2.view({x.size(0), 1, -1}), _hidden);
    auto z = res_lstm.output;
    _hidden = res_lstm.state;

    auto pi = z_pi->forward(z);
    //std::cout << "DEBUGforward: pi:";
    //for(size_t i = 0; i < 3; ++i)
    //    std::cout << pi.size(i) << ",";
    //std::cout << std::endl;
    //std::cout << "DEBUGforward: sequence:" << sequence << " _nb_gaussians:" << _nb_gaussians << " _z_dim:" << _z_dim << std::endl;
    pi = pi.view({-1, sequence, _nb_gaussians, _z_dim});
    pi = torch::softmax(pi, 2);
    pi /= _temperature;

    auto sigma = torch::exp(z_sigma->forward(z)).view({-1, sequence, _nb_gaussians, _z_dim});
    sigma *= std::pow(_temperature, 0.5);

    auto mu = z_mu->forward(z).view({-1, sequence, _nb_gaussians, _z_dim});
    return {pi, sigma, mu};
}

void MDNLSTMImpl::reset_hidden() {
    _hidden = torch::zeros({2, _nb_layers, _sequence_dim, _hidden_units});
    //_hidden = torch::zeros({2, 1, 1, 6});
    //_cell = torch::zeros({_nb_layers, _sequence_dim, _hidden_units});
}

