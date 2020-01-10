#include "WMEE/include/autoencoder.h"
#include <utility>

AEImpl::AEImpl(int64_t image_size, int64_t h_dim, int64_t z_dim, double learning_rate) :
        fc1(image_size, h_dim),
        fc2(h_dim, z_dim),
        fc3(z_dim, h_dim),
        fc4(h_dim, image_size) {
    register_module("fc1", fc1);
    register_module("fc2", fc2);
    register_module("fc3", fc3);
    register_module("fc4", fc4);
    //optim.reset(new torch::optim::SGD(this->parameters(), 1e-3)); // XXX
    //optim.reset(new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(1e-4))); // XXX
    optim.reset(new torch::optim::Adam(this->parameters(), torch::optim::AdamOptions(learning_rate))); // XXX
}

torch::Tensor AEImpl::encode(torch::Tensor x) {
//    auto h = torch::relu(fc1->forward(x));
//    return torch::relu(fc2->forward(h));
    auto h = torch::tanh(fc1->forward(x));
    return torch::tanh(fc2->forward(h));
}

torch::Tensor AEImpl::decode(torch::Tensor z) {
//    auto h = torch::relu(fc3->forward(z));
//    return torch::sigmoid(fc4->forward(h));
    auto h = torch::tanh(fc3->forward(z));
    return torch::tanh(fc4->forward(h));
}

AEOutput AEImpl::forward(torch::Tensor x) {
    auto z = encode(x);
    auto x_reconstructed = decode(z);
    return {x_reconstructed, z};
}
