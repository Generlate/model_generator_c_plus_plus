#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <torch/torch.h>

class NeuralNetwork : public torch::nn::Module
{
public:
    // Constructor
    NeuralNetwork();

    // Forward pass
    torch::Tensor forward(const torch::Tensor &x);

private:
    // Layers
    torch::nn::Linear hidden1{nullptr};
    torch::nn::Linear hidden2{nullptr};
    torch::nn::Linear hidden3{nullptr};
    torch::nn::Linear output{nullptr};
};

#endif // NEURAL_NETWORK_H
