/** Takes in tensors and returns new tensors. */

#include "neural_network.h"

NeuralNetwork::NeuralNetwork(int inputSize)
{
    if (inputSize <= 0)
    {
        throw std::invalid_argument("Error: Invalid input size.");
    }
    hidden1 = register_module("hidden1", torch::nn::Linear(inputSize, 8));
    hidden2 = register_module("hidden2", torch::nn::Linear(8, 8));
    hidden3 = register_module("hidden3", torch::nn::Linear(8, 8));
    output = register_module("output", torch::nn::Linear(8, 1));
}

torch::Tensor NeuralNetwork::forward(torch::Tensor x)
{
    x = torch::relu(hidden1->forward(x));
    x = torch::relu(hidden2->forward(x));
    x = torch::relu(hidden3->forward(x));
    x = output->forward(x);
    return x;
}

void NeuralNetwork::save(torch::serialize::OutputArchive &archive) const
{
    torch::nn::Module::save(archive);
}

void NeuralNetwork::load(torch::serialize::InputArchive &archive)
{
    torch::nn::Module::load(archive);
}
