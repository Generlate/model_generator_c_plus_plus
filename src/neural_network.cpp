/** Takes in tensors and returns new tensors. */

#include "neural_network.h"

// Calls Pytorch functions to process a tensor through a neural network.
// Constructor definition
NeuralNetwork::NeuralNetwork()
    : hidden1(register_module("hidden1", torch::nn::Linear(60, 80))),
      hidden2(register_module("hidden2", torch::nn::Linear(80, 80))),
      hidden3(register_module("hidden3", torch::nn::Linear(80, 80))),
      output(register_module("output", torch::nn::Linear(80, 1)))
{
    // Constructor does not need to do anything else
}

// Initializes a Neural network with three hidden layers.

// Applies a relu activation function.
// Forward pass definition
torch::Tensor NeuralNetwork::forward(const torch::Tensor &x)
{
    x = torch::relu(hidden1(x));
    x = torch::relu(hidden2(x));
    x = torch::relu(hidden3(x));
    x = output(x);
    return x;
}
