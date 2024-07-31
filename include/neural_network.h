#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <stdexcept>

struct NeuralNetwork : torch::nn::Module
{
    torch::nn::Linear hidden1{nullptr}, hidden2{nullptr}, hidden3{nullptr}, output{nullptr};

    NeuralNetwork(int inputSize);

    torch::Tensor forward(torch::Tensor x);

    void save(torch::serialize::OutputArchive &archive) const override;

    void load(torch::serialize::InputArchive &archive) override;
};
