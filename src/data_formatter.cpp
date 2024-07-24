#include "data_processor.h"
#include "data_loader.h" // Include the header where TrainingDataLoader and TestingDataLoader are defined
#include <torch/torch.h>
#include <iostream>
#include <stdexcept>

// Helper function to convert data to tensor
torch::Tensor convert_to_tensor(const std::vector<std::vector<float>> &data)
{
    std::vector<torch::Tensor> tensors;
    for (const auto &vec : data)
    {
        tensors.push_back(torch::tensor(vec));
    }
    return torch::stack(tensors);
}

// Format training data
torch::Tensor format_training_data(const TrainingDataLoader &training_loader)
{
    std::vector<std::vector<float>> number_lists(24);
    for (const auto &file_content : training_loader.file_contents)
    {
        for (size_t i = 0; i < file_content.size() && i < number_lists.size(); ++i)
        {
            number_lists[i].push_back(file_content[i]);
        }
    }

    std::vector<torch::Tensor> tensors;
    for (auto &vec : number_lists)
    {
        tensors.push_back(torch::tensor(vec).to(torch::kFloat32));
    }

    return torch::stack(tensors, 0);
}

// Format testing data
torch::Tensor format_testing_data(const TestingDataLoader &testing_loader)
{
    std::vector<torch::Tensor> tensors;
    for (size_t i = 0; i < testing_loader.size(); ++i)
    {
        tensors.push_back(testing_loader[i]);
    }
    return torch::stack(tensors, 0);
}
