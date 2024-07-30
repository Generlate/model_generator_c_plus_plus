#pragma once

#include <string>
#include <vector>
#include <torch/torch.h> // Ensure you have PyTorch C++ API installed and properly linked

// Forward declarations of classes for loading data
class TrainingDataLoader;
class TestingDataLoader;

// Function prototypes
torch::Tensor format_training_data(const TrainingDataLoader &training_loader);
torch::Tensor format_testing_data(const TestingDataLoader &testing_loader);
