#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <torch/torch.h>
#include <fstream>
#include <iostream>

// TrainingDataLoader for training datasets
class TrainingDataLoader : public Dataset
{
public:
    TrainingDataLoader(const std::string &training_data_directory, int training_number_of_files);

    size_t size() const override;
    torch::Tensor operator[](size_t index) const override;
};

// TestingDataLoader for testing datasets
class TestingDataLoader : public Dataset
{

public:
    std::vector<std::string> file_paths;

    TestingDataLoader(const std::string &testing_data_directory, int testing_number_of_files);

    size_t size() const override;
    torch::Tensor operator[](size_t index) const override;
};

#endif // DATA_LOADER_H
