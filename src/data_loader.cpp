/** Module to get data from a directory and convert it to a python dataset. */

#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <iostream>
#include <vector>

class Dataset : public torch::data::datasets::Dataset<Dataset>
{
public:
    // Constructor to initialize the dataset with file paths or other parameters
    Dataset(const std::string &file_path)
    {
        // Load data from file or other sources
        std::ifstream file(file_path);
        std::string line;
        while (std::getline(file, line))
        {
            data_.push_back(std::stof(line));
        }
    }

    // Override the get method to return a single data sample
    torch::data::Example<> get(size_t index) override
    {
        return {torch::tensor(data_[index]), torch::tensor(labels_[index])};
    }

    // Override the size method to return the size of the dataset
    torch::optional<size_t> size() const override
    {
        return data_.size();
    }

    std::vector<float> data_;
    std::vector<float> labels_;
};

// Allows a training dataset to be created from a directory of files.
// Initializes a Dataset by loading files from a directory.
// The sorted file paths limited to the number specified.
// TrainingDataLoader implementation
TrainingDataLoader::TrainingDataLoader(const std::string &training_data_directory, int training_number_of_files)
{
    std::vector<std::string> file_paths;
    for (const auto &entry : std::filesystem::directory_iterator(training_data_directory))
    {
        file_paths.push_back(entry.path().string());
    }
    std::sort(file_paths.begin(), file_paths.end());
    file_paths.resize(training_number_of_files);
    file_contents = load_file_contents(file_paths);
}

// Loads file contents.
// TestingDataLoader implementation
TestingDataLoader::TestingDataLoader(const std::string &testing_data_directory, int testing_number_of_files)
{
    std::vector<std::string> file_names;
    for (const auto &entry : std::filesystem::directory_iterator(testing_data_directory))
    {
        file_names.push_back(entry.path().string());
    }
    std::sort(file_names.begin(), file_names.end());
    file_names.resize(testing_number_of_files);
    file_paths = file_names;
}

// Allows a testing dataset to be created from a directory of files.
// To fill.
// Return the dataset's length.
// Return file contents as a tensor.
// Loads file contents.
// Organize the dataset.
