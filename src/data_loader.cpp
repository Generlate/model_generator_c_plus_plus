/** Module to get data from a directory and convert it to a python dataset. */

#include "data_loader.h"
#include <fstream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <torch/torch.h>
#include <torch/script.h>

// Helper function to process file content
std::vector<float> DataLoader::organize_content(const std::string &content)
{
    std::vector<float> flattened_list;
    std::istringstream stream(content);
    std::string line;
    while (std::getline(stream, line))
    {
        if (!line.empty())
        {
            std::istringstream line_stream(line);
            float value;
            while (line_stream >> value)
            {
                flattened_list.push_back(round(value));
            }
        }
    }
    return flattened_list;
}

// Load contents from files
std::vector<std::vector<float>> DataLoader::load_file_contents(const std::vector<std::string> &file_paths)
{
    std::vector<std::vector<float>> file_contents;
    for (const auto &file_path : file_paths)
    {
        std::ifstream file(file_path);
        if (!file.is_open())
        {
            throw std::runtime_error("Unable to open file: " + file_path);
        }
        std::string content;
        std::string line;
        int line_count = 0;
        while (std::getline(file, line) && line_count < 10)
        {
            if (line_count >= 2)
            {
                content += line + "\n";
            }
            ++line_count;
        }
        file_contents.push_back(organize_content(content));
    }
    return file_contents;
}

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

// Returns the dataset's length.
size_t TrainingDataLoader::size() const
{
    return file_contents.size();
}

// Return as a tensor.
torch::Tensor TrainingDataLoader::operator[](size_t index) const
{
    if (index >= file_contents.size())
    {
        throw std::out_of_range("Index out of range");
    }
    return torch::tensor(file_contents[index]);
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

// Organize the dataset.
size_t TestingDataLoader::size() const
{
    return file_paths.size();
}

// Allows a testing dataset to be created from a directory of files.
// To fill.
// Return the dataset's length.
// Return file contents as a tensor.
// Loads file contents.
// Organize the dataset.
torch::Tensor TestingDataLoader::operator[](size_t index) const
{
    if (index >= file_paths.size())
    {
        throw std::out_of_range("Index out of range");
    }
    std::ifstream file(file_paths[index]);
    if (!file.is_open())
    {
        throw std::runtime_error("Unable to open file: " + file_paths[index]);
    }
    std::string content;
    std::string line;
    int line_count = 0;
    while (std::getline(file, line) && line_count < 10)
    {
        if (line_count >= 2)
        {
            content += line + "\n";
        }
        ++line_count;
    }
    auto flattened_list = organize_content(content);
    auto tensor = torch::tensor(flattened_list, torch::kFloat32).view({-1, 1});
    return tensor;
}
