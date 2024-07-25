/** Module to get data from a directory and convert it to a python dataset. */

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <torch/torch.h>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <filesystem>
#include <torch/script.h>

// TrainingDataLoader for training datasets
// Allows a training dataset to be created from a directory of files.

// class TrainingDataLoader : public torch::data::Dataset<TrainingDataLoader>
//{
// public:
//     TrainingDataLoader(const std::string &training_data_directory, int training_number_of_files);
//  Initializes a Dataset by loading files from a directory.
//  The sorted file paths limited to the number specified.

//    size_t size() const override;
//    torch::Tensor operator[](size_t index) const override;

//    int len();
// Returns the dataset's length.
//    void getitem();
// Return as a tensor.
//    void load_file_contents();
//    std::vector<float> load_file_contents(const std::string &file_path);
// Loads file contents.
//  void organize_content();
//   std::vector<float> organize_content(const std::vector<float> &content);
// Organize the dataset.
//};

// TestingDataLoader for testing datasets
// class TestingDataLoader : public torch::data::Dataset<TestingDataLoader>
// {
// Allows a testing dataset to be created from a directory of files.

// public:
// std::vector<std::string> file_paths;

// TestingDataLoader(const std::string &testing_data_directory, int testing_number_of_files);
// To fill.

// size_t size() const override;
// torch::Tensor operator[](size_t index) const override;

// void len();
// Return the dataset's length.
// void getitem();
// Return file contents as a tensor.
// void load_file_contents();
// Loads file contents.
// void organize_content();
// Organize the dataset.
// };

class TrainingDataLoader
{
public:
    TrainingDataLoader(const std::string &Dataset);
    void printName() const;
    std::string getName() const; // Optional getter

private:
    std::string dataset_;
};

class TestingDataLoader
{
public:
    TestingDataLoader(const std::string &Dataset);
    void printName2() const;
    std::string getName2() const; // Optional getter

private:
    std::string dataset_;
};

#endif // DATA_LOADER_H
