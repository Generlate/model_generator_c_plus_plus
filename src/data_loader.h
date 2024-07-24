/** Module to get data from a directory and convert it to a python dataset. */

#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <torch/torch.h>
#include <fstream>
#include <iostream>

class Dataset
{
public:
}

// TrainingDataLoader for training datasets
// Allows a training dataset to be created from a directory of files.

class TrainingDataLoader : public Dataset
{
public:
    TrainingDataLoader(const std::string &training_data_directory, int training_number_of_files);
    // Initializes a Dataset by loading files from a directory.
    // The sorted file paths limited to the number specified.

    size_t size() const override;
    torch::Tensor operator[](size_t index) const override;

    int len();
    // Returns the dataset's length.
    void getitem();
    // Return as a tensor.
    void load_file_contents();
    // Loads file contents.
    void organize_content();
    // Organize the dataset.
};

// TestingDataLoader for testing datasets
class TestingDataLoader : public Dataset
{
    // Allows a testing dataset to be created from a directory of files.

public:
    std::vector<std::string> file_paths;

    TestingDataLoader(const std::string &testing_data_directory, int testing_number_of_files);
    // To fill.

    size_t size() const override;
    torch::Tensor operator[](size_t index) const override;

    void len();
    // Return the dataset's length.
    void getitem();
    // Return file contents as a tensor.
    void load_file_contents();
    // Loads file contents.
    void organize_content();
    // Organize the dataset.
};

#endif // DATA_LOADER_H
