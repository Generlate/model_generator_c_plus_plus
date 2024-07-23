#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include <torch/torch.h>

// DataLoader base class for shared functionality
class DataLoader
{
protected:
    std::vector<std::vector<float>> file_contents;

    std::vector<float> organize_content(const std::string &content);

public:
    virtual size_t size() const = 0;
    virtual torch::Tensor operator[](size_t index) const = 0;

    static std::vector<std::vector<float>> load_file_contents(const std::vector<std::string> &file_paths);
};

// TrainingDataLoader for training datasets
class TrainingDataLoader : public DataLoader
{
public:
    TrainingDataLoader(const std::string &training_data_directory, int training_number_of_files);

    size_t size() const override;
    torch::Tensor operator[](size_t index) const override;
};

// TestingDataLoader for testing datasets
class TestingDataLoader : public DataLoader
{
private:
    std::vector<std::string> file_paths;

public:
    TestingDataLoader(const std::string &testing_data_directory, int testing_number_of_files);

    size_t size() const override;
    torch::Tensor operator[](size_t index) const override;
};

#endif // DATA_LOADER_H
