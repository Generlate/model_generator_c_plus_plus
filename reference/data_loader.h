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

struct MeshData
{
    std::vector<float> vertices; // Flattened vertex list (x, y, z) per vertex
    std::vector<int> faces;      // Flattened face list (vertex indices) per face
};

class DataLoader
{
public:
    DataLoader(const std::string &directory);
    std::vector<MeshData> loadDataset();

private:
    std::string directory;
    MeshData parseOFFFile(const std::string &filePath);
};

torch::Tensor meshDataToTensor(const MeshData &meshData);

// class TestingDataLoader : public torch::data::Dataset<TestingDataLoader>
//{
// public:
//     TestingDataLoader(const std::string &dataset);

// Implement the get method
//    torch::data::Example<> get(size_t index) override;

// Implement the size method
//    torch::optional<size_t> size() const override;

// Print the dataset name
//    void printName() const;

// Optional getter for the dataset name
//    std::string getName() const;

// private:
//     std::string dataset_;
// };

#endif // DATA_LOADER_H
