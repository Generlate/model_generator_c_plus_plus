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

// Allows a training dataset to be created from a directory of files.
// Initializes a Dataset by loading files from a directory.
// The sorted file paths limited to the number specified.
// TrainingDataLoader implementation
// TrainingDataLoader::TrainingDataLoader(const std::string &training_data_directory, int training_number_of_files)
//{

DataLoader::DataLoader(const std::string &dir) : directory(dir) {}

MeshData DataLoader::parseOFFFile(const std::string &filePath)
{
    std::ifstream file(filePath);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open file: " + filePath);
    }

    MeshData meshData;
    std::string line;

    // Read header line
    std::getline(file, line);
    if (line != "OFF")
    {
        throw std::runtime_error("Invalid OFF file: " + filePath);
    }

    // Read number of vertices, faces, and edges
    int numVertices, numFaces, numEdges;
    file >> numVertices >> numFaces >> numEdges;

    // Read vertices
    meshData.vertices.resize(numVertices * 3);
    for (int i = 0; i < numVertices; ++i)
    {
        float x, y, z;
        file >> x >> y >> z;
        meshData.vertices[i * 3] = x;
        meshData.vertices[i * 3 + 1] = y;
        meshData.vertices[i * 3 + 2] = z;
    }

    // Read faces
    meshData.faces.reserve(numFaces * 3);
    for (int i = 0; i < numFaces; ++i)
    {
        int numFaceVertices;
        file >> numFaceVertices;
        for (int j = 0; j < numFaceVertices; ++j)
        {
            int vertexIndex;
            file >> vertexIndex;
            meshData.faces.push_back(vertexIndex);
        }
    }

    return meshData;
}

std::vector<MeshData> DataLoader::loadDataset()
{
    std::vector<MeshData> dataset;

    for (const auto &entry : std::filesystem::directory_iterator(directory))
    {
        if (entry.path().extension() == ".off")
        {
            MeshData meshData = parseOFFFile(entry.path().string());
            dataset.push_back(meshData);
        }
    }

    return dataset;
}

torch::Tensor meshDataToTensor(const MeshData &meshData)
{
    // Convert mesh data to tensor with correct options
    auto vertexOptions = torch::TensorOptions().dtype(torch::kFloat32);
    torch::Tensor vertices = torch::from_blob(
                                 meshData.vertices.data(),
                                 {static_cast<long>(meshData.vertices.size() / 3), 3},
                                 vertexOptions)
                                 .clone(); // Use clone to ensure tensor owns its data

    auto faceOptions = torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor faces = torch::from_blob(
                              meshData.faces.data(),
                              {static_cast<long>(meshData.faces.size())},
                              faceOptions)
                              .clone(); // Use clone to ensure tensor owns its data

    // For demonstration, returning vertices tensor. Modify as needed.
    return vertices;
}

// TestingDataLoader::TestingDataLoader(const std::string &Dataset) : dataset_(Dataset) {}

// Implement the get method
// torch::data::Example<> TestingDataLoader::get(size_t index) {
// Create a tensor for the data (example)
//   torch::Tensor data = torch::tensor({1.0, 2.0, 3.0});
// Create a tensor for the target (example)
//    torch::Tensor target = torch::tensor({1.0});

// Return the data and target as an Example
//    return {data, target};
//}

// Implement the size method
// torch::optional<size_t> TestingDataLoader::size() const {
// Return the size of the dataset (example)
//    return 100; // You should replace this with the actual size of your dataset
//}

// Print the dataset name
// void TestingDataLoader::printName() const {
//    std::cout << "Dataset: " << dataset_ << std::endl;
//}

// Optional getter for the dataset name
// std::string TestingDataLoader::getName() const {
//    return dataset_;
//}