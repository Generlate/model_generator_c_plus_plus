/** Runs a formatted dataset through a neural network, formats the output to be viewed as a 3d object and file name. */

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <filesystem>
// #include "neural_network.h"
// #include "data_formatter.h"
// #include "data_loader.h"

// namespace fs = std::filesystem;

// int main()
// {
// Load datasets
// load_datasets();

// Create an instance of the neural network
//     NeuralNetwork model;

// Define the loss function and OPTIMIZER
//    auto CRITERION = torch::nn::MSELoss();
//   auto OPTIMIZER = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(0.01));

// Set number of epochs. I found three to give the lowest loss score.
//    const int NUMBER_OF_EPOCHS = 3;

// The training loop
//    for (int EPOCH = 0; EPOCH < NUMBER_OF_EPOCHS; ++EPOCH)
//   {
// Get the next testing data tensor.
// Generate box coordinates
//      auto output = model.forward(TRAINING_COMBINED_TENSOR);

// Compare the generated coordinates against the test coordinates
//        auto loss = CRITERION(output, TESTING_COMBINED_TENSOR);

// Print the loss every EPOCH
//        std::cout << "Epoch: " << EPOCH + 1 << ", Loss: " << loss.item<double>() << std::endl;

// Backward pass and optimize
//        OPTIMIZER.zero_grad();
//        loss.backward();
//       OPTIMIZER.step();
//    }

// List the generated coordinates
//    auto output = model.forward(TRAINING_COMBINED_TENSOR);
//    auto array = output.detach().cpu().flatten();

// Format to .off
//    std::string FORMATTED_ARRAY = "OFF\n8 6 0\n";
//    for (size_t i = 0; i < array.size(0); ++i)
//    {
//        FORMATTED_ARRAY += std::to_string(array[i].item<float>() * 22) + " ";
//        if ((i + 1) % 3 == 0)
//        {
//            FORMATTED_ARRAY += "\n";
//        }
//    }

//    std::string ADDITIONAL_STRING = R"(
// 4 0 1 2 3
// 4 1 5 6 2
// 4 5 4 7 6
// 4 4 0 3 7
// 4 3 2 6 7
// 4 4 5 1 0)";

//    FORMATTED_ARRAY += ADDITIONAL_STRING;

//    std::string file_path = "./generated_boxes/generated_box.off";

// Check if the file already exists
//    if (fs::exists(file_path))
//    {
// Find the next available file name by incrementing a counter
//       int FILE_COUNTER = 1;
//        std::string incremented_file_path;
//        do
//        {
//            incremented_file_path = file_path.substr(0, file_path.find_last_of('.')) + "_" + std::to_string(FILE_COUNTER) + ".off";
//            ++FILE_COUNTER;
//        } while (fs::exists(incremented_file_path));

//        file_path = incremented_file_path;
//    }

// Save the .off file
//    std::ofstream FILE(file_path);
//    if (FILE.is_open())
//    {
//        FILE << FORMATTED_ARRAY;
//        FILE.close();
//        std::cout << "File generated successfully. Saved as: " << file_path << std::endl;
//    }
//    else
//    {
//        std::cerr << "Unable to open file " << file_path << std::endl;
//    }

//   return 0;
//}

namespace fs = std::filesystem;

bool readOffFile(const std::string &filename, std::vector<torch::Tensor> &vertices, std::vector<torch::Tensor> &faces)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line); // Read the OFF header
    if (line != "OFF")
    {
        std::cerr << "Invalid OFF file: " << filename << std::endl;
        return false;
    }

    size_t numVertices, numFaces, numEdges;
    file >> numVertices >> numFaces >> numEdges;

    std::vector<std::vector<float>> vertexList(numVertices, std::vector<float>(3));
    std::vector<std::vector<int>> faceList(numFaces);

    for (size_t i = 0; i < numVertices; ++i)
    {
        file >> vertexList[i][0] >> vertexList[i][1] >> vertexList[i][2];
    }

    size_t maxFaceVertices = 0;
    for (size_t i = 0; i < numFaces; ++i)
    {
        size_t numFaceVertices;
        file >> numFaceVertices;
        faceList[i].resize(numFaceVertices);
        maxFaceVertices = std::max(maxFaceVertices, numFaceVertices);
        for (size_t j = 0; j < numFaceVertices; ++j)
        {
            file >> faceList[i][j];
        }
    }

    // Convert to torch::Tensor
    std::vector<float> vertexData;
    for (const auto &v : vertexList)
    {
        vertexData.insert(vertexData.end(), v.begin(), v.end());
    }
    torch::Tensor vertexTensor = torch::from_blob(vertexData.data(), {static_cast<long>(numVertices), 3}, torch::kFloat32).clone();
    vertices.push_back(vertexTensor);

    std::vector<int64_t> faceData;
    for (const auto &f : faceList)
    {
        faceData.push_back(f.size());
        faceData.insert(faceData.end(), f.begin(), f.end());
    }

    // Ensure face tensor has the right dimensions
    torch::Tensor faceTensor = torch::from_blob(faceData.data(), {static_cast<long>(numFaces), static_cast<long>(maxFaceVertices)}, torch::kInt64).clone();
    faces.push_back(faceTensor);

    return true;
}

// Function to load all .off files in a directory
void loadDatasetFromDirectory(const std::string &directory, std::vector<torch::Tensor> &vertices, std::vector<torch::Tensor> &faces)
{
    for (const auto &entry : fs::directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".off")
        {
            std::string filePath = entry.path().string();
            if (!readOffFile(filePath, vertices, faces))
            {
                std::cerr << "Failed to load file: " << filePath << std::endl;
            }
        }
    }
}

// Print tensor utility function
void printTensor(const torch::Tensor &tensor, const std::string &name)
{
    if (tensor.scalar_type() == torch::kFloat32)
    {
        auto accessor = tensor.accessor<float, 2>(); // Adjust for your tensor's type and dimensions
        for (int64_t i = 0; i < tensor.size(0); ++i)
        {
            for (int64_t j = 0; j < tensor.size(1); ++j)
            {
                std::cout << accessor[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
    else if (tensor.scalar_type() == torch::kInt64)
    {
        auto accessor = tensor.accessor<int64_t, 2>(); // Adjust for your tensor's type and dimensions
        for (int64_t i = 0; i < tensor.size(0); ++i)
        {
            for (int64_t j = 0; j < tensor.size(1); ++j)
            {
                std::cout << accessor[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }
}

// Function to print the entire dataset
void printDataset(const std::vector<torch::Tensor> &vertices, const std::vector<torch::Tensor> &faces, const std::string &datasetName)
{
    std::cout << datasetName << " Dataset:" << std::endl;
    for (size_t i = 0; i < vertices.size(); ++i)
    {
        std::cout << "Vertex Tensor " << i << ":" << std::endl;
        printTensor(vertices[i], "Vertex");
    }

    for (size_t i = 0; i < faces.size(); ++i)
    {
        std::cout << "Face Tensor " << i << ":" << std::endl;
        printTensor(faces[i], "Face");
    }
}

int main()
{
    std::string training_directory = "../assets/datasets/austens_boxes/training";
    std::string testing_directory = "../assets/datasets/austens_boxes/testing";

    std::vector<torch::Tensor> training_vertices;
    std::vector<torch::Tensor> training_faces;
    std::vector<torch::Tensor> testing_vertices;
    std::vector<torch::Tensor> testing_faces;

    loadDatasetFromDirectory(training_directory, training_vertices, training_faces);
    loadDatasetFromDirectory(testing_directory, testing_vertices, testing_faces);

    // Print the training dataset
    printDataset(training_vertices, training_faces, "Training");

    // Print the testing dataset
    printDataset(testing_vertices, testing_faces, "Testing");

    return 0;
}