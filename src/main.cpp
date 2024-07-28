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

// Extract numeric part from filename for sorting
int extractNumberFromFilename(const std::string &filename)
{
    std::regex regex("\\d+");
    std::smatch match;
    if (std::regex_search(filename, match, regex) && !match.empty())
    {
        return std::stoi(match.str());
    }
    return 0;
}

// Function to read an OFF file
bool readOffFile(const std::string &filename, std::vector<std::pair<std::string, torch::Tensor>> &vertices)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "Error opening file: " << filename << std::endl;
        return false;
    }

    std::string line;
    std::getline(file, line);
    if (line != "OFF")
    {
        std::cerr << "Invalid OFF file: " << filename << std::endl;
        return false;
    }

    size_t numVertices, numFaces, numEdges;
    file >> numVertices >> numFaces >> numEdges;

    if (!file.good())
    {
        std::cerr << "Error reading metadata from file: " << filename << std::endl;
        return false;
    }

    std::vector<float> vertexData(numVertices * 3);
    for (size_t i = 0; i < numVertices; ++i)
    {
        file >> vertexData[i * 3] >> vertexData[i * 3 + 1] >> vertexData[i * 3 + 2];
    }

    if (!file.good())
    {
        std::cerr << "Error reading vertices from file: " << filename << std::endl;
        return false;
    }

    // Convert to torch::Tensor
    torch::Tensor vertexTensor = torch::from_blob(vertexData.data(), {static_cast<long>(numVertices), 3}, torch::kFloat32).clone();
    vertices.emplace_back(filename, std::move(vertexTensor));

    return true;
}

// Function to load all .off files in a directory
void loadDatasetFromDirectory(const std::string &directory, std::vector<std::pair<std::string, torch::Tensor>> &vertices)
{
    std::vector<std::string> filePaths;

    for (const auto &entry : fs::directory_iterator(directory))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".off")
        {
            filePaths.push_back(entry.path().string());
        }
    }

    std::sort(filePaths.begin(), filePaths.end(), [](const std::string &a, const std::string &b)
              { return extractNumberFromFilename(a) < extractNumberFromFilename(b); });

    for (const auto &filePath : filePaths)
    {
        if (!readOffFile(filePath, vertices))
        {
            std::cerr << "Failed to load file: " << filePath << std::endl;
        }
    }
}

// Create a tensor of tensors (flattened data)
torch::Tensor combineTensors(const std::vector<std::pair<std::string, torch::Tensor>> &vertices)
{
    std::vector<torch::Tensor> flattenedTensors;

    for (const auto &entry : vertices)
    {
        torch::Tensor flatTensor = entry.second.view(-1); // Flatten the tensor
        flattenedTensors.push_back(flatTensor);
    }

    // Concatenate all flattened tensors into one tensor
    torch::Tensor result = torch::cat(flattenedTensors, 0).view({static_cast<long>(flattenedTensors.size()), -1});
    return result;
}

// Print tensor utility function
/* void printTensor(const torch::Tensor &tensor)
{
    if (tensor.scalar_type() == torch::kFloat32)
    {
        auto accessor = tensor.accessor<float, 2>(); // Adjust for the tensor's dimensions
        for (int64_t i = 0; i < tensor.size(0); ++i)
        {
            std::cout << "[";
            for (int64_t j = 0; j < tensor.size(1); ++j)
            {
                std::cout << accessor[i][j];
                if (j < tensor.size(1) - 1)
                    std::cout << ", "; // Add a comma between values
            }
            std::cout << "]";
            if (i < tensor.size(0) - 1)
                std::cout << ", "; // Add a comma between rows
            std::cout << std::endl;
        }
    }
} */

// Function to print the entire dataset
/* void printDataset(const std::vector<std::pair<std::string, torch::Tensor>> &vertices, const std::string &datasetName)
{
    std::cout << "Dataset: " << datasetName << std::endl;
    for (const auto &entry : vertices)
    {
        std::cout << "File: " << entry.first << std::endl;
        printTensor(entry.second);
        std::cout << std::endl;
    }
} */

// Function to create the training number lists
std::vector<torch::Tensor> createTrainingNumberLists(const torch::Tensor &trainingTensors)
{
    std::vector<torch::Tensor> training_number_lists;

    for (int index = 0; index < 24; ++index)
    {
        std::vector<float> values;

        for (int i = 0; i < trainingTensors.size(0); ++i)
        {
            if (index < trainingTensors.size(1))
            {
                values.push_back(trainingTensors[i][index].item<float>());
            }
        }

        training_number_lists.push_back(torch::tensor(values));
    }

    return training_number_lists;
}

// Define the neural network structure
struct NeuralNetwork : torch::nn::Module
{
    torch::nn::Linear hidden1{nullptr}, hidden2{nullptr}, hidden3{nullptr}, output{nullptr};

    NeuralNetwork(int64_t input_size)
    {
        hidden1 = register_module("hidden1", torch::nn::Linear(input_size, 80));
        hidden2 = register_module("hidden2", torch::nn::Linear(80, 80));
        hidden3 = register_module("hidden3", torch::nn::Linear(80, 80));
        output = register_module("output", torch::nn::Linear(80, 1));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(hidden1->forward(x)); // First hidden layer with ReLU
        x = torch::relu(hidden2->forward(x)); // Second hidden layer with ReLU
        x = torch::relu(hidden3->forward(x)); // Third hidden layer with ReLU
        x = output->forward(x);               // Output layer
        return x;
    }
};

std::string formatToOFF(const std::vector<float> &array)
{
    std::ostringstream formatted_array;
    formatted_array << "OFF\n8 6 0\n";

    for (size_t i = 0; i < array.size(); ++i)
    {
        formatted_array << array[i] << " ";
        if ((i + 1) % 3 == 0)
        {
            formatted_array << "\n";
        }
    }

    // Get the current formatted string and remove the trailing newline
    std::string vertex_section = formatted_array.str();
    if (!vertex_section.empty() && vertex_section.back() == '\n')
    {
        vertex_section.pop_back();
    }

    // Adding the additional string
    std::string additional_string = R"(
4 0 1 2 3
4 1 5 6 2
4 5 4 7 6
4 4 0 3 7
4 3 2 6 7
4 4 5 1 0)";
    vertex_section += additional_string;

    return vertex_section;
}

int main()
{

    std::string training_directory = "../assets/datasets/austens_boxes/training";
    std::string testing_directory = "../assets/datasets/austens_boxes/testing";

    std::vector<std::pair<std::string, torch::Tensor>> training_vertices;
    std::vector<std::pair<std::string, torch::Tensor>> testing_vertices;

    loadDatasetFromDirectory(training_directory, training_vertices);
    loadDatasetFromDirectory(testing_directory, testing_vertices);

    // Create tensor of tensors
    torch::Tensor trainingTensors = combineTensors(training_vertices);
    torch::Tensor testingTensors = combineTensors(testing_vertices);

    // Print the tensor of tensors
    // std::cout << training_vertices << std::endl;
    // std::cout << trainingTensors << std::endl;
    auto transposed_training_tensor = trainingTensors.transpose(0, 1);
    // std::cout << transposed_training_tensor << std::endl;
    auto transposed_testing_tensor = testingTensors.transpose(0, 1);
    // std::cout << transposed_testing_tensor << std::endl;

    // std::cout << "Testing Tensors:" << std::endl;
    // printTensor(testingTensors);

    // Print the shape of trainingTensors
    // std::cout << "Shape of trainingTensors: " << trainingTensors.sizes() << std::endl;

    std::vector<torch::Tensor> training_number_lists = createTrainingNumberLists(trainingTensors);
    // std::cout << "Training Number Lists" << training_number_lists << std::endl;

    // Set random seed for reproducibility
    torch::manual_seed(1);

    // Create model with the appropriate input size
    int64_t input_size = transposed_training_tensor.size(1); // Assuming trainingTensors is 2D
    NeuralNetwork model(input_size);

    // Define the loss function and optimizer.
    auto CRITERION = torch::nn::MSELoss();
    auto OPTIMIZER = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(0.01));

    // Set number of epochs. I found three to give the lowest loss score.
    const int NUMBER_OF_EPOCHS = 3;
    // TESTING_DATA_ITERATOR = iter(TESTING_INPUT)

    std::string file_path = "./generated_boxes/generated_box.off";

    // Use trainingTensors as input to the model
    auto TRAINING_INPUT = transposed_training_tensor;
    auto TESTING_INPUT = transposed_testing_tensor;

    // The training loop
    for (int EPOCH = 0; EPOCH < NUMBER_OF_EPOCHS; ++EPOCH)
    {
        auto output = model.forward(TRAINING_INPUT);

        // Compute the loss for each of the 10 targets and average the results
        torch::Tensor total_loss = torch::zeros({});
        for (int i = 0; i < TESTING_INPUT.size(1); ++i)
        {
            auto target = TESTING_INPUT.narrow(1, i, 1); // Select column i from TESTING_INPUT
            auto loss = CRITERION(output, target);
            total_loss += loss;
        }
        total_loss /= TESTING_INPUT.size(1);

        std::cout << "Epoch: " << EPOCH + 1 << ", Loss: " << total_loss.item<double>() << std::endl;

        OPTIMIZER.zero_grad();
        total_loss.backward();
        OPTIMIZER.step();
    }

    // List the generated coordinates
    auto output = model.forward(TRAINING_INPUT);

    // Print the output tensor
    // std::cout << "Output Tensor:" << std::endl;
    // std::cout << output << std::endl;

    // List the generated coordinates.
    torch::Tensor detached_output = output.detach().cpu();
    std::vector<float> output_array(detached_output.data_ptr<float>(), detached_output.data_ptr<float>() + detached_output.numel());
    // std::cout << output_array << std::endl;

    std::string output_formatted = formatToOFF(output_array);
    std::cout << output_formatted << std::endl;

    // FORMATTED_ARRAY += ADDITIONAL_STRING;

    // std::cout << "File generated successfully. Saved as:" << file_path << std::endl;
    // std::cout << trainingTensors << std::endl;

    return 0;
}