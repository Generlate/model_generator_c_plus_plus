/** Runs a formatted dataset through a neural network, formats the output to be viewed as a 3d object and file name. */

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <filesystem>
// #include "neural_network.h"
// #include "data_formatter.h"
// #include "data_loader.h"

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

// Define the neural network structure
struct NeuralNetwork : torch::nn::Module
{
    torch::nn::Linear hidden1{nullptr}, hidden2{nullptr}, hidden3{nullptr}, output{nullptr};

    NeuralNetwork(int64_t inputSize)
    {
        hidden1 = register_module("hidden1", torch::nn::Linear(inputSize, 8));
        hidden2 = register_module("hidden2", torch::nn::Linear(8, 8));
        hidden3 = register_module("hidden3", torch::nn::Linear(8, 8));
        output = register_module("output", torch::nn::Linear(8, 1));
    }

    torch::Tensor forward(torch::Tensor x)
    {
        x = torch::relu(hidden1->forward(x));
        x = torch::relu(hidden2->forward(x));
        x = torch::relu(hidden3->forward(x));
        x = output->forward(x);
        return x;
    }

    // Implement the save method
    void save(torch::serialize::OutputArchive &archive) const override
    {
        torch::nn::Module::save(archive);
    }

    // Implement the load method
    void load(torch::serialize::InputArchive &archive) override
    {
        torch::nn::Module::load(archive);
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

// Function to save the .off file
void saveOffFile(std::string filePath, const std::string &formattedArray)
{
    // Check if the file already exists
    if (std::filesystem::exists(filePath))
    {
        // Find the next available file name by incrementing a counter
        int fileCounter = 1;
        std::string fileName, fileExtension, incrementedFilePath;

        // Split the file path into name and extension
        std::string::size_type pos = filePath.find_last_of('.');
        if (pos != std::string::npos)
        {
            fileName = filePath.substr(0, pos);
            fileExtension = filePath.substr(pos);
        }
        else
        {
            fileName = filePath;
            fileExtension = "";
        }

        // Find the next available file name
        do
        {
            incrementedFilePath = fileName + "_" + std::to_string(fileCounter) + fileExtension;
            fileCounter++;
        } while (std::filesystem::exists(incrementedFilePath));

        filePath = incrementedFilePath;
    }

    // Open the file in write mode
    std::ofstream file(filePath);

    // Check if the file is open
    if (file.is_open())
    {
        // Write the formatted array to the file
        file << formattedArray;

        // Close the file
        file.close();

        // Print success message
        std::cout << "File generated successfully. Saved as: " << filePath << std::endl;
    }
    else
    {
        // Print error message if the file could not be opened
        std::cerr << "Error: Could not open the file for writing." << std::endl;
    }
}

int main()
{
    std::string training_directory = "../assets/datasets/austens_boxes/training";
    std::string target_directory = "../assets/datasets/austens_boxes/target";

    std::vector<std::pair<std::string, torch::Tensor>> training_vertices;
    std::vector<std::pair<std::string, torch::Tensor>> target_vertices;

    loadDatasetFromDirectory(training_directory, training_vertices);
    loadDatasetFromDirectory(target_directory, target_vertices);

    // Create tensor of tensors
    torch::Tensor trainingTensors = combineTensors(training_vertices);
    torch::Tensor targetTensors = combineTensors(target_vertices);
    int64_t inputSize = training_vertices.size();

    // Create a neural network model
    NeuralNetwork model(inputSize);
    model.to(torch::kCPU);

    // Print the tensor of tensors
    // std::cout << training_vertices << std::endl;
    // std::cout << trainingTensors << std::endl;
    torch::Tensor transposed_training_tensor = trainingTensors.transpose(0, 1);
    // std::cout << transposed_training_tensor << std::endl;
    torch::Tensor transposed_target_tensor = targetTensors.transpose(0, 1);
    // std::cout << transposed_target_tensor << std::endl;

    // Print the shape of trainingTensors
    // std::cout << "Shape of trainingTensors: " << trainingTensors.sizes() << std::endl;

    // Set random seed for reproducibility
    torch::manual_seed(1);
    // todo: check if this needs to be played with.

    // Create model with the appropriate input size
    int64_t number_of_features = transposed_training_tensor.size(1);

    // Define the loss function and optimizer.
    auto CRITERION = torch::nn::MSELoss();
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.01));

    // Set number of epochs. I found three to give the lowest loss score.
    const int NUMBER_OF_EPOCHS = 2;
    const int NOISE = 200;

    // Use trainingTensors as input to the model
    torch::Tensor TRAINING_INPUT = transposed_training_tensor * NOISE;
    torch::Tensor TRAINING_TARGET = transposed_target_tensor.mean(1, /*keepdim=*/true); // Shape: {24, 1}

    for (int epoch = 0; epoch < NUMBER_OF_EPOCHS; ++epoch)
    {
        model.train();

        optimizer.zero_grad();

        torch::Tensor output = model.forward(TRAINING_INPUT);

        // torch::Tensor target = TRAINING_TARGET.select(1, epoch % TRAINING_TARGET.size(1)).view({-1, 1});
        torch::Tensor target = TRAINING_TARGET;

        torch::Tensor loss = CRITERION(output, target);

        loss.backward();
        optimizer.step();

        std::cout << "Epoch [" << epoch + 1 << "/" << NUMBER_OF_EPOCHS << "], Loss: " << loss.item<float>() << std::endl;
    }

    // List the generated coordinates
    torch::Tensor output = model.forward(TRAINING_INPUT);

    std::shared_ptr net = std::make_shared<NeuralNetwork>(inputSize);
    // std::cout << net << std::endl;

    // Save the model
    std::string model_save_path = "../assets/generated_boxes/model.pt";
    torch::save(net, model_save_path);
    // todo: check if the net should be changed to model

    // List the generated coordinates.
    torch::Tensor detached_output = output.detach().cpu();
    std::vector<float> output_array(detached_output.data_ptr<float>(), detached_output.data_ptr<float>() + detached_output.numel());
    // std::cout << output_array << std::endl;

    std::string output_formatted = formatToOFF(output_array);
    // std::cout << output_formatted << std::endl;

    std::string file_path = "../assets/generated_boxes/generated_box.off";

    saveOffFile(file_path, output_formatted);

    return 0;
}
