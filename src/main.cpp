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

    torch::Tensor vertexTensor = torch::from_blob(vertexData.data(), {static_cast<long>(numVertices), 3}, torch::kFloat32).clone();
    vertices.emplace_back(filename, std::move(vertexTensor));

    return true;
}
// todo: review this function

void loadOffFilesFromDirectory(const std::string &directory, std::vector<std::pair<std::string, torch::Tensor>> &vertices)
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

torch::Tensor combineTensors(const std::vector<std::pair<std::string, torch::Tensor>> &vertices)
{
    std::vector<torch::Tensor> flattenedTensors;

    for (const auto &entry : vertices)
    {
        torch::Tensor flatTensor = entry.second.view(-1);
        flattenedTensors.push_back(flatTensor);
    }

    torch::Tensor result = torch::cat(flattenedTensors, 0).view({static_cast<long>(flattenedTensors.size()), -1});
    return result;
}

struct NeuralNetwork : torch::nn::Module
{
    torch::nn::Linear hidden1{nullptr}, hidden2{nullptr}, hidden3{nullptr}, output{nullptr};

    NeuralNetwork(int inputSize)
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

    void save(torch::serialize::OutputArchive &archive) const override
    {
        torch::nn::Module::save(archive);
    }

    void load(torch::serialize::InputArchive &archive) override
    {
        torch::nn::Module::load(archive);
    }
};

std::string formatToOFF(const torch::Tensor &tensor)
{

    std::ostringstream formatted_array;
    formatted_array << "OFF\n8 6 0\n";

    torch::Tensor flattened = tensor.view({-1});
    auto data = flattened.accessor<float, 1>();

    for (size_t i = 0; i < 8; ++i)
    {
        formatted_array << data[i * 3] << " " << data[i * 3 + 1] << " " << data[i * 3 + 2] << "\n";
    }

    std::string vertex_section = formatted_array.str();
    if (!vertex_section.empty() && vertex_section.back() == '\n')
    {
        vertex_section.pop_back(); // Remove the trailing newline
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

void saveOffFile(std::string filePath, const std::string &formattedArray)
{
    if (std::filesystem::exists(filePath))
    {
        int fileCounter = 1;
        std::string fileName, fileExtension, incrementedFilePath;

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

        do
        {
            incrementedFilePath = fileName + "_" + std::to_string(fileCounter) + fileExtension;
            fileCounter++;
        } while (std::filesystem::exists(incrementedFilePath));

        filePath = incrementedFilePath;
    }

    std::ofstream file(filePath);

    if (file.is_open())
    {
        file << formattedArray;

        file.close();

        std::cout << "File generated successfully. Saved as: " << filePath << std::endl;
    }
    else
    {
        std::cerr << "Error: Could not open the file for writing." << std::endl;
    }
}

void trainModel(NeuralNetwork &model,
                torch::Tensor &TRAINING_INPUT,
                torch::Tensor &TRAINING_TARGET)
{
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.01));

    torch::nn::MSELoss lossFunction;

    const short NUMBER_OF_EPOCHS = 2;
    for (short EPOCH = 0; EPOCH < NUMBER_OF_EPOCHS; ++EPOCH)
    {
        model.train();
        optimizer.zero_grad();
        torch::Tensor output = model.forward(TRAINING_INPUT);
        torch::Tensor loss = lossFunction(output, TRAINING_TARGET);
        loss.backward();
        optimizer.step();
        std::cout << "Epoch [" << EPOCH + 1 << "/" << NUMBER_OF_EPOCHS << "], Loss: " << loss.item<float>() << std::endl;
    }
}

int main()
{
    std::string training_directory = "../assets/datasets/austens_boxes/training";
    std::string target_directory = "../assets/datasets/austens_boxes/target";

    std::vector<std::pair<std::string, torch::Tensor>> training_vertices;
    std::vector<std::pair<std::string, torch::Tensor>> target_vertices;

    loadOffFilesFromDirectory(training_directory, training_vertices);
    loadOffFilesFromDirectory(target_directory, target_vertices);

    torch::Tensor trainingTensors = combineTensors(training_vertices);
    torch::Tensor targetTensors = combineTensors(target_vertices);
    int inputSize = training_vertices.size();

    NeuralNetwork model(inputSize);

    torch::Tensor transposed_training_tensor = trainingTensors.transpose(0, 1);
    torch::Tensor transposed_target_tensor = targetTensors.transpose(0, 1);

    torch::manual_seed(1);

    const short NOISE = 200;

    torch::Tensor TRAINING_INPUT = transposed_training_tensor * NOISE;
    torch::Tensor TRAINING_TARGET = transposed_target_tensor.mean(1, /*keepdim=*/true);

    trainModel(model, TRAINING_INPUT, TRAINING_TARGET);

    torch::Tensor output = model.forward(TRAINING_INPUT).detach();
    std::string output_formatted = formatToOFF(output);
    std::string file_path = "../assets/generated_boxes/generated_box.off";
    saveOffFile(file_path, output_formatted);

    std::shared_ptr<NeuralNetwork> net = std::make_shared<NeuralNetwork>(inputSize);
    std::string model_save_path = "./model.pt";
    torch::save(net, model_save_path);
    // todo: check if the net should be changed to model

    return 0;
}

// todo: change training_vertices to load with shape {1, 24} and correctly join to make trainingTensors (do it with targetTensors too)
// todo: check whether training_vertices can be loaded as {24, 1} and trainingTensors {24, 90} to skip transposing them
// todo: check if functions can be simplified
// todo: make separate .cpp and .h files  to improve readability
// todo: put .cpp in src/ and .h into include/
// todo: use Doxygen
// todo: check if i should do more or less error handling
// todo compare to hazel
// todo: update ReadMe
// todo: delete reference/
// todo: check line breaks
// todo: make main.cpp the neural network definition, make a trainer file and make a cube generator file
// todo: check with rider
// todo: go through numeric types and check appropriateness
// todo: check what's assigned to a variable and what's just a function transform
// todo: check main() to see if some variables and functions should be moved into other functions (for readability)
// todo: check allcaps on variables
// todo: compare to python project