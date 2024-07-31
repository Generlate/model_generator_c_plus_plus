/** Runs a formatted dataset through a neural network, formats the output to be viewed as a 3d object and file name. */

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <ranges>
// #include "neural_network.h"
// #include "data_formatter.h"
// #include "data_loader.h"

namespace fs = std::filesystem;

class CubeGenerator
{
public:
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

    CubeGenerator(int argc, char **argv)
    {
        training_directory = "../assets/datasets/austens_boxes/training";
        target_directory = "../assets/datasets/austens_boxes/target";
    }

    std::string training_directory;
    std::string target_directory;

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
        if (!file)
        {
            std::cerr << "Error opening file: " << filename << '\n';
            return false;
        }

        std::string line;
        if (!std::getline(file, line) || line != "OFF")
        {
            std::cerr << "Invalid OFF file: " << filename << '\n';
            return false;
        }

        size_t numVertices, numFaces, numEdges;
        if (!(file >> numVertices >> numFaces >> numEdges) || !file.good())
        {
            std::cerr << "Error reading metadata from file: " << filename << '\n';
            return false;
        }

        std::vector<float> vertexData(numVertices * 3);
        for (size_t i = 0; i < numVertices; ++i)
        {
            if (!(file >> vertexData[i * 3] >> vertexData[i * 3 + 1] >> vertexData[i * 3 + 2]) || !file.good())
            {
                std::cerr << "Error reading vertices from file: " << filename << '\n';
                return false;
            }
        }

        torch::Tensor vertexTensor = torch::from_blob(vertexData.data(), {static_cast<long>(numVertices), 3}, torch::kFloat32).clone();
        vertices.emplace_back(filename, std::move(vertexTensor));

        return true;
    }

    std::vector<std::pair<std::string, torch::Tensor>> loadOffFilesFromDirectory(const std::string &directory)
    {
        std::vector<std::pair<std::string, torch::Tensor>> vertices;
        std::vector<std::string> filePaths;

        for (const auto &entry : fs::directory_iterator(directory))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".off")
            {
                filePaths.push_back(entry.path().string());
            }
        }

        std::sort(filePaths.begin(), filePaths.end(), [this](const std::string &a, const std::string &b)
                  { return extractNumberFromFilename(a) < extractNumberFromFilename(b); });

        for (const std::string &filePath : filePaths)
        {
            if (!readOffFile(filePath, vertices))
            {
                std::cerr << "Failed to load file: " << filePath << '\n';
            }
        }

        return vertices;
    }

    torch::Tensor combineTensors(const std::vector<std::pair<std::string, torch::Tensor>> &vertices)
    {
        std::vector<torch::Tensor> flattenedTensors;

        for (const std::pair<std::string, torch::Tensor> &entry : vertices)
        {
            torch::Tensor flatTensor = entry.second.view(-1);
            flattenedTensors.push_back(flatTensor);
        }

        torch::Tensor result = torch::stack(flattenedTensors);
        return result;
    }

    void trainModel(NeuralNetwork &model, torch::Tensor &TRAINING_INPUT, torch::Tensor &TRAINING_TARGET)
    {
        torch::nn::MSELoss lossFunction;
        torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.01));

        const short NUMBER_OF_EPOCHS = 2;
        for (short EPOCH : std::views::iota(0, NUMBER_OF_EPOCHS))
        {
            model.train();
            optimizer.zero_grad();
            torch::Tensor output = model.forward(TRAINING_INPUT);
            torch::Tensor loss = lossFunction(output, TRAINING_TARGET);
            loss.backward();
            optimizer.step();
            std::cout << "Epoch [" << EPOCH + 1 << "/" << NUMBER_OF_EPOCHS << "], Loss: " << loss.item<float>() << '\n';
        }
    }

    std::string formatToOFF(const torch::Tensor &tensor)
    {
        std::ostringstream formatted_array;
        formatted_array << "OFF\n8 6 0\n";

        auto flattened = tensor.view({-1});
        size_t numVertices = 8;
        for (auto i : std::views::iota(0u, numVertices))
        {
            const size_t baseIndex = i * 3;
            formatted_array << flattened[baseIndex].item<float>() << " "
                            << flattened[baseIndex + 1].item<float>() << " "
                            << flattened[baseIndex + 2].item<float>() << "\n";
        }

        std::string vertex_section = formatted_array.str();
        if (!vertex_section.empty() && vertex_section.back() == '\n')
        {
            vertex_section.pop_back();
        }

        vertex_section += R"(
4 0 1 2 3
4 1 5 6 2
4 5 4 7 6
4 4 0 3 7
4 3 2 6 7
4 4 5 1 0)";

        return vertex_section;
    }

    void saveOffFile(const std::string &originalFilePath, const std::string &formattedArray)
    {
        std::string filePath = originalFilePath;
        std::string fileName = filePath.substr(0, filePath.find_last_of('.'));
        std::string fileExtension = filePath.substr(filePath.find_last_of('.'));

        int fileCounter = 1;
        while (std::filesystem::exists(filePath))
        {
            filePath = fileName + "_" + std::to_string(fileCounter++) + fileExtension;
        }

        if (std::ofstream file(filePath); file)
        {
            file << formattedArray;
            std::cout << "File generated successfully. Saved as: " << filePath << '\n';
        }
        else
        {
            std::cerr << "Error: Could not open the file for writing." << '\n';
        }
    }

    int run()
    {
        std::vector<std::pair<std::string, torch::Tensor>> training_vector = loadOffFilesFromDirectory(training_directory);
        std::vector<std::pair<std::string, torch::Tensor>> target_vector = loadOffFilesFromDirectory(target_directory);

        const short NOISE = 200;
        torch::Tensor trainingTensor = combineTensors(training_vector).transpose(0, 1) * NOISE;
        torch::Tensor targetTensor = combineTensors(target_vector).transpose(0, 1);

        int inputSize = training_vector.size();
        NeuralNetwork model(inputSize);
        torch::manual_seed(1);

        torch::Tensor TRAINING_TARGET = targetTensor.mean(1, true);

        trainModel(model, trainingTensor, TRAINING_TARGET);

        torch::Tensor output = model.forward(trainingTensor).detach();
        std::string output_formatted = formatToOFF(output);
        std::string file_path = "../assets/generated_boxes/generated_box.off";
        saveOffFile(file_path, output_formatted);

        std::shared_ptr<NeuralNetwork> net = std::make_shared<NeuralNetwork>(inputSize);
        std::string model_save_path = "./model.pt";
        torch::save(net, model_save_path);

        return 0;
    }
};

int main(int argc, char **argv)
{
    CubeGenerator app(argc, argv);
    return app.run();
}

// todo: go through numeric types and check appropriateness
// todo: check pointer types
// todo: check what's assigned to a variable and what's just a function transform
// todo: check allcaps on variables
// todo: make separate .cpp and .h files  to improve readability
// todo: put .cpp in src/ and .h into include/
// todo: use Doxygen
// todo: make main.cpp the neural network definition, make a trainer file and make a cube generator file
// todo: check if i should do more or less error handling
// todo compare to hazel
// todo: delete reference/
// todo: check line breaks
// todo: add app --version function and --help function