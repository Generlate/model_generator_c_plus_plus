#include "cube_generator.h"

namespace fs = std::filesystem;

CubeGenerator::CubeGenerator(int argc, char **argv)
{
}

int CubeGenerator::ExtractNumberFromFilename(const std::string &filename)
{
    try
    {
        std::regex regex("\\d+");
        std::smatch match;
        if (std::regex_search(filename, match, regex) && !match.empty())
        {
            return std::stoi(match.str());
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Error extracting number from filename: " << e.what() << std::endl;
    }
    return 0;
}

bool CubeGenerator::ReadOffFile(const std::string &filename, std::vector<std::pair<std::string, torch::Tensor>> &vertices)
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

    size_t num_vertices, num_faces, num_edges;
    if (!(file >> num_vertices >> num_faces >> num_edges) || !file.good())
    {
        std::cerr << "Error reading metadata from file: " << filename << '\n';
        return false;
    }

    std::vector<float> vertexData(num_vertices * 3);
    for (size_t i = 0; i < num_vertices; ++i)
    {
        if (!(file >> vertexData[i * 3] >> vertexData[i * 3 + 1] >> vertexData[i * 3 + 2]) || !file.good())
        {
            std::cerr << "Error reading vertices from file: " << filename << '\n';
            return false;
        }
    }

    torch::Tensor vertex_tensor = torch::from_blob(vertexData.data(), {static_cast<long>(num_vertices), 3}, torch::kFloat).clone();
    vertices.emplace_back(filename, std::move(vertex_tensor));

    return true;
}

std::vector<std::pair<std::string, torch::Tensor>> CubeGenerator::LoadOffFilesFromDirectory(const std::string &directory)
{
    std::vector<std::pair<std::string, torch::Tensor>> vertices;

    try
    {
        std::vector<std::string> file_paths;

        for (const auto &entry : fs::directory_iterator(directory))
        {
            if (entry.is_regular_file() && entry.path().extension() == ".off")
            {
                file_paths.push_back(entry.path().string());
            }
        }

        std::sort(file_paths.begin(), file_paths.end(), [this](const std::string &a, const std::string &b)
                  { return ExtractNumberFromFilename(a) < ExtractNumberFromFilename(b); });

        for (const std::string &file_path : file_paths)
        {
            if (!ReadOffFile(file_path, vertices))
            {
                std::cerr << "Failed to load file: " << file_path << '\n';
            }
        }
    }
    catch (const std::exception &e)
    {
        std::cerr << "Exception caught while loading OFF files: " << e.what() << std::endl;
    }

    return vertices;
}

torch::Tensor CubeGenerator::CombineTensors(const std::vector<std::pair<std::string, torch::Tensor>> &vertices)
{
    std::vector<torch::Tensor> flattened_tensors;

    for (const std::pair<std::string, torch::Tensor> &entry : vertices)
    {
        torch::Tensor flat_tensor = entry.second.view(-1);
        flattened_tensors.push_back(flat_tensor);
    }

    torch::Tensor result = torch::stack(flattened_tensors);

    return result;
}

void CubeGenerator::TrainModel(NeuralNetwork &model, torch::Tensor &TRAINING_INPUT, torch::Tensor &TRAINING_TARGET)
{
    torch::nn::MSELoss loss_function;
    torch::optim::Adam optimizer(model.parameters(), torch::optim::AdamOptions(0.01));

    const short NUMBER_OF_EPOCHS = 2;
    for (short EPOCH : std::views::iota(0, NUMBER_OF_EPOCHS))
    {
        model.train();
        optimizer.zero_grad();
        torch::Tensor output = model.forward(TRAINING_INPUT);
        torch::Tensor loss = loss_function(output, TRAINING_TARGET);
        loss.backward();
        optimizer.step();
        std::cout << "Epoch [" << EPOCH + 1 << "/" << NUMBER_OF_EPOCHS << "], Loss: " << loss.item<float>() << '\n';
    }
}

std::string CubeGenerator::FormatToOFF(const torch::Tensor &tensor)
{
    std::ostringstream formatted_array;
    formatted_array << "OFF\n8 6 0\n";
    auto flattened = tensor.view({-1});
    size_t num_vertices = 8;

    for (auto i : std::views::iota(0u, num_vertices))
    {
        const size_t BASE_INDEX = i * 3;
        formatted_array << flattened[BASE_INDEX].item<float>() << " "
                        << flattened[BASE_INDEX + 1].item<float>() << " "
                        << flattened[BASE_INDEX + 2].item<float>() << "\n";
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

void CubeGenerator::SaveOffFile(const std::string &originalFilePath, const std::string &formattedArray)
{
    std::string file_path = originalFilePath;
    std::string file_name = file_path.substr(0, file_path.find_last_of('.'));
    std::string file_extension = file_path.substr(file_path.find_last_of('.'));
    int file_counter = 1;

    while (std::filesystem::exists(file_path))
    {
        file_path = file_name + "_" + std::to_string(file_counter++) + file_extension;
    }

    if (std::ofstream file(file_path); file)
    {
        file << formattedArray;
        std::cout << "File generated successfully. Saved as: " << file_path << '\n';
    }
    else
    {
        std::cerr << "Error: Could not open the file for writing." << '\n';
    }
}

int CubeGenerator::run()
{
    std::string TRAINING_DIRECTORY = "../assets/datasets/austens_boxes/training";
    std::string TARGET_DIRECTORY = "../assets/datasets/austens_boxes/target";

    std::vector<std::pair<std::string, torch::Tensor>> training_vector = LoadOffFilesFromDirectory(TRAINING_DIRECTORY);
    std::vector<std::pair<std::string, torch::Tensor>> target_vector = LoadOffFilesFromDirectory(TARGET_DIRECTORY);

    int input_size = training_vector.size();
    NeuralNetwork model(input_size);
    torch::manual_seed(1);
    const short NOISE = 200;

    torch::Tensor training_tensor = CombineTensors(training_vector).transpose(0, 1) * NOISE;
    torch::Tensor target_tensor = CombineTensors(target_vector).transpose(0, 1);

    torch::Tensor TRAINING_TARGET = target_tensor.mean(1, true);

    TrainModel(model, training_tensor, TRAINING_TARGET);

    torch::Tensor output = model.forward(training_tensor).detach();

    std::string output_formatted = FormatToOFF(output);
    std::string FILE_PATH = "../assets/generated_boxes/generated_box.off";
    SaveOffFile(FILE_PATH, output_formatted);

    std::shared_ptr<NeuralNetwork> net = std::make_shared<NeuralNetwork>(input_size);
    std::string MODEL_SAVE_PATH = "./model.pt";
    torch::save(net, MODEL_SAVE_PATH);

    return 0;
}