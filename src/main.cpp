#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include "neural_network.h"
#include "data_formatter.h"

namespace fs = std::filesystem;

int main()
{
    // Load datasets
    load_datasets();

    // Create an instance of the neural network
    NeuralNetwork model;

    // Define the loss function and optimizer
    auto criterion = torch::nn::MSELoss();
    auto optimizer = torch::optim::SGD(model.parameters(), torch::optim::SGDOptions(0.01));

    // Set number of epochs
    const int number_of_epochs = 3;

    // The training loop
    for (int epoch = 0; epoch < number_of_epochs; ++epoch)
    {
        // Generate box coordinates
        auto output = model.forward(TRAINING_COMBINED_TENSOR);

        // Compare the generated coordinates against the test coordinates
        auto loss = criterion(output, TESTING_COMBINED_TENSOR);

        // Print the loss every epoch
        std::cout << "Epoch: " << epoch + 1 << ", Loss: " << loss.item<double>() << std::endl;

        // Backward pass and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    // List the generated coordinates
    auto output = model.forward(TRAINING_COMBINED_TENSOR);
    auto array = output.detach().cpu().flatten();

    // Format to .off
    std::string formatted_array = "OFF\n8 6 0\n";
    for (size_t i = 0; i < array.size(0); ++i)
    {
        formatted_array += std::to_string(array[i].item<float>() * 22) + " ";
        if ((i + 1) % 3 == 0)
        {
            formatted_array += "\n";
        }
    }

    std::string additional_string = R"(
4 0 1 2 3
4 1 5 6 2
4 5 4 7 6
4 4 0 3 7
4 3 2 6 7
4 4 5 1 0)";

    formatted_array += additional_string;

    std::string file_path = "./generated_boxes/generated_box.off";

    // Check if the file already exists
    if (fs::exists(file_path))
    {
        // Find the next available file name by incrementing a counter
        int file_counter = 1;
        std::string incremented_file_path;
        do
        {
            incremented_file_path = file_path.substr(0, file_path.find_last_of('.')) + "_" + std::to_string(file_counter) + ".off";
            ++file_counter;
        } while (fs::exists(incremented_file_path));

        file_path = incremented_file_path;
    }

    // Save the .off file
    std::ofstream file(file_path);
    if (file.is_open())
    {
        file << formatted_array;
        file.close();
        std::cout << "File generated successfully. Saved as: " << file_path << std::endl;
    }
    else
    {
        std::cerr << "Unable to open file " << file_path << std::endl;
    }

    return 0;
}
