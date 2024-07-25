/** Runs a formatted dataset through a neural network, formats the output to be viewed as a 3d object and file name. */

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <filesystem>
// #include "neural_network.h"
// #include "data_formatter.h"
#include "data_loader.h"

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

int main(int argc, char *argv[])
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <name>" << std::endl;
        return 1;
    }

    std::string name = argv[1];

    TrainingDataLoader person(name);

    // Print using member function
    person.printName();

    // Print using getter (optional)
    std::cout << "File generated successfully. Saved as: " << person.getName() << std::endl;

    TestingDataLoader person2(name);

    // Print using member function
    person2.printName2();

    // Print using getter (optional)
    std::cout << "File generated successfully. Saved as: " << person2.getName2() << std::endl;

    return 0;
}