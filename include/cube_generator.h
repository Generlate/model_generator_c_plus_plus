#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <ranges>
#include "neural_network.h"

class CubeGenerator
{
public:
    static constexpr const char *VERSION = "CubeGenerator 0.0.1";
    static constexpr const char *HELP = "help page";

    CubeGenerator(int argc, char **argv);

    int extractNumberFromFilename(const std::string &filename);
    bool readOffFile(const std::string &filename, std::vector<std::pair<std::string, torch::Tensor>> &vertices);
    std::vector<std::pair<std::string, torch::Tensor>> loadOffFilesFromDirectory(const std::string &directory);
    torch::Tensor combineTensors(const std::vector<std::pair<std::string, torch::Tensor>> &vertices);
    void trainModel(NeuralNetwork &model, torch::Tensor &TRAINING_INPUT, torch::Tensor &TRAINING_TARGET);
    std::string formatToOFF(const torch::Tensor &tensor);
    void saveOffFile(const std::string &originalFilePath, const std::string &formattedArray);
    int run();
};