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
    CubeGenerator(int argc, char **argv);

    int ExtractNumberFromFilename(const std::string &filename);
    bool ReadOffFile(const std::string &filename, std::vector<std::pair<std::string, torch::Tensor>> &vertices);
    std::vector<std::pair<std::string, torch::Tensor>> LoadOffFilesFromDirectory(const std::string &directory);
    torch::Tensor CombineTensors(const std::vector<std::pair<std::string, torch::Tensor>> &vertices);
    void TrainModel(NeuralNetwork &model, torch::Tensor &TRAINING_INPUT, torch::Tensor &TRAINING_TARGET);
    std::string FormatToOFF(const torch::Tensor &tensor);
    void SaveOffFile(const std::string &originalFilePath, const std::string &formattedArray);
    int run();
};