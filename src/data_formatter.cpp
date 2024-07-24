/** Formats the dataset to the proper format for the neural network to process. */

#include "data_processor.h"
#include "data_loader.h" // Include the header where TrainingDataLoader and TestingDataLoader are defined
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <stdexcept>

// Specify where the training and testing boxes are stored and how many to load.

// Apply the class that loads and formats the box data.

// Get numbers from the files and convert to tensor.

// Get testing tensors for each file.

// Combine tensors into a single tensor.