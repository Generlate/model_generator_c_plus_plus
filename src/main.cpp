/** Runs a formatted dataset through a neural network, formats the output to be viewed as a 3d object and file name. */

#include "neural_network.h"
#include "cube_generator.h"

int main(int argc, char **argv)
{
    CubeGenerator app(argc, argv);
    return app.run();
}

// todo: make separate .cpp and .h files  to improve readability
// todo: put .cpp in src/ and .h into include/
// todo: make main.cpp the neural network definition, make a trainer file and make a cube generator file
// todo: use Doxygen
// todo compare to hazel
// todo: delete reference/