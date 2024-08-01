/** Runs a formatted dataset through a neural network, formats the output to be viewed as a 3d object and file name. */

#include "neural_network.h"
#include "cli_args.h"
#include "cube_generator.h"

int main(int argc, char **argv)
{
    if (cliArgs(argc, argv))
    {
        return 0;
    }

    CubeGenerator app(argc, argv);
    return app.run();
}

// TODO: use Doxygen
// TODO: compare to hazel
// TODO: add a package manager (NuGet, Conan, vcpkg)
// TODO: change to c++ standard 2020 to match google syle guide
// TODO: convert structs to class
// TODO: check names against google's style