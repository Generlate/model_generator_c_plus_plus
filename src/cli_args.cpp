#include "cli_args.h"

const char *VERSION = "CubeGenerator 0.0.1";
const char *HELP = "Help page: \n --version: gets version information. \n --help: explains commands";

bool cliArgs(int argc, char **argv)
{
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if (arg == "--version")
        {
            std::cout << VERSION << std::endl;
            return true;
        }
        else if (arg == "--help")
        {
            std::cout << HELP << std::endl;
            return true;
        }
    }
    return false;
}