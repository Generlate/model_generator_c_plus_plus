#include "algorithmic_box_generator.h"
#include <iostream>
#include <fstream>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

const int VERTICES[8][3] = {
    {-1, -1, -1}, // Vertex 0
    {1, -1, -1},  // Vertex 1
    {1, 1, -1},   // Vertex 2
    {-1, 1, -1},  // Vertex 3
    {-1, -1, 1},  // Vertex 4
    {1, -1, 1},   // Vertex 5
    {1, 1, 1},    // Vertex 6
    {-1, 1, 1}    // Vertex 7
};

const int FACES[6][4] = {
    {0, 1, 2, 3}, // Face 0
    {1, 5, 6, 2}, // Face 1
    {5, 4, 7, 6}, // Face 2
    {4, 0, 3, 7}, // Face 3
    {3, 2, 6, 7}, // Face 4
    {4, 5, 1, 0}  // Face 5
};

void generate_box(int width, int height, int depth, const std::string &filename)
{
    std::ofstream file(filename);
    if (file.is_open())
    {
        file << "OFF\n";
        file << 8 << " " << 6 << " 0\n";

        for (int i = 0; i < 8; ++i)
        {
            file << VERTICES[i][0] * width << " " << VERTICES[i][1] * height << " " << VERTICES[i][2] * depth << "\n";
        }

        for (int i = 0; i < 6; ++i)
        {
            file << 4 << " ";
            for (int j = 0; j < 4; ++j)
            {
                file << FACES[i][j] << " ";
            }
            file << "\n";
        }

        file.close();
    }
    else
    {
        std::cerr << "Unable to open file " << filename << std::endl;
    }
}

void generate_multiple_boxes(int num_boxes)
{
    srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < num_boxes; ++i)
    {
        int width = rand() % 10 + 1;
        int height = rand() % 10 + 1;
        int depth = rand() % 10 + 1;
        std::string filename = "box" + std::to_string(i) + ".off";

        generate_box(width, height, depth, filename);

        std::cout << "Box " << i << " with dimensions "
                  << width << "x" << height << "x" << depth
                  << " exported to " << filename << "." << std::endl;
    }
}

int main()
{
    generate_multiple_boxes(100);
    return 0;
}
