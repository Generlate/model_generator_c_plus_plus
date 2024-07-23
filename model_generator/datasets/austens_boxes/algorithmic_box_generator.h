#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

const std::vector<std::vector<int>> VERTICES = {
    {-1, -1, -1}, // Vertex 0
    {1, -1, -1},  // Vertex 1
    {1, 1, -1},   // Vertex 2
    {-1, 1, -1},  // Vertex 3
    {-1, -1, 1},  // Vertex 4
    {1, -1, 1},   // Vertex 5
    {1, 1, 1},    // Vertex 6
    {-1, 1, 1}    // Vertex 7
};

const std::vector<std::vector<int>> FACES = {
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
        file << VERTICES.size() << " " << FACES.size() << " 0\n";

        for (const auto &vertex : VERTICES)
        {
            file << vertex[0] * width << " " << vertex[1] * height << " " << vertex[2] * depth << "\n";
        }

        for (const auto &face : FACES)
        {
            file << face.size() << " ";
            for (const auto &vertex_index : face)
            {
                file << vertex_index << " ";
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
