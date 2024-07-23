#ifndef BOX_GENERATOR_H
#define BOX_GENERATOR_H

#include <string>
#include <vector>

void generate_box(int width, int height, int depth, const std::string &filename);
void generate_multiple_boxes(int num_boxes);

#endif // BOX_GENERATOR_H
