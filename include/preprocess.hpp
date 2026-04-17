#include <vector>

struct PreprocessResult {
    std::vector<float> tensor;  // CHW, size = 3 * input_w * input_h
    float scale;
    int pad_x;
    int pad_y;
};