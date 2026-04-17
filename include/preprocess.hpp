#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>

struct PreprocessResult {
    std::vector<float> tensor;  // CHW, size = 3 * input_w * input_h
    float scale;
    int pad_x;
    int pad_y;
};

PreprocessResult preprocessYOLO(const cv::Mat& image, int input_w = 1280, int input_h = 1280);