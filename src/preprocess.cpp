#include "preprocess.hpp"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cstring>
#include <vector>
#include <stdexcept>

PreprocessResult preprocessYOLO(const cv::Mat& image, int input_w, int input_h) {
    if (image.empty()) {
        throw std::runtime_error("Input image is empty");
    }
    if (image.channels() != 3) {
        throw std::runtime_error("Expected 3-channel color image");
    }

    // 1. BGR -> RGB
    cv::Mat rgb;
    cv::cvtColor(image, rgb, cv::COLOR_BGR2RGB);

    // 2. Compute letterbox resize
    float scale = std::min(
        static_cast<float>(input_w) / static_cast<float>(rgb.cols),
        static_cast<float>(input_h) / static_cast<float>(rgb.rows)
    );

    int new_w = static_cast<int>(std::round(rgb.cols * scale));
    int new_h = static_cast<int>(std::round(rgb.rows * scale));

    cv::Mat resized;
    cv::resize(rgb, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    // 3. Pad to target size
    int pad_x = (input_w - new_w) / 2;
    int pad_y = (input_h - new_h) / 2;
    int pad_right = input_w - new_w - pad_x;
    int pad_bottom = input_h - new_h - pad_y;

    cv::Mat letterboxed;
    cv::copyMakeBorder(
        resized,
        letterboxed,
        pad_y, pad_bottom,
        pad_x, pad_right,
        cv::BORDER_CONSTANT,
        cv::Scalar(114, 114, 114)
    );

    // 4. Convert to float32 and normalize to [0,1]
    cv::Mat float_img;
    letterboxed.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // 5. HWC -> CHW
    std::vector<cv::Mat> channels(3);
    cv::split(float_img, channels);

    std::vector<float> tensor(3 * input_w * input_h);
    size_t channel_size = static_cast<size_t>(input_w) * input_h;

    for (int c = 0; c < 3; ++c) {
        std::memcpy(
            tensor.data() + c * channel_size,
            channels[c].data,
            channel_size * sizeof(float)
        );
    }

    return {tensor, scale, pad_x, pad_y};
}
