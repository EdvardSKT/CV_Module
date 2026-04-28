#pragma once

#include <cstdint>
#include <mutex>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "utilities/image_processing.hpp"

struct SharedFrame {
    cv::Mat frame;
    std::vector<Detection> detections;
    std::mutex mutex;
    uint64_t frame_id = 0;
};
