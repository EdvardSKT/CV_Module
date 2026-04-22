#pragma once

#include <cstdint>
#include <mutex>

#include <opencv2/core/mat.hpp>

struct SharedFrame {
    cv::Mat frame;
    std::mutex mutex;
    uint64_t frame_id = 0;
};
