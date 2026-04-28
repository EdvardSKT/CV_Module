#pragma once

#include <atomic>
#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"
#include "utilities/shared_types.hpp"

void video_file_cv_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    const std::string& mp4_path,
    SharedFrame* shared_frame = nullptr
);
