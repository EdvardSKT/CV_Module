#pragma once

#include <atomic>
#include <chrono>
#include <utility>
#include <vector>

#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"
#include "utilities/shared_types.hpp"

void cv_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    SharedFrame* shared_frame = nullptr
);
