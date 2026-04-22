#pragma once

#include <atomic>
#include <chrono>
#include <utility>
#include <vector>

#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"

void queue_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    int& fd
);
