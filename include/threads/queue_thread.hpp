#pragma once

#include <atomic>
#include <chrono>
#include <utility>
#include <vector>

#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"

struct BatteryTrack {
    RobotCoordinate coordinate;
    std::chrono::steady_clock::time_point last_seen_timestamp;
    int match_counter = 0;
    bool confirmed = false;
    bool notified = false;
    bool is_active_target = false;
};

void queue_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    int& fd
);
