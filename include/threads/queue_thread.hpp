#pragma once

#include <atomic>
#include <chrono>
#include <utility>
#include <vector>
#include <string>

#include "utilities/BoundedChannel.hpp"
#include "utilities/coordinate_converter.hpp"
#include "utilities/image_processing.hpp"

struct BatteryTrack {
    RobotCoordinate coordinate;
    std::chrono::steady_clock::time_point last_seen_timestamp;
    int match_counter = 0;
    bool confirmed = false;
    bool notified = false;
    std::string battery_type = "?";
};

void queue_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    int& fd,
    BoundedChannel<double>* battery_y_offset_ch = nullptr
);

void stationary_batteries_queue_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    BoundedChannel<RobotCoordinate>* position_ch
);
