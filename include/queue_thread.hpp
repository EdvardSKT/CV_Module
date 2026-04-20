#pragma once

#include "image_processing.hpp"
#include "BoundedChannel.hpp"

#include <atomic>

void queue_loop(std::atomic<bool>& running, BoundedChannel<std::vector<DetectionCenter>>& ch, int& fd);