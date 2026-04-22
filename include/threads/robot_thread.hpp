#pragma once

#include <atomic>

#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"

void robot_loop(BoundedChannel<DetectionCenter>& active_target_ch, std::atomic<bool>& running);
