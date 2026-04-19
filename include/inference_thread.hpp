#pragma once

#include <atomic>

#include "shared_types.hpp"

void inference_loop(SharedFrame& camera_frame, SharedFrame& display_frame, std::atomic<bool>& running);
