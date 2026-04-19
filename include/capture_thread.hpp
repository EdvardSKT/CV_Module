#pragma once

#include <atomic>

#include "shared_types.hpp"

void capture_loop(SharedFrame& shared_frame, std::atomic<bool>& running);
