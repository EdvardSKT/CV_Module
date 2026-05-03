#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "threads/queue_thread.hpp"
#include "utilities/BoundedChannel.hpp"
#include "vendor/YMConnect.h"

struct RobotLoopConfig {
    std::string controller_ip = "192.168.1.31";
    UINT32 pick_finished_address = 10010;
    UINT32 ready_for_offset_address = 10011;
    std::chrono::milliseconds ready_poll_interval{10};
    std::chrono::milliseconds pick_finished_poll_interval{100};
};

class RobotController {
public:
    virtual ~RobotController() = default;

    virtual bool is_connected() const = 0;
    virtual StatusInfo read_bit(UINT32 address, bool& value) = 0;
    virtual StatusInfo write_position_variable(const RobotPositionVariableData& value) = 0;
};

class FakeRobotController final : public RobotController {
public:
    explicit FakeRobotController(std::chrono::milliseconds pick_cycle_time = std::chrono::milliseconds{2000});

    bool is_connected() const override;
    StatusInfo read_bit(UINT32 address, bool& value) override;
    StatusInfo write_position_variable(const RobotPositionVariableData& value) override;

    void set_ready_for_offset(bool ready);
    std::vector<RobotPositionVariableData> written_positions() const;

private:
    mutable std::mutex mutex_;
    std::chrono::milliseconds pick_cycle_time_;
    bool ready_for_offset_{true};
    bool pick_finished_{false};
    bool connected_{true};
    std::vector<RobotPositionVariableData> written_positions_;
    std::chrono::steady_clock::time_point pick_finished_at_{};
};

std::unique_ptr<RobotController> make_ymconnect_robot_controller(
    const std::string& controller_ip,
    StatusInfo& status
);

void robot_loop(
    BoundedChannel<double>& battery_y_offset_ch,
    std::atomic<bool>& running
);

void robot_loop(
    BoundedChannel<double>& battery_y_offset_ch,
    std::atomic<bool>& running,
    RobotController& controller,
    const RobotLoopConfig& config = {}
);
