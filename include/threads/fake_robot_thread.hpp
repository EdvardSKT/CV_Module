#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

#include "threads/queue_thread.hpp"
#include "utilities/BoundedChannel.hpp"

using UINT16 = std::uint16_t;
using UINT32 = std::uint32_t;
using DOUBLE64 = double;

namespace AxisIndex {
enum CartesianAxis {
    X = 0,
    Y,
    Z,
    Rx,
    Ry,
    Rz
};
}  // namespace AxisIndex

using CoordinateArray = std::array<DOUBLE64, 6>;

enum class CoordinateType {
    UserCoordinate
};

struct StatusInfo {
    int code = 0;
    std::string message;

    bool IsOk() const;
};

std::ostream& operator<<(std::ostream& os, const StatusInfo& status);

struct PositionData {
    CoordinateType coordinateType = CoordinateType::UserCoordinate;
    UINT32 userCoordinateNumber = 0;
    CoordinateArray axisData{};
};

struct RobotPositionVariableData {
    PositionData positionData{};
    UINT16 variableIndex = 0;
};

struct RobotLoopConfig {
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

void robot_loop(
    BoundedChannel<BatteryTrack>& active_target_ch,
    std::atomic<bool>& running
);

void robot_loop(
    BoundedChannel<BatteryTrack>& active_target_ch,
    std::atomic<bool>& running,
    RobotController& controller,
    const RobotLoopConfig& config = {}
);
