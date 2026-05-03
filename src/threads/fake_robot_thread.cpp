#include "threads/fake_robot_thread.hpp"

#include <iostream>
#include <thread>

namespace {

const UINT32 CONVEYOR_USER_COORDINATE_NUMBER = 1;
const UINT16 BATTERY_OFFSET_VARIABLE_NUMBER = 0;

StatusInfo ok_status()
{
    return StatusInfo{};
}

RobotPositionVariableData generate_robot_position_variable(const BatteryTrack& active_target)
{
    const DOUBLE64 cross_track_offset = active_target.coordinate.y;

    RobotPositionVariableData robot_position_variable_data{};
    CoordinateArray battery_offset{};

    battery_offset.at(AxisIndex::CartesianAxis::X) = 0;
    battery_offset.at(AxisIndex::CartesianAxis::Y) = cross_track_offset;
    battery_offset.at(AxisIndex::CartesianAxis::Z) = 0;
    battery_offset.at(AxisIndex::CartesianAxis::Rx) = 0;
    battery_offset.at(AxisIndex::CartesianAxis::Ry) = 0;
    battery_offset.at(AxisIndex::CartesianAxis::Rz) = 0;

    robot_position_variable_data.variableIndex = BATTERY_OFFSET_VARIABLE_NUMBER;
    robot_position_variable_data.positionData.coordinateType = CoordinateType::UserCoordinate;
    robot_position_variable_data.positionData.userCoordinateNumber = CONVEYOR_USER_COORDINATE_NUMBER;
    robot_position_variable_data.positionData.axisData = battery_offset;

    return robot_position_variable_data;
}

void log_status_if_error(const char* action, const StatusInfo& status)
{
    if (!status.IsOk()) {
        std::cerr << action << " failed: " << status << std::endl;
    }
}

}  // namespace

bool StatusInfo::IsOk() const
{
    return code == 0;
}

std::ostream& operator<<(std::ostream& os, const StatusInfo& status)
{
    os << "StatusInfo{code=" << status.code;
    if (!status.message.empty()) {
        os << ", message=\"" << status.message << "\"";
    }
    os << "}";
    return os;
}

FakeRobotController::FakeRobotController(std::chrono::milliseconds pick_cycle_time)
    : pick_cycle_time_(pick_cycle_time)
{
}

bool FakeRobotController::is_connected() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return connected_;
}

StatusInfo FakeRobotController::read_bit(UINT32 address, bool& value)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!connected_) {
        return StatusInfo{-1, "Fake robot controller is disconnected"};
    }

    constexpr UINT32 pick_finished_address = 10010;
    constexpr UINT32 ready_for_offset_address = 10011;

    if (address == ready_for_offset_address) {
        value = ready_for_offset_;
        return ok_status();
    }

    if (address == pick_finished_address) {
        if (!pick_finished_ && pick_finished_at_ != std::chrono::steady_clock::time_point{} &&
            std::chrono::steady_clock::now() >= pick_finished_at_) {
            pick_finished_ = true;
            ready_for_offset_ = true;
        }

        value = pick_finished_;
        return ok_status();
    }

    return StatusInfo{-2, "Unsupported fake IO address"};
}

StatusInfo FakeRobotController::write_position_variable(const RobotPositionVariableData& value)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!connected_) {
        return StatusInfo{-1, "Fake robot controller is disconnected"};
    }

    written_positions_.push_back(value);
    ready_for_offset_ = false;
    pick_finished_ = false;
    pick_finished_at_ = std::chrono::steady_clock::now() + pick_cycle_time_;

    const auto& axis_data = value.positionData.axisData;
    std::cout << "[fake_robot] received offset write: variable=" << value.variableIndex
              << " x=" << axis_data.at(AxisIndex::CartesianAxis::X)
              << " y=" << axis_data.at(AxisIndex::CartesianAxis::Y)
              << " z=" << axis_data.at(AxisIndex::CartesianAxis::Z) << std::endl;

    return ok_status();
}

void FakeRobotController::set_ready_for_offset(bool ready)
{
    std::lock_guard<std::mutex> lock(mutex_);
    ready_for_offset_ = ready;
}

std::vector<RobotPositionVariableData> FakeRobotController::written_positions() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return written_positions_;
}

void robot_loop(
    BoundedChannel<BatteryTrack>& active_target_ch,
    std::atomic<bool>& running
)
{
    FakeRobotController controller;
    robot_loop(active_target_ch, running, controller, RobotLoopConfig{});
}

void robot_loop(
    BoundedChannel<BatteryTrack>& active_target_ch,
    std::atomic<bool>& running,
    RobotController& controller,
    const RobotLoopConfig& config
)
{
    bool ready_for_offset = false;
    bool pick_finished = false;

    while (running) {
        auto incoming = active_target_ch.recv();
        if (!incoming) {
            std::cout << "nothing on the channel" << std::endl;
            break;
        }

        BatteryTrack active_target = *incoming;
        const RobotPositionVariableData battery_offset = generate_robot_position_variable(active_target);

        while (running && !ready_for_offset) {
            std::this_thread::sleep_for(config.ready_poll_interval);
            const StatusInfo status = controller.read_bit(config.ready_for_offset_address, ready_for_offset);
            log_status_if_error("Read READY_FOR_OFFSET", status);
            if (!status.IsOk()) {
                running = false;
                return;
            }
        }
        ready_for_offset = false;

        const StatusInfo write_status = controller.write_position_variable(battery_offset);
        std::cout << write_status << std::endl;
        if (!write_status.IsOk()) {
            running = false;
            return;
        }

        while (running && !pick_finished) {
            std::this_thread::sleep_for(config.pick_finished_poll_interval);
            const StatusInfo status = controller.read_bit(config.pick_finished_address, pick_finished);
            log_status_if_error("Read PICK_FINISHED", status);
            if (!status.IsOk()) {
                running = false;
                return;
            }
        }
        pick_finished = false;
    }
}
