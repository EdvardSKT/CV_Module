#include "threads/robot_thread.hpp"
#include "vendor/YMConnect.h"

#include <iostream>
#include <thread>
#include <utility>
#include <map>

namespace {

const UINT32 CONVEYOR_USER_COORDINATE_NUMBER = 1;
const UINT16 BATTERY_OFFSET_VARIABLE_NUMBER = 0;

const DOUBLE64 X_OFFSET = 8;

const std::map<std::string, DOUBLE64> BATTERY_Z_OFFSETS_MM = {
    {"Alkalisk Ax2", -5.0},
    {"Alkalisk Ax3", 0.0},
    {"Lithium knappecelle", -10.0},
    {"Super Alkalisk",8.0}
};

StatusInfo ok_status()
{
    return StatusInfo{};
}

RobotPositionVariableData generate_robot_position_variable(const double& battery_y_offset)
{
    const DOUBLE64 cross_track_offset = battery_y_offset;

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

RobotPositionVariableData generate_robot_position_variable(const RobotCoordinate& battery_position)
{
    const DOUBLE64 x = battery_position.x;
    const DOUBLE64 y = battery_position.y;

    RobotPositionVariableData robot_position_variable_data{};
    CoordinateArray battery_offset{};

    battery_offset.at(AxisIndex::CartesianAxis::X) = x*1000 + X_OFFSET;
    battery_offset.at(AxisIndex::CartesianAxis::Y) = y*1000;
    battery_offset.at(AxisIndex::CartesianAxis::Z) = BATTERY_Z_OFFSETS_MM.at(battery_position.battery_type);
    battery_offset.at(AxisIndex::CartesianAxis::Rx) = 0;
    battery_offset.at(AxisIndex::CartesianAxis::Ry) = 0;
    battery_offset.at(AxisIndex::CartesianAxis::Rz) = 180;

    robot_position_variable_data.variableIndex = BATTERY_OFFSET_VARIABLE_NUMBER;
    robot_position_variable_data.positionData.coordinateType = CoordinateType::BaseCoordinate;
    robot_position_variable_data.positionData.axisData = battery_offset;

    return robot_position_variable_data;
}

class YmConnectRobotController final : public RobotController {
public:
    explicit YmConnectRobotController(MotomanController* controller)
        : controller_(controller)
    {
    }

    ~YmConnectRobotController() override
    {
        if (controller_ != nullptr) {
            YMConnect::CloseConnection(controller_);
        }
    }

    bool is_connected() const override
    {
        return controller_ != nullptr;
    }

    StatusInfo read_bit(UINT32 address, bool& value) override
    {
        if (controller_ == nullptr || controller_->Io == nullptr) {
            return StatusInfo{-1, "YMConnect controller is not connected"};
        }

        return controller_->Io->ReadBit(address, value);
    }

    StatusInfo write_position_variable(const RobotPositionVariableData& value) override
    {
        if (controller_ == nullptr || controller_->Variables == nullptr || controller_->Variables->RobotPositionVariable == nullptr) {
            return StatusInfo{-1, "YMConnect robot position variable interface is unavailable"};
        }

        return controller_->Variables->RobotPositionVariable->Write(value);
    }

private:
    MotomanController* controller_{nullptr};
};

void log_status_if_error(const char* action, const StatusInfo& status)
{
    if (!status.IsOk()) {
        std::cerr << action << " failed: " << status << std::endl;
    }
}

}  // namespace

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

std::unique_ptr<RobotController> make_ymconnect_robot_controller(
    const std::string& controller_ip,
    StatusInfo& status
)
{
    auto* controller = YMConnect::OpenConnection(controller_ip, status);
    return std::make_unique<YmConnectRobotController>(controller);
}

void robot_loop(
    BoundedChannel<RobotCoordinate>& position_ch,
    std::atomic<bool>& running
)
{
    RobotLoopConfig config{};
    StatusInfo status{};
    auto controller = make_ymconnect_robot_controller(config.controller_ip, status);

    if (!status.IsOk() || !controller->is_connected()) {
        std::cerr << "Failed to connect to robot controller at " << config.controller_ip
                  << ": " << status << std::endl;
        running = false;
        return;
    }

    robot_loop(position_ch, running, *controller, config);
}

void robot_loop(
    BoundedChannel<RobotCoordinate>& position_ch,
    std::atomic<bool>& running,
    RobotController& controller,
    const RobotLoopConfig& config
)
{
    bool ready_for_offset = false;
    bool pick_finished = false;

    while (running) {
        auto incoming = position_ch.recv();
        if (!incoming) {
            std::cout << "nothing on the channel" << std::endl;
            break;
        }

        RobotCoordinate battery_position = *incoming;
        const RobotPositionVariableData battery_offset = generate_robot_position_variable(battery_position);

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

// STATIONARY BATTERIES


void robot_loop_2(
    BoundedChannel<RobotCoordinate>& position_ch,
    std::atomic<bool>& running
)
{
    StatusInfo status;
    MotomanController* c = YMConnect::OpenConnection("192.168.1.20", status); // Open a connection to the robot controller

    UINT32 write_addr = 10011;
    UINT32 read_addr = 10010;

    if (status.StatusCode != 0)
    {
        std::cout << status << std::endl;
        return;
    }

    while(running){

        std::cout << "Reading from channel..." << std::endl;
        auto incoming = position_ch.recv();
        if (!incoming) {
            std::cout << "nothing on the channel" << std::endl;
            break;
        }
        RobotCoordinate battery_position = *incoming;
        std::cout << "Received battery position on channel: " << battery_position.x << " , " << battery_position.y << std::endl;

        if(battery_position.x > 0.84 && battery_position.x < 1.13 && battery_position.y > -0.2 && battery_position.y < 0.2)
        {
            const RobotPositionVariableData pos = generate_robot_position_variable(battery_position);

            bool robot_ready = false;
            status = c->Io->ReadBit(read_addr, robot_ready);
            while(running && !robot_ready){
                std::cout << "Robot not ready..." << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                status = c->Io->ReadBit(read_addr, robot_ready);
            }
            robot_ready = false;

            status = c->Variables->RobotPositionVariable->Write(pos);
            while(running && status.StatusCode != 0){
                std::cout << "Could not write position variable: " << status << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                status = c->Variables->RobotPositionVariable->Write(pos);
            }
            std::cout << "Wrote position variable to robot: " << battery_position.x << " , " << battery_position.y << std::endl;
    
            status = c->Io->WriteBit(write_addr, 1); // IF THIS FAILS THE PROGRAM JUST CONTINUES... NOT GOOD
            while(running && status.StatusCode != 0){
                std::cout << "Could not write bit: " << status << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                status = c->Io->WriteBit(write_addr, 1);
            }
            std::cout << "Wrote 'position ready' bit to robot." << std::endl;
            
            bool robot_is_picking = true;
            std::cout << "Robot is picking..." << std::endl;
            status = c->Io->ReadBit(10011, robot_is_picking);
            while(running && robot_is_picking){
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
                status = c->Io->ReadBit(10011, robot_is_picking);
            }
        } else {
            std::cout << "Coordinate out of reach, skipping..." << std::endl;
        }
    }

    YMConnect::CloseConnection(c);
    return;
}