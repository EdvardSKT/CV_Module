#include "threads/robot_thread.hpp"

#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"
#include "vendor/YMConnect.h"

#include <atomic>
#include <iostream>

namespace {

const UINT32 CONVEYOR_USER_COORDINATE_NUMBER = 1;
const UINT16 BATTERY_OFFSET_VARIABLE_NUMBER = 0;

}

void robot_loop(BoundedChannel<DetectionCenter>& active_target_ch, std::atomic<bool>& running) {
    StatusInfo status{};
    auto c = YMConnect::OpenConnection("192.168.1.31", status);

    while (running) {
        auto incoming = active_target_ch.recv();
        if(!incoming){
            std::cout << "nothing on the channel" << std::endl;
            break;
        }

        DetectionCenter active_target = *incoming;
        DOUBLE64 crossTrack_offset = active_target.center.y;

        RobotPositionVariableData robotPositionVariableData{};
        CoordinateArray battery_offset{};

        battery_offset.at(AxisIndex::CartesianAxis::X) = 0;
        battery_offset.at(AxisIndex::CartesianAxis::Y) = crossTrack_offset;
        battery_offset.at(AxisIndex::CartesianAxis::Z) = 0;
        battery_offset.at(AxisIndex::CartesianAxis::Rx) = 0;
        battery_offset.at(AxisIndex::CartesianAxis::Ry) = 0;
        battery_offset.at(AxisIndex::CartesianAxis::Rz) = 0;

        robotPositionVariableData.variableIndex = BATTERY_OFFSET_VARIABLE_NUMBER;
        robotPositionVariableData.positionData.coordinateType = CoordinateType::UserCoordinate;
        robotPositionVariableData.positionData.userCoordinateNumber = CONVEYOR_USER_COORDINATE_NUMBER;
        robotPositionVariableData.positionData.axisData = battery_offset;

        status = c->Variables->RobotPositionVariable->Write(robotPositionVariableData);

        std::cout << status << std::endl;
    }

    YMConnect::CloseConnection(c);
};
