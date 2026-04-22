#include "image_processing.hpp"
#include "BoundedChannel.hpp"
#include "queue_thread.hpp"
#include "coordinate_converter.hpp"

#include <string>
#include <vector>
#include <cmath>
#include <atomic>

#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <iostream>
#include <string>

#include <opencv2/core/types.hpp>

namespace
{

const float CONVEYOR_SPEED = 0.3;
const float Y_DISTANCE_THRESHOLD = 0.001; // TODO: FIND REALISTIC VALUE
const float X_TRAVEL_UNCERTAINTY = 0.001; // TODO: FIND REALISTIC VALUE
const float HOME_X_POSITION = 0.5;
const float END_X_POSITION = 2.5;

const int NEEDED_DETECTIONS_FOR_CONFIRMATION = 3;

struct BatteryTrack {
    RobotCoordinate coordinate;
    std::chrono::steady_clock::time_point timestamp;
    int match_counter = 0;
    bool confirmed = false;
};

std::vector<RobotCoordinate> find_new_detections(const std::vector<RobotCoordinate> incoming_detections, const std::vector<RobotCoordinate> existing_detections){

    std::vector<RobotCoordinate> new_detections;

    for(auto new_det : incoming_detections){
        for(auto old_det : existing_detections){

            double traveled_time = 1; // TODO: IMPLEMENT TIME FOR EACH FRAME AND USE IT HERE 
            double predicted_x = old_det.x + CONVEYOR_SPEED*traveled_time;

            double distance_from_predicted_x = (new_det.x - predicted_x);
            double distance_y = (new_det.y - old_det.y);

            if(std::fabs(distance_y) <= Y_DISTANCE_THRESHOLD && std::fabs(distance_from_predicted_x) <= X_TRAVEL_UNCERTAINTY){
                new_detections.push_back(new_det);
                break;
            }
        }
    }

    return new_detections;
};

std::vector<BatteryTrack> update_existing_detections(const std::pair<std::vector<RobotCoordinate>, std::chrono::steady_clock::time_point>& incoming_detections, std::vector<BatteryTrack> existing_detections){
    auto timestamp = incoming_detections.second;
    
    if(existing_detections.size() == 0){
        for(const auto& det : incoming_detections.first){
            existing_detections.push_back(BatteryTrack{det, timestamp});
        }
        return existing_detections;
    }

    existing_detections.erase(
        std::remove_if(existing_detections.begin(), existing_detections.end(),
            [](const BatteryTrack& t) {
                return t.coordinate.x > END_X_POSITION;
            }),
        existing_detections.end()
    );
    
    std::vector<bool> used(existing_detections.size(), false);
    
    for(const auto& inc_det : incoming_detections.first){
        for(size_t i = 0; i < existing_detections.size(); i++){
            if(used.at(i)){
                continue;
            }

            BatteryTrack& exi_det = existing_detections.at(i);

            std::chrono::duration<double> traveled_time = timestamp - exi_det.timestamp;
            double traveled_time_seconds = traveled_time.count();

            double predicted_x = exi_det.coordinate.x + CONVEYOR_SPEED*traveled_time_seconds;

            double distance_from_predicted_x = inc_det.x - predicted_x;
            double distance_y = inc_det.y - exi_det.coordinate.y;

            if(std::fabs(distance_y) <= Y_DISTANCE_THRESHOLD && std::fabs(distance_from_predicted_x) <= X_TRAVEL_UNCERTAINTY){
                used.at(i) = true;

                exi_det.coordinate = inc_det;
                exi_det.timestamp = timestamp;

                exi_det.match_counter++;

                if(!exi_det.confirmed && exi_det.match_counter == NEEDED_DETECTIONS_FOR_CONFIRMATION){
                    exi_det.confirmed = true;
                }

                break;
            }
        }
    }
    return existing_detections;
}

float find_pulse_delay_ms(double along_track_position){
    float stamp = 0; // TODO: add stamp to coordinates
    return (HOME_X_POSITION - along_track_position)/CONVEYOR_SPEED * 1000 - stamp;
};

}

void queue_loop(std::atomic<bool>& running, BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch, int& fd)
{
    std::vector<BatteryTrack> existing_detections;

    while (running) {
        auto incoming = ch.recv();
        if(!incoming){
            break;
        }

        std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point> incoming_detections_camera_frame = std::move(incoming.value());
        std::pair<std::vector<RobotCoordinate>, std::chrono::steady_clock::time_point> incoming_detections = {convert_multiple(incoming_detections_camera_frame.first), incoming_detections_camera_frame.second};

        existing_detections = update_existing_detections(incoming_detections, existing_detections);

        
    }

};