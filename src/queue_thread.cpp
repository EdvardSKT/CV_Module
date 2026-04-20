#include "image_processing.hpp"
#include "BoundedChannel.hpp"
#include "queue_thread.hpp"

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


std::vector<DetectionCenter> find_new_detections(std::vector<DetectionCenter> incoming_detections, std::vector<DetectionCenter> existing_detections){

    std::vector<DetectionCenter> new_detections;

    for(auto new_det : incoming_detections){
        for(auto old_det : existing_detections){

            double traveled_time = 1; // TODO: IMPLEMENT TIME FOR EACH FRAME AND USE IT HERE 
            double predicted_x = old_det.center.x + CONVEYOR_SPEED*traveled_time;

            double distance_from_predicted_x = (new_det.center.x - predicted_x);
            double distance_y = (new_det.center.y - old_det.center.y);

            if(std::fabs(distance_y) <= Y_DISTANCE_THRESHOLD && std::fabs(distance_from_predicted_x) <= X_TRAVEL_UNCERTAINTY){
                new_detections.push_back(new_det);
                break;
            }
        }
    }

    return new_detections;
};

float find_pulse_delay_ms(DetectionCenter detection){
    float current_x = detection.center.x;
    return (HOME_X_POSITION - current_x)/CONVEYOR_SPEED * 1000;
};

}

void queue_loop(std::atomic<bool>& running, BoundedChannel<std::vector<DetectionCenter>>& ch, int& fd)
{
    std::vector<DetectionCenter> existing_detections;

    while (running) {
        auto incoming = ch.recv();
        if(!incoming){
            break;
        }

        std::vector<DetectionCenter> incoming_detections = std::move(incoming.value());

        auto new_detections = find_new_detections(incoming_detections, existing_detections);

        for(auto det : new_detections){
            float delay_ms = find_pulse_delay_ms(det);
            std::string msg = std::to_string(delay_ms) + "\n";

            write(fd, msg.c_str(), msg.size());
        }

    }

};