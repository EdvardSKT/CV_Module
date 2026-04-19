#include "image_processing.hpp"

#include <string>
#include <vector>
#include <cmath>

#include <opencv2/core/types.hpp>

namespace
{

const float CONVEYOR_SPEED = 0.3;
const float Y_DISTANCE_THRESHOLD = 0.001; // TODO: FIND REALISTIC VALUE
const float X_TRAVEL_UNCERTAINTY = 0.001; // TODO: FIND REALISTIC VALUE


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

}

void queue_loop(std::atomic<bool>& running)
{

    while (running) {
        
    }

};