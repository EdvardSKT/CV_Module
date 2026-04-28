#include "threads/queue_thread.hpp"

#include "utilities/BoundedChannel.hpp"
#include "utilities/coordinate_converter.hpp"
#include "utilities/image_processing.hpp"

#include <string>
#include <vector>
#include <cmath>
#include <atomic>
#include <algorithm>
#include <cerrno>
#include <cstring>

#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <iostream>
#include <string>

#include <opencv2/core/types.hpp>

namespace
{

const float CONVEYOR_SPEED = 0.1;
const float Y_DISTANCE_THRESHOLD = 0.01; // TODO: FIND REALISTIC VALUE
const float X_TRAVEL_UNCERTAINTY = 0.01; // TODO: FIND REALISTIC VALUE
const float HOME_X_POSITION = 3.0;
const float END_X_POSITION = 3.5;

const int NEEDED_DETECTIONS_FOR_CONFIRMATION = 3;
const bool LOG_TRACKING = true;

void update_existing_detections(const std::pair<std::vector<RobotCoordinate>, std::chrono::steady_clock::time_point>& incoming_detections, std::vector<BatteryTrack>& existing_detections, std::chrono::steady_clock::time_point& tracks_timestamp){
    auto detection_timestamp = incoming_detections.second;

    size_t num_existing = existing_detections.size();    
    std::vector<bool> used(num_existing, false);
    
    for(const auto& inc_det : incoming_detections.first){
        bool add_track = true;
        for(size_t i = 0; i < num_existing; i++){
            if(used.at(i)){
                continue;
            }

            BatteryTrack& exi_det = existing_detections.at(i);

            std::chrono::duration<double> traveled_time = detection_timestamp - tracks_timestamp;
            double traveled_time_seconds = traveled_time.count();

            double predicted_x = exi_det.coordinate.x + CONVEYOR_SPEED*traveled_time_seconds;

            double distance_from_predicted_x = inc_det.x - predicted_x;
            double distance_y = inc_det.y - exi_det.coordinate.y;

            if(std::fabs(distance_y) <= Y_DISTANCE_THRESHOLD && std::fabs(distance_from_predicted_x) <= X_TRAVEL_UNCERTAINTY){
                used.at(i) = true;

                exi_det.coordinate = inc_det;
                exi_det.last_seen_timestamp = detection_timestamp;

                exi_det.match_counter++;

                if(!exi_det.confirmed && exi_det.match_counter == NEEDED_DETECTIONS_FOR_CONFIRMATION){
                    exi_det.confirmed = true;
                    if (LOG_TRACKING) {
                        std::cout << "Confirmed detection at x=" << exi_det.coordinate.x
                                  << ", y=" << exi_det.coordinate.y << "\n";
                    }
                }

                add_track = false;
                break;
            }
        }

        if(!add_track){
            continue;
        }
        
        existing_detections.push_back(BatteryTrack{inc_det, detection_timestamp});
    }

    double dt = std::chrono::duration<double>(detection_timestamp - tracks_timestamp).count();

    for(size_t i = 0; i < num_existing; i++){
        if(!used.at(i)){
            auto& track = existing_detections.at(i);
            track.coordinate.x += dt*CONVEYOR_SPEED;
        }
    }

    tracks_timestamp = detection_timestamp;

    existing_detections.erase(
        std::remove_if(existing_detections.begin(), existing_detections.end(),
            [](const BatteryTrack& t) {
                return t.coordinate.x > END_X_POSITION;
            }),
        existing_detections.end()
    );

    return;
}

}

void queue_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    int& fd,
    BoundedChannel<BatteryTrack>* active_target_ch
)
{
    std::vector<BatteryTrack> existing_detections;
    std::chrono::steady_clock::time_point tracks_timestamp;

    while (running) {
        auto incoming = ch.recv();
        if(!incoming){
            break;
        }

        std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point> incoming_detections_camera_frame = std::move(incoming.value());
        std::pair<std::vector<RobotCoordinate>, std::chrono::steady_clock::time_point> incoming_detections = {convert_multiple(incoming_detections_camera_frame.first), incoming_detections_camera_frame.second};

        if (LOG_TRACKING) {
            std::cout << "Queue frame: " << incoming_detections.first.size()
                      << " detection(s)";
            for (size_t i = 0; i < incoming_detections.first.size(); ++i) {
                const auto& coordinate = incoming_detections.first.at(i);
                std::cout << " | d" << i
                          << " x=" << coordinate.x
                          << ", y=" << coordinate.y;
            }
            std::cout << "\n";
        }

        update_existing_detections(incoming_detections, existing_detections, tracks_timestamp);

        std::chrono::duration<double> dt = std::chrono::steady_clock::now() - tracks_timestamp;
        double dt_seconds = dt.count();

        for(auto& det : existing_detections) {
            if(det.confirmed && !det.notified){
                double time_until_home_ms = ((HOME_X_POSITION - det.coordinate.x)/CONVEYOR_SPEED - dt_seconds)*1000;

                if(time_until_home_ms < 0){
                    if (LOG_TRACKING) {
                        std::cout << "Confirmed detection already passed home: x="
                                  << det.coordinate.x << ", delay="
                                  << time_until_home_ms << " ms\n";
                    }
                    continue;
                }

                const auto delay_ms = static_cast<long long>(std::llround(time_until_home_ms));
                std::string msg = std::to_string(delay_ms) + "\n";

                const ssize_t bytes_written = write(fd, msg.c_str(), msg.size());
                if (bytes_written < 0) {
                    std::cerr << "Serial write failed: " << std::strerror(errno) << "\n";
                    continue;
                }
                if (static_cast<size_t>(bytes_written) != msg.size()) {
                    std::cerr << "Serial write incomplete: wrote " << bytes_written
                              << " of " << msg.size() << " bytes\n";
                    continue;
                }

                std::cout << "Sent relay delay: " << delay_ms << " ms as bytes ";
                for (const unsigned char byte : msg) {
                    std::cout << "0x" << std::hex << static_cast<int>(byte) << " ";
                }
                std::cout << std::dec << "\n";

                if (active_target_ch != nullptr) {
                    BatteryTrack active_target = det;
                    active_target.is_active_target = true;
                    if (!active_target_ch->send(std::move(active_target))) {
                        std::cerr << "Failed to forward active target to robot thread\n";
                        running = false;
                        break;
                    }
                }

                det.notified = true;
            }
        }
    }
};
