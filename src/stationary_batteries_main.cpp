#include <atomic>
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "threads/computer_vision_thread.hpp"
#include "threads/queue_thread.hpp"
#include "threads/robot_thread.hpp"
#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"
#include "utilities/shared_types.hpp"

#include <opencv2/highgui.hpp>

namespace {

std::atomic<bool> running{true};

void signal_handler(int)
{
    running = false;
}

}  // namespace

int main()
{
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>> detection_ch(10);
    BoundedChannel<RobotCoordinate> position_ch(10);
    SharedFrame display_frame;

    std::signal(SIGINT, signal_handler);

    std::thread detector_thread(cv_loop, std::ref(running), std::ref(detection_ch), &display_frame);
    std::thread queue_thread(
        stationary_batteries_queue_loop,
        std::ref(running),
        std::ref(detection_ch),
        &position_ch
    );
    std::thread robot_thread(robot_loop_2, std::ref(position_ch), std::ref(running));

    uint64_t last_displayed_frame_id = 0;
    bool display_window_created = false;
    while (running) {
        cv::Mat frame;
        std::vector<Detection> detections;
        {
            std::lock_guard<std::mutex> lock(display_frame.mutex);
            if (display_frame.frame_id != last_displayed_frame_id) {
                display_frame.frame.copyTo(frame);
                detections = display_frame.detections;
                last_displayed_frame_id = display_frame.frame_id;
            }
        }

        if (!frame.empty()) {
            draw_detections(frame, detections);
            cv::imshow("Computer Vision", frame);
            display_window_created = true;
            if (cv::waitKey(1) == 27) {
                running = false;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    if (display_window_created) {
        cv::destroyWindow("Computer Vision");
    }

    detection_ch.close();
    position_ch.close();

    detector_thread.join();
    queue_thread.join();
    robot_thread.join();

    return 0;
}
