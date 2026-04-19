#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <mutex>
#include <thread>

#include "capture_thread.hpp"
#include "inference_thread.hpp"
#include "shared_types.hpp"

namespace {

std::atomic<bool> running{true};

void signal_handler(int) {
    running = false;
}

}  // namespace

int main() {
    SharedFrame camera_frame;
    SharedFrame display_frame;

    std::signal(SIGINT, signal_handler);

    std::thread camera_thread(capture_loop, std::ref(camera_frame), std::ref(running));
    std::thread inference_thread(inference_loop, std::ref(camera_frame), std::ref(display_frame), std::ref(running));

    while (running) {
        cv::Mat frame_to_show;
        {
            std::lock_guard<std::mutex> lock(display_frame.mutex);
            if (!display_frame.frame.empty()) {
                frame_to_show = display_frame.frame.clone();
            }
        }

        if (frame_to_show.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cv::imshow("output", frame_to_show);
        if (cv::waitKey(25) == 27) {
            running = false;
        }
    }

    camera_thread.join();
    inference_thread.join();
    return 0;
}
