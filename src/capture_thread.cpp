#include "capture_thread.hpp"

#include <iostream>

#include <opencv2/opencv.hpp>

void capture_loop(SharedFrame& shared_frame, std::atomic<bool>& running) {
    cv::VideoCapture camera_feed(0);

    if (!camera_feed.isOpened()) {
        std::cerr << "Could not open camera\n";
        running = false;
        return;
    }

    cv::Mat frame;
    while (running && camera_feed.read(frame)) {
        std::lock_guard<std::mutex> lock(shared_frame.mutex);
        shared_frame.frame = frame.clone();
        shared_frame.frame_id++;
    }
}
