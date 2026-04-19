#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <mutex>
#include <thread>

#include "preprocess.hpp"

void draw_x(cv::Mat& img) {
    int cx = img.cols / 2;
    int cy = img.rows / 2;
    int size = 20;

    cv::Point p1(cx - size, cy - size);
    cv::Point p2(cx + size, cy + size);
    cv::Point p3(cx - size, cy + size);
    cv::Point p4(cx + size, cy - size);

    cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 2);
    cv::line(img, p3, p4, cv::Scalar(0, 0, 255), 2);
}

struct SharedFrame {
    cv::Mat frame;
    std::mutex frame_mtx;
    uint64_t frame_id = 0;
};

std::atomic<bool> running{true};

void signal_handler(int) {
    running = false;
}

void capture_loop(SharedFrame& sharedFrame, std::atomic<bool>& running) {
    cv::VideoCapture camera_feed(0);

    if (!camera_feed.isOpened()) {
        std::cerr << "Could not open camera\n";
        running = false;
        return;
    }

    cv::Mat frame;

    while (running && camera_feed.read(frame)) {
        std::lock_guard<std::mutex> lock(sharedFrame.frame_mtx);
        sharedFrame.frame_id++;
        sharedFrame.frame = frame.clone();
    }
}

void inference_loop(SharedFrame& sharedFrame, std::atomic<bool>& running, cv::Mat& display_frame) {
    constexpr int kInputWidth = 1280;
    constexpr int kInputHeight = 1280;
    uint64_t prevFrameId{0};

    cv::dnn::Net net = cv::dnn::readNetFromONNX("models/best.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    while (running) {
        cv::Mat currFrame;
        uint64_t currFrameId{0};

        {
            std::lock_guard<std::mutex> lock(sharedFrame.frame_mtx);
            currFrameId = sharedFrame.frame_id;

            if (currFrameId != prevFrameId) {
                currFrame = sharedFrame.frame.clone();
            }
        }

        if (currFrameId == prevFrameId) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        prevFrameId = currFrameId;

        if (currFrame.empty()) {
            continue;
        }

        PreprocessResult prep;
        try {
            prep = preprocessYOLO(currFrame, kInputWidth, kInputHeight);
        } catch (const std::exception& e) {
            std::cerr << "Preprocessing failed: " << e.what() << "\n";
            continue;
        }

        int blob_sizes[] = {1, 3, kInputHeight, kInputWidth};
        cv::Mat blob(4, blob_sizes, CV_32F, prep.tensor.data());

        net.setInput(blob);
        cv::Mat out = net.forward();

        cv::Mat debug_frame = currFrame.clone();
        draw_x(debug_frame);

        cv::putText(
            debug_frame,
            "Forward pass OK",
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2
        );

        std::string output_shape = "Output: ";
        for (int i = 0; i < out.dims; ++i) {
            output_shape += std::to_string(out.size[i]);
            if (i + 1 < out.dims) {
                output_shape += "x";
            }
        }

        cv::putText(
            debug_frame,
            output_shape,
            cv::Point(20, 80),
            cv::FONT_HERSHEY_SIMPLEX,
            0.8,
            cv::Scalar(0, 255, 0),
            2
        );

        display_frame = debug_frame;
        std::cout << output_shape << "\n";
    }
}

int main() {
    SharedFrame sharedFrame;
    std::signal(SIGINT, signal_handler);
    cv::Mat display_frame;

    std::thread camera_thread(capture_loop, std::ref(sharedFrame), std::ref(running));
    std::thread inference_thread(inference_loop, std::ref(sharedFrame), std::ref(running), std::ref(display_frame));

    while (running) {
        if (display_frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cv::imshow("output", display_frame);
        if (cv::waitKey(25) == 27) {
            running = false;
        }
    }

    camera_thread.join();
    inference_thread.join();

    return 0;
}
