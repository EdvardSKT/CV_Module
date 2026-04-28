#include "threads/video_file_computer_vision_thread.hpp"

#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace {

constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;

std::filesystem::path resolve_model_path() {
    const std::filesystem::path cwd_model = std::filesystem::current_path() / "models" / "best.onnx";
    if (std::filesystem::exists(cwd_model)) {
        return cwd_model;
    }

#ifdef CV_MODULE_SOURCE_DIR
    const std::filesystem::path source_model = std::filesystem::path(CV_MODULE_SOURCE_DIR) / "models" / "best.onnx";
    if (std::filesystem::exists(source_model)) {
        return source_model;
    }
#endif

    throw std::runtime_error("Could not find models/best.onnx from the current directory or project source directory");
}

}  // namespace

void video_file_cv_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    const std::string& mp4_path,
    SharedFrame* shared_frame
) {
    cv::VideoCapture video(mp4_path);
    if (!video.isOpened()) {
        std::cerr << "Could not open video file: " << mp4_path << "\n";
        running = false;
        return;
    }

    cv::dnn::Net net;
    try {
        const auto model_path = resolve_model_path();
        std::cout << "Loading model: " << model_path << "\n";
        net = cv::dnn::readNetFromONNX(model_path.string());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load ONNX model: " << e.what() << "\n";
        running = false;
        return;
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    const double fps = video.get(cv::CAP_PROP_FPS);
    const int delay_ms = fps > 0.0 ? static_cast<int>(1000.0 / fps) : 0;

    cv::Mat frame;
    while (running && video.read(frame)) {
        const auto timestamp = std::chrono::steady_clock::now();

        PreprocessResult prep;
        try {
            prep = preprocessYOLO(frame, kInputWidth, kInputHeight);
        } catch (const std::exception& e) {
            std::cerr << "Preprocessing failed: " << e.what() << "\n";
            continue;
        }

        int blob_sizes[] = {1, 3, kInputHeight, kInputWidth};
        cv::Mat blob(4, blob_sizes, CV_32F, prep.tensor.data());

        cv::Mat out;
        try {
            net.setInput(blob);
            out = net.forward();
        } catch (const cv::Exception& e) {
            std::cerr << "Forward pass failed: " << e.what() << "\n";
            running = false;
            return;
        }

        const auto detections = decode_detections(out, prep, frame.size());
        const auto current_detections = detection_centers_from_detections(detections);

        if (shared_frame != nullptr) {
            std::lock_guard<std::mutex> lock(shared_frame->mutex);
            frame.copyTo(shared_frame->frame);
            shared_frame->detections = detections;
            ++shared_frame->frame_id;
        }

        if (!ch.send(std::make_pair(current_detections, timestamp))) {
            break;
        }

        if (delay_ms > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }
    }

    running = false;
}
