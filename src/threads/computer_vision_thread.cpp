#include "threads/computer_vision_thread.hpp"

#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

namespace {

// INFERENCE RELATED CONSTANTS AND HELPER FUNCTIONS
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

void cv_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    SharedFrame* shared_frame
){

    // Open camera feed
    cv::VideoCapture camera_feed("/dev/video42", cv::CAP_V4L2);

    if (!camera_feed.isOpened()) {
        std::cerr << "Could not open camera\n";
        running = false;
        return;
    }

    // Open and configure inference model
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

    // Set backend and target
    try {
        if (cv::cuda::getCudaEnabledDeviceCount() > 0) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Using CUDA for OpenCV DNN\n";
        } else {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
            std::cout << "CUDA unavailable, using CPU\n";
        }
    } catch (const cv::Exception& e) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }


    cv::Mat frame;
    while (running && camera_feed.read(frame)) {

        // Time of image
        auto timestamp = std::chrono::steady_clock::now();
        
        // Preprocess image
        PreprocessResult prep;
        try {
            prep = preprocessYOLO(frame, kInputWidth, kInputHeight);
        } catch (const std::exception& e) {
            std::cerr << "Preprocessing failed: " << e.what() << "\n";
            continue;
        }

        int blob_sizes[] = {1, 3, kInputHeight, kInputWidth};
        cv::Mat blob(4, blob_sizes, CV_32F, prep.tensor.data());

        // Forward pass
        cv::Mat out;
        try {
            net.setInput(blob);
            out = net.forward();
        } catch (const cv::Exception& e) {
            std::cerr << "Forward pass failed: " << e.what() << "\n";
            running = false;
            return;
        }

        // Extract detections
        const auto detections = decode_detections(out, prep, frame.size());
        const auto current_detections = detection_centers_from_detections(detections);

        if (shared_frame != nullptr) {
            std::lock_guard<std::mutex> lock(shared_frame->mutex);
            frame.copyTo(shared_frame->frame);
            shared_frame->detections = detections;
            ++shared_frame->frame_id;
        }

        // Send detections on channel 
        if (!ch.send(std::make_pair(current_detections, timestamp))) {
            break;
        }
    }
}
