#include "inference_thread.hpp"

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <thread>

#include "image_processing.hpp"

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

void inference_loop(SharedFrame& camera_frame, SharedFrame& display_frame, std::atomic<bool>& running) {
    uint64_t prev_frame_id{0};

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

    while (running) {
        cv::Mat current_frame;
        uint64_t current_frame_id{0};

        {
            std::lock_guard<std::mutex> lock(camera_frame.mutex);
            current_frame_id = camera_frame.frame_id;
            if (current_frame_id != prev_frame_id) {
                current_frame = camera_frame.frame.clone();
            }
        }

        if (current_frame_id == prev_frame_id || current_frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        prev_frame_id = current_frame_id;

        PreprocessResult prep;
        try {
            prep = preprocessYOLO(current_frame, kInputWidth, kInputHeight);
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

        const auto detections = decode_detections(out, prep, current_frame.size());
        const auto detection_centers = postprocess_detection_centers(out, prep, current_frame.size());

        cv::Mat annotated_frame = current_frame.clone();
        draw_detections(annotated_frame, detections);

        const std::string overlay = "Detections: " + std::to_string(detections.size());
        cv::putText(
            annotated_frame,
            overlay,
            cv::Point(20, 40),
            cv::FONT_HERSHEY_SIMPLEX,
            1.0,
            cv::Scalar(0, 255, 0),
            2
        );

        {
            std::lock_guard<std::mutex> lock(display_frame.mutex);
            display_frame.frame = annotated_frame;
            display_frame.frame_id++;
        }

        std::cout << overlay << "\n";
        for (const auto& det : detection_centers) {
            std::cout << "  battery_type=" << det.battery_type
                      << " center=(" << det.center.x << ", " << det.center.y << ")"
                      << " conf=" << det.confidence << "\n";
        }
    }
}
