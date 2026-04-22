#include "threads/computer_vision_thread.hpp"

#include "utilities/BoundedChannel.hpp"
#include "utilities/coordinate_converter.hpp"
#include "utilities/image_processing.hpp"

#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace {

// INFERENCE RELATED CONSTANTS AND HELPER FUNCTIONS
constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;
constexpr double kMockConveyorSpeed = 0.3;
constexpr double kMockStartX = 1.35;
constexpr double kMockY = 0.8552;
constexpr double kMockSpawnIntervalSeconds = 2.0;
constexpr double kMockVisibleDurationSeconds = 5.0;
constexpr auto kMockFramePeriod = std::chrono::milliseconds(100);
constexpr auto kMockTimestampStep = kMockFramePeriod;
constexpr int kMockFrameCount = 100;
constexpr int kMockBatteryCount = 4;

const cv::Matx33d kMockCameraMatrix{
    874.62127813, 0.0, 951.80754784,
    0.0, 877.90862779, 544.96955520,
    0.0, 0.0, 1.0
};

const cv::Vec4d kMockDistortionCoefficients{
    0.04717069,
    -0.04607454,
    0.16142146,
    -0.11573583
};

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

cv::Point robot_coordinate_to_pixel(const RobotCoordinate& coordinate, const ConverterConfig& config = {})
{
    std::array<cv::Point2d, 1> origin_pixel{config.origin_pixel};
    std::array<cv::Point2d, 1> origin_u{};

    cv::fisheye::undistortPoints(
        origin_pixel,
        origin_u,
        kMockCameraMatrix,
        kMockDistortionCoefficients,
        cv::Matx33d::eye()
    );

    const cv::Point2d detection_u{
        origin_u[0].x + (coordinate.y - config.robot_offset_y) / config.camera_height_m,
        origin_u[0].y + (coordinate.x - config.robot_offset_x) / config.camera_height_m
    };

    std::array<cv::Point2d, 1> undistorted{detection_u};
    std::array<cv::Point2d, 1> distorted{};

    cv::fisheye::distortPoints(
        undistorted,
        distorted,
        kMockCameraMatrix,
        kMockDistortionCoefficients
    );

    return {
        static_cast<int>(std::lround(distorted[0].x)),
        static_cast<int>(std::lround(distorted[0].y))
    };
}

}  // namespace

void cv_loop(std::atomic<bool>& running, BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch){

    // Open camera feed
    cv::VideoCapture camera_feed(0);

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

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

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
        const auto current_detections = postprocess_detection_centers(out, prep, frame.size());

        // Send detections on channel 
        if (!ch.send(std::make_pair(current_detections, timestamp))) {
            break;
        }
    }
}

void mock_cv_loop(std::atomic<bool>& running, BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch)
{
    std::cout << "Starting mock detector" << std::endl;

    const auto start_time = std::chrono::steady_clock::now();
    auto next_frame_time = start_time;
    int frame_index = 0;

    while (running && frame_index < kMockFrameCount) {
        const auto timestamp = start_time + frame_index * kMockTimestampStep;
        const double elapsed_seconds = std::chrono::duration<double>(timestamp - start_time).count();

        std::vector<DetectionCenter> detections;
        std::cout << "Mock frame " << frame_index
                  << ": elapsed=" << elapsed_seconds << "s";

        for (int battery_index = 0; battery_index < kMockBatteryCount; ++battery_index) {
            const double battery_age_seconds =
                elapsed_seconds - battery_index * kMockSpawnIntervalSeconds;

            if (battery_age_seconds < 0.0 ||
                battery_age_seconds > kMockVisibleDurationSeconds) {
                continue;
            }

            const RobotCoordinate mock_coordinate{
                kMockStartX + kMockConveyorSpeed * battery_age_seconds,
                kMockY
            };

            detections.push_back(DetectionCenter{
                robot_coordinate_to_pixel(mock_coordinate),
                "mock_battery",
                0.99F
            });

            std::cout << " | b" << battery_index
                      << " x=" << mock_coordinate.x
                      << ", y=" << mock_coordinate.y;
        }

        std::cout << "\n";

        if (!ch.send(std::make_pair(std::move(detections), timestamp))) {
            break;
        }

        next_frame_time += kMockFramePeriod;
        ++frame_index;
        std::this_thread::sleep_until(next_frame_time);
    }

    std::cout << "Mock detector finished after " << frame_index << " frame(s)" << std::endl;
    running = false;
}
