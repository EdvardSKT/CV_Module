#include "threads/mock_computer_vision_thread.hpp"

#include "utilities/BoundedChannel.hpp"
#include "utilities/coordinate_converter.hpp"
#include "utilities/image_processing.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <iostream>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace {

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

void mock_cv_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch
)
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
