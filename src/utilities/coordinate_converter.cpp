#include "utilities/coordinate_converter.hpp"

#include <array>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace {

const cv::Matx33d kCameraMatrix{
    874.62127813, 0.0, 951.80754784,
    0.0, 877.90862779, 544.96955520,
    0.0, 0.0, 1.0
};

const cv::Vec4d kDistortionCoefficients{
    0.04717069,
    -0.04607454,
    0.16142146,
    -0.11573583
};

cv::Point2d undistort_pixel_to_normalized(const cv::Point2d& pixel)
{
    std::array<cv::Point2d, 1> distorted{pixel};
    std::array<cv::Point2d, 1> undistorted{};

    cv::fisheye::undistortPoints(
        distorted,
        undistorted,
        kCameraMatrix,
        kDistortionCoefficients,
        cv::Matx33d::eye()
    );

    return undistorted[0];
}

}  // namespace

RobotCoordinate convert(const cv::Point2d& detection_pixel, const ConverterConfig& config)
{
    const cv::Point2d origin_u = undistort_pixel_to_normalized(config.origin_pixel);
    const cv::Point2d detection_u = undistort_pixel_to_normalized(detection_pixel);

    // Mirrors converter_node.py: x/y are intentionally swapped when mapping to robot axes.
    return {
        (detection_u.y - origin_u.y) * config.camera_height_m + config.robot_offset_x,
        (detection_u.x - origin_u.x) * config.camera_height_m + config.robot_offset_y
    };
}

RobotCoordinate convert(const DetectionCenter& detection, const ConverterConfig& config)
{
    return convert(
        cv::Point2d{
            static_cast<double>(detection.center.y),
            static_cast<double>(detection.center.x)
        },
        config
    );
}

std::vector<RobotCoordinate> convert_multiple(const std::vector<DetectionCenter>& detections, const ConverterConfig& config)
{
    std::vector<RobotCoordinate> converted;
    converted.reserve(detections.size());

    for(auto det : detections){
        converted.push_back(convert(det));
    }

    return converted;
}
