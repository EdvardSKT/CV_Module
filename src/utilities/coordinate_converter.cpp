#include "utilities/coordinate_converter.hpp"

#include <array>
#include <cmath>
#include <mutex>
#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include <librealsense2/rsutil.h>

namespace {

std::mutex calibration_mutex;
CameraCalibration camera_calibration;

CameraCalibration get_camera_calibration()
{
    std::lock_guard<std::mutex> lock(calibration_mutex);
    return camera_calibration;
}

rs2_intrinsics to_realsense_intrinsics(const CameraCalibration& calibration)
{
    rs2_intrinsics intrinsics{};
    intrinsics.width = calibration.width;
    intrinsics.height = calibration.height;
    intrinsics.ppx = static_cast<float>(calibration.cx);
    intrinsics.ppy = static_cast<float>(calibration.cy);
    intrinsics.fx = static_cast<float>(calibration.fx);
    intrinsics.fy = static_cast<float>(calibration.fy);
    intrinsics.model = static_cast<rs2_distortion>(calibration.realsense_distortion_model);
    for (int i = 0; i < 5; ++i) {
        intrinsics.coeffs[i] = static_cast<float>(calibration.distortion[i]);
    }

    return intrinsics;
}

cv::Point2d realsense_deproject_pixel_to_normalized(const rs2_intrinsics& intrinsics, const cv::Point2d& pixel)
{
    const float realsense_pixel[2]{
        static_cast<float>(pixel.x),
        static_cast<float>(pixel.y)
    };
    float point[3]{};
    rs2_deproject_pixel_to_point(point, &intrinsics, realsense_pixel, 1.0f);

    if (point[2] == 0.0f) {
        return cv::Point2d{point[0], point[1]};
    }

    return cv::Point2d{
        static_cast<double>(point[0] / point[2]),
        static_cast<double>(point[1] / point[2])
    };
}

cv::Point2d realsense_projective_pixel_to_normalized(const rs2_intrinsics& intrinsics, const cv::Point2d& pixel)
{
    cv::Point2d normalized{
        (pixel.x - static_cast<double>(intrinsics.ppx)) / static_cast<double>(intrinsics.fx),
        (pixel.y - static_cast<double>(intrinsics.ppy)) / static_cast<double>(intrinsics.fy)
    };

    constexpr double kEpsilon = 1e-4;
    for (int iteration = 0; iteration < 10; ++iteration) {
        const float point[3]{
            static_cast<float>(normalized.x),
            static_cast<float>(normalized.y),
            1.0f
        };
        float projected[2]{};
        rs2_project_point_to_pixel(projected, &intrinsics, point);

        const cv::Point2d error{
            static_cast<double>(projected[0]) - pixel.x,
            static_cast<double>(projected[1]) - pixel.y
        };
        if (std::abs(error.x) + std::abs(error.y) < 1e-3) {
            break;
        }

        const float point_dx[3]{
            static_cast<float>(normalized.x + kEpsilon),
            static_cast<float>(normalized.y),
            1.0f
        };
        const float point_dy[3]{
            static_cast<float>(normalized.x),
            static_cast<float>(normalized.y + kEpsilon),
            1.0f
        };
        float projected_dx[2]{};
        float projected_dy[2]{};
        rs2_project_point_to_pixel(projected_dx, &intrinsics, point_dx);
        rs2_project_point_to_pixel(projected_dy, &intrinsics, point_dy);

        const double j00 = (static_cast<double>(projected_dx[0]) - static_cast<double>(projected[0])) / kEpsilon;
        const double j10 = (static_cast<double>(projected_dx[1]) - static_cast<double>(projected[1])) / kEpsilon;
        const double j01 = (static_cast<double>(projected_dy[0]) - static_cast<double>(projected[0])) / kEpsilon;
        const double j11 = (static_cast<double>(projected_dy[1]) - static_cast<double>(projected[1])) / kEpsilon;
        const double determinant = j00 * j11 - j01 * j10;
        if (std::abs(determinant) < 1e-9) {
            break;
        }

        const double delta_x = (j11 * error.x - j01 * error.y) / determinant;
        const double delta_y = (-j10 * error.x + j00 * error.y) / determinant;
        normalized.x -= delta_x;
        normalized.y -= delta_y;

        if (std::abs(delta_x) + std::abs(delta_y) < 1e-8) {
            break;
        }
    }

    return normalized;
}

cv::Point2d realsense_pixel_to_normalized(const CameraCalibration& calibration, const cv::Point2d& pixel)
{
    const rs2_intrinsics intrinsics = to_realsense_intrinsics(calibration);
    if (intrinsics.model == RS2_DISTORTION_NONE || intrinsics.model == RS2_DISTORTION_INVERSE_BROWN_CONRADY) {
        return realsense_deproject_pixel_to_normalized(intrinsics, pixel);
    }

    return realsense_projective_pixel_to_normalized(intrinsics, pixel);
}

cv::Point2d undistort_pixel_to_normalized(const cv::Point2d& pixel)
{
    const CameraCalibration calibration = get_camera_calibration();
    if (calibration.distortion_model == DistortionModel::RealSenseNative) {
        return realsense_pixel_to_normalized(calibration, pixel);
    }

    const cv::Matx33d camera_matrix{
        calibration.fx, 0.0, calibration.cx,
        0.0, calibration.fy, calibration.cy,
        0.0, 0.0, 1.0
    };

    std::array<cv::Point2d, 1> distorted{pixel};
    std::array<cv::Point2d, 1> undistorted{};

    if (calibration.distortion_model == DistortionModel::Fisheye) {
        const cv::Vec4d distortion_coefficients{
            calibration.distortion[0],
            calibration.distortion[1],
            calibration.distortion[2],
            calibration.distortion[3]
        };
        cv::fisheye::undistortPoints(
            distorted,
            undistorted,
            camera_matrix,
            distortion_coefficients,
            cv::Matx33d::eye()
        );
    } else {
        cv::undistortPoints(
            distorted,
            undistorted,
            camera_matrix,
            calibration.distortion,
            cv::noArray(),
            cv::Matx33d::eye()
        );
    }

    return undistorted[0];
}

cv::Point2f undistort_pixel_to_normalized2(const cv::Point2f& pixel)
{
    const CameraCalibration calibration = get_camera_calibration();
    if (calibration.distortion_model == DistortionModel::RealSenseNative) {
        const cv::Point2d normalized = realsense_pixel_to_normalized(calibration, cv::Point2d(pixel));
        return cv::Point2f{
            static_cast<float>(normalized.x),
            static_cast<float>(normalized.y)
        };
    }

    const cv::Matx33d camera_matrix{
        calibration.fx, 0.0, calibration.cx,
        0.0, calibration.fy, calibration.cy,
        0.0, 0.0, 1.0
    };

    std::array<cv::Point2f, 1> distorted{pixel};
    std::array<cv::Point2f, 1> undistorted{};

    if (calibration.distortion_model == DistortionModel::Fisheye) {
        const cv::Vec4d distortion_coefficients{
            calibration.distortion[0],
            calibration.distortion[1],
            calibration.distortion[2],
            calibration.distortion[3]
        };
        cv::fisheye::undistortPoints(
            distorted,
            undistorted,
            camera_matrix,
            distortion_coefficients,
            cv::Matx33d::eye()
        );
    } else {
        cv::undistortPoints(
            distorted,
            undistorted,
            camera_matrix,
            calibration.distortion,
            cv::noArray(),
            cv::Matx33d::eye()
        );
    }

    return undistorted[0];
}

}  // namespace

void set_camera_calibration(const CameraCalibration& calibration)
{
    std::lock_guard<std::mutex> lock(calibration_mutex);
    camera_calibration = calibration;
}

void vector_undistort_pixel_to_normalized(std::vector<cv::Point2f>& pixels)
{
    for (auto& p : pixels) {
        const cv::Point2d undistorted =
            undistort_pixel_to_normalized(cv::Point2d(p));

        p = cv::Point2f(
            static_cast<float>(undistorted.x),
            static_cast<float>(undistorted.y)
        );
    }
}


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
            static_cast<double>(detection.center.x),
            static_cast<double>(detection.center.y)
        },
        config
    );
}

std::vector<RobotCoordinate> convert_multiple(const std::vector<DetectionCenter>& detections, const ConverterConfig& config)
{
    std::vector<RobotCoordinate> converted;
    converted.reserve(detections.size());

    for(auto det : detections){
        converted.push_back(convert(det, config));
    }

    return converted;
}

RobotCoordinate convert_with_homography(const DetectionCenter& detection, const cv::Mat& H)
{
    std::array<cv::Point2f, 1> undistorted{ 
        undistort_pixel_to_normalized2(
            cv::Point2f{
                static_cast<float>(detection.center.x),
                static_cast<float>(detection.center.y)
            }
        )
    };
    std::array<cv::Point2f, 1> converted;

    cv::perspectiveTransform(undistorted, converted, H);

    return RobotCoordinate{converted[0].x, converted[0].y, detection.battery_type};
}

std::vector<RobotCoordinate> convert_multiple_with_homography(const std::vector<DetectionCenter>& detections, const cv::Mat& H)
{
    std::vector<RobotCoordinate> converted;
    converted.reserve(detections.size());

    for(auto det : detections){
        converted.push_back(convert_with_homography(det, H));
    }

    return converted;
}
