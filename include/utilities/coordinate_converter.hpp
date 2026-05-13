#pragma once

#include <string>

#include <opencv2/core/types.hpp>

#include "utilities/image_processing.hpp"

struct RobotCoordinate {
    double x;
    double y;
    std::string battery_type = "";
};

struct ConverterConfig {
    double scale_x = 5.0;
    double scale_y = 0.005;
    double robot_offset_x = 1.307;
    double robot_offset_y = 0.8552;
    double camera_height_m = 0.39;
    cv::Point2d origin_pixel{518.0, 128.0};
};

enum class DistortionModel {
    Fisheye,
    BrownConrady,
    RealSenseNative
};

struct CameraCalibration {
    int width = 0;
    int height = 0;
    double fx = 874.62127813;
    double fy = 877.90862779;
    double cx = 951.80754784;
    double cy = 544.96955520;
    cv::Vec<double, 5> distortion{0.04717069, -0.04607454, 0.16142146, -0.11573583, 0.0};
    DistortionModel distortion_model = DistortionModel::Fisheye;
    int realsense_distortion_model = 0;
};

void set_camera_calibration(const CameraCalibration& calibration);

void vector_undistort_pixel_to_normalized(std::vector<cv::Point2f>& pixels);

RobotCoordinate convert(const cv::Point2d& detection_pixel, const ConverterConfig& config = {});

RobotCoordinate convert(const DetectionCenter& detection, const ConverterConfig& config = {});

std::vector<RobotCoordinate> convert_multiple(const std::vector<DetectionCenter>& detections, const ConverterConfig& config = {});

RobotCoordinate convert_with_homography(const DetectionCenter& detection, const cv::Mat& H);

std::vector<RobotCoordinate> convert_multiple_with_homography(const std::vector<DetectionCenter>& detections, const cv::Mat& H);
