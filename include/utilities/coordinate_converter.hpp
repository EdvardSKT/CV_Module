#pragma once

#include <string>

#include <opencv2/core/types.hpp>

#include "utilities/image_processing.hpp"

struct RobotCoordinate {
    double x;
    double y;
};

struct ConverterConfig {
    double scale_x = 5.0;
    double scale_y = 0.005;
    double robot_offset_x = 1.307;
    double robot_offset_y = 0.8552;
    double camera_height_m = 0.39;
    cv::Point2d origin_pixel{518.0, 128.0};
};

RobotCoordinate convert(const cv::Point2d& detection_pixel, const ConverterConfig& config = {});

RobotCoordinate convert(const DetectionCenter& detection, const ConverterConfig& config = {});

std::vector<RobotCoordinate> convert_multiple(const std::vector<DetectionCenter>& detections, const ConverterConfig& config = {});
