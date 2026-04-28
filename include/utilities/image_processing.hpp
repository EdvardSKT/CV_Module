#pragma once

#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

struct PreprocessResult {
    std::vector<float> tensor;  // CHW, size = 3 * input_w * input_h
    float scale;
    int pad_x;
    int pad_y;
};

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

struct DetectionCenter {
    cv::Point center;
    std::string battery_type;
    float confidence;
};

PreprocessResult preprocessYOLO(const cv::Mat& image, int input_w = 1280, int input_h = 1280);

std::vector<Detection> decode_detections(const cv::Mat& output, const PreprocessResult& prep, const cv::Size& image_size);

std::vector<DetectionCenter> detection_centers_from_detections(const std::vector<Detection>& detections);

std::vector<DetectionCenter> postprocess_detection_centers(
    const cv::Mat& output,
    const PreprocessResult& prep,
    const cv::Size& image_size
);

void draw_detections(cv::Mat& image, const std::vector<Detection>& detections);
