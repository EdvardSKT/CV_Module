#include "image_processing.hpp"
#include "BoundedChannel.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <atomic>
#include <iostream>
#include <vector>

#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

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

// DETECTION QUEUE RELATED CONSTANTS AND HELPER FUNCTIONS

const float CONVEYOR_SPEED = 0.3;
const float Y_DISTANCE_THRESHOLD = 0.001; // TODO: FIND REALISTIC VALUE
const float X_TRAVEL_UNCERTAINTY = 0.001; // TODO: FIND REALISTIC VALUE
const float HOME_X_POSITION = 0.5;


std::vector<DetectionCenter> find_new_detections(std::vector<DetectionCenter> incoming_detections, std::vector<DetectionCenter> existing_detections){

    std::vector<DetectionCenter> new_detections;

    for(auto new_det : incoming_detections){
        for(auto old_det : existing_detections){

            double traveled_time = 1; // TODO: IMPLEMENT TIME FOR EACH FRAME AND USE IT HERE 
            double predicted_x = old_det.center.x + CONVEYOR_SPEED*traveled_time;

            double distance_from_predicted_x = (new_det.center.x - predicted_x);
            double distance_y = (new_det.center.y - old_det.center.y);

            if(std::fabs(distance_y) <= Y_DISTANCE_THRESHOLD && std::fabs(distance_from_predicted_x) <= X_TRAVEL_UNCERTAINTY){
                new_detections.push_back(new_det);
                break;
            }
        }
    }

    return new_detections;
};

float find_pulse_delay_ms(double along_track_position){
    float stamp = 0; // TODO: add stamp to coordinates
    return (HOME_X_POSITION - along_track_position)/CONVEYOR_SPEED * 1000 - stamp;
};

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

    // Create variable for registered detections
    std::vector<DetectionCenter> registered_detections{};

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
        std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point> msg{current_detections, timestamp};
        ch.send(msg);
    }
}