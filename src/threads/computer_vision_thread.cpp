#include "threads/computer_vision_thread.hpp"

#include "utilities/BoundedChannel.hpp"
#include "utilities/coordinate_converter.hpp"
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

#include <librealsense2/rs.hpp>

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

void set_camera_calibration_from_realsense(const rs2_intrinsics& intrinsics)
{
    CameraCalibration calibration{};
    calibration.width = intrinsics.width;
    calibration.height = intrinsics.height;
    calibration.fx = intrinsics.fx;
    calibration.fy = intrinsics.fy;
    calibration.cx = intrinsics.ppx;
    calibration.cy = intrinsics.ppy;
    calibration.distortion = {
        intrinsics.coeffs[0],
        intrinsics.coeffs[1],
        intrinsics.coeffs[2],
        intrinsics.coeffs[3],
        intrinsics.coeffs[4]
    };
    calibration.distortion_model = DistortionModel::RealSenseNative;
    calibration.realsense_distortion_model = static_cast<int>(intrinsics.model);

    set_camera_calibration(calibration);

    std::cout << "RealSense color intrinsics: "
              << "width=" << intrinsics.width
              << ", height=" << intrinsics.height
              << ", fx=" << intrinsics.fx
              << ", fy=" << intrinsics.fy
              << ", ppx=" << intrinsics.ppx
              << ", ppy=" << intrinsics.ppy
              << ", distortion_model=" << intrinsics.model
              << "\n";
}

rs2::pipeline_profile start_realsense_color_pipeline(rs2::pipeline& camera_pipeline)
{
    try {
        rs2::config camera_config;
        camera_config.enable_stream(RS2_STREAM_COLOR, 1280, 720, RS2_FORMAT_BGR8, 30);
        return camera_pipeline.start(camera_config);
    } catch (const rs2::error& e) {
        std::cerr << "Could not start preferred RealSense color stream: " << e.what()
                  << "\nFalling back to default BGR color stream\n";
    }

    rs2::config fallback_config;
    fallback_config.enable_stream(RS2_STREAM_COLOR, RS2_FORMAT_BGR8);
    return camera_pipeline.start(fallback_config);
}

}  // namespace

void cv_loop(
    std::atomic<bool>& running,
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>>& ch,
    SharedFrame* shared_frame
){

    rs2::pipeline camera_pipeline;
    try {
        const rs2::pipeline_profile profile = start_realsense_color_pipeline(camera_pipeline);
        const rs2::video_stream_profile color_profile =
            profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        set_camera_calibration_from_realsense(color_profile.get_intrinsics());
    } catch (const rs2::error& e) {
        std::cerr << "Could not start RealSense camera: " << e.what() << "\n";
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

    while (running) {
        rs2::frameset frames;
        try {
            frames = camera_pipeline.wait_for_frames();
        } catch (const rs2::error& e) {
            std::cerr << "Failed to read RealSense frame: " << e.what() << "\n";
            running = false;
            return;
        }

        const rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) {
            continue;
        }

        cv::Mat frame(
            cv::Size(color_frame.get_width(), color_frame.get_height()),
            CV_8UC3,
            const_cast<void*>(color_frame.get_data()),
            cv::Mat::AUTO_STEP
        );

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
