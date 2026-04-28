#include "utilities/image_processing.hpp"

#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/core/cuda.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

namespace {

constexpr int kInputWidth = 640;
constexpr int kInputHeight = 640;

std::filesystem::path resolve_model_path(const char* explicit_path) {
    if (explicit_path != nullptr) {
        const std::filesystem::path path = explicit_path;
        if (std::filesystem::exists(path)) {
            return path;
        }
        throw std::runtime_error("Model path does not exist: " + path.string());
    }

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

void configure_backend(cv::dnn::Net& net) {
    try {
        const int cuda_devices = cv::cuda::getCudaEnabledDeviceCount();
        if (cuda_devices > 0) {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
            std::cout << "Using CUDA for OpenCV DNN (" << cuda_devices << " CUDA device(s) visible)\n";
            return;
        }
    } catch (const cv::Exception& e) {
        std::cerr << "CUDA device check failed, using CPU: " << e.what() << "\n";
    }

    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    std::cout << "Using CPU for OpenCV DNN\n";
}

void print_usage(const char* program) {
    std::cerr << "Usage: " << program << " <video.mp4> [model.onnx] [--no-display]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const std::string video_path = argv[1];
    const bool no_display =
        (argc >= 3 && std::string(argv[2]) == "--no-display") ||
        (argc >= 4 && std::string(argv[3]) == "--no-display");
    const char* model_arg = nullptr;
    if (argc >= 3 && std::string(argv[2]) != "--no-display") {
        model_arg = argv[2];
    }

    cv::VideoCapture video(video_path);
    if (!video.isOpened()) {
        std::cerr << "Could not open video file: " << video_path << "\n";
        return 1;
    }

    cv::dnn::Net net;
    try {
        const auto model_path = resolve_model_path(model_arg);
        std::cout << "Loading model: " << model_path << "\n";
        net = cv::dnn::readNetFromONNX(model_path.string());
    } catch (const std::exception& e) {
        std::cerr << "Failed to load ONNX model: " << e.what() << "\n";
        return 1;
    }

    configure_backend(net);

    const double fps = video.get(cv::CAP_PROP_FPS);
    const int delay_ms = fps > 0.0 ? std::max(1, static_cast<int>(1000.0 / fps)) : 1;

    cv::Mat frame;
    int frame_index = 0;
    while (video.read(frame)) {
        ++frame_index;

        PreprocessResult prep;
        try {
            prep = preprocessYOLO(frame, kInputWidth, kInputHeight);
        } catch (const std::exception& e) {
            std::cerr << "Frame " << frame_index << ": preprocessing failed: " << e.what() << "\n";
            continue;
        }

        int blob_sizes[] = {1, 3, kInputHeight, kInputWidth};
        cv::Mat blob(4, blob_sizes, CV_32F, prep.tensor.data());

        cv::Mat out;
        const auto inference_start = std::chrono::steady_clock::now();
        try {
            net.setInput(blob);
            out = net.forward();
        } catch (const cv::Exception& e) {
            std::cerr << "Frame " << frame_index << ": forward pass failed: " << e.what() << "\n";
            return 1;
        }
        const auto inference_end = std::chrono::steady_clock::now();

        const auto detections = decode_detections(out, prep, frame.size());
        const auto centers = detection_centers_from_detections(detections);
        const double inference_ms = std::chrono::duration<double, std::milli>(inference_end - inference_start).count();

        std::cout << "Frame " << frame_index << ": " << detections.size()
                  << " detection(s), inference " << cv::format("%.2f", inference_ms) << " ms\n";
        for (const auto& center : centers) {
            std::cout << "  " << center.battery_type
                      << " center=(" << center.center.x << ", " << center.center.y << ")"
                      << " confidence=" << cv::format("%.2f", center.confidence) << "\n";
        }

        if (!no_display) {
            draw_detections(frame, detections);
            cv::imshow("CUDA video test", frame);
            const int key = cv::waitKey(delay_ms);
            if (key == 27 || key == 'q') {
                break;
            }
        }
    }

    if (!no_display) {
        cv::destroyWindow("CUDA video test");
    }

    return 0;
}
