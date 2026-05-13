#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <iostream>
#include <mutex>
#include <string>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "threads/computer_vision_thread.hpp"
#include "threads/queue_thread.hpp"
#include "threads/robot_thread.hpp"
#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"
#include "utilities/shared_types.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>

namespace {

std::atomic<bool> running{true};
constexpr double kDefaultRecordingFps = 30.0;
const char* kDefaultRecordingDirectory = "videos";
const char* kDefaultRecordingPrefix = "stationary_batteries_detections";

void signal_handler(int)
{
    running = false;
}

std::filesystem::path make_default_recording_path()
{
    const auto now = std::chrono::system_clock::now();
    const auto seconds = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
    return std::filesystem::path(kDefaultRecordingDirectory) /
           (std::string(kDefaultRecordingPrefix) + "_" + std::to_string(seconds) + ".avi");
}

std::filesystem::path make_unique_recording_path(const std::filesystem::path& requested_path)
{
    if (!std::filesystem::exists(requested_path)) {
        return requested_path;
    }

    const std::filesystem::path directory = requested_path.parent_path();
    const std::string stem = requested_path.stem().string();
    const std::string extension = requested_path.extension().string();

    for (int suffix = 1; suffix < 10000; ++suffix) {
        const std::filesystem::path candidate = directory / (stem + "_" + std::to_string(suffix) + extension);
        if (!std::filesystem::exists(candidate)) {
            return candidate;
        }
    }

    return directory / (stem + "_fallback_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + extension);
}

}  // namespace

int main(int argc, char* argv[])
{
    std::filesystem::path recording_path = make_default_recording_path();
    double recording_fps = kDefaultRecordingFps;

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--record-output") == 0 && i + 1 < argc) {
            recording_path = argv[++i];
        } else if (std::strcmp(argv[i], "--record-fps") == 0 && i + 1 < argc) {
            try {
                recording_fps = std::stod(argv[++i]);
            } catch (const std::exception&) {
                std::cerr << "Invalid --record-fps value, using " << kDefaultRecordingFps << " fps\n";
                recording_fps = kDefaultRecordingFps;
            }
        } else {
            std::cerr << "Unknown argument: " << argv[i] << "\n"
                      << "Usage: " << argv[0] << " [--record-output path] [--record-fps fps]\n";
        }
    }

    recording_path = make_unique_recording_path(recording_path);
    const std::filesystem::path pick_time_csv_path =
        recording_path.parent_path() / (recording_path.stem().string() + "_pick_times.csv");

    std::error_code create_directory_error;
    const std::filesystem::path recording_directory = recording_path.parent_path();
    if (!recording_directory.empty()) {
        std::filesystem::create_directories(recording_directory, create_directory_error);
        if (create_directory_error) {
            std::cerr << "Failed to create recording directory " << recording_directory
                      << ": " << create_directory_error.message() << "\n";
        }
    }

    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>> detection_ch(10);
    BoundedChannel<RobotCoordinate> position_ch(10);
    SharedFrame display_frame;

    std::signal(SIGINT, signal_handler);

    std::thread detector_thread(cv_loop, std::ref(running), std::ref(detection_ch), &display_frame);
    std::thread queue_thread(
        stationary_batteries_queue_loop,
        std::ref(running),
        std::ref(detection_ch),
        &position_ch
    );
    std::thread robot_thread(robot_loop_2, std::ref(position_ch), std::ref(running), pick_time_csv_path.string());

    uint64_t last_displayed_frame_id = 0;
    bool display_window_created = false;
    cv::VideoWriter detection_video;
    bool recording_disabled = recording_fps <= 0.0;
    while (running) {
        cv::Mat frame;
        std::vector<Detection> detections;
        {
            std::lock_guard<std::mutex> lock(display_frame.mutex);
            if (display_frame.frame_id != last_displayed_frame_id) {
                display_frame.frame.copyTo(frame);
                detections = display_frame.detections;
                last_displayed_frame_id = display_frame.frame_id;
            }
        }

        if (!frame.empty()) {
            draw_detections(frame, detections);

            if (!recording_disabled && !detection_video.isOpened()) {
                if (create_directory_error) {
                    recording_disabled = true;
                }

                if (!recording_disabled) {
                    const int fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
                    detection_video.open(recording_path.string(), fourcc, recording_fps, frame.size(), frame.channels() == 3);
                    if (detection_video.isOpened()) {
                        std::cout << "Recording detection video to " << recording_path
                                  << " at " << recording_fps << " fps\n";
                    } else {
                        std::cerr << "Failed to open detection video writer for " << recording_path << "\n";
                        recording_disabled = true;
                    }
                }
            }

            if (detection_video.isOpened()) {
                detection_video.write(frame);
            }

            cv::imshow("Computer Vision", frame);
            display_window_created = true;
            if (cv::waitKey(1) == 27) {
                running = false;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    if (display_window_created) {
        cv::destroyWindow("Computer Vision");
    }
    detection_video.release();

    detection_ch.close();
    position_ch.close();

    detector_thread.join();
    queue_thread.join();
    robot_thread.join();

    return 0;
}
