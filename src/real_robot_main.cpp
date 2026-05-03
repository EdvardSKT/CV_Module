#include <atomic>
#include <chrono>
#include <csignal>
#include <fcntl.h>
#include <iostream>
#include <mutex>
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

namespace {

std::atomic<bool> running{true};

void signal_handler(int)
{
    running = false;
}

}  // namespace

int main()
{
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>> detection_ch(10);
    BoundedChannel<double> battery_y_offset_ch(10);
    SharedFrame display_frame;

    std::signal(SIGINT, signal_handler);

    const char* port = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0";

    int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    struct termios tty{};
    tcgetattr(fd, &tty);

    cfsetospeed(&tty, B115200);
    cfsetispeed(&tty, B115200);

    tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;
    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~(PARENB | PARODD);
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CRTSCTS;

    tty.c_iflag = 0;
    tty.c_oflag = 0;
    tty.c_lflag = 0;
    tty.c_cc[VMIN] = 0;
    tty.c_cc[VTIME] = 5;

    if (tcsetattr(fd, TCSANOW, &tty) != 0) {
        perror("tcsetattr");
        close(fd);
        return 1;
    }

    tcflush(fd, TCIOFLUSH);
    std::cout << "Opened serial port " << port << " at 115200 baud\n";
    std::cout << "Waiting for controller serial boot/reset...\n";
    std::this_thread::sleep_for(std::chrono::seconds(2));

    auto real_robot_loop = static_cast<void (*)(
        BoundedChannel<double>&,
        std::atomic<bool>&)>(robot_loop);

    std::thread detector_thread(cv_loop, std::ref(running), std::ref(detection_ch), &display_frame);
    std::thread queue_thread(
        queue_loop,
        std::ref(running),
        std::ref(detection_ch),
        std::ref(fd),
        &battery_y_offset_ch
    );
    std::thread robot_thread(real_robot_loop, std::ref(battery_y_offset_ch), std::ref(running));

    uint64_t last_displayed_frame_id = 0;
    bool display_window_created = false;
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

    detection_ch.close();
    battery_y_offset_ch.close();

    detector_thread.join();
    queue_thread.join();
    robot_thread.join();

    close(fd);

    return 0;
}
