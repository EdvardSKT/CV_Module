#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <atomic>
#include <chrono>
#include <csignal>
#include <mutex>
#include <thread>
#include <fcntl.h>
#include <unistd.h>
#include <termios.h>
#include <iostream>
#include <string>

#include "capture_thread.hpp"
#include "inference_thread.hpp"
#include "queue_thread.hpp"
#include "shared_types.hpp"
#include "BoundedChannel.hpp"


namespace {

std::atomic<bool> running{true};

void signal_handler(int) {
    running = false;
}

}  // namespace

int main() {
    SharedFrame camera_frame;
    SharedFrame display_frame;

    BoundedChannel<std::vector<DetectionCenter>> ch(10);

    std::signal(SIGINT, signal_handler);

    // Open serial communication
    const char* port = "/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0";

    int fd = open(port, O_RDWR | O_NOCTTY | O_SYNC);
    if (fd < 0) {
        perror("open");
        return 1;
    }

    // Configure serial
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

    tcsetattr(fd, TCSANOW, &tty);

    // Start worker threads

    std::thread camera_thread(capture_loop, std::ref(camera_frame), std::ref(running));
    std::thread inference_thread(inference_loop, std::ref(camera_frame), std::ref(display_frame), std::ref(running));
    std::thread queue_thread(queue_loop, std::ref(running), std::ref(ch), std::ref(fd));

    while (running) {
        cv::Mat frame_to_show;
        {
            std::lock_guard<std::mutex> lock(display_frame.mutex);
            if (!display_frame.frame.empty()) {
                frame_to_show = display_frame.frame.clone();
            }
        }

        if (frame_to_show.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        cv::imshow("output", frame_to_show);
        if (cv::waitKey(25) == 27) {
            running = false;
        }
    }

    camera_thread.join();
    inference_thread.join();
    return 0;
}
