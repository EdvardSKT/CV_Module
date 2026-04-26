#include <atomic>
#include <chrono>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <iostream>
#include <termios.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#include "threads/computer_vision_thread.hpp"
#include "threads/mock_computer_vision_thread.hpp"
#include "threads/queue_thread.hpp"
#include "utilities/BoundedChannel.hpp"
#include "utilities/image_processing.hpp"


namespace {

std::atomic<bool> running{true};

void signal_handler(int) {
    running = false;
}

}  // namespace

int main(int argc, char* argv[]) {
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>> ch(10);

    bool use_mock_detector = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--mock-detector") == 0) {
            use_mock_detector = true;
        }
    }

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

    std::thread computer_vision_thread(
        use_mock_detector ? mock_cv_loop : cv_loop,
        std::ref(running),
        std::ref(ch)
    );
    std::thread queue_thread(
        queue_loop,
        std::ref(running),
        std::ref(ch),
        std::ref(fd),
        nullptr
    );

    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    ch.close();
    computer_vision_thread.join();
    queue_thread.join();
    close(fd);

    return 0;
}
