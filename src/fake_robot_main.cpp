#include "threads/mock_computer_vision_thread.hpp"
#include "threads/queue_thread.hpp"
#include "threads/robot_thread.hpp"

#include <atomic>
#include <chrono>
#include <fcntl.h>
#include <iostream>
#include <termios.h>
#include <thread>
#include <unistd.h>

int main()
{
    BoundedChannel<std::pair<std::vector<DetectionCenter>, std::chrono::steady_clock::time_point>> detection_ch(10);
    BoundedChannel<double> battery_y_offset_ch(10);
    std::atomic<bool> running{true};

    const char* port = "/dev/ttyUSB0";

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

    FakeRobotController fake_robot;

    auto robot_loop_with_controller = static_cast<void (*)(
        BoundedChannel<double>&,
        std::atomic<bool>&,
        RobotController&,
        const RobotLoopConfig&)>(robot_loop);

    std::thread detector_thread(mock_cv_loop, std::ref(running), std::ref(detection_ch));
    std::thread queue_thread(
        queue_loop,
        std::ref(running),
        std::ref(detection_ch),
        std::ref(fd),
        &battery_y_offset_ch
    );
    std::thread robot_thread(
        robot_loop_with_controller,
        std::ref(battery_y_offset_ch),
        std::ref(running),
        std::ref(fake_robot),
        RobotLoopConfig{}
    );

    while (running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(25));
    }

    detection_ch.close();
    battery_y_offset_ch.close();

    detector_thread.join();
    queue_thread.join();
    robot_thread.join();

    close(fd);

    const auto writes = fake_robot.written_positions();
    std::cout << "[fake_robot] total writes captured: " << writes.size() << std::endl;

    if (!writes.empty()) {
        const auto& axis_data = writes.back().positionData.axisData;
        std::cout << "[fake_robot] last y offset: "
                  << axis_data.at(AxisIndex::CartesianAxis::Y) << std::endl;
    }

    return writes.empty() ? 1 : 0;
}
