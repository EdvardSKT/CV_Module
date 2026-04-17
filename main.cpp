#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

#include <atomic>
#include <csignal>
#include <thread>

void draw_x(cv::Mat& img) {
    int cx = img.cols / 2;
    int cy = img.rows / 2;
    int size = 20; // half length of the X arms

    cv::Point p1(cx - size, cy - size);
    cv::Point p2(cx + size, cy + size);
    cv::Point p3(cx - size, cy + size);
    cv::Point p4(cx + size, cy - size);

    cv::line(img, p1, p2, cv::Scalar(0, 0, 255), 2); // red
    cv::line(img, p3, p4, cv::Scalar(0, 0, 255), 2);
}

struct SharedFrame {
    cv::Mat frame;
    std::mutex frame_mtx;
    uint64_t frame_id = 0;
};

std::atomic<bool> running{true};

void signal_handler(int) 
{
    running = false;
}

void capture_loop(SharedFrame& sharedFrame, std::atomic<bool>& running)
{

    cv::VideoCapture camera_feed(0);

    if(!camera_feed.isOpened()) return;

    cv::Mat frame;

    while (running && camera_feed.read(frame)){
        {
            std::lock_guard<std::mutex> lock(sharedFrame.frame_mtx);
            sharedFrame.frame_id++;
            sharedFrame.frame = frame.clone();
        }
    }
}

void inference_loop(SharedFrame& sharedFrame, std::atomic<bool>& running, cv::Mat& display_frame) {
    uint64_t prevFrameId{0};

    cv::dnn::Net net = cv::dnn::readNetFromONNX("weights.onnx");
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

    while (running) {
        cv::Mat currFrame;
        uint64_t currFrameId{0};

        {
            std::lock_guard<std::mutex> lock(sharedFrame.frame_mtx);
            currFrameId = sharedFrame.frame_id;

            if (currFrameId != prevFrameId) {
                currFrame = sharedFrame.frame;
            }
        }

        if (currFrameId == prevFrameId) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }

        prevFrameId = currFrameId;

        if (currFrame.empty()) {
            continue;
        }

        cv::Mat blob;
        cv::dnn::blobFromImage(
            currFrame,
            blob,
            1.0 / 255.0,
            cv::Size(640, 640),
            cv::Scalar(),
            true,
            false
        );

        net.setInput(blob);
        cv::Mat out = net.forward();

        std::cout << "dims=" << out.dims << "\n";
        for (int i = 0; i < out.dims; ++i) {
            std::cout << out.size[i] << " ";
        }
        std::cout << "\n";
    }
}

int main() 
{
    SharedFrame sharedFrame;
    std::signal(SIGINT, signal_handler);
    cv::Mat display_frame;

    std::thread camera_thread(capture_loop, std::ref(sharedFrame), std::ref(running));
    std::thread inference_thread(inference_loop, std::ref(sharedFrame), std::ref(running), std::ref(display_frame));

    while(running){
        if(display_frame.empty()){
            continue;
        }
        cv::imshow("output", display_frame);
        cv::waitKey(25);
    }

    camera_thread.join();
    inference_thread.join();

    return 0;
}