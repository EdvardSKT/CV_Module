#include "utilities/image_processing.hpp"
#include "utilities/coordinate_converter.hpp"

#include "opencv2/core/persistence.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/opencv.hpp"

#include <cstring>
#include <exception>
#include <iostream>
#include <string>
#include <vector>


namespace 
{

const std::vector<cv::Point2f> turntablePoints = {
    cv::Point2f{1.0294f, -0.1723f},
    cv::Point2f{0.8526f, 0.0009f},
    cv::Point2f{1.0237f, 0.1791f},
    cv::Point2f{1.2012f, -0.0065f},
    cv::Point2f{1.0671f, 0.0047f},
    cv::Point2f{0.9992f, -0.0238f}
};

void print_usage(const char* program)
{
    std::cout << "Usage: " << program << " [--camera /dev/video42] [--output homography_calibration.yml]\n";
}

}

int main(int argc, char* argv[])
{
    std::string camera_device = "/dev/video42";
    std::string output_path = "homography_calibration.yml";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--camera") == 0 && i + 1 < argc) {
            camera_device = argv[++i];
        } else if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << argv[i] << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    std::vector<cv::Point2f> imagePoints;
    try {
        get_calibration_points(imagePoints, static_cast<int>(turntablePoints.size()), camera_device);
    } catch (const std::exception& e) {
        std::cerr << "Calibration capture failed: " << e.what() << "\n";
        return 1;
    }

    if (imagePoints.size() != turntablePoints.size()) {
        std::cerr << "Calibration cancelled after " << imagePoints.size()
                  << " points; expected " << turntablePoints.size() << ". Nothing saved.\n";
        return 1;
    }

    cv::FileStorage fs(output_path, cv::FileStorage::WRITE);
    if (!fs.isOpened()) {
        std::cerr << "Could not open output file: " << output_path << "\n";
        return 1;
    }
    
    std::cout << "Saved " << imagePoints.size() << " calibration point pairs to "
              << output_path << "\n";


    vector_undistort_pixel_to_normalized(imagePoints);
    cv::Mat H = cv::findHomography(imagePoints, turntablePoints);

    fs << "camera_device" << camera_device;
    fs << "image_points" << imagePoints;
    fs << "turntable_points" << turntablePoints;
    fs << "homography" << H;
    fs.release();

    std::vector<cv::Point2f> testPoint;
    get_calibration_points(testPoint, 1, camera_device);
    vector_undistort_pixel_to_normalized(testPoint);

    std::vector<cv::Point2f> result;

    cv::perspectiveTransform(testPoint, result, H);

    std::cout << "Table coordinate: "
          << result[0].x << ", "
          << result[0].y << std::endl;

    return 0;
}
