#include "utilities/image_processing.hpp"
#include "utilities/coordinate_converter.hpp"

#include "opencv2/core/persistence.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/opencv.hpp"

#include <librealsense2/rs.hpp>

#include <cstring>
#include <exception>
#include <iostream>
#include <stdexcept>
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
    std::cout << "Usage: " << program << " [--output homography_calibration.yml]\n";
}

CameraCalibration camera_calibration_from_realsense(const rs2_intrinsics& intrinsics)
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
    return calibration;
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

cv::Mat capture_realsense_color_frame(rs2::pipeline& camera_pipeline)
{
    cv::Mat frame;

    for (int i = 0; i < 20; ++i) {
        const rs2::frameset frames = camera_pipeline.wait_for_frames();
        const rs2::video_frame color_frame = frames.get_color_frame();
        if (!color_frame) {
            continue;
        }

        cv::Mat color(
            cv::Size(color_frame.get_width(), color_frame.get_height()),
            CV_8UC3,
            const_cast<void*>(color_frame.get_data()),
            cv::Mat::AUTO_STEP
        );
        frame = color.clone();
    }

    if (frame.empty()) {
        throw std::runtime_error("Could not capture RealSense color frame");
    }

    return frame;
}

void print_realsense_intrinsics(const rs2_intrinsics& intrinsics)
{
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

}

int main(int argc, char* argv[])
{
    std::string output_path = "homography_calibration.yml";

    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
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

    rs2::pipeline camera_pipeline;
    rs2_intrinsics color_intrinsics{};
    try {
        const rs2::pipeline_profile profile = start_realsense_color_pipeline(camera_pipeline);
        const rs2::video_stream_profile color_profile =
            profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
        color_intrinsics = color_profile.get_intrinsics();
        set_camera_calibration(camera_calibration_from_realsense(color_intrinsics));
        print_realsense_intrinsics(color_intrinsics);
    } catch (const rs2::error& e) {
        std::cerr << "Could not start RealSense camera: " << e.what() << "\n";
        return 1;
    }

    std::vector<cv::Point2f> imagePoints;
    try {
        const cv::Mat calibration_frame = capture_realsense_color_frame(camera_pipeline);
        get_calibration_points_from_image(imagePoints, static_cast<int>(turntablePoints.size()), calibration_frame);
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

    fs << "camera_device" << "realsense";
    fs << "camera_width" << color_intrinsics.width;
    fs << "camera_height" << color_intrinsics.height;
    fs << "camera_fx" << color_intrinsics.fx;
    fs << "camera_fy" << color_intrinsics.fy;
    fs << "camera_ppx" << color_intrinsics.ppx;
    fs << "camera_ppy" << color_intrinsics.ppy;
    fs << "camera_distortion_model" << static_cast<int>(color_intrinsics.model);
    fs << "camera_distortion_coefficients" << std::vector<float>{
        color_intrinsics.coeffs[0],
        color_intrinsics.coeffs[1],
        color_intrinsics.coeffs[2],
        color_intrinsics.coeffs[3],
        color_intrinsics.coeffs[4]
    };
    fs << "image_points" << imagePoints;
    fs << "turntable_points" << turntablePoints;
    fs << "homography" << H;
    fs.release();

    std::vector<cv::Point2f> testPoint;
    try {
        const cv::Mat test_frame = capture_realsense_color_frame(camera_pipeline);
        get_calibration_points_from_image(testPoint, 1, test_frame);
    } catch (const std::exception& e) {
        std::cerr << "Test point capture failed: " << e.what() << "\n";
        return 1;
    }
    vector_undistort_pixel_to_normalized(testPoint);

    std::vector<cv::Point2f> result;

    cv::perspectiveTransform(testPoint, result, H);

    std::cout << "Table coordinate: "
          << result[0].x << ", "
          << result[0].y << std::endl;

    return 0;
}
