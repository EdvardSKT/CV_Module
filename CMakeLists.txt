cmake_minimum_required(VERSION 3.16)
project(CV_MODULE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn calib3d)

add_executable(cv_module
    src/main.cpp
    src/threads/computer_vision_thread.cpp
    src/threads/video_file_computer_vision_thread.cpp
    src/threads/mock_computer_vision_thread.cpp
    src/threads/queue_thread.cpp
    src/utilities/coordinate_converter.cpp
    src/utilities/image_processing.cpp
)

target_include_directories(cv_module PRIVATE include)
target_link_libraries(cv_module PRIVATE ${OpenCV_LIBS})
target_compile_definitions(cv_module PRIVATE CV_MODULE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(real_robot_main
    src/real_robot_main.cpp
    src/threads/computer_vision_thread.cpp
    src/threads/video_file_computer_vision_thread.cpp
    src/threads/queue_thread.cpp
    src/threads/robot_thread.cpp
    src/utilities/coordinate_converter.cpp
    src/utilities/image_processing.cpp
)

target_include_directories(real_robot_main PRIVATE include)
target_link_libraries(real_robot_main PRIVATE
    ${OpenCV_LIBS}
    ${CMAKE_SOURCE_DIR}/include/libYMConnect.so
)
target_compile_definitions(real_robot_main PRIVATE CV_MODULE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

add_executable(fake_robot_demo
    src/fake_robot_main.cpp
    src/threads/mock_computer_vision_thread.cpp
    src/threads/queue_thread.cpp
    src/threads/robot_thread.cpp
    src/utilities/coordinate_converter.cpp
    src/utilities/image_processing.cpp
)

target_include_directories(fake_robot_demo PRIVATE include)
target_link_libraries(fake_robot_demo PRIVATE
    ${OpenCV_LIBS}
    ${CMAKE_SOURCE_DIR}/include/libYMConnect.so
)

add_executable(cuda_video_test
    src/cuda_video_test_main.cpp
    src/utilities/image_processing.cpp
)

target_include_directories(cuda_video_test PRIVATE include)
target_link_libraries(cuda_video_test PRIVATE ${OpenCV_LIBS})
target_compile_definitions(cuda_video_test PRIVATE CV_MODULE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
