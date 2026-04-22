cmake_minimum_required(VERSION 3.16)
project(CV_MODULE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn calib3d)

add_executable(cv_module
    src/main.cpp
    src/threads/computer_vision_thread.cpp
    src/threads/queue_thread.cpp
    src/threads/robot_thread.cpp
    src/utilities/coordinate_converter.cpp
    src/utilities/image_processing.cpp
)

target_include_directories(cv_module PRIVATE include)
target_link_libraries(cv_module PRIVATE
    ${OpenCV_LIBS}
    ${CMAKE_SOURCE_DIR}/include/libYMConnect.so
)
target_compile_definitions(cv_module PRIVATE CV_MODULE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
