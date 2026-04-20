cmake_minimum_required(VERSION 3.16)
project(CV_MODULE LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui videoio dnn)

add_executable(cv_module
    src/main.cpp
    src/image_processing.cpp
    src/capture_thread.cpp
    src/inference_thread.cpp
    src/queue_thread.cpp
)

target_include_directories(cv_module PRIVATE include)
target_link_libraries(cv_module PRIVATE ${OpenCV_LIBS})
target_compile_definitions(cv_module PRIVATE CV_MODULE_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
