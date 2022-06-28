cmake_minimum_required(VERSION 3.22)
project(calibrate_homography_using_opencv_cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(calibrate_homography_using_opencv_cpp main.cpp)

find_package(OpenCV 4.5.5 REQUIRED)
target_link_directories(calibrate_homography_using_opencv_cpp PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(calibrate_homography_using_opencv_cpp ${OpenCV_LIBS})