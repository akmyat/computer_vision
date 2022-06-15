cmake_minimum_required(VERSION 3.22)
project(denseOpticalFlow)

set(CMAKE_CXX_STANDARD 14)

add_executable(denseOpticalFlow main.cpp)

find_package(OpenCV 4.5.5 REQUIRED)
target_include_directories(denseOpticalFlow PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(denseOpticalFlow ${OpenCV_LIBS})
