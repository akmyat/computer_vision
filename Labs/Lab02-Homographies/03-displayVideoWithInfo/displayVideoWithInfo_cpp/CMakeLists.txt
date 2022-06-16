cmake_minimum_required(VERSION 3.22)
project(displayVideoWithInfo_cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(displayVideoWithInfo_cpp main.cpp)

find_package(OpenCV 4.5.5 REQUIRED)
target_include_directories(displayVideoWithInfo_cpp PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(displayVideoWithInfo_cpp ${OpenCV_LIBS})