cmake_minimum_required(VERSION 3.22)
project(opticalFlowLucasKanade)

set(CMAKE_CXX_STANDARD 14)

add_executable(opticalFlowLucasKanade main.cpp)

find_package(OpenCV 4.5.5 REQUIRED)
target_include_directories(opticalFlowLucasKanade PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opticalFlowLucasKanade ${OpenCV_LIBS})