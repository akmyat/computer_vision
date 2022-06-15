cmake_minimum_required(VERSION 3.22)
project(showCircle)

set(CMAKE_CXX_STANDARD 14)

add_executable(showCircle main.cpp)

find_package(OpenCV 4.5.5 REQUIRED)
target_include_directories(showCircle PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(showCircle ${OpenCV_LIBS})
