cmake_minimum_required(VERSION 3.22)
project(readVideo)

set(CMAKE_CXX_STANDARD 14)

add_executable(readVideo main.cpp)

find_package(OpenCV 4.5.5 REQUIRED)
target_include_directories(readVideo PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(readVideo ${OpenCV_LIBS})
