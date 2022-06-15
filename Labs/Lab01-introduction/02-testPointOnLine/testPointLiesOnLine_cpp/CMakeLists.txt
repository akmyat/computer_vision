cmake_minimum_required(VERSION 3.22)
project(testPointLiesOnLine)

set(CMAKE_CXX_STANDARD 14)

add_executable(testPointLiesOnLine main.cpp matplotlib-cpp.h)

find_package(OpenCV 4.5.5 REQUIRED)
target_include_directories(testPointLiesOnLine PUBLIC ${OpenCV_INCLUDE_DIRS})

find_package(Python3 COMPONENTS Interpreter Development NumPy)
include_directories(${Python3_INCLUDE_DIRS})

target_link_libraries(testPointLiesOnLine ${OpenCV_LIBS} ${Python3_LIBRARIES})
