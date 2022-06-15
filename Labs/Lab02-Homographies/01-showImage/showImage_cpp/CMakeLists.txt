cmake_minimum_required(VERSION 3.22)
project(showImage_cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(showImage_cpp main.cpp)

find_package(OpenCV 4.5.5 REQUIRED)
target_include_directories(showImage_cpp PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(showImage_cpp ${OpenCV_LIBS})