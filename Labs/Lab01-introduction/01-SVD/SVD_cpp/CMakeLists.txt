cmake_minimum_required(VERSION 3.22)
project(SVD)

set(CMAKE_CXX_STANDARD 14)

add_executable(SVD main.cpp)

find_package(OpenCV 4.5.5 REQUIRED)
target_include_directories(SVD PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(SVD ${OpenCV_LIBS})