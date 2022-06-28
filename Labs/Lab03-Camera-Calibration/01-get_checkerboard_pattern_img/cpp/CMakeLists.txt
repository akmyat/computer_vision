cmake_minimum_required(VERSION 3.22)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

add_executable(cpp main.cpp)

find_package(OpenCV 4.5.5 REQUIRED)
target_link_directories(cpp PUBLIC ${OpenCV_INCLUDE_DIRES})
target_link_libraries(cpp ${OpenCV_LIBS})