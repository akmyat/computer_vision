cmake_minimum_required(VERSION 3.22)
project(sfm)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

add_executable(sfm main.cpp)

find_package(Ceres QUIET)

set(OpenCV_DIR "" CACHE PATH "/mnt/ntfs/Data/code/CV/source/opencv/build")
find_package(OpenCV 4.5.5 REQUIRED COMPONENTS core calib3d features2d sfm viz)

find_package(Eigen3 REQUIRED)

set(OpenMVS_DIR "" CACHE PATH "/mnt/ntfs/Data/code/CV/source/openMVS/build")
find_package(OpenMVS REQUIRED)

find_package(Boost REQUIRED COMPONENTS filesystem graph)

include_directories(${EIGEN3_INCLUDE_DIR} ${OpenMVS_INCLUDE_DIRS} ${Boost_INCLUDE_DIR})

message(STATUS ${OpenCV_LIBRARIES} ${OpenMVS_LIBRARIES})

target_link_libraries(sfm
        ${OpenCV_LIBRARIES}
        ${Boost_LIBRARIES}
#        ${OpenMVS_LIBRARIES}
        )
