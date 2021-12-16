cmake_minimum_required(VERSION 3.1)
project(hasift)

set(OpenCV_DIR thirdParty/opencv_3.4.13/share/OpenCV)
find_package(OpenCV 3 REQUIRED)

file(GLOB_RECURSE SRC_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=hidden -Wall -g -O3 -fPIC -std=c++11 -fopenmp")

include_directories(
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/src/hessaff      
)

include_directories(${INC_DIRS})

add_executable(hasift ${SRC_FILES})
target_link_libraries(hasift -Wl,--whole-archive ${OpenCV_LIBS} -Wl,--no-whole-archive)
