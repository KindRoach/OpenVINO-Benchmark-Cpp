cmake_minimum_required(VERSION 3.22)
project(OpenVINO_Benchmark_Cpp)

set(CMAKE_CXX_STANDARD 23)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV REQUIRED)

add_executable(sync_decode opencv_benchmark/sync_decode.cpp)
target_include_directories(sync_decode PRIVATE
        "lib/argparse/include"
        ${OpenCV_INCLUDE_DIRS}
        "lib/spdlog/include")
target_link_libraries(sync_decode PRIVATE ${OpenCV_LIBS})


add_executable(async_decode opencv_benchmark/async_decode.cpp)
target_include_directories(async_decode PRIVATE
        "lib/argparse/include"
        ${OpenCV_INCLUDE_DIRS}
        "lib/spdlog/include")
target_link_libraries(async_decode PRIVATE ${OpenCV_LIBS})
