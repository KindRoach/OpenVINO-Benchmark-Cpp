cmake_minimum_required(VERSION 3.22)
project(OpenVINO_Benchmark_Cpp)

set(CMAKE_CXX_STANDARD 23)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV REQUIRED)

add_executable(run_decode run_decode.cpp)
target_include_directories(run_decode PRIVATE
        "lib/argparse/include"
        ${OpenCV_INCLUDE_DIRS}
        "lib/spdlog/include")
target_link_libraries(run_decode PRIVATE ${OpenCV_LIBS})
