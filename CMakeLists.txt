cmake_minimum_required(VERSION 3.22)
project(OpenVINO-Benchmark-Cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(argparse CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(OpenVINO REQUIRED)
find_package(OpenCV REQUIRED)


add_executable(run_decode run_decode.cpp)
target_link_libraries(run_decode PRIVATE
        argparse::argparse
        spdlog::spdlog
        ${OpenCV_LIBS}
)


file(GLOB INFER_SOURCES infer/*.cpp)
file(GLOB INFER_HEADERS infer/*.hpp)
add_executable(run_infer run_infer.cpp ${INFER_SOURCES} ${INFER_HEADERS})
target_link_libraries(run_infer PRIVATE
        argparse::argparse
        spdlog::spdlog
        openvino::runtime
        ${OpenCV_LIBS}
)
