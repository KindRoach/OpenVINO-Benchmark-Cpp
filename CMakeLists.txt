cmake_minimum_required(VERSION 3.22)
project(OpenVINO-Benchmark-Cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(run_decode run_decode.cpp)
target_include_directories(run_decode PRIVATE
        "lib/argparse/include"
        ${OpenCV_INCLUDE_DIRS}
        "lib/spdlog/include")
target_link_libraries(run_decode PRIVATE ${OpenCV_LIBS})

file(GLOB INFER_SOURCES infer/*.cpp)
file(GLOB INFER_HEADERS infer/*.hpp)

add_executable(run_infer run_infer.cpp ${INFER_SOURCES} ${INFER_HEADERS})
target_include_directories(run_infer PRIVATE
        "lib/argparse/include"
        ${OpenCV_INCLUDE_DIRS}
        "lib/spdlog/include")
target_link_libraries(run_infer PRIVATE ${OpenCV_LIBS} openvino::runtime)
