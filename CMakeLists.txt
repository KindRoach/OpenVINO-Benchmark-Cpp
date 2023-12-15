cmake_minimum_required(VERSION 3.22)
project(OpenVINO-Benchmark-Cpp)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")

include(FetchContent)

FetchContent_Declare(spdlog GIT_REPOSITORY https://github.com/gabime/spdlog.git GIT_TAG v1.12.0)
FetchContent_Declare(argparse GIT_REPOSITORY https://github.com/p-ranav/argparse.git GIT_TAG v3.0)
FetchContent_MakeAvailable(argparse spdlog)

if (MSVC)
    list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/lib/opencv/build")
    list(APPEND CMAKE_PREFIX_PATH "${PROJECT_SOURCE_DIR}/lib/openvino/runtime")
endif ()

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

add_executable(run_decode run_decode.cpp)
target_link_libraries(run_decode PRIVATE spdlog argparse ${OpenCV_LIBS})

file(GLOB INFER_SOURCES infer/*.cpp)
file(GLOB INFER_HEADERS infer/*.hpp)

add_executable(run_infer run_infer.cpp ${INFER_SOURCES} ${INFER_HEADERS})
target_link_libraries(run_infer PRIVATE spdlog argparse ${OpenCV_LIBS} openvino::runtime)
