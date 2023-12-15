file(GLOB OVBC_SOURCE *.cpp)
file(GLOB OVBC_HEADER *.hpp)

add_library(ovbc ${OVBC_SOURCE} ${OVBC_HEADER})
target_include_directories(ovbc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ovbc PUBLIC
        spdlog
        argparse
        openvino::runtime
        ${OpenCV_LIBS}
)
