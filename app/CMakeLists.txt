add_executable(run_decode run_decode.cpp)
target_link_libraries(run_decode PRIVATE ovbc)

add_executable(run_infer run_infer.cpp)
target_link_libraries(run_infer PRIVATE ovbc)
