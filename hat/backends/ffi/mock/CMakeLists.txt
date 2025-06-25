cmake_minimum_required(VERSION 3.22.1)
project(opencl_backend)

set(CMAKE_CXX_STANDARD 14)

if ("${MOCK_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (MOCK_BACKEND "${CMAKE_SOURCE_DIR}")
    message("MOCK_BACKEND=${MOCK_BACKEND}")
endif()

if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/../shared")
    message("SHARED_BACKEND=${SHARED_BACKEND}")
endif()


message("MOCK")

include_directories(
    ${SHARED_BACKEND}/src/main/native/include
)
link_directories(
    ${CMAKE_BINARY_DIR}
)

add_library(mock_backend SHARED
    ${MOCK_BACKEND}/src/main/native/cpp/mock_backend.cpp
)
target_link_libraries(mock_backend
        shared_backend
)

add_executable(mock_info
    ${MOCK_BACKEND}/src/main/native/cpp/info.cpp
)


target_link_libraries(mock_info
    mock_backend
)








