cmake_minimum_required(VERSION 3.22.1)
project(spirv_backend)

set(CMAKE_CXX_STANDARD 14)

if ("${SPIRV_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (SPIRV_BACKEND "${CMAKE_SOURCE_DIR}")
    message("SPIRV_BACKEND=${SPIRV_BACKEND}")
endif()

if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/../shared")
    message("SHARED_BACKEND=${SHARED_BACKEND}")
endif()

message("SPIRV")

include_directories(
    ${SPIRV_BACKEND}/src/main/native/include
    ${SHARED_BACKEND}/src/main/native/include
    ${SPIRV_INCLUDE_DIR}
)
link_directories(
    ${CMAKE_BINARY_DIR}
)

add_library(spirv_backend SHARED
    ${SPIRV_BACKEND}/src/main/native/cpp/spirv_backend.cpp
)

add_executable(spirv_info
    ${SPIRV_BACKEND}/src/main/native/cpp/info.cpp
)

target_link_libraries(spirv_info
        shared_backend
    spirv_backend
)


