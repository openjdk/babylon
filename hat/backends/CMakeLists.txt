cmake_minimum_required(VERSION 3.22.1)
project(backends)

add_custom_target(copy_libs)

if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/shared")
    message("SHARED_BACKEND=${SHARED_BACKEND}")
endif()

set(CMAKE_CXX_STANDARD 14)

include_directories(
    ${SHARED_BACKEND}/include
)

if ("${OPENCL_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (OPENCL_BACKEND "${CMAKE_SOURCE_DIR}/opencl")
    message("OPENCL_BACKEND=${OPENCL_BACKEND}")
endif()
add_subdirectory(opencl)

if ("${CUDA_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (CUDA_BACKEND "${CMAKE_SOURCE_DIR}/cuda")
    message("CUDA_BACKEND=${CUDA_BACKEND}")
endif()
add_subdirectory(cuda)

if ("${SPIRV_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (SPIRV_BACKEND "${CMAKE_SOURCE_DIR}/spirv")
    message("SPIRV_BACKEND=${SPIRV_BACKEND}")
endif()
add_subdirectory(spirv)

if ("${PTX_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (PTX_BACKEND "${CMAKE_SOURCE_DIR}/ptx")
    message("PTX_BACKEND=${PTX_BACKEND}")
endif()
add_subdirectory(ptx)

if ("${MOCK_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (MOCK_BACKEND "${CMAKE_SOURCE_DIR}/mock")
    message("MOCK_BACKEND=${MOCK_BACKEND}")
endif()
add_subdirectory(mock)

add_executable(schemadump
        ${SHARED_BACKEND}/cpp/schema.cpp
        ${SHARED_BACKEND}/cpp/schemadump.cpp
)

