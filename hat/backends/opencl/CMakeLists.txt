cmake_minimum_required(VERSION 3.22.1)
project(opencl_backend)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL)
if(OPENCL_FOUND)
    message("OPENCL")
    if ("${OPENCL_BACKEND}EMPTY" STREQUAL "EMPTY")
        set (OPENCL_BACKEND "${CMAKE_SOURCE_DIR}")
        message("OPENCL_BACKEND=${OPENCL_BACKEND}")
    endif()
    if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
        set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/../shared")
        message("SHARED_BACKEND=${SHARED_BACKEND}")
    endif()

    if (APPLE)
       set(OPENCL_INCLUDE_DIR "-framework OpenCL")
       set(OPENCL_LIB "-framework OpenCL")
    else()
       set(OPENCL_LIB "OpenCL")
    endif()

    include_directories(
        ${OPENCL_BACKEND}/include
        ${SHARED_BACKEND}/include
        ${OPENCL_INCLUDE_DIR}
    )
    link_directories(
        ${CMAKE_BINARY_DIR}
    )

    add_library(opencl_backend SHARED
        ${SHARED_BACKEND}/cpp/shared.cpp
        ${OPENCL_BACKEND}/cpp/opencl_backend.cpp
    )

    target_link_libraries(opencl_backend
        ${OPENCL_LIB}
    )

    add_executable(opencl_info
        ${OPENCL_BACKEND}/cpp/info.cpp
    )

    target_link_libraries(opencl_info
        opencl_backend
        ${OPENCL_LIB}
    )
    add_custom_target(opencl_natives DEPENDS opencl_info opencl_backend)
endif()
