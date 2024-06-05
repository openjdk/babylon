cmake_minimum_required(VERSION 3.22.1)
project(cuda_backend)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message("CUDA")

    include_directories(
            ${CUDAToolkit_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/${SHARED_PATH_ADJUST}/shared/include
            ${CMAKE_SOURCE_DIR}/${CUDA_PATH_ADJUST}/include
    )

    link_directories(
            ${CMAKE_BINARY_DIR}
            ${CUDAToolkit_LIBRARY_DIR}
    )

    add_library(cuda_backend SHARED
            ${CMAKE_SOURCE_DIR}/${SHARED_PATH_ADJUST}/cpp/shared.cpp
            ${CMAKE_SOURCE_DIR}/${CUDA_PATH_ADJUST}/cpp/cuda_backend.cpp
    )

    target_link_libraries(cuda_backend
            -lcudart
            -lcuda
    )

    add_executable(cuda_info
            ${CMAKE_SOURCE_DIR}/${CUDA_PATH_ADJUST}/cpp/info.cpp
    )

    target_link_libraries(cuda_info
            cuda_backend
            -lcudart
            -lcuda
    )
    add_custom_target(cuda_natives DEPENDS cuda_info cuda_backend)


endif()