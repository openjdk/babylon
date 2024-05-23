cmake_minimum_required(VERSION 3.22.1)
project(backends)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL)
find_package(CUDAToolkit)

if(OPENCL_FOUND)
    message("OPENCL")
    if (APPLE)
        include_directories(
                ${CMAKE_SOURCE_DIR}/opencl/shared/include
                ${CMAKE_SOURCE_DIR}/opencl/include
                ${CMAKE_SOURCE_DIR}/shared/include
                "-framework OpenCL"
        )
        link_directories(
                ${CMAKE_BINARY_DIR}
        )

        add_library(opencl_backend SHARED
                ${CMAKE_SOURCE_DIR}/shared/cpp/shared.cpp
                ${CMAKE_SOURCE_DIR}/opencl/cpp/opencl_backend.cpp
        )

        target_link_libraries(opencl_backend
                "-framework OpenCL"
        )

        add_executable(opencl_info
                ${CMAKE_SOURCE_DIR}/opencl/cpp/info.cpp
        )

        target_link_libraries(opencl_info
                opencl_backend
                "-framework OpenCL"
        )
        add_custom_target(backends DEPENDS opencl_info opencl_backend)
    else()
        message("OPENCL but not on APPLE")
    endif()

endif()

if(CUDAToolkit_FOUND)
    message("CUDA")

    include_directories(
            ${CUDAToolkit_INCLUDE_DIR}
            ${CMAKE_SOURCE_DIR}/shared/include
            ${CMAKE_SOURCE_DIR}/cuda/include
    )

    link_directories(
            ${CMAKE_BINARY_DIR}
            ${CUDAToolkit_LIBRARY_DIR}
    )

    add_library(cuda_backend SHARED
            ${CMAKE_SOURCE_DIR}/shared/cpp/shared.cpp
            ${CMAKE_SOURCE_DIR}/cuda/cpp/cuda_backend.cpp
    )

    target_link_libraries(cuda_backend
            -lcudart
            -lcuda
    )

    add_executable(cuda_info
            ${CMAKE_SOURCE_DIR}/cuda/cpp/info.cpp
    )

    target_link_libraries(cuda_info
            cuda_backend
            -lcudart
            -lcuda
    )
    add_custom_target(backends DEPENDS cuda_info cuda_backend)

endif()
