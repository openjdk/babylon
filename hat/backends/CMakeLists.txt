cmake_minimum_required(VERSION 3.22.1)
project(backends)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL)
find_package(CUDAToolkit)

if(OPENCL_FOUND)
    message("OPENCL")
    if (APPLE)
       set(OPENCL_INCLUDE_DIR "-framework OpenCL")
       set(OPENCL_LIB "-framework OpenCL")
    else()
       #set(OPENCL_INCLUDE_DIR "-framework OpenCL")  
       set(OPENCL_LIB "OpenCL")
    endif()

    include_directories(
        ${CMAKE_SOURCE_DIR}/opencl/shared/include
        ${CMAKE_SOURCE_DIR}/opencl/include
        ${CMAKE_SOURCE_DIR}/shared/include
        ${OPENCL_INCLUDE_DIR}
    )
    link_directories(
        ${CMAKE_BINARY_DIR}
    )

    add_library(opencl_backend SHARED
        ${CMAKE_SOURCE_DIR}/shared/cpp/shared.cpp
        ${CMAKE_SOURCE_DIR}/opencl/cpp/opencl_backend.cpp
    )

    target_link_libraries(opencl_backend
        ${OPENCL_LIB}
    )

    add_executable(opencl_info
        ${CMAKE_SOURCE_DIR}/opencl/cpp/info.cpp
    )

    target_link_libraries(opencl_info
        opencl_backend
        ${OPENCL_LIB}
    )
    add_custom_target(opencl_natives DEPENDS opencl_info opencl_backend)
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
    add_custom_target(cuda_natives DEPENDS cuda_info cuda_backend)

endif()
