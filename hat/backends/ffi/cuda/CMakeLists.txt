cmake_minimum_required(VERSION 3.22.1)
project(cuda_backend)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message("CUDA")
    if ("${CUDA_BACKEND}EMPTY" STREQUAL "EMPTY")
	    set (CUDA_BACKEND "${CMAKE_SOURCE_DIR}")
	    message("CUDA_BACKEND=${CUDA_BACKEND}")
    endif()

    if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
        set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/../shared")
        message("SHARED_BACKEND=${SHARED_BACKEND}")
    endif()

    include_directories(
            ${CUDAToolkit_INCLUDE_DIRS} # was ${CUDAToolkit_INCLUDE_DIR}
	        ${SHARED_BACKEND}/src/main/native/include
	        ${CUDA_BACKEND}/src/main/native/include
    )

    link_directories(
            ${CMAKE_BINARY_DIR}
            ${CUDAToolkit_LIBRARY_DIR}
    )

    add_library(cuda_backend SHARED
            ${CUDA_BACKEND}/src/main/native/include/cuda_backend.h
	        ${CUDA_BACKEND}/src/main/native/cpp/cuda_backend.cpp
            ${CUDA_BACKEND}/src/main/native/cpp/cuda_backend_buffer.cpp
            ${CUDA_BACKEND}/src/main/native/cpp/cuda_backend_kernel.cpp
            ${CUDA_BACKEND}/src/main/native/cpp/cuda_backend_module.cpp
            ${CUDA_BACKEND}/src/main/native/cpp/cuda_backend_queue.cpp
    )

    target_link_libraries(cuda_backend
            shared_backend
            -lcuda
    )

    add_executable(cuda_info
	    ${CUDA_BACKEND}/src/main/native/cpp/info.cpp
    )

    add_executable(cuda_squares
            ${CUDA_BACKEND}/src/main/native/cpp/squares.cpp
    )

    target_link_libraries(cuda_squares
            cuda_backend

    )

    target_link_libraries(cuda_info
            cuda_backend

    )
endif()
