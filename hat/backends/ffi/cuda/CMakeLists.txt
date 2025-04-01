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
            ${CUDAToolkit_INCLUDE_DIR}
	        ${SHARED_BACKEND}/include
	        ${CUDA_BACKEND}/include
    )

    link_directories(
            ${CMAKE_BINARY_DIR}
            ${CUDAToolkit_LIBRARY_DIR}
    )

    add_library(cuda_backend SHARED
            ${SHARED_BACKEND}/cpp/shared.cpp
            ${SHARED_BACKEND}/include/shared.h
            ${SHARED_BACKEND}/cpp/buffer.cpp
            ${SHARED_BACKEND}/include/buffer.h
            ${SHARED_BACKEND}/cpp/schema_cursor.cpp
            ${SHARED_BACKEND}/include/schema_cursor.h
            ${SHARED_BACKEND}/cpp/buffer_cursor.cpp
            ${SHARED_BACKEND}/include/buffer_cursor.h
            ${SHARED_BACKEND}/cpp/hex.cpp
            ${SHARED_BACKEND}/include/hex.h
            ${SHARED_BACKEND}/cpp/json.cpp
            ${SHARED_BACKEND}/include/json.h
            ${SHARED_BACKEND}/cpp/fsutil.cpp
            ${SHARED_BACKEND}/include/fsutil.h
            ${SHARED_BACKEND}/cpp/strutil.cpp
            ${SHARED_BACKEND}/include/strutil.h
            ${CUDA_BACKEND}/include/cuda_backend.h
	    ${CUDA_BACKEND}/cpp/cuda_backend.cpp
    )

    target_link_libraries(cuda_backend
            -lcudart
            -lcuda
    )

    add_executable(cuda_info
	    ${CUDA_BACKEND}/cpp/info.cpp
    )

    target_link_libraries(cuda_info
            cuda_backend
            -lcudart
            -lcuda
    )
endif()
