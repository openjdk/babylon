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
    add_custom_target(cuda_natives DEPENDS cuda_info cuda_backend)

    add_custom_target(copy_cuda_libs DEPENDS cuda_info cuda_backend
        COMMAND echo cp ${CMAKE_BINARY_DIR}/cuda/libcuda_backend.* ${CMAKE_SOURCE_DIR}/../maven-build
        COMMAND cp ${CMAKE_BINARY_DIR}/cuda/cuda_info ${CMAKE_SOURCE_DIR}/../maven-build
   )
add_dependencies(copy_libs copy_cuda_libs)

endif()
