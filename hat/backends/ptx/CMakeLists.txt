cmake_minimum_required(VERSION 3.22.1)
project(ptx_backend)

set(CMAKE_CXX_STANDARD 14)

find_package(CUDAToolkit)
if(CUDAToolkit_FOUND)
    message("PTX")
    if ("${PTX_BACKEND}EMPTY" STREQUAL "EMPTY")
	    set (PTX_BACKEND "${CMAKE_SOURCE_DIR}")
	    message("PTX_BACKEND=${PTX_BACKEND}")
    endif()
    if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
        set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/../shared")
        message("SHARED_BACKEND=${SHARED_BACKEND}")
    endif()

    include_directories(
            ${CUDAToolkit_INCLUDE_DIR}
	    ${SHARED_BACKEND}/include
	    ${PTX_BACKEND}/include
    )

    link_directories(
            ${CMAKE_BINARY_DIR}
            ${CUDAToolkit_LIBRARY_DIR}
    )

    add_library(ptx_backend SHARED
	    ${SHARED_BACKEND}/cpp/shared.cpp
	    ${PTX_BACKEND}/cpp/ptx_backend.cpp
    )

    target_link_libraries(ptx_backend
            -lcudart
            -lcuda
    )

    add_executable(ptx_info
	    ${PTX_BACKEND}/cpp/info.cpp
    )

    target_link_libraries(ptx_info
            ptx_backend
            -lcudart
            -lcuda
    )

endif()


