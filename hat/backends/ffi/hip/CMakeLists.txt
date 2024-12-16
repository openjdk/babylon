cmake_minimum_required(VERSION 3.22.1)
project(cuda_backend)

set(CMAKE_CXX_STANDARD 14)

find_package(hip)
if(hip_FOUND)
	message("HIP")

	if ("${HIP_BACKEND}EMPTY" STREQUAL "EMPTY")
		set (HIP_BACKEND "${CMAKE_SOURCE_DIR}")
		message("HIP_BACKEND=${HIP_BACKEND}")
    endif()

    if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
        set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/../shared")
        message("SHARED_BACKEND=${SHARED_BACKEND}")
    endif()

    include_directories(
            ${HIP_INCLUDE_DIR}
            ${SHARED_BACKEND}/include
            ${HIP_BACKEND}/include
    )

    link_directories(
            ${CMAKE_BINARY_DIR}
            ${HIP_LIBRARY_DIR}
    )

    add_library(hip_backend SHARED
	    ${SHARED_BACKEND}/cpp/shared.cpp
	    ${HIP_BACKEND}/cpp/hip_backend.cpp
    )

    target_link_libraries(hip_backend
	    PRIVATE hip::host
    )

    add_executable(hip_info
	    ${HIP_BACKEND}/cpp/info.cpp
    )

    target_link_libraries(hip_info
            hip_backend
            hip::host
    )
endif()
