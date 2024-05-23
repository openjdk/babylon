cmake_minimum_required(VERSION 3.28)
project(backends)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL)
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
