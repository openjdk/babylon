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
        ${OPENCL_BACKEND}/src/main/native/include
        ${SHARED_BACKEND}/src/main/native/include
        ${OPENCL_INCLUDE_DIR}
    )
    link_directories(
        ${CMAKE_BINARY_DIR}
    )

    add_library(opencl_backend SHARED
        ${OPENCL_BACKEND}/src/main/native/include/opencl_backend.h
        ${OPENCL_BACKEND}/src/main/native/cpp/opencl_backend.cpp
        ${OPENCL_BACKEND}/src/main/native/cpp/opencl_backend_buffer.cpp
        ${OPENCL_BACKEND}/src/main/native/cpp/opencl_backend_info.cpp
        ${OPENCL_BACKEND}/src/main/native/cpp/opencl_backend_queue.cpp
        ${OPENCL_BACKEND}/src/main/native/cpp/opencl_backend_kernel.cpp
        ${OPENCL_BACKEND}/src/main/native/cpp/opencl_backend_program.cpp
    )

    target_link_libraries(opencl_backend
            shared_backend
        ${OPENCL_LIB}
    )

    add_executable(opencl_info
            ${OPENCL_BACKEND}/src/main/native/include/opencl_backend.h
            ${OPENCL_BACKEND}/src/main/native/cpp/info.cpp
    )
    add_executable(opencl_squares
            ${OPENCL_BACKEND}/src/main/native/include/opencl_backend.h
            ${OPENCL_BACKEND}/src/main/native/cpp/squares.cpp
        )

    target_link_libraries(opencl_squares
        opencl_backend

    )
    target_link_libraries(opencl_info
        opencl_backend

    )

    add_custom_command(OUTPUT ${CMAKE_BINARY_DIR}/extractFile
            COMMAND echo jextract stuff
            #COMMAND BANNER jextract stuff
            COMMAND touch ${CMAKE_BINARY_DIR}/extractFile
            #COMMAND ${CMAKE_COMMAND} -E touch ${LIBFOO_TAR_HEADERS}
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            #DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/libfoo/foo.tar"
            COMMENT "Extracting opencl"
            VERBATIM
    )

    add_custom_target(extract DEPENDS ${CMAKE_BINARY_DIR}/extractFile)

endif()
