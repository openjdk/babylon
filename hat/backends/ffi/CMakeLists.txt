cmake_minimum_required(VERSION 3.22.1)
project(backends)
set(CMAKE_CXX_STANDARD 14)

if ("${HAT_TARGET}EMPTY" STREQUAL "EMPTY")
    message("HAT_TARGET is empty")
else ()
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${HAT_TARGET})
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${HAT_TARGET})
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${HAT_TARGET})
    message("Binaries in ${HAT_TARGET} HAT_TARGET")
endif ()

if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/shared")
    message("SHARED_BACKEND=${SHARED_BACKEND}")
endif()

include_directories(
    ${SHARED_BACKEND}/include
)

if ("${OPENCL_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (OPENCL_BACKEND "${CMAKE_SOURCE_DIR}/opencl")
    message("OPENCL_BACKEND=${OPENCL_BACKEND}")
endif()
add_subdirectory(opencl)

if ("${CUDA_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (CUDA_BACKEND "${CMAKE_SOURCE_DIR}/cuda")
    message("CUDA_BACKEND=${CUDA_BACKEND}")
endif()
add_subdirectory(cuda)

if ("${SPIRV_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (SPIRV_BACKEND "${CMAKE_SOURCE_DIR}/spirv")
    message("SPIRV_BACKEND=${SPIRV_BACKEND}")
endif()
add_subdirectory(spirv)

#if ("${PTX_BACKEND}EMPTY" STREQUAL "EMPTY")
#    set (PTX_BACKEND "${CMAKE_SOURCE_DIR}/ptx")
#    message("PTX_BACKEND=${PTX_BACKEND}")
#endif()
#add_subdirectory(ptx)

if ("${HIP_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (HIP_BACKEND "${CMAKE_SOURCE_DIR}/hip")
    message("HIP_BACKEND=${HIP_BACKEND}")
endif()
add_subdirectory(hip)

if ("${MOCK_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (MOCK_BACKEND "${CMAKE_SOURCE_DIR}/mock")
    message("MOCK_BACKEND=${MOCK_BACKEND}")
endif()
add_subdirectory(mock)

add_executable(schemadump
        ${SHARED_BACKEND}/cpp/schema.cpp
        ${SHARED_BACKEND}/cpp/schemadump.cpp
        ${SHARED_BACKEND}/cpp/schema_cursor.cpp
        ${SHARED_BACKEND}/include/schema_cursor.h
        ${SHARED_BACKEND}/include/schema.h
)

