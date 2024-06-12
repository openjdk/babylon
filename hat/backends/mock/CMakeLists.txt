cmake_minimum_required(VERSION 3.22.1)
project(opencl_backend)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCL)
if ("${MOCK_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (MOCK_BACKEND "${CMAKE_SOURCE_DIR}")
    message("MOCK_BACKEND=${MOCK_BACKEND}")
endif()
if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
    set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/../shared")
    message("SHARED_BACKEND=${SHARED_BACKEND}")
endif()


message("MOCK")

include_directories(
    ${MOCK_BACKEND}/include
    ${SHARED_BACKEND}/include
    ${MOCK_INCLUDE_DIR}
)
link_directories(
    ${CMAKE_BINARY_DIR}
)

add_library(mock_backend SHARED
    ${SHARED_BACKEND}/cpp/shared.cpp
    ${MOCK_BACKEND}/cpp/mock_backend.cpp
)


add_executable(mock_info
    ${MOCK_BACKEND}/cpp/info.cpp
)

target_link_libraries(mock_info
    mock_backend
)
add_custom_target(mock_natives DEPENDS mock_info mock_backend)
add_custom_target(copy_mock_libs DEPENDS mock_info mock_backend
    COMMAND cp ${CMAKE_BINARY_DIR}/mock/libmock_backend.* ${CMAKE_SOURCE_DIR}/../maven-build
    COMMAND cp ${CMAKE_BINARY_DIR}/mock/mock_info ${CMAKE_SOURCE_DIR}/../maven-build
)
add_dependencies(copy_libs copy_mock_libs)




