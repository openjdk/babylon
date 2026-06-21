cmake_minimum_required(VERSION 3.22.1)
project(discovery)
find_package(OpenGL REQUIRED)
if(NOT DEFINED PROPS_OUTPUT_DIR)
    set(PROPS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

file(WRITE "${PROPS_OUTPUT_DIR}/paths.properties"
    "mac.frameworks.dir=${CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES}\n"
    "opengl.found=${OpenCL_FOUND}\n"
    "opengl.header=${OpenCL_INCLUDE_DIR}\n"
    "opengl.lib=${OpenCL_LIBRARY}\n"
)
