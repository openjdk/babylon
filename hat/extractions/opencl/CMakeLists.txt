cmake_minimum_required(VERSION 3.22.1)
project(discovery)
find_package(OpenCL REQUIRED)
if(NOT DEFINED PROPS_OUTPUT_DIR)
    set(PROPS_OUTPUT_DIR "${CMAKE_CURRENT_BINARY_DIR}")
endif()

file(WRITE "${PROPS_OUTPUT_DIR}/paths.properties"
    "mac.frameworks.dir=${CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES}\n"
    "opencl.found=${OpenCL_FOUND}\n"
    #"opencl.header=${OpenCL_INCLUDE_DIR}\n"
    #"opencl.lib=${OpenCL_LIBRARY}\n"
    #"opencl.version=${OpenCL_VERSION_STRING}\n"
)
