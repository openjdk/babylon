cmake_minimum_required(VERSION 3.22.1)
project(discovery)
#foreach (_variableName ${_variableNames})
#   message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
find_package(OpenCL REQUIRED)
# Generate the properties file for Maven
# Note: ${OpenCL_LIBRARY} and ${OpenCL_INCLUDE_DIR} are set by find_package
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/opencl_paths.properties"
    "mac.frameworks.dir=${CMAKE_C_IMPLICIT_LINK_FRAMEWORK_DIRECTORIES}\n"
    "opencl.found=${OpenCL_FOUND}\n"
    "opencl.header=${OpenCL_INCLUDE_DIR}\n"
    "opencl.lib=${OpenCL_LIBRARY}\n"
    "opencl.version=${OpenCL_VERSION_STRING}\n"
)
