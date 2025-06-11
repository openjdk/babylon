cmake_minimum_required(VERSION 3.22.1)
project(cuda_backend)

set(CMAKE_CXX_STANDARD 14)

# this is the only way I can think of to get cmake to find FindHIP.cmake
set(ROCM_PATH "/opt/rocm")
list(APPEND CMAKE_MODULE_PATH "${ROCM_PATH}/lib/cmake/hip")
# /opt/rocm-6.3.4/lib/cmake/hip/FindHIP.cmake
# Weirdly the above does not set HIP_INCLUDE_DIR
# This seems to set expected VARS but is not 
# invoked  /opt/rocm-6.3.4/lib/cmake/hip/hip-config.cmake
find_package(HIP)
get_cmake_property(_variableNames VARIABLES)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()
if(HIP_FOUND)
   if(HIP_READY) #  get rid of this once we figur out how to build
    set(HIP_INCLUDE_DIR "/opt/rocm/include")
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
       ${hip_LIB_INSTALL_DIR}
    )

    add_library(hip_backend SHARED
       ${SHARED_BACKEND}/cpp/shared.cpp
       ${HIP_BACKEND}/cpp/hip_backend.cpp
    )

    #target_link_libraries(hip_backend
    #   PRIVATE hip::host
    #)

    add_executable(hip_info
       ${HIP_BACKEND}/cpp/info.cpp
    )

    target_link_libraries(hip_info
       hip_backend
     #  hip::host
    )
  else()
     message("FOUND HIP but CMakefile still broken")
  endif()
endif()
