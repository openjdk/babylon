cmake_minimum_required(VERSION 3.22.1)
project(cuda_backend)

set(CMAKE_CXX_STANDARD 14)
if(UNIX)
   if(NOT DEFINED ROCM_PATH)
      set(ROCM_PATH "/opt/rocm" CACHE STRING "Default ROCM installation directory.")
   endif()
   list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}")
endif()
find_package(HIP)
#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
if(HIP_FOUND)
    set(CMAKE_CXX_COMPILER  ${HIP_HIPCC_EXECUTABLE})
    set(CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE})

    if ("${HIP_BACKEND}EMPTY" STREQUAL "EMPTY")
      set (HIP_BACKEND "${CMAKE_SOURCE_DIR}")
      message("HIP_BACKEND=${HIP_BACKEND}")
    endif()

    if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
        set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/../shared")
        message("SHARED_BACKEND=${SHARED_BACKEND}")
    endif()

    include_directories(
       hip::host
       ${HIP_INCLUDE_DIR}
       ${SHARED_BACKEND}/src/main/native/include
       ${HIP_BACKEND}/src/main/native/include
    )

    link_directories(
       ${CMAKE_BINARY_DIR}
      # ${hip_LIB_INSTALL_DIR}
    )

    add_library(hip_backend SHARED
       ${HIP_BACKEND}/src/main/native/cpp/hip_backend.cpp
    )

    target_link_libraries(hip_backend
       PRIVATE hip::host
    )

    add_executable(hip_info
       ${HIP_BACKEND}/src/main/native/cpp/info.cpp
    )

    target_link_libraries(hip_info
       hip_backend
       hip::host
    )
else()
   messge("NO HIP for YOU!")
endif()
