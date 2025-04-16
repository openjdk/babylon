cmake_minimum_required(VERSION 3.22.1)
project(extractions)

set(OSX_SDK ${CMAKE_OSX_SYSROOT})
set(OSX_SDK_LIBRARY_FRAMEWORKS ${OSX_SDK}/System/Library/Frameworks)
set(OSX_SYS_LIBRARY_FRAMEWORKS /System/Library/Frameworks)

set(EXTRACTIONS "")

find_package(OpenGL)
if(OPENGL_FOUND)
   add_subdirectory(opengl)
   set(EXTRACTIONS ${EXTRACTIONS} extract_opengl)
else()
   message("NO OPENGL")
endif()

find_package(OpenCL)
if(OPENCL_FOUND)
   add_subdirectory(opencl)
   set(EXTRACTIONS ${EXTRACTIONS} extract_opencl)
else()
   message("NO OPENCL")
endif()

find_package(CUDA)
if(CUDA_FOUND)
   add_subdirectory(cuda)
   set(EXTRACTIONS ${EXTRACTIONS} extract_cuda)
else()
   message("NO CUDA")
endif()

add_custom_target(extract DEPENDS ${EXTRACTIONS})


