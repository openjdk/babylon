cmake_minimum_required(VERSION 3.22.1)
project(extract_cuda)

#We can assume that 'parent' has guarded us with find_package(XxxxXX) + if(XXXXXX_FOUND)

set(JEXTRACT_PACKAGE cuda)
set(JEXTRACT_SOURCE ${CMAKE_SOURCE_DIR}/${JEXTRACT_PACKAGE}/src/main/java)
set(JEXTRACT_HEADER ${JEXTRACT_SOURCE}/${JEXTRACT_PACKAGE}/${JEXTRACT_PACKAGE}_h.java)  

if (Apple)

else()
add_custom_command(OUTPUT  ${JEXTRACT_HEADER}
   COMMAND mkdir -p ${JEXTRACT_SOURCE}
   COMMAND jextract 
       --target-package ${JEXTRACT_PACKAGE} 
       --output ${JEXTRACT_SOURCE} 
       --library :${CUDA_cuda_driver_LIBRARY}
       --header-class-name ${JEXTRACT_PACKAGE}_h
       ${CUDAToolkit_INCLUDE_DIR}/cuda.h
       WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
       VERBATIM
   )
#CUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12.2/include
#CUDA_cuda_driver_LIBRARY =/usr/lib/aarch64-linux-gnu/libcuda.so

endif()

add_custom_target(extract_cuda DEPENDS  ${JEXTRACT_HEADER})
