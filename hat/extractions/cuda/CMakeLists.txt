cmake_minimum_required(VERSION 3.22.1)
project(extract_cuda)

#We can assume that 'parent' has guarded us with find_package(XxxxXX) + if(XXXXXX_FOUND)

set(JEXTRACT_PACKAGE cuda)
set(JEXTRACT_SOURCE ${CMAKE_SOURCE_DIR}/${JEXTRACT_PACKAGE}/src/main/java)
set(JEXTRACT_HEADER ${JEXTRACT_SOURCE}/${JEXTRACT_PACKAGE}/${JEXTRACT_PACKAGE}_h.java)  

add_custom_command(OUTPUT  ${JEXTRACT_HEADER}
   COMMAND mkdir -p ${JEXTRACT_SOURCE}
   #COMMAND echo -F${OSX_SDK_LIBRARY_FRAMEWORKS} > ${CMAKE_BINARY_DIR}/compile_flags.txt
   COMMAND jextract 
       --target-package ${JEXTRACT_PACKAGE} 
       --output ${JEXTRACT_SOURCE} 
       --library :${OSX_SYS_LIBRARY_FRAMEWORKS}/OpenCL.framework/OpenCL 
       --header-class-name ${JEXTRACT_PACKAGE}_h
       ${OSX_SDK_LIBRARY_FRAMEWORKS}/OpenCL.framework/Headers/opencl.h
       WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
       VERBATIM
   )

add_custom_target(extract_opencl DEPENDS  ${JEXTRACT_HEADER})
