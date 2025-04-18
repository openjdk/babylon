cmake_minimum_required(VERSION 3.22.1)
project(extract_opencl)

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()
find_package(OpenCL)
if(OPENCL_FOUND)
   set(JEXTRACT_PACKAGE opencl)
   set(JEXTRACT_SOURCE ${CMAKE_SOURCE_DIR}/src/main/java)
   set(JEXTRACT_HEADER ${JEXTRACT_SOURCE}/${JEXTRACT_PACKAGE}/${JEXTRACT_PACKAGE}_h.java)  

   if (APPLE)
      set(OSX_SDK ${CMAKE_OSX_SYSROOT})
      set(OSX_SDK_LIBRARY_FRAMEWORKS ${OSX_SDK}/System/Library/Frameworks)
      set(OSX_SYS_LIBRARY_FRAMEWORKS /System/Library/Frameworks)
   
     add_custom_command(OUTPUT  ${JEXTRACT_HEADER}
        COMMAND echo creating ${JEXTRACT_SOURCE}
        COMMAND mkdir -p ${JEXTRACT_SOURCE}
        COMMAND echo -F${OSX_SDK_LIBRARY_FRAMEWORKS} > ${CMAKE_BINARY_DIR}/compile_flags.txt
        COMMAND jextract 
            --target-package ${JEXTRACT_PACKAGE} 
            --output ${JEXTRACT_SOURCE} 
            --library :${OSX_SYS_LIBRARY_FRAMEWORKS}/OpenCL.framework/OpenCL 
           --header-class-name ${JEXTRACT_PACKAGE}_h
              ${OSX_SDK_LIBRARY_FRAMEWORKS}/OpenCL.framework/Headers/opencl.h
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
            VERBATIM
       )
   else()
   
   endif()

   add_custom_target(extract_opencl DEPENDS  ${JEXTRACT_HEADER})
else()
   message("NO OPENCL")
endif()
