cmake_minimum_required(VERSION 3.22.1)
project(extract_opengl)

#get_cmake_property(_variableNames VARIABLES)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

find_package(OpenGL)
if(OPENGL_FOUND)
   set(JEXTRACT_PACKAGE opengl)
   set(JEXTRACT_SOURCE ${CMAKE_SOURCE_DIR}/${JEXTRACT_PACKAGE}/src/main/java)
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
             --library :${OSX_SYS_LIBRARY_FRAMEWORKS}/OpenGL.framework/OpenGL
             --library :${OSX_SYS_LIBRARY_FRAMEWORKS}/GLUT.framework/GLUT
             --header-class-name ${JEXTRACT_PACKAGE}_h
             ${OSX_SDK_LIBRARY_FRAMEWORKS}/GLUT.framework/Headers/glut.h
         WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
         VERBATIM
      )
   else()
      add_custom_command(OUTPUT  ${JEXTRACT_HEADER}
         COMMAND echo creating ${JEXTRACT_SOURCE}
         COMMAND mkdir -p ${JEXTRACT_SOURCE}
         COMMAND jextract
             --target-package ${JEXTRACT_PACKAGE}
             --output ${JEXTRACT_SOURCE}
             --library :/usr/lib/${CMAKE_CXX_LIBRARY_ARCHITECTURE}/libOpenGL.so
#             --library :/usr/lib/${CMAKE_CXX_LIBRARY_ARCHITECTURE}/libGLU.so
             --library :/usr/lib/${CMAKE_CXX_LIBRARY_ARCHITECTURE}/libglut.so.3.12.0
             --header-class-name ${JEXTRACT_PACKAGE}_h
             /usr/include/GL/glut.h
         WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
         VERBATIM
      )

   endif()

   add_custom_target(extract_opengl DEPENDS  ${JEXTRACT_HEADER})
else()
   message("NO OPENGL")
endif()
