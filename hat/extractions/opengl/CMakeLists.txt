cmake_minimum_required(VERSION 3.22.1)
project(extract_opengl)

#We can assume that 'parent' has guarded us with find_package(XxxxXX) + if(XXXXXX_FOUND)

set(JEXTRACT_PACKAGE opengl)
set(JEXTRACT_SOURCE ${CMAKE_SOURCE_DIR}/${JEXTRACT_PACKAGE}/src/main/java)
set(JEXTRACT_HEADER ${JEXTRACT_SOURCE}/${JEXTRACT_PACKAGE}/${JEXTRACT_PACKAGE}_h.java)  

if (APPLE)
add_custom_command(OUTPUT  ${JEXTRACT_HEADER}
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
   COMMAND mkdir -p ${JEXTRACT_SOURCE}
   COMMAND jextract 
       --target-package ${JEXTRACT_PACKAGE} 
       --output ${JEXTRACT_SOURCE} 
       --library :/usr/lib/aarch64-linux-gnu/libOpenGL.so
       --library :/usr/lib/aarch64-linux-gnu/libGLU.so
       --header-class-name ${JEXTRACT_PACKAGE}_h
       /usr/include/GL/glut.h
   WORKING_DIRECTORY "${CMAKE_BINARY_DIR}"
   VERBATIM
)

# --target-package opengl \
#    --output /home/gfrost/github/grfrost/babylon-grfrost-fork/hat/stage/jextract \
#    --library :/usr/lib/aarch64-linux-gnu/libOpenGL.so:/usr/lib/aarch64-linux-gnu/libGLU.so\
#     /usr/include/GL/glut.h

#    --library :/usr/lib/aarch64-linux-gnu/libOpenGL.so:/usr/lib/aarch64-linux-gnu/libGLX.so:/usr/lib/aarch64-linux-gnu/libGLU.so/libOpenGL.so \ /usr/include/GL/glut.h

endif()

add_custom_target(extract_opengl DEPENDS  ${JEXTRACT_HEADER})
