cmake_minimum_required(VERSION 3.21)  #limited by jetson project(HAT)

project(hat)

set(BABYLON_JDK ${CMAKE_SOURCE_DIR}/..)
if (EXISTS ${BABYLON_JDK})
   message("BABYLON_JDK ${BABYLON_JDK} OK")
else()
   message(FATAL_ERROR "BABYLON_JDK ${BABYLON_JDK} does not point to anywhere useful ")
endif()

set(BEEHIVE $ENV{HOME}/github/beehive-spirv-toolkit)
if (EXISTS ${BEEHIVE})
   message("BEEHIVE ${BEEHIVE}")
else()
   message(WARNING "BEEHIVE ${BEEHIVE} does not point to anywhere useful.  We can build most of the project but can't build SPIRV backend without BEEHIVE")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_LINKER_COMMANDS ON)

if(is$ENV{ECHO_ONLY} STREQUAL isON )
   set(DOIT echo)
   set_property(GLOBAL PROPERTY TARGET_MESSAGES OFF)
   message("ECHO_ONLY=ON! I hope you know the implications ")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CXXFLAGS "-g -Wno-deprecate-declarations")
set(CFLAGS "-g -Wno-deprecate-declarations")

find_package(OpenCL)         # Note we don't ask for 'required'
find_package(CUDAToolkit)    # Note we don't ask for 'required'

#The following stanza just locates the specific image for babylon
if(${CMAKE_SYSTEM_NAME} STREQUAL Darwin )
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL arm64 )
       set(JAVA_SYS_TYPE "macosx-aarch64")
    else()
       set(JAVA_SYS_TYPE "macosx-x86_64")
    endif()
elseif(${CMAKE_SYSTEM_NAME} STREQUAL Linux )
    if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL aarch64 )
       set(JAVA_SYS_TYPE "linux-aarch64")
    else()
       set(JAVA_SYS_TYPE "linux-x86_64")
    endif()
else()
   message(FATAL_ERROR "What machine is this?")
endif()

set(JAVA_HOME ${BABYLON_JDK}/build/${JAVA_SYS_TYPE}-server-release/jdk)

if (EXISTS ${JAVA_HOME})
   message("JAVA_HOME ${JAVA_HOME}")
else()
   message(FATAL_ERROR "JAVA_HOME ${JAVA_HOME} does not point to anywhere useful ")
endif()

## We have JAVA_HOME so now setup clean targets

#set_property(DIRECTORY PROPERTY ADDITIONAL_MAKE_CLEAN_FILES
#   ${CMAKE_BINARY_DIR}/hat ${CMAKE_BINARY_DIR}/mandel ${CMAKE_BINARY_DIR}/violajones ${CMAKE_BINARY_DIR}/experiments ${CMAKE_BINARY_DIR}/heal ${CMAKE_BINARY_DIR}/view 
#   ${CMAKE_BINARY_DIR}/backends
#)

add_custom_target(backend_jars)
add_custom_target(backend_libs)
add_custom_target(example_jars)
add_custom_target(info_executables)

##WORK STARTS HERE

### Compilation dependencies list is
###
###    examples -> hat
###
###    opencl_info(exe)
###                   \
###                     -> libopencl_backend(.so|.dylib)
###                                                      \
###                                                       \
###    spirv_info(exe)  -> libspirv_backend(.so|.dylib) ->     backend(.so|.dylib)
###                                                       /
###                                                     /
###                     -> libcuda_backend(.so|.dylib)
###                   /
###    cuda_info(exe)
###
###
### Runtime dependencies depending on the backend ;)
###
###  mandel (opencl) -> examples        > hat
###                                      /
###                     opencl_backend
###                                      \
###                                       libopencl_backend(.so|.dylib)
###


### build/hat/classes
### build/hat/hat.jar
###
###  (!hat!)) -> tools

set(HAT_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/hat/src/main/java)
set(HAT_CLASS_DIR ${CMAKE_BINARY_DIR}/hat/classes)
set(HAT_JAR ${CMAKE_BINARY_DIR}/hat/hat.jar)
file(GLOB_RECURSE HAT_JAVA_SOURCE_FILES ${HAT_SOURCE_ROOT}/*.java)

add_custom_target(hat.javac
   COMMAND ${DOIT}
     mkdir -p ${HAT_CLASS_DIR}
   COMMAND ${DOIT}
     ${JAVA_HOME}/bin/javac
        --enable-preview --source=23 -g
        -d ${HAT_CLASS_DIR}
        --source-path ${HAT_SOURCE_ROOT}
        --add-exports=java.base/jdk.internal=ALL-UNNAMED
        --add-exports=java.base/jdk.internal.vm.annotation=ALL-UNNAMED
        ${HAT_JAVA_SOURCE_FILES}
)

add_custom_target(hat.jar DEPENDS hat.javac
   COMMAND  ${DOIT}
      ${JAVA_HOME}/bin/jar
        --create
        --no-manifest
        --file ${HAT_JAR}
        -C ${HAT_CLASS_DIR} .
)



### Examples start here

### Squares

set(SQUARES_CLASS_DIR ${CMAKE_BINARY_DIR}/squares/classes)
set(SQUARES_JAR ${CMAKE_BINARY_DIR}/squares/squares.jar)
set(SQUARES_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/examples/squares/src/main/java)
file(GLOB_RECURSE SQUARES_JAVA_SOURCE_FILES ${SQUARES_SOURCE_ROOT}/*.java)

add_custom_target(squares.javac DEPENDS hat.jar
   COMMAND ${DOIT}
     mkdir -p ${SQUARES_CLASS_DIR}
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/javac
       --enable-preview --source=23 -g -Xlint:unchecked
       -d ${SQUARES_CLASS_DIR} --class-path ${HAT_JAR}
       --source-path ${SQUARES_SOURCE_ROOT}
       ${SQUARES_JAVA_SOURCE_FILES}
)

add_custom_target(squares.jar DEPENDS squares.javac
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/jar --create --no-manifest
       --file  ${SQUARES_JAR}
       -C ${SQUARES_CLASS_DIR} .
)
add_dependencies(example_jars squares.jar)

### Heal

set(HEAL_CLASS_DIR ${CMAKE_BINARY_DIR}/heal/classes)
set(HEAL_JAR ${CMAKE_BINARY_DIR}/heal/heal.jar)
set(HEAL_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/examples/heal/src/java)
file(GLOB_RECURSE HEAL_JAVA_SOURCE_FILES ${HEAL_SOURCE_ROOT}/*.java)

add_custom_target(heal.javac DEPENDS hat.jar
   COMMAND ${DOIT}
     mkdir -p ${HEAL_CLASS_DIR}
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/javac
       --enable-preview --source=23 -g -Xlint:unchecked
       -d ${HEAL_CLASS_DIR} --class-path ${HAT_JAR}
       --source-path ${HEAL_SOURCE_ROOT}
       ${HEAL_JAVA_SOURCE_FILES}
)

add_custom_target(heal.jar DEPENDS heal.javac
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/jar --create --no-manifest
       --file  ${HEAL_JAR}
       -C ${HEAL_CLASS_DIR} .
       -C ${CMAKE_SOURCE_DIR}/examples/heal/src/resources .

)
add_dependencies(example_jars heal.jar)

### View

set(VIEW_CLASS_DIR ${CMAKE_BINARY_DIR}/view/classes)
set(VIEW_JAR ${CMAKE_BINARY_DIR}/view/view.jar)
set(VIEW_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/examples/view/src/java)
file(GLOB_RECURSE VIEW_JAVA_SOURCE_FILES ${VIEW_SOURCE_ROOT}/*.java)

add_custom_target(view.javac DEPENDS hat.jar
   COMMAND ${DOIT}
     mkdir -p ${VIEW_CLASS_DIR}
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/javac
       --enable-preview --source=23 -g -Xlint:unchecked
       -d ${VIEW_CLASS_DIR} --class-path ${HAT_JAR}
       --source-path ${VIEW_SOURCE_ROOT}
       ${VIEW_JAVA_SOURCE_FILES}
)

add_custom_target(view.jar DEPENDS view.javac
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/jar --create --no-manifest
       --file  ${VIEW_JAR}
       -C ${VIEW_CLASS_DIR} .
       -C ${CMAKE_SOURCE_DIR}/examples/view/src/resources .

)
add_dependencies(example_jars view.jar)



### Mandel

set(MANDEL_CLASS_DIR ${CMAKE_BINARY_DIR}/mandel/classes)
set(MANDEL_JAR ${CMAKE_BINARY_DIR}/mandel/mandel.jar)
set(MANDEL_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/examples/mandel/src/main/java)
file(GLOB_RECURSE MANDEL_JAVA_SOURCE_FILES ${MANDEL_SOURCE_ROOT}/*.java)

add_custom_target(mandel.javac DEPENDS hat.jar
   COMMAND ${DOIT}
     mkdir -p ${MANDEL_CLASS_DIR}
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/javac
       --enable-preview --source=23 -g -Xlint:unchecked
       -d ${MANDEL_CLASS_DIR} --class-path ${HAT_JAR}
       --source-path ${MANDEL_SOURCE_ROOT}
       ${MANDEL_JAVA_SOURCE_FILES}
)

add_custom_target(mandel.jar DEPENDS mandel.javac
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/jar --create --no-manifest
       --file  ${MANDEL_JAR}
       -C ${MANDEL_CLASS_DIR} .
)
add_dependencies(example_jars mandel.jar)

#ViolaJones


set(VIOLAJONES_CLASS_DIR ${CMAKE_BINARY_DIR}/violajones/classes)
set(VIOLAJONES_JAR ${CMAKE_BINARY_DIR}/violajones/violajones.jar)
set(VIOLAJONES_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/examples/violajones/src/main/java)
file(GLOB_RECURSE VIOLAJONES_JAVA_SOURCE_FILES ${VIOLAJONES_SOURCE_ROOT}/*.java)

add_custom_target(violajones.javac DEPENDS hat.jar
   COMMAND ${DOIT}
     mkdir -p ${VIOLAJONES_CLASS_DIR}
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/javac
       --enable-preview --source=23 -g -Xlint:unchecked
       -d ${VIOLAJONES_CLASS_DIR} --class-path ${HAT_JAR}
       --source-path ${VIOLAJONES_SOURCE_ROOT}
       ${VIOLAJONES_JAVA_SOURCE_FILES}
)

add_custom_target(violajones.jar DEPENDS violajones.javac
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/jar --create --no-manifest
       --file  ${VIOLAJONES_JAR}
       -C ${VIOLAJONES_CLASS_DIR} .
       -C ${CMAKE_SOURCE_DIR}/examples/violajones/src/main/resources .
)
add_dependencies(example_jars violajones.jar)

#### We delegate to the cmake config for backends which is clion compatible. 

set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/backends/shared")
set (PTX_BACKEND "${CMAKE_SOURCE_DIR}/backends/ptx")
set (OPENCL_BACKEND "${CMAKE_SOURCE_DIR}/backends/opencl")
set (CUDA_BACKEND "${CMAKE_SOURCE_DIR}/backends/cuda")
set (SPIRV_BACKEND "${CMAKE_SOURCE_DIR}/backends/spirv")
set (MOCK_BACKEND "${CMAKE_SOURCE_DIR}/backends/mock")
add_subdirectory("backends")

#### We can always build the Java artifacts for all of our backends
#### we do need to guard the building of native artifacts
#### with if (CUDAToolkit_FOUND) or if (OpenCL_FOUND) etc because our build
#### machine may not have the appropriate headers or libs to link to
#### Clearly at runtime the java artifacts will need to load the relevant .so/.dylib

##OpenCL Backend
###opencl_backend.jar

set(OPENCL_BACKEND_JAVA_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/backends/opencl/java)
file(GLOB_RECURSE OPENCL_BACKEND_JAVA_SOURCE_FILES ${OPENCL_BACKEND_JAVA_SOURCE_ROOT}/*.java)
set(OPENCL_BACKEND_CLASS_DIR ${CMAKE_BINARY_DIR}/opencl_backend/classes)
set(OPENCL_BACKEND_SERVICES_DIR ${CMAKE_SOURCE_DIR}/backends/opencl/services)
set(OPENCL_BACKEND_JAR ${CMAKE_BINARY_DIR}/opencl_backend/opencl_backend.jar)

add_custom_target(opencl_backend.javac DEPENDS hat.jar
   COMMAND ${DOIT}
     mkdir -p ${OPENCL_BACKEND_CLASS_DIR}
   COMMAND ${DOIT}
     ${JAVA_HOME}/bin/javac
       --enable-preview --source=23 -g -Xlint:unchecked
       -d ${OPENCL_BACKEND_CLASS_DIR} -classpath ${HAT_JAR}
       --source-path ${OPENCL_BACKEND_JAVA_SOURCE_ROOT}
       ${OPENCL_BACKEND_JAVA_SOURCE_FILES}
)

add_custom_target(opencl_backend.jar DEPENDS opencl_backend.javac
   COMMAND ${DOIT}
     ${JAVA_HOME}/bin/jar
       --create --no-manifest
       --file  ${OPENCL_BACKEND_JAR}
       -C ${OPENCL_BACKEND_SERVICES_DIR} .
       -C ${OPENCL_BACKEND_CLASS_DIR} .
)
add_dependencies(backend_jars opencl_backend.jar)
find_package(OpenCL)
if(OPENCL_FOUND)  
    add_dependencies(info_executables opencl_info)
    add_dependencies(backend_libs opencl_backend)
endif()


##CUDA Backend
set(CUDA_BACKEND_JAVA_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/backends/cuda/java)
file(GLOB_RECURSE CUDA_BACKEND_JAVA_SOURCE_FILES ${CUDA_BACKEND_JAVA_SOURCE_ROOT}/*.java)
set(CUDA_BACKEND_CLASS_DIR ${CMAKE_BINARY_DIR}/cuda_backend/classes)
set(CUDA_BACKEND_SERVICES_DIR ${CMAKE_SOURCE_DIR}/backends/cuda/services)
set(CUDA_BACKEND_JAR ${CMAKE_BINARY_DIR}/cuda_backend/cuda_backend.jar)

add_custom_target(cuda_backend.javac DEPENDS hat.jar
   COMMAND ${DOIT}
      mkdir -p ${CUDA_BACKEND_CLASS_DIR}
   COMMAND ${DOIT}
      ${JAVA_HOME}/bin/javac
        --enable-preview --source=23 -g -Xlint:unchecked
        -d ${CUDA_BACKEND_CLASS_DIR} -classpath ${HAT_JAR}
        --source-path ${CUDA_BACKEND_JAVA_SOURCE_ROOT}
        ${CUDA_BACKEND_JAVA_SOURCE_FILES}
)

add_custom_target(cuda_backend.jar DEPENDS cuda_backend.javac
    COMMAND ${DOIT}
      ${JAVA_HOME}/bin/jar
        --create --no-manifest
        --file  ${CUDA_BACKEND_JAR}
        -C ${CUDA_BACKEND_SERVICES_DIR} .
        -C ${CUDA_BACKEND_CLASS_DIR} .
)
add_dependencies(backend_jars cuda_backend.jar)
find_package(CUDAToolkit)

if(CUDAToolkit_FOUND)
    add_dependencies(info_executables cuda_info)
    add_dependencies(backend_libs cuda_backend)
endif()
##End of CUDA Backend

if (EXISTS ${BEEHIVE})
##SPRIV Backend
##spirv_backend.jar

set(SPIRV_BACKEND_JAVA_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/backends/spirv/java)
set(SPIRV_INTEL_BACKEND_JAVA_SOURCE_ROOT ${BABYLON_JDK}/cr-examples/spirv/src/main/java)
set(SPIRV_BEEHIVE_BACKEND_JAVA_SOURCE_ROOT ${BEEHIVE}/lib/src/main/java)
file(GLOB_RECURSE SPIRV_BACKEND_JAVA_SOURCE_FILES ${SPIRV_BACKEND_JAVA_SOURCE_ROOT}/hat/*.java  )
file(GLOB_RECURSE SPIRV_INTEL_BACKEND_JAVA_SOURCE_FILES ${SPIRV_INTEL_BACKEND_JAVA_SOURCE_ROOT}/intel/*.java)
file(GLOB_RECURSE SPIRV_BEEHIVE_BACKEND_JAVA_SOURCE_FILES ${SPIRV_BEEHIVE_BACKEND_JAVA_SOURCE_ROOT}/uk/*.java)
set(SPIRV_BACKEND_CLASS_DIR ${CMAKE_BINARY_DIR}/spirv_backend/classes)
set(SPIRV_BACKEND_SERVICES_DIR ${CMAKE_SOURCE_DIR}/backends/spirv/services)
set(SPIRV_BACKEND_JAR ${CMAKE_BINARY_DIR}/spirv_backend/spirv_backend.jar)

add_custom_target(spirv_backend.javac DEPENDS hat.jar
   COMMAND ${DOIT}
      mkdir -p ${SPIRV_BACKEND_CLASS_DIR}
   COMMAND ${DOIT}
      ${JAVA_HOME}/bin/javac
        --enable-preview --source=23 -g -Xlint:unchecked
        -d ${SPIRV_BACKEND_CLASS_DIR} -classpath ${HAT_JAR}
        --source-path ${SPIRV_BACKEND_JAVA_SOURCE_ROOT}:${SPIRV_INTEL_BACKEND_JAVA_SOURCE_ROOT}:${SPIRV_BEEHIVE_BACKEND_JAVA_SOURCE_ROOT}
        ${SPIRV_BACKEND_JAVA_SOURCE_FILES}
        ${SPIRV_INTEL_BACKEND_JAVA_SOURCE_FILES}
        ${SPIRV_BEEHIVE_BACKEND_JAVA_SOURCE_FILES}
)

add_custom_target(spirv_backend.jar DEPENDS spirv_backend.javac
    COMMAND ${DOIT}
      ${JAVA_HOME}/bin/jar
        --create --no-manifest
        --file  ${SPIRV_BACKEND_JAR}
        -C ${SPIRV_BACKEND_SERVICES_DIR} .
        -C ${SPIRV_BACKEND_CLASS_DIR} .
)
add_dependencies(backend_jars spirv_backend.jar)



add_custom_target(spirv_info_exec DEPENDS spirv_backend.jar spirv_backend
   COMMAND  ${DOIT}
      ${JAVA_HOME}/bin/java
        --enable-preview --enable-native-access=ALL-UNNAMED
        --add-exports=java.base/jdk.internal=ALL-UNNAMED
        -classpath ${SPIRV_BACKEND_JAR}:${HAT_JAR}
        -Djava.library.path=${CMAKE_BINARY_DIR}
        hat.backend.SpirvDeviceInfo
  )
add_dependencies(info_executables spirv_info)
add_dependencies(backend_libs spirv_backend)
## end of SPIRV
else()
   message("BEEHIVE! not found")
endif()

##PTX Backend
##ptx_backend.jar

set(PTX_BACKEND_JAVA_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/backends/ptx/java)
file(GLOB_RECURSE PTX_BACKEND_JAVA_SOURCE_FILES ${PTX_BACKEND_JAVA_SOURCE_ROOT}/*.java)
set(PTX_BACKEND_CLASS_DIR ${CMAKE_BINARY_DIR}/ptx_backend/classes)
set(PTX_BACKEND_SERVICES_DIR ${CMAKE_SOURCE_DIR}/backends/ptx/services)
set(PTX_BACKEND_JAR ${CMAKE_BINARY_DIR}/ptx_backend/ptx_backend.jar)

add_custom_target(ptx_backend.javac DEPENDS hat.jar
   COMMAND ${DOIT}
      mkdir -p ${PTX_BACKEND_CLASS_DIR}
   COMMAND ${DOIT}
      ${JAVA_HOME}/bin/javac
        --enable-preview --source=23 -g -Xlint:unchecked
        -d ${PTX_BACKEND_CLASS_DIR} -classpath ${HAT_JAR}
        --source-path ${PTX_BACKEND_JAVA_SOURCE_ROOT}
        ${PTX_BACKEND_JAVA_SOURCE_FILES}
)

add_custom_target(ptx_backend.jar DEPENDS ptx_backend.javac
    COMMAND ${DOIT}
      ${JAVA_HOME}/bin/jar
        --create --no-manifest
        --file  ${PTX_BACKEND_JAR}
        -C ${PTX_BACKEND_SERVICES_DIR} .
        -C ${PTX_BACKEND_CLASS_DIR} .
)
add_dependencies(backend_jars ptx_backend.jar)
  add_custom_target(ptx_info_exec DEPENDS ptx_backend.jar ptx_backend
      COMMAND  ${DOIT}
        ${JAVA_HOME}/bin/java
          --enable-preview --enable-native-access=ALL-UNNAMED
          --add-exports=java.base/jdk.internal=ALL-UNNAMED
          -classpath ${PTX_BACKEND_JAR}:${HAT_JAR}
          -Djava.library.path=${CMAKE_BINARY_DIR}
          hat.backend.SpirvDeviceInfo
  )
add_dependencies(info_executables ptx_info)   
add_dependencies(backend_libs ptx_backend)    
#endif()

## end of PTX

##MOCK Backend
##mock_backend.jar

set(MOCK_BACKEND_JAVA_SOURCE_ROOT ${CMAKE_SOURCE_DIR}/backends/mock/java)
file(GLOB_RECURSE MOCK_BACKEND_JAVA_SOURCE_FILES ${MOCK_BACKEND_JAVA_SOURCE_ROOT}/*.java)
set(MOCK_BACKEND_CLASS_DIR ${CMAKE_BINARY_DIR}/mock_backend/classes)
set(MOCK_BACKEND_SERVICES_DIR ${CMAKE_SOURCE_DIR}/backends/mock/services)
set(MOCK_BACKEND_JAR ${CMAKE_BINARY_DIR}/mock_backend/mock_backend.jar)

add_custom_target(mock_backend.javac DEPENDS hat.jar
   COMMAND ${DOIT}
      mkdir -p ${MOCK_BACKEND_CLASS_DIR}
   COMMAND ${DOIT}
      ${JAVA_HOME}/bin/javac
        --enable-preview --source=23 -g -Xlint:unchecked
        -d ${MOCK_BACKEND_CLASS_DIR} -classpath ${HAT_JAR}
        --source-path ${MOCK_BACKEND_JAVA_SOURCE_ROOT}
        ${MOCK_BACKEND_JAVA_SOURCE_FILES}
)

add_custom_target(mock_backend.jar DEPENDS mock_backend.javac
    COMMAND ${DOIT}
      ${JAVA_HOME}/bin/jar
        --create --no-manifest
        --file  ${MOCK_BACKEND_JAR}
        -C ${MOCK_BACKEND_SERVICES_DIR} .
        -C ${MOCK_BACKEND_CLASS_DIR} .
)
add_dependencies(backend_jars mock_backend.jar)


  add_custom_target(mock_info_exec DEPENDS mock_backend.jar mock_backend
      COMMAND  ${DOIT}
        ${JAVA_HOME}/bin/java
          --enable-preview --enable-native-access=ALL-UNNAMED
          --add-exports=java.base/jdk.internal=ALL-UNNAMED
          -classpath ${MOCK_BACKEND_JAR}:${HAT_JAR}
          -Djava.library.path=${CMAKE_BINARY_DIR}
          hat.backend.MockDeviceInfo
  )

add_dependencies(info_executables mock_info) 
add_dependencies(backend_libs mock_backend)  
#endif()

## end of MOCK

message("AVAILABLE_BACKEND_JAR = ${AVAILABLE_BACKEND_JAR}")

#  squares executable
add_custom_target(squares_java DEPENDS squares.jar
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${SQUARES_JAR}:${CMAKE_SOURCE_DIR}/backends/shared/services
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}
       squares.Squares
)

add_custom_target(squares_opencl DEPENDS squares.jar opencl_backend.jar opencl_backend
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${SQUARES_JAR}:${OPENCL_BACKEND_JAR}
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}/backends/opencl
       squares.Squares
)

add_custom_target(squares_cuda DEPENDS squares.jar cuda_backend.jar cuda_backend
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${SQUARES_JAR}:${CUDA_BACKEND_JAR}
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}/backends/cuda
       squares.Squares
)
add_custom_target(squares_ptx DEPENDS squares.jar ptx_backend.jar ptx_backend
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${SQUARES_JAR}:${PTX_BACKEND_JAR}
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}/backends/ptx
       squares.Squares
)

#  heal executable
add_custom_target(heal_java DEPENDS heal.jar
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${HEAL_JAR}:${CMAKE_SOURCE_DIR}/backends/shared/services
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}
       heal.HealingBrushDisplay
)
#  view executable
add_custom_target(view_java DEPENDS view.jar
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${VIEW_JAR}:${CMAKE_SOURCE_DIR}/backends/shared/services
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}
       view.Main
)

#  mandel executable
add_custom_target(mandel_java DEPENDS mandel.jar
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${MANDEL_JAR}:${CMAKE_SOURCE_DIR}/backends/shared/services
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}
       mandel.MandelCompute
)

add_custom_target(mandel_opencl DEPENDS mandel.jar opencl_backend.jar opencl_backend
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${MANDEL_JAR}:${OPENCL_BACKEND_JAR}
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}/backends/opencl
       mandel.MandelCompute
)

add_custom_target(mandel_cuda DEPENDS mandel.jar cuda_backend.jar cuda_backend
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${MANDEL_JAR}:${CUDA_BACKEND_JAR}
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}/backends/cuda
       mandel.MandelCompute
)

#  violajones executable
add_custom_target(violajones_java DEPENDS violajones.jar
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${VIOLAJONES_JAR}:${CMAKE_SOURCE_DIR}/backends/shared/services
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}
       violajones.ViolaJonesCompute
)

add_custom_target(violajones_opencl DEPENDS violajones.jar opencl_backend.jar opencl_backend
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${VIOLAJONES_JAR}:${OPENCL_BACKEND_JAR}
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}/backends/opencl
       violajones.ViolaJonesCompute
)

add_custom_target(violajones_cuda DEPENDS violajones.jar cuda_backend.jar cuda_backend
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${VIOLAJONES_JAR}:${CUDA_BACKEND_JAR}
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}/backends/cuda
       violajones.ViolaJonesCompute
)

add_custom_target(violajones_headless_java DEPENDS violajones.jar
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${VIOLAJONES_JAR}:${CMAKE_SOURCE_DIR}/backends/shared/services
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Dheadless=true
       violajones.ViolaJonesCompute
)

add_custom_target(violajones_headless_opencl DEPENDS violajones.jar opencl_backend.jar opencl_backend
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${VIOLAJONES_JAR}:${OPENCL_BACKEND_JAR}
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}/backends/opencl
       -Dheadless=true
       violajones.ViolaJonesCompute
)

add_custom_target(violajones_headless_cuda DEPENDS violajones.jar cuda_backend.jar cuda_backend
   COMMAND  ${DOIT}
     ${JAVA_HOME}/bin/java
       --enable-preview --enable-native-access=ALL-UNNAMED
       --class-path ${HAT_JAR}:${VIOLAJONES_JAR}:${CUDA_BACKEND_JAR}
       --add-exports=java.base/jdk.internal=ALL-UNNAMED
       -Djava.library.path=${CMAKE_BINARY_DIR}/backends/cuda
       -Dheadless=true
       violajones.ViolaJonesCompute
)


add_custom_target(mkjar
   COMMAND ${DOIT}
      ${JAVA_HOME}/bin/jar
      --create
      --no-manifest
      --file  ${CMAKE_BINARY_DIR}/hat.jar
      -C ${CMAKE_SOURCE_DIR} CMakeLists.txt
      -C ${CMAKE_SOURCE_DIR} docs
      -C ${CMAKE_SOURCE_DIR} backends
      -C ${CMAKE_SOURCE_DIR} examples
      -C ${CMAKE_SOURCE_DIR} hat
)

add_custom_target(hat DEPENDS hat.jar backend_jars backend_libs  example_jars info_executables)



