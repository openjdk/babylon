cmake_minimum_required(VERSION 3.22.1)
project(shared_backend)

set(CMAKE_CXX_STANDARD 17)

message("SHARED_BACKEND")
if ("${SHARED_BACKEND}EMPTY" STREQUAL "EMPTY")
   set (SHARED_BACKEND "${CMAKE_SOURCE_DIR}/../shared")
   message("SHARED_BACKEND=${SHARED_BACKEND}")
endif()

include_directories(
   ${SHARED_BACKEND}/src/main/native/include
)

add_library(shared_backend SHARED
    ${SHARED_BACKEND}/src/main/native/cpp/shared.cpp
    ${SHARED_BACKEND}/src/main/native/include/shared.h


    ${SHARED_BACKEND}/src/main/native/include/config.h
)
 # ${SHARED_BACKEND}/src/main/native/cpp/strutil.cpp
 # ${SHARED_BACKEND}/src/main/native/include/strutil.h
 #  ${SHARED_BACKEND}/src/main/native/cpp/hex.cpp
 # ${SHARED_BACKEND}/src/main/native/include/hex.h
