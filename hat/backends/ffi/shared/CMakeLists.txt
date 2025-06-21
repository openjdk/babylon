cmake_minimum_required(VERSION 3.22.1)
project(shared_backend)

set(CMAKE_CXX_STANDARD 14)

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
    ${SHARED_BACKEND}/src/main/native/cpp/buffer.cpp
    ${SHARED_BACKEND}/src/main/native/include/buffer.h
    ${SHARED_BACKEND}/src/main/native/cpp/schema_cursor.cpp
    ${SHARED_BACKEND}/src/main/native/include/schema_cursor.h
    ${SHARED_BACKEND}/src/main/native/cpp/buffer_cursor.cpp
    ${SHARED_BACKEND}/src/main/native/include/buffer_cursor.h
    ${SHARED_BACKEND}/src/main/native/cpp/hex.cpp
    ${SHARED_BACKEND}/src/main/native/include/hex.h
    ${SHARED_BACKEND}/src/main/native/cpp/json.cpp
    ${SHARED_BACKEND}/src/main/native/include/json.h
    ${SHARED_BACKEND}/src/main/native/cpp/fsutil.cpp
    ${SHARED_BACKEND}/src/main/native/include/fsutil.h
    ${SHARED_BACKEND}/src/main/native/cpp/strutil.cpp
    ${SHARED_BACKEND}/src/main/native/include/strutil.h
        ${SHARED_BACKEND}/src/main/native/include/config.h
)

add_executable(schemadump
    ${SHARED_BACKEND}/src/main/native/cpp/schema.cpp
    ${SHARED_BACKEND}/src/main/native/cpp/schemadump.cpp
    ${SHARED_BACKEND}/src/main/native/cpp/schema_cursor.cpp
    ${SHARED_BACKEND}/src/main/native/include/schema_cursor.h
    ${SHARED_BACKEND}/src/main/native/include/schema.h
)
