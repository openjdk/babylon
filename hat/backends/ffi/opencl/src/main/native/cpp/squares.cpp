/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

#include "opencl_backend.h"
class KernelContextWithBufferState {
public:
    int x;
    int maxX;
    BufferState bufferState;
};
struct ArgArray_2 {
    int argc;
    u8_t pad12[12];
    KernelArg argv[2];
};


struct S32Array1024WithBufferState {
    int length;
    int array[1024];
    BufferState bufferState;
};
int main(int argc, char **argv) {
    OpenCLBackend openclBackend(0
            | Backend::Config::INFO_BIT
            | Backend::Config::Config::TRACE_CALLS_BIT
            | Backend::Config::Config::TRACE_COPIES_BIT
    );

    //std::string cudaPath =  "/home/gfrost/github/grfrost/babylon-grfrost-fork/hat/squares.cuda";
    OpenCLSource openclSource((char *) R"(
 #define NDRANGE_OPENCL
 #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
 #pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
 #ifndef NULL
 #define NULL 0
 #endif
 #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
 #pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
 typedef char s8_t;
 typedef char byte;
 typedef char boolean;
 typedef unsigned char u8_t;
 typedef short s16_t;
 typedef unsigned short u16_t;
 typedef unsigned int u32_t;
 typedef int s32_t;
 typedef float f32_t;
 typedef long s64_t;
 typedef unsigned long u64_t;
 typedef struct KernelContext_s{
     int x;
     int maxX;
 }KernelContext_t;
 typedef struct S32Array_s{
     int length;
     int array[1];
 }S32Array_t;



 inline int squareit(
     int v
 ){
     return v*v;
 }


 __kernel void squareKernel(
     __global KernelContext_t *global_kc, __global S32Array_t* s32Array
 ){
     KernelContext_t mine;
     KernelContext_t* kc=&mine;
     kc->x=get_global_id(0);
     kc->maxX=global_kc->maxX;
     if(kc->x<kc->maxX){
         int value = s32Array->array[(long)kc->x];
         s32Array->array[(long)kc->x]=squareit(value);
     }
     return;
 }
    )");

    auto *program =openclBackend.compileProgram(openclSource);
    const int maxX = 32;
    auto *kernelContextWithBufferState = bufferOf<KernelContextWithBufferState>("kernelContext");
    kernelContextWithBufferState->x=0;
    kernelContextWithBufferState->maxX=maxX;

    auto *s32Array1024WithBufferState = bufferOf<S32Array1024WithBufferState>("s32ArrayX1024");

    s32Array1024WithBufferState->length=maxX;

    for (int i=0; i < s32Array1024WithBufferState->length; i++){
        s32Array1024WithBufferState->array[i]=i;
    }

    ArgArray_2 args2Array{.argc = 2, .argv={
            {.idx = 0, .variant = '&',.value = {.buffer ={.memorySegment = static_cast<void *>(kernelContextWithBufferState), .sizeInBytes = sizeof(KernelContextWithBufferState), .access = RO_BYTE}}},
            {.idx = 1, .variant = '&',.value = {.buffer ={.memorySegment = static_cast<void *>(s32Array1024WithBufferState), .sizeInBytes = sizeof(S32Array1024WithBufferState), .access = RW_BYTE}}}
    }};
    const auto kernel = program->getOpenCLKernel((char*)"squareKernel");
    kernel->ndrange( reinterpret_cast<ArgArray_s *>(&args2Array));
    for (int i=0; i < s32Array1024WithBufferState->length; i++){
        std::cout << i << " array[" << i << "]=" << s32Array1024WithBufferState->array[i] << std::endl;
    }
}

