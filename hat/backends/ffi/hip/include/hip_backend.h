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
#pragma once
#define HIP_TYPES
#ifdef __APPLE__

#define LongUnsignedNewline "%llu\n"
#define Size_tNewline "%lu\n"
#define LongHexNewline "(0x%llx)\n"
#define alignedMalloc(size, alignment) memalign(alignment, size)
#define SNPRINTF snprintf
#else

#include <malloc.h>

#define LongHexNewline "(0x%lx)\n"
#define LongUnsignedNewline "%lu\n"
#define Size_tNewline "%lu\n"
#if defined (_WIN32)
#include "windows.h"
#define alignedMalloc(size, alignment) _aligned_malloc(size, alignment)
#define SNPRINTF _snprintf
#else
#define alignedMalloc(size, alignment) memalign(alignment, size)
#define SNPRINTF  snprintf
#endif
#endif

#include <iostream>
#include <hip/hip_runtime.h>
//#include <builtin_types.h>

#include "shared.h"

#include <fstream>

#include<vector>
#include <thread>

/*
struct WHERE{
    const char* f;
    int l;
    cudaError_enum e;
    const char* t;
    void report() const{
        if (e == CUDA_SUCCESS){
           // std::cout << t << "  OK at " << f << " line " << l << std::endl;
        }else {
            const char *buf;
            cuGetErrorName(e, &buf);
            std::cerr << t << " CUDA error = " << e << " " << buf <<std::endl<< "      " << f << " line " << l << std::endl;
            exit(-1);
        }
    }
};

*/
class PtxSource: public Text  {
public:
    PtxSource();
    PtxSource(size_t len);
    PtxSource(size_t len, char *text);
    PtxSource(char *text);
    ~PtxSource() = default;
   // static PtxSource *nvcc(const char *cudaSource, size_t len);
};

class HipSource: public Text  {
public:
    HipSource();
    HipSource(size_t len);
    HipSource(size_t len, char *text);
    HipSource(char *text);
    ~HipSource() = default;

};

class HipBackend : public Backend {
public:
class HipQueue: public Backend::Queue {
    public:
         std::thread::id streamCreationThread;
        //CUstream cuStream;
        hipStream_t cuStream;
        explicit HipQueue(Backend *backend);
        void init();
         void wait() override;

         void release() override;

         void computeStart() override;

         void computeEnd() override;

         void copyToDevice(Buffer *buffer) override;

         void copyFromDevice(Buffer *buffer) override;

        void dispatch(KernelContext *kernelContext, CompilationUnit::Kernel *kernel) override;

        ~HipQueue() override;

};

  class HipBuffer : public Backend::Buffer {
    public:
        //CUdeviceptr devicePtr;
       hipDevice_t devicePtr; 
        HipBuffer(Backend *backend, BufferState *bufferState);
        ~HipBuffer() override;
    };

    class HipProgram : public Backend::CompilationUnit {
        class HipKernel : public Backend::CompilationUnit::Kernel {


        private:
            hipFunction_t kernel;
            hipStream_t hipStream;
        public:
            HipKernel(Backend::CompilationUnit *program, char* name, hipFunction_t kernel);

            ~HipKernel() override;

            //long ndrange( void *argArray);
        };

    private:
        hipModule_t module;
        HipSource hipSource;
        PtxSource ptxSource;
        Log log;

    public:
        HipProgram(Backend *backend, Backend::CompilationUnit::BuildInfo *buildInfo, hipModule_t module);
        ~HipProgram();

        long getHipKernel(char *name);
        long getHipKernel(int nameLen, char *name);

        bool programOK();
    };

private:
    hipDevice_t device;
    hipCtx_t context;
public:
    void info();

     HipBackend(int mode);
    HipBackend();
    ~HipBackend();

    int getMaxComputeUnits();

};

