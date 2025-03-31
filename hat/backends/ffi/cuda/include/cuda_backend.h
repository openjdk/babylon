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
#define CUDA_TYPES
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
#include <cuda.h>
#include <builtin_types.h>
#include <cuda_runtime_api.h>
#define CUDA_TYPES

#include "shared.h"

#include <fstream>

#include<vector>

//extern void __checkCudaErrors(CUresult err, const char *file, const int line);

//#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)
class Text {
public:
    size_t len;
    char *text;
    bool isCopy;

    Text(size_t len, char *text, bool isCopy);
    Text(char *text, bool isCopy);
    Text(size_t len);
    virtual ~Text();
};

class PtxSource: public Text  {
public:
    PtxSource();
    PtxSource(size_t len);
    PtxSource(char *text);
    ~PtxSource() = default;
    static PtxSource *nvcc(const char *cudaSource, size_t len);
};
class CudaSource:public Text  {
public:
    CudaSource(size_t len);
    CudaSource(char* text);
    ~CudaSource() = default;
};
class Log:public Text  {
public:
    Log(size_t len);
    Log(char* text);
    ~Log()  = default;
};


class CudaBackend : public Backend {
public:
class CudaConfig : public Backend::Config{
    public:
        CudaConfig(int mode);
         ~CudaConfig()=default;
    };
class CudaQueue: public Backend::Queue {
    public:
        cudaStream_t cudaStream;
        CudaQueue(Backend *backend);
        void showEvents(int width);
        void wait();
        void release();
        void computeStart();
        void computeEnd();
        void inc(int bits);
        void inc(int bits, const char *arg);
        void inc(int bits, int arg);
        void marker(int bits);
        void marker(int bits, const char *arg);
        void marker(int bits, int arg);
        void markAsCopyToDeviceAndInc(int argn);
        void markAsCopyFromDeviceAndInc(int argn);
        void markAsNDRangeAndInc();
        void markAsStartComputeAndInc();
        void markAsEndComputeAndInc();
        void markAsEnterKernelDispatchAndInc();
        void markAsLeaveKernelDispatchAndInc();
        virtual ~CudaQueue();
    };

    class CudaBuffer : public Backend::Buffer {
    public:
        CUdeviceptr devicePtr;
        CudaBuffer(Backend *backend,Arg_s *arg, BufferState_s *bufferStateS);
        void copyToDevice();
        void copyFromDevice();
        virtual ~CudaBuffer();
    };

    class CudaModule : public Backend::CompilationUnit {
        class CudaKernel : public Backend::CompilationUnit::Kernel {

        private:
            CUfunction function;

        public:
            CudaKernel(Backend::CompilationUnit *program, char* name, CUfunction function);
            ~CudaKernel() override;
            long ndrange( void *argArray);
        };

    private:
        CUmodule module;
        CudaSource cudaSource;
        PtxSource ptxSource;
        Log log;

    public:

        CudaModule(Backend *backend, char *cudaSrc,   char *log, bool ok, CUmodule module);
        ~CudaModule();
        long getKernel(int nameLen, char *name);
        bool programOK();
    };

private:
    CUdevice device;
    CUcontext context;

    CudaConfig cudaConfig;
    CudaQueue cudaQueue;
public:
    void info();

    long compile(int len, char *source);
    void computeStart();
    void computeEnd();
    bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength);

    CudaBackend(int mode);

    CudaBackend();

    ~CudaBackend();
};
extern "C" long getCudaBackend(int mode);

