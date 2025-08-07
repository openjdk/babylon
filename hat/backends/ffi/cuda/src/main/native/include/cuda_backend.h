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

#include "shared.h"

#include <fstream>
#include <thread>

struct WHERE{
    const char* f;
    int l;
    cudaError_enum e;
    const char* t;
    void report() const {
        if (e != CUDA_SUCCESS){
            const char *buf;
            cuGetErrorName(e, &buf);
            std::cerr << t << " CUDA error = " << e << " " << buf <<std::endl<< "      " << f << " line " << l << std::endl;
            exit(-1);
        }
    }
};

#define CUDA_CHECK(err, functionName) { \
    WHERE{.f =__FILE__, \
          .l=__LINE__, \
          .e = err, \
          .t = functionName \
         }.report(); \
}

class PtxSource final : public Text  {
public:
    PtxSource();
    explicit PtxSource(size_t len);
    PtxSource(size_t len, char *text);
    PtxSource(size_t len, char *text, bool isCopy);
    explicit PtxSource(char *text);
    ~PtxSource() override = default;
};

class CudaSource final :public Text  {
public:
    CudaSource(size_t len, char *text, bool isCopy);
    explicit CudaSource(size_t len);
    explicit CudaSource(char* text);
    CudaSource();
    ~CudaSource() override = default;
};

class CudaBackend final : public Backend {
public:
class CudaQueue final : public Backend::Queue {
    public:
        std::thread::id streamCreationThread;
        CUstream cuStream;
        explicit CudaQueue(Backend *backend);
        void init();
        void wait() override;

         void release() override;

         void computeStart() override;

         void computeEnd() override;

         void copyToDevice(Buffer *buffer) override;

         void copyFromDevice(Buffer *buffer) override;

        int estimateThreadsPerBlock(int dimensions);

        void dispatch(KernelContext *kernelContext, CompilationUnit::Kernel *kernel) override;

        ~CudaQueue() override;
};

    class CudaBuffer final : public Buffer {
    public:
        CUdeviceptr devicePtr;
        CudaBuffer(Backend *backend, BufferState *bufferState);
        ~CudaBuffer() override;
    };

    class CudaModule final : public CompilationUnit {
        CUmodule module;
        CudaSource cudaSource;
        PtxSource ptxSource;
        Log log;

    public:
        class CudaKernel final : public Kernel {

        public:
            bool setArg(KernelArg *arg) override;
            bool setArg(KernelArg *arg, Buffer *buffer) override;
            CudaKernel(Backend::CompilationUnit *program, char* name, CUfunction function);
            ~CudaKernel() override;
            static CudaKernel * of(long kernelHandle);
            static CudaKernel * of(Backend::CompilationUnit::Kernel *kernel);

            CUfunction function;
            void *argslist[100]{};
        };
        CudaModule(Backend *backend, char *cudaSrc,   char *log, bool ok, CUmodule module);
        ~CudaModule() override;
        static CudaModule * of(long moduleHandle);
        //static CudaModule * of(CompilationUnit *compilationUnit);
        Kernel *getKernel(int nameLen, char *name) override;
        CudaKernel *getCudaKernel(char *name);
        CudaKernel *getCudaKernel(int nameLen, char *name);
        bool programOK();
    };

private:
    CUresult initStatus;
    CUdevice device;
    CUcontext context;
public:
    void info() override;
    CudaModule * compile(const CudaSource *cudaSource);
    CudaModule * compile(const CudaSource &cudaSource);
    CudaModule * compile(const PtxSource *ptxSource);
    CudaModule * compile(const PtxSource &ptxSource);
    static PtxSource *nvcc(const CudaSource *cudaSource);
    CompilationUnit * compile(int len, char *source) override;
    void computeStart() override;
    void computeEnd() override;
    CudaBuffer * getOrCreateBuffer(BufferState *bufferState) override;
    bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) override;

    explicit CudaBackend(int mode);

    ~CudaBackend() override;
    static CudaBackend * of(long backendHandle);
    static CudaBackend * of(Backend *backend);
};
