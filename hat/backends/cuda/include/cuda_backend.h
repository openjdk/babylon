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

#define CUDA_TYPES

#include "shared.h"

#include <fstream>

#include<vector>

extern void __checkCudaErrors(CUresult err, const char *file, const int line);

#define checkCudaErrors(err)  __checkCudaErrors (err, __FILE__, __LINE__)

class Ptx {
public:
    size_t len;
    char *text;

    Ptx(size_t len);

    ~Ptx();

    static Ptx *nvcc(const char *cudaSource, size_t len);
};

class CudaBackend : public Backend {
public:
    class CudaConfig : public Backend::Config {
    public:
        boolean gpu;
    };

    class CudaProgram : public Backend::Program {
        class CudaKernel : public Backend::Program::Kernel {
            class CudaBuffer : public Backend::Program::Kernel::Buffer {
            public:
                CUdeviceptr devicePtr;

                CudaBuffer(Backend::Program::Kernel *kernel, Arg_t *arg);

                void copyToDevice();

                void copyFromDevice();

                virtual ~CudaBuffer();
            };

        private:
            CUfunction function;
        public:
            CudaKernel(Backend::Program *program, std::string name, CUfunction function);

            ~CudaKernel();

            long ndrange(int range, void *argArray);
        };

    private:
        CUmodule module;
        Ptx *ptx;

    public:
        CudaProgram(Backend *backend, BuildInfo *buildInfo, Ptx *ptx, CUmodule module);

        ~CudaProgram();

        long getKernel(int nameLen, char *name);

        bool programOK();
    };

private:
    CUdevice device;
    CUcontext context;
public:

    CudaBackend(CudaConfig *config, int configSchemaLen, char *configSchema);

    CudaBackend();

    ~CudaBackend();

    int getMaxComputeUnits();

    void info();

    long compileProgram(int len, char *source);

    static const char *errorMsg(CUresult status);

};

