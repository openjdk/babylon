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
#define PTX_TYPES
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

class Ptx {
public:
    size_t len;
    char *text;

    Ptx(size_t len);

    ~Ptx();

    static Ptx *nvcc(const char *ptxSource, size_t len);
};

class PtxBackend : public Backend {
public:
    class PtxConfig : public Backend::Config {
    public:
        boolean gpu;
    };

    class PtxProgram : public Backend::Program {
        class PtxKernel : public Backend::Program::Kernel {
            class PtxBuffer : public Backend::Program::Kernel::Buffer {
            public:
                CUdeviceptr devicePtr;

                PtxBuffer(Backend::Program::Kernel *kernel, Arg_s *arg);

                void copyToDevice();

                void copyFromDevice();

                virtual ~PtxBuffer();
            };

        private:
            CUfunction function;
            cudaStream_t cudaStream;
        public:
            PtxKernel(Backend::Program *program, char* name, CUfunction function);

            ~PtxKernel() override;

            long ndrange( void *argArray);
        };

    private:
        CUmodule module;
        Ptx *ptx;

    public:
        PtxProgram(Backend *backend, BuildInfo *buildInfo, Ptx *ptx, CUmodule module);

        ~PtxProgram();

        long getKernel(int nameLen, char *name);

        bool programOK();
    };

private:
    CUdevice device;
    CUcontext context;
public:

    PtxBackend(PtxConfig *config, int configSchemaLen, char *configSchema);

    PtxBackend();

    ~PtxBackend();

    int getMaxComputeUnits();

    void info();

    long compileProgram(int len, char *source);

};

