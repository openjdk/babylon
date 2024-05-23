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

struct Ptx {
    size_t len;
    char *text;

    Ptx(size_t len)
            : len(len), text(len > 0 ? new char[len] : nullptr) {}

    Ptx() {
        if (len > 0 && text != nullptr) {
            delete[] text;
        }
    }

    static Ptx *nvcc(const char *cudaSource, size_t len) {
        Ptx *ptx = nullptr;
        const char *cudaPath = "./tmp2.cu";
        const char *ptxPath = "./tmp2.ptx";
        const char *stderrPath = "./tmp2.stderr";
        const char *stdoutPath = "./tmp2.stdout";
        // we are going to fork exec nvcc
        int pid;
        if ((pid = fork()) == 0) {
            std::ofstream cuda;
            cuda.open(cudaPath);
            cuda.write(cudaSource, len);
            cuda.close();

            const char *path = "/usr/local/cuda-12.2/bin/nvcc";
            // const char *argv[]{"nvcc", "-v", "-ptx", cudaPath, "-o", ptxPath, nullptr};
            const char *argv[]{"nvcc", "-ptx", cudaPath, "-o", ptxPath, nullptr};
            // We are the child so exec nvcc.
            //close(1); // stdout
            //close(2); // stderr
            //open(stderrPath, O_RDWR); // stdout
            //open(stdoutPath, O_RDWR); // stderr
            execvp(path, (char *const *) argv);
        } else if (pid < 0) {
            // fork failed.
            std::cerr << "fork of nvcc failed" << std::endl;
            std::exit(1);
        } else {
            std::cerr << "fork suceeded" << std::endl;
            std::ifstream ptxStream(ptxPath);
            ptxStream.seekg(0, ptxStream.end);
            size_t ptxLen = ptxStream.tellg();
            if (ptxLen > 0) {
                ptx = new Ptx(ptxLen + 1);
                ptxStream.seekg(0, ptxStream.beg);
                ptxStream.read(ptx->text, ptx->len);
                ptx->text[ptx->len] = '\0';
            }
            ptxStream.close();
        }
        std::cout << "returning PTX" << std::endl;
        return ptx;
    }
};

class CudaBackend : public Backend {
public:
    class CudaConfig : public Backend::Config {
    public:
        boolean gpu;
    };

    class CudaProgram : public Backend::Program {
        class CudaKernel : public Backend::Program::Kernel {
            class CudaBuffer {
            public:
                void *ptr;
                size_t sizeInBytes;
                CUdeviceptr devicePtr;

                CudaBuffer(void *ptr, size_t sizeInBytes);

                virtual ~CudaBuffer();
            };

        private:
            CUfunction function;
        public:
            CudaKernel(Backend::Program *program, CUfunction function);

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

};

