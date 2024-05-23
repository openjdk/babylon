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
#define CL_TARGET_OPENCL_VERSION 120

#ifdef __APPLE__
#include <opencl/opencl.h>
#define LongUnsignedNewline "%llu\n"
#define Size_tNewline "%lu\n"
#define LongHexNewline "(0x%llx)\n"
#define alignedMalloc(size, alignment) memalign(alignment, size)
#define SNPRINTF snprintf
#else
#include <CL/cl.h>
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
#include "shared.h"

class OpenCLBackend : public Backend {
public:
    class OpenCLConfig : public Backend::Config {
    public:
        boolean gpu;
        boolean junk;
    };

    class OpenCLProgram : public Backend::Program {
        class OpenCLKernel : public Backend::Program::Kernel {

        class OpenCLBuffer : public Backend::Program::Kernel::Buffer {
            public:
                void *ptr;
                size_t sizeInBytes;
                cl_mem clMem;
                void copyToDevice();
                void copyFromDevice();
                OpenCLBuffer(void *ptr, size_t sizeInBytes, cl_context context);

                virtual ~OpenCLBuffer();
            };

        private:
            cl_kernel kernel;
        public:
            OpenCLKernel(Backend::Program *program, cl_kernel kernel);

            ~OpenCLKernel();

            long ndrange(int range, void *argArray);
        };

    private:
        cl_program program;

    public:
        OpenCLProgram(Backend *backend, BuildInfo *buildInfo, cl_program program);

        ~OpenCLProgram();

        long getKernel(int nameLen, char *name);

        bool programOK();
    };

public:
    cl_platform_id platform_id;
    cl_context context;
    cl_command_queue command_queue;
    cl_device_id device_id;
    size_t eventMax;
    cl_event *events;
    size_t eventc;

    OpenCLBackend();

    OpenCLBackend(OpenCLConfig *config, int configSchemaLen, char *configSchema);

    ~OpenCLBackend();

    int getMaxComputeUnits();

    void info();

    long compileProgram(int len, char *source);

protected:

    void allocEvents(int max);

    void releaseEvents();

    void waitForEvents();

    void showEvents(int width);

public:
    static const char *errorMsg(cl_int status);
};

