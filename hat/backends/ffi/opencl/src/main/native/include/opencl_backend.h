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
// The following looks like it is not used (at least to CLION) but it is. ;) don't remove
#define CL_TARGET_OPENCL_VERSION 120
#ifdef __APPLE__
#include <opencl/opencl.h>
#else
#include <CL/cl.h>
#include <malloc.h>
#if defined (_WIN32)
       #include "windows.h"
#endif
#endif
#include "shared.h"

class OpenCLSource final : public Text {
public:
    explicit OpenCLSource(size_t len);

    explicit OpenCLSource(char *text);

    OpenCLSource();

    ~OpenCLSource() override = default;
};

extern void __checkOpenclErrors(cl_int status, const char *functionName, const char *file, const int line);

#define OPENCL_CHECK(err, functionName) __checkOpenclErrors (err, functionName, __FILE__, __LINE__)

class OpenCLBackend final : public Backend {
public:
    class OpenCLBuffer final : public Backend::Buffer {
    public:
        cl_mem clMem;

        OpenCLBuffer(Backend *backend, BufferState *bufferState);

        ~OpenCLBuffer() override;
    };

    class OpenCLProgram final : public Backend::CompilationUnit {
    public:
        class OpenCLKernel : public Backend::CompilationUnit::Kernel {
        public:
            cl_kernel kernel;

            OpenCLKernel(Backend::CompilationUnit *compilationUnit, char *name, cl_kernel kernel);

            bool setArg(KernelArg *arg) override;

            bool setArg(KernelArg *arg, Buffer *buffer) override;

            ~OpenCLKernel() override;
        };

    private:
        cl_program program;

    public:
        OpenCLProgram(Backend *backend, char *src, char *log, bool ok, cl_program program);

        ~OpenCLProgram() override;

        OpenCLKernel *getOpenCLKernel(char *name);

        OpenCLKernel *getOpenCLKernel(int nameLen, char *name);

        CompilationUnit::Kernel *getKernel(int nameLen, char *name) override;
    };

    class OpenCLQueue : public Backend::ProfilableQueue {
    public:
        cl_command_queue command_queue;
        cl_event *events;

        cl_event *eventListPtr() const;

        cl_event *nextEventPtr() const;

        explicit OpenCLQueue(Backend *backend);

        void wait() override;

        void release() override;

        void computeStart() override;

        void computeEnd() override;

        void showEvents(int width) override;

        void inc(int bits) override;

        void inc(int bits, const char *arg) override;

        void marker(int bits) override;

        void marker(int bits, const char *arg) override;

        void markAsStartComputeAndInc() override;

        void markAsEndComputeAndInc() override;

        void markAsEnterKernelDispatchAndInc() override;

        void markAsLeaveKernelDispatchAndInc() override;

        void copyToDevice(Buffer *buffer) override;

        void copyFromDevice(Buffer *buffer) override;


        void dispatch(KernelContext *kernelContext, CompilationUnit::Kernel *kernel) override;

        ~OpenCLQueue() override;
    };

    cl_platform_id platform_id;
    cl_context context;
    cl_device_id device_id;

    explicit OpenCLBackend(int configBits);

    ~OpenCLBackend() override;

    OpenCLBuffer *getOrCreateBuffer(BufferState *bufferState) override;

    OpenCLProgram *compileProgram(OpenCLSource &openclSource);

    OpenCLProgram *compileProgram(const OpenCLSource *openclSource);

    OpenCLProgram *compileProgram(int len, char *source);

    CompilationUnit *compile(int len, char *source) override;

    void computeStart() override;

    void computeEnd() override;

    bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) override;

    void info() override;

    static const char *errorMsg(cl_int status);
};


struct PlatformInfo {
    struct DeviceInfo {
        OpenCLBackend *openclBackend;
        cl_int maxComputeUnits;
        cl_int maxWorkItemDimensions;
        cl_device_type deviceType{};
        size_t maxWorkGroupSize;
        cl_ulong globalMemSize;
        cl_ulong localMemSize;
        cl_ulong maxMemAllocSize;
        char *profile;
        char *deviceVersion;
        size_t *maxWorkItemSizes;
        char *driverVersion;
        char *cVersion;
        char *name;
        char *extensions;
        char *builtInKernels;
        char *deviceTypeStr;

        explicit DeviceInfo(OpenCLBackend *openclBackend);

        ~DeviceInfo();
    };

    OpenCLBackend *openclBackend;
    char *versionName;
    char *vendorName;
    char *name;
    DeviceInfo deviceInfo;

    explicit PlatformInfo(OpenCLBackend *openclBackend);

    ~PlatformInfo();
};
