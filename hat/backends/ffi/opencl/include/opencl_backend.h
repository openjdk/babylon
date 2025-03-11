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
#else
   #include <CL/cl.h>
   #include <malloc.h>
   #if defined (_WIN32)
       #include "windows.h"
   #endif
#endif

#include "shared.h"

extern void __checkOpenclErrors(cl_int status, const char *file, const int line);

#define checkOpenCLErrors(err)  __checkOpenclErrors (err, __FILE__, __LINE__)

class OpenCLBackend : public Backend {
public:
    class OpenCLConfig{
    public:
        const static  int GPU_BIT =1<<1;
        const static  int CPU_BIT =1<<2;
        const static  int MINIMIZE_COPIES_BIT =1<<3;
        const static  int TRACE_BIT =1<<4;
        const static  int PROFILE_BIT =1<<5;
        const static  int SHOW_CODE_BIT = 1 << 6;
        const static  int SHOW_KERNEL_MODEL_BIT = 1 << 7;
        const static  int SHOW_COMPUTE_MODEL_BIT = 1 <<8;
        const static  int INFO_BIT = 1 <<9;
        const static  int TRACE_COPIES_BIT = 1 <<10;
        const static  int TRACE_SKIPPED_COPIES_BIT = 1 <<11;
        int mode;
        bool gpu;
        bool cpu;
        bool minimizeCopies;
        bool trace;
        bool profile;
        bool showCode;
        bool info;
        bool traceCopies;
         bool traceSkippedCopies;
        OpenCLConfig(int mode);
        virtual ~OpenCLConfig();
    };
    class OpenCLQueue {
    public:
       static const int CopyToDeviceBits= 1<<20;
       static const int CopyFromDeviceBits= 1<<21;
       static const int NDRangeBits =1<<22;
       static const int StartComputeBits= 1<<23;
       static const int EndComputeBits= 1<<24;
       static const int EnterKernelDispatchBits= 1<<25;
       static const int LeaveKernelDispatchBits= 1<<26;
       static const int HasConstCharPtrArgBits = 1<<27;
       static const int hasIntArgBits = 1<<28;
       OpenCLBackend *openclBackend;
       size_t eventMax;
       cl_event *events;
       int *eventInfoBits;
       const char **eventInfoConstCharPtrArgs;
       size_t eventc;
       cl_command_queue command_queue;

       OpenCLQueue(OpenCLBackend *openclBackend);
       cl_event *eventListPtr();
       cl_event *nextEventPtr();
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
       virtual ~OpenCLQueue();
    };

    class OpenCLProgram : public Backend::Program {
        public:
        class OpenCLKernel : public Backend::Program::Kernel {
            public:
            class OpenCLBuffer : public Backend::Program::Kernel::Buffer {
            public:
                cl_mem clMem;
                void copyToDevice();
                void copyFromDevice();
                OpenCLBuffer(Backend::Program::Kernel *kernel, Arg_s *arg);
                virtual ~OpenCLBuffer();
            };
        private:
            const char *name;
            cl_kernel kernel;
        public:
            OpenCLKernel(Backend::Program *program, char* name,cl_kernel kernel);
            ~OpenCLKernel();
            long ndrange( void *argArray);
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
    cl_device_id device_id;
    OpenCLConfig openclConfig;
    OpenCLQueue openclQueue;
    OpenCLBackend(int mode, int platform, int device);
    ~OpenCLBackend();
    int getMaxComputeUnits();
    bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength);
    void info();
    void computeStart();
    void computeEnd();
    void dumpSled(std::ostream &out,void *argArray);
    char *dumpSchema(std::ostream &out,int depth, char *ptr, void *data);
    long compileProgram(int len, char *source);
    char *strInfo(cl_device_info device_info);
    cl_int cl_int_info( cl_device_info device_info);
    cl_ulong cl_ulong_info( cl_device_info device_info);
    size_t size_t_info( cl_device_info device_info);
    char *strPlatformInfo(cl_platform_info platform_info);
public:
    static const char *errorMsg(cl_int status);
};
extern "C" long getOpenCLBackend(int mode, int platform, int device, int unused);