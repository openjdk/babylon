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
    // These must sync with hat/backend/ffi/Mode.java
        // Bits 0-3 select platform id 0..5
        // Bits 4-7 select device id 0..15
        const static  int START_BIT_IDX = 16;
        const static  int MINIMIZE_COPIES_BIT =1<<START_BIT_IDX;
        const static  int TRACE_BIT =1<<17;
        const static  int PROFILE_BIT =1<<18;
        const static  int SHOW_CODE_BIT = 1 << 19;
        const static  int SHOW_KERNEL_MODEL_BIT = 1 << 20;
        const static  int SHOW_COMPUTE_MODEL_BIT = 1 <<21;
        const static  int INFO_BIT = 1<<22;
        const static  int TRACE_COPIES_BIT = 1 <<23;
        const static  int TRACE_SKIPPED_COPIES_BIT = 1 <<24;
        const static  int TRACE_ENQUEUES_BIT = 1 <<25;
        const static  int TRACE_CALLS_BIT = 1 <<26;
        const static  int SHOW_WHY_BIT = 1 <<27;
        const static  int USE_STATE_BIT = 1 <<28;
        const static  int SHOW_STATE_BIT = 1 <<29;
        const static  int END_BIT_IDX = 30;

        const static  char *bitNames[]; // See below for out of line definition
        int configBits;
        bool minimizeCopies;
        bool alwaysCopy;
        bool trace;
        bool profile;
        bool showCode;
        bool info;
        bool traceCopies;
        bool traceSkippedCopies;
        bool traceEnqueues;
        bool traceCalls;
        bool showWhy;
        bool useState;
        bool showState;
        int platform; //0..15
        int device; //0..15
        OpenCLConfig(int mode);
        virtual ~OpenCLConfig();
    };
    class OpenCLQueue {
    public:
       const static  int START_BIT_IDX =20;
       static const int CopyToDeviceBits= 1<<START_BIT_IDX;
       static const int CopyFromDeviceBits= 1<<21;
       static const int NDRangeBits =1<<22;
       static const int StartComputeBits= 1<<23;
       static const int EndComputeBits= 1<<24;
       static const int EnterKernelDispatchBits= 1<<25;
       static const int LeaveKernelDispatchBits= 1<<26;
       static const int HasConstCharPtrArgBits = 1<<27;
       static const int hasIntArgBits = 1<<28;
       const static  int END_BIT_IDX = 27;
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
                BufferState_s * bufferState;
                void copyToDevice();
                void copyFromDevice();
                bool shouldCopyToDevice(Arg_s *arg);
                bool shouldCopyFromDevice(Arg_s *arg);
                OpenCLBuffer(Backend::Program::Kernel *kernel, Arg_s *arg, BufferState_s *bufferState);
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
    OpenCLBackend(int configBits);
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
extern "C" long getOpenCLBackend(int configBits);
#ifdef opencl_backend_cpp
const  char *OpenCLBackend::OpenCLConfig::bitNames[] = {
              "MINIMIZE_COPIES",
              "TRACE",
              "PROFILE",
              "SHOW_CODE",
              "SHOW_KERNEL_MODEL",
              "SHOW_COMPUTE_MODEL",
              "INFO",
              "TRACE_COPIES",
              "TRACE_SKIPPED_COPIES",
              "TRACE_ENQUEUES",
              "TRACE_CALLS"
              "SHOW_WHY_BIT",
              "USE_STATE_BIT",
              "SHOW_STATE_BIT"
        };
#endif