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
    class OpenCLConfig : public Backend::Config{
    public:
        OpenCLConfig(int mode);
        virtual ~OpenCLConfig();
    };
    class OpenCLQueue : public Backend::Queue {
    public:
       cl_command_queue command_queue;
       cl_event *events;
       OpenCLQueue(Backend *backend);
       cl_event *eventListPtr();
       cl_event *nextEventPtr();

        virtual void showEvents(int width);
        virtual void wait();
        virtual void release();
        virtual void computeStart();
        virtual void computeEnd();
        virtual void inc(int bits);
        virtual void inc(int bits, const char *arg);
        virtual  void inc(int bits, int arg);
        virtual void marker(int bits);
        virtual void marker(int bits, const char *arg);
        virtual void marker(int bits, int arg);
        virtual void markAsCopyToDeviceAndInc(int argn);
        virtual  void markAsCopyFromDeviceAndInc(int argn);
        virtual void markAsNDRangeAndInc();
        virtual void markAsStartComputeAndInc();
        virtual  void markAsEndComputeAndInc();
        virtual  void markAsEnterKernelDispatchAndInc();
        virtual void markAsLeaveKernelDispatchAndInc();

       virtual ~OpenCLQueue();
    };
    class OpenCLBuffer : public Backend::Buffer {
            public:
                cl_mem clMem;
                void copyToDevice();
                void copyFromDevice();
                bool shouldCopyToDevice(Arg_s *arg);
                bool shouldCopyFromDevice(Arg_s *arg);
                OpenCLBuffer(Backend *backend, Arg_s *arg, BufferState_s *bufferState);
                virtual ~OpenCLBuffer();
            };
    class OpenCLProgram : public Backend::CompilationUnit {
        public:
        class OpenCLKernel : public Backend::CompilationUnit::Kernel {
            public:

        private:
            const char *name;
            cl_kernel kernel;
        public:
            OpenCLKernel(Backend::CompilationUnit *compilationUnit, char* name,cl_kernel kernel);
            ~OpenCLKernel();
            long ndrange( void *argArray);
        };
    private:
        cl_program program;
    public:
        OpenCLProgram(Backend *backend, char *src, char *log, bool ok, cl_program program);
        ~OpenCLProgram();
        long getKernel(int nameLen, char *name);
        bool compilationUnitOK();
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
    long compile(int len, char *source);

public:
    static const char *errorMsg(cl_int status);
};
extern "C" long getOpenCLBackend(int configBits);


struct PlatformInfo{
    struct DeviceInfo{
      OpenCLBackend *openclBackend;
      cl_int maxComputeUnits;
      cl_int maxWorkItemDimensions;
      cl_device_type deviceType;
      size_t maxWorkGroupSize;
      cl_ulong globalMemSize;
      cl_ulong localMemSize;
      cl_ulong maxMemAllocSize;
      char *profile;
      char *deviceVersion;
      size_t *maxWorkItemSizes ;
      char *driverVersion;
      char *cVersion;
      char *name;
      char *extensions;
      char *builtInKernels;
      char *deviceTypeStr;
      DeviceInfo(OpenCLBackend *openclBackend);
      ~DeviceInfo();
    };
  OpenCLBackend *openclBackend;
  char *versionName;
  char *vendorName;
  char *name;
  DeviceInfo deviceInfo;

  PlatformInfo(OpenCLBackend *openclBackend);
  ~PlatformInfo();
};


