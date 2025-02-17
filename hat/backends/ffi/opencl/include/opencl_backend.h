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

extern void __checkOpenclErrors(cl_int status, const char *file, const int line);

#define checkOpenCLErrors(err)  __checkOpenclErrors (err, __FILE__, __LINE__)

class OpenCLBackend : public Backend {
public:
    class OpenCLConfig : public Backend::Config {
    public:
        const static  int GPU_BIT =1<<1;
        const static  int CPU_BIT =1<<2;
        const static  int MINIMIZE_COPIES_BIT =1<<3;
        const static  int TRACE_BIT =1<<4;
        bool gpu;
        bool minimizeCopies;
        bool trace;
        OpenCLConfig(int mode):
           Backend::Config(mode),
           gpu((mode&GPU_BIT)==GPU_BIT),
           minimizeCopies((mode&MINIMIZE_COPIES_BIT)==MINIMIZE_COPIES_BIT),
           trace((mode&TRACE_BIT)==TRACE_BIT){
           printf("native gpu %d\n",gpu);
           printf("native minimizeCopies %d\n", minimizeCopies);
           printf("native trace %d\n", trace);
        }
        virtual ~OpenCLConfig(){}
    };
    class OpenCLQueue {
    public:
       size_t eventMax;
       cl_event *events;
       size_t eventc;
       cl_command_queue command_queue;
       OpenCLQueue()
          : eventMax(256), events(new cl_event[eventMax]), eventc(0){
       }
       cl_event *eventListPtr(){
          return (eventc == 0) ? nullptr : events;
       }
        cl_event *nextEventPtr(){
            return &events[eventc];
        }
        void showEvents(int width);
         void wait(){
                 cl_int status = clWaitForEvents(eventc, events);

                    if (status != CL_SUCCESS) {
                        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                        exit(1);
                    }
        }
        void release(){
         cl_int status = CL_SUCCESS;

            for (int i = 0; i < eventc; i++) {
                status = clReleaseEvent(events[i]);
                if (status != CL_SUCCESS) {
                    std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                    exit(1);
                }
            }
               eventc = 0;
            }




       virtual ~OpenCLQueue(){
        clReleaseCommandQueue(command_queue);
        delete []events;
       }
    };

    class OpenCLProgram : public Backend::Program {
        class OpenCLKernel : public Backend::Program::Kernel {
            class OpenCLBuffer : public Backend::Program::Kernel::Buffer {
            public:
                cl_mem clMem;
                void copyToDevice();
                void copyFromDevice();
                OpenCLBuffer(Backend::Program::Kernel *kernel, Arg_s *arg);
                virtual ~OpenCLBuffer();
            };

        private:
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
     OpenCLQueue openclQueue;
    OpenCLBackend(int mode, int platform, int device);
    ~OpenCLBackend();
    int getMaxComputeUnits();
    bool getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength);
    void info();
    void dumpSled(std::ostream &out,void *argArray);
    char *dumpSchema(std::ostream &out,int depth, char *ptr, void *data);
    long compileProgram(int len, char *source);

public:
    static const char *errorMsg(cl_int status);
};