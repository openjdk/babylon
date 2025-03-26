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

#include "opencl_backend.h"

bool OpenCLBackend::getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) {
    if (openclConfig.traceCalls){
      std::cout << "getBufferFromDeviceIfDirty(" <<std::hex << (long)memorySegment << "," << std::dec<< memorySegmentLength <<"){"<<std::endl;
    }
    if (openclConfig.minimizeCopies){
       BufferState_s * bufferState = BufferState_s::of(memorySegment,memorySegmentLength);
       if (bufferState->state == BufferState_s::DEVICE_OWNED){
          static_cast<OpenCLProgram::OpenCLKernel::OpenCLBuffer *>(bufferState->vendorPtr)->copyFromDevice();
          if (openclConfig.traceEnqueues | openclConfig.traceCopies){
             std::cout << "copying buffer from device (from java access) "<< std::endl;
          }
          openclQueue.wait();
          openclQueue.release();
       }else{
          std::cout << "HOW DID WE GET HERE 1 attempting  to get buffer but buffer is not device dirty"<<std::endl;
          std::exit(1);
       }
    }else{
     std::cerr << "HOW DID WE GET HERE ? java side should avoid calling getBufferFromDeviceIfDirty as we are not minimising buffers!"<<std::endl;
     std::exit(1);
    }
    if (openclConfig.traceCalls){
      std::cout << "}getBufferFromDeviceIfDirty()"<<std::endl;
    }
    return true;
}

OpenCLBackend::OpenCLBackend(int configBits )
        : Backend(configBits), openclConfig(mode), openclQueue(this) {

    cl_int status;
    cl_uint platformc = 0;
    if ((status = clGetPlatformIDs(0, NULL, &platformc)) != CL_SUCCESS) {
        std::cerr << "clGetPlatformIDs (to get count) failed " << errorMsg(status)<<std::endl;
        std::exit(1);
        return;
    }

    if (openclConfig.platform >= platformc){
        std::cerr << "We only have "<<platformc<<" platform"<<((platformc>1)?"s":"")<<" (platform[0]-platform["<<(platformc-1)<<"] inclusive) you requested platform["<<openclConfig.platform<<"]"<< std::endl;
        std::exit(1);
        return;
    }
    cl_platform_id *platforms = new cl_platform_id[platformc];
    if ((status = clGetPlatformIDs(platformc, platforms, NULL)) != CL_SUCCESS) {
        std::cerr << "clGetPlatformIDs failed " << errorMsg(status)<<std::endl;
        std::exit(1);
        return;
    }
    cl_uint devicec = 0;
        platform_id = platforms[openclConfig.platform];
        if ((status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, NULL, &devicec)) != CL_SUCCESS) {
            if (status != CL_SUCCESS){
               std::cerr << "clGetDeviceIDs (to get count) failed " << errorMsg(status)<<std::endl;
            }
            delete[] platforms;
            return;
        }
       if (openclConfig.device >= devicec){
            std::cerr << "Platform["<<openclConfig.platform<<"] only has "<<devicec<<" device"<<((devicec>1)?"s":"")<<" (device[0]-device["<<(devicec-1)<<"] inclusive) and you requested device["<<openclConfig.device<<"]"<< std::endl;
            std::cerr << "No device available " << errorMsg(CL_DEVICE_NOT_AVAILABLE)<<std::endl;
              delete[] platforms;
            std::exit(1);
            return;
        }

    if (devicec == 0) {
        status = CL_DEVICE_NOT_AVAILABLE;
        std::cerr << "No device available " << errorMsg(status)<<std::endl;
          delete[] platforms;
        return;
    }
    cl_device_id *device_ids = new cl_device_id[devicec];             // compute device id
    if ((status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, devicec, device_ids, NULL)) != CL_SUCCESS) {
        std::cerr << "clGetDeviceIDs failed " << errorMsg(status)<<std::endl;
        delete[] platforms;
        delete[] device_ids;
        return;
    }
    if ((context = clCreateContext(nullptr, 1, &device_ids[openclConfig.device], NULL, NULL, &status)) == NULL || status != CL_SUCCESS) {
        std::cerr << "clCreateContext failed " << errorMsg(status)<<std::endl;
        delete[] platforms;
        delete[] device_ids;
        return;
    }

    cl_command_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE;

    if ((openclQueue.command_queue = clCreateCommandQueue(context, device_ids[openclConfig.device], queue_props, &status)) == NULL ||
        status != CL_SUCCESS) {
        std::cerr << "clCreateCommandQueue failed " << errorMsg(status)<<std::endl;
        clReleaseContext(context);
        delete[] platforms;
        delete[] device_ids;
        return;
    }

    device_id = device_ids[openclConfig.device];
    delete[] device_ids;
    delete[] platforms;

}

OpenCLBackend::~OpenCLBackend() {
    clReleaseContext(context);

}

void OpenCLBackend::computeStart() {
  if (openclConfig.trace){
     std::cout <<"compute start" <<std::endl;
  }
  openclQueue.computeStart();
}
void OpenCLBackend::computeEnd() {
  openclQueue.computeEnd();
 openclQueue.wait();

 if (openclConfig.profile){
     openclQueue.showEvents(100);
 }
 openclQueue.release();
 if (openclConfig.trace){
     std::cout <<"compute end" <<std::endl;
 }
}

long OpenCLBackend::compileProgram(int len, char *source) {
    size_t srcLen = ::strlen(source);
    char *src = new char[srcLen + 1];
    ::strncpy(src, source, srcLen);
    src[srcLen] = '\0';
    if(openclConfig.trace){
        std::cout << "native compiling " << src << std::endl;
    }
    cl_int status;
    cl_program program;
    if ((program = clCreateProgramWithSource(context, 1, (const char **) &src, nullptr, &status)) == nullptr ||
        status != CL_SUCCESS) {
        std::cerr << "clCreateProgramWithSource failed" << std::endl;
        delete[] src;
        return 0;
    }

    if ((status = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr)) != CL_SUCCESS) {
        std::cerr << "clBuildProgram failed" << std::endl;
        // dont return we may still be able to get log!
    }
    size_t logLen = 0;

    BuildInfo *buildInfo = nullptr;
    if ((status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen)) != CL_SUCCESS) {
        std::cerr << "clGetBuildInfo (getting log size) failed" << std::endl;
        buildInfo = new BuildInfo(src, nullptr, true);
    } else {
        cl_build_status buildStatus;
        clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_STATUS, sizeof(buildStatus), &buildStatus, nullptr);
        if (logLen > 0) {
            char *log = new char[logLen + 1];
            if ((status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logLen + 1, (void *) log,
                                                nullptr)) != CL_SUCCESS) {
                std::cerr << "clGetBuildInfo (getting log) failed" << std::endl;
                delete[] log;
                log = nullptr;
            } else {
                log[logLen] = '\0';
                if (logLen > 1) {
                    std::cerr << "logLen = " << logLen << " log  = " << log << std::endl;
                }
            }
            buildInfo = new BuildInfo(src, log, true);
        } else {
            buildInfo = new BuildInfo(src, nullptr, true);
        }
    }

    return reinterpret_cast<long>(new OpenCLProgram(this, buildInfo, program));
}

const char *OpenCLBackend::errorMsg(cl_int status) {
    static struct {
        cl_int code;
        const char *msg;
    } error_table[] = {
            {CL_SUCCESS,                         "success"},
            {CL_DEVICE_NOT_FOUND,                "device not found",},
            {CL_DEVICE_NOT_AVAILABLE,            "device not available",},
            {CL_COMPILER_NOT_AVAILABLE,          "compiler not available",},
            {CL_MEM_OBJECT_ALLOCATION_FAILURE,   "mem object allocation failure",},
            {CL_OUT_OF_RESOURCES,                "out of resources",},
            {CL_OUT_OF_HOST_MEMORY,              "out of host memory",},
            {CL_PROFILING_INFO_NOT_AVAILABLE,    "profiling not available",},
            {CL_MEM_COPY_OVERLAP,                "memcopy overlaps",},
            {CL_IMAGE_FORMAT_MISMATCH,           "image format mismatch",},
            {CL_IMAGE_FORMAT_NOT_SUPPORTED,      "image format not supported",},
            {CL_BUILD_PROGRAM_FAILURE,           "build program failed",},
            {CL_MAP_FAILURE,                     "map failed",},
            {CL_INVALID_VALUE,                   "invalid value",},
            {CL_INVALID_DEVICE_TYPE,             "invalid device type",},
            {CL_INVALID_PLATFORM,                "invlaid platform",},
            {CL_INVALID_DEVICE,                  "invalid device",},
            {CL_INVALID_CONTEXT,                 "invalid context",},
            {CL_INVALID_QUEUE_PROPERTIES,        "invalid queue properties",},
            {CL_INVALID_COMMAND_QUEUE,           "invalid command queue",},
            {CL_INVALID_HOST_PTR,                "invalid host ptr",},
            {CL_INVALID_MEM_OBJECT,              "invalid mem object",},
            {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "invalid image format descriptor ",},
            {CL_INVALID_IMAGE_SIZE,              "invalid image size",},
            {CL_INVALID_SAMPLER,                 "invalid sampler",},
            {CL_INVALID_BINARY,                  "invalid binary",},
            {CL_INVALID_BUILD_OPTIONS,           "invalid build options",},
            {CL_INVALID_PROGRAM,                 "invalid program ",},
            {CL_INVALID_PROGRAM_EXECUTABLE,      "invalid program executable",},
            {CL_INVALID_KERNEL_NAME,             "invalid kernel name",},
            {CL_INVALID_KERNEL_DEFINITION,       "invalid definition",},
            {CL_INVALID_KERNEL,                  "invalid kernel",},
            {CL_INVALID_ARG_INDEX,               "invalid arg index",},
            {CL_INVALID_ARG_VALUE,               "invalid arg value",},
            {CL_INVALID_ARG_SIZE,                "invalid arg size",},
            {CL_INVALID_KERNEL_ARGS,             "invalid kernel args",},
            {CL_INVALID_WORK_DIMENSION,          "invalid work dimension",},
            {CL_INVALID_WORK_GROUP_SIZE,         "invalid work group size",},
            {CL_INVALID_WORK_ITEM_SIZE,          "invalid work item size",},
            {CL_INVALID_GLOBAL_OFFSET,           "invalid global offset",},
            {CL_INVALID_EVENT_WAIT_LIST,         "invalid event wait list",},
            {CL_INVALID_EVENT,                   "invalid event",},
            {CL_INVALID_OPERATION,               "invalid operation",},
            {CL_INVALID_GL_OBJECT,               "invalid gl object",},
            {CL_INVALID_BUFFER_SIZE,             "invalid buffer size",},
            {CL_INVALID_MIP_LEVEL,               "invalid mip level",},
            {CL_INVALID_GLOBAL_WORK_SIZE,        "invalid global work size",},
            {-9999,                              "enqueueNdRangeKernel Illegal read or write to a buffer",},
            {0,                                  NULL},
    };
    for (int i = 0; error_table[i].msg != NULL; i++) {
        if (error_table[i].code == status) {
            //std::cerr << " clerror '" << error_table[i].msg << "'" << std::endl;
            return error_table[i].msg;
        }
    }
    static char unknown[256];
     #if defined (_WIN32)
        _snprintf
     #else
        snprintf
     #endif
     (unknown, sizeof(unknown), "unmapped string for  error %d", status);
    return unknown;
}


long getOpenCLBackend(int configBits) {
 // std::cerr << "Opencl Driver mode=" << mode << " platform=" << platform << " device=" << device << std::endl;

    return reinterpret_cast<long>(new OpenCLBackend(configBits));
}


void __checkOpenclErrors(cl_int status, const char *file, const int line) {
    if (CL_SUCCESS != status) {
        std::cerr << "Opencl Driver API error = " << status << " from file " << file << " line " << line << std::endl;
        exit(-1);
    }
}

