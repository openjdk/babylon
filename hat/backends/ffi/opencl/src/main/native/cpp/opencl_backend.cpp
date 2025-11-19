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

OpenCLBackend::OpenCLBuffer *OpenCLBackend::getOrCreateBuffer(BufferState *bufferState) {
    OpenCLBuffer *openclBuffer = nullptr;
    if (bufferState->vendorPtr == nullptr || bufferState->state == BufferState::NEW_STATE) {
        openclBuffer = new OpenCLBuffer(this, bufferState);
        if (config->trace) {
            std::cout << "We allocated arg buffer " << std::endl;
        }
    } else {
        if (config->trace) {
            std::cout << "Were reusing  buffer  buffer " << std::endl;
        }
        openclBuffer = static_cast<OpenCLBuffer *>(bufferState->vendorPtr);
    }
    return openclBuffer;
}

bool OpenCLBackend::getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) {
    if (config->traceCalls) {
        std::cout << "getBufferFromDeviceIfDirty(" << std::hex << (long) memorySegment << "," << std::dec <<
                memorySegmentLength << "){" << std::endl;
    }
    if (config->minimizeCopies) {
        const BufferState *bufferState = BufferState::of(memorySegment, memorySegmentLength);
        if (bufferState->state == BufferState::DEVICE_OWNED) {
            queue->copyFromDevice(static_cast<Buffer *>(bufferState->vendorPtr));
            if (config->traceEnqueues | config->traceCopies) {
                std::cout << "copying buffer from device (from java access) " << std::endl;
            }
            queue->wait();
            queue->release();
        } else {
            std::cout << "HOW DID WE GET HERE 1 attempting  to get buffer but buffer is not device dirty" << std::endl;
            std::exit(1);
        }
    } else {
        std::cerr <<
                "HOW DID WE GET HERE ? java side should avoid calling getBufferFromDeviceIfDirty as we are not minimising buffers!"
                << std::endl;
        std::exit(1);
    }
    if (config->traceCalls) {
        std::cout << "}getBufferFromDeviceIfDirty()" << std::endl;
    }
    return true;
}

OpenCLBackend::OpenCLBackend(int configBits)
    : Backend(new Config(configBits), new OpenCLQueue(this)) {

    if (config->info) {
        std::cerr << "[INFO] Config Bits = " << std::hex << configBits << std::dec << std::endl;
    }

    cl_int status;
    cl_uint platformc = 0;
    OPENCL_CHECK(clGetPlatformIDs(0, nullptr, &platformc), "clGetPlatformIDs");

    if (config->platform >= platformc) {
        std::cerr << "We only have " << platformc << " platform" << ((platformc > 1) ? "s" : "") <<
                " (platform[0]-platform[" << (platformc - 1) << "] inclusive) you requested platform[" << config->
                platform << "]" << std::endl;
        std::exit(1);
    }
    auto *platforms = new cl_platform_id[platformc];
    OPENCL_CHECK(clGetPlatformIDs(platformc, platforms, nullptr), "clGetPlatformIDs");

    cl_uint numDevices = 0;
    platform_id = platforms[config->platform];
    if ((status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices)) != CL_SUCCESS) {
        if (status != CL_SUCCESS) {
            std::cerr << "clGetDeviceIDs (to get count) failed " << errorMsg(status) << std::endl;
        }
        delete[] platforms;
        return;
    }
    if (config->device >= numDevices) {
        std::cerr << "Platform[" << config->platform << "] only has " << numDevices << " device" << (
                    (numDevices > 1) ? "s" : "") << " (device[0]-device[" << (numDevices - 1) <<
                "] inclusive) and you requested device[" << config->device << "]" << std::endl;
        std::cerr << "No device available " << errorMsg(CL_DEVICE_NOT_AVAILABLE) << std::endl;
        delete[] platforms;
        std::exit(1);
    }

    if (numDevices == 0) {
        status = CL_DEVICE_NOT_AVAILABLE;
        std::cerr << "No device available " << errorMsg(status) << std::endl;
        delete[] platforms;
        return;
    }
    auto *device_ids = new cl_device_id[numDevices]; // compute device id
    if ((status = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_ALL, numDevices, device_ids, nullptr)) != CL_SUCCESS) {
        std::cerr << "clGetDeviceIDs failed " << errorMsg(status) << std::endl;
        delete[] platforms;
        delete[] device_ids;
        return;
    }
    if ((context = clCreateContext(nullptr, 1, &device_ids[config->device], nullptr, nullptr, &status)) == nullptr ||
        status != CL_SUCCESS) {
        std::cerr << "clCreateContext failed " << errorMsg(status) << std::endl;
        delete[] platforms;
        delete[] device_ids;
        return;
    }

    cl_command_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE;
    const auto openCLQueue = dynamic_cast<OpenCLQueue *>(queue);
    if ((openCLQueue->command_queue = clCreateCommandQueue(context, device_ids[config->device], queue_props, &status))
        == nullptr || status != CL_SUCCESS) {
        std::cerr << "clCreateCommandQueue failed " << errorMsg(status) << std::endl;
        clReleaseContext(context);
        delete[] platforms;
        delete[] device_ids;
        return;
    }

    device_id = device_ids[config->device];
    delete[] device_ids;
    delete[] platforms;
}

OpenCLBackend::~OpenCLBackend() {
    clReleaseContext(context);
}

void OpenCLBackend::computeStart() {
    if (config->trace) {
        std::cout << "compute start" << std::endl;
    }
    queue->computeStart();
}

void OpenCLBackend::computeEnd() {
    queue->computeEnd();
    queue->wait();

    if (config->profile) {
        const auto openCLQueue = dynamic_cast<OpenCLQueue *>(queue);
        openCLQueue->showEvents(100);
    }
    queue->release();
    if (config->trace) {
        std::cout << "compute end" << std::endl;
    }
}

OpenCLBackend::OpenCLProgram *OpenCLBackend::compileProgram(OpenCLSource &openclSource) {
    return compileProgram(&openclSource);
}

OpenCLBackend::OpenCLProgram *OpenCLBackend::compileProgram(const OpenCLSource *openclSource) {
    return compileProgram(openclSource->len, openclSource->text);
}

OpenCLBackend::OpenCLProgram *OpenCLBackend::compileProgram(int len, char *text) {
    return dynamic_cast<OpenCLProgram *>(compile(len, text));
}

Backend::CompilationUnit *OpenCLBackend::compile(int len, char *source) {
    const size_t srcLen = ::strlen(source);
    auto src = new char[srcLen + 1];
    strncpy(src, source, srcLen);
    src[srcLen] = '\0';
    if (config->trace) {
        std::cout << "native compiling " << src << std::endl;
    }
    cl_int status;
    cl_program program;
    if ((program = clCreateProgramWithSource(context, 1, (const char **) &src, nullptr, &status)) == nullptr ||
        status != CL_SUCCESS) {
        std::cerr << "clCreateProgramWithSource failed" << std::endl;
        delete[] src;
        return nullptr;
    }

    cl_int buildStatus = clBuildProgram(program, 0, nullptr, nullptr, nullptr, nullptr);
    if (buildStatus != CL_SUCCESS) {
        std::cerr << "buildStatus =failed" << std::endl;
    }
    size_t logLen = 0;
    OpenCLProgram *openclProgram = nullptr;
    if ((status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logLen)) != CL_SUCCESS) {
        std::cerr << "clGetBuildInfo (getting log size) failed" << std::endl;
        //openclProgram->buildInfo = new Backend::CompilationUnit::BuildInfo(openclProgram, src, nullptr, false);
        openclProgram = new OpenCLProgram(this, src, nullptr, buildStatus == CL_SUCCESS, program);
    } else {
        //  cl_build_status buildStatus;
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
                if (logLen > 2) {
                    std::cerr << "logLen = " << logLen << " log  = " << log << std::endl;
                }
            }
            openclProgram = new OpenCLProgram(this, src, log, buildStatus == CL_SUCCESS, program);
        } else {
            openclProgram = new OpenCLProgram(this, src, nullptr, buildStatus == CL_SUCCESS, program);
        }
    }
    return openclProgram;
}

const char *OpenCLBackend::errorMsg(cl_int status) {
    static struct {
        cl_int code;
        const char *msg;
    } error_table[] = {
        // @formatter:off
                {CL_SUCCESS, "success"},
                {CL_DEVICE_NOT_FOUND, "device not found",},
                {CL_DEVICE_NOT_AVAILABLE, "device not available",},
                {CL_COMPILER_NOT_AVAILABLE, "compiler not available",},
                {CL_MEM_OBJECT_ALLOCATION_FAILURE, "mem object allocation failure",},
                {CL_OUT_OF_RESOURCES, "out of resources",},
                {CL_OUT_OF_HOST_MEMORY, "out of host memory",},
                {CL_PROFILING_INFO_NOT_AVAILABLE, "profiling not available",},
                {CL_MEM_COPY_OVERLAP, "memcopy overlaps",},
                {CL_IMAGE_FORMAT_MISMATCH, "image format mismatch",},
                {CL_IMAGE_FORMAT_NOT_SUPPORTED, "image format not supported",},
                {CL_BUILD_PROGRAM_FAILURE, "build program failed",},
                {CL_MAP_FAILURE, "map failed",},
                {CL_INVALID_VALUE, "invalid value",},
                {CL_INVALID_DEVICE_TYPE, "invalid device type",},
                {CL_INVALID_PLATFORM, "invlaid platform",},
                {CL_INVALID_DEVICE, "invalid device",},
                {CL_INVALID_CONTEXT, "invalid context",},
                {CL_INVALID_QUEUE_PROPERTIES, "invalid queue properties",},
                {CL_INVALID_COMMAND_QUEUE, "invalid command queue",},
                {CL_INVALID_HOST_PTR, "invalid host ptr",},
                {CL_INVALID_MEM_OBJECT, "invalid mem object",},
                {CL_INVALID_IMAGE_FORMAT_DESCRIPTOR, "invalid image format descriptor ",},
                {CL_INVALID_IMAGE_SIZE, "invalid image size",},
                {CL_INVALID_SAMPLER, "invalid sampler",},
                {CL_INVALID_BINARY, "invalid binary",},
                {CL_INVALID_BUILD_OPTIONS, "invalid build options",},
                {CL_INVALID_PROGRAM, "invalid program ",},
                {CL_INVALID_PROGRAM_EXECUTABLE, "invalid program executable",},
                {CL_INVALID_KERNEL_NAME, "invalid kernel name",},
                {CL_INVALID_KERNEL_DEFINITION, "invalid definition",},
                {CL_INVALID_KERNEL, "invalid kernel",},
                {CL_INVALID_ARG_INDEX, "invalid arg index",},
                {CL_INVALID_ARG_VALUE, "invalid arg value",},
                {CL_INVALID_ARG_SIZE, "invalid arg size",},
                {CL_INVALID_KERNEL_ARGS, "invalid kernel args",},
                {CL_INVALID_WORK_DIMENSION, "invalid work dimension",},
                {CL_INVALID_WORK_GROUP_SIZE, "invalid work group size",},
                {CL_INVALID_WORK_ITEM_SIZE, "invalid work item size",},
                {CL_INVALID_GLOBAL_OFFSET, "invalid global offset",},
                {CL_INVALID_EVENT_WAIT_LIST, "invalid event wait list",},
                {CL_INVALID_EVENT, "invalid event",},
                {CL_INVALID_OPERATION, "invalid operation",},
                {CL_INVALID_GL_OBJECT, "invalid gl object",},
                {CL_INVALID_BUFFER_SIZE, "invalid buffer size",},
                {CL_INVALID_MIP_LEVEL, "invalid mip level",},
                {CL_INVALID_GLOBAL_WORK_SIZE, "invalid global work size",},
                {-9999, "enqueueNdRangeKernel Illegal read or write to a buffer",},
                {0, nullptr},
                // @formatter:on
            };
    for (int i = 0; error_table[i].msg != nullptr; i++) {
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

extern "C" long getBackend(int configBits) {
    return reinterpret_cast<long>(new OpenCLBackend(configBits));
}

void __checkOpenclErrors(cl_int status, const char *functionName, const char *file, const int line) {
    if (CL_SUCCESS != status) {
        std::cerr << "Opencl Error ( " << functionName << ") with error code: " << status << " from file " << file <<
                " line " << line << std::endl;
        exit(-1);
    }
}

OpenCLSource::OpenCLSource()
    : Text(0L) {
}

OpenCLSource::OpenCLSource(const size_t len)
    : Text(len) {
}

OpenCLSource::OpenCLSource(char *text)
    : Text(text, false) {
}
