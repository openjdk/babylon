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

OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::OpenCLBuffer(void *ptr, size_t sizeInBytes, cl_context context)
        : Backend::Program::Kernel::Buffer(ptr, sizeInBytes) {
    cl_int status;
    clMem = clCreateBuffer(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE, sizeInBytes, ptr, &status);
}
void OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::copyToDevice() {

}
void OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::copyFromDevice() {

}
OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::~OpenCLBuffer() {
    clReleaseMemObject(clMem);
}

OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLKernel(Backend::Program *program, cl_kernel kernel)
        : Backend::Program::Kernel(program), kernel(kernel) {
}

OpenCLBackend::OpenCLProgram::OpenCLKernel::~OpenCLKernel() {
    clReleaseKernel(kernel);
}

long OpenCLBackend::OpenCLProgram::OpenCLKernel::ndrange(int range, void *argArray) {
    //std::cout<<"ndrange("<<range<<") "<< std::endl;
    ArgSled argSled((ArgArray_t *) argArray);
    cl_int status;
    OpenCLBackend *backend = (OpenCLBackend *) program->backend;

    bool verbose = false;

    // std::cout << "allocing events "<< ((argSled.argc()*3)+1)<< std::endl;

    backend->allocEvents(argSled.argc() * 3 + 1);

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_t *arg = argSled.arg(i);

        if (arg->variant == '&') {
            arg->value.buffer.vendorPtr = new OpenCLBuffer(
                    (void *) arg->value.buffer.memorySegment,
                    (size_t) arg->value.buffer.sizeInBytes,
                    backend->context);
            OpenCLBuffer *clbuf = ((OpenCLBuffer *) arg->value.buffer.vendorPtr);
            if ((status = clEnqueueWriteBuffer(backend->command_queue, clbuf->clMem, CL_FALSE, 0, clbuf->sizeInBytes, clbuf->ptr, backend->eventc, ((backend->eventc == 0) ? NULL : backend->events),
                    &(backend->events[backend->eventc]))) !=
                CL_SUCCESS) {
                std::cerr << "write failed!" << errorMsg(status) << std::endl;
            }
            backend->eventc++;
            clSetKernelArg(kernel, arg->idx, sizeof(cl_mem), &((OpenCLBuffer *) arg->value.buffer.vendorPtr)->clMem);

        } else if (arg->variant == 'I') {
            clSetKernelArg(kernel, arg->idx, sizeof(arg->value.s32), (void *) &arg->value.s32);
        } else if (arg->variant == 'F') {
            clSetKernelArg(kernel, arg->idx, sizeof(arg->value.f32), (void *) &arg->value.f32);
        }
    }
    size_t globalSize = range;
    size_t dims = 1;
    if ((status = clEnqueueNDRangeKernel(
            backend->command_queue,
            kernel,
            dims,
            nullptr,
            &globalSize,
            nullptr,
            backend->eventc,
            ((backend->eventc == 0) ? nullptr : backend->events),
            &(backend->events[backend->eventc]))) != CL_SUCCESS) {
#ifdef VERBOSE
        std::cout <<  " globalSize=" << globalSize << " " << error(status) << std::endl;
#endif
    }
    backend->eventc++;
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_t *arg = argSled.arg(i);

        if (arg->variant == '&') {
            OpenCLBuffer *clBuf = ((OpenCLBuffer *) arg->value.buffer.vendorPtr);
            if ((status = clEnqueueReadBuffer(backend->command_queue, clBuf->clMem, CL_FALSE, 0,
                    clBuf->sizeInBytes, clBuf->ptr, backend->eventc, ((backend->eventc == 0) ? NULL : backend->events),
                    &(backend->events[backend->eventc]))) !=
                CL_SUCCESS) {
                std::cout << "read failed!";
            }
            backend->eventc++;
        }
    }
    backend->waitForEvents();
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_t *arg = argSled.arg(i);
        if (arg->variant == '&') {
            delete ((OpenCLBuffer *) arg->value.buffer.vendorPtr);
        }
    }
    backend->releaseEvents();
    return 0;
}


OpenCLBackend::OpenCLProgram::OpenCLProgram(Backend *backend, BuildInfo *buildInfo, cl_program program)
        : Backend::Program(backend, buildInfo), program(program) {
}

OpenCLBackend::OpenCLProgram::~OpenCLProgram() {
    clReleaseProgram(program);
}

long OpenCLBackend::OpenCLProgram::getKernel(int nameLen, char *name) {
    cl_int status;
    cl_kernel kernel = clCreateKernel(program, name, &status);
    return (long) new OpenCLKernel(this, kernel);
}

bool OpenCLBackend::OpenCLProgram::programOK() {
    return true;
}

OpenCLBackend::OpenCLBackend(OpenCLBackend::OpenCLConfig *openclConfig, int configSchemaLen, char *configSchema)
        : Backend((Backend::Config *) openclConfig, configSchemaLen, configSchema), eventMax(0), events(nullptr), eventc(0) {

    if (openclConfig == nullptr) {
        std::cout << "openclConfig == null" << std::endl;
    } else {
        std::cout << "openclConfig->gpu" << (openclConfig->gpu ? "true" : "false") << std::endl;
        std::cout << "openclConfig->schema" << configSchema << std::endl;
    }
    cl_device_type requestedType = openclConfig == nullptr ? CL_DEVICE_TYPE_GPU : openclConfig->gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

    cl_int status;
    cl_uint platformc = 0;
    if ((status = clGetPlatformIDs(0, NULL, &platformc)) != CL_SUCCESS) {
        return;
    }
    cl_platform_id *platforms = new cl_platform_id[platformc];
    if ((status = clGetPlatformIDs(platformc, platforms, NULL)) != CL_SUCCESS) {
        return;
    }

    cl_uint devicec = 0;
    for (unsigned int i = 0; devicec == 0 && i < platformc; ++i) {
        platform_id = platforms[i];
        if ((status = clGetDeviceIDs(platform_id, requestedType, 0, NULL, &devicec)) != CL_SUCCESS) {
            delete[] platforms;
            return;
        }
    }
    if (devicec == 0) {
        status = CL_DEVICE_NOT_AVAILABLE;
        return;
    }
    cl_device_id *device_ids = new cl_device_id[devicec];             // compute device id
    if ((status = clGetDeviceIDs(platform_id, requestedType, devicec, device_ids, NULL)) != CL_SUCCESS) {
        delete[] platforms;
        delete[] device_ids;
        return;
    }
    if ((context = clCreateContext(0, 1, device_ids, NULL, NULL, &status)) == NULL || status != CL_SUCCESS) {
        delete[] platforms;
        delete[] device_ids;
        return;
    }

    cl_command_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE;

    if ((command_queue = clCreateCommandQueue(context, device_ids[0], queue_props, &status)) == NULL || status != CL_SUCCESS) {
        clReleaseContext(context);
        delete[] platforms;
        delete[] device_ids;
        return;
    }
    device_id = device_ids[0];
    delete[] device_ids;
    delete[] platforms;
}

OpenCLBackend::OpenCLBackend()
        : OpenCLBackend(nullptr, 0, nullptr) {

}

OpenCLBackend::~OpenCLBackend() {
    clReleaseContext(context);
    clReleaseCommandQueue(command_queue);
}

void OpenCLBackend::allocEvents(int max) {
    if (events != nullptr || eventc != 0) {
        std::cerr << "opencl state issue, we might have leaked events!" << std::endl;
    }
    eventMax = max;
    eventc = 0;
    events = new cl_event[eventMax];
}

void OpenCLBackend::releaseEvents() {
    for (int i = 0; i < eventc; i++) {
        clReleaseEvent(events[i]);
    }
    delete[] events;
    eventMax = 0;
    eventc = 0;
    events = nullptr;
}

void OpenCLBackend::waitForEvents() {
    clWaitForEvents(eventc, events);
}

void OpenCLBackend::showEvents(int width) {
    cl_ulong *samples = new cl_ulong[4 * eventc]; // queued, submit, start, end
    int sample = 0;
    cl_ulong min;
    cl_ulong max;
    for (int event = 0; event < eventc; event++) {
        for (int type = 0; type < 4; type++) {
            cl_profiling_info info;
            switch (type) {
                case 0:
                    info = CL_PROFILING_COMMAND_QUEUED;
                    break;
                case 1:
                    info = CL_PROFILING_COMMAND_SUBMIT;
                    break;
                case 2:
                    info = CL_PROFILING_COMMAND_START;
                    break;
                case 3:
                    info = CL_PROFILING_COMMAND_END;
                    break;
            }

            if ((clGetEventProfilingInfo(events[event], info, sizeof(samples[sample]), &samples[sample], NULL)) != CL_SUCCESS) {
                std::cerr << "failed to get profile info " << info << std::endl;
            }
            if (sample == 0) {
                min = max = samples[sample];
            } else {
                if (samples[sample] < min) {
                    min = samples[sample];
                }
                if (samples[sample] > max) {
                    max = samples[sample];
                }
            }
            sample++;
        }
    }
    sample = 0;
    int range = (max - min);
    int scale = range / width;  // range per char
    std::cout << "Range: " << range << "(ns)" << std::endl;
    std::cout << "Scale: " << scale << " range (ns) per char" << std::endl;

    for (int event = 0; event < eventc; event++) {
        cl_ulong queue = (samples[sample++] - min) / scale;
        cl_ulong submit = (samples[sample++] - min) / scale;
        cl_ulong start = (samples[sample++] - min) / scale;
        cl_ulong end = (samples[sample++] - min) / scale;
        for (int c = 0; c < 80; c++) {
            if (c > queue) {
                if (c > submit) {
                    if (c > start) {
                        if (c > end) {
                            std::cout << " ";
                        } else {
                            std::cout << "=";
                        }
                    } else {
                        std::cout << "#";
                    }
                } else {
                    std::cout << "+";
                }
            } else {
                std::cout << " ";
            }
        }
        std::cout << std::endl;

    }
    delete[] samples;
}

int OpenCLBackend::getMaxComputeUnits() {
    std::cout << "getMaxComputeUnits()" << std::endl;
    cl_uint value;
    clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(value), &value, nullptr);
    return value;
}

void OpenCLBackend::info() {
    cl_int status;
    fprintf(stderr, "platform{\n");
    char platformVersionName[512];
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(platformVersionName), platformVersionName, NULL);
    char platformVendorName[512];
    char platformName[512];
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_VENDOR, sizeof(platformVendorName), platformVendorName, NULL);
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
    fprintf(stderr, "   CL_PLATFORM_VENDOR..\"%s\"\n", platformVendorName);
    fprintf(stderr, "   CL_PLATFORM_VERSION.\"%s\"\n", platformVersionName);
    fprintf(stderr, "   CL_PLATFORM_NAME....\"%s\"\n", platformName);


    cl_device_type deviceType;
    status = clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
    fprintf(stderr, "         CL_DEVICE_TYPE..................... ");
    if (deviceType & CL_DEVICE_TYPE_DEFAULT) {
        deviceType &= ~CL_DEVICE_TYPE_DEFAULT;
        fprintf(stderr, "Default ");
    }
    if (deviceType & CL_DEVICE_TYPE_CPU) {
        deviceType &= ~CL_DEVICE_TYPE_CPU;
        fprintf(stderr, "CPU ");
    }
    if (deviceType & CL_DEVICE_TYPE_GPU) {
        deviceType &= ~CL_DEVICE_TYPE_GPU;
        fprintf(stderr, "GPU ");
    }
    if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
        deviceType &= ~CL_DEVICE_TYPE_ACCELERATOR;
        fprintf(stderr, "Accelerator ");
    }
    fprintf(stderr, LongHexNewline, deviceType);

    cl_uint maxComputeUnits;
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(maxComputeUnits), &maxComputeUnits, NULL);
    fprintf(stderr, "         CL_DEVICE_MAX_COMPUTE_UNITS........ %u\n", maxComputeUnits);

    cl_uint maxWorkItemDimensions;
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDimensions), &maxWorkItemDimensions, NULL);
    fprintf(stderr, "         CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. %u\n", maxWorkItemDimensions);

    size_t *maxWorkItemSizes = new size_t[maxWorkItemDimensions];
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxWorkItemDimensions, maxWorkItemSizes, NULL);
    for (unsigned dimIdx = 0; dimIdx < maxWorkItemDimensions; dimIdx++) {
        fprintf(stderr, "             dim[%d] = %ld\n", dimIdx, maxWorkItemSizes[dimIdx]);
    }

    size_t maxWorkGroupSize;
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    fprintf(stderr, "         CL_DEVICE_MAX_WORK_GROUP_SIZE...... " Size_tNewline, maxWorkGroupSize);

    cl_ulong maxMemAllocSize;
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);
    fprintf(stderr, "         CL_DEVICE_MAX_MEM_ALLOC_SIZE....... " LongUnsignedNewline, maxMemAllocSize);

    cl_ulong globalMemSize;
    status = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
    fprintf(stderr, "         CL_DEVICE_GLOBAL_MEM_SIZE.......... " LongUnsignedNewline, globalMemSize);

    cl_ulong localMemSize;
    status = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    fprintf(stderr, "         CL_DEVICE_LOCAL_MEM_SIZE........... " LongUnsignedNewline, localMemSize);

    char profile[2048];
    status = clGetDeviceInfo(device_id, CL_DEVICE_PROFILE, sizeof(profile), &profile, NULL);
    fprintf(stderr, "         CL_DEVICE_PROFILE.................. %s\n", profile);

    char deviceVersion[2048];
    status = clGetDeviceInfo(device_id, CL_DEVICE_VERSION, sizeof(deviceVersion), &deviceVersion, NULL);
    fprintf(stderr, "         CL_DEVICE_VERSION.................. %s\n", deviceVersion);

    char driverVersion[2048];
    status = clGetDeviceInfo(device_id, CL_DRIVER_VERSION, sizeof(driverVersion), &driverVersion, NULL);
    fprintf(stderr, "         CL_DRIVER_VERSION.................. %s\n", driverVersion);

    char cVersion[2048];
    status = clGetDeviceInfo(device_id, CL_DEVICE_OPENCL_C_VERSION, sizeof(cVersion), &cVersion, NULL);
    fprintf(stderr, "         CL_DEVICE_OPENCL_C_VERSION......... %s\n", cVersion);

    char name[2048];
    status = clGetDeviceInfo(device_id, CL_DEVICE_NAME, sizeof(name), &name, NULL);
    fprintf(stderr, "         CL_DEVICE_NAME..................... %s\n", name);
    char extensions[2048];
    status = clGetDeviceInfo(device_id, CL_DEVICE_EXTENSIONS, sizeof(extensions), &extensions, NULL);
    fprintf(stderr, "         CL_DEVICE_EXTENSIONS............... %s\n", extensions);
    char builtInKernels[2048];
    status = clGetDeviceInfo(device_id, CL_DEVICE_BUILT_IN_KERNELS, sizeof(builtInKernels), &builtInKernels, NULL);
    fprintf(stderr, "         CL_DEVICE_BUILT_IN_KERNELS......... %s\n", builtInKernels);

    fprintf(stderr, "      }\n");
}

long OpenCLBackend::compileProgram(int len, char *source) {
    size_t srcLen = ::strlen(source);
    char *src = new char[srcLen + 1];
    ::strncpy(src, source, srcLen);
    src[srcLen] = '\0';
    //std::cout << "native compiling " << src << std::endl;
    cl_int status;
    cl_program program;
    if ((program = clCreateProgramWithSource(context, 1, (const char **) &src, nullptr, &status)) == nullptr || status != CL_SUCCESS) {
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
            if ((status = clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, logLen + 1, (void *) log, nullptr)) != CL_SUCCESS) {
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

    return (long) new OpenCLProgram(this, buildInfo, program);
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
            {0, NULL},
    };
    static char unknown[256];
    int ii;

    for (ii = 0; error_table[ii].msg != NULL; ii++) {
        if (error_table[ii].code == status) {
//std::cerr << " clerror '" << error_table[ii].msg << "'" << std::endl;
            return error_table[ii].msg;
        }
    }
    SNPRINTF(unknown, sizeof(unknown), "unknown error %d", status);
    return unknown;
}


long getBackend(void *config, int configSchemaLen, char *configSchema) {
    // Dynamic cast?
    OpenCLBackend::OpenCLConfig *openclConfig = (OpenCLBackend::OpenCLConfig *) config;

    return (long) new OpenCLBackend(openclConfig, configSchemaLen, configSchema);
}