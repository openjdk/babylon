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

#define INFO 1

/*
  OpenCLBuffer
  */

OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::OpenCLBuffer(Backend::Program::Kernel *kernel, Arg_s *arg)
        : Backend::Program::Kernel::Buffer(kernel, arg) {
    /*
     *   (void *) arg->value.buffer.memorySegment,
     *   (size_t) arg->value.buffer.sizeInBytes);
     */
    cl_int status;
    auto openclBackend = dynamic_cast<OpenCLBackend *>(kernel->program->backend);
    clMem = clCreateBuffer(openclBackend->context,
                           CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                           arg->value.buffer.sizeInBytes,
                           arg->value.buffer.memorySegment,
                           &status);
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }

    BufferState_s * bufferState = BufferState_s::of(
      arg->value.buffer.memorySegment,
      arg->value.buffer.sizeInBytes
      );
    if (INFO){
       bufferState->dump("on allocation before assign");
    }
    bufferState->vendorPtr =  static_cast<void *>(this);
    if (INFO){
        bufferState->dump("after assign ");
    }
    if (INFO){
         std::cout << "created buffer " << std::endl;
    }
}


void OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::copyToDevice() {

    /*
     *   (void *) arg->value.buffer.memorySegment,
     *   (size_t) arg->value.buffer.sizeInBytes);
     */
    auto openclKernel = dynamic_cast<OpenCLKernel *>(kernel);
    auto openclBackend = dynamic_cast<OpenCLBackend *>(openclKernel->program->backend);
    auto openclQueue = dynamic_cast<OpenCLQueue *>(openclKernel->program->backend->queue);
    auto openclConfig = dynamic_cast<OpenCLConfig *>(openclKernel->program->backend->config);
    cl_int status = clEnqueueWriteBuffer( openclQueue->command_queue,
                                         clMem,
                                         CL_FALSE,
                                         0,
                                         arg->value.buffer.sizeInBytes,
                                         arg->value.buffer.memorySegment,
                                         openclQueue->eventc,
                                         (openclQueue->eventc == 0) ? NULL : openclQueue->events,
                                         &(openclQueue->events[openclQueue->eventc]));

    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    openclQueue->eventc++;
    if(openclConfig->trace){
        std::cout << "enqueued buffer copyToDevice " << std::endl;
    }
}

void OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::copyFromDevice() {
    auto openclKernel = dynamic_cast<OpenCLKernel *>(kernel);
    auto openclBackend = dynamic_cast<OpenCLBackend *>(openclKernel->program->backend);
       auto openclQueue = dynamic_cast<OpenCLQueue *>(openclKernel->program->backend->queue);
        auto openclConfig = dynamic_cast<OpenCLConfig *>(openclKernel->program->backend->config);
    cl_int status = clEnqueueReadBuffer( openclQueue->command_queue,
                                        clMem,
                                        CL_FALSE,
                                        0,
                                        arg->value.buffer.sizeInBytes,
                                        arg->value.buffer.memorySegment,
                                        openclQueue->eventc,
                                        (openclQueue->eventc == 0) ? NULL : openclQueue->events,
                                        &(openclQueue->events[openclQueue->eventc]));

    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    openclQueue->eventc++;
    if(openclConfig->trace){
       std::cout << "enqueued buffer copyFromDevice " << std::endl;
    }
}

OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::~OpenCLBuffer() {
    clReleaseMemObject(clMem);
}

/*
  OpenCLKernel
  */

OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLKernel(Backend::Program *program, char* name, cl_kernel kernel)
        : Backend::Program::Kernel(program, name), kernel(kernel){
}

OpenCLBackend::OpenCLProgram::OpenCLKernel::~OpenCLKernel() {
    clReleaseKernel(kernel);
}

long OpenCLBackend::OpenCLProgram::OpenCLKernel::ndrange(void *argArray) {
   // std::cout << "ndrange(" << range << ") " << std::endl;
    ArgSled argSled(static_cast<ArgArray_s *>(argArray));
    OpenCLConfig *openclConfig = dynamic_cast<OpenCLConfig*>(program->backend->config);
    if (openclConfig->trace){
       Sled::show(std::cout, argArray);
    }
   // if (events != nullptr || eventc != 0) {
     //   std::cerr << "opencl issue, we might have leaked events!" << std::endl;
    //}
   // eventMax = argSled.argc() * 4 + 1;
    //eventc = 0;
   // events = new cl_event[eventMax];
    OpenCLQueue *openclQueue = dynamic_cast<OpenCLQueue *>(program->backend->queue);
    NDRange *ndrange = nullptr;
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
               auto openclBuffer = new OpenCLBuffer(this, arg);
                BufferState_s * bufferState = BufferState_s::of(
                             arg->value.buffer.memorySegment,
                             arg->value.buffer.sizeInBytes
                             );

                if (arg->idx == 0){
                    ndrange = static_cast<NDRange *>(arg->value.buffer.memorySegment);
                }
                if (!openclConfig->minimizeCopies){
                    openclBuffer->copyToDevice();
                    std::cout << "copying arg " << arg->idx <<" to device "<< std::endl;
                }
                cl_int status = clSetKernelArg(kernel, arg->idx, sizeof(cl_mem), &openclBuffer->clMem);
                if (status != CL_SUCCESS) {
                    std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                    exit(1);
                }
                if (INFO){
                   std::cout << "set buffer arg " << arg->idx << std::endl;
                }
                break;
            }
            case 'I':
            case 'F': {
                cl_int status = clSetKernelArg(kernel, arg->idx, sizeof(arg->value.x32), (void *) &arg->value);
                if (status != CL_SUCCESS) {
                    std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                    exit(1);
                }
                if (INFO){
                   std::cout << "set I or F arg " << arg->idx << std::endl;
                }
                break;
            }
            case 'S':
            case 'C': {
                cl_int status = clSetKernelArg(kernel, arg->idx, sizeof(arg->value.x16), (void *) &arg->value);
                if (status != CL_SUCCESS) {
                    std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                    exit(1);
                }
                if (INFO){
                   std::cout << "set S or C arg " << arg->idx << std::endl;
                }
                break;
            }
            case 'J':
            case 'D': {
                cl_int status = clSetKernelArg(kernel, arg->idx, sizeof(arg->value.x64), (void *) &arg->value);
                if (status != CL_SUCCESS) {
                    std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                    exit(1);
                }
                if (INFO){
                   std::cout << "set J or D arg " << arg->idx << std::endl;
                }
                break;
            }
            default: {
                std::cerr << "unexpected variant (ndrange) " << (char) arg->variant << std::endl;
                exit(1);
            }
        }
    }

    size_t globalSize = ndrange->maxX;
    if (INFO){
       std::cout << "ndrange = " << ndrange->maxX << std::endl;
    }
    size_t dims = 1;
    cl_int status = clEnqueueNDRangeKernel(
            openclQueue->command_queue,
            kernel,
            dims,
            nullptr,
            &globalSize,
            nullptr,
            openclQueue->eventc,
            (openclQueue->eventc == 0) ? nullptr : openclQueue->events,
            &(openclQueue->events[openclQueue->eventc]));
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if (INFO){
       std::cout << "enqueued dispatch  " << std::endl;
       std::cout <<  " globalSize=" << globalSize << " " << std::endl;
    }

    openclQueue->eventc++;
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            BufferState_s * bufferState = BufferState_s::of(
              arg->value.buffer.memorySegment,
              arg->value.buffer.sizeInBytes
              );
            if (!openclConfig->minimizeCopies){
               static_cast<OpenCLBuffer *>(bufferState->vendorPtr)->copyFromDevice();
               std::cout << "copying arg " << arg->idx <<" from device "<< std::endl;
               if (openclConfig->trace){
                  bufferState->dump("After copy from device");
               }
            }

        }
    }
    status = clWaitForEvents(openclQueue->eventc, openclQueue->events);
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    for (int i = 0; i < openclQueue->eventc; i++) {
        status = clReleaseEvent(openclQueue->events[i]);
        if (status != CL_SUCCESS) {
            std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
            exit(1);
        }
    }
   // delete[] events;
    //eventMax = 0;
    // This should be GUARDED !!!!!!!!!!!!!!!!!
   openclQueue->eventc = 0;
    //events = nullptr;
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            BufferState_s * bufferState = BufferState_s::of(
                      arg->value.buffer.memorySegment,
                      arg->value.buffer.sizeInBytes
            );
            if (!openclConfig->minimizeCopies){
               delete static_cast<OpenCLBuffer *>(bufferState->vendorPtr);
               bufferState->vendorPtr = nullptr;
               if (openclConfig->trace){
                  bufferState->dump("After deleting buffer ");
               }
            }
        }
    }
    return 0;
}

/*
  OpenCLProgram
  */
OpenCLBackend::OpenCLProgram::OpenCLProgram(Backend *backend, BuildInfo *buildInfo, cl_program program)
        : Backend::Program(backend, buildInfo), program(program) {
}

OpenCLBackend::OpenCLProgram::~OpenCLProgram() {
    clReleaseProgram(program);
}

long OpenCLBackend::OpenCLProgram::getKernel(int nameLen, char *name) {
    cl_int status;
    cl_kernel kernel = clCreateKernel(program, name, &status);
    return (long) new OpenCLKernel(this,name, kernel);
}

bool OpenCLBackend::OpenCLProgram::programOK() {
    return true;
}
/*
  OpenCLBackend
  */
bool OpenCLBackend::getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) {
    std::cout << "attempting  to get buffer from device (if dirty) from OpenCLBackend "<<std::endl;
    return true;
}

OpenCLBackend::OpenCLBackend(int mode, int platform, int device )
        : Backend(mode, platform, device, new OpenCLConfig(mode),  new OpenCLQueue()) {
    OpenCLConfig *openclConfig = dynamic_cast<OpenCLConfig *>(config);
     OpenCLQueue *openclQueue = dynamic_cast<OpenCLQueue *>(queue);
    if (openclConfig->trace){
        std::cout << "openclConfig->gpu" << (openclConfig->gpu ? "true" : "false") << std::endl;
        std::cout << "openclConfig->minimizeCopies" << (openclConfig->minimizeCopies ? "true" : "false") << std::endl;
    }
    cl_device_type requestedType =openclConfig->gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

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

    if ((openclQueue->command_queue = clCreateCommandQueue(context, device_ids[0], queue_props, &status)) == NULL ||
        status != CL_SUCCESS) {
        clReleaseContext(context);
        delete[] platforms;
        delete[] device_ids;
        return;
    }

    device_id = device_ids[0];
    delete[] device_ids;
    delete[] platforms;
}

OpenCLBackend::~OpenCLBackend() {
    clReleaseContext(context);
    clReleaseCommandQueue(dynamic_cast<OpenCLQueue *>(queue)->command_queue);
}

void OpenCLBackend::OpenCLProgram::OpenCLKernel::showEvents(int width) {
    OpenCLQueue * openclQueue =  dynamic_cast<OpenCLQueue *>(program->backend->queue);
    cl_ulong *samples = new cl_ulong[4 * openclQueue->eventc]; // queued, submit, start, end
    int sample = 0;
    cl_ulong min;
    cl_ulong max;
    for (int event = 0; event < openclQueue->eventc; event++) {
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

            if ((clGetEventProfilingInfo(openclQueue->events[event], info, sizeof(samples[sample]), &samples[sample], NULL)) !=
                CL_SUCCESS) {
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

    for (int event = 0; event < openclQueue->eventc; event++) {
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
    if (INFO){
       std::cout << "getMaxComputeUnits()" << std::endl;
    }
    cl_uint value;
    cl_int status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(value), &value, nullptr);
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    return value;

}

void OpenCLBackend::info() {
    cl_int status;
    fprintf(stderr, "platform{\n");
    char platformVersionName[512];
    status = clGetPlatformInfo(platform_id, CL_PLATFORM_VERSION, sizeof(platformVersionName), platformVersionName,
                               NULL);
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
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(maxWorkItemDimensions),
                             &maxWorkItemDimensions, NULL);
    fprintf(stderr, "         CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. %u\n", maxWorkItemDimensions);

    size_t *maxWorkItemSizes = new size_t[maxWorkItemDimensions];
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxWorkItemDimensions,
                             maxWorkItemSizes, NULL);
  //  for (unsigned dimIdx = 0; dimIdx < maxWorkItemDimensions; dimIdx++) {
    //    fprintf(stderr, "             dim[%d] = %ld\n", dimIdx, maxWorkItemSizes[dimIdx]);
   // }

    size_t maxWorkGroupSize;
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize,
                             NULL);

    fprintf(stderr, "         CL_DEVICE_MAX_WORK_GROUP_SIZE...... "
    Size_tNewline, maxWorkGroupSize);


    cl_ulong maxMemAllocSize;
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(maxMemAllocSize), &maxMemAllocSize, NULL);
    fprintf(stderr, "         CL_DEVICE_MAX_MEM_ALLOC_SIZE....... "
    LongUnsignedNewline, maxMemAllocSize);

    cl_ulong globalMemSize;
    status = clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
    fprintf(stderr, "         CL_DEVICE_GLOBAL_MEM_SIZE.......... "
    LongUnsignedNewline, globalMemSize);

    cl_ulong localMemSize;
    status = clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(localMemSize), &localMemSize, NULL);
    fprintf(stderr, "         CL_DEVICE_LOCAL_MEM_SIZE........... "
    LongUnsignedNewline, localMemSize);

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
    OpenCLConfig *openclConfig = dynamic_cast<OpenCLConfig*>(config);
    if(openclConfig->trace){
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
    static char unknown[256];
    int ii;

    for (ii = 0; error_table[ii].msg != NULL; ii++) {
        if (error_table[ii].code == status) {
            //std::cerr << " clerror '" << error_table[ii].msg << "'" << std::endl;
            return error_table[ii].msg;
        }
    }
    SNPRINTF(unknown, sizeof(unknown), "unmapped string for  error %d", status);
    return unknown;
}


long getBackend(int mode, int platform, int device) {
  std::cerr << "Opencl Driver mode=" << mode << " platform=" << platform << " device=" << device << std::endl;

    return reinterpret_cast<long>(new OpenCLBackend(mode, platform, device));
}


void __checkOpenclErrors(cl_int status, const char *file, const int line) {
    if (CL_SUCCESS != status) {
        std::cerr << "Opencl Driver API error = " << status << " from file " << file << " line " << line << std::endl;
        exit(-1);
    }
}

