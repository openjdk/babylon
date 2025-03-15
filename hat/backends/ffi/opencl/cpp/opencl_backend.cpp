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
#define opencl_backend_cpp
#include "opencl_backend.h"

OpenCLBackend::OpenCLConfig::OpenCLConfig(int configBits):
       configBits(configBits),
       minimizeCopies((configBits&MINIMIZE_COPIES_BIT)==MINIMIZE_COPIES_BIT),
       alwaysCopy(!minimizeCopies),
       trace((configBits&TRACE_BIT)==TRACE_BIT),
       traceCopies((configBits&TRACE_COPIES_BIT)==TRACE_COPIES_BIT),
       traceEnqueues((configBits&TRACE_ENQUEUES_BIT)==TRACE_ENQUEUES_BIT),
       traceCalls((configBits&TRACE_CALLS_BIT)==TRACE_CALLS_BIT),
       traceSkippedCopies((configBits&TRACE_SKIPPED_COPIES_BIT)==TRACE_SKIPPED_COPIES_BIT),
       info((configBits&INFO_BIT)==INFO_BIT),
       showCode((configBits&SHOW_CODE_BIT)==SHOW_CODE_BIT),
       profile((configBits&PROFILE_BIT)==PROFILE_BIT),
       showWhy((configBits&SHOW_WHY_BIT)==SHOW_WHY_BIT),
       useState((configBits&USE_STATE_BIT)==USE_STATE_BIT),
       showState((configBits&SHOW_STATE_BIT)==SHOW_STATE_BIT),

       platform((configBits&0xf)),
       device((configBits&0xf0)>>4){
       if (info){
          std::cout << "native showCode " << showCode <<std::endl;
          std::cout << "native info " << info<<std::endl;
          std::cout << "native minimizeCopies " << minimizeCopies<<std::endl;
          std::cout << "native alwaysCopy " << alwaysCopy<<std::endl;
          std::cout << "native trace " << trace<<std::endl;
          std::cout << "native traceSkippedCopies " << traceSkippedCopies<<std::endl;
          std::cout << "native traceCalls " << traceCalls<<std::endl;
          std::cout << "native traceCopies " << traceCopies<<std::endl;
          std::cout << "native traceEnqueues " << traceEnqueues<<std::endl;
          std::cout << "native profile " << profile<<std::endl;
          std::cout << "native showWhy " << showWhy<<std::endl;
           std::cout << "native useState " << useState<<std::endl;
            std::cout << "native showState " << showState<<std::endl;
          std::cout << "native platform " << platform<<std::endl;
          std::cout << "native device " << device<<std::endl;
       }
 }
 OpenCLBackend::OpenCLConfig::~OpenCLConfig(){
 }


/*
  OpenCLBuffer
  */

OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::OpenCLBuffer(Backend::Program::Kernel *kernel, Arg_s *arg, BufferState_s *bufferState)
        : Backend::Program::Kernel::Buffer(kernel, arg), bufferState(bufferState) {
    cl_int status;
    OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(kernel->program->backend);
    clMem = clCreateBuffer(
        openclBackend->context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
        bufferState->length,// arg->value.buffer.sizeInBytes,
        arg->value.buffer.memorySegment,
        &status);

    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
     bufferState->vendorPtr =  static_cast<void *>(this);
    if (openclBackend->openclConfig.traceCopies){
        std::cout << "created buffer for arg idx "<< arg->idx << std::endl;
    }

}

bool OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::shouldCopyToDevice( Arg_s *arg){
   OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(kernel->program->backend);
   bool kernelReadsFromThisArg = (arg->value.buffer.access==RW_BYTE) || (arg->value.buffer.access==RO_BYTE);
   bool isHostDirtyOrNew = bufferState->isHostDirty() | bufferState->isHostNew();

   bool result=  (kernelReadsFromThisArg & isHostDirtyOrNew);

   if (openclBackend->openclConfig.showWhy){
     std::cout<<
          "config.alwaysCopy="<<openclBackend->openclConfig.alwaysCopy
          << " | arg.RW="<<(arg->value.buffer.access==RW_BYTE)
          << " | arg.RO="<<(arg->value.buffer.access==RO_BYTE)
          << " | kernel.needsToRead="<<  kernelReadsFromThisArg
          << " | buffer.hostDirty="<< bufferState->isHostDirty()
          << " | buffer.hostNew="<< bufferState->isHostNew()
          << " | buffer.deviceDirty="<< bufferState->isDeviceDirty()
          <<" so "
            ;
    }
    if (result && bufferState->isDeviceDirty()){
            result= false;
      }
   return openclBackend->openclConfig.alwaysCopy |result;
}
bool OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::shouldCopyFromDevice(Arg_s *arg){
   OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(kernel->program->backend);

   bool kernelWroteToThisArg = (arg->value.buffer.access==WO_BYTE) |  (arg->value.buffer.access==RW_BYTE);
   bool result = kernelWroteToThisArg;
   if (openclBackend->openclConfig.showWhy){
       std::cout<<
         "config.alwaysCopy="<<openclBackend->openclConfig.alwaysCopy
            << " | arg.WO="<<(arg->value.buffer.access==WO_BYTE)
            << " | arg.RW="<<(arg->value.buffer.access==RW_BYTE)
            << " | kernel.wroteToThisArg="<<  kernelWroteToThisArg
            <<" so " ;
      }
   return openclBackend->openclConfig.alwaysCopy;
}


void OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::copyToDevice() {
    OpenCLKernel *openclKernel = dynamic_cast<OpenCLKernel *>(kernel);
    OpenCLBackend *openclBackend = dynamic_cast<OpenCLBackend *>(openclKernel->program->backend);
    cl_int status = clEnqueueWriteBuffer(
       openclBackend->openclQueue.command_queue,
       clMem,
       CL_FALSE,
       0,
       bufferState->length, // arg->value.buffer.sizeInBytes,
       arg->value.buffer.memorySegment,
       openclBackend->openclQueue.eventc,
       openclBackend->openclQueue.eventListPtr(),
       openclBackend->openclQueue.nextEventPtr()
    );
    openclBackend->openclQueue.markAsCopyToDeviceAndInc(arg->idx);

    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if(openclBackend->openclConfig.traceCopies){
        std::cout << "enqueued buffer for arg idx " << arg->idx << " in OpenCLBuffer::copyToDevice()" << std::endl;
    }
}

void OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::copyFromDevice() {
    OpenCLKernel * openclKernel = dynamic_cast<OpenCLKernel *>(kernel);
    OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(openclKernel->program->backend);

    cl_int status = clEnqueueReadBuffer(
       openclBackend->openclQueue.command_queue,
       clMem,
       CL_FALSE,
       0,
       bufferState->length,//arg->value.buffer.sizeInBytes,
       arg->value.buffer.memorySegment,
       openclBackend->openclQueue.eventc,
       openclBackend->openclQueue.eventListPtr(),
       openclBackend->openclQueue.nextEventPtr()
    );
    openclBackend->openclQueue.markAsCopyFromDeviceAndInc(arg->idx);
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if(openclBackend->openclConfig.traceCopies){
       std::cout << "enqueued buffer for arg idx " << arg->idx << " in OpenCLBuffer::copyFromDevice()" << std::endl;
    }

   // std::cout << "setting device dirty"<<std::endl;
    bufferState->setDeviceDirty();

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
    if (status != CL_SUCCESS){
       std::cerr << "Failed to get kernel "<<name<<" "<<errorMsg(status)<<std::endl;
    }
    return (long) new OpenCLKernel(this,name, kernel);
}

bool OpenCLBackend::OpenCLProgram::programOK() {
    return true;
}
/*
  OpenCLBackend
  */
bool OpenCLBackend::getBufferFromDeviceIfDirty(void *memorySegment, long memorySegmentLength) {
    if (openclConfig.minimizeCopies){
       BufferState_s * bufferState = BufferState_s::of(memorySegment,memorySegmentLength);
       if (bufferState->isDeviceDirty()){
          std::cout << "from getBufferFromDeviceIfDirty Buffer is device dirty so attempting to get buffer from device from OpenCLBackend "<<std::endl;
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

   char *OpenCLBackend::strInfo( cl_device_info device_info){
     size_t sz;
     cl_int  status = clGetDeviceInfo(device_id, device_info, 0, nullptr,  &sz);
     char *ptr = new char[sz+1];
     status = clGetDeviceInfo(device_id, device_info, sz, ptr,nullptr);
     return ptr;
  }

   cl_int OpenCLBackend::cl_int_info( cl_device_info device_info){
     cl_uint v;
     cl_int status = clGetDeviceInfo(device_id, device_info, sizeof(v), &v, nullptr);
     return v;
  }
   cl_ulong OpenCLBackend::cl_ulong_info( cl_device_info device_info){
     cl_ulong v;
     cl_int status = clGetDeviceInfo(device_id, device_info, sizeof(v), &v, nullptr);
     return v;
  }
   size_t OpenCLBackend::size_t_info( cl_device_info device_info){
     size_t v;
     cl_int status = clGetDeviceInfo(device_id, device_info, sizeof(v), &v, nullptr);
     return v;
  }

  char *OpenCLBackend::strPlatformInfo(cl_platform_info platform_info){
       size_t sz;
       cl_int  status = clGetPlatformInfo(platform_id, platform_info, 0, nullptr,  &sz);
       char *ptr = new char[sz+1];
       status = clGetPlatformInfo(platform_id, platform_info, sz, ptr,nullptr);
       return ptr;
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

struct PlatformInfo{
  OpenCLBackend *openclBackend;
  char *versionName;
  char *vendorName;
  char *name;

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

  DeviceInfo(OpenCLBackend *openclBackend):
       openclBackend(openclBackend),
       maxComputeUnits(openclBackend->cl_int_info( CL_DEVICE_MAX_COMPUTE_UNITS)),
       maxWorkItemDimensions(openclBackend->cl_int_info( CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)),
       maxWorkGroupSize(openclBackend->size_t_info( CL_DEVICE_MAX_WORK_GROUP_SIZE)),
       maxWorkItemSizes( new size_t[maxWorkItemDimensions]),
       maxMemAllocSize(openclBackend->cl_ulong_info(CL_DEVICE_MAX_MEM_ALLOC_SIZE)),
       globalMemSize(openclBackend->cl_ulong_info( CL_DEVICE_GLOBAL_MEM_SIZE)),
       localMemSize(openclBackend->cl_ulong_info( CL_DEVICE_LOCAL_MEM_SIZE)),
       profile(openclBackend->strInfo( CL_DEVICE_PROFILE)),
       deviceVersion(openclBackend->strInfo(  CL_DEVICE_VERSION)),
       driverVersion(openclBackend->strInfo(  CL_DRIVER_VERSION)),
       cVersion(openclBackend->strInfo(  CL_DEVICE_OPENCL_C_VERSION)),
       name(openclBackend->strInfo(  CL_DEVICE_NAME)),
       extensions(openclBackend->strInfo(  CL_DEVICE_EXTENSIONS)),
       builtInKernels(openclBackend->strInfo( CL_DEVICE_BUILT_IN_KERNELS)){

       clGetDeviceInfo(openclBackend->device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * maxWorkItemDimensions, maxWorkItemSizes, NULL);
       clGetDeviceInfo(openclBackend->device_id, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
       char buf[512];
       buf[0]='\0';
       if (CL_DEVICE_TYPE_CPU == (deviceType & CL_DEVICE_TYPE_CPU)) {
          std::strcat(buf, "CPU ");
       }
       if (CL_DEVICE_TYPE_GPU == (deviceType & CL_DEVICE_TYPE_GPU)) {
          std::strcat(buf, "GPU ");
       }
       if (CL_DEVICE_TYPE_ACCELERATOR == (deviceType & CL_DEVICE_TYPE_ACCELERATOR)) {
          std::strcat(buf, "ACC ");
       }
       deviceTypeStr = new char[std::strlen(buf)];
       std::strcpy(deviceTypeStr, buf);
  }
  ~DeviceInfo(){
     delete [] deviceTypeStr;
     delete [] profile;
     delete [] deviceVersion;
     delete [] driverVersion;
     delete [] cVersion;
     delete [] name;
     delete [] extensions;
     delete [] builtInKernels;
     delete [] maxWorkItemSizes;
  }
};
  DeviceInfo deviceInfo;
  PlatformInfo(OpenCLBackend *openclBackend):
     openclBackend(openclBackend),
     versionName(openclBackend->strPlatformInfo(CL_PLATFORM_VERSION)),
     vendorName(openclBackend->strPlatformInfo(CL_PLATFORM_VENDOR)),
     name(openclBackend->strPlatformInfo(CL_PLATFORM_NAME)),
     deviceInfo(openclBackend){
  }
  ~PlatformInfo(){
     delete [] versionName;
     delete [] vendorName;
     delete [] name;
  }
};

void OpenCLBackend::info() {
    PlatformInfo platformInfo(this);
    cl_int status;
    std::cerr << "platform{" <<std::endl;
    std::cerr << "   CL_PLATFORM_VENDOR..\"" << platformInfo.vendorName <<"\""<<std::endl;
    std::cerr << "   CL_PLATFORM_VERSION.\"" << platformInfo.versionName <<"\""<<std::endl;
    std::cerr << "   CL_PLATFORM_NAME....\"" << platformInfo.name <<"\""<<std::endl;
    std::cerr << "         CL_DEVICE_TYPE..................... " <<  platformInfo.deviceInfo.deviceTypeStr << " "<<  platformInfo.deviceInfo.deviceType<<std::endl;
    std::cerr << "         CL_DEVICE_MAX_COMPUTE_UNITS........ " <<  platformInfo.deviceInfo.maxComputeUnits<<std::endl;
    std::cerr << "         CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. " <<  platformInfo.deviceInfo.maxWorkItemDimensions << " {";
    for (unsigned dimIdx = 0; dimIdx <  platformInfo.deviceInfo.maxWorkItemDimensions; dimIdx++) {
        std::cerr<<  platformInfo.deviceInfo.maxWorkItemSizes[dimIdx] << " ";
    }
    std::cerr<< "}"<<std::endl;
     std::cerr <<  "         CL_DEVICE_MAX_WORK_GROUP_SIZE...... "<<  platformInfo.deviceInfo.maxWorkGroupSize<<std::endl;
     std::cerr <<  "         CL_DEVICE_MAX_MEM_ALLOC_SIZE....... "<<  platformInfo.deviceInfo.maxMemAllocSize<<std::endl;
     std::cerr <<  "         CL_DEVICE_GLOBAL_MEM_SIZE.......... "<<  platformInfo.deviceInfo.globalMemSize<<std::endl;
     std::cerr <<  "         CL_DEVICE_LOCAL_MEM_SIZE........... "<<  platformInfo.deviceInfo.localMemSize<<std::endl;
     std::cerr <<  "         CL_DEVICE_PROFILE.................. "<<  platformInfo.deviceInfo.profile<<std::endl;
     std::cerr <<  "         CL_DEVICE_VERSION.................. "<<  platformInfo.deviceInfo.deviceVersion<<std::endl;
     std::cerr <<  "         CL_DRIVER_VERSION.................. "<<  platformInfo.deviceInfo.driverVersion<<std::endl;
     std::cerr <<  "         CL_DEVICE_OPENCL_C_VERSION......... "<<  platformInfo.deviceInfo.cVersion<<std::endl;
     std::cerr <<  "         CL_DEVICE_NAME..................... "<<  platformInfo.deviceInfo.name<<std::endl;
     std::cerr <<  "         CL_DEVICE_EXTENSIONS............... "<<  platformInfo.deviceInfo.extensions<<std::endl;
     std::cerr <<  "         CL_DEVICE_BUILT_IN_KERNELS......... "<<  platformInfo.deviceInfo.builtInKernels<<std::endl;
     std::cerr <<  "}"<<std::endl;
}

int OpenCLBackend::getMaxComputeUnits() {
    PlatformInfo platformInfo(this);
    return platformInfo.deviceInfo.maxComputeUnits;
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

