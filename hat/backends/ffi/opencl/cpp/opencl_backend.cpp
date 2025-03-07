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

OpenCLBackend::OpenCLConfig::OpenCLConfig(int mode):
       mode(mode),
       gpu((mode&GPU_BIT)==GPU_BIT),
       cpu((mode&CPU_BIT)==CPU_BIT),
       minimizeCopies((mode&MINIMIZE_COPIES_BIT)==MINIMIZE_COPIES_BIT),
       trace((mode&TRACE_BIT)==TRACE_BIT),
       traceCopies((mode&TRACE_COPIES_BIT)==TRACE_COPIES_BIT),
       info((mode&INFO_BIT)==INFO_BIT),
       showCode((mode&SHOW_CODE_BIT)==SHOW_CODE_BIT),
       profile((mode&PROFILE_BIT)==PROFILE_BIT){
       if (info){
          std::cout << "native show_code " << showCode <<std::endl;
          std::cout <<  "native info " << info<<std::endl;
          std::cout << "native gpu " << gpu<<std::endl;
          std::cout << "native cpu " << cpu<<std::endl;
          std::cout << "native minimizeCopies " << minimizeCopies<<std::endl;
          std::cout << "native trace " << trace<<std::endl;
          std::cout << "native traceCopies " << traceCopies<<std::endl;
          std::cout << "native profile " << profile<<std::endl;
       }
 }
 OpenCLBackend::OpenCLConfig::~OpenCLConfig(){
 }


/*
  OpenCLBuffer
  */

OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::OpenCLBuffer(Backend::Program::Kernel *kernel, Arg_s *arg)
        : Backend::Program::Kernel::Buffer(kernel, arg) {
    cl_int status;
    OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(kernel->program->backend);
    clMem = clCreateBuffer(
        openclBackend->context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
        arg->value.buffer.sizeInBytes,
        arg->value.buffer.memorySegment,
        &status);

    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }

    BufferState_s * bufferState = BufferState_s::of(arg);
     bufferState->vendorPtr =  static_cast<void *>(this);
    if (openclBackend->openclConfig.traceCopies){
        std::cout << "created buffer for arg idx "<< arg->idx << std::endl;
    }

}


void OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::copyToDevice() {
    OpenCLKernel *openclKernel = dynamic_cast<OpenCLKernel *>(kernel);
    OpenCLBackend *openclBackend = dynamic_cast<OpenCLBackend *>(openclKernel->program->backend);
    cl_int status = clEnqueueWriteBuffer(
       openclBackend->openclQueue.command_queue,
       clMem,
       CL_FALSE,
       0,
       arg->value.buffer.sizeInBytes,
       arg->value.buffer.memorySegment,
       openclBackend->openclQueue.eventc,
       openclBackend->openclQueue.eventListPtr(),
       openclBackend->openclQueue.nextEventPtr()
    );
    openclBackend->openclQueue.markAsCopyToDeviceAndInc();

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
       arg->value.buffer.sizeInBytes,
       arg->value.buffer.memorySegment,
       openclBackend->openclQueue.eventc,
       openclBackend->openclQueue.eventListPtr(),
       openclBackend->openclQueue.nextEventPtr()
    );
    openclBackend->openclQueue.markAsCopyFromDeviceAndInc();
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if(openclBackend->openclConfig.traceCopies){
       std::cout << "enqueued buffer for arg idx " << arg->idx << " in OpenCLBuffer::copyFromDevice()" << std::endl;
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
    OpenCLBackend *openclBackend = dynamic_cast<OpenCLBackend*>(program->backend);
    openclBackend->openclQueue.marker(openclBackend->openclQueue.EnterKernelDispatchBits);
    if (openclBackend->openclConfig.trace){
       Sled::show(std::cout, argArray);
    }
    NDRange *ndrange = nullptr;
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
               if (openclBackend->openclConfig.trace){
                  std::cout << "arg["<<i<<"] = "<< std::hex << (int)(arg->value.buffer.access);
                  switch (arg->value.buffer.access){
                      case RO_BYTE: std::cout << " RO";break;
                      case WO_BYTE: std::cout << " WO";break;
                      case RW_BYTE: std::cout << " RW";break;
                      default: std::cout << "JUNK!!!!"; break;
                  }
                  std::cout << std::endl;
               }
               if ((arg->value.buffer.access == RO_BYTE ) || (arg->value.buffer.access == RW_BYTE ) ||(arg->value.buffer.access == WO_BYTE )){
                 // OK
               }else{
                  std::cerr << "arg["<<i<<"] = "<< std::hex << (int)(arg->value.buffer.access) << std::endl;
                  std::exit(1);
               }

               BufferState_s * bufferState = BufferState_s::of(arg);
               OpenCLBuffer * openclBuffer =nullptr;
               if (bufferState->isHostNew()){
                  openclBuffer = new OpenCLBuffer(this, arg);
                  if (openclBackend->openclConfig.trace){
                     std::cout << "We allocated arg "<<i<<" buffer "<<std::endl;
                  }
                  bufferState->clearHostNew();
               }else{
                  if (openclBackend->openclConfig.trace){
                      std::cout << "Were reusing  arg "<<i<<" buffer "<<std::endl;
                  }
                  openclBuffer=  static_cast<OpenCLBuffer*>(bufferState->vendorPtr);
                }
                if (arg->idx == 0){
                    ndrange = static_cast<NDRange *>(arg->value.buffer.memorySegment);
                }
                if (openclBackend->openclConfig.minimizeCopies){
                   // is the buffer GPU dirty. If so we should not need to copy

                     if (bufferState->isDeviceDirty() && bufferState->isHostDirty()){
                           std::cerr <<" WHY is buffer host and device dirty for arg " << arg->idx << "  This should not happen!"<< std::endl;
                           exit(1);
                        }


                    if (bufferState->isHostDirty()){
                       if (openclBackend->openclConfig.traceCopies){
                          std::cout << "HOST is dirty (java side changed code) so copying arg " << arg->idx <<" to device "<< std::endl;
                       }
                       bufferState->clearHostDirty();
                       openclBuffer->copyToDevice();

                    }else{
                       if (openclBackend->openclConfig.traceCopies){
                           std::cout << "HOST is not dirty (java side has not changed code) so not copying arg " << arg->idx <<" to device "<< std::endl;
                       }
                    }

                }else{
                    if (openclBackend->openclConfig.traceCopies){
                        std::cout << "copying arg " << arg->idx <<" to device "<< std::endl;
                    }
                    openclBuffer->copyToDevice();
                }
                cl_int status = clSetKernelArg(kernel, arg->idx, sizeof(cl_mem), &openclBuffer->clMem);
                if (status != CL_SUCCESS) {
                    std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                    exit(1);
                }
                if (openclBackend->openclConfig.trace){
                   std::cout << "set buffer arg " << arg->idx << std::endl;
                }
                break;
            }
             case 'B':
             case 'S':
             case 'C':
             case 'I':
             case 'F':
             case 'J':
             case 'D':
             {
                cl_int status = clSetKernelArg(kernel, arg->idx, arg->size(), (void *) &arg->value);
                if (status != CL_SUCCESS) {
                    std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                    exit(1);
                }
                if (openclBackend->openclConfig.trace){
                   std::cerr << "set " <<arg->variant << " " << arg->idx << std::endl;
                }
                break;
            }
            default: {
                std::cerr << "unexpected variant setting args in OpenCLkernel::ndrange " << (char) arg->variant << std::endl;
                exit(1);
            }
        }
    }

    size_t globalSize = ndrange->maxX;
    if (openclBackend->openclConfig.trace){
       std::cout << "ndrange = " << ndrange->maxX << std::endl;
    }
    size_t dims = 1;
    cl_int status = clEnqueueNDRangeKernel(
            openclBackend->openclQueue.command_queue,
            kernel,
            dims,
            nullptr,
            &globalSize,
            nullptr,
            openclBackend->openclQueue.eventc,
            openclBackend->openclQueue.eventListPtr(),
            openclBackend->openclQueue.nextEventPtr());
    openclBackend->openclQueue.markAsNDRangeAndInc();
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if (openclBackend->openclConfig.trace){
       std::cout << "enqueued kernel dispatch globalSize=" << globalSize << std::endl;
    }
   if (openclBackend->openclConfig.minimizeCopies){
     openclBackend->openclQueue.wait();
   }else{
       for (int i = 1; i < argSled.argc(); i++) { // note i = 1... we don't need to copy back the KernelContext
          Arg_s *arg = argSled.arg(i);
          if (arg->variant == '&') {
             BufferState_s * bufferState = BufferState_s::of(arg );
             static_cast<OpenCLBuffer *>(bufferState->vendorPtr)->copyFromDevice();
             if (openclBackend->openclConfig.traceCopies){
                std::cout << "copying arg " << arg->idx <<" from device "<< std::endl;
                bufferState->dump("After copy from device");
             }
             bufferState->setDeviceDirty();
          }
       }
         openclBackend->openclQueue.wait();
    }
      openclBackend->openclQueue.marker(openclBackend->openclQueue.LeaveKernelDispatchBits);
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

   // if (openclConfig->trace){
       if (openclConfig.minimizeCopies){
         std::cout << "attempting  to get buffer from device (if dirty) from OpenCLBackend "<<std::endl;
       //}else{
       //  std::cout << "skipping attempt  to get buffer from device (if dirty) from OpenCLBackend (we are not minimizing copies) "<<std::endl;
       }
   // }

    return true;
}

OpenCLBackend::OpenCLBackend(int mode, int platform, int device )
        : Backend(mode), openclConfig(mode), openclQueue(this) {
    if (openclConfig.trace){
        std::cout << "openclConfig->gpu" << (openclConfig.gpu ? "true" : "false") << std::endl;
        std::cout << "openclConfig->minimizeCopies" << (openclConfig.minimizeCopies ? "true" : "false") << std::endl;
    }
    cl_device_type requestedType =openclConfig.gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

    cl_int status;
    cl_uint platformc = 0;
    if ((status = clGetPlatformIDs(0, NULL, &platformc)) != CL_SUCCESS) {
        if (status != CL_SUCCESS){
           std::cerr << "clGetPlatformIDs (to get count) failed " << errorMsg(status)<<std::endl;
        }
        return;
    }
    cl_platform_id *platforms = new cl_platform_id[platformc];
    if ((status = clGetPlatformIDs(platformc, platforms, NULL)) != CL_SUCCESS) {
         if (status != CL_SUCCESS){
            std::cerr << "clGetPlatformIDs failed " << errorMsg(status)<<std::endl;
         }
        return;
    }

    cl_uint devicec = 0;
    for (unsigned int i = 0; devicec == 0 && i < platformc; ++i) {
        platform_id = platforms[i];
        if ((status = clGetDeviceIDs(platform_id, requestedType, 0, NULL, &devicec)) != CL_SUCCESS) {
            if (status != CL_SUCCESS){
               std::cerr << "clGetDeviceIDs (to get count) failed " << errorMsg(status)<<std::endl;
            }
            delete[] platforms;
            return;
        }
    }
    if (devicec == 0) {
        status = CL_DEVICE_NOT_AVAILABLE;
        std::cerr << "No device available " << errorMsg(status)<<std::endl;
        return;
    }
    cl_device_id *device_ids = new cl_device_id[devicec];             // compute device id
    if ((status = clGetDeviceIDs(platform_id, requestedType, devicec, device_ids, NULL)) != CL_SUCCESS) {

        std::cerr << "clGetDeviceIDs failed " << errorMsg(status)<<std::endl;
        delete[] platforms;
        delete[] device_ids;
        return;
    }
    if ((context = clCreateContext(0, 1, device_ids, NULL, NULL, &status)) == NULL || status != CL_SUCCESS) {
        std::cerr << "clCreateContext failed " << errorMsg(status)<<std::endl;
        delete[] platforms;
        delete[] device_ids;
        return;
    }

    cl_command_queue_properties queue_props = CL_QUEUE_PROFILING_ENABLE;

    if ((openclQueue.command_queue = clCreateCommandQueue(context, device_ids[0], queue_props, &status)) == NULL ||
        status != CL_SUCCESS) {
        std::cerr << "clCreateCommandQueue failed " << errorMsg(status)<<std::endl;
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

}
/*
  static char *strInfo(cl_device_id device_id, cl_device_info device_info){
     size_t sz;
     cl_int  status = clGetDeviceInfo(device_id, device_info, 0, nullptr,  &sz);
     char *ptr = new char[sz+1];
     status = clGetDeviceInfo(device_id, device_info, sz, ptr,nullptr);
     return ptr;
  }

  static cl_int cl_int_info(cl_device_id device_id, cl_device_info device_info){
     cl_uint v;
     cl_int status = clGetDeviceInfo(device_id, device_info, sizeof(v), &v, nullptr);
     return v;
  }
  static cl_ulong cl_ulong_info(cl_device_id device_id, cl_device_info device_info){
     cl_ulong v;
     cl_int status = clGetDeviceInfo(device_id, device_info, sizeof(v), &v, nullptr);
     return v;
  }
  static size_t size_t_info(cl_device_id device_id, cl_device_info device_info){
     size_t v;
     cl_int status = clGetDeviceInfo(device_id, device_info, sizeof(v), &v, nullptr);
     return v;
  }

  static char *strInfo(cl_platform_id platform_id,cl_platform_info platform_info){
       size_t sz;
       cl_int  status = clGetPlatformInfo(platform_id, platform_info, 0, nullptr,  &sz);
       char *ptr = new char[sz+1];
       status = clGetPlatformInfo(platform_id, platform_info, sz, ptr,nullptr);
       return ptr;
  }
  */
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


long getOpenCLBackend(int mode, int platform, int device, int unused) {
 // std::cerr << "Opencl Driver mode=" << mode << " platform=" << platform << " device=" << device << std::endl;

    return reinterpret_cast<long>(new OpenCLBackend(mode, platform, device));
}


void __checkOpenclErrors(cl_int status, const char *file, const int line) {
    if (CL_SUCCESS != status) {
        std::cerr << "Opencl Driver API error = " << status << " from file " << file << " line " << line << std::endl;
        exit(-1);
    }
}

