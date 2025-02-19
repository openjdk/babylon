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
//#include <iomanip>
#ifdef __APPLE__
    #define LongUnsignedNewline "%llu\n"
    #define Size_tNewline "%lu\n"
    #define LongHexNewline "(0x%llx)\n"
 //  #define alignedMalloc(size, alignment) memalign(alignment, size)
#else
    #include <malloc.h>
    #define LongHexNewline "(0x%lx)\n"
    #define LongUnsignedNewline "%lu\n"
    #define Size_tNewline "%lu\n"
    #if defined (_WIN32)
        #include "windows.h"
     //   #define alignedMalloc(size, alignment) _aligned_malloc(size, alignment)
    #else
     //  #define alignedMalloc(size, alignment) memalign(alignment, size)
    #endif
#endif

OpenCLBackend::OpenCLConfig::OpenCLConfig(int mode):
       mode(mode),
       gpu((mode&GPU_BIT)==GPU_BIT),
       cpu((mode&CPU_BIT)==CPU_BIT),
       minimizeCopies((mode&MINIMIZE_COPIES_BIT)==MINIMIZE_COPIES_BIT),
       trace((mode&TRACE_BIT)==TRACE_BIT),
       profile((mode&PROFILE_BIT)==PROFILE_BIT){
       printf("native gpu %d\n",gpu);
       printf("native cpu %d\n",cpu);
       printf("native minimizeCopies %d\n", minimizeCopies);
       printf("native trace %d\n", trace);
       printf("native profile %d\n",profile);
 }
 OpenCLBackend::OpenCLConfig::~OpenCLConfig(){
 }

 OpenCLBackend::OpenCLQueue::OpenCLQueue()
  : eventMax(256), events(new cl_event[eventMax]), eventc(0){
 }

 cl_event *OpenCLBackend::OpenCLQueue::eventListPtr(){
   return (eventc == 0) ? nullptr : events;
  }
 cl_event *OpenCLBackend::OpenCLQueue::nextEventPtr(){
              return &events[eventc];
 }

void OpenCLBackend::OpenCLQueue::showEvents(int width) {

    cl_ulong *samples = new cl_ulong[4 * eventc]; // queued, submit, start, end
    int sample = 0;
    cl_ulong min;
    cl_ulong max;
    cl_profiling_info profiling_info_arr[]={CL_PROFILING_COMMAND_QUEUED,CL_PROFILING_COMMAND_SUBMIT,CL_PROFILING_COMMAND_START,CL_PROFILING_COMMAND_END} ;
    const char* profiling_info_name_arr[]={"CL_PROFILING_COMMAND_QUEUED","CL_PROFILING_COMMAND_SUBMIT","CL_PROFILING_COMMAND_START","CL_PROFILING_COMMAND_END"} ;

    for (int event = 0; event < eventc; event++) {
        for (int type = 0; type < 4; type++) {
            if ((clGetEventProfilingInfo(events[event], profiling_info_arr[type], sizeof(samples[sample]), &samples[sample], NULL)) !=
                CL_SUCCESS) {
                std::cerr << "failed to get profile info " << profiling_info_name_arr[type] << std::endl;
            }
            if (sample == 0) {
                if (type == 0){
                   min = max = samples[sample];
                }
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
    std::cout << "Range: " <<min<< "-" <<max<< "("<< range << "ns)"
        <<  "  (" << scale << "ns) per char"
        << " +:submitted, #:started, =:end  "<< std::endl;

    for (int event = 0; event < eventc; event++) {
        cl_command_type command_type;
        clGetEventInfo(events[event],CL_EVENT_COMMAND_TYPE,sizeof(command_type), &command_type, nullptr);
        switch (command_type){
          case CL_COMMAND_NDRANGE_KERNEL: std::cout <<   "kernel "; break;
          case CL_COMMAND_READ_BUFFER: std::cout <<    "  read "; break;
          case CL_COMMAND_WRITE_BUFFER: std::cout << " write "; break;
          default: std::cout <<                    " other "; break;
        }
        long eventStart=samples[sample];
        cl_ulong queue = (samples[sample++] - min) / scale;
        cl_ulong submit = (samples[sample++] - min) / scale;
        cl_ulong start = (samples[sample++] - min) / scale;
        long eventEnd=samples[sample];
        cl_ulong end = (samples[sample++] - min) / scale;

        std::cout << std::setw(8)<< (eventEnd-eventStart) << "(ns) ";
        for (int c = 0; c < width; c++) {
            char ch = ' ';
            if (c >= queue && c<submit) {
                ch = '+';
            }else if (c>= submit && c<start){
                ch = '#';
            }else if (c>= start && c<end){
                ch = '=';
            }
            std::cout << ch;
        }
        std::cout << std::endl;
    }
    delete[] samples;
}
 void OpenCLBackend::OpenCLQueue::wait(){
     cl_int status = clWaitForEvents(eventc, events);
      if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
       }
 }
  void OpenCLBackend::OpenCLQueue::release(){
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

     OpenCLBackend::OpenCLQueue::~OpenCLQueue(){
              clReleaseCommandQueue(command_queue);
              delete []events;
             }

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

    BufferState_s * bufferState = BufferState_s::of(arg);
    if (openclBackend->openclConfig.trace){
       bufferState->dump("on allocation before assign");
    }
    bufferState->vendorPtr =  static_cast<void *>(this);
    if (openclBackend->openclConfig.trace){
        bufferState->dump("after assign ");
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
    cl_int status = clEnqueueWriteBuffer( openclBackend->openclQueue.command_queue,
                                         clMem,
                                         CL_FALSE,
                                         0,
                                         arg->value.buffer.sizeInBytes,
                                         arg->value.buffer.memorySegment,
                                         openclBackend->openclQueue.eventc,
                                         openclBackend->openclQueue.eventListPtr(),
                                         openclBackend->openclQueue.nextEventPtr()
                                       );
    openclBackend->openclQueue.eventc++;

    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }

    if(openclBackend->openclConfig.trace){
        std::cout << "enqueued buffer copyToDevice " << std::endl;
    }
}

void OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLBuffer::copyFromDevice() {
    auto openclKernel = dynamic_cast<OpenCLKernel *>(kernel);
    auto openclBackend = dynamic_cast<OpenCLBackend *>(openclKernel->program->backend);

    cl_int status = clEnqueueReadBuffer( openclBackend->openclQueue.command_queue,
                                        clMem,
                                        CL_FALSE,
                                        0,
                                        arg->value.buffer.sizeInBytes,
                                        arg->value.buffer.memorySegment,
                                        openclBackend->openclQueue.eventc,
                                        openclBackend->openclQueue.eventListPtr(),
                                        openclBackend->openclQueue.nextEventPtr()
                                        );
    openclBackend->openclQueue.eventc++;
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if(openclBackend->openclConfig.trace){
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
    OpenCLBackend *openclBackend = dynamic_cast<OpenCLBackend*>(program->backend);
    if (openclBackend->openclConfig.trace){
       Sled::show(std::cout, argArray);
    }
    NDRange *ndrange = nullptr;
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
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
                if (!openclBackend->openclConfig.minimizeCopies){
                    openclBuffer->copyToDevice();
                    if (openclBackend->openclConfig.trace){
                        std::cout << "copying arg " << arg->idx <<" to device "<< std::endl;
                    }
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
    openclBackend->openclQueue.eventc++;
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if (openclBackend->openclConfig.trace){
       std::cout << "enqueued dispatch  " << std::endl;
       std::cout <<  " globalSize=" << globalSize << " " << std::endl;
    }

    for (int i = 1; i < argSled.argc(); i++) { // We don't need to copy back the KernelContext
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            BufferState_s * bufferState = BufferState_s::of(arg );
            if (!openclBackend->openclConfig.minimizeCopies){
               static_cast<OpenCLBuffer *>(bufferState->vendorPtr)->copyFromDevice();
               if (openclBackend->openclConfig.trace){
                std::cout << "copying arg " << arg->idx <<" from device "<< std::endl;
                  bufferState->dump("After copy from device");
               }
            }
        }
    }
    openclBackend->openclQueue.wait();
   // openclBackend->openclQueue.release(); release in computeEnd

    /* NOte that we have leaked a clmem in the OpenCLBuffer attached to the Arg. **/
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
        : Backend(mode), openclConfig(mode), openclQueue() {


    if (openclConfig.trace){
        std::cout << "openclConfig->gpu" << (openclConfig.gpu ? "true" : "false") << std::endl;
        std::cout << "openclConfig->minimizeCopies" << (openclConfig.minimizeCopies ? "true" : "false") << std::endl;
    }
    cl_device_type requestedType =openclConfig.gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;

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

    if ((openclQueue.command_queue = clCreateCommandQueue(context, device_ids[0], queue_props, &status)) == NULL ||
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

}


int OpenCLBackend::getMaxComputeUnits() {
    if (openclConfig.trace){
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
void OpenCLBackend::computeStart() {
  if (openclConfig.trace){
     std::cout <<"compute start" <<std::endl;
  }
}
void OpenCLBackend::computeEnd() {
 if (openclConfig.profile){
     openclQueue.showEvents(100);
 }
 openclQueue.release();
 if (openclConfig.trace){
     std::cout <<"compute end" <<std::endl;
 }
}

struct PlatformInfo{
static char *str(cl_platform_id platform_id,cl_platform_info platform_info){
     size_t sz;
     cl_int  status = clGetPlatformInfo(platform_id, platform_info, 0, nullptr,  &sz);
     char *ptr = new char[sz+1];
     status = clGetPlatformInfo(platform_id, platform_info, sz, ptr,nullptr);
     return ptr;
}
  char *versionName;
  char *vendorName;
  char *name;
  PlatformInfo(cl_platform_id platform_id):
     versionName(str(platform_id, CL_PLATFORM_VERSION)),
     vendorName(str(platform_id, CL_PLATFORM_VENDOR)),
     name(str(platform_id, CL_PLATFORM_NAME)){
  }
  ~PlatformInfo(){
     delete [] versionName;
     delete [] vendorName;
     delete [] name;
  }
};

struct DeviceInfo{
static char *str(cl_device_id device_id, cl_device_info device_info){
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
  cl_int maxComputeUnits;
  cl_int maxWorkItemDimensions;
  cl_device_type deviceType;
  char *deviceTypeStr;

  DeviceInfo(cl_device_id device_id):
       maxComputeUnits(cl_int_info(device_id, CL_DEVICE_MAX_COMPUTE_UNITS)),
        maxWorkItemDimensions(cl_int_info(device_id, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS)) {
       clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, NULL);
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
  }
};
void OpenCLBackend::info() {
    PlatformInfo platformInfo(platform_id);
    cl_int status;
    fprintf(stderr, "platform{\n");
    fprintf(stderr, "   CL_PLATFORM_VENDOR..\"%s\"\n", platformInfo.vendorName);
    fprintf(stderr, "   CL_PLATFORM_VERSION.\"%s\"\n", platformInfo.versionName);
    fprintf(stderr, "   CL_PLATFORM_NAME....\"%s\"\n", platformInfo.name);
    DeviceInfo deviceInfo(device_id);
    fprintf(stderr, "         CL_DEVICE_TYPE..................... %s ", deviceInfo.deviceTypeStr);
    fprintf(stderr, LongHexNewline, deviceInfo.deviceType);
    fprintf(stderr, "         CL_DEVICE_MAX_COMPUTE_UNITS........ %u\n", deviceInfo.maxComputeUnits);
    fprintf(stderr, "         CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. %u\n", deviceInfo.maxWorkItemDimensions);

    size_t *maxWorkItemSizes = new size_t[deviceInfo.maxWorkItemDimensions];
    status = clGetDeviceInfo(device_id, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * deviceInfo.maxWorkItemDimensions,
                             maxWorkItemSizes, NULL);
    for (unsigned dimIdx = 0; dimIdx < deviceInfo.maxWorkItemDimensions; dimIdx++) {
        fprintf(stderr, "             dim[%d] = %ld\n", dimIdx, maxWorkItemSizes[dimIdx]);
    }

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
    static char unknown[256];
    int ii;

    for (ii = 0; error_table[ii].msg != NULL; ii++) {
        if (error_table[ii].code == status) {
            //std::cerr << " clerror '" << error_table[ii].msg << "'" << std::endl;
            return error_table[ii].msg;
        }
    }
     #if defined (_WIN32)
        _snprintf
     #else
        snprintf
     #endif
     (unknown, sizeof(unknown), "unmapped string for  error %d", status);
    return unknown;
}


long getOpenCLBackend(int mode, int platform, int device, int unused) {
  std::cerr << "Opencl Driver mode=" << mode << " platform=" << platform << " device=" << device << std::endl;

    return reinterpret_cast<long>(new OpenCLBackend(mode, platform, device));
}


void __checkOpenclErrors(cl_int status, const char *file, const int line) {
    if (CL_SUCCESS != status) {
        std::cerr << "Opencl Driver API error = " << status << " from file " << file << " line " << line << std::endl;
        exit(-1);
    }
}

