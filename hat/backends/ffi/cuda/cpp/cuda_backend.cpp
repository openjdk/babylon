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

#include <sys/wait.h>
#include <chrono>
#include "cuda_backend.h"

Text::Text(size_t len, char *text, bool isCopy)
        : len(len), text(text), isCopy(isCopy) {
    std::cout << "in Text len="<<len<<" isCopy="<<isCopy << std::endl;
}
Text::Text(char *text, bool isCopy)
        : len(std::strlen(text)), text(text), isCopy(isCopy) {
    std::cout << "in Text len="<<len<<" isCopy="<<isCopy << std::endl;
}
Text::Text(size_t len)
        : len(len), text(len > 0 ? new char[len] : nullptr), isCopy(true) {
    std::cout << "in Text len="<<len<<" isCopy="<<isCopy << std::endl;
}
Text::~Text(){
    if (isCopy && text){
        delete[] text;
    }
    text = nullptr;
    isCopy = false;
    len = 0;
}

PtxSource::PtxSource()
        : Text(0L) {
}
PtxSource::PtxSource(size_t len)
        : Text(len) {
}
PtxSource::PtxSource(char *text)
        : Text(text, false) {
}
CudaSource::CudaSource(size_t len)
        : Text(len) {
}
CudaSource::CudaSource(char *text)
        : Text(text, false) {
}
CudaSource::CudaSource(size_t len, char *text, bool isCopy)
   :Text(len, text, isCopy){

}
Log::Log(size_t len)
        : Text(len) {
}
Log::Log(char *text)
        : Text(text, false) {
}
uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

std::string tmpFileName(uint64_t time, const std::string& suffix){
    std::stringstream timestamp;
    timestamp << "./tmp" << time << suffix;
    return timestamp.str();
}

PtxSource *PtxSource::nvcc(const char *cudaSource, size_t len) {
    CudaSource cSource(len,(char*)cudaSource,false);

    uint64_t time = timeSinceEpochMillisec();
    std::string ptxPath = tmpFileName(time, ".ptx");
    std::string cudaPath = tmpFileName(time, ".cu");
    // we are going to fork exec nvcc
    int pid;
    cSource.writeTmp(cudaPath);
    if ((pid = fork()) == 0) {
        const char *path = "/usr/local/cuda-12.2/bin/nvcc";
        const char *argv[]{"nvcc", "-ptx", cudaPath.c_str(), "-o", ptxPath.c_str(), nullptr};
       // std::cerr << "child about to exec nvcc" << std::endl;
       // std::cerr << "path " << path<< " " << argv[1]<< " " << argv[2]<< " " << argv[3]<< " " << argv[4]<< std::endl;
        int stat = execvp(path, (char *const *) argv);
        std::cerr << " nvcc stat = "<<stat << " errno="<< errno<< " '"<< std::strerror(errno)<< "'"<<std::endl;
        std::exit(errno);
    } else if (pid < 0) {
        // fork failed.
        std::cerr << "fork of nvcc failed" << std::endl;
        std::exit(1);
    } else {
        int status;
       // std::cerr << "parent waiting for child nvcc exec" << std::endl;
        pid_t result = wait(&status);
        //std::cerr << "child finished should be safe to read "<< ptxPath << std::endl;
        PtxSource *ptx= new PtxSource();
        ptx->readTmp(ptxPath);
        return ptx;
    }
    std::cerr << "we should never get here !";
    exit(1);
    return nullptr;
}

/*
//http://mercury.pr.erau.edu/~siewerts/extra/code/digital-media/CUDA/cuda_work/samples/0_Simple/matrixMulDrv/matrixMulDrv.cpp
 */
CudaBackend::CudaBuffer::CudaBuffer(Backend *backend, Arg_s *arg, BufferState_s *bufferState)
        : Buffer(backend, arg,bufferState), devicePtr() {
    /*
     *   (void *) arg->value.buffer.memorySegment,
     *   (size_t) arg->value.buffer.sizeInBytes);
     */
  //  std::cout << "cuMemAlloc()" << std::endl;
    CUresult status = cuMemAlloc(&devicePtr, (size_t) arg->value.buffer.sizeInBytes);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuMemFree() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
  //  std::cout << "devptr " << std::hex<<  (long)devicePtr <<std::dec <<std::endl;
  bufferState->vendorPtr= static_cast<void *>(this);
}

CudaBackend::CudaBuffer::~CudaBuffer() {

 //   std::cout << "cuMemFree()"
  //          << "devptr " << std::hex<<  (long)devicePtr <<std::dec
   //         << std::endl;
    CUresult  status = cuMemFree(devicePtr);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuMemFree() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    bufferState->vendorPtr= nullptr;
}

void CudaBackend::CudaBuffer::copyToDevice() {
    auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
 //   std::cout << "copyToDevice() 0x"   << std::hex<<arg->value.buffer.sizeInBytes<<std::dec << " "<< arg->value.buffer.sizeInBytes << " "
 //             << "devptr " << std::hex<<  (long)devicePtr <<std::dec
 //             << std::endl;
    char *ptr = (char*)arg->value.buffer.memorySegment;

    CUresult status = cuMemcpyHtoDAsync(devicePtr, arg->value.buffer.memorySegment, arg->value.buffer.sizeInBytes,cudaBackend->cudaQueue.cudaStream);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuMemcpyHtoDAsync() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    status = static_cast<CUresult >(cudaStreamSynchronize(cudaBackend->cudaQueue.cudaStream));
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
}

void CudaBackend::CudaBuffer::copyFromDevice() {
    auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
  //  auto cudaKernel = dynamic_cast<CudaKernel*>(kernel);
 //   std::cout << "copyFromDevice() 0x" << std::hex<<arg->value.buffer.sizeInBytes<<std::dec << " "<< arg->value.buffer.sizeInBytes << " "
 //             << "devptr " << std::hex<<  (long)devicePtr <<std::dec
  //            << std::endl;
    char *ptr = (char*)arg->value.buffer.memorySegment;

    CUresult status =cuMemcpyDtoHAsync(arg->value.buffer.memorySegment, devicePtr, arg->value.buffer.sizeInBytes,cudaBackend->cudaQueue.cudaStream);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    cudaError_t t1 = cudaStreamSynchronize(cudaBackend->cudaQueue.cudaStream);
    if (static_cast<cudaError_t>(CUDA_SUCCESS) != t1) {
        std::cerr << "CUDA error = " << t1
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(t1))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }

}

CudaBackend::CudaModule::CudaKernel::CudaKernel(Backend::CompilationUnit *program,char * name, CUfunction function)
        : Backend::CompilationUnit::Kernel(program, name), function(function) {
}

CudaBackend::CudaModule::CudaKernel::~CudaKernel() = default;

long CudaBackend::CudaModule::CudaKernel::ndrange(void *argArray) {
  //  std::cout << "ndrange(" << range << ") " << name << std::endl;
    auto cudaBackend = dynamic_cast<CudaBackend*>(compilationUnit->backend);

    ArgSled argSled(static_cast<ArgArray_s *>(argArray));
 //   Schema::dumpSled(std::cout, argArray);
    void *argslist[argSled.argc()];
    NDRange *ndrange = nullptr;
#ifdef VERBOSE
    std::cerr << "there are " << argSled.argc() << "args " << std::endl;
#endif
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
                if (arg->idx == 0){
                    ndrange = static_cast<NDRange *>(arg->value.buffer.memorySegment);
                }
                auto cudaBuffer = new CudaBackend::CudaBuffer(cudaBackend, arg, BufferState_s::of(arg));
                cudaBuffer->copyToDevice();
                argslist[arg->idx] = static_cast<void *>(&cudaBuffer->devicePtr);
                break;
            }
            case 'I':
            case 'F':
            case 'J':
            case 'D':
            case 'C':
            case 'S': {
                argslist[arg->idx] = static_cast<void *>(&arg->value);
                break;
            }
            default: {
                std::cerr << " unhandled variant " << (char) arg->variant << std::endl;
                break;
            }
        }
    }

    int range = ndrange->maxX;
    int rangediv1024 = range / 1024;
    int rangemod1024 = range % 1024;
    if (rangemod1024 > 0) {
        rangediv1024++;
    }
   // std::cout << "Running the kernel..." << std::endl;
  //  std::cout << "   Requested range   = " << range << std::endl;
  //  std::cout << "   Range mod 1024    = " << rangemod1024 << std::endl;
   // std::cout << "   Actual range 1024 = " << (rangediv1024 * 1024) << std::endl;
    auto status= static_cast<CUresult>(cudaStreamSynchronize(cudaBackend->cudaQueue.cudaStream));
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }

    status= cuLaunchKernel(function,
                                   rangediv1024, 1, 1,
                                   1024, 1, 1,
                                   0, cudaBackend->cudaQueue.cudaStream,
                    argslist, 0);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuLaunchKernel() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    status= static_cast<CUresult>(cudaStreamSynchronize(cudaBackend->cudaQueue.cudaStream));
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }

    //std::cout << "Kernel complete..."<<cudaGetErrorString(t)<<std::endl;

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            static_cast<CudaBuffer *>(BufferState_s::of(arg)->vendorPtr)->copyFromDevice();

        }
    }
    status=   static_cast<CUresult>(cudaStreamSynchronize(cudaBackend->cudaQueue.cudaStream));
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            delete static_cast<CudaBuffer *>(BufferState_s::of(arg)->vendorPtr);
            BufferState_s::of(arg)->vendorPtr = nullptr;

        }
    }
   // cudaStreamDestroy(cudaStream);
    return (long) 0;
}


CudaBackend::CudaModule::CudaModule(Backend *backend, char *src, char  *log,  bool ok, CUmodule module)
        : Backend::CompilationUnit(backend, src, log, ok), cudaSource(src), ptxSource(),log(log), module(module) {
}

CudaBackend::CudaModule::~CudaModule() = default;

long CudaBackend::CudaModule::getKernel(int nameLen, char *name) {
    CUfunction function;
    CUresult status= cuModuleGetFunction(&function, module, name);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuModuleGetFunction() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    long kernelHandle =  reinterpret_cast<long>(new CudaKernel(this, name, function));
    return kernelHandle;
}

bool CudaBackend::CudaModule::programOK() {
    return true;
}

CudaBackend::CudaBackend(int mode)
        : Backend(mode), cudaConfig(mode), cudaQueue(this), device(),context()  {
  //  std::cout << "CudaBackend constructor " << ((cudaConfig == nullptr) ? "cudaConfig== null" : "got cudaConfig")
    //          << std::endl;
    int deviceCount = 0;
    CUresult err = cuInit(0);
    if (err == CUDA_SUCCESS) {
        cuDeviceGetCount(&deviceCount);
        std::cout << "CudaBackend device count" << std::endl;
        cuDeviceGet(&device, 0);
        std::cout << "CudaBackend device ok" << std::endl;
        cuCtxCreate(&context, 0, device);
        std::cout << "CudaBackend context created ok" << std::endl;
    } else {
        std::cout << "CudaBackend failed, we seem to have the runtime library but no device, no context, nada "
                  << std::endl;
        exit(1);
    }
}

//CudaBackend::CudaBackend() : CudaBackend(nullptr, 0, nullptr) {
//
//}

CudaBackend::~CudaBackend() {
    std::cout << "freeing context" << std::endl;
    CUresult status = cuCtxDestroy(context);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuCtxDestroy(() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
}

void CudaBackend::info() {
    char name[100];
    cuDeviceGetName(name, sizeof(name), device);
    std::cout << "> Using device 0: " << name << std::endl;

    // get compute capabilities and the devicename
    int major = 0, minor = 0;
    cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device);
    cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device);
    std::cout << "> GPU Device has major=" << major << " minor=" << minor << " compute capability" << std::endl;

    int warpSize;
    cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device);
    std::cout << "> GPU Device has warpSize " << warpSize << std::endl;

    int threadsPerBlock;
    cuDeviceGetAttribute(&threadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device);
    std::cout << "> GPU Device has threadsPerBlock " << threadsPerBlock << std::endl;

    int cores;
    cuDeviceGetAttribute(&cores, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device);
    std::cout << "> GPU Cores " << cores << std::endl;

    size_t totalGlobalMem;
    cuDeviceTotalMem(&totalGlobalMem, device);
    std::cout << "  Total amount of global memory:   " << (unsigned long long) totalGlobalMem << std::endl;
    std::cout << "  64-bit Memory Address:           " <<
              ((totalGlobalMem > (unsigned long long) 4 * 1024 * 1024 * 1024L) ? "YES" : "NO") << std::endl;

}

long CudaBackend::compile(int len, char *source) {
    PtxSource *ptx = PtxSource::nvcc(source, len);
    CUmodule module;
    std::cout << "inside compileProgram" << std::endl;
    std::cout << "cuda " << source << std::endl;
    if (ptx->text != nullptr) {
        std::cout << "ptx " << ptx->text << std::endl;

        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 2;
        auto jitOptions = new CUjit_option[jitNumOptions];
        void **jitOptVals = new void *[jitNumOptions];

        // set up size of compilation log buffer
        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;
        int jitLogBufferSize = 8192;
        jitOptVals[0] = (void *) (size_t) jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;
        int status = cuModuleLoadDataEx(&module, ptx->text, jitNumOptions, jitOptions, (void **) jitOptVals);

        printf("> PTX JIT log:\n%s\n", jitLogBuffer);
        return reinterpret_cast<long>(new CudaModule(this,  ptx->text,jitLogBuffer,true, module));

        //delete ptx;
    } else {
        std::cout << "no ptx content!" << std::endl;
        exit(1);
    }
}
extern "C" long getCudaBackend(int mode) {
    long backendHandle= reinterpret_cast<long>(new CudaBackend(mode));
    std::cout << "getBackend() -> backendHandle=" << std::hex << backendHandle << std::dec << std::endl;
    return backendHandle;
}

CudaBackend::CudaQueue::CudaQueue(Backend *backend)
        : Backend::Queue(backend){
        cudaStreamCreate(&cudaStream);
    }



void CudaBackend::CudaQueue::showEvents(int width) {

}
void CudaBackend::CudaQueue::wait(){
    if (eventc > 0){
      //  cl_int status = clWaitForEvents(eventc, events);
      //  if (status != CL_SUCCESS) {
          //  std::cerr << "failed clWaitForEvents" << CudaBackend::errorMsg(status) << std::endl;
           // exit(1);
       // }
    }
}
void clCallback(void *){
    std::cerr<<"start of compute"<<std::endl;
}

void CudaBackend::CudaQueue::marker(int bits){
   // cl_int status = clEnqueueMarkerWithWaitList(
          //  command_queue,
           // this->eventc, this->eventListPtr(),this->nextEventPtr()
   // );
   // if (status != CL_SUCCESS){
     //   std::cerr << "failed to clEnqueueMarkerWithWaitList "<<errorMsg(status)<< std::endl;
     //   std::exit(1);
  //  }
   // inc(bits);
}
void CudaBackend::CudaQueue::marker(int bits, const char* arg){
   // cl_int status = clEnqueueMarkerWithWaitList(
          //  command_queue,
          //  this->eventc, this->eventListPtr(),this->nextEventPtr()
  //  );
   // if (status != CL_SUCCESS){
     //   std::cerr << "failed to clEnqueueMarkerWithWaitList "<<errorMsg(status)<< std::endl;
      //  std::exit(1);
   // }
   // inc(bits, arg);
}
void CudaBackend::CudaQueue::marker(int bits, int arg){
    //cl_int status = clEnqueueMarkerWithWaitList(
          //  command_queue,
        //    this->eventc, this->eventListPtr(),this->nextEventPtr()
  //  );
   // if (status != CL_SUCCESS){
    //    std::cerr << "failed to clEnqueueMarkerWithWaitList "<<errorMsg(status)<< std::endl;
    //    std::exit(1);
  //  }
 //   inc(bits, arg);
}

void CudaBackend::CudaQueue::computeStart(){
    wait(); // should be no-op
    release(); // also ;
    marker(StartComputeBits);
}



void CudaBackend::CudaQueue::computeEnd(){
    marker(EndComputeBits);
}

void CudaBackend::CudaQueue::inc(int bits){
    if (eventc+1 >= eventMax){
        std::cerr << "CudaBackend::CudaQueue event list overflowed!!" << std::endl;
    }else{
        eventInfoBits[eventc]=bits;
    }
    eventc++;
}
void CudaBackend::CudaQueue::inc(int bits, const char *arg){
    if (eventc+1 >= eventMax){
        std::cerr << "CudaBackend::CudaQueue event list overflowed!!" << std::endl;
    }else{
        eventInfoBits[eventc]=bits|HasConstCharPtrArgBits;
        eventInfoConstCharPtrArgs[eventc]=arg;
    }
    eventc++;
}
void CudaBackend::CudaQueue::inc(int bits, int arg){
    if (eventc+1 >= eventMax){
        std::cerr << "CudaBackend::CudaQueue event list overflowed!!" << std::endl;
    }else{
        eventInfoBits[eventc]=bits|arg|hasIntArgBits;
    }
    eventc++;
}

void CudaBackend::CudaQueue::markAsEndComputeAndInc(){
    inc(EndComputeBits);
}
void CudaBackend::CudaQueue::markAsStartComputeAndInc(){
    inc(StartComputeBits);
}
void CudaBackend::CudaQueue::markAsNDRangeAndInc(){
    inc(NDRangeBits);
}
void CudaBackend::CudaQueue::markAsCopyToDeviceAndInc(int argn){
    inc(CopyToDeviceBits, argn);
}
void CudaBackend::CudaQueue::markAsCopyFromDeviceAndInc(int argn){
    inc(CopyFromDeviceBits, argn);
}
void CudaBackend::CudaQueue::markAsEnterKernelDispatchAndInc(){
    inc(EnterKernelDispatchBits);
}
void CudaBackend::CudaQueue::markAsLeaveKernelDispatchAndInc(){
    inc(LeaveKernelDispatchBits);
}

void CudaBackend::CudaQueue::release(){
   // cl_int status = CL_SUCCESS;
  //  for (int i = 0; i < eventc; i++) {
   //     status = clReleaseEvent(events[i]);
    //    if (status != CL_SUCCESS) {
      //      std::cerr << CudaBackend::errorMsg(status) << std::endl;
      //      exit(1);
   //     }
   // }//
 //   eventc = 0;
}

CudaBackend::CudaQueue::~CudaQueue(){
   // clReleaseCommandQueue(command_queue);
   // delete []events;

}

CudaBackend::CudaConfig::CudaConfig(int mode)
   : Backend::Config(mode){

}
void CudaBackend::computeEnd(){

}
void CudaBackend::computeStart(){

}
bool CudaBackend::getBufferFromDeviceIfDirty(void *memorySegment, long size){
return true;
}