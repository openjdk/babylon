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
#include <cuda_runtime_api.h>
#include "ptx_backend.h"

Ptx::Ptx(size_t len)
        : len(len), text(len > 0 ? new char[len] : nullptr) {
    std::cout << "in Ptx with buffer allocated "<<len << std::endl;
}

Ptx::~Ptx() {
    if (len > 0 && text != nullptr) {
        std::cout << "in ~Ptx with deleting allocated "<<len << std::endl;
        delete[] text;
    }
}

uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

Ptx *Ptx::nvcc(const char *ptxSource, size_t len) {
    Ptx *ptx = nullptr;
    uint64_t time = timeSinceEpochMillisec();
    std::stringstream timestampPtx;
    timestampPtx << "./tmp" << time << ".ptx";
    const char *ptxPath = strdup(timestampPtx.str().c_str());
   // std::cout << "ptx " << ptxPath << std::endl;
    // we are going to fork exec nvcc
    int pid;
    if ((pid = fork()) == 0) {
        std::ofstream ptx;
        std::stringstream timestampPtx;
        timestampPtx << "./tmp" << time << ".cu";
        const char *ptxPath = strdup(timestampPtx.str().c_str());
        std::cout << "ptx " << ptxPath << std::endl;
        ptx.open(ptxPath, std::ofstream::trunc);
        ptx.write(ptxSource, len);
        ptx.close();
        const char *path = "/usr/bin/nvcc";
        //const char *path = "/usr/local/cuda-12.2/bin/nvcc";
        const char *argv[]{"nvcc", "-ptx", ptxPath, "-o", ptxPath, nullptr};
        // we can't free ptxPath or ptxpath in child because we need them in exec, no prob through
        // because we get a new proc so they are released to os
        execvp(path, (char *const *) argv);

    } else if (pid < 0) {
        // fork failed.
        std::cerr << "fork of nvcc failed" << std::endl;
        std::exit(1);
    } else {
        int status;
     //   std::cerr << "fork suceeded waiting for child" << std::endl;
        pid_t result = wait(&status);
        std::cerr << "child finished" << std::endl;
        std::ifstream ptxStream;
        ptxStream.open(ptxPath);
      //  if (ptxStream.is_open()) {
            ptxStream.seekg(0, std::ios::end);
            size_t ptxLen = ptxStream.tellg();
            ptxStream.close();
            ptxStream.open(ptxPath);
            free((void *) ptxPath);
            ptxPath = nullptr;
            if (ptxLen > 0) {
                std::cerr << "ptx len "<< ptxLen << std::endl;
                ptx = new Ptx(ptxLen + 1);
                std::cerr << "about to read  "<< ptx->len << std::endl;
                ptxStream.read(ptx->text, ptx->len);
                ptxStream.close();
                std::cerr << "about to read  "<< ptx->len << std::endl;
                ptx->text[ptx->len - 1] = '\0';
                std::cerr << "read text "<< ptx->text << std::endl;

            } else {
                std::cerr << "no ptx! ptxLen == 0?";
                exit(1);
            }
      //  }else{
        //    std::cerr << "no ptx!";
       //     exit(1);
      //  }
    }
    std::cout << "returning PTX" << std::endl;
    return ptx;
}

/*
//http://mercury.pr.erau.edu/~siewerts/extra/code/digital-media/CUDA/cuda_work/samples/0_Simple/matrixMulDrv/matrixMulDrv.cpp
 */
PtxBackend::PtxProgram::PtxKernel::PtxBuffer::PtxBuffer(Backend::Program::Kernel *kernel, Arg_s *arg)
        : Buffer(kernel, arg), devicePtr() {
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
    arg->value.buffer.vendorPtr = static_cast<void *>(this);
}

PtxBackend::PtxProgram::PtxKernel::PtxBuffer::~PtxBuffer() {

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
    arg->value.buffer.vendorPtr = nullptr;
}

void PtxBackend::PtxProgram::PtxKernel::PtxBuffer::copyToDevice() {
    auto ptxKernel = dynamic_cast<PtxKernel*>(kernel);
 //   std::cout << "copyToDevice() 0x"   << std::hex<<arg->value.buffer.sizeInBytes<<std::dec << " "<< arg->value.buffer.sizeInBytes << " "
 //             << "devptr " << std::hex<<  (long)devicePtr <<std::dec
 //             << std::endl;
    char *ptr = (char*)arg->value.buffer.memorySegment;

    unsigned long ifacefacade1 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-16);
    unsigned long ifacefacade2 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-8);

    if (ifacefacade1 != 0x1face00000facadeL && ifacefacade1 != ifacefacade2) {
        std::cerr<<"End of buf marker before HtoD"<< std::hex << ifacefacade1 << ifacefacade2<< " buffer corrupt !" <<std::endl
                <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }


    CUresult status = cuMemcpyHtoDAsync(devicePtr, arg->value.buffer.memorySegment, arg->value.buffer.sizeInBytes,ptxKernel->cudaStream);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuMemcpyHtoDAsync() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    status = static_cast<CUresult >(cudaStreamSynchronize(ptxKernel->cudaStream));
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
}

void PtxBackend::PtxProgram::PtxKernel::PtxBuffer::copyFromDevice() {
    auto ptxKernel = dynamic_cast<PtxKernel*>(kernel);
 //   std::cout << "copyFromDevice() 0x" << std::hex<<arg->value.buffer.sizeInBytes<<std::dec << " "<< arg->value.buffer.sizeInBytes << " "
 //             << "devptr " << std::hex<<  (long)devicePtr <<std::dec
  //            << std::endl;
    char *ptr = (char*)arg->value.buffer.memorySegment;

    unsigned long ifacefacade1 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-16);
    unsigned long ifacefacade2 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-8);

    if (ifacefacade1 != 0x1face00000facadeL || ifacefacade1 != ifacefacade2) {
        std::cerr<<"end of buf marker before  DtoH"<< std::hex << ifacefacade1 << ifacefacade2<< std::dec<< " buffer corrupt !"<<std::endl
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    CUresult status =cuMemcpyDtoHAsync(arg->value.buffer.memorySegment, devicePtr, arg->value.buffer.sizeInBytes,ptxKernel->cudaStream);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    cudaError_t t1 = cudaStreamSynchronize(ptxKernel->cudaStream);
    if (static_cast<cudaError_t>(CUDA_SUCCESS) != t1) {
        std::cerr << "CUDA error = " << t1
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(t1))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    ifacefacade1 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-16);
    ifacefacade2 = *reinterpret_cast<unsigned long*>(ptr+arg->value.buffer.sizeInBytes-8);

    if (ifacefacade1 != 0x1face00000facadeL || ifacefacade1 != ifacefacade2) {
        std::cerr<<"end of buf marker after  DtoH"<< std::hex << ifacefacade1 << ifacefacade2<< std::dec<< " buffer corrupt !"<<std::endl
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
}

PtxBackend::PtxProgram::PtxKernel::PtxKernel(Backend::Program *program,char * name, CUfunction function)
        : Backend::Program::Kernel(program, name), function(function),cudaStream() {
}

PtxBackend::PtxProgram::PtxKernel::~PtxKernel() = default;

long PtxBackend::PtxProgram::PtxKernel::ndrange(void *argArray) {
  //  std::cout << "ndrange(" << range << ") " << name << std::endl;

    cudaStreamCreate(&cudaStream);
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
                auto ptxBuffer = new PtxBuffer(this, arg);
                ptxBuffer->copyToDevice();
                argslist[arg->idx] = static_cast<void *>(&ptxBuffer->devicePtr);
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
    auto status= static_cast<CUresult>(cudaStreamSynchronize(cudaStream));
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }

    status= cuLaunchKernel(function,
                                   rangediv1024, 1, 1,
                                   1024, 1, 1,
                                   0, cudaStream,
                    argslist, 0);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuLaunchKernel() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    status= static_cast<CUresult>(cudaStreamSynchronize(cudaStream));
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
            static_cast<PtxBuffer *>(arg->value.buffer.vendorPtr)->copyFromDevice();

        }
    }
    status=   static_cast<CUresult>(cudaStreamSynchronize(cudaStream));
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            delete static_cast<PtxBuffer *>(arg->value.buffer.vendorPtr);
            arg->value.buffer.vendorPtr = nullptr;
        }
    }
    cudaStreamDestroy(cudaStream);
    return (long) 0;
}


PtxBackend::PtxProgram::PtxProgram(Backend *backend, BuildInfo *buildInfo, Ptx *ptx, CUmodule module)
        : Backend::Program(backend, buildInfo), ptx(ptx), module(module) {
}

PtxBackend::PtxProgram::~PtxProgram() = default;

long PtxBackend::PtxProgram::getKernel(int nameLen, char *name) {
    CUfunction function;
    CUresult status= cuModuleGetFunction(&function, module, name);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuModuleGetFunction() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    long kernelHandle =  reinterpret_cast<long>(new PtxKernel(this, name, function));
    return kernelHandle;
}

bool PtxBackend::PtxProgram::programOK() {
    return true;
}

PtxBackend::PtxBackend(PtxBackend::PtxConfig *ptxConfig, int
configSchemaLen, char *configSchema)
        : Backend((Backend::Config*) ptxConfig, configSchemaLen, configSchema), device(),context()  {
  //  std::cout << "PtxBackend constructor " << ((ptxConfig == nullptr) ? "ptxConfig== null" : "got ptxConfig")
    //          << std::endl;
    int deviceCount = 0;
    CUresult err = cuInit(0);
    if (err == CUDA_SUCCESS) {
        cuDeviceGetCount(&deviceCount);
        std::cout << "PtxBackend device count" << std::endl;
        cuDeviceGet(&device, 0);
        std::cout << "PtxBackend device ok" << std::endl;
        cuCtxCreate(&context, 0, device);
        std::cout << "PtxBackend context created ok" << std::endl;
    } else {
        std::cout << "PtxBackend failed, we seem to have the runtime library but no device, no context, nada "
                  << std::endl;
        exit(1);
    }
}

PtxBackend::PtxBackend() : PtxBackend(nullptr, 0, nullptr) {

}

PtxBackend::~PtxBackend() {
    std::cout << "freeing context" << std::endl;
    CUresult status = cuCtxDestroy(context);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuCtxDestroy(() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
}

int PtxBackend::getMaxComputeUnits() {
    std::cout << "getMaxComputeUnits()" << std::endl;
    int value = 1;
    return value;
}

void PtxBackend::info() {
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

long PtxBackend::compileProgram(int len, char *source) {
    Ptx *ptx = Ptx::nvcc(source, len);
    CUmodule module;
    std::cout << "inside compileProgram" << std::endl;
    std::cout << "ptx " << source << std::endl;
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
        return reinterpret_cast<long>(new PtxProgram(this, nullptr, ptx, module));

        //delete ptx;
    } else {
        std::cout << "no ptx content!" << std::endl;
        exit(1);
    }
}

long getBackend(void *config, int configSchemaLen, char *configSchema) {
    long backendHandle= reinterpret_cast<long>(
            new PtxBackend(static_cast<PtxBackend::PtxConfig *>(config), configSchemaLen,
                            configSchema));
    std::cout << "getBackend() -> backendHandle=" << std::hex << backendHandle << std::dec << std::endl;
    return backendHandle;
}



