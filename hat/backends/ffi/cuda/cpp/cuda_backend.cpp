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
CudaSource::CudaSource()
        : Text(0) {
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
    cSource.write(cudaPath);
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
        ptx->read(ptxPath);
        return ptx;
    }
    std::cerr << "we should never get here !";
    exit(1);
    return nullptr;
}


CudaBackend::CudaBackend(int mode)
        : Backend(mode), cudaConfig(mode), cudaQueue(this), device(),context()  {
  //  std::cout << "CudaBackend constructor " << ((cudaConfig == nullptr) ? "cudaConfig== null" : "got cudaConfig")
    //          << std::endl;
    int deviceCount = 0;
    CUresult status = cuInit(0);
    if (status == CUDA_SUCCESS) {
        status  = cuDeviceGetCount(&deviceCount);
        if (status != CUDA_SUCCESS) {
            std::cerr
                    << "cuDeviceGetCount() failed we seem to have the runtime library but no device, CUDA error = "
                    << status
                    << " " << cudaGetErrorString(static_cast<cudaError_t>(status))
                    << " " << __FILE__ << " line " << __LINE__ << std::endl;
            exit(-1);
        }
        std::cout << "CudaBackend device count = "<< deviceCount << std::endl;
        status= cuDeviceGet(&device, 0);
        if (status != CUDA_SUCCESS) {
            std::cerr
                    << "cuDeviceGet() failed we seem to have the runtime library but no device, CUDA error = "
                    << status
                    << " " << cudaGetErrorString(static_cast<cudaError_t>(status))
                    << " " << __FILE__ << " line " << __LINE__ << std::endl;
            exit(-1);
        }
        std::cout << "CudaBackend device ok (id = "<<device<<")" << std::endl;

        status = cuCtxCreate(&context, 0, device);
        if (status != CUDA_SUCCESS) {
            std::cerr
                    << "cuCtxCreate() failed we seem to have the runtime library found a device, but cant create context, CUDA error = "
                    << status
                    << " " << cudaGetErrorString(static_cast<cudaError_t>(status))
                    << " " << __FILE__ << " line " << __LINE__ << std::endl;
            exit(-1);
        }
        std::cout << "CudaBackend context created ok (id="<<context<<")" << std::endl;
    } else {
        std::cerr << "cuInit() failed we seem to have the runtime library but no device, no context, nada CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
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

PtxSource *CudaBackend::nvcc(CudaSource *cudaSource){
    uint64_t time = timeSinceEpochMillisec();
    std::string ptxPath = tmpFileName(time, ".ptx");
    std::string cudaPath = tmpFileName(time, ".cu");
    // we are going to fork exec nvcc so we need to write the cuda source to disk
    int pid;
    cudaSource->write(cudaPath);
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
        ptx->read(ptxPath);
        return ptx;
    }
    std::cerr << "we should never get here !";
    exit(1);
    return nullptr;

}
CudaBackend::CudaModule * CudaBackend::compile(CudaSource &cudaSource) {
    return compile(&cudaSource);
}
CudaBackend::CudaModule * CudaBackend::compile(CudaSource *cudaSource) {
    PtxSource *ptx = nvcc(cudaSource);
    CUmodule module;
    std::cout << "inside compile" << std::endl;
    std::cout << "cuda " << cudaSource->text << std::endl;
    if (ptx->text != nullptr) {
        std::cout << "ptx " << ptx->text << std::endl;
        Log *infLog = new Log(8192);
        Log *errLog = new Log(8192);
        const unsigned int optc = 5;
        auto jitOptions = new CUjit_option[optc];
        void **jitOptVals = new void *[optc];


        jitOptions[0] = CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES;jitOptVals[0] = (void *) (size_t) infLog->len;
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER; jitOptVals[1] = infLog->text;
        jitOptions[2] = CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES;jitOptVals[2] = (void *) (size_t) errLog->len;
        jitOptions[3] = CU_JIT_ERROR_LOG_BUFFER; jitOptVals[3] = errLog->text;
        jitOptions[4] = CU_JIT_GENERATE_LINE_INFO;jitOptVals[4] = (void *)1;
        int status = cuModuleLoadDataEx(&module, ptx->text,
                                        optc, jitOptions, (void **) jitOptVals);
        if (CUDA_SUCCESS != status) {
            std::cerr << "cuModuleLoadDataEx() CUDA error = " << status
                      <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                      <<" " << __FILE__ << " line " << __LINE__ << std::endl;
            exit(-1);
        }
        std::cout <<"> PTX JIT inflog:"<<std::endl  << infLog->text << std::endl;
        std::cout <<"> PTX JIT errlog:"<<std::endl  << errLog->text << std::endl;
        return new CudaModule(this,  ptx->text,infLog->text,true, module);

        //delete ptx;
    } else {
        std::cout << "no ptx content!" << std::endl;
        exit(1);
    }
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
        if (CUDA_SUCCESS != status) {
            std::cerr << "cuModuleLoadDataEx() CUDA error = " << status
                      <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                      <<" " << __FILE__ << " line " << __LINE__ << std::endl;
            exit(-1);
        }
        printf("> PTX JIT log:\n%s\n", jitLogBuffer);
        return reinterpret_cast<long>(new CudaModule(this,  ptx->text,jitLogBuffer,true, module));

        //delete ptx;
    } else {
        std::cout << "no ptx content!" << std::endl;
        exit(1);
    }
}
extern "C" long getBackend(int mode) {
    long backendHandle= reinterpret_cast<long>(new CudaBackend(mode));
    std::cout << "getBackend() -> backendHandle=" << std::hex << backendHandle << std::dec << std::endl;
    return backendHandle;
}

void clCallback(void *){
    std::cerr<<"start of compute"<<std::endl;
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

CudaBackend * CudaBackend::of(long backendHandle){
    return reinterpret_cast<CudaBackend *>(backendHandle);
}
CudaBackend * CudaBackend::of(Backend *backend){
    return dynamic_cast<CudaBackend *>(backend);
}