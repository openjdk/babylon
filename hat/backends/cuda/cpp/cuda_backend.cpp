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

Ptx::Ptx(size_t len)
        : len(len), text(len > 0 ? new char[len] : nullptr) {}

Ptx::~Ptx() {
    if (len > 0 && text != nullptr) {
        delete[] text;
    }
}

uint64_t timeSinceEpochMillisec() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

Ptx *Ptx::nvcc(const char *cudaSource, size_t len) {

    uint64_t time = timeSinceEpochMillisec();
    std::stringstream timestampCuda;
    timestampCuda << "./tmp" << time << ".cu";
    std::stringstream timestampPtx;
    timestampPtx << "./tmp" << time << ".ptx";
    Ptx *ptx = nullptr;
    const char *cudaPath = strdup(timestampCuda.str().c_str());
    const char *ptxPath = strdup(timestampPtx.str().c_str());
    std::cout << "cuda " << cudaPath << std::endl;
    std::cout << "ptx " << ptxPath << std::endl;
    // we are going to fork exec nvcc
    int pid;
    if ((pid = fork()) == 0) {
        std::ofstream cuda;
        cuda.open(cudaPath, std::ofstream::trunc);
        cuda.write(cudaSource, len);
        cuda.close();

        const char *path = "/usr/local/cuda-12.2/bin/nvcc";
        // const char *argv[]{"nvcc", "-v", "-ptx", cudaPath, "-o", ptxPath, nullptr};
        const char *argv[]{"nvcc", "-ptx", cudaPath, "-o", ptxPath, nullptr};
        // We are the child so exec nvcc.
        //close(1); // stdout
        //close(2); // stderr
        //open(stderrPath, O_RDWR); // stdout
        //open(stdoutPath, O_RDWR); // stderr
        execvp(path, (char *const *) argv);
    } else if (pid < 0) {
        // fork failed.
        std::cerr << "fork of nvcc failed" << std::endl;
        std::exit(1);
    } else {
        int status;
        std::cerr << "fork suceeded waitikbng for child" << std::endl;
        pid_t result = wait(&status);
        std::cerr << "child finished" << std::endl;
        std::ifstream ptxStream(ptxPath);
        ptxStream.seekg(0, ptxStream.end);
        size_t ptxLen = ptxStream.tellg();
        if (ptxLen > 0) {
            ptx = new Ptx(ptxLen + 1);
            ptxStream.seekg(0, ptxStream.beg);
            ptxStream.read(ptx->text, ptx->len);
            ptx->text[ptx->len] = '\0';
            ptx->text[ptx->len - 1] = '\0';
        }
        ptxStream.close();
    }
    std::cout << "returning PTX" << std::endl;
    return ptx;
}

/*
//http://mercury.pr.erau.edu/~siewerts/extra/code/digital-media/CUDA/cuda_work/samples/0_Simple/matrixMulDrv/matrixMulDrv.cpp
 */
CudaBackend::CudaProgram::CudaKernel::CudaBuffer::CudaBuffer(Backend::Program::Kernel *kernel, Arg_t *arg)
        : Buffer(kernel, arg) {
    /*
     *   (void *) arg->value.buffer.memorySegment,
     *   (size_t) arg->value.buffer.sizeInBytes);
     */
    std::cout << "cuMemAlloc()" << std::endl;
    checkCudaErrors(cuMemAlloc(&devicePtr, (size_t) arg->value.buffer.sizeInBytes));
    arg->value.buffer.vendorPtr = static_cast<void*>(this);
}

CudaBackend::CudaProgram::CudaKernel::CudaBuffer::~CudaBuffer() {
    std::cout << "cuMemFree()" << std::endl;
    checkCudaErrors(cuMemFree(devicePtr));
}

void CudaBackend::CudaProgram::CudaKernel::CudaBuffer::copyToDevice() {
    std::cout << "copyToDevice()" << std::endl;
    checkCudaErrors(cuMemcpyHtoD(devicePtr, arg->value.buffer.memorySegment, arg->value.buffer.sizeInBytes));
}

void CudaBackend::CudaProgram::CudaKernel::CudaBuffer::copyFromDevice() {
    std::cout << "copyFromDevice()" << std::endl;
    checkCudaErrors(cuMemcpyDtoH(arg->value.buffer.memorySegment, devicePtr, arg->value.buffer.sizeInBytes));

}

CudaBackend::CudaProgram::CudaKernel::CudaKernel(Backend::Program *program, CUfunction function)
        : Backend::Program::Kernel(program), function(function) {
}

CudaBackend::CudaProgram::CudaKernel::~CudaKernel() {

    // releaseCUfunction function
}

long CudaBackend::CudaProgram::CudaKernel::ndrange(int range, void *argArray) {
    //std::cout<<"ndrange("<<range<<") "<< std::endl;
    ArgSled argSled(static_cast<ArgArray_t *>(argArray));
    void *argslist[argSled.argc()];
#ifdef VERBOSE
    std::cerr << "there are " << argSled.argc() << "args " << std::endl;
#endif
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_t *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
                auto cudaBuffer = new CudaBuffer(this, arg);
                cudaBuffer->copyToDevice();
                argslist[arg->idx] = static_cast<void*>(&cudaBuffer->devicePtr);
                break;
            }
            case 'I':
            case 'F':
            case 'J':
            case 'D':
            case 'C':
            case 'S':
            {
                argslist[arg->idx] = static_cast<void*>(&arg->value);
                break;
            }
            default: {
                std::cerr << " unhandled variant " << (char) arg->variant << std::endl;
                break;
            }
        }
    }


#ifdef VERBOSE
    std::cout << "Running the kernel... range = "<< range << "range mod 512 " << (range%512)<< std::endl;
#endif

    checkCudaErrors(cuLaunchKernel(function,
                                   range / 1024, 1, 1,
                                   1024, 1, 1,
                                   0, 0,
                                   argslist, 0));

    //cudaError_t t = cudaDeviceSynchronize();


    //std::cout << "Kernel complete..."<<cudaGetErrorString(t)<<std::endl;

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_t *arg = argSled.arg(i);
        if (arg->variant == '&') {
            static_cast<CudaBuffer *>(arg->value.buffer.vendorPtr)->copyFromDevice();
        }
    }

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_t *arg = argSled.arg(i);
        if (arg->variant == '&') {
            delete static_cast<CudaBuffer *>(arg->value.buffer.vendorPtr);
            arg->value.buffer.vendorPtr= nullptr;
        }
    }
    return (long) 0;
}


CudaBackend::CudaProgram::CudaProgram(Backend *backend, BuildInfo *buildInfo, Ptx *ptx, CUmodule module)
        : Backend::Program(backend, buildInfo), ptx(ptx), module(module) {
}

CudaBackend::CudaProgram::~CudaProgram() {
}

long CudaBackend::CudaProgram::getKernel(int nameLen, char *name) {
    CUfunction function;
    std::cout << "trying to get kernelFunction " << name << std::endl;
    checkCudaErrors(
            cuModuleGetFunction(&function, module, name)
    );
    return reinterpret_cast<long>(new CudaKernel(this, function));
}

bool CudaBackend::CudaProgram::programOK() {
    return true;
}

CudaBackend::CudaBackend(CudaBackend::CudaConfig *cudaConfig, int
configSchemaLen, char *configSchema)
        : Backend((Backend::Config
*) cudaConfig, configSchemaLen, configSchema) {
    std::cout << "CudaBackend constructor " << ((cudaConfig == nullptr) ? "cudaConfig== null" : "got cudaConfig")
              << std::endl;
    int deviceCount = 0;
    CUresult err = cuInit(0);
    if (err == CUDA_SUCCESS) {
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
        std::cout << "CudaBackend device count" << std::endl;
        checkCudaErrors(cuDeviceGet(&device, 0));
        std::cout << "CudaBackend device ok" << std::endl;
        checkCudaErrors(cuCtxCreate(&context, 0, device));
        std::cout << "CudaBackend context created ok" << std::endl;
    } else {
        std::cout << "CudaBackend failed, we seem to have the runtime library but no device, no context, nada "
                  << std::endl;
        exit(1);
    }
}

CudaBackend::CudaBackend() : CudaBackend(nullptr, 0, nullptr) {

}

CudaBackend::~CudaBackend() {
    std::cout << "freeing context" << std::endl;
    checkCudaErrors(cuCtxDestroy(context));
}

int CudaBackend::getMaxComputeUnits() {
    std::cout << "getMaxComputeUnits()" << std::endl;
    int value = 1;
    return value;
}

void CudaBackend::info() {
    char name[100];
    cuDeviceGetName(name, sizeof(name), device);
    std::cout << "> Using device 0: " << name << std::endl;

    // get compute capabilities and the devicename
    int major = 0, minor = 0;
    checkCudaErrors(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
    checkCudaErrors(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
    std::cout << "> GPU Device has major=" << major << " minor=" << minor << " compute capability" << std::endl;

    int warpSize;
    checkCudaErrors(cuDeviceGetAttribute(&warpSize, CU_DEVICE_ATTRIBUTE_WARP_SIZE, device));
    std::cout << "> GPU Device has warpSize " << warpSize << std::endl;

    int threadsPerBlock;
    checkCudaErrors(cuDeviceGetAttribute(&threadsPerBlock, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, device));
    std::cout << "> GPU Device has threadsPerBlock " << threadsPerBlock << std::endl;

    int cores;
    checkCudaErrors(cuDeviceGetAttribute(&cores, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
    std::cout << "> GPU Cores " << cores << std::endl;

    size_t totalGlobalMem;
    checkCudaErrors(cuDeviceTotalMem(&totalGlobalMem, device));
    std::cout << "  Total amount of global memory:   " << (unsigned long long) totalGlobalMem << std::endl;
    std::cout << "  64-bit Memory Address:           " <<
              ((totalGlobalMem > (unsigned long long) 4 * 1024 * 1024 * 1024L) ? "YES" : "NO") << std::endl;

}

long CudaBackend::compileProgram(int len, char *source) {
    Ptx *ptx = Ptx::nvcc(source, len);

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
        int jitLogBufferSize = 1024;
        jitOptVals[0] = (void *) (size_t) jitLogBufferSize;

        // set up pointer to the compilation log buffer
        jitOptions[1] = CU_JIT_INFO_LOG_BUFFER;
        char *jitLogBuffer = new char[jitLogBufferSize];
        jitOptVals[1] = jitLogBuffer;
        int status = cuModuleLoadDataEx(&module, ptx->text, jitNumOptions, jitOptions, (void **) jitOptVals);

        printf("> PTX JIT log:\n%s\n", jitLogBuffer);
        return reinterpret_cast<long>(new CudaProgram(this, nullptr, ptx, module));

        //delete ptx;
    } else {
        std::cout << "no ptx content!/" << std::endl;
        exit(1);
    }
}

long getBackend(void *config, int configSchemaLen, char *configSchema) {
    return reinterpret_cast<long>(new CudaBackend(static_cast<CudaBackend::CudaConfig *>(config), configSchemaLen, configSchema));
}


void __checkCudaErrors(CUresult err, const char *file, const int line) {
    if (CUDA_SUCCESS != err) {
        std::cerr << "CUDA Driver API error = " << err << " from file " << file << " line " << line << std::endl;
        exit(-1);
    }
}