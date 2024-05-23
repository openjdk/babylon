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

#include "cuda_backend.h"

/*
//http://mercury.pr.erau.edu/~siewerts/extra/code/digital-media/CUDA/cuda_work/samples/0_Simple/matrixMulDrv/matrixMulDrv.cpp
 */
CudaBackend::CudaProgram::CudaKernel::CudaBuffer::CudaBuffer(void *ptr, size_t sizeInBytes)
        : ptr(ptr), sizeInBytes(sizeInBytes) {
    cuMemAlloc(&devicePtr, (size_t) sizeInBytes);
}

CudaBackend::CudaProgram::CudaKernel::CudaBuffer::~CudaBuffer() {

}

CudaBackend::CudaProgram::CudaKernel::CudaKernel(Backend::Program *program, CUfunction function)
        : Backend::Program::Kernel(program), function(function) {
}

CudaBackend::CudaProgram::CudaKernel::~CudaKernel() {
    // releaseCUfunction function
}

long CudaBackend::CudaProgram::CudaKernel::ndrange(int range, void *argArray) {
    //std::cout<<"ndrange("<<range<<") "<< std::endl;
    ArgSled argSled((ArgArray_t *) argArray);

    CudaBackend *backend = (CudaBackend *) program->backend;
    bool verbose = false;
    void *argslist[argSled.argc()];
    // #ifdef VERBOSE
    std::cerr << "there are " << argSled.argc() << "args " << std::endl;
    // #endif
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_t *arg = argSled.arg(i);
        if (arg->variant == '&') {

            CudaBuffer *cudaBuffer = new CudaBuffer(
                    (void *) arg->value.buffer.memorySegment,
                    (size_t) arg->value.buffer.sizeInBytes);
            //std::cout << "copying out!"<<std::endl;
            checkCudaErrors(cuMemcpyHtoD(cudaBuffer->devicePtr, cudaBuffer->ptr, cudaBuffer->sizeInBytes));
            argslist[arg->idx] = (void *) &cudaBuffer->devicePtr;
            arg->value.buffer.vendorPtr = (void *) cudaBuffer;
        } else if (arg->variant == 'I') {
            argslist[arg->idx] = &arg->value.s32;
        } else if (arg->variant == 'F') {
            argslist[arg->idx] = &arg->value.f32;
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
#ifdef VERBOSE
        std::cout << "looking at ! "<<arg->argc<<std::endl;
#endif
        if (arg->variant == '&') {
            //   if ((arg->access &ACCESS_WO || arg->access &ACCESS_RW)){
            //#ifdef VERBOSE
            //std::cout << "copying back!"<<std::endl;
            CudaBuffer *cudaBuffer = (CudaBuffer *) arg->value.buffer.vendorPtr;
            checkCudaErrors(cuMemcpyDtoH(cudaBuffer->ptr, cudaBuffer->devicePtr, cudaBuffer->sizeInBytes));
            //#endif
            //arg->value.buffer.vendorPtr
            // checkCudaErrors( cuMemcpyDtoH(arg->_1d.ptr, arg->_1d.buf, arg->sizeInBytes) );
            //hexdump(arg->_1d.ptr, sizeof(int) * arg->_1d.elements);
            //       }else{
            //#ifdef VERBOSE
            //          std::cout << "skipping copying back!"<<std::endl;
            //#endif
            //       }
        }
    }

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_t *arg = argSled.arg(i);
        if (arg->variant == '&') {
#ifdef VERBOSE
            std::cout << "releasing arg "<<arg->argc<< " "<<std::endl;
#endif
            CudaBuffer *cudaBuffer = (CudaBuffer *) arg->value.buffer.vendorPtr;
            //  checkCudaErrors( cuMemcpyDtoH(cudaBuffer->ptr, cudaBuffer->devicePtr, cudaBuffer->sizeInBytes) );

            checkCudaErrors(cuMemFree(cudaBuffer->devicePtr));
            delete cudaBuffer;
        } else {
#ifdef VERBOSE
            std::cout << "not releasing arg "<<arg->idx<< " "<<std::endl;
#endif
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
    return (long) new CudaKernel(this, function);
}

bool CudaBackend::CudaProgram::programOK() {
    return true;
}

CudaBackend::CudaBackend(CudaBackend::CudaConfig *cudaConfig, int configSchemaLen, char *configSchema)
        : Backend((Backend::Config *) cudaConfig, configSchemaLen, configSchema) {
    std::cout << "CudaBackend constructor" << std::endl;
    CUresult err = cuInit(0);
    int deviceCount = 0;
    if (err == CUDA_SUCCESS) {
        checkCudaErrors(cuDeviceGetCount(&deviceCount));
    }
    checkCudaErrors(cuDeviceGet(&device, 0));
    checkCudaErrors(cuCtxCreate(&context, 0, device));
    std::cout << "created context" << std::endl;
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
    cuDeviceGetName(name, 100, device);
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

        //ContextDevice *contextDevice= (ContextDevice*) handle;

        std::cout << "ptx " << ptx->text << std::endl;


        // in this branch we use compilation with parameters
        const unsigned int jitNumOptions = 2;
        CUjit_option *jitOptions = new CUjit_option[jitNumOptions];
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
        delete ptx;
    } else {
        std::cout << "no ptx content!/" << std::endl;
    }
    return (long) new CudaProgram(this, nullptr, ptx, module);
}

long getBackend(void *config, int configSchemaLen, char *configSchema) {
    // Dynamic cast?
    CudaBackend::CudaConfig *cudaConfig = (CudaBackend::CudaConfig *) config;

    return (long) new CudaBackend(cudaConfig, configSchemaLen, configSchema);
}


