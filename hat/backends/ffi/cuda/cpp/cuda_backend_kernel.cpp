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


CudaBackend::CudaModule::CudaKernel::CudaKernel(Backend::CompilationUnit *program,char * name, CUfunction function)
        : Backend::CompilationUnit::Kernel(program, name), function(function) {
}

CudaBackend::CudaModule::CudaKernel::~CudaKernel() = default;
/*
long CudaBackend::CudaModule::CudaKernel::ndrange(void *argArray) {

    auto cudaBackend = CudaBackend::of(compilationUnit->backend);
    if (cudaBackend->cudaConfig.traceCalls) {
        std::cout << "ndrange(" <<  ") " << name << std::endl;
    }
    ArgSled argSled(static_cast<ArgArray_s *>(argArray));
    void *argslist[argSled.argc()];

    NDRange *ndrange = nullptr;
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
   // std::cout << "   Requested range   = " << range << std::endl;
   // std::cout << "   Range mod 1024    = " << rangemod1024 << std::endl;
   // std::cout << "   Actual range 1024 = " << (rangediv1024 * 1024) << std::endl;
  //  auto status= static_cast<CUresult>(cudaStreamSynchronize(cudaBackend->cudaQueue.cuStream));

  //  cudaBackend->cudaQueue.wait();
    auto status= cuLaunchKernel(function,
                                   rangediv1024, 1, 1,
                                   1024, 1, 1,
                                   0, cudaBackend->cudaQueue.cuStream ,
                    argslist, nullptr);

    WHERE{.f=__FILE__, .l=__LINE__, .e=status, .t="cuLaunchKernel"}.report();
 //   cudaBackend->cudaQueue.wait();


    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            auto bufferState = BufferState_s::of(arg)->vendorPtr;
            auto cudaBuffer = static_cast<CudaBuffer *>(bufferState);
            cudaBuffer->copyFromDevice();
        }
    }
    cudaBackend->cudaQueue.wait();

    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        if (arg->variant == '&') {
            auto bufferState = BufferState_s::of(arg)->vendorPtr;
            auto cudaBuffer = static_cast<CudaBuffer *>(bufferState);
            delete cudaBuffer;

        }
    }

    return (long) 0;
} */

CudaBackend::CudaModule::CudaKernel * CudaBackend::CudaModule::CudaKernel::of(long kernelHandle){
    return reinterpret_cast<CudaBackend::CudaModule::CudaKernel *>(kernelHandle);
}
CudaBackend::CudaModule::CudaKernel * CudaBackend::CudaModule::CudaKernel::of(Backend::CompilationUnit::Kernel *kernel){
    return dynamic_cast<CudaBackend::CudaModule::CudaKernel *>(kernel);
}

bool CudaBackend::CudaModule::CudaKernel::setArg(KernelArg *arg){
    argslist[arg->idx] = static_cast<void *>(&arg->value);
    return true;
}
bool CudaBackend::CudaModule::CudaKernel::setArg(KernelArg *arg, Buffer *buffer) {
    argslist[arg->idx] = static_cast<void *>(&dynamic_cast<CudaBuffer *>(buffer)->devicePtr);
    return true;
}
