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
#include <thread>
#include "cuda_backend.h"

CudaBackend::CudaQueue::CudaQueue(Backend *backend)
        : Backend::Queue(backend),cuStream(),streamCreationThread() {
}
void CudaBackend::CudaQueue::init(){
    streamCreationThread = std::this_thread::get_id();
    if (backend->config->traceCalls){
        std::cout << "init() 0x"
                  << " thread=" <<streamCreationThread
                  << std::endl;
    }

    WHERE{.f=__FILE__ , .l=__LINE__,
          .e=cuStreamCreate(&cuStream,CU_STREAM_DEFAULT),
          .t= "cuStreamCreate"
    }.report();

    if (backend->config->traceCalls){
        std::cout << "exiting init() 0x"
                  << " custream=" <<std::hex<<streamCreationThread <<std::dec
                  << std::endl;
    }
    }

//void CudaBackend::CudaQueue::sync(const char *file, int line) const {

//}


void CudaBackend::CudaQueue::wait(){
    WHERE{.f=__FILE__, .l=__LINE__,
          .e=cuStreamSynchronize(cuStream),
          .t= "cuStreamSynchronize"
    }.report();
}


void CudaBackend::CudaQueue::computeStart(){
    wait(); // should be no-op
    release(); // also ;
}



void CudaBackend::CudaQueue::computeEnd(){

}




void CudaBackend::CudaQueue::release(){

}

CudaBackend::CudaQueue::~CudaQueue(){
   // delete []events;
    WHERE{.f=__FILE__, .l=__LINE__,
            .e=cuStreamDestroy(cuStream),
            .t= "cuStreamDestroy"
    }.report();
}

void CudaBackend::CudaQueue::copyToDevice(Buffer *buffer) {
    //auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    auto *cudaBuffer = dynamic_cast<CudaBuffer *>(buffer);
    std::thread::id thread_id = std::this_thread::get_id();
    if (thread_id != streamCreationThread){
        std::cout << "copyToDevice()  thread=" <<thread_id<< " != "<< streamCreationThread<< std::endl;
    }
    if (backend->config->traceCalls) {

        std::cout << "copyToDevice() 0x"
                << std::hex<<cudaBuffer->bufferState->length<<std::dec << "/"
                << cudaBuffer->bufferState->length << " "
                << "devptr=" << std::hex<<  (long)cudaBuffer->devicePtr <<std::dec
                << " thread=" <<thread_id
                  << std::endl;
    }
    WHERE{.f=__FILE__, .l=__LINE__,
            .e=cuMemcpyHtoDAsync(
                    cudaBuffer->devicePtr,
                    cudaBuffer->bufferState->ptr,
                    cudaBuffer->bufferState->length,
                    dynamic_cast<CudaQueue*>(backend->queue)->cuStream),
            .t="cuMemcpyHtoDAsync"
    }.report();

}

void CudaBackend::CudaQueue::copyFromDevice(Buffer *buffer) {
    auto *cudaBuffer = dynamic_cast<CudaBuffer *>(buffer);
    //auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    std::thread::id thread_id = std::this_thread::get_id();
    if (thread_id != streamCreationThread){
        std::cout << "copyFromDevice()  thread=" <<thread_id<< " != "<< streamCreationThread<< std::endl;
    }
    if (backend->config->traceCalls) {

        std::cout << "copyFromDevice() 0x"
                  << std::hex<<cudaBuffer->bufferState->length<<std::dec << "/"
                  << cudaBuffer->bufferState->length << " "
                  << "devptr=" << std::hex<<  (long)cudaBuffer->devicePtr <<std::dec
                << " thread=" <<thread_id
                  << std::endl;
    }


    WHERE{.f=__FILE__, .l=__LINE__,
            .e=cuMemcpyDtoHAsync(
                    cudaBuffer->bufferState->ptr,
                    cudaBuffer->devicePtr,
                    cudaBuffer->bufferState->length,
                                 dynamic_cast<CudaQueue*>(backend->queue)->cuStream),
            .t="cuMemcpyDtoHAsync"
    }.report();

}

void CudaBackend::CudaQueue::dispatch(KernelContext *kernelContext, CompilationUnit::Kernel *kernel) {
    auto cudaKernel = dynamic_cast<CudaModule::CudaKernel *>(kernel);

    int range = kernelContext->maxX;
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
    std::thread::id thread_id = std::this_thread::get_id();
    if (thread_id != streamCreationThread){
        std::cout << "dispatch()  thread=" <<thread_id<< " != "<< streamCreationThread<< std::endl;
    }

    auto status = cuLaunchKernel(cudaKernel->function,
                                 rangediv1024, 1, 1,
                                 1024, 1, 1,
                                 0, cuStream,
                                 cudaKernel->argslist, nullptr);

    WHERE{.f=__FILE__, .l=__LINE__, .e=status, .t="cuLaunchKernel"}.report();
}