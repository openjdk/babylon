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

void CudaBackend::CudaQueue::wait(){
    CUDA_CHECK(cuStreamSynchronize(cuStream), "cuStreamSynchronize");
}


void CudaBackend::CudaQueue::computeStart() {
    wait(); // should be no-op
    release(); // also ;
}

void CudaBackend::CudaQueue::computeEnd() {

}

void CudaBackend::CudaQueue::release() {

}

CudaBackend::CudaQueue::~CudaQueue() {
    CUDA_CHECK(cuStreamDestroy(cuStream), "cuStreamDestroy");
}

void CudaBackend::CudaQueue::copyToDevice(Buffer *buffer) {
    const auto *cudaBuffer = dynamic_cast<CudaBuffer *>(buffer);
    const std::thread::id thread_id = std::this_thread::get_id();
    if (thread_id != streamCreationThread){
        std::cout << "copyToDevice()  thread=" <<thread_id<< " != "<< streamCreationThread<< std::endl;
    }
    if (backend->config->traceCalls) {

        std::cout << "copyToDevice() 0x"
                << std::hex<<cudaBuffer->bufferState->length<<std::dec << "/"
                << cudaBuffer->bufferState->length << " "
                << "devptr=" << std::hex<<  static_cast<long>(cudaBuffer->devicePtr) <<std::dec
                << " thread=" <<thread_id
                  << std::endl;
    }

    CUDA_CHECK(cuMemcpyHtoDAsync(cudaBuffer->devicePtr,
                    cudaBuffer->bufferState->ptr,
                    cudaBuffer->bufferState->length,
                    dynamic_cast<CudaQueue*>(backend->queue)->cuStream), "cuMemcpyHtoDAsync");
}

void CudaBackend::CudaQueue::copyFromDevice(Buffer *buffer) {
    const auto *cudaBuffer = dynamic_cast<CudaBuffer *>(buffer);
    const std::thread::id thread_id = std::this_thread::get_id();
    if (thread_id != streamCreationThread){
        std::cout << "copyFromDevice()  thread=" <<thread_id<< " != "<< streamCreationThread<< std::endl;
    }
    if (backend->config->traceCalls) {

        std::cout << "copyFromDevice() 0x"
                  << std::hex<<cudaBuffer->bufferState->length<<std::dec << "/"
                  << cudaBuffer->bufferState->length << " "
                  << "devptr=" << std::hex<<  static_cast<long>(cudaBuffer->devicePtr) <<std::dec
                << " thread=" <<thread_id
                  << std::endl;
    }

    CUDA_CHECK(cuMemcpyDtoHAsync(cudaBuffer->bufferState->ptr,
                                cudaBuffer->devicePtr,
                                cudaBuffer->bufferState->length,
                                dynamic_cast<CudaQueue*>(backend->queue)->cuStream),
                                "cuMemcpyDtoHAsync");

}

// TODO: Improve heuristics to decide a better block size, if possible.
// The following is just a rough number to fit into a modern NVIDIA GPU.
int CudaBackend::CudaQueue::estimateThreadsPerBlock(int dimensions) {
    switch (dimensions) {
        case 1: return 256;
        case 2: return 16;
        case 3: return 16;
        default: return 1;
    }
}

void CudaBackend::CudaQueue::dispatch(KernelContext *kernelContext, CompilationUnit::Kernel *kernel) {
    const auto cudaKernel = dynamic_cast<CudaModule::CudaKernel *>(kernel);

    int threadsPerBlockX;
    int threadsPerBlockY = 1;
    int threadsPerBlockZ = 1;

    // The local and global mesh dimensions match by design from the Java APIs
    const int dimensions = kernelContext->globalMesh.dimensions;
    if (kernelContext -> localMesh.maxX > 0) {
        threadsPerBlockX = kernelContext -> localMesh.maxX;
    } else {
        threadsPerBlockX = estimateThreadsPerBlock(dimensions);
    }
    if (kernelContext-> localMesh.maxY > 0) {
        threadsPerBlockY = kernelContext-> localMesh.maxY;
    } else if (dimensions > 1) {
        threadsPerBlockY = estimateThreadsPerBlock(dimensions);
    }
    if (kernelContext-> localMesh.maxZ > 0) {
        threadsPerBlockZ = kernelContext-> localMesh.maxZ;
    } else if (dimensions > 2) {
        threadsPerBlockZ = estimateThreadsPerBlock(dimensions);
    }

    int blocksPerGridX = (kernelContext->globalMesh.maxX + threadsPerBlockX - 1) / threadsPerBlockX;
    int blocksPerGridY = 1;
    int blocksPerGridZ = 1;

    if (dimensions > 1) {
        blocksPerGridY = (kernelContext->globalMesh.maxY + threadsPerBlockY - 1) / threadsPerBlockY;
    }
    if (dimensions > 2) {
        blocksPerGridZ = (kernelContext->globalMesh.maxZ + threadsPerBlockZ - 1) / threadsPerBlockZ;
    }

    // Enable debug information with trace. Use HAT=INFO
    if (backend->config->info) {
        std::cout << "Dispatching the CUDA kernel" << std::endl;
        std::cout << "   \\_ BlocksPerGrid   = [" << blocksPerGridX << "," << blocksPerGridY << "," << blocksPerGridZ << "]" << std::endl;
        std::cout << "   \\_ ThreadsPerBlock = [" << threadsPerBlockX << "," << threadsPerBlockY << "," << threadsPerBlockZ << "]" << std::endl;
    }

    const std::thread::id thread_id = std::this_thread::get_id();
    if (thread_id != streamCreationThread){
        std::cout << "dispatch()  thread=" <<thread_id<< " != "<< streamCreationThread<< std::endl;
    }

    const auto status = cuLaunchKernel(cudaKernel->function, //
                                 blocksPerGridX, blocksPerGridY, blocksPerGridZ, //
                                 threadsPerBlockX, threadsPerBlockY, threadsPerBlockZ, //
                                 0, //
                                 cuStream, //
                                 cudaKernel->argslist, //
                                 nullptr);

    CUDA_CHECK(status, "cuLaunchKernel");
}
