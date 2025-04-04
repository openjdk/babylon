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

/*
//http://mercury.pr.erau.edu/~siewerts/extra/code/digital-media/CUDA/cuda_work/samples/0_Simple/matrixMulDrv/matrixMulDrv.cpp
 */
CudaBackend::CudaBuffer::CudaBuffer(Backend *backend, Arg_s *arg, BufferState_s *bufferState)
        : Buffer(backend, arg,bufferState), devicePtr() {
    /*
     *   (void *) arg->value.buffer.memorySegment,
     *   (size_t) arg->value.buffer.sizeInBytes);
     */
    auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    if (cudaBackend->cudaConfig.traceCalls) {
        std::cout << "cuMemAlloc()" << std::endl;
    }
    CUresult status = cuMemAlloc(&devicePtr, (size_t) arg->value.buffer.sizeInBytes);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuMemAlloc() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
   // std::cout << "devptr " << std::hex<<  (long)devicePtr <<std::dec <<std::endl;
  bufferState->vendorPtr= static_cast<void *>(this);
}

CudaBackend::CudaBuffer::~CudaBuffer() {
    auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    if (cudaBackend->cudaConfig.traceCalls) {
        std::cout << "cuMemFree()"
                  << "devptr " << std::hex << (long) devicePtr << std::dec
                  << std::endl;
    }
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
    if (cudaBackend->cudaConfig.traceCalls) {
        std::cout << "copyToDevice() 0x" << std::hex << arg->value.buffer.sizeInBytes << std::dec << " "
                  << arg->value.buffer.sizeInBytes << " "
                  << "devptr " << std::hex << (long) devicePtr << std::dec
                  << std::endl;
    }


    CUresult status = cuMemcpyHtoDAsync(devicePtr, arg->value.buffer.memorySegment,
                                        arg->value.buffer.sizeInBytes,cudaBackend->cudaQueue.cuStream);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuMemcpyHtoDAsync() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    status = static_cast<CUresult >(cudaStreamSynchronize(cudaBackend->cudaQueue.cuStream));
    if (CUDA_SUCCESS != status) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
}

void CudaBackend::CudaBuffer::copyFromDevice() {
    auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    if (cudaBackend->cudaConfig.traceCalls) {
        std::cout << "copyFromDevice() 0x" << std::hex<<arg->value.buffer.sizeInBytes<<std::dec << " "<< arg->value.buffer.sizeInBytes << " "
                     << "devptr " << std::hex<<  (long)devicePtr <<std::dec
                    << std::endl;
    }
    CUresult status =cuMemcpyDtoHAsync(arg->value.buffer.memorySegment, devicePtr, arg->value.buffer.sizeInBytes,
                                       cudaBackend->cudaQueue.cuStream);
    if (CUDA_SUCCESS != status) {
        std::cerr << "cuMemcpyDtoHAsync() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    cudaError_t t1 = cudaStreamSynchronize(cudaBackend->cudaQueue.cuStream);
    if (static_cast<cudaError_t>(CUDA_SUCCESS) != t1) {
        std::cerr << "cudaStreamSynchronize() CUDA error = " << t1
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(t1))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }

}

