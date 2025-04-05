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
        std::cout << "CudaBuffer()" << std::endl;
    }

    WHERE{.f=__FILE__, .l=__LINE__,
            .e=cuMemAlloc(&devicePtr, (size_t) arg->value.buffer.sizeInBytes),
            .t="cuMemAlloc"
    }.report();
    if (cudaBackend->cudaConfig.traceCalls) {
        std::cout << "devptr " << std::hex<<  (long)devicePtr <<std::dec <<std::endl;
    }
  bufferState->vendorPtr= static_cast<void *>(this);
}

CudaBackend::CudaBuffer::~CudaBuffer() {
    auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    if (cudaBackend->cudaConfig.traceCalls) {
        std::cout << "~CudaBuffer()"<< "devptr " << std::hex << (long) devicePtr << std::dec<< std::endl;
    }
    WHERE{.f=__FILE__, .l=__LINE__,
            .e=cuMemFree(devicePtr),
            .t="cuMemFree"
    }.report();
    bufferState->vendorPtr= nullptr;
}

void CudaBackend::CudaBuffer::copyToDevice() {
    auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    if (cudaBackend->cudaConfig.traceCalls) {
        std::cout << "copyToDevice() 0x"
                  << std::hex << arg->value.buffer.sizeInBytes << std::dec << "/"
                  << arg->value.buffer.sizeInBytes << " "
                  << "devptr " << std::hex << (long) devicePtr << std::dec
                  << std::endl;
    }
    WHERE{.f=__FILE__, .l=__LINE__,
            .e=cuMemcpyHtoDAsync(devicePtr, arg->value.buffer.memorySegment,
                                 arg->value.buffer.sizeInBytes,cudaBackend->cudaQueue.cuStream),
            .t="cuMemcpyHtoDAsync"
    }.report();
    cudaBackend->cudaQueue.wait();
}

void CudaBackend::CudaBuffer::copyFromDevice() {
    auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    if (cudaBackend->cudaConfig.traceCalls) {
        std::cout << "copyFromDevice() 0x"
                     << std::hex<<arg->value.buffer.sizeInBytes<<std::dec << "/"
                     << arg->value.buffer.sizeInBytes << " "
                     << "devptr " << std::hex<<  (long)devicePtr <<std::dec
                    << std::endl;
    }
    WHERE{.f=__FILE__, .l=__LINE__,
            .e=cuMemcpyDtoHAsync(arg->value.buffer.memorySegment, devicePtr, arg->value.buffer.sizeInBytes,
                                 cudaBackend->cudaQueue.cuStream),
            .t="cuMemcpyDtoHAsync"
    }.report();

    cudaBackend->cudaQueue.wait();

}

