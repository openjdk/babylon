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

/*
//http://mercury.pr.erau.edu/~siewerts/extra/code/digital-media/CUDA/cuda_work/samples/0_Simple/matrixMulDrv/matrixMulDrv.cpp
 */
CudaBackend::CudaBuffer::CudaBuffer(Backend *backend,  BufferState *bufferState)
        : Buffer(backend, bufferState), devicePtr() {

    const auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    if (cudaBackend->config->traceCalls) {
        std::cout << "CudaBuffer()" << std::endl;
    }

    WHERE{.f=__FILE__, .l=__LINE__,
            .e=cuMemAlloc(&devicePtr, static_cast<size_t>(bufferState->length)),
            .t="cuMemAlloc"
    }.report();
    if (cudaBackend->config->traceCalls) {
        std::cout << "devptr=" << std::hex<<  static_cast<long>(devicePtr) << "stream=" <<dynamic_cast<CudaQueue *>(backend->queue)->cuStream <<std::dec <<std::endl;
    }

  bufferState->vendorPtr= static_cast<void *>(this);
}

CudaBackend::CudaBuffer::~CudaBuffer() {
    const auto cudaBackend = dynamic_cast<CudaBackend*>(backend);
    if (cudaBackend->config->traceCalls) {
        const std::thread::id thread_id = std::this_thread::get_id();

        std::cout << "~CudaBuffer()"<< "devptr =" << std::hex << (long) devicePtr << std::dec
                << " thread=" <<thread_id
        <<std::endl;
    }
    WHERE{.f=__FILE__, .l=__LINE__,
            .e=cuMemFree(devicePtr),
            .t="cuMemFree"
    }.report();
    bufferState->vendorPtr= nullptr;
}


