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

#include "opencl_backend.h"


/*
  OpenCLBuffer
  */

OpenCLBackend::OpenCLBuffer::OpenCLBuffer(Backend *backend,  BufferState_s *bufferState)
        : Backend::Buffer(backend, bufferState) {
    cl_int status;
    OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(backend);
    clMem = clCreateBuffer(
        openclBackend->context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
        bufferState->length,
        bufferState->ptr,
        &status);

    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
     bufferState->vendorPtr =  static_cast<void *>(this);


}
bool OpenCLBackend::OpenCLBuffer::copyToDevice(int accessBits) {
    bool kernelWroteToThisArg = (accessBits==WO_BYTE) |  (accessBits==RW_BYTE);
    if (backend->config->showWhy){
        std::cout<<
                 "config.alwaysCopy="<<backend->config->alwaysCopy
                 << " | arg.WO="<<(accessBits==WO_BYTE)
                 << " | arg.RW="<<(accessBits==RW_BYTE)
                 << " | kernel.wroteToThisArg="<<  kernelWroteToThisArg
                 << "Buffer state = "<< BufferState_s::stateNames[bufferState->state]
                 <<" so " ;
    }
    bool copying= backend->config->alwaysCopy;


   //  std::cout << "copyTo(" <<std::hex << (long) arg->value.buffer.memorySegment << "," << std::dec<<   bufferState->length <<")"<<std::endl;
    if (copying) {
        auto openCLQueue = dynamic_cast<OpenCLQueue *>(backend->queue);
        cl_int status = clEnqueueWriteBuffer(
                openCLQueue->command_queue,
                clMem,
                CL_FALSE,
                0,
                bufferState->length, // arg->value.buffer.sizeInBytes,
                bufferState->ptr,
                openCLQueue->eventc,
                openCLQueue->eventListPtr(),
                openCLQueue->nextEventPtr()
        );

        if (status != CL_SUCCESS) {
            std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
            exit(1);
        }
        openCLQueue->markAsCopyToDeviceAndInc();
    }
    return copying;
}

bool OpenCLBackend::OpenCLBuffer::copyFromDevice(int accessBits) {
    auto * openclBackend = dynamic_cast<OpenCLBackend *>(backend);

    bool kernelReadsFromThisArg = (accessBits==RW_BYTE) || (accessBits==RO_BYTE);
    bool copying =
            openclBackend->config->alwaysCopy
            ||  (bufferState->state == BufferState_s::NEW_STATE)
            || ((bufferState->state == BufferState_s::HOST_OWNED));

    if (openclBackend->config->showWhy){
        std::cout<<
                 "config.alwaysCopy="<<openclBackend->config->alwaysCopy
                 << " | arg.RW="<<(accessBits==RW_BYTE)
                 << " | arg.RO="<<(accessBits==RO_BYTE)
                 << " | kernel.needsToRead="<<  kernelReadsFromThisArg
                 << " | Buffer state = "<< BufferState_s::stateNames[bufferState->state]
                 <<" so "
                ;
    }
    if (copying) {

        auto openCLQueue = dynamic_cast<OpenCLQueue *>(backend->queue);
        cl_int status = clEnqueueReadBuffer(
                openCLQueue->command_queue,
                clMem,
                CL_FALSE,
                0,
                bufferState->length,//arg->value.buffer.sizeInBytes,
                bufferState->ptr,
                openCLQueue->eventc,
                openCLQueue->eventListPtr(),
                openCLQueue->nextEventPtr()
        );
        if (status != CL_SUCCESS) {
            std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
            exit(1);
        }
        openCLQueue->markAsCopyFromDeviceAndInc();
    }
    return copying;
}

OpenCLBackend::OpenCLBuffer::~OpenCLBuffer() {
    clReleaseMemObject(clMem);
}


