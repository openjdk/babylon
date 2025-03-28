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

OpenCLBackend::OpenCLBuffer::OpenCLBuffer(Backend *backend, Arg_s *arg, BufferState_s *bufferState)
        : Backend::Buffer(backend, arg), bufferState(bufferState) {
    cl_int status;
    OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(backend);
    clMem = clCreateBuffer(
        openclBackend->context,
        CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
        bufferState->length,// arg->value.buffer.sizeInBytes,
        arg->value.buffer.memorySegment,
        &status);

    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
     bufferState->vendorPtr =  static_cast<void *>(this);
    if (openclBackend->openclConfig.traceCopies){
        std::cout << "created buffer for arg idx "<< arg->idx << std::endl;
    }

}

bool OpenCLBackend::OpenCLBuffer::shouldCopyToDevice( Arg_s *arg){
//std::cout << "shouldCopyToDevice( Arg_s *arg)" <<std::endl;
// std::cout <<std::hex;
//// std::cout << "arg=="<<((long) arg) <<std::endl;
// std::cout << "arg->idx=="<<arg->idx <<std::endl;
 //  std::cout << "bufferState=="<<((long) bufferState) <<std::endl;
  //  std::cout << "kernel=="<<((long) kernel) <<std::endl;
  //    std::cout << "kernel->name=="<<kernel->name <<std::endl;
  //   std::cout << "kernel->program=="<<((long) kernel->program) <<std::endl;
   //   std::cout << "kernel->program->backend=="<<((long) kernel->program->backend) <<std::endl;
   //   std::cout <<std::dec;
         OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(backend);


   bool kernelReadsFromThisArg = (arg->value.buffer.access==RW_BYTE) || (arg->value.buffer.access==RO_BYTE);
   bool isAlwaysCopyingOrNewStateOrHostOwned =
        openclBackend->openclConfig.alwaysCopy
        ||  (bufferState->state == BufferState_s::NEW_STATE)
        || ((bufferState->state == BufferState_s::HOST_OWNED));

   if (openclBackend->openclConfig.showWhy){
       std::cout<<
                   "config.alwaysCopy="<<openclBackend->openclConfig.alwaysCopy
                   << " | arg.RW="<<(arg->value.buffer.access==RW_BYTE)
                   << " | arg.RO="<<(arg->value.buffer.access==RO_BYTE)
                   << " | kernel.needsToRead="<<  kernelReadsFromThisArg
                   << " | Buffer state = "<< BufferState_s::stateNames[bufferState->state]
                   <<" so "
                     ;
     }
     return isAlwaysCopyingOrNewStateOrHostOwned;
}
bool OpenCLBackend::OpenCLBuffer::shouldCopyFromDevice(Arg_s *arg){
   OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(backend);
 bool kernelWroteToThisArg = (arg->value.buffer.access==WO_BYTE) |  (arg->value.buffer.access==RW_BYTE);
       if (openclBackend->openclConfig.showWhy){
           std::cout<<
             "config.alwaysCopy="<<openclBackend->openclConfig.alwaysCopy
                << " | arg.WO="<<(arg->value.buffer.access==WO_BYTE)
                << " | arg.RW="<<(arg->value.buffer.access==RW_BYTE)
                << " | kernel.wroteToThisArg="<<  kernelWroteToThisArg
                << "Buffer state = "<< BufferState_s::stateNames[bufferState->state]
                <<" so " ;
       }
       return openclBackend->openclConfig.alwaysCopy;
}


void OpenCLBackend::OpenCLBuffer::copyToDevice() {
  //  OpenCLKernel *openclKernel = dynamic_cast<OpenCLKernel *>(kernel);
    OpenCLBackend *openclBackend = dynamic_cast<OpenCLBackend *>(backend);
   //  std::cout << "copyTo(" <<std::hex << (long) arg->value.buffer.memorySegment << "," << std::dec<<   bufferState->length <<")"<<std::endl;

    cl_int status = clEnqueueWriteBuffer(
       openclBackend->openclQueue.command_queue,
       clMem,
       CL_FALSE,
       0,
       bufferState->length, // arg->value.buffer.sizeInBytes,
       arg->value.buffer.memorySegment,
       openclBackend->openclQueue.eventc,
       openclBackend->openclQueue.eventListPtr(),
       openclBackend->openclQueue.nextEventPtr()
    );
    openclBackend->openclQueue.markAsCopyToDeviceAndInc(arg->idx);

    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if(openclBackend->openclConfig.traceCopies){
        std::cout << "enqueued buffer for arg idx " << arg->idx << " in OpenCLBuffer::copyToDevice()" << std::endl;
    }
}

void OpenCLBackend::OpenCLBuffer::copyFromDevice() {
//    OpenCLKernel * openclKernel = dynamic_cast<OpenCLKernel *>(kernel);
    OpenCLBackend * openclBackend = dynamic_cast<OpenCLBackend *>(backend);
 //  std::cout << "copyFrom(" <<std::hex << (long) arg->value.buffer.memorySegment << "," << std::dec<<   bufferState->length <<")"<<std::endl;

    cl_int status = clEnqueueReadBuffer(
       openclBackend->openclQueue.command_queue,
       clMem,
       CL_FALSE,
       0,
       bufferState->length,//arg->value.buffer.sizeInBytes,
       arg->value.buffer.memorySegment,
       openclBackend->openclQueue.eventc,
       openclBackend->openclQueue.eventListPtr(),
       openclBackend->openclQueue.nextEventPtr()
    );
    openclBackend->openclQueue.markAsCopyFromDeviceAndInc(arg->idx);
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if(openclBackend->openclConfig.traceCopies){
       std::cout << "enqueued buffer for arg idx " << arg->idx << " in OpenCLBuffer::copyFromDevice()" << std::endl;
    }
}

OpenCLBackend::OpenCLBuffer::~OpenCLBuffer() {
    clReleaseMemObject(clMem);
}


