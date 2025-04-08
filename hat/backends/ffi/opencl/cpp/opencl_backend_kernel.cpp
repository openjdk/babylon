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
  OpenCLKernel
  */

OpenCLBackend::OpenCLProgram::OpenCLKernel::OpenCLKernel(Backend::CompilationUnit *compilationUnit, char* name, cl_kernel kernel)
    : Backend::CompilationUnit::Kernel(compilationUnit, name), kernel(kernel){
}

OpenCLBackend::OpenCLProgram::OpenCLKernel::~OpenCLKernel() {
    clReleaseKernel(kernel);
}


/*
void dispatchKernel(Kernel kernel, KernelContext kc, Arg ... args) {
    for (int argn = 0; argn<args.length; argn++){
      Arg arg = args[argn];
      if (alwaysCopyBuffers || (((arg.flags &JavaDirty)==JavaDirty) && kernel.readsFrom(arg))) {
         enqueueCopyToDevice(arg);
      }
    }
    enqueueKernel(kernel);
    waitForKernel();

    for (int argn = 0; argn<args.length; argn++){
      Arg arg = args[argn];
      if (alwaysCopyBuffers){
         enqueueCopyFromDevice(arg);
         arg.flags = 0;
      }else{
          if (kernel.writesTo(arg)) {
             arg.flags = DeviceDirty;
          }else{
             arg.flags = 0;
          }
      }
    }

}
*/

bool OpenCLBackend::OpenCLProgram::OpenCLKernel::setArg(Arg_s *arg, Buffer *buffer){
    OpenCLBuffer * openCLBuffer = dynamic_cast<OpenCLBuffer *>(buffer);
    cl_int status = clSetKernelArg(kernel, arg->idx, sizeof(cl_mem), &openCLBuffer->clMem);
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
        return false;
    }
    return true;
}
bool OpenCLBackend::OpenCLProgram::OpenCLKernel::setArg(Arg_s *arg) {
    cl_int status = clSetKernelArg(kernel, arg->idx, arg->size(), (void *) &arg->value);
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
        return false;
    }
    return true;
}


long OpenCLBackend::OpenCLProgram::OpenCLKernel::ndrange(void *argArray) {

   // std::cout << "ndrange(" << range << ") " << std::endl;
    ArgSled argSled(static_cast<ArgArray_s *>(argArray));
    Backend *backend = dynamic_cast<Backend*>(compilationUnit->backend);

    OpenCLQueue * openCLQueue = dynamic_cast<OpenCLQueue *>(backend->queue);
    openCLQueue->marker(Backend::Queue::EnterKernelDispatchBits,
     (dynamic_cast<Backend::CompilationUnit::Kernel*>(this))->name);
    if (backend->config->traceCalls){
       std::cout << "ndrange(\"" <<  (dynamic_cast<Backend::CompilationUnit::Kernel*>(this))->name<< "\"){"<<std::endl;
        std::cout << "Kernel name '"<< (dynamic_cast<Backend::CompilationUnit::Kernel*>(this))->name<<"'"<<std::endl;
    }
    if (backend->config->trace){
       Sled::show(std::cout, argArray);
    }
    NDRange *ndrange = nullptr;
    for (int i = 0; i < argSled.argc(); i++) {
        Arg_s *arg = argSled.arg(i);
        switch (arg->variant) {
            case '&': {
               if (arg->idx == 0){
                   ndrange = static_cast<NDRange *>(arg->value.buffer.memorySegment);
               }
               if (backend->config->trace){
                  std::cout << "arg["<<i<<"] = "<< std::hex << (int)(arg->value.buffer.access);
                  switch (arg->value.buffer.access){
                      case RO_BYTE: std::cout << " RO";break;
                      case WO_BYTE: std::cout << " WO";break;
                      case RW_BYTE: std::cout << " RW"; break;
                  }
                  std::cout << std::endl;
               }

               BufferState_s * bufferState = BufferState_s::of(arg);

               //Sanity check the buffers
               // These sanity check finds errors passing memory segments which are not Buffers

               if (bufferState->ptr != arg->value.buffer.memorySegment){
                   std::cerr <<"bufferState->ptr !=  arg->value.buffer.memorySegment"<<std::endl;
                   std::exit(1);
               }

               if ((bufferState->vendorPtr == 0L) && (bufferState->state != BufferState_s::NEW_STATE)){
                   std::cerr << "Warning:  Unexpected initial state for arg "<< i
                      <<" of kernel '"<<(dynamic_cast<Backend::CompilationUnit::Kernel*>(this))->name<<"'"
                      << " state=" << bufferState->state<< " '"
                      << BufferState_s::stateNames[bufferState->state]<< "'"
                      << " vendorPtr" << bufferState->vendorPtr<<std::endl;
               }
               // End of sanity checks

               Buffer * buffer = backend->getOrCreateBuffer(bufferState);

                if (buffer->copyToDevice(arg->value.buffer.access)){
                }else if (backend->config->traceSkippedCopies){
                    std::cout << "NOT copying arg " << arg->idx <<" to device "<< std::endl;
                }
                setArg(arg, buffer);
                if (backend->config->trace){
                   std::cout << "set buffer arg " << arg->idx << std::endl;
                }
                break;
            }
             case 'B':
             case 'S':
             case 'C':
             case 'I':
             case 'F':
             case 'J':
             case 'D':
             {
                 setArg(arg);

                 if (backend->config->trace){
                    std::cerr << "set " <<arg->variant << " " << arg->idx << std::endl;
                 }
                 break;
            }
            default: {
                std::cerr << "unexpected variant setting args in OpenCLkernel::ndrange " << (char) arg->variant << std::endl;
                exit(1);
            }
        }
    }

    size_t globalSize = ndrange->maxX;
    if (backend->config->trace){
       std::cout << "ndrange = " << ndrange->maxX << std::endl;
    }
    size_t dims = 1;
    cl_int status = clEnqueueNDRangeKernel(
            openCLQueue->command_queue,
            kernel,
            dims,
            nullptr,
            &globalSize,
            nullptr,
            openCLQueue->eventc,
            openCLQueue->eventListPtr(),
            openCLQueue->nextEventPtr());
    openCLQueue->markAsNDRangeAndInc();
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if (backend->config->trace | backend->config->traceEnqueues){
       std::cout << "enqueued kernel dispatch \"" << (dynamic_cast<Backend::CompilationUnit::Kernel*>(this))->name <<
       "\" globalSize=" << globalSize << std::endl;
    }


       for (int i = 0; i < argSled.argc(); i++) { // note i = 1... we don't need to copy back the KernelContext
          Arg_s *arg = argSled.arg(i);
          if (arg->variant == '&') {
             BufferState_s * bufferState = BufferState_s::of(arg );
             OpenCLBuffer *openclBuffer = static_cast<OpenCLBuffer *>(bufferState->vendorPtr);
             if ( openclBuffer->copyFromDevice(arg->value.buffer.access)){
                if (backend->config->traceCopies||backend->config->traceEnqueues){
                     std::cout << "copying arg " << arg->idx <<" from device "<< std::endl;
                }
             }else{
                 if (backend->config->traceSkippedCopies){
                     std::cout << "NOT copying arg " << arg->idx <<" from device "<< std::endl;
                 }
             }
             bufferState->state = BufferState_s::DEVICE_OWNED; // This seems wrong
          }
       }
    openCLQueue->marker(Backend::Queue::LeaveKernelDispatchBits,name);
    backend->queue->wait();
    backend->queue->release();
      if (backend->config->traceCalls){
          std::cout << "\"" << name<< "\"}"<<std::endl;
      }
      return 0;
}
