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
void dispatchKernel(Kernel kernel, KernelContext kc, Arg ... args) {
    for (int argn = 0; argn<args.length; argn++){
      Arg arg = args[argn];
      if (!minimizingBuffers || (((arg.flags &JavaDirty)==JavaDirty) && kernel.readsFrom(arg))) {
         enqueueCopyToDevice(arg);
      }
    }
    enqueueKernel(kernel);
    waitForKernel();

    for (int argn = 0; argn<args.length; argn++){
      Arg arg = args[argn];
      if (!minimizingBuffers){
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
bool shouldCopyToDevice(BufferState_s *bufferState, Arg_s *arg ){
   bool kernelReadsFromThisArg = (arg->value.buffer.access==RW_BYTE) || (arg->value.buffer.access==RO_BYTE);
   bool isHostDirtyOrNew = bufferState->isHostDirty() | bufferState->isHostNew();

   bool result=  (kernelReadsFromThisArg & isHostDirtyOrNew);
   if (result && bufferState->isDeviceDirty()){
         std::cout << "already still on GPU!"<<std::endl;
         result= false;
   }
   return result;
}
bool shouldCopyFromDevice( BufferState_s *bufferState, Arg_s *arg ){
   bool kernelWroteToThisArg = (arg->value.buffer.access==WO_BYTE) |  (arg->value.buffer.access==RW_BYTE);
   bool result = kernelWroteToThisArg;
   //if (!result){
    //  std::cout << "shouldCopyFromDevice false"<<std::endl;
  // }
   return result;
}

long OpenCLBackend::OpenCLProgram::OpenCLKernel::ndrange(void *argArray) {

   // std::cout << "ndrange(" << range << ") " << std::endl;
    ArgSled argSled(static_cast<ArgArray_s *>(argArray));
    OpenCLBackend *openclBackend = dynamic_cast<OpenCLBackend*>(program->backend);
  //  std::cout << "Kernel name '"<< (dynamic_cast<Backend::Program::Kernel*>(this))->name<<"'"<<std::endl;
    openclBackend->openclQueue.marker(openclBackend->openclQueue.EnterKernelDispatchBits,
     (dynamic_cast<Backend::Program::Kernel*>(this))->name);
    if (openclBackend->openclConfig.trace){
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
               if (openclBackend->openclConfig.trace){
                  std::cout << "arg["<<i<<"] = "<< std::hex << (int)(arg->value.buffer.access);
                  switch (arg->value.buffer.access){
                      case RO_BYTE: std::cout << " RO";break;
                      case WO_BYTE: std::cout << " WO";break;
                      case RW_BYTE: std::cout << " RW"; break;
                  }
                  std::cout << std::endl;
               }

               BufferState_s * bufferState = BufferState_s::of(arg);
               OpenCLBuffer * openclBuffer =nullptr;
               if (bufferState->isHostNew()){
                  openclBuffer = new OpenCLBuffer(this, arg);
                  if (openclBackend->openclConfig.trace){
                     std::cout << "We allocated arg "<<i<<" buffer "<<std::endl;
                  }
                  bufferState->clearHostNew();
               }else{
                  if (openclBackend->openclConfig.trace){
                      std::cout << "Were reusing  arg "<<i<<" buffer "<<std::endl;
                  }
                  openclBuffer=  static_cast<OpenCLBuffer*>(bufferState->vendorPtr);
                }
                if (!openclBackend->openclConfig.minimizeCopies
                   || shouldCopyToDevice(bufferState, arg)){

                       if (openclBackend->openclConfig.traceCopies){
                        //  std::cout << "We are not minimising copies OR (HOST is JAVA dirty and the kernel is READS this arg) so copying arg " << arg->idx <<" to device "<< std::endl;
                       }
                       bufferState->clearHostDirty();
                       if (openclBackend->openclConfig.traceEnqueues){
                           std::cout << "copying arg " << arg->idx <<" to device "<< std::endl;
                       }
                       openclBuffer->copyToDevice();

                    }else{
                     if (openclBackend->openclConfig.traceSkippedCopies){
                                          std::cout << "NOT copying arg " << arg->idx <<" to device "<< std::endl;
                                                       // bufferState->dump("After copy from device");
                                     }
                    }

                cl_int status = clSetKernelArg(kernel, arg->idx, sizeof(cl_mem), &openclBuffer->clMem);
                if (status != CL_SUCCESS) {
                    std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                    exit(1);
                }
                if (openclBackend->openclConfig.trace){
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
                cl_int status = clSetKernelArg(kernel, arg->idx, arg->size(), (void *) &arg->value);
                if (status != CL_SUCCESS) {
                    std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
                    exit(1);
                }
                if (openclBackend->openclConfig.trace){
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
    if (openclBackend->openclConfig.trace){
       std::cout << "ndrange = " << ndrange->maxX << std::endl;
    }
    size_t dims = 1;
    cl_int status = clEnqueueNDRangeKernel(
            openclBackend->openclQueue.command_queue,
            kernel,
            dims,
            nullptr,
            &globalSize,
            nullptr,
            openclBackend->openclQueue.eventc,
            openclBackend->openclQueue.eventListPtr(),
            openclBackend->openclQueue.nextEventPtr());
    openclBackend->openclQueue.markAsNDRangeAndInc();
    if (status != CL_SUCCESS) {
        std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
        exit(1);
    }
    if (openclBackend->openclConfig.trace | openclBackend->openclConfig.traceEnqueues){
       std::cout << "enqueued kernel dispatch globalSize=" << globalSize << std::endl;
    }


       for (int i = 1; i < argSled.argc(); i++) { // note i = 1... we don't need to copy back the KernelContext
          Arg_s *arg = argSled.arg(i);
          if (arg->variant == '&') {
             BufferState_s * bufferState = BufferState_s::of(arg );
             if (!openclBackend->openclConfig.minimizeCopies || shouldCopyFromDevice(bufferState,arg)){
                static_cast<OpenCLBuffer *>(bufferState->vendorPtr)->copyFromDevice();
                //if (openclBackend->openclConfig.traceCopies){
                    //std::cout << "copying arg " << arg->idx <<" from device "<< std::endl;
                   // bufferState->dump("After copy from device");
                //}
                if (openclBackend->openclConfig.traceEnqueues){
                   std::cout << "copying arg " << arg->idx <<" from device "<< std::endl;
                }
                bufferState->setDeviceDirty();
             }else{
                 if (openclBackend->openclConfig.traceSkippedCopies){
                      std::cout << "NOT copying arg " << arg->idx <<" from device "<< std::endl;
                                   // bufferState->dump("After copy from device");
                 }
             }
          }
       }



      openclBackend->openclQueue.marker(openclBackend->openclQueue.LeaveKernelDispatchBits,
           (dynamic_cast<Backend::Program::Kernel*>(this))->name
      );
      openclBackend->openclQueue.wait();
      openclBackend->openclQueue.release();
    return 0;
}
