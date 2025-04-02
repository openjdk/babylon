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

CudaBackend::CudaQueue::CudaQueue(Backend *backend)
        : Backend::Queue(backend),cudaStream(){
    auto status =  cudaStreamCreate(&cudaStream);
    if (::cudaSuccess != status) {
        std::cerr << "cudaStreamCreate() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
    }



void CudaBackend::CudaQueue::showEvents(int width) {

}
void CudaBackend::CudaQueue::wait(){
    if (eventc > 0){
      //  cl_int status = clWaitForEvents(eventc, events);
      //  if (status != CL_SUCCESS) {
          //  std::cerr << "failed clWaitForEvents" << CudaBackend::errorMsg(status) << std::endl;
           // exit(1);
       // }
    }
}

void CudaBackend::CudaQueue::marker(int bits){
   // cl_int status = clEnqueueMarkerWithWaitList(
          //  command_queue,
           // this->eventc, this->eventListPtr(),this->nextEventPtr()
   // );
   // if (status != CL_SUCCESS){
     //   std::cerr << "failed to clEnqueueMarkerWithWaitList "<<errorMsg(status)<< std::endl;
     //   std::exit(1);
  //  }
   // inc(bits);
}
void CudaBackend::CudaQueue::marker(int bits, const char* arg){
   // cl_int status = clEnqueueMarkerWithWaitList(
          //  command_queue,
          //  this->eventc, this->eventListPtr(),this->nextEventPtr()
  //  );
   // if (status != CL_SUCCESS){
     //   std::cerr << "failed to clEnqueueMarkerWithWaitList "<<errorMsg(status)<< std::endl;
      //  std::exit(1);
   // }
   // inc(bits, arg);
}
void CudaBackend::CudaQueue::marker(int bits, int arg){
    //cl_int status = clEnqueueMarkerWithWaitList(
          //  command_queue,
        //    this->eventc, this->eventListPtr(),this->nextEventPtr()
  //  );
   // if (status != CL_SUCCESS){
    //    std::cerr << "failed to clEnqueueMarkerWithWaitList "<<errorMsg(status)<< std::endl;
    //    std::exit(1);
  //  }
 //   inc(bits, arg);
}

void CudaBackend::CudaQueue::computeStart(){
    wait(); // should be no-op
    release(); // also ;
    marker(StartComputeBits);
}



void CudaBackend::CudaQueue::computeEnd(){
    marker(EndComputeBits);
}

void CudaBackend::CudaQueue::inc(int bits){
    if (eventc+1 >= eventMax){
        std::cerr << "CudaBackend::CudaQueue event list overflowed!!" << std::endl;
    }else{
        eventInfoBits[eventc]=bits;
    }
    eventc++;
}
void CudaBackend::CudaQueue::inc(int bits, const char *arg){
    if (eventc+1 >= eventMax){
        std::cerr << "CudaBackend::CudaQueue event list overflowed!!" << std::endl;
    }else{
        eventInfoBits[eventc]=bits|HasConstCharPtrArgBits;
        eventInfoConstCharPtrArgs[eventc]=arg;
    }
    eventc++;
}
void CudaBackend::CudaQueue::inc(int bits, int arg){
    if (eventc+1 >= eventMax){
        std::cerr << "CudaBackend::CudaQueue event list overflowed!!" << std::endl;
    }else{
        eventInfoBits[eventc]=bits|arg|hasIntArgBits;
    }
    eventc++;
}

void CudaBackend::CudaQueue::markAsEndComputeAndInc(){
    inc(EndComputeBits);
}
void CudaBackend::CudaQueue::markAsStartComputeAndInc(){
    inc(StartComputeBits);
}
void CudaBackend::CudaQueue::markAsNDRangeAndInc(){
    inc(NDRangeBits);
}
void CudaBackend::CudaQueue::markAsCopyToDeviceAndInc(int argn){
    inc(CopyToDeviceBits, argn);
}
void CudaBackend::CudaQueue::markAsCopyFromDeviceAndInc(int argn){
    inc(CopyFromDeviceBits, argn);
}
void CudaBackend::CudaQueue::markAsEnterKernelDispatchAndInc(){
    inc(EnterKernelDispatchBits);
}
void CudaBackend::CudaQueue::markAsLeaveKernelDispatchAndInc(){
    inc(LeaveKernelDispatchBits);
}

void CudaBackend::CudaQueue::release(){
   // cl_int status = CL_SUCCESS;
  //  for (int i = 0; i < eventc; i++) {
   //     status = clReleaseEvent(events[i]);
    //    if (status != CL_SUCCESS) {
      //      std::cerr << CudaBackend::errorMsg(status) << std::endl;
      //      exit(1);
   //     }
   // }//
 //   eventc = 0;
}

CudaBackend::CudaQueue::~CudaQueue(){
   // clReleaseCommandQueue(command_queue);
   // delete []events;
   auto status = cudaStreamDestroy(cudaStream);
    if (::cudaSuccess != status) {
        std::cerr << "cudaStreamDestroy() CUDA error = " << status
                  <<" " << cudaGetErrorString(static_cast<cudaError_t>(status))
                  <<" " << __FILE__ << " line " << __LINE__ << std::endl;
        exit(-1);
    }
}
