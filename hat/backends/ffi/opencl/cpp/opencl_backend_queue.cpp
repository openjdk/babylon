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
While based on OpenCL's event list, I think we need to use a MOD eventMax queue.

So
*/
 OpenCLBackend::OpenCLQueue::OpenCLQueue(OpenCLBackend *openclBackend)
    :openclBackend(openclBackend), eventMax(10000), events(new cl_event[eventMax]), eventInfoBits(new int[eventMax]), eventc(0){
 }

 cl_event *OpenCLBackend::OpenCLQueue::eventListPtr(){
    return (eventc == 0) ? nullptr : events;
 }
 cl_event *OpenCLBackend::OpenCLQueue::nextEventPtr(){
    return &events[eventc];
 }

void OpenCLBackend::OpenCLQueue::showEvents(int width) {
    const int  SAMPLE_TYPES=4;
    cl_ulong *samples = new cl_ulong[SAMPLE_TYPES * eventc]; // queued, submit, start, end, complete
    int sample = 0;
    cl_ulong min;
    cl_ulong max;
    cl_profiling_info profiling_info_arr[]={CL_PROFILING_COMMAND_QUEUED,CL_PROFILING_COMMAND_SUBMIT,CL_PROFILING_COMMAND_START,CL_PROFILING_COMMAND_END} ;
    const char* profiling_info_name_arr[]={"CL_PROFILING_COMMAND_QUEUED","CL_PROFILING_COMMAND_SUBMIT","CL_PROFILING_COMMAND_START","CL_PROFILING_COMMAND_END" } ;

    for (int event = 0; event < eventc; event++) {
        for (int type = 0; type < SAMPLE_TYPES; type++) {
            if ((clGetEventProfilingInfo(events[event], profiling_info_arr[type], sizeof(samples[sample]), &samples[sample], NULL)) !=
                CL_SUCCESS) {
                std::cerr << "failed to get profile info " << profiling_info_name_arr[type] << std::endl;
            }
            if (sample == 0) {
                if (type == 0){
                   min = max = samples[sample];
                }
            } else {
                if (samples[sample] < min) {
                    min = samples[sample];
                }
                if (samples[sample] > max) {
                    max = samples[sample];
                }
            }
            sample++;
        }
    }
    sample = 0;
    int range = (max - min);
    int scale = range / width;  // range per char
    std::cout << "Range: " <<min<< "-" <<max<< "("<< range << "ns)"
        <<  "  (" << scale << "ns) per char"
        << " +:submitted, .:started, =:end  "<< std::endl;

    for (int event = 0; event < eventc; event++) {
      /*  cl_command_type command_type;
        clGetEventInfo(events[event],CL_EVENT_COMMAND_TYPE,sizeof(command_type), &command_type, nullptr);
        switch (command_type){
          case CL_COMMAND_MARKER:         std::cout <<   "marker "; break;
          case CL_COMMAND_USER:           std::cout <<   "  user "; break;
          case CL_COMMAND_NDRANGE_KERNEL: std::cout <<   "kernel "; break;
          case CL_COMMAND_READ_BUFFER:    std::cout <<   "  read "; break;
          case CL_COMMAND_WRITE_BUFFER:   std::cout <<   " write "; break;
          default: std::cout <<                          " other "; break;
        } */
        int bits = eventInfoBits[event];
        if ((bits&CopyToDeviceBits)==CopyToDeviceBits){
           std::cout <<   "  write ";
        }
        if ((bits&CopyFromDeviceBits)==CopyFromDeviceBits){
           std::cout <<   "   read ";
        }
        if ((bits&StartComputeBits)==StartComputeBits){
           std::cout <<   "  start ";
        }
        if ((bits&EndComputeBits)==EndComputeBits){
           std::cout <<   "    end ";
        }
        if ((bits&NDRangeBits)==NDRangeBits){
           std::cout <<   " kernel ";
        }
        if ((bits&EnterKernelDispatchBits)==EnterKernelDispatchBits){
           std::cout <<   "  enter ";
        }
        if ((bits&LeaveKernelDispatchBits)==LeaveKernelDispatchBits){
           std::cout <<   "  leave ";
        }


        cl_ulong queue = (samples[sample++] - min) / scale;
        cl_ulong submit = (samples[sample++] - min) / scale;
        cl_ulong start = (samples[sample++] - min) / scale;
        cl_ulong end = (samples[sample++] - min) / scale;

        std::cout << std::setw(20)<< (queue-end) << "(ns) ";
        for (int c = 0; c < width; c++) {
            char ch = ' ';
            if (c >= queue && c<=submit) {
                ch = '+';
            }else if (c>submit && c<start){
                ch = '.';
            }else if (c>=start && c<end){
                ch = '=';
            }
            std::cout << ch;
        }
        std::cout << std::endl;
    }
    delete[] samples;
}
void OpenCLBackend::OpenCLQueue::wait(){
    if (eventc > 0){
       cl_int status = clWaitForEvents(eventc, events);
       if (status != CL_SUCCESS) {
          std::cerr << "failed clWaitForEvents" << OpenCLBackend::errorMsg(status) << std::endl;
          exit(1);
       }
    }
 }
 void clCallback(void *){
      std::cerr<<"start of compute"<<std::endl;
 }
  void OpenCLBackend::OpenCLQueue::marker(int bits){
  cl_int status = clEnqueueMarkerWithWaitList(
      command_queue,
      this->eventc, this->eventListPtr(),this->nextEventPtr()
      );
        if (status != CL_SUCCESS){
             std::cerr << "failed to clEnqueueMarkerWithWaitList "<<errorMsg(status)<< std::endl;
             std::exit(1);
         }
      inc(bits);
  }

 void OpenCLBackend::OpenCLQueue::computeStart(){
   wait(); // should be no-op
   release(); // also ;
   marker(StartComputeBits);
 }



 void OpenCLBackend::OpenCLQueue::computeEnd(){
   marker(EndComputeBits);
 }

 void OpenCLBackend::OpenCLQueue::inc(int bits){
    if (eventc+1 >= eventMax){
       std::cerr << "OpenCLBackend::OpenCLQueue event list overflowed!!" << std::endl;
    }else{
        eventInfoBits[eventc]=bits;
    }
    eventc++;
 }

 void OpenCLBackend::OpenCLQueue::markAsEndComputeAndInc(){
     inc(EndComputeBits);
 }
 void OpenCLBackend::OpenCLQueue::markAsStartComputeAndInc(){
     inc(StartComputeBits);
 }
 void OpenCLBackend::OpenCLQueue::markAsNDRangeAndInc(){
     inc(NDRangeBits);
 }
 void OpenCLBackend::OpenCLQueue::markAsCopyToDeviceAndInc(){
     inc(CopyToDeviceBits);
 }
 void OpenCLBackend::OpenCLQueue::markAsCopyFromDeviceAndInc(){
     inc(CopyFromDeviceBits);
 }
 void OpenCLBackend::OpenCLQueue::markAsEnterKernelDispatchAndInc(){
     inc(EnterKernelDispatchBits);
 }
 void OpenCLBackend::OpenCLQueue::markAsLeaveKernelDispatchAndInc(){
     inc(LeaveKernelDispatchBits);
 }

 void OpenCLBackend::OpenCLQueue::release(){
     cl_int status = CL_SUCCESS;
     for (int i = 0; i < eventc; i++) {
         status = clReleaseEvent(events[i]);
         if (status != CL_SUCCESS) {
             std::cerr << OpenCLBackend::errorMsg(status) << std::endl;
             exit(1);
         }
     }
     eventc = 0;
 }

 OpenCLBackend::OpenCLQueue::~OpenCLQueue(){
     clReleaseCommandQueue(command_queue);
     delete []events;
     delete []eventInfoBits;
 }