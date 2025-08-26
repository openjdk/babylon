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
OpenCLBackend::OpenCLQueue::OpenCLQueue(Backend *backend)
    : ProfilableQueue(backend, 10000),
      command_queue(),
      events(new cl_event[eventMax]) {
}

cl_event *OpenCLBackend::OpenCLQueue::eventListPtr() const {
    return (eventc == 0) ? nullptr : events;
}

cl_event *OpenCLBackend::OpenCLQueue::nextEventPtr() const {
    return &events[eventc];
}

void OpenCLBackend::OpenCLQueue::showEvents(const int width) {
    constexpr int SAMPLE_TYPES = 4;
    auto *samples = new cl_ulong[SAMPLE_TYPES * eventc]; // queued, submit, start, end, complete
    int sample = 0;
    cl_ulong min = CL_LONG_MAX;
    cl_ulong max = CL_LONG_MIN;

    for (int event = 0; event < eventc; event++) {
        for (int type = 0; type < SAMPLE_TYPES; type++) {
            cl_profiling_info profiling_info_arr[] = {
                CL_PROFILING_COMMAND_QUEUED,CL_PROFILING_COMMAND_SUBMIT,CL_PROFILING_COMMAND_START,
                CL_PROFILING_COMMAND_END
            };
            if ((clGetEventProfilingInfo(events[event], profiling_info_arr[type], sizeof(samples[sample]),
                                         &samples[sample], NULL)) !=
                CL_SUCCESS) {
                const char *profiling_info_name_arr[] = {
                    "CL_PROFILING_COMMAND_QUEUED", "CL_PROFILING_COMMAND_SUBMIT", "CL_PROFILING_COMMAND_START",
                    "CL_PROFILING_COMMAND_END"
                };
                std::cerr << "failed to get profile info " << profiling_info_name_arr[type] << std::endl;
            }
            if (sample == 0) {
                if (type == 0) {
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
    const cl_ulong range = (max - min);
    const cl_ulong scale = range / width; // range per char
    std::cout << "Range: " << min << "-" << max << "(" << range << "ns)"
            << "  (" << scale << "ns) per char"
            << " +:submitted, .:started, =:end  " << std::endl;

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
        const int bits = eventInfoBits[event];
        if ((bits & CopyToDeviceBits) == CopyToDeviceBits) {
            std::cout << "  write " << (bits & 0xffff) << " ";
        }
        if ((bits & CopyFromDeviceBits) == CopyFromDeviceBits) {
            std::cout << "   read " << (bits & 0xffff) << " ";
        }
        if ((bits & StartComputeBits) == StartComputeBits) {
            std::cout << "  start    ";
        }
        if ((bits & EndComputeBits) == EndComputeBits) {
            std::cout << "    end    ";
        }
        if ((bits & NDRangeBits) == NDRangeBits) {
            std::cout << " kernel    ";
        }
        if ((bits & EnterKernelDispatchBits) == EnterKernelDispatchBits) {
            if ((bits & HasConstCharPtrArgBits) == HasConstCharPtrArgBits) {
                std::cout << eventInfoConstCharPtrArgs[event] << std::endl;
            }
            std::cout << "  enter{   ";
        }
        if ((bits & LeaveKernelDispatchBits) == LeaveKernelDispatchBits) {
            // std::cout <<   "  leave    ";
            if ((bits & HasConstCharPtrArgBits) == HasConstCharPtrArgBits) {
                std::cout << eventInfoConstCharPtrArgs[event] << std::endl;
            }
            std::cout << " }leave    ";
        }


        const cl_ulong queue = (samples[sample++] - min) / scale;
        const cl_ulong submit = (samples[sample++] - min) / scale;
        const cl_ulong start = (samples[sample++] - min) / scale;
        const cl_ulong end = (samples[sample++] - min) / scale;

        std::cout << std::setw(20) << (queue - end) << "(ns) ";
        for (int c = 0; c < width; c++) {
            char ch = ' ';
            if (c >= queue && c <= submit) {
                ch = '+';
            } else if (c > submit && c < start) {
                ch = '.';
            } else if (c >= start && c < end) {
                ch = '=';
            }
            std::cout << ch;
        }
        std::cout << std::endl;
    }
    delete[] samples;
}

void OpenCLBackend::OpenCLQueue::wait() {
    if (eventc > 0) {
        OPENCL_CHECK(clWaitForEvents(eventc, events), "clWaitForEvents");
    }
}

void OpenCLBackend::OpenCLQueue::marker(int bits) {
    cl_int status = clEnqueueMarkerWithWaitList(
        command_queue,
        this->eventc,
        this->eventListPtr(),
        this->nextEventPtr()
    );
    if (status != CL_SUCCESS) {
        std::cerr << "failed to clEnqueueMarkerWithWaitList " << errorMsg(status) << std::endl;
        std::exit(1);
    }
    inc(bits);
}

void OpenCLBackend::OpenCLQueue::marker(int bits, const char *arg) {
    OPENCL_CHECK(clEnqueueMarkerWithWaitList(
                     command_queue,
                     this->eventc,
                     this->eventListPtr(),
                     this->nextEventPtr()),
                 "clEnqueueMarkerWithWaitList");

    inc(bits, arg);
}

void OpenCLBackend::OpenCLQueue::computeStart() {
    wait(); // should be no-op
    release(); // also ;
    marker(StartComputeBits);
}

void OpenCLBackend::OpenCLQueue::computeEnd() {
    marker(EndComputeBits);
}

void OpenCLBackend::OpenCLQueue::inc(const int bits) {
    if (eventc + 1 >= eventMax) {
        std::cerr << "OpenCLBackend::OpenCLQueue event list overflowed!!" << std::endl;
    } else {
        eventInfoBits[eventc] = bits;
    }
    eventc++;
}

void OpenCLBackend::OpenCLQueue::inc(const int bits, const char *arg) {
    if (eventc + 1 >= eventMax) {
        std::cerr << "OpenCLBackend::OpenCLQueue event list overflowed!!" << std::endl;
    } else {
        eventInfoBits[eventc] = bits | HasConstCharPtrArgBits;
        eventInfoConstCharPtrArgs[eventc] = arg;
    }
    eventc++;
}

void OpenCLBackend::OpenCLQueue::markAsEndComputeAndInc() {
    inc(EndComputeBits);
}

void OpenCLBackend::OpenCLQueue::markAsStartComputeAndInc() {
    inc(StartComputeBits);
}

void OpenCLBackend::OpenCLQueue::markAsEnterKernelDispatchAndInc() {
    inc(EnterKernelDispatchBits);
}

void OpenCLBackend::OpenCLQueue::markAsLeaveKernelDispatchAndInc() {
    inc(LeaveKernelDispatchBits);
}

void OpenCLBackend::OpenCLQueue::release() {
    // TODO: possible check ALL events before return from the macro
    for (int i = 0; i < eventc; i++) {
        OPENCL_CHECK(clReleaseEvent(events[i]), "clReleaseEvent");
    }
    eventc = 0;
}

OpenCLBackend::OpenCLQueue::~OpenCLQueue() {
    OPENCL_CHECK(clReleaseCommandQueue(command_queue), "clReleaseCommandQueue");
    delete []events;
}

void OpenCLBackend::OpenCLQueue::dispatch(KernelContext *kernelContext, Backend::CompilationUnit::Kernel *kernel) {
    size_t numDimensions = kernelContext->dimensions;

    size_t global_work_size[] {
        static_cast<size_t>(kernelContext->maxX),  // to be replaced with gsx
        static_cast<size_t>(kernelContext->maxY),  // to be replaced with gsy
        static_cast<size_t>(kernelContext->maxZ)   // to be replaced with gsz
    };

    size_t local_work_size[] = {
        static_cast<size_t>(kernelContext->lsx),
        static_cast<size_t>(kernelContext->lsy),
        static_cast<size_t>(kernelContext->lsz),
    };

    if (backend->config->info) {
        std::cout << "[INFO] OpenCLBackend::OpenCLQueue::dispatch" << std::endl;
        std::cout << "[INFO] numDimensions: " << numDimensions << std::endl;
        std::cout << "[INFO] GLOBAL [" << global_work_size[0] << "," << global_work_size[1] << "," << global_work_size[2] << "]" << std::endl;
        if (kernelContext->lsx > 0) {
            std::cout << "[INFO] LOCAL  [" << local_work_size[0] << "," << local_work_size[1] << "," << local_work_size[2] << "]" << std::endl;
        } else {
            std::cout << "[INFO] LOCAL  [ nullptr ] // The driver will setup a default value" << std::endl;
        }
    }

    cl_event kernelEvent = nullptr;
    const cl_int status = clEnqueueNDRangeKernel(
        command_queue,
        dynamic_cast<OpenCLProgram::OpenCLKernel *>(kernel)->kernel,
        numDimensions,
        nullptr,
        global_work_size,
        kernelContext->lsx > 0 ? local_work_size : nullptr,
        eventc,
        eventListPtr(),
        nextEventPtr());
        //&kernelEvent);

    // clWaitForEvents(1, &kernelEvent);
    // cl_ulong start_time, end_time;
    // clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start_time, NULL);
    // clGetEventProfilingInfo(kernelEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end_time, NULL);
    //
    // std::cout << "[INFO] Kernel Time: " << (end_time - start_time) << std::endl;
    // events[eventc] = kernelEvent;

    inc(NDRangeBits);
    // markAsNDRangeAndInc();

    OPENCL_CHECK(status, "clEnqueueNDRangeKernel");
    if (backend->config->trace | backend->config->traceEnqueues) {
        std::cout << "enqueued kernel dispatch \"" << kernel->name << "\" globalSize=" << kernelContext->maxX <<
                std::endl;
    }
}

void OpenCLBackend::OpenCLQueue::copyToDevice(Buffer *buffer) {
    auto openclBuffer = dynamic_cast<OpenCLBuffer *>(buffer);
    cl_int status = clEnqueueWriteBuffer(
        command_queue,
        openclBuffer->clMem,
        CL_FALSE,
        0,
        buffer->bufferState->length,
        buffer->bufferState->ptr,
        eventc,
        eventListPtr(),
        nextEventPtr()
    );

    OPENCL_CHECK(status, "clEnqueueWriteBuffer");

    inc(CopyToDeviceBits);
    //  markAsCopyToDeviceAndInc();
}

void OpenCLBackend::OpenCLQueue::copyFromDevice(Buffer *buffer) {
    auto openclBuffer = dynamic_cast<OpenCLBuffer *>(buffer);
    cl_int status = clEnqueueReadBuffer(
        command_queue,
        openclBuffer->clMem,
        CL_FALSE,
        0,
        buffer->bufferState->length,
        buffer->bufferState->ptr,
        eventc,
        eventListPtr(),
        nextEventPtr()
    );
    OPENCL_CHECK(status, "clEnqueueReadBuffer");
    inc(CopyFromDeviceBits);
    //markAsCopyFromDeviceAndInc();
}
