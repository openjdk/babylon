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
package wrap.clwrap;

import hat.buffer.Buffer;
import hat.ifacemapper.BufferState;
import opencl.opencl_h;
import wrap.ArenaHolder;
import wrap.Wrap;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.List;

import static java.lang.foreign.MemorySegment.NULL;
import static opencl.opencl_h.CL_DEVICE_BUILT_IN_KERNELS;
import static opencl.opencl_h.CL_DEVICE_MAX_COMPUTE_UNITS;
import static opencl.opencl_h.CL_DEVICE_NAME;
import static opencl.opencl_h.CL_DEVICE_TYPE_ALL;
import static opencl.opencl_h.CL_DEVICE_VENDOR;
import static opencl.opencl_h.CL_MEM_READ_WRITE;
import static opencl.opencl_h.CL_MEM_USE_HOST_PTR;
import static opencl.opencl_h.CL_PROGRAM_BUILD_LOG;
import static opencl.opencl_h.CL_QUEUE_PROFILING_ENABLE;
import static opencl.opencl_h.CL_SUCCESS;

// https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
public class CLPlatform implements ArenaHolder {
    public static List<CLPlatform> platforms(Arena arena) {
        var arenaWrapper = ArenaHolder.wrap(arena);
        List<CLPlatform> platforms = new ArrayList<>();
        var platformc = arenaWrapper.intPtr(0);
        if ((opencl_h.clGetPlatformIDs(0, NULL, platformc.ptr())) != CL_SUCCESS()) {
            System.out.println("Failed to get opencl platforms");
        } else {
            var platformIds = arenaWrapper.ptrArr(platformc.get());
            if ((opencl_h.clGetPlatformIDs(platformc.get(), platformIds.ptr(), NULL)) != CL_SUCCESS()) {
                System.out.println("Failed getting platform ids");
            } else {
                for (int i = 0; i < platformc.get(); i++) {
                    platforms.add(new CLPlatform(arena, platformIds.get(i)));
                }
            }
        }
        return platforms;
    }

    public static class CLDevice implements ArenaHolder {
        final CLPlatform platform;
        final MemorySegment deviceId;

        @Override
        public Arena arena() {
            return platform.arena();
        }

        int intDeviceInfo(int query) {
            var value = intPtr(0);
            if ((opencl_h.clGetDeviceInfo(deviceId, query, value.sizeof(), value.ptr(), NULL)) != CL_SUCCESS()) {
                throw new RuntimeException("Failed to get query " + query);
            }
            return value.get();
        }

        String strDeviceInfo(int query) {
            var value = cstr(2048);
            if ((opencl_h.clGetDeviceInfo(deviceId, query, value.len(), value.ptr(), NULL)) != CL_SUCCESS()) {
                throw new RuntimeException("Failed to get query " + query);
            }
            return value.get();
        }

        public int computeUnits() {
            return intDeviceInfo(CL_DEVICE_MAX_COMPUTE_UNITS());
        }

        public String deviceName() {
            return strDeviceInfo(CL_DEVICE_NAME());
        }

        public String deviceVendor() {
            return strDeviceInfo(CL_DEVICE_VENDOR());
        }

        public String builtInKernels() {
            return strDeviceInfo(CL_DEVICE_BUILT_IN_KERNELS());
        }

        CLDevice(CLPlatform platform, MemorySegment deviceId) {
            this.platform = platform;
            this.deviceId = deviceId;
        }

        public static class CLContext implements ArenaHolder {
            CLDevice device;
            MemorySegment context;
            MemorySegment queue;

            @Override
            public Arena arena() {
                return device.arena();
            }

            CLContext(CLDevice device, MemorySegment context) {
                this.device = device;
                this.context = context;
                var status = device.platform.status;

                var queue_props = CL_QUEUE_PROFILING_ENABLE();
                if ((this.queue = opencl_h.clCreateCommandQueue(context, device.deviceId, queue_props, status.ptr())) == NULL) {
                    opencl_h.clReleaseContext(context);
                } else {
                    if (!status.isOK()) {
                        opencl_h.clReleaseContext(context);
                    }
                }
            }

            static public class CLProgram implements ArenaHolder {
                CLContext context;
                String source;
                MemorySegment program;
                String log;

                @Override
                public Arena arena() {
                    return context.arena();
                }

                CLProgram(CLContext context, String source) {
                    this.context = context;
                    this.source = source;
                    var sourceList = ptrArr(source);
                    var status = context.device.platform.status;
                    if ((program = opencl_h.clCreateProgramWithSource(context.context, 1, sourceList.ptr(), NULL, status.ptr())) == NULL) {
                        if (!status.isOK()) {
                            throw new RuntimeException("failed to createProgram " + status.get());
                        }
                        throw new RuntimeException("failed to createProgram");
                    } else {
                        var deviceIdList = ptrArr(context.device.deviceId);
                        if ((status.set(opencl_h.clBuildProgram(program, 1, deviceIdList.ptr(), NULL, NULL, NULL))) != CL_SUCCESS()) {
                            System.err.println("failed to build program " + status);
                            System.err.println("Source "+source);
                           // System.exit(1);
                        }
                        var logLen = longPtr(1L);
                        if ((status.set(opencl_h.clGetProgramBuildInfo(program, context.device.deviceId, CL_PROGRAM_BUILD_LOG(), 0, NULL, logLen.ptr()))) != CL_SUCCESS()) {
                            System.err.println("failed to get log build " + status.get());
                            System.exit(1);
                        } else {
                            var logPtr = cstr(1 + logLen.get());
                            if ((status.set(opencl_h.clGetProgramBuildInfo(program, context.device.deviceId, opencl_h.CL_PROGRAM_BUILD_LOG(), logLen.get(), logPtr.ptr(), NULL))) != opencl_h.CL_SUCCESS()) {
                                System.out.println("clGetBuildInfo (getting log) failed ");
                            } else {
                                log = logPtr.get();
                                System.out.println("log\n" +log);
                            }
                        }
                    }
                }

                public static class CLKernel implements ArenaHolder {
                    CLProgram program;
                    MemorySegment kernel;
                    String kernelName;

                    @Override
                    public Arena arena() {
                        return program.arena();
                    }

                    public CLKernel(CLProgram program, String kernelName) {
                        this.program = program;
                        this.kernelName = kernelName;
                        var kernelNameCStr = this.cstr(kernelName);
                        var status = program.context.device.platform.status;
                        kernel = opencl_h.clCreateKernel(program.program, kernelNameCStr.ptr(), status.ptr());
                        if (!status.isOK()) {
                            System.out.println("failed to create kernel '"+kernelName+"'" + status);
                        }
                    }


                    public void run(CLWrapComputeContext clWrapComputeContext, int range, Object... args) {
                        var status = CLStatusPtr.of(arena());
                        for (int i = 0; i < args.length; i++) {
                            if (args[i] instanceof CLWrapComputeContext.MemorySegmentState memorySegmentState) {
                                if (memorySegmentState.clMemPtr == null) {
                                    memorySegmentState.clMemPtr = CLWrapComputeContext.ClMemPtr.of(arena(), opencl_h.clCreateBuffer(program.context.context,
                                            CL_MEM_USE_HOST_PTR() | CL_MEM_READ_WRITE(),
                                            memorySegmentState.memorySegment.byteSize(),
                                            memorySegmentState.memorySegment,
                                            status.ptr()));
                                    if (!status.isOK()) {
                                        throw new RuntimeException("failed to create memory buffer for arg["+i+" " + status.get());
                                    }
                                }
                                if (memorySegmentState.copyToDevice) {
                                    status.set(opencl_h.clEnqueueWriteBuffer(program.context.queue,
                                            memorySegmentState.clMemPtr.get(),
                                            clWrapComputeContext.blockInt(),
                                            0,
                                            memorySegmentState.memorySegment.byteSize(),
                                            memorySegmentState.memorySegment,
                                            clWrapComputeContext.eventc(),
                                            clWrapComputeContext.eventsPtr(),
                                            clWrapComputeContext.nextEventPtrSlot()
                                    ));
                                    if (!status.isOK()) {
                                        System.err.println("failed to enqueue write for arg["+i+" " + status);
                                        System.exit(1);
                                    }
                                }

                                status.set(opencl_h.clSetKernelArg(kernel, i, memorySegmentState.clMemPtr.sizeof(), memorySegmentState.clMemPtr.ptr()));
                                if (!status.isOK()) {
                                    System.err.println("failed to set arg["+i+" " + status);
                                    System.exit(1);
                                }
                            } else if (args[i] instanceof Buffer buffer) {
                                //  System.out.println("Arg "+i+" is a buffer so checking if we need to write");
                                BufferState bufferState = BufferState.of(buffer);

                                //System.out.println("Before possible write"+ bufferState);
                                MemorySegment memorySegment = Buffer.getMemorySegment(buffer);

                                CLWrapComputeContext.ClMemPtr clmem = clWrapComputeContext.clMemMap.computeIfAbsent(memorySegment, k ->
                                        CLWrapComputeContext.ClMemPtr.of(arena(), opencl_h.clCreateBuffer(program.context.context,
                                                CL_MEM_USE_HOST_PTR() | CL_MEM_READ_WRITE(),
                                                memorySegment.byteSize(),
                                                memorySegment,
                                                status.ptr()))
                                );
                                if (bufferState.getState()==BufferState.HOST_OWNED) {

                                    //System.out.println("arg " + args[i] + " isHostDirty copying in");
                                    status.set(opencl_h.clEnqueueWriteBuffer(program.context.queue,
                                            clmem.get(),
                                            clWrapComputeContext.blockInt(),
                                            0,
                                            memorySegment.byteSize(),
                                            memorySegment,
                                            clWrapComputeContext.eventc(),
                                            clWrapComputeContext.eventsPtr(),
                                            clWrapComputeContext.nextEventPtrSlot()
                                    ));
                                    if (!status.isOK()) {
                                        System.err.println("failed to enqueue write for arg["+i+" " + status);
                                        System.exit(1);
                                    }
                                } else {

                                    //  System.out.println("arg "+args[i]+" is not HostDirty not copying in");
                                }
                                //     System.out.println("After possible write "+ bufferState);
                                status.set(opencl_h.clSetKernelArg(kernel, i, clmem.sizeof(), clmem.ptr()));
                                if (!status.isOK()) {
                                    System.err.println("failed to set arg["+i+"]" + status);
                                    System.exit(1);
                                }

                            } else {
                                Wrap.Ptr ptr = switch (args[i]) {
                                    case Integer intArg -> intPtr(intArg);
                                    case Float floatArg -> floatPtr(floatArg);
                                    case Double doubleArg -> doublePtr(doubleArg);
                                    case Long longArg -> longPtr(longArg);
                                    case Short shortArg -> shortPtr(shortArg);
                                    default -> throw new IllegalStateException("Unexpected value: " + args[i]);
                                };
                                status.set(opencl_h.clSetKernelArg(kernel, i, ptr.sizeof(), ptr.ptr()));
                                if (!status.isOK()) {
                                    System.err.println("failed to set arg["+i+"] " + status);

                                    System.exit(1);
                                }

                            }
                        }

                        // We need to store x,y,z sizes so this is a kind of int3
                        var globalSize = this.ofInts(range, 0, 0);
                        status.set(opencl_h.clEnqueueNDRangeKernel(
                                        program.context.queue,
                                        kernel,
                                        1, // this must match the # of dims we are using in this case 1 of 3
                                        NULL,
                                        globalSize.ptr(),
                                        NULL,
                                        clWrapComputeContext.eventc(),
                                        clWrapComputeContext.eventsPtr(),
                                        clWrapComputeContext.nextEventPtrSlot()
                                )
                        );
                        if (!status.isOK()) {
                            System.out.println("failed to enqueue NDRange " + status);
                        }

                        if (clWrapComputeContext.alwaysBlock) {
                            opencl_h.clFlush(program.context.queue);
                        }

                        for (int i = 0; i < args.length; i++) {
                            if (args[i] instanceof CLWrapComputeContext.MemorySegmentState memorySegmentState) {
                                if (memorySegmentState.copyFromDevice) {
                                    status.set(opencl_h.clEnqueueReadBuffer(program.context.queue,
                                            memorySegmentState.clMemPtr.get(),
                                            clWrapComputeContext.blockInt(),
                                            0,
                                            memorySegmentState.memorySegment.byteSize(),
                                            memorySegmentState.memorySegment,
                                            clWrapComputeContext.eventc(),
                                            clWrapComputeContext.eventsPtr(),
                                            clWrapComputeContext.nextEventPtrSlot()
                                    ));
                                    if (!status.isOK()) {
                                        System.out.println("failed to enqueue read " + status);
                                    }
                                }
                            } else if (args[i] instanceof Buffer buffer) {
                                //   System.out.println("Arg "+i+" is a buffer so checking if we need to read");
                                BufferState bufferState = BufferState.of(buffer);
                                MemorySegment memorySegment = Buffer.getMemorySegment(buffer);
                                CLWrapComputeContext.ClMemPtr clmem = clWrapComputeContext.clMemMap.get(memorySegment);
                                // System.out.println("Before possible read "+ bufferState);
                                if (bufferState.getState() == BufferState.HOST_OWNED) {
                                  //  System.out.println("arg " + args[i] + " isDeviceDirty copying out");
                                    status.set(opencl_h.clEnqueueReadBuffer(program.context.queue,
                                            clmem.get(),
                                            clWrapComputeContext.blockInt(),
                                            0,
                                            memorySegment.byteSize(),
                                            memorySegment,
                                            clWrapComputeContext.eventc(),
                                            clWrapComputeContext.eventsPtr(),
                                            clWrapComputeContext.nextEventPtrSlot()
                                    ));
                                    if (!status.isOK()) {
                                        System.out.println("failed to enqueue read " + status);
                                    }
                                } else {
                                    //   System.out.println("arg "+args[i]+" isnot DeviceDirty not copying out");
                                }

                            }
                        }
                        // if (!computeContext.alwaysBlock) {
                        clWrapComputeContext.waitForEvents();
                        //  }
                    }
                }

                public CLKernel getKernel(String kernelName) {
                    return new CLKernel(this, kernelName);
                }
            }

            public CLProgram buildProgram(String source) {
                var program = new CLProgram(this, source);
                return program;
            }
        }

        public CLContext createContext() {
            var status = platform.status;
            MemorySegment context;
            var deviceIds = ptrArr(this.deviceId);
            if ((context = opencl_h.clCreateContext(NULL, 1, deviceIds.ptr(), NULL, NULL, status.ptr())) == NULL) {
                System.out.println("Failed to get context  ");
                return null;
            } else {
                if (!status.isOK()) {
                    System.out.println("failed to get context  " + status);
                }
                return new CLContext(this, context);
            }
        }
    }

    int intPlatformInfo(int query) {
        var value = intPtr(0);
        if ((opencl_h.clGetPlatformInfo(platformId, query, value.sizeof(), value.ptr(), NULL)) != opencl_h.CL_SUCCESS()) {
            throw new RuntimeException("Failed to get query " + query);
        }
        return value.get();
    }

    String strPlatformInfo(int query) {

        var value = cstr(2048);
        int status;
        if ((status = opencl_h.clGetPlatformInfo(platformId, query, value.len(), value.ptr(), NULL)) != opencl_h.CL_SUCCESS()) {
            throw new RuntimeException("Failed to get query " + query);
        }
        return value.get();
    }

    private Arena secretarena;
    MemorySegment platformId;
    public List<CLDevice> devices = new ArrayList<>();
    final CLStatusPtr status;

    public String platformName() {
        return strPlatformInfo(opencl_h.CL_PLATFORM_NAME());
    }

    String vendorName() {
        return strPlatformInfo(opencl_h.CL_PLATFORM_VENDOR());
    }

    String version() {
        return strPlatformInfo(opencl_h.CL_PLATFORM_VERSION());
    }

    @Override
    public Arena arena() {
        return secretarena;
    }

    public CLPlatform(Arena arena, MemorySegment platformId) {
        this.secretarena = arena;
        this.platformId = platformId;
        this.status = CLStatusPtr.of(arena());
        var devicec = intPtr(0);
        if ((status.set(opencl_h.clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL(), 0, NULL, devicec.ptr()))) != opencl_h.CL_SUCCESS()) {
            System.err.println("Failed getting devicec for platform 0 ");
        } else {
            //  System.out.println("platform 0 has " + devicec + " device" + ((devicec > 1) ? "s" : ""));
            var deviceIdList = ptrArr(devicec.get());
            if ((status.set(opencl_h.clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL(), devicec.get(), deviceIdList.ptr(), devicec.ptr()))) != opencl_h.CL_SUCCESS()) {
                System.err.println("Failed getting deviceids  for platform 0 ");
            } else {
                // System.out.println("We have "+devicec+" device ids");
                for (int i = 0; i < devicec.get(); i++) {
                    devices.add(new CLDevice(this, deviceIdList.get(i)));
                }
            }
        }
    }
}
