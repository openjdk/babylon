package hat.backend.jextracted;
/*
 * Copyright (c) 2020, Oracle and/or its affiliates. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 *   - Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *
 *   - Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in the
 *     documentation and/or other materials provided with the distribution.
 *
 *   - Neither the name of Oracle nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
 * IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

import opencl.opencl_h;

import java.io.IOException;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.ArrayList;
import java.util.List;

//import static java.lang.foreign.ValueLayout.JAVA_INT;
import static opencl.opencl_h.CL_DEVICE_TYPE_ALL;
import static opencl.opencl_h.CL_MEM_READ_WRITE;
import static opencl.opencl_h.CL_MEM_USE_HOST_PTR;
import static opencl.opencl_h.CL_QUEUE_PROFILING_ENABLE;

public class CLWrap {
    public static MemorySegment NULL = MemorySegment.NULL;

    // https://streamhpc.com/blog/2013-04-28/opencl-error-codes/
    static class Platform {
        static class Device {
            final Platform platform;
            final MemorySegment deviceId;

            int intDeviceInfo(int query) {
                var value = 0;
                if ((opencl_h.clGetDeviceInfo(deviceId, query, opencl_h.C_INT.byteSize(), platform.intValuePtr, NULL)) != opencl_h.CL_SUCCESS()) {
                    System.out.println("Failed to get query " + query);
                } else {
                    value = platform.intValuePtr.get(opencl_h.C_INT, 0);
                }
                return value;
            }

            String strDeviceInfo(int query) {
                String value = null;
                if ((opencl_h.clGetDeviceInfo(deviceId, query, 2048, platform.byte2048ValuePtr, platform.intValuePtr)) != opencl_h.CL_SUCCESS()) {
                    System.out.println("Failed to get query " + query);
                } else {
                    int len = platform.intValuePtr.get(opencl_h.C_INT, 0);
                    byte[] bytes = platform.byte2048ValuePtr.toArray(ValueLayout.JAVA_BYTE);
                    value = new String(bytes).substring(0, len - 1);
                }
                return value;
            }

            int computeUnits() {
                return intDeviceInfo(opencl_h.CL_DEVICE_MAX_COMPUTE_UNITS());
            }

            String deviceName() {
                return strDeviceInfo(opencl_h.CL_DEVICE_NAME());
            }

            String builtInKernels() {
                return strDeviceInfo(opencl_h.CL_DEVICE_BUILT_IN_KERNELS());
            }

            Device(Platform platform, MemorySegment deviceId) {
                this.platform = platform;
                this.deviceId = deviceId;
            }

            public static class Context {
                Device device;
                MemorySegment context;
                MemorySegment queue;

                Context(Device device, MemorySegment context) {
                    this.device = device;
                    this.context = context;
                    var statusPtr = device.platform.openCL.arena.allocateFrom(opencl_h.C_INT, 1);

                    var queue_props = CL_QUEUE_PROFILING_ENABLE();
                    if ((this.queue = opencl_h.clCreateCommandQueue(context, device.deviceId, queue_props, statusPtr)) == NULL) {
                        int status = statusPtr.get(opencl_h.C_INT, 0);
                        opencl_h.clReleaseContext(context);
                        // delete[] platforms;
                        // delete[] device_ids;
                        return;
                    }

                }

                static public class Program {
                    Context context;
                    String source;
                    MemorySegment program;
                    String log;

                    Program(Context context, String source) {
                        this.context = context;
                        this.source = source;
                        MemorySegment sourcePtr = context.device.platform.openCL.arena.allocateFrom(source);
                        var sourcePtrPtr = context.device.platform.openCL.arena.allocateFrom(opencl_h.C_POINTER, sourcePtr);
                    //    sourcePtrPtr.set(opencl_h.C_POINTER, 0, sourcePtr);
                        var sourceLenPtr = context.device.platform.openCL.arena.allocateFrom(opencl_h.C_LONG,  source.length());
                    //    sourceLenPtr.set(opencl_h.C_LONG, 0, source.length());
                        var statusPtr = context.device.platform.openCL.arena.allocateFrom(opencl_h.C_INT, 0);
                        if ((program = opencl_h.clCreateProgramWithSource(context.context, 1, sourcePtrPtr, sourceLenPtr, statusPtr)) == NULL) {
                            int status = statusPtr.get(opencl_h.C_INT, 0);
                            if (status != opencl_h.CL_SUCCESS()) {
                                System.out.println("failed to createProgram " + status);
                            }
                            System.out.println("failed to createProgram");
                        } else {
                            int status = statusPtr.get(opencl_h.C_INT, 0);
                            if (status != opencl_h.CL_SUCCESS()) {
                                System.out.println("failed to create program " + status);
                            }
                            var deviceIdPtr = context.device.platform.openCL.arena.allocateFrom(opencl_h.C_POINTER, context.device.deviceId);
                          //  deviceIdPtr.set(opencl_h.C_POINTER, 0, context.device.deviceId);
                            if ((status = opencl_h.clBuildProgram(program, 1, deviceIdPtr, NULL, NULL, NULL)) != opencl_h.CL_SUCCESS()) {
                                System.out.println("failed to build" + status);
                                // dont return we may still be able to get log!
                            }

                            var logLenPtr = context.device.platform.openCL.arena.allocate(opencl_h.C_LONG, 1);

                            if ((status = opencl_h.clGetProgramBuildInfo(program, context.device.deviceId, opencl_h.CL_PROGRAM_BUILD_LOG(), 0, NULL, logLenPtr)) != opencl_h.CL_SUCCESS()) {
                                System.out.println("failed to get log build " + status);
                            } else {
                                long logLen = logLenPtr.get(opencl_h.C_LONG, 0);
                                var logPtr = context.device.platform.openCL.arena.allocate(opencl_h.C_CHAR, 1 + logLen);
                                if ((status = opencl_h.clGetProgramBuildInfo(program, context.device.deviceId, opencl_h.CL_PROGRAM_BUILD_LOG(), logLen, logPtr, logLenPtr)) != opencl_h.CL_SUCCESS()) {
                                    System.out.println("clGetBuildInfo (getting log) failed");
                                } else {
                                    byte[] bytes = logPtr.toArray(ValueLayout.JAVA_BYTE);
                                    log = new String(bytes).substring(0, (int) logLen);
                                }
                            }
                        }
                    }

                    public static class Kernel {
                        Program program;
                        MemorySegment kernel;
                        String kernelName;

                        public Kernel(Program program, String kernelName) {
                            this.program = program;
                            this.kernelName = kernelName;
                            var statusPtr = program.context.device.platform.openCL.arena.allocateFrom(opencl_h.C_INT, opencl_h.CL_SUCCESS());
                            MemorySegment kernelNamePtr = program.context.device.platform.openCL.arena.allocateFrom(kernelName);
                            kernel = opencl_h.clCreateKernel(program.program, kernelNamePtr, statusPtr);
                            int status = statusPtr.get(opencl_h.C_INT, 0);
                            if (status != opencl_h.CL_SUCCESS()) {
                                System.out.println("failed to create kernel " + status);
                            }
                        }

                        public void run(int range, Object... args) {
                            var bufPtr = program.context.device.platform.openCL.arena.allocate(opencl_h.cl_mem, args.length);
                            var statusPtr = program.context.device.platform.openCL.arena.allocateFrom(opencl_h.C_INT, opencl_h.CL_SUCCESS());
                            int status;
                            var eventMax = args.length * 4 + 1;
                            int eventc = 0;
                            var eventsPtr = program.context.device.platform.openCL.arena.allocate(opencl_h.cl_event, eventMax);
                            boolean block = false;// true;
                            for (int i = 0; i < args.length; i++) {
                                if (args[i] instanceof MemorySegment memorySegment) {
                                    MemorySegment clMem = opencl_h.clCreateBuffer(program.context.context,
                                            CL_MEM_USE_HOST_PTR() | CL_MEM_READ_WRITE(),
                                            memorySegment.byteSize(),
                                            memorySegment,
                                            statusPtr);
                                    status = statusPtr.get(opencl_h.C_INT, 0);
                                    if (status != opencl_h.CL_SUCCESS()) {
                                        System.out.println("failed to create memory buffer " + status);
                                    }
                                    bufPtr.set(opencl_h.cl_mem, i * opencl_h.cl_mem.byteSize(), clMem);
                                    status = opencl_h.clEnqueueWriteBuffer(program.context.queue,
                                            clMem,
                                            block ? opencl_h.CL_TRUE() : opencl_h.CL_FALSE(), //block?
                                            0,
                                            memorySegment.byteSize(),
                                            memorySegment,
                                            block ? 0 : eventc,
                                            block ? NULL : ((eventc == 0) ? NULL : eventsPtr),
                                            block ? NULL : eventsPtr.asSlice(eventc * opencl_h.cl_event.byteSize(), opencl_h.cl_event)
                                    );
                                    if (status != opencl_h.CL_SUCCESS()) {
                                        System.out.println("failed to enqueue write " + status);
                                    }
                                    if (!block) {
                                        eventc++;
                                    }
                                    var clMemPtr = program.context.device.platform.openCL.arena.allocateFrom(opencl_h.C_POINTER, clMem);

                                    status = opencl_h.clSetKernelArg(kernel, i, opencl_h.C_POINTER.byteSize(), clMemPtr);
                                    if (status != opencl_h.CL_SUCCESS()) {
                                        System.out.println("failed to set arg " + status);
                                    }
                                } else {
                                    bufPtr.set(opencl_h.cl_mem, i * opencl_h.cl_mem.byteSize(), NULL);
                                    switch (args[i]){
                                        case Integer intArg->{
                                            var intPtr = program.context.device.platform.openCL.arena.allocateFrom(opencl_h.C_INT, intArg);
                                            status = opencl_h.clSetKernelArg(kernel, i, opencl_h.C_INT.byteSize(), intPtr);
                                            if (status != opencl_h.CL_SUCCESS()) {
                                                System.out.println("failed to set arg " + status);
                                            }
                                        }
                                        case Float floatArg->{
                                            var floatPtr = program.context.device.platform.openCL.arena.allocateFrom(opencl_h.C_FLOAT, floatArg);
                                            status = opencl_h.clSetKernelArg(kernel, i, opencl_h.C_FLOAT.byteSize(), floatPtr);
                                            if (status != opencl_h.CL_SUCCESS()) {
                                                System.out.println("failed to set arg " + status);
                                            }
                                        }
                                        default -> throw new IllegalStateException("Unexpected value: " + args[i]);
                                    }
                                }
                            }

                            // We need to store x,y,z sizes so this is a kind of int3
                            var globalSizePtr = program.context.device.platform.openCL.arena.allocate(opencl_h.C_INT, 3);
                            globalSizePtr.set(opencl_h.C_INT, 0, range);
                            globalSizePtr.set(opencl_h.C_INT, 1*opencl_h.C_INT.byteSize(), 0);
                            globalSizePtr.set(opencl_h.C_INT, 2*opencl_h.C_INT.byteSize(), 0);
                            status = opencl_h.clEnqueueNDRangeKernel(
                                    program.context.queue,
                                    kernel,
                                    1, // this must match the # of dims we are using in this case 1 of 3
                                    NULL,
                                    globalSizePtr,
                                    NULL,
                                    block ? 0 : eventc,
                                    block ? NULL : ((eventc == 0) ? NULL : eventsPtr),
                                    block ? NULL : eventsPtr.asSlice(eventc * opencl_h.cl_event.byteSize(), opencl_h.cl_event
                                    )
                            );
                            if (status != opencl_h.CL_SUCCESS()) {
                                System.out.println("failed to enqueue NDRange " + status);
                            }

                            if (block) {
                                opencl_h.clFlush(program.context.queue);
                            } else {
                                eventc++;
                                status = opencl_h.clWaitForEvents(eventc, eventsPtr);
                                if (status != opencl_h.CL_SUCCESS()) {
                                    System.out.println("failed to wait for ndrange events " + status);
                                }
                            }

                            for (int i = 0; i < args.length; i++) {
                                if (args[i] instanceof MemorySegment memorySegment) {
                                    MemorySegment clMem = bufPtr.get(opencl_h.cl_mem, (long) i * opencl_h.cl_mem.byteSize());
                                    status = opencl_h.clEnqueueReadBuffer(program.context.queue,
                                            clMem,
                                            block ? opencl_h.CL_TRUE() : opencl_h.CL_FALSE(),
                                            0,
                                            memorySegment.byteSize(),
                                            memorySegment,
                                            block ? 0 : eventc,
                                            block ? NULL : ((eventc == 0) ? NULL : eventsPtr),
                                            block ? NULL : eventsPtr.asSlice(eventc * opencl_h.cl_event.byteSize(), opencl_h.cl_event)// block?NULL:readEventPtr
                                    );
                                    if (status != opencl_h.CL_SUCCESS()) {
                                        System.out.println("failed to enqueue read " + status);
                                    }
                                    if (!block) {
                                        eventc++;
                                    }
                                }
                            }
                            if (!block) {
                                status = opencl_h.clWaitForEvents(eventc, eventsPtr);
                                if (status != opencl_h.CL_SUCCESS()) {
                                    System.out.println("failed to wait for events " + status);
                                }
                            }
                            for (int i = 0; i < args.length; i++) {
                                if (args[i] instanceof MemorySegment memorySegment) {
                                    MemorySegment clMem = bufPtr.get(opencl_h.cl_mem, (long) i * opencl_h.cl_mem.byteSize());
                                    status = opencl_h.clReleaseMemObject(clMem);
                                    if (status != opencl_h.CL_SUCCESS()) {
                                        System.out.println("failed to release memObject " + status);
                                    }
                                }
                            }
                        }
                    }

                    public Kernel getKernel(String kernelName) {
                        return new Kernel(this, kernelName);
                    }
                }

                public Program buildProgram(String source) {
                    var program = new Program(this, source);
                    return program;
                }
            }

            public Context createContext() {

                var statusPtr = platform.openCL.arena.allocateFrom(opencl_h.C_INT, 0);
                MemorySegment context;
                var deviceIds = platform.openCL.arena.allocateFrom(opencl_h.C_POINTER, this.deviceId);
                if ((context = opencl_h.clCreateContext(NULL, 1, deviceIds, NULL, NULL, statusPtr)) == NULL) {
                    int status = statusPtr.get(opencl_h.C_INT, 0);
                    System.out.println("Failed to get context  ");
                    return null;
                } else {
                    int status = statusPtr.get(opencl_h.C_INT, 0);
                    if (status != opencl_h.CL_SUCCESS()) {
                        System.out.println("failed to get context  " + status);
                    }
                    return new Context(this, context);
                }
            }
        }

        int intPlatformInfo(int query) {
            var value = 0;
            if ((opencl_h.clGetPlatformInfo(platformId, query, opencl_h.C_INT.byteSize(), intValuePtr, NULL)) != opencl_h.CL_SUCCESS()) {
                System.out.println("Failed to get query " + query);
            } else {
                value = intValuePtr.get(opencl_h.C_INT, 0);
            }
            return value;
        }

        String strPlatformInfo(int query) {
            String value = null;
            int status;
            if ((status = opencl_h.clGetPlatformInfo(platformId, query, 2048, byte2048ValuePtr, intValuePtr)) != opencl_h.CL_SUCCESS()) {
                System.err.println("Failed to get query " + query);
            } else {
                int len = intValuePtr.get(opencl_h.C_INT, 0);
                byte[] bytes = byte2048ValuePtr.toArray(ValueLayout.JAVA_BYTE);
                value = new String(bytes).substring(0, len - 1);
            }
            return value;
        }

        CLWrap openCL;
        MemorySegment platformId;
        List<Device> devices = new ArrayList<>();
        final MemorySegment intValuePtr;
        final MemorySegment byte2048ValuePtr;

        String platformName() {
            return strPlatformInfo(opencl_h.CL_PLATFORM_NAME());
        }

        String vendorName() {
            return strPlatformInfo(opencl_h.CL_PLATFORM_VENDOR());
        }

        String version() {
            return strPlatformInfo(opencl_h.CL_PLATFORM_VERSION());
        }

        public Platform(CLWrap openCL, MemorySegment platformId) {
            this.openCL = openCL;
            this.platformId = platformId;
            this.intValuePtr = openCL.arena.allocateFrom(opencl_h.C_INT, 0);
            this.byte2048ValuePtr = openCL.arena.allocate(opencl_h.C_CHAR, 2048);
            var devicecPtr = openCL.arena.allocateFrom(opencl_h.C_INT, 0);
            int status;
            if ((status = opencl_h.clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL(), 0, NULL, devicecPtr)) != opencl_h.CL_SUCCESS()) {
                System.err.println("Failed getting devicec for platform 0 ");
            } else {
                int devicec = devicecPtr.get(opencl_h.C_INT, 0);
                //  System.out.println("platform 0 has " + devicec + " device" + ((devicec > 1) ? "s" : ""));
                var deviceIdsPtr = openCL.arena.allocate(opencl_h.C_POINTER, devicec);
                if ((status = opencl_h.clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL(), devicec, deviceIdsPtr, devicecPtr)) != opencl_h.CL_SUCCESS()) {
                    System.err.println("Failed getting deviceids  for platform 0 ");
                } else {
                    // System.out.println("We have "+devicec+" device ids");
                    for (int i = 0; i < devicec; i++) {
                        devices.add(new Device(this, deviceIdsPtr.get(opencl_h.C_POINTER, i * opencl_h.C_POINTER.byteSize())));
                    }
                }
            }
        }
    }

    List<Platform> platforms = new ArrayList<>();

    Arena arena;

    CLWrap(Arena arena) {
        this.arena = arena;
        var platformcPtr = arena.allocateFrom(opencl_h.C_INT, 0);

        if ((opencl_h.clGetPlatformIDs(0, NULL, platformcPtr)) != opencl_h.CL_SUCCESS()) {
            System.out.println("Failed to get opencl platforms");
        } else {
            int platformc = platformcPtr.get(opencl_h.C_INT, 0);
            // System.out.println("There are "+platformc+" platforms");
            var platformIdsPtr = arena.allocate(opencl_h.C_POINTER, platformc);
            if ((opencl_h.clGetPlatformIDs(platformc, platformIdsPtr, platformcPtr)) != opencl_h.CL_SUCCESS()) {
                System.out.println("Failed getting platform ids");
            } else {
                for (int i = 0; i < platformc; i++) {
                    // System.out.println("We should have the ids");
                    platforms.add(new Platform(this, platformIdsPtr.get(opencl_h.C_POINTER, i)));
                }
            }
        }
    }


    public static void main(String[] args) throws IOException {
        try (var arena = Arena.ofConfined()) {
            CLWrap openCL = new CLWrap(arena);

            Platform.Device[] selectedDevice = new Platform.Device[1];
            openCL.platforms.forEach(platform -> {
                System.out.println("Platform Name " + platform.platformName());
                platform.devices.forEach(device -> {
                    System.out.println("   Compute Units     " + device.computeUnits());
                    System.out.println("   Device Name       " + device.deviceName());
                    System.out.println("   Built In Kernels  " + device.builtInKernels());
                    selectedDevice[0] = device;
                });
            });
            var context = selectedDevice[0].createContext();
            var program = context.buildProgram("""
                    __kernel void squares(__global int* in,__global int* out ){
                        int gid = get_global_id(0);
                        out[gid] = in[gid]*in[gid];
                    }
                    """);
            var kernel = program.getKernel("squares");
            var in = arena.allocate(opencl_h.C_INT, 512);
            var out = arena.allocate(opencl_h.C_INT, 512);
            for (int i = 0; i < 512; i++) {
                in.set(opencl_h.C_INT, (int) i * opencl_h.C_INT.byteSize(), i);
            }
            kernel.run(512, in, out);
            for (int i = 0; i < 512; i++) {
                System.out.println(i + " " + out.get(opencl_h.C_INT, (int) i * opencl_h.C_INT.byteSize()));
            }
        }
    }
}
