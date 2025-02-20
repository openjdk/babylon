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
package hat.backend.ffi;


import hat.ComputeContext;
import hat.NDRange;
import hat.buffer.Buffer;
import hat.buffer.BufferTracker;
import hat.callgraph.KernelCallGraph;

import java.lang.invoke.MethodHandle;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class OpenCLBackend extends C99FFIBackend implements BufferTracker {
    public record Mode(int bits) {
        private static final int GPU_BIT = 1 << 1;
        private static final int CPU_BIT = 1 << 2;
        private static final int MINIMIZE_COPIES_BIT = 1 << 3;
        private static final int TRACE_BIT = 1 << 4;
        private static final int PROFILE_BIT = 1 << 5;

        public static Mode of(String name) {
            return switch (name){
                case "GPU" -> GPU();
                case "CPU" -> CPU();
                case "MINIMIZE_COPIES" -> MINIMIZE_COPIES();
                case "TRACE" -> TRACE();
                case "PROFILE" -> PROFILE();
                default -> Mode.of(0);
            };
        }
        public static Mode of() {
            List<Mode> modes = new ArrayList<>();
            if (( ((System.getenv("HAT") instanceof String e)?e:"")+
                    ((System.getProperty("HAT") instanceof String p)?p:"")) instanceof String opts) {
                Arrays.stream(opts.split(",")).forEach(opt ->
                        modes.add(of(opt))
                );
            }
            if (System.getenv("HAT_GPU") != null || System.getProperty("HAT_GPU") != null) {
                modes.add(GPU());
            }
            if (System.getenv("HAT_CPU") != null || System.getProperty("HAT_CPU") != null) {
                modes.add(CPU());
            }
            if (System.getenv("HAT_TRACE") != null || System.getProperty("HAT_TRACE") != null) {
                modes.add(TRACE());
            }
            if (System.getenv("HAT_PROFILE") != null || System.getProperty("HAT_PROFILE") != null) {
                modes.add(PROFILE());
            }
            if (System.getenv("HAT_MINIMIZE_COPIES") != null || System.getProperty("HAT_PROFILE") !=null) {
                modes.add(MINIMIZE_COPIES());
            }
           return of(modes);
        }
        public static Mode of(int bits) {

            return new Mode(bits);
        }
        public static Mode of(List<Mode> modes) {
            int allBits = 0;
            for (Mode mode : modes) {
                allBits |= mode.bits;
            }
            return new Mode(allBits);
        }
        public static Mode of(Mode ...modes) {
           return of(List.of(modes));
        }
        public Mode and(Mode ...modes) {
            return Mode.of(Mode.of(List.of(modes)).bits&bits);
        }
        public Mode or(Mode ...modes) {
            return Mode.of(Mode.of(List.of(modes)).bits|bits);
        }
        public static Mode CPU() {
            return new Mode(CPU_BIT);
        }
        public boolean isCPU() {
            return (bits&CPU_BIT)==CPU_BIT;
        }
        public static Mode GPU() {
            return new Mode(GPU_BIT);
        }
        public boolean isGPU() {
            return (bits&GPU_BIT)==GPU_BIT;
        }
        public static Mode PROFILE() {
            return new Mode(PROFILE_BIT);
        }
        public boolean isPROFILE() {
            return (bits&PROFILE_BIT)==PROFILE_BIT;
        }
        public static Mode TRACE() {
            return new Mode(TRACE_BIT);
        }
        public boolean isTRACE() {
            return (bits&TRACE_BIT)==TRACE_BIT;
        }
        public static Mode MINIMIZE_COPIES() {
            return new Mode(MINIMIZE_COPIES_BIT);
        }
        public boolean isMINIMIZE_COPIES() {
            return (bits&MINIMIZE_COPIES_BIT)==MINIMIZE_COPIES_BIT;
        }

        @Override
        public String toString() {
            StringBuilder builder = new StringBuilder();
            if (isCPU()) {
                if (!builder.isEmpty()){
                    builder.append("|");
                }
                builder.append("CPU");
            }
            if (isGPU()) {
                if (!builder.isEmpty()){
                    builder.append("|");
                }
                builder.append("GPU");
            }
            if (isTRACE()) {
                if (!builder.isEmpty()){
                    builder.append("|");
                }
                builder.append("TRACE");
            }
            if (isPROFILE()) {
                if (!builder.isEmpty()){
                    builder.append("|");
                }
                builder.append("PROFILE");
            }
            if (isMINIMIZE_COPIES()) {
                if (!builder.isEmpty()){
                    builder.append("|");
                }
                builder.append("MINIMIZE_COPIES");
            }

            return builder.toString();
        }
    }

    final MethodHandle getBackend_MH;
    public long getBackend(int mode, int platform, int device, int unused) {
        try {
            backendHandle = (long) getBackend_MH.invoke(mode, platform, device, unused);
        } catch (Throwable throwable) {
            throw new IllegalStateException(throwable);
        }
        return backendHandle;
    }
    public OpenCLBackend(Mode mode) {
        super("opencl_backend");
        System.out.println(mode);
        getBackend_MH  =  nativeLibrary.longFunc("getOpenCLBackend",JAVA_INT,JAVA_INT, JAVA_INT, JAVA_INT);
        getBackend(mode.bits,0, 0, 0 );
        info();
    }

    public OpenCLBackend() {
        this(Mode.of().or(Mode.GPU()));
    }


    @Override
    public void computeContextHandoff(ComputeContext computeContext) {
        //System.out.println("OpenCL backend received computeContext");
        injectBufferTracking(computeContext.computeCallGraph.entrypoint);
    }

    @Override
    public void dispatchKernel(KernelCallGraph kernelCallGraph, NDRange ndRange, Object... args) {
        //System.out.println("OpenCL backend dispatching kernel " + kernelCallGraph.entrypoint.method);
        CompiledKernel compiledKernel = kernelCallGraphCompiledCodeMap.computeIfAbsent(kernelCallGraph, (_) -> {
            String code = createCode(kernelCallGraph, new OpenCLHatKernelBuilder(), args);
            System.out.println(code);
            long programHandle = compileProgram(code);
            if (programOK(programHandle)) {
                long kernelHandle = getKernel(programHandle, kernelCallGraph.entrypoint.method.getName());
                return new CompiledKernel(this, kernelCallGraph, code, kernelHandle, args);
            } else {
                throw new IllegalStateException("opencl failed to compile ");
            }
        });
        compiledKernel.dispatch(ndRange,args);

    }

    @Override
    public void preMutate(Buffer b) {
        if (b.isDeviceDevice()){
            getBufferFromDeviceIfDirty(b); // This might block to fetch from device
            b.clearDeviceDirty();
        }
    }

    @Override
    public void postMutate(Buffer b) {
       b.setHostDirty();

    }

    @Override
    public void preAccess(Buffer b) {
        if (b.isDeviceDevice()){
            getBufferFromDeviceIfDirty(b);
            b.clearDeviceDirty();// this should reset deviceDirty!
        }
    }

    @Override
    public void postAccess(Buffer b) {
       // a no op
    }

    @Override
    public void preEscape(Buffer b) {
            getBufferFromDeviceIfDirty(b).clearDeviceDirty(); //  we have to assume the escaped buffer is about to be accessed
    }

    @Override
    public void postEscape(Buffer b) {
        b.setHostDirty(); // We have no choice but to assume escaped buffer has been mutates
    }
}
