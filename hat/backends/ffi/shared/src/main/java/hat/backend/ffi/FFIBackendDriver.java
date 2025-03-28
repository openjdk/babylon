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


import hat.backend.Backend;
import hat.buffer.ArgArray;
import hat.buffer.Buffer;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.util.HashMap;
import java.util.Map;

public abstract class FFIBackendDriver implements Backend {
    public boolean isAvailable() {
        return ffiLib.available;
    }


    public static class BackendBridge {
        // CUDA this combines Device+Stream+Context
        // OpenCL this combines Platform+Device+Queue+Context
        public static class CompilationUnitBridge {
            // CUDA calls this a Module
            // OpenCL calls this a program
            public static class KernelBridge {
                // CUDA calls this a Function
                // OpenCL calls this a Program
                CompilationUnitBridge compilationUnitBridge;
                long handle;
                final FFILib.VoidHandleMethodPtr releaseKernel_MPtr;
                String name;
                final FFILib.LongLongAddressMethodPtr ndrange_MPtr;
                KernelBridge(CompilationUnitBridge compilationUnitBridge, String name, long handle) {
                    this.compilationUnitBridge = compilationUnitBridge;
                    this.handle = handle;
                    this.releaseKernel_MPtr = compilationUnitBridge.backendBridge.ffiLib.voidHandleFunc("releaseKernel");
                    this.ndrange_MPtr = compilationUnitBridge.backendBridge.ffiLib.longLongAddressFunc("ndrange");
                    this.name = name;
                }



                public void ndRange(ArgArray argArray) {
                    this.ndrange_MPtr.invoke(handle, Buffer.getMemorySegment(argArray));
                }
                void release() {
                    releaseKernel_MPtr.invoke(handle);
                }
            }

            BackendBridge backendBridge;
            String source;
            final FFILib.VoidHandleMethodPtr releaseCompilationUnit_MPtr;
            final FFILib.BooleanHandleMethodPtr compilationUnitOK_MPtr;
            final FFILib.LongHandleIntAddressMethodPtr getKernel_MPtr;


            long handle;
            Map<String, KernelBridge> kernels = new HashMap<>();

            CompilationUnitBridge(BackendBridge backendBridge, long handle, String source) {
                this.backendBridge = backendBridge;
                this.handle = handle;
                this.source = source;
                this.releaseCompilationUnit_MPtr = backendBridge.ffiLib.voidHandleFunc("releaseCompilationUnit");
                this.compilationUnitOK_MPtr = backendBridge.ffiLib.booleanHandleFunc("compilationUnitOK");
                this.getKernel_MPtr = backendBridge.ffiLib.longHandleIntAddressFunc("getKernel");



            }

            void release() {
                this.releaseCompilationUnit_MPtr.invoke(handle);
            }

            boolean ok() {
                return this.compilationUnitOK_MPtr.invoke(handle);
            }

            public KernelBridge getKernel(String kernelName) {
                KernelBridge kernelBridge = kernels.computeIfAbsent(kernelName, _ ->
                        new KernelBridge(this, kernelName,
                                getKernel_MPtr.invoke(handle, kernelName.length(), Arena.global().allocateFrom(kernelName)))
                );
                return kernelBridge;


            }


        }

        FFILib ffiLib;
        long handle;

        Map<Long, CompilationUnitBridge> compilationUnits = new HashMap<>();
        final FFILib.LongHandleIntAddressMethodPtr compile_MPtr;
        final FFILib.VoidHandleMethodPtr computeStart_MPtr;
        final FFILib.VoidHandleMethodPtr computeEnd_MPtr;
        final FFILib.VoidAddressMethodPtr dumpArgArray_MPtr;

        final FFILib.VoidHandleMethodPtr info_MPtr;
        final FFILib.BooleanHandleAddressLongMethodPtr getBufferFromDeviceIfDirty_MPtr;
        BackendBridge(FFILib ffiLib) {
            this.ffiLib = ffiLib;
            this.compile_MPtr = ffiLib.longHandleIntAddressFunc("compile");
            this.dumpArgArray_MPtr = ffiLib.voidAddressFunc("dumpArgArray");
            this.info_MPtr = ffiLib.voidHandleFunc("info");
            this.computeStart_MPtr = ffiLib.voidHandleFunc("computeStart");
            this.computeEnd_MPtr = ffiLib.voidHandleFunc("computeEnd");
            this.getBufferFromDeviceIfDirty_MPtr = ffiLib.booleanHandleAddressLongFunc("getBufferFromDeviceIfDirty");
        }


        void release() {

        }

        private CompilationUnitBridge compilationUnit(long handle, String source) {
            return compilationUnits.computeIfAbsent(handle, _ ->
                    new CompilationUnitBridge(this, handle, source)
            );
        }

        public CompilationUnitBridge compile(String source) {
            var compilationUnitHandle = compile_MPtr.invoke(handle, source.length(), Arena.global().allocateFrom(source));
            return compilationUnit(compilationUnitHandle, source);
        }

        public Buffer getBufferFromDeviceIfDirty(Buffer buffer) {
            MemorySegment memorySegment = Buffer.getMemorySegment(buffer);
            boolean ok = getBufferFromDeviceIfDirty_MPtr.invoke(handle, memorySegment, memorySegment.byteSize());
            if (!ok) {
                throw new IllegalStateException("Failed to get buffer from backend");
            }
            return buffer;

        }

        public void computeStart() {
            computeStart_MPtr.invoke(handle);
        }

        public void computeEnd() {
            computeEnd_MPtr.invoke(handle);
        }

        public void info() {
            info_MPtr.invoke(handle);
        }

        public void dumpArgArray(ArgArray argArray) {
            dumpArgArray_MPtr.invoke(Buffer.getMemorySegment(argArray));
        }


    }



    public final FFILib ffiLib;
    public final BackendBridge backendBridge;

    public FFIBackendDriver(String libName) {
        this.ffiLib = new FFILib(libName);
        this.backendBridge = new BackendBridge(ffiLib);

    }


}
