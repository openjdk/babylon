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
import hat.buffer.BufferTracker;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

public abstract class FFIBackendDriver implements Backend {
    public boolean isAvailable() {
        return nativeLibrary.available;
    }
    final MethodHandle computeStart_MH;
    final MethodHandle computeEnd_MH;
    final MethodHandle dumpArgArray_MH;
    final MethodHandle getDevice_MH;
    final MethodHandle releaseDevice_MH;
    final MethodHandle getMaxComputeUnits_MH;
    final MethodHandle compileProgram_MH;
    final MethodHandle releaseProgram_MH;
    final MethodHandle getKernel_MH;
    final MethodHandle programOK_MH;
    final MethodHandle releaseKernel_MH;
    final MethodHandle ndrange_MH;
    final MethodHandle info_MH;
    final MethodHandle getBufferFromDeviceIfDirty_MH;
    public long backendHandle = 0;
    public final FFILib nativeLibrary;

    public FFIBackendDriver(String libName) {
        this.nativeLibrary = new FFILib(libName);
        this.dumpArgArray_MH = nativeLibrary.voidFunc("dumpArgArray", ADDRESS);
        this.getDevice_MH = nativeLibrary.longFunc("getDeviceHandle");
        this.releaseDevice_MH = nativeLibrary.voidFunc("releaseDeviceHandle", JAVA_LONG);
        this.getMaxComputeUnits_MH = nativeLibrary.intFunc("getMaxComputeUnits", JAVA_LONG);
        this.compileProgram_MH = nativeLibrary.longFunc("compileProgram", JAVA_LONG, JAVA_INT, ADDRESS);
        this.releaseProgram_MH = nativeLibrary.voidFunc("releaseProgram", JAVA_LONG);
        this.getKernel_MH = nativeLibrary.longFunc("getKernel", JAVA_LONG, JAVA_INT, ADDRESS);
        this.programOK_MH = nativeLibrary.booleanFunc("programOK", JAVA_LONG);
        this.releaseKernel_MH = nativeLibrary.voidFunc("releaseKernel", JAVA_LONG);
        this.ndrange_MH = nativeLibrary.longFunc("ndrange", JAVA_LONG,  ADDRESS);
        this.info_MH = nativeLibrary.voidFunc("info", JAVA_LONG);
        this.computeStart_MH = nativeLibrary.voidFunc("computeStart", JAVA_LONG);
        this.computeEnd_MH = nativeLibrary.voidFunc("computeEnd", JAVA_LONG);
        this.getBufferFromDeviceIfDirty_MH = nativeLibrary.booleanFunc("getBufferFromDeviceIfDirty",JAVA_LONG, ADDRESS, JAVA_LONG);
    }

    public Buffer getBufferFromDeviceIfDirty(Buffer buffer) {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        if (this instanceof BufferTracker) {
            try {
                MemorySegment memorySegment = Buffer.getMemorySegment(buffer);
                boolean ok = (Boolean) getBufferFromDeviceIfDirty_MH.invoke(backendHandle, memorySegment, memorySegment.byteSize());
                if (!ok){
                    throw new IllegalStateException("Failed to get buffer from backend");
                }

            } catch (Throwable throwable) {
                throw new IllegalStateException(throwable);
            }
        }
        return buffer;

    }

    public int getGetMaxComputeUnits() {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            return (int) getMaxComputeUnits_MH.invoke(backendHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void computeStart() {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            computeStart_MH.invoke(backendHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
    public void computeEnd() {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            computeEnd_MH.invoke(backendHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
    public void info() {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            info_MH.invoke(backendHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void dumpArgArray(ArgArray argArray) {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            dumpArgArray_MH.invoke(Buffer.getMemorySegment(argArray));
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public long compileProgram(String source) {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            var arena = Arena.global();
            var cstr = arena.allocateFrom(source);
            return (Long) compileProgram_MH.invoke(backendHandle, source.length(), cstr);

        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void ndRange(long kernelHandle,  ArgArray argArray) {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            this.ndrange_MH.invoke(kernelHandle, Buffer.getMemorySegment(argArray));
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public boolean programOK(long programHandle) {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            return (Boolean) programOK_MH.invoke(programHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public long getKernel(long programHandle, String kernelName) {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            var arena = Arena.global();
            var cstr = arena.allocateFrom(kernelName);
            return ((Long) getKernel_MH.invoke(programHandle, kernelName.length(), cstr)).longValue();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void releaseKernel(long kernelHandle) {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            releaseKernel_MH.invoke(kernelHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void releaseProgram(long programHandle) {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            releaseProgram_MH.invoke(programHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void release() {
        if (backendHandle == 0L) {
            throw new IllegalStateException("no backend handle");
        }
        try {
            releaseDevice_MH.invoke(backendHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
}
