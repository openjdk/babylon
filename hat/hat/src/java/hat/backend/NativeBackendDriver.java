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
package hat.backend;


import hat.buffer.ArgArray;
import hat.buffer.BackendConfig;
import hat.buffer.Buffer;
import hat.buffer.S32Array;

import java.lang.foreign.Arena;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

public abstract class NativeBackendDriver implements Backend {

    public boolean isAvailable() {
        return nativeLibrary.available;
    }

    final MethodHandle getBackend_MH;
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

    public long backendHandle = 0;


    public final NativeLib nativeLibrary;

    public NativeBackendDriver(String libName) {
        this.nativeLibrary = new NativeLib(libName);
        this.dumpArgArray_MH = nativeLibrary.voidFunc("dumpArgArray", ADDRESS);
        this.getDevice_MH = nativeLibrary.longFunc("getDeviceHandle");
        this.releaseDevice_MH = nativeLibrary.voidFunc("releaseDeviceHandle", JAVA_LONG);
        this.getMaxComputeUnits_MH = nativeLibrary.intFunc("getMaxComputeUnits", JAVA_LONG);
        this.compileProgram_MH = nativeLibrary.longFunc("compileProgram", JAVA_LONG, JAVA_INT, ADDRESS);
        this.releaseProgram_MH = nativeLibrary.voidFunc("releaseProgram", JAVA_LONG);
        this.getKernel_MH = nativeLibrary.longFunc("getKernel", JAVA_LONG, JAVA_INT, ADDRESS);
        this.programOK_MH = nativeLibrary.booleanFunc("programOK", JAVA_LONG);
        this.releaseKernel_MH = nativeLibrary.voidFunc("releaseKernel", JAVA_LONG);
        this.ndrange_MH = nativeLibrary.longFunc("ndrange", JAVA_LONG, JAVA_INT, ADDRESS);
        this.info_MH = nativeLibrary.voidFunc("info", JAVA_LONG);
        this.getBackend_MH = nativeLibrary.longFunc("getBackend", ADDRESS, JAVA_INT, ADDRESS);

    }
    public long getBackend(BackendConfig backendConfig){

        try {
            if (backendConfig==null){
                backendHandle = (long) getBackend_MH.invoke(MemorySegment.NULL, 0, MemorySegment.NULL);
            }else {
                String schema = backendConfig.schema();
                var arena = Arena.global();
                var cstr = arena.allocateFrom(schema);
                backendHandle = (long) getBackend_MH.invoke(backendConfig.memorySegment(), schema.length(), cstr);
            }
        } catch (Throwable throwable) {
            throw new IllegalStateException(throwable);
        }
        return  backendHandle;
    }
    public int getGetMaxComputeUnits() {
        if (backendHandle == 0L){
            throw new IllegalStateException("no backend handle");
        }
        try {
            return (int) getMaxComputeUnits_MH.invoke(backendHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }
    public void info() {
        if (backendHandle == 0L){
            throw new IllegalStateException("no backend handle");
        }
        try {
            info_MH.invoke(backendHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void dumpArgArray(ArgArray argArray) {
        try {
            dumpArgArray_MH.invoke(argArray.memorySegment());
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public long compileProgram(String source) {
        if (backendHandle == 0L){
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

    public void  ndRange(long kernelHandle, int range, ArgArray argArray ) {
        try {
             this.ndrange_MH.invoke(kernelHandle, range, argArray.memorySegment());
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public boolean programOK(long programHandle) {
        try {
            return (Boolean) programOK_MH.invoke(programHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public long getKernel(long programHandle, String kernelName) {
        try {
            var arena = Arena.global();
            var cstr = arena.allocateFrom(kernelName);
            return ((Long) getKernel_MH.invoke(programHandle, kernelName.length(), cstr)).longValue();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void releaseKernel(long kernelHandle) {
        try {
            releaseKernel_MH.invoke(kernelHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    public void releaseProgram(long programHandle) {
        try {
            releaseProgram_MH.invoke(programHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }


    public void release() {
        if (backendHandle == 0L){
            throw new IllegalStateException("no backend handle");
        }
        try {
            releaseDevice_MH.invoke(backendHandle);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }


}
