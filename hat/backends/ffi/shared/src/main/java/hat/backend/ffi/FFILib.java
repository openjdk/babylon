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

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.ADDRESS;
import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

public class FFILib {
    final public String name;
    public final boolean available;

    final public Linker nativeLinker;

    final public SymbolLookup loaderLookup;

    public static class MethodPtr{
        final FFILib ffiLib;
        final FunctionDescriptor functionDescriptor;
        final MethodHandle mh;
        final String name;

        MethodPtr(FFILib ffiLib, FunctionDescriptor descriptor, String name) {
            this.ffiLib = ffiLib;
            this.functionDescriptor= descriptor;
            this.mh = ffiLib.loaderLookup.find(name)
                    .map(symbolSegment -> ffiLib.nativeLinker.downcallHandle(symbolSegment, descriptor))
                    .orElse(null);
            if (this.mh == null) {
                System.err.println("Could not find method " + name + " in " + ffiLib.name);
            }
            this.name = name;
        }

    }

    public static class VoidAddressMethodPtr extends MethodPtr{
        VoidAddressMethodPtr(FFILib ffiLib, String name) {
            super(ffiLib,FunctionDescriptor.ofVoid(ADDRESS), name);
        }
        public void invoke(MemorySegment memorySegment) {
            try {
                mh.invoke(memorySegment);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static class VoidHandleMethodPtr extends MethodPtr{
        VoidHandleMethodPtr(FFILib ffiLib, String name) {
            super(ffiLib, FunctionDescriptor.ofVoid(JAVA_LONG), name);
        }
        public void invoke(long handle) {
            if (handle == 0) {
                throw new RuntimeException("handle is zero");
            }
            try {
                mh.invoke(handle);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static class BooleanHandleMethodPtr extends MethodPtr{
        BooleanHandleMethodPtr(FFILib ffiLib, String name) {
            super(ffiLib, FunctionDescriptor.of(JAVA_BOOLEAN,JAVA_LONG),name);
        }
        public boolean invoke(long handle) {
            if (handle == 0L) {
                throw new IllegalArgumentException("handle is zero");
            }
            try {
                return (boolean)mh.invoke(handle);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static class BooleanHandleAddressLongMethodPtr extends MethodPtr{
        BooleanHandleAddressLongMethodPtr(FFILib ffiLib, String name) {
            super(ffiLib, FunctionDescriptor.of(JAVA_BOOLEAN,JAVA_LONG,ADDRESS,JAVA_LONG), name);
        }
        public boolean invoke(long handle,MemorySegment memorySegment, long len) {
            if (handle == 0L) {
                throw new IllegalArgumentException("handle is zero");
            }
            try {
                return (boolean)mh.invoke(handle, memorySegment, len);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static class LongHandleIntAddressMethodPtr extends MethodPtr{
        LongHandleIntAddressMethodPtr(FFILib ffiLib, String name) {
            super(ffiLib, FunctionDescriptor.of(JAVA_LONG,JAVA_LONG,JAVA_INT,ADDRESS), name);
        }
        public long invoke(long handle, int i, MemorySegment memorySegment) {
            if (handle == 0L) {
                throw new IllegalArgumentException("handle is zero");
            }
            try {
                return (long)mh.invoke(handle, i, memorySegment);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static class LongIntMethodPtr extends MethodPtr{
        LongIntMethodPtr(FFILib ffiLib, String name) {
            super(ffiLib,FunctionDescriptor.of(JAVA_LONG,JAVA_INT), name);
        }
        public long invoke( int i) {
            try {
                return (long)mh.invoke(i);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
    }

    public static class LongLongAddressMethodPtr extends MethodPtr{
        LongLongAddressMethodPtr(FFILib ffiLib, String name) {
            super(ffiLib,FunctionDescriptor.of(JAVA_LONG,JAVA_LONG,ADDRESS), name);
        }
        public long invoke(long l,  MemorySegment memorySegment) {
            try {
                return (long)mh.invoke(l, memorySegment);
            } catch (Throwable e) {
                throw new RuntimeException(e);
            }
        }
    }

    public FFILib(String name) {
        this.name = name;

        boolean nonFinalAvailable = true;
        try {
            Runtime.getRuntime().loadLibrary(name);
        } catch (UnsatisfiedLinkError e) {
            nonFinalAvailable = false;
        }
        this.available = nonFinalAvailable;
        this.nativeLinker = Linker.nativeLinker();
        this.loaderLookup = SymbolLookup.loaderLookup();
    }


    public VoidAddressMethodPtr voidAddressFunc(String name) {
        return new VoidAddressMethodPtr(this, name);
    }

    public VoidHandleMethodPtr voidHandleFunc(String name) {
        return new VoidHandleMethodPtr(this, name);
    }
    public BooleanHandleMethodPtr booleanHandleFunc(String name) {
        return new BooleanHandleMethodPtr(this, name);
    }
    public BooleanHandleAddressLongMethodPtr booleanHandleAddressLongFunc(String name) {
        return new BooleanHandleAddressLongMethodPtr(this, name);
    }
    public LongHandleIntAddressMethodPtr longHandleIntAddressFunc(String name) {
        return new LongHandleIntAddressMethodPtr(this, name);
    }
    public LongIntMethodPtr longIntFunc(String name) {
        return new LongIntMethodPtr(this, name);
    }
    public LongLongAddressMethodPtr longLongAddressFunc(String name) {
        return new LongLongAddressMethodPtr(this, name);
    }

}
