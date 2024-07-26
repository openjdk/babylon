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

import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.JAVA_BOOLEAN;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

public class NativeLib {
    final public String name;
    public final boolean available;


    final public Linker nativeLinker;

    final public SymbolLookup loaderLookup;

    NativeLib(String name) {
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


    MethodHandle voidFunc(String name, MemoryLayout... args) {
        return loaderLookup.find(name)
                .map(symbolSegment -> nativeLinker.downcallHandle(symbolSegment,
                        FunctionDescriptor.ofVoid(args)))
                .orElse(null);
    }

    MethodHandle typedFunc(String name, MemoryLayout returnLayout, MemoryLayout... args) {
        return loaderLookup.find(name)
                .map(symbolSegment -> nativeLinker.downcallHandle(symbolSegment,
                        FunctionDescriptor.of(returnLayout, args)))
                .orElse(null);
    }

    MethodHandle longFunc(String name, MemoryLayout... args) {
        return typedFunc(name, JAVA_LONG, args);
    }

    MethodHandle booleanFunc(String name, MemoryLayout... args) {
        return typedFunc(name, JAVA_BOOLEAN, args);
    }

    MethodHandle intFunc(String name, MemoryLayout... args) {
        return typedFunc(name, JAVA_INT, args);
    }
}
