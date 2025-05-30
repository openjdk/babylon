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
package wrap;

import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;
import java.lang.invoke.VarHandle;

public record Scalar(String name, MemorySegment memorySegment,
                     VarHandle varHandle) {

    public static Scalar of(MemorySegment memorySegment, MemoryLayout memoryLayout, String name) {
        return of(memorySegment, memoryLayout, MemoryLayout.PathElement.groupElement(name));
    }

    public static Scalar of(MemorySegment memorySegment, MemoryLayout memoryLayout, String name, Object initial) {
        return of(memorySegment, memoryLayout, name).set(initial);
    }

    public static Scalar of(MemorySegment memorySegment, MemoryLayout memoryLayout,
                            MemoryLayout.PathElement... pathElements) {
        VarHandle vh = memoryLayout.varHandle(pathElements);
        String name = null;
        for (int i = 0; i < pathElements.length; i++) {
            MemoryLayout.PathElement pathElement = pathElements[i];
            // Why can't I access LayoutPath?
            if (pathElement.toString().isEmpty()) {
                name = pathElement.toString();
            }
        }
        return new Scalar(name, memorySegment, vh);

    }

    public Scalar set(Object o) {
        varHandle.set(memorySegment, 0, o);
        return this;
    }

    public Object get() {
        return varHandle.get(memorySegment, 0);
    }

    public int i32() {
        return (int) get();
    }

    public long i64() {
        return (long) get();
    }

    public float f32() {
        return (float) get();
    }

    public double f64() {
        return (double) get();
    }
}
