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

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;

public interface ArenaHolder {
    public static ArenaHolder wrap(Arena arena) {
        return ()-> arena;
    }

    Arena arena();

    default Wrap.IntPtr intPtr(int value){
        return Wrap.IntPtr.of(arena(), value);
    }
    default Wrap.LongPtr longPtr(long value){
        return Wrap.LongPtr.of(arena(), value);
    }
    default Wrap.FloatPtr floatPtr(float value){
        return Wrap.FloatPtr.of(arena(), value);
    }
    default Wrap.IntArr ofInts(int ...values){
        return  Wrap.IntArr.of(arena(), values);
    }
    default Wrap.FloatArr ofFloats(float ...values){
        return  Wrap.FloatArr.of(arena(), values);
    }
    default Wrap.CStrPtr cstr(MemorySegment segment){
        return Wrap.CStrPtr.of( segment);
    }

    default Wrap.CStrPtr cstr(String s){
        return Wrap.CStrPtr.of(arena(), s);
    }
    default Wrap.CStrPtr cstr(long size){
        return Wrap.CStrPtr.of(arena(), (int)size);
    }
    default Wrap.CStrPtr cstr(int size){
        return Wrap.CStrPtr.of(arena(), size);
    }
    default Wrap.PtrArr ptrArr(MemorySegment ... memorySegments) {
        return Wrap.PtrArr.of(arena(), memorySegments);
    }
    default Wrap.PtrArr ptrArr(Wrap.Ptr ...ptrs) {
        return Wrap.PtrArr.of(arena(), ptrs);
    }
    default Wrap.PtrArr ptrArr(int len) {
        return Wrap.PtrArr.of(arena(), len);
    }

    default Wrap.PtrArr ptrArr(String ...strings) {
        return Wrap.PtrArr.of(arena(), strings);
    }
}
