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
package hat.buffer;

import hat.Accelerator;
import hat.ifacemapper.Schema;

import java.lang.foreign.MemorySegment;
import java.util.function.Function;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface S32LocalArray extends Buffer {

    int length();
    int array(long idx);
    void array(long idx, int i);

    Schema<S32LocalArray> schema = Schema.of(S32LocalArray.class, s32Array->s32Array
            .arrayLen("length").array("array"));

    static S32LocalArray create(Accelerator accelerator, int length){
        return schema.allocate(accelerator, length);
    }

    static S32LocalArray create(Accelerator accelerator, int length, Function<Integer,Integer> filler){
        return schema.allocate(accelerator, length).fill(filler);
    }

    static S32LocalArray createFrom(Accelerator accelerator, int[] arr) {
        return create( accelerator, arr.length).copyFrom(arr);
    }

    default S32LocalArray copyFrom(int[] ints) {
        MemorySegment.copy(ints, 0, Buffer.getMemorySegment(this), JAVA_INT, 4, length());
        return this;
    }

    default S32LocalArray copyTo(int[] ints) {
        MemorySegment.copy(Buffer.getMemorySegment(this), JAVA_INT, 4, ints, 0, length());
        return this;
    }

    default S32LocalArray fill(Function<Integer, Integer> filler) {
        for (int i = 0; i < length(); i++) {
            array(i, filler.apply(i));
        }
        return this;
    }
}
