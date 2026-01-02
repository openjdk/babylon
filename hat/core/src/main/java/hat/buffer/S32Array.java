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

import jdk.incubator.code.Reflect;
import optkl.util.carriers.CommonCarrier;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;

import java.lang.foreign.MemorySegment;
import java.util.function.Function;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public interface S32Array extends Buffer {
    @Reflect default void  schema(){array(length());}
    int length();
    int array(long idx);
    void array(long idx, int i);

    long ARRAY_OFFSET = JAVA_INT.byteSize();
    Schema<S32Array> schema = Schema.of(S32Array.class);

    @Reflect static S32Array create(CommonCarrier cc, int length){
        return schema.allocate(cc, length);
    }
    @Reflect default S32Array fill(Function<Integer, Integer> filler) {
        for (int i = 0; i < length(); i++) {
            array(i, filler.apply(i));
        }
        return this;
    }
    @Reflect static S32Array create(CommonCarrier cc, int length, Function<Integer,Integer> filler){
        return create(cc,length).fill(filler);
    }
    static S32Array createFrom(CommonCarrier cc, int[] arr){
        return create( cc, arr.length).copyfrom(arr);
    }
    @Reflect default S32Array copyfrom(int[] ints) {
        MemorySegment.copy(ints, 0, MappableIface.getMemorySegment(this), JAVA_INT, ARRAY_OFFSET, length());
        return this;
    }
    @Reflect default int[] copyTo(int[] ints) {
        MemorySegment.copy(MappableIface.getMemorySegment(this), JAVA_INT, ARRAY_OFFSET, ints, 0, length());
        return ints;
    }

    @Reflect default int[] arrayView() {
        return this.copyTo(new int[this.length()]);
    }
}
