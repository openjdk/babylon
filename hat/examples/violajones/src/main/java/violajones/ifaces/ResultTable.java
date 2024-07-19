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
package violajones.ifaces;

import hat.Accelerator;
import hat.buffer.After;
import hat.buffer.Atomic;
import hat.buffer.BoundBy;
import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;

public interface ResultTable extends Buffer {

    interface Result extends Struct {
        float x();
        float y();
        float width();
        float height();
        void x(float x);
        void y(float y);
        void width(float width);
        void height(float height);
    }
    @Atomic
    void atomicResultTableCount(int atomicResultTableCount);
    int atomicResultTableCount();

    @After("atomicResultTableCount")
    int length();
    void length(int length);

    @BoundBy("length")
    Result result(long idx);


    default int atomicResultTableCountInc() {
        int index = atomicResultTableCount();
        atomicResultTableCount(index + 1);
        return index;
    }

    Schema<ResultTable> schema = Schema.of(ResultTable.class, resultTable->resultTable
            .atomic("atomicResultTableCount")
            .arrayLen("length")
            .array("result", array->array
                    .fields("x", "y", "width", "height")
            )
    );

    static ResultTable create(MethodHandles.Lookup lookup, BufferAllocator bufferAllocator,int length){
        var instance = schema.allocate(lookup,bufferAllocator,length);
        instance.length(length);
        instance.atomicResultTableCount(0);
        return instance;
    }

    static ResultTable create(Accelerator accelerator, int length){
       return create(accelerator.lookup, accelerator,length);
    }
}
