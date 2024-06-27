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
package experiments;

import hat.Schema;
import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.Arena;
import java.lang.foreign.GroupLayout;

public interface ResultTable extends Buffer{
    interface Result extends Buffer.StructChild {
        float x();
        void x(float x);
        float y();
        void y(float y);
        float width();
        void width(float width);
        float height();
        void height(float height);
    }
    void atomicResultTableCount(int count);
    int atomicResultTableCount();
    int length();
    Result result(long idx);
    Schema<ResultTable> schema = Schema.of(ResultTable.class, resultTable->resultTable
            .atomic("atomicResultTableCount")
            .arrayLen("length").array("result", array->array.fields("x","y","width","height"))
    );

    public static void main(String[] args) {
        BufferAllocator bufferAllocator = new BufferAllocator() {
            public <T extends Buffer> T allocate(SegmentMapper<T> s) {
                return s.allocate(Arena.global());
            }
        };
        ResultTable.schema.toText(t->System.out.print(t));
        System.out.println();
        Schema.BoundLayout boundLayout = ResultTable.schema.collectLayouts(1000);
        System.out.println(boundLayout.groupLayout);
        System.out.println("[i4(length)i4(atomicResultTableCount)[1000:[f4(x)f4(y)f4(width)f4(height)](Result)](result)](ResultTable)");
       // var boundSchema = ResultTable.schema.allocate(bufferAllocator, 100);

      //  var resultTable = ResultTable.schema.allocate(bufferAllocator, 100).instance;
      //  int resultTableLen = resultTable.length();
      //  System.out.println(Buffer.getLayout(resultTable));
    }

}
