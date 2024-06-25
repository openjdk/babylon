/*
 * Copyright (c) 2024 Intel Corporation. All rights reserved.
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


import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.SegmentMapper;

import java.lang.foreign.Arena;

public class SchemaLayoutTest {

    public static void main(String[] args) {
        BufferAllocator bufferAllocator= new BufferAllocator() {
            public <T extends Buffer> T allocate(SegmentMapper<T> s) {return s.allocate(Arena.global());}
        };
        hat.buffer.S32Array os32  = hat.buffer.S32Array.create(bufferAllocator,100);
        System.out.println("Layout from hat S32Array "+ Buffer.getLayout(os32));

        var s32Array = S32Array.schema.allocate(bufferAllocator, 100);
        int s23ArrayLen = s32Array.length();
        System.out.println("Layout from schema "+Buffer.getLayout(s32Array));
        ResultTable.schema.toText(t->System.out.print(t));

        var resultTable = ResultTable.schema.allocate(bufferAllocator, 100);
        int resultTableLen = resultTable.length();
        System.out.println(Buffer.getLayout(resultTable));


        Cascade.schema.toText(t->System.out.print(t));
        var cascadelayout = Cascade.schema.layout(10,10,10);
        System.out.println(cascadelayout);
        var cascade = Cascade.schema.allocate(bufferAllocator,10,10,10);

        System.out.println(Buffer.getLayout(cascade));
        //var layout = Cascade.schema.field.layout();

   //     System.out.println(layout);
    }
}

