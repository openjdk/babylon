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

import hat.Accelerator;
import hat.backend.DebugBackend;
import hat.buffer.S32Array2D;
import hat.ifacemapper.BoundSchema;
import hat.buffer.Buffer;

import java.lang.foreign.GroupLayout;
import java.lang.foreign.MemoryLayout;
import java.lang.invoke.MethodHandles;

public class S32ArrayTest implements Buffer {

    public static void main(String[] args) {
        Accelerator accelerator = new Accelerator(MethodHandles.lookup(),new DebugBackend());

        hat.buffer.S32Array s32Array  = hat.buffer.S32Array.create(accelerator, 100);
        GroupLayout groupLayout = (GroupLayout) Buffer.getLayout(s32Array);
        System.out.println("Layout from buffer "+groupLayout);
        BoundSchema<?> boundSchema = Buffer.getBoundSchema(s32Array);
        System.out.println("BoundSchema from buffer  "+boundSchema);

        BoundSchema.FieldLayout<?> fieldLayout =  boundSchema.rootBoundSchemaNode().getName("array");
        long arrayOffset = fieldLayout.offset();
        MemoryLayout layaout = fieldLayout.layout();
        if (fieldLayout instanceof BoundSchema.ArrayFieldLayout arrayFieldLayout){
            System.out.println("isArray");
            arrayFieldLayout.offset(0);
        }
        S32Array2D.schema.toText(t->System.out.print(t));
    }

}
