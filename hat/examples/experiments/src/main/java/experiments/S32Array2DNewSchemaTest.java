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
import hat.backend.Backend;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;
import optkl.util.carriers.CommonCarrier;

import java.lang.foreign.GroupLayout;
import java.lang.invoke.MethodHandles;

public class S32Array2DNewSchemaTest implements Buffer {
    public interface S32Arr2D extends Buffer {
         @Reflect default void schema(){array(width()*height());};
         Schema<S32Arr2D> schema = Schema.of(S32Arr2D.class);
        // Schema<S32Arr2D> oldSchema = Schema.of(S32Arr2D.class, s32Array -> s32Array.arrayLen("width", "height").array("array"));
        int width();
        int height();
        int array(long idx);
        void array(long idx, int i);
        static S32Arr2D create(CommonCarrier cc, int width, int height) {
            return schema.allocate(cc, width, height);
        }
    }
    public static void main(String[] args) {
        var lookup = MethodHandles.lookup();
        Accelerator accelerator = new Accelerator(lookup, Backend.FIRST);

        S32Arr2D s32Arr2D  = S32Arr2D.create(accelerator, 100,200);
        GroupLayout groupLayout = (GroupLayout) MappableIface.getLayout(s32Arr2D);
        System.out.println("Layout from buffer "+groupLayout);

        if ( MappableIface.getBoundSchema(s32Arr2D)
                .rootBoundSchemaNode()
                .getName("array") instanceof BoundSchema.ArrayFieldLayout arrayFieldLayout){
            arrayFieldLayout.elementOffset(0);
            arrayFieldLayout.elementLayout(0);
            if (arrayFieldLayout instanceof BoundSchema.BoundArrayFieldLayout boundArrayFieldLayout){
                boundArrayFieldLayout.dimFields.forEach(dimLayout->{
                    System.out.println(dimLayout.field.name + " offset=@"+dimLayout.offset());
                });
            }
        }
        System.out.println("-----");
        S32Arr2D.schema.toText(System.out::print);
    }

}
