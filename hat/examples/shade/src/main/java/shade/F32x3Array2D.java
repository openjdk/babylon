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
package shade;

import hat.Accelerator;
import hat.buffer.S08x3RGBImage;
import hat.buffer.S32Array2D;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.BoundSchema;
import optkl.ifacemapper.Buffer;
import optkl.ifacemapper.MappableIface;
import optkl.ifacemapper.Schema;

public interface F32x3Array2D extends Buffer {
 //   @Reflect
   // default void schema(){array(width()*height()*depth());};  we need to accept more than one *....
   // Schema<F32x3Array2D> schema = Schema.of(F32x3Array2D.class);

    long width();
    long height();

    float array(long idx);
    void array(long idx, float f);
    Schema<F32x3Array2D> schema = Schema.of(F32x3Array2D.class, s -> s
            .arrayLen("width", "height").stride(3).array("array")
    );
    static F32x3Array2D create(Accelerator accelerator, int width, int height) {
        return  BoundSchema.of(accelerator ,schema,width,height,3).allocate();
    }


    default float r(int x, int y){
        return array(3*(y*width()+x+0));
    }
    default float g(int x, int y){
        return array(3*(y*width()+x+1));
    }
    default float b(int x, int y){
        return array(3*(y*width()+x+2));
    }
    default void r(int x, int y, float r){
        array(3*(y*width()+x+0),r);
    }
    default void g(int x, int y, float g){
        array(3*(y*width()+x+1),g);
    }
    default void b(int x, int y, float b){
        array(3*(y*width()+x+2),b);
    }

    default void clear(){
        MappableIface.getMemorySegment(this).fill((byte)0);
    }

}
