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
package hat.types;

import jdk.incubator.code.Reflect;
import jdk.incubator.code.dialect.java.JavaType;
import optkl.IfaceValue;

import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

// This is immutable
public interface ivec2 extends  IfaceValue.vec {
    Shape shape = Shape.of( JavaType.INT,2);
    int x();
    int y();
    AtomicInteger count = new AtomicInteger(0);
    AtomicBoolean collect = new AtomicBoolean(false);
    //   if (collect.get())count.getAndIncrement();

    // A mutable form needed for interface mapping.
    interface Field extends ivec2 {
        @Reflect default void schema(){x();y();}
        void x(int x);
        void y(int y);
        default ivec2 of(int x, int y){
            x(x);y(y);
            return this;
        }
        default ivec2 of(ivec2 ivec2){
            of(ivec2.x(),ivec2.y());
            return this;
        }
    }


     static ivec2 ivec2(int x, int y) {
        record Impl(int x, int y) implements ivec2 { }
      //   if (collect.get())count.getAndIncrement();
        return new Impl(x, y);
    }
    static ivec2 ivec2(ivec2 ivec2) {return ivec2(ivec2.x(), ivec2.y());}
    static ivec2 ivec2(int scalar) {return ivec2(scalar,scalar);}

}
