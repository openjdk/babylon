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
import optkl.IfaceValue.Matrix;

public interface mat2 extends Matrix{
    Shape shape = IfaceValue.Matrix.Shape.of( JavaType.FLOAT,2,2);
    float _00();

    float _01();

    float _10();

    float _11();

    // A mutable variant needed for interface mapping to memory segments
    interface Field extends mat2 {
        @Reflect
        default void schema(){_00();_01();_10();_11();}
        void _00(float _00);
        void _01(float _01);
        void _10(float _10);
        void _11(float _11);
        default mat2 of(float _00, float _01, float _10, float _11){
            _00(_00);_01(_01);_10(_10);_11(_11);
            return this;
        }
        default mat2 of(mat2 mat2){
            of(mat2._00(),mat2._01(),mat2._10(),mat2._11());
            return this;
        }
    }

    static mat2 mat2(float _00, float _01, float _10, float _11) {
        record Impl(float _00, float _01, float _10, float _11) implements mat2 {
        }
        return new Impl(_00, _01, _10, _11);
    }
    static mat2 mat2(mat2 mat2) {return mat2(mat2._00(), mat2._01(), mat2._10(), mat2._11());}
    static mat2 mat2(float scalar) {return mat2(scalar,scalar,scalar,scalar);}

    static mat2 add(mat2 l, mat2 r) {return mat2(l._00()+r._00(),l._01()+r._01(), l._10()+r._10(),l._11()+r._11());}
    default mat2 add(mat2 rhs){return add(this,rhs);}
    default mat2 add(float scalar){return add(this,mat2(scalar));}

    static mat2 sub(mat2 l, mat2 r) {return mat2(l._00()-r._00(),l._01()-r._01(), l._10()-r._10(),l._11()-r._11());}
    default mat2 sub(float scalar) {return sub(this, mat2(scalar));}
    default mat2 sub(mat2 rhs){return sub(this,rhs);}

    static mat2 mul(mat2 l, mat2 r) {return mat2(l._00()*r._00(),l._01()*r._01(), l._10()*r._10(),l._11()*r._11());}
    static vec2 mul(mat2 l, vec2 r) {return vec2.vec2(
            l._00()*r.x()+l._01()*r.y(),
            l._10()*r.x()+l._11()*r.y()
    );}

    default mat2 mul(float scalar) {return mul(this, mat2(scalar));}
    default mat2 mul(mat2 rhs){return mul(this,rhs);}

    static mat2 div(mat2 l, mat2 r) {return mat2(l._00()/r._00(),l._01()/r._01(), l._10()/r._10(),l._11()/r._11());}
    default mat2 div(float scalar) {return div(this, mat2(scalar));}
    default mat2 div(mat2 rhs){return div(this,rhs);}

}
