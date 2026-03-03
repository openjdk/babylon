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

public interface mat3 extends IfaceValue.mat {
    Shape shape = IfaceValue.mat.Shape.of( JavaType.FLOAT,3,3);
    float _00();
    float _01();
    float _02();
    float _10();
    float _11();
    float _12();
    float _20();
    float _21();
    float _22();

    AtomicInteger count = new AtomicInteger(0);
    AtomicBoolean collect = new AtomicBoolean(false);
    //   if (collect.get())count.getAndIncrement();
    // A mutable variant needed for interface mapping
    interface Field extends mat3 {
        @Reflect
        default void schema(){_00();_01();_02();_10();_11();_12();_20();_21();_22();}
        void _00(float _00);
        void _01(float _01);
        void _02(float _02);
        void _10(float _10);
        void _11(float _11);
        void _12(float _12);
        void _20(float _20);
        void _21(float _21);
        void _22(float _22);
        default mat3 of(float _00, float _01, float _02, float _10, float _11, float _12, float _20, float _21, float _22) {
            _00(_00);_01(_01);_02(_02);_10(_10);_11(_11);_12(_12);_20(_20);_21(_21);_22(_22);
            return this;
        }
        default mat3 of(mat3 mat3){
            of(mat3._00(),mat3._01(),mat3._02(),mat3._10(),mat3._11(),mat3._12(),mat3._20(),mat3._21(),mat3._22() );
            return this;
        }
    }




    static mat3 mat3(float _00, float _01, float _02,float _10, float _11, float _12, float _20, float _21, float _22) {
        record Impl(float _00, float _01, float _02, float _10, float _11, float _12, float _20, float _21, float _22) implements mat3 {
        }
       // if (collect.get())count.getAndIncrement();
        return new Impl(_00, _01,_02, _10, _11, _12, _20, _21, _22);
    }
    static mat3 mat3(mat3 mat3) {return mat3(mat3._00(), mat3._01(), mat3._02(), mat3._10(), mat3._11(), mat3._12(), mat3._20(), mat3._21(), mat3._22());}
    static mat3 mat3(float scalar) {return mat3(scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar,scalar);}

    static mat3 add(mat3 l, mat3 r) {return mat3(
            l._00()+r._00(),l._01()+r._01(),l._02()+r._02(),
            l._10()+r._10(),l._11()+r._11(),l._12()+r._12(),
            l._20()+r._20(),l._21()+r._21(),l._22()+r._22()
    );}
    //default mat3 add(mat3 rhs){return add(this,rhs);}
    //default mat3 add(float scalar){return add(this,mat3(scalar));}

    static mat3 sub(mat3 l, mat3 r) {return mat3(
            l._00()-r._00(),l._01()-r._01(),l._02()-r._02(),
            l._10()-r._10(),l._11()-r._11(),l._12()-r._12(),
            l._20()-r._20(),l._21()-r._21(),l._22()-r._22()
    );}
    //default mat3 sub(float scalar) {return sub(this, mat3(scalar));}
    //default mat3 sub(mat3 rhs){return sub(this,rhs);}

    static mat3 mul(mat3 l, mat3 r) {return mat3(
            l._00()*r._00(),l._01()*r._01(),l._02()*r._02(),
            l._10()*r._10(),l._11()*r._11(),l._12()*r._12(),
            l._20()*r._20(),l._21()*r._21(),l._22()*r._22()
    );}
   // default mat3 mul(float scalar) {return mul(this, mat3(scalar));}
   // default mat3 mul(mat3 rhs){return mul(this,rhs);}

    static mat3 div(mat3 l, mat3 r) {return mat3(
            l._00()/r._00(),l._01()/r._01(),l._02()/r._02(),
            l._10()/r._10(),l._11()/r._11(),l._12()/r._12(),
            l._20()/r._20(),l._21()/r._21(),l._22()/r._22()
    );}
   // default mat3 div(float scalar) {return div(this, mat3(scalar));}
   // default mat3 div(mat3 rhs){return div(this,rhs);}

}
