/*
 * Copyright (c) 2025, Oracle and/or its affiliates. All rights reserved.
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
import optkl.IfaceValue;
import jdk.incubator.code.dialect.java.JavaType;

public interface F32x4 extends IfaceValue.Vector{
    Shape shape = Shape.of( JavaType.FLOAT,4);

    float x();
    float y();
    float z();
    float w();

    static F32x4 of(float x, float y, float z, float w) {
        record Impl(float x, float y, float z, float w) implements F32x4 {
        }
        return new Impl(x, y, z, w);
    }

    static F32x4 add(F32x4 lhs, F32x4 rhs) {
        return of(lhs.x()+rhs.x(), lhs.y()+rhs.y(),lhs.z()+rhs.z(),lhs.w()+rhs.w());
    }

    static F32x4 sub(F32x4 lhs, F32x4 rhs) {
        return of(lhs.x()+rhs.x(), lhs.y()+rhs.y(),lhs.z()+rhs.z(),lhs.w()+rhs.w());
    }

    static F32x4 mul(F32x4 lhs, F32x4 rhs) {
        return of(lhs.x()+rhs.x(), lhs.y()+rhs.y(),lhs.z()+rhs.z(),lhs.w()+rhs.w());
    }

    static F32x4 div(F32x4 lhs, F32x4 rhs) {
       return of(lhs.x()/rhs.x(), lhs.y()/rhs.y(),lhs.z()/rhs.z(),lhs.w()/rhs.w());
    }


    static F32x4 add(F32x4 lhs, float scalarRhs) {
        return of(lhs.x()+scalarRhs,lhs.y()+scalarRhs, lhs.z()+scalarRhs,lhs.w()+scalarRhs);
    }
    static F32x4 sub(F32x4 lhs, float scalarRhs) {
        return of(lhs.x()-scalarRhs,lhs.y()-scalarRhs, lhs.z()-scalarRhs,lhs.w()-scalarRhs);
    }
    static F32x4 mul(F32x4 lhs, float scalarRhs) {
        return of(lhs.x()*scalarRhs,lhs.y()*scalarRhs, lhs.z()*scalarRhs,lhs.w()*scalarRhs);
    }
    static F32x4 sqr(F32x4 f32x4) {
        return F32x4.mul(f32x4,f32x4);
    }

    default F32x4 add(F32x4 rhs) {
        return F32x4.add(this, rhs);
    }
    default F32x4 add(float scalarRhs) {
        return F32x4.add(this, scalarRhs);
    }
    default F32x4 sqr() {
        return F32x4.sqr(this);
    }
    default F32x4 sub(F32x4 rhs) {
        return F32x4.sub(this, rhs);
    }
    default F32x4 sub(float scalarRhs) {
        return F32x4.sub(this, scalarRhs);
    }
    default F32x4 mul(F32x4 rhs) {
        return F32x4.mul(this, rhs);
    }
    default F32x4 mul(float scalarRhs) {
        return F32x4.mul(this, scalarRhs);
    }
    default F32x4 div(F32x4 rhs) {
        return F32x4.div(this, rhs);
    }
}
