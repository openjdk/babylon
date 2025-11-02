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
package hat.buffer;

import hat.types._V4;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.util.function.BiFunction;
import java.util.stream.IntStream;

public interface Float4 extends _V4 {

    float x();
    float y();
    float z();
    float w();

    @CodeReflection
    @Override
    default PrimitiveType type() {
        return JavaType.FLOAT;
    }

    record MutableImpl(float x, float y, float z, float w) implements Float4 {
        public void x(float x) {}
        public void y(float y) {}
        public void z(float z) {}
        public void w(float w) {}
    }

    record ImmutableImpl(float x, float y, float z, float w) implements Float4 {
    }

    /**
     * Make a Mutable implementation (for the device side - e.g., the GPU) from an immutable implementation.
     *
     * @param float4
     * @return {@link Float4.MutableImpl}
     */
    static Float4.MutableImpl makeMutable(Float4 float4) {
        return new MutableImpl(float4.x(), float4.y(), float4.z(), float4.w());
    }

    static Float4 of(float x, float y, float z, float w) {
        return new ImmutableImpl(x, y, z, w);
    }

    // Not implemented for the GPU yet
    default Float4 lanewise(Float4 other, BiFunction<Float, Float, Float> f) {
        float[] backA = this.toArray();
        float[] backB = other.toArray();
        float[] backC = new float[backA.length];
        IntStream.range(0, backA.length).forEach(j -> {
            backC[j] = f.apply(backA[j], backB[j]);
        });
        return of(backC[0], backC[1], backC[2], backC[3]);
    }

    static Float4 add(Float4 vA, Float4 vB) {
        return vA.lanewise(vB, Float::sum);
    }

    static Float4 sub(Float4 vA, Float4 vB) {
        return vA.lanewise(vB, (a, b) -> a - b);
    }

    static Float4 mul(Float4 vA, Float4 vB) {
        return vA.lanewise(vB, (a, b) -> a * b);
    }

    static Float4 div(Float4 vA, Float4 vB) {
        return vA.lanewise(vB, (a, b) -> a / b);
    }

    default Float4 add(Float4 vb) {
        return Float4.add(this, vb);
    }

    default Float4 sub(Float4 vb) {
        return Float4.sub(this, vb);
    }

    default Float4 mul(Float4 vb) {
        return Float4.mul(this, vb);
    }

    default Float4 div(Float4 vb) {
        return Float4.div(this, vb);
    }

    // Not implemented for the GPU yet
    default float[] toArray() {
        return new float[] { x(), y(), z(), w() };
    }
}
