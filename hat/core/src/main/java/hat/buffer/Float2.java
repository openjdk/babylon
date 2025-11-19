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

import hat.types._V2;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.dialect.java.JavaType;
import jdk.incubator.code.dialect.java.PrimitiveType;

import java.util.function.BiFunction;
import java.util.stream.IntStream;

public interface Float2 extends _V2 {

    float x();
    float y();

    @CodeReflection
    @Override
    default PrimitiveType type() {
        return JavaType.FLOAT;
    }

    record MutableImpl(float x, float y) implements Float2 {
        public void x(float x) {}
        public void y(float y) {}
    }

    record ImmutableImpl(float x, float y) implements Float2 {
    }

    /**
     * Make a Mutable implementation (for the device side - e.g., the GPU) from an immutable implementation.
     *
     * @param {@link Float2}
     * @return {@link Float2.MutableImpl}
     */
    static Float2.MutableImpl makeMutable(Float2 float2) {
        return new MutableImpl(float2.x(), float2.y());
    }

    static Float2 of(float x, float y) {
        return new ImmutableImpl(x, y);
    }

    // Not implemented for the GPU yet
    default Float2 lanewise(Float2 other, BiFunction<Float, Float, Float> f) {
        float[] backA = this.toArray();
        float[] backB = other.toArray();
        float[] backC = new float[backA.length];
        IntStream.range(0, backA.length).forEach(j -> {
            backC[j] = f.apply(backA[j], backB[j]);
        });
        return of(backC[0], backC[1]);
    }

    static Float2 add(Float2 vA, Float2 vB) {
        return vA.lanewise(vB, Float::sum);
    }

    static Float2 sub(Float2 vA, Float2 vB) {
        return vA.lanewise(vB, (a, b) -> a - b);
    }

    static Float2 mul(Float2 vA, Float2 vB) {
        return vA.lanewise(vB, (a, b) -> a * b);
    }

    static Float2 div(Float2 vA, Float2 vB) {
        return vA.lanewise(vB, (a, b) -> a / b);
    }

    default Float2 add(Float2 vb) {
        return Float2.add(this, vb);
    }

    default Float2 sub(Float2 vb) {
        return Float2.sub(this, vb);
    }

    default Float2 mul(Float2 vb) {
        return Float2.mul(this, vb);
    }

    default Float2 div(Float2 vb) {
        return Float2.div(this, vb);
    }

    // Not implemented for the GPU yet
    default float[] toArray() {
        return new float[] { x(), y()};
    }
}
