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

import java.util.function.BiFunction;

public interface Float4 extends HatVector {

    float x();
    float y();
    float z();
    float w();
    void x(float x);
    void y(float y);
    void z(float z);
    void w(float w);

    record Float4Impl(float x, float y, float z, float w) implements Float4 {
        @Override
        public void x(float x) {}

        @Override
        public void y(float y) {}

        @Override
        public void z(float z) {}

        @Override
        public void w(float w) {}
    }

    static Float4 of(float x, float y, float z, float w) {
        return new Float4Impl(x, y, z, w);
    }

    default Float4 lanewise(Float4 other, BiFunction<Float, Float, Float> f) {
        float[] backA = this.toArray();
        float[] backB = other.toArray();
        float[] backC = new float[backA.length];
        for (int j = 0; j < backA.length; j++) {
            var r = f.apply(backA[j],  backB[j]);
            backC[j] = r;
        }
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

    default float[] toArray() {
        return new float[] { x(), y(), z(), w() };
    }
}
