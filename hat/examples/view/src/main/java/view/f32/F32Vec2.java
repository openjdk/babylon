
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
package view.f32;

import java.util.ArrayList;
import java.util.List;

public interface F32Vec2 {
    float x();
    float y();

    static void reset(int markedVec2) {
        F32Vec2.f32Vec2Pool.count = markedVec2;
    }

    class F32Vec2Pool extends FloatPool<F32Vec2.F32Vec2Pool> {
        public static int X = 0;
        public static int Y = 1;
        public record Idx(F32Vec2Pool pool, int idx) implements Pool.Idx<F32Vec2.F32Vec2Pool>, F32Vec2 {
            private int xIdx() {
                return pool.stride * idx + X;
            }

            @Override
            public float x() {
                return pool.entries[xIdx()];
            }

            private int yIdx() {
                return pool.stride * idx + Y;
            }

            @Override
            public float y() {
                return pool.entries[yIdx()];
            }

        }

        F32Vec2Pool(int max) {
            super(2, max);
        }

        @Override
        F32Vec2.F32Vec2Pool.Idx idx(int idx) {
            return new F32Vec2.F32Vec2Pool.Idx(this, idx);
        }

        public F32Vec2.F32Vec2Pool.Idx of(float x, float y) {
            F32Vec2.F32Vec2Pool.Idx i = idx(count++);
            entries[i.xIdx()] = x;
            entries[i.yIdx()] = y;
            return i;
        }
    }
    F32Vec2Pool f32Vec2Pool = new F32Vec2Pool(12000);
}
