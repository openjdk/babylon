
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

    static void reset(int markedVec2) {
        F32Vec2.f32Vec2Pool.count = markedVec2;
    }

    class F32Vec2Pool extends FloatPool<F32Vec2Pool> {
        static public int X = 0;
        static public int Y = 1;
        record Idx(F32Vec2Pool pool, int idx) implements Pool.Idx<F32Vec2Pool>{

        }
        F32Vec2Pool() {
            super(2,12800);
        }
        @Override
        Idx idx(int idx) {
            return new Idx(this, idx);
        }
    }
    F32Vec2Pool f32Vec2Pool = new F32Vec2Pool();

    record F32Vec2Impl(int id, float x, float y) implements F32Vec2{ }
     static F32Vec2Impl of(float x, float y) {
        f32Vec2Pool.entries[f32Vec2Pool.count * f32Vec2Pool.stride + F32Vec2Pool.X] = x;
        f32Vec2Pool.entries[f32Vec2Pool.count * f32Vec2Pool.stride + F32Vec2Pool.Y] = y;
        return  new F32Vec2Impl(f32Vec2Pool.count++, x,y);
    }
}
