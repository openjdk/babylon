
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

public class F32Vec2 {

    public  static final int X = 0;
    public static final int Y = 1;
    public static class Pool extends FloatPool{
        Pool( int max) {
            super(2, max);
        }
    }
public static     Pool pool = new Pool(12800);

    public static int createVec2(float x, float y) {
        pool.entries[pool.count * pool.stride + X] = x;
        pool.entries[pool.count * pool.stride + Y] = y;

        return pool.count++;
    }

    static int mulScaler(int i, float s) {
        i *= pool.stride;
        return createVec2(pool.entries[i + X] * s, pool.entries[i + Y] * s);
    }

    static int addScaler(int i, float s) {
        i *= pool.stride;
        return createVec2(pool.entries[i + X] + s, pool.entries[i + Y] + s);
    }

    static int divScaler(int i, float s) {
        i *= pool.stride;
        return createVec2(pool.entries[i + X] / s, pool.entries[i + Y] / s);
    }

    static int addVec2(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return createVec2(pool.entries[lhs + X] + pool.entries[rhs + X], pool.entries[lhs + Y] + pool.entries[rhs + Y]);
    }

    static int subVec2(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return createVec2(pool.entries[lhs + X] - pool.entries[rhs + X], pool.entries[lhs + Y] - pool.entries[rhs + Y]);
    }


    static float dotProd(int lhs, int rhs) {
        lhs *= pool.stride;
        rhs *= pool.stride;
        return pool.entries[lhs + X] * pool.entries[rhs + X] + pool.entries[lhs + Y] * pool.entries[rhs + Y];
    }

    static String asString(int i) {
        i *= pool.stride;
        return pool.entries[i + X] + "," + pool.entries[i + Y];
    }
}
