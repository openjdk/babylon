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
package view.f32.pool;

import view.f32.F32x2;
import view.f32.factories.Factory2;

public class F32x2Pool extends F32Pool<F32x2,F32x2Pool> implements F32x2.Factory {
    public static int X = 0;
    public static int Y = 1;

    public record PoolEntry(F32x2Pool pool, int idx) implements Pool.PoolEntry<F32x2,F32x2Pool>, F32x2 {
        private int xIdx() {
            return pool.stride * idx + X;
        }

        @Override
        public float x() {
            return pool.floatEntries[xIdx()];
        }

        private int yIdx() {
            return pool.stride * idx + Y;
        }

        @Override
        public float y() {
            return pool.floatEntries[yIdx()];
        }

    }

    public F32x2Pool(int max) {
        super(2, max);
    }

    @Override
    public F32x2 entry(int idx) {
        return (F32x2) new PoolEntry(this, idx);
    }

    @Override
    public F32x2 of(Float x, Float y) {
        PoolEntry i = (PoolEntry) entry(count++);
        floatEntries[i.xIdx()] = x;
        floatEntries[i.yIdx()] = y;
        return i;
    }
}
