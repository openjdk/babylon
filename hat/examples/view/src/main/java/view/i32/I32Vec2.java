
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
package view.i32;

public class I32Vec2 {
    public static final int SIZE = 2;
    public  static final int MAX = 800;
    public  static final int X = 0;
    public static final int Y = 1;

    public static int count = 0;
    public static int entries[] = new int[MAX * SIZE];

    static int createVec2(int x, int y) {
        entries[count * SIZE + X] = x;
        entries[count * SIZE + Y] = y;
        return count++;
    }


    static int mulScaler(int i, int s) {
        i *= SIZE;
        return createVec2(entries[i + X] * s, entries[i + Y] * s);
    }

    static int addScaler(int i, int s) {
        i *= SIZE;
        return createVec2(entries[i + X] + s, entries[i + Y] + s);
    }

    static int divScaler(int i, int s) {
        i *= SIZE;
        return createVec2(entries[i + X] / s, entries[i + Y] / s);
    }

    static int addVec2(int lhs, int rhs) {
        lhs *= SIZE;
        rhs *= SIZE;
        return createVec2(entries[lhs + X] + entries[rhs + X], entries[lhs + Y] + entries[rhs + Y]);
    }

    static int subVec2(int lhs, int rhs) {
        lhs *= SIZE;
        rhs *= SIZE;
        return createVec2(entries[lhs + X] - entries[rhs + X], entries[lhs + Y] - entries[rhs + Y]);
    }


    static float dotProd(int lhs, int rhs) {
        lhs *= SIZE;
        rhs *= SIZE;
        return entries[lhs + X] * entries[rhs + X] + entries[lhs + Y] * entries[rhs + Y];
    }

    static String asString(int i) {
        i *= SIZE;
        return entries[i + X] + "," + entries[i + Y];
    }
}
