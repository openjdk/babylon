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


import view.f32.factories.Factory16;
import view.f32.pool.F32x4x4Pool;

public interface F32x4x4 {
    float x0y0();

    float x1y0();

    float x2y0();

    float x3y0();

    float x0y1();

    float x1y1();

    float x2y1();

    float x3y1();

    float x0y2();

    float x1y2();

    float x2y2();

    float x3y2();

    float x0y3();

    float x1y3();

    float x2y3();

    float x3y3();

    default String asString() {

        return String.format("""
                        |%5.2f, %5.2f, %5.2f, %5.2f|
                        |%5.2f, %5.2f, %5.2f, %5.2f|
                        |%5.2f, %5.2f, %5.2f, %5.2f|
                        |%5.2f, %5.2f, %5.2f, %5.2f|
                        """,
                x0y0(), x1y0(), x2y0(), x3y0(),
                x0y1(), x1y1(), x2y1(), x3y1(),
                x0y2(), x1y2(), x2y2(), x3y2(),
                x0y3(), x1y3(), x2y3(), x3y3());
    }



    @FunctionalInterface
    interface Factory extends Factory16<
            Float,Float,Float,Float,
            Float,Float,Float,Float,
            Float,Float,Float,Float,
            Float,Float,Float,Float,F32x4x4>{

    }


}
