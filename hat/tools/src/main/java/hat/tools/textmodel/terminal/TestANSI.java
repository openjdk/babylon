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
package hat.tools.textmodel.terminal;
public class TestANSI {
    static void main(String[] args) {
        var colorizer = ANSI.of(System.out);
        var image = colorizer.img(200, 200);
        for (int frame = 0; frame < 40; frame++) {
            image.clean().home();
            image.clean().home();

            double realMin = -2.0;
            double realMax = 1.0;
            double imagMin = -1.5;
            double imagMax = 1.5;
            int maxIterations = frame / 2; // Maximum number of iterations for each point
            double escapeRadiusSquared = 4.0; // If |z|^2 exceeds this, the point escapes

            for (int y = 0; y < image.height; y++) {
                for (int x = 0; x < image.width; x++) {
                    double cReal = realMin + (double) x / image.width * (realMax - realMin);
                    double cImag = imagMin + (double) y / image.height * (imagMax - imagMin);
                    double zReal = 0.0;
                    double zImag = 0.0;
                    int iteration = 0;
                    while (zReal * zReal + zImag * zImag < escapeRadiusSquared && iteration < maxIterations) {
                        double nextZReal = zReal * zReal - zImag * zImag + cReal;
                        double nextZImag = 2 * zReal * zImag + cImag;
                        zReal = nextZReal;
                        zImag = nextZImag;
                        iteration++;
                    }
                    if (iteration == maxIterations) {
                        image.set(x, y);
                    }
                }
            }
            image.map().write();
            image.delay(100);
        }
    }
}


