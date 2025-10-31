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

/*
 *  Based on mesh descriptions found here
 *      https://6502disassembly.com/a2-elite/
 *      https://6502disassembly.com/a2-elite/meshes.html
 *
 */
package view;

import view.f32.F32Triangle2D;
import view.f32.F32Vec2;

import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.stream.IntStream;

public record Graphics2DRenderer(int width, int height, DisplayMode displayMode) implements Renderer {
    static private Renderer of(int width, int height, DisplayMode displayMode) {

        return new Graphics2DRenderer(width, height,displayMode);
    }

    static public Renderer wireOf(int width, int height) {
        return of(width, height, DisplayMode.WIRE);
    }

    static public Renderer fillOf(int width, int height) {
        return of(width, height, DisplayMode.FILL);
    }


    @Override
    public void render(boolean old) {
       // IntStream.range(0, width * height).parallel().forEach(id -> kernel(id, old));
        //System.arraycopy(offscreenRgb, 0, ((DataBufferInt) image.getRaster().getDataBuffer()).getData(), 0, offscreenRgb.length);
    }

    @Override
    public void paint(Graphics2D g) {/*
        g.drawImage(image, 0, 0, width, height, null); */
    }


}
