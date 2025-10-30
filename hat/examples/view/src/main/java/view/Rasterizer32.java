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

import java.util.stream.IntStream;

public record Rasterizer32(View view, DisplayMode displayMode) implements Renderer {
    static public Rasterizer of(View view, DisplayMode displayMode){
        return new Rasterizer(view, displayMode);
    }

    private void accept(int gid) {
        int x = gid % view.image.getWidth();
        int y = gid / view.image.getHeight();
        int col = 0x404040;
        for (F32.TriangleVec2 t: F32.TriangleVec2.arr) {
            var v0 =  t.v0();
            var v1 = t.v1();
            var v2 = t.v2();
            if (displayMode.filled && F32.TriangleVec2.intriangle(x, y, v0.x(), v0.y(), v1.x(),v1.y(),v2.x(),v2.y())) {
                col = t.rgb();
            } else if (displayMode.wire && F32.TriangleVec2.onedge(x, y, v0.x(), v0.y(), v1.x(),v1.y(),v2.x(),v2.y())) {
                col =t.rgb();
            }
        }
        view.offscreenRgb[gid] = col;
    }
@Override
    public void render() {
        IntStream.range(0, view.image.getHeight()*view.image.getWidth()).parallel().forEach(this::accept);
        view().update();
    }
}
