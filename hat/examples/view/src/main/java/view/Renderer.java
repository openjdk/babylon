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

import java.awt.Graphics2D;
import java.awt.Image;

public interface Renderer {
    enum DisplayMode {
        FILL(false, true, false),
        WIRE(true, false, false),
        WIRE_SHOW_HIDDEN(true, false, true),
        WIRE_AND_FILL(true, true, false);
        final public boolean wire;
        final public boolean filled;
        final public boolean showHidden;

        DisplayMode(boolean wire, boolean filled, boolean showHidden) {
            this.wire = wire;
            this.filled = filled;
            this.showHidden = showHidden;
        }
    }

    DisplayMode displayMode();

    void render(boolean old);

    void paint(Graphics2D g);

    Image image();

    int width();

    int height();
}
