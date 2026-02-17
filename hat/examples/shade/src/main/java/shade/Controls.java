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
package shade;

import hat.util.ui.Menu;
import hat.util.ui.SevenSegmentDisplay;

import javax.swing.JMenuBar;
import javax.swing.JToggleButton;

public class Controls {
    public  final Menu menu;
    private SevenSegmentDisplay vectorsAndMats7Seg;
    private SevenSegmentDisplay shaderUs7Seg;
    private SevenSegmentDisplay fps7Seg;
    private SevenSegmentDisplay frame7Seg;
    private SevenSegmentDisplay elapsedMs7Seg;
    private JToggleButton running;

    public Controls() {
        this.menu = new Menu(new JMenuBar())
                .exit()
                .label("Vectors + Mats").sevenSegment(10, 15, $ -> vectorsAndMats7Seg = $).space(20)
                .label("Shader Time (us)").sevenSegment(6, 15, $ -> shaderUs7Seg = $).space(20)
                .label("Frame ").sevenSegment(6, 15, $ -> frame7Seg = $).space(20)
                .label("Elapsed (ms)").sevenSegment(6, 15, $ -> elapsedMs7Seg = $).space(20)
                .label("Frames (per sec)").sevenSegment(4, 15, $ -> fps7Seg = $).space(20)
                .toggle("Stop","Go",true, $->running=$,_->{})
                .space(40);
    }

    Controls vectorsAndMats(int v) {
        vectorsAndMats7Seg.set(v);
        return this;
    }

    Controls shaderUs(int v) {
        shaderUs7Seg.set(v);
        return this;
    }

    boolean running() {
        return running.isSelected();
    }

    Controls fps(int v) {
        fps7Seg.set(v);
        return this;
    }

    Controls frame(int v) {
        frame7Seg.set(v);
        return this;
    }

    Controls elapsedMs(int v) {
        elapsedMs7Seg.set(v);
        return this;
    }
}
