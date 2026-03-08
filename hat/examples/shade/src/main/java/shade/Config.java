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

import hat.Accelerator;
import hat.util.ui.Menu;
import hat.util.ui.SevenSegmentDisplay;

import javax.swing.JMenuBar;
import javax.swing.JToggleButton;

public class Config {
    private final Accelerator accelerator;

    private final int width;
    private final int height;
    private final String shaderName;
    private final Shader shader;


    public Accelerator accelerator() {
        return accelerator;
    }

    public int width() {
        return width;
    }


    public int height() {
        return height;
    }


    public Shader shader() {
        return shader;
    }


    public String shaderName() {
        return shaderName;
    }


    public boolean running() {
        return running != null && running.isSelected();
    }


    private final Menu menu;

    public Menu menu() {
        return menu;
    }

    private final boolean showActualFps;

    private boolean showFps() {
        return showActualFps;
    }

    private final boolean showShaderTimeUs;

    private boolean showShaderTimeUs() {
        return showShaderTimeUs;
    }

    private final boolean showFrameNumber;

    private boolean showFrameCount() {
        return showFrameNumber;
    }

    private SevenSegmentDisplay shaderTimeUs7Seg;
    private SevenSegmentDisplay actualFps7Seg;
    private SevenSegmentDisplay frameCount7Seg;
    private SevenSegmentDisplay elapsedMs7Seg;
    private JToggleButton running;

    public Config(
            Accelerator accelerator,
            int width,
            int height,
            String shaderName,
            Shader shader,
            boolean showActualFps,
            boolean showShaderTimeUs,
            boolean showFrameNumber
    ) {
        this.accelerator = accelerator;
        this.width = width;
        this.height = height;
        this.shaderName = shaderName;
        this.shader = shader;
        this.showActualFps = showActualFps;
        this.showShaderTimeUs = showShaderTimeUs;

        this.showFrameNumber = showFrameNumber;
        this.menu = new Menu(new JMenuBar()).exit();
        if (showShaderTimeUs) {
            this.menu
                    .label("Shader (us)")
                    .sevenSegment(6, 15, $ -> shaderTimeUs7Seg = $).space(20);
        }
        if (showFrameNumber) {
            this.menu
                    .label("Frame ").sevenSegment(6, 15, $ -> frameCount7Seg = $).space(20);
        }
        if (showActualFps) {
            this.menu.label("FPS").sevenSegment(4, 15, $ -> actualFps7Seg = $).space(20);
        }
        this.menu.space(40);
    }


    Config shaderTimeUs(int v) {
        if (showShaderTimeUs) {
            shaderTimeUs7Seg.set(v);
        }
        return this;
    }


    Config actualFps(int v) {
        if (showActualFps) {
            actualFps7Seg.set(v);
        }
        return this;
    }

    Config frameNumber(int v) {
        if (showFrameNumber) {
            frameCount7Seg.set(v);
        }
        return this;
    }


    public static Config of(
            Accelerator accelerator,
            int width,
            int height,
            String shaderName,
            Shader shader,
            boolean showActualFps,
            boolean showShaderTimeUs,
            boolean showFrameNumber) {
        return new Config(accelerator, width, height, shaderName, shader,  showActualFps, showShaderTimeUs,  showFrameNumber);
    }


    public static Config of(Accelerator accelerator, int width, int height, String name, Shader shader) {
        return new Config(accelerator, width, height,  name, shader, false,  false, false);
    }

    public static Config of(Accelerator accelerator, int width, int height,  Shader shader) {
        return new Config(accelerator, width, height,  shader.getClass().getSimpleName(), shader,
                Boolean.parseBoolean(System.getProperty("showActualFps", "true")),
                Boolean.parseBoolean(System.getProperty("showShaderTimeUs", "true")),
                Boolean.parseBoolean(System.getProperty("showFrameCount", "false")));
    }
}
