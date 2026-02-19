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
import hat.backend.Backend;
import hat.util.ui.Menu;
import hat.util.ui.SevenSegmentDisplay;

import javax.swing.JFrame;
import javax.swing.JMenuBar;
import javax.swing.JToggleButton;
import java.awt.Rectangle;
import java.lang.invoke.MethodHandles;

public class ShaderFrame {
    Accelerator acc;
    JFrame jFrame;
    FrameControls frameControls;

    public ShaderFrame(Accelerator acc, FrameControls frameControls) {
        this.acc = acc;
        this.frameControls = frameControls;
        jFrame = new JFrame(frameControls.shaderName());
        jFrame.setJMenuBar(frameControls.menu().menuBar());
        ShaderViewer imagePanel = new ShaderViewer(acc, frameControls);
        int frameWidth = frameControls.width() + Math.min(1600, 1600 - frameControls.width());
        int frameHeight = frameControls.height() + Math.min(1000, 1000 - frameControls.height());
        jFrame.setBounds(new Rectangle(frameWidth, frameHeight));
        jFrame.setContentPane(imagePanel.panel);
        jFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        jFrame.setVisible(true);
        imagePanel.start();
    }

    public static ShaderFrame of(Accelerator acc, FrameControls frameControls ) {
        return new ShaderFrame(acc,frameControls);
    }

    public static ShaderFrame of(Accelerator acc,  int width, int height , int targetFps, String name, Shader shader ) {
        return ShaderFrame.of(acc, new FrameControls(true,width,height,targetFps,name,shader));
    }

    public static ShaderFrame of(FrameControls frameControls) {
        return of(new Accelerator(MethodHandles.lookup(), Backend.FIRST),frameControls);
    }

    public static ShaderFrame of( int width, int height , int targetFps, String name, Shader shader ) {
        return of( new FrameControls(false,width,height,targetFps,name,shader));
    }


    public static class FrameControls implements Controller {

        private final boolean useHat;
        private final boolean showAllocations;

        private final int width;
        private final int height;
        private final int targetFps;
        private final String shaderName;
        private final Shader shader;


        @Override
        public boolean useHat() {
            return useHat;
        }

        @Override public boolean showAllocations() {
            return showAllocations;
        }
        @Override
        public int width() {
            return width;
        }

        @Override
        public int height() {
            return height;
        }

        @Override
        public int targetFps() {
            return targetFps;
        }

        @Override
        public Shader shader() {
            return shader;
        }

        @Override
        public String shaderName() {
            return shaderName;
        }

        @Override
        public boolean running() {
            return running != null && running.isSelected();
        }


        private final Menu menu;

        public Menu menu() {
            return menu;
        }
        private final boolean showTargetFps;

        private boolean showTargetFps() {
            return showTargetFps;
        }

        private final boolean showActualFps;

        private boolean showFps() {
            return showActualFps;
        }

        private final boolean showShaderTimeUs;

        private boolean showShaderTimeUs() {
            return showShaderTimeUs;
        }



        private final boolean showElapsedMs;

        private boolean showElapsedMs() {
            return showElapsedMs;
        }

        private final boolean showFrameNumber;

        private boolean showFrameCount() {
            return showFrameNumber;
        }

        private SevenSegmentDisplay allocations7Seg;
        private SevenSegmentDisplay shaderTimeUs7Seg;
        private SevenSegmentDisplay targetFps7Seg;
        private SevenSegmentDisplay actualFps7Seg;
        private SevenSegmentDisplay frameCount7Seg;
        private SevenSegmentDisplay elapsedMs7Seg;
        private JToggleButton running;

        public FrameControls(
                boolean useHat,
                int width,
                int height,
                int targetFps,
               String shaderName,
                Shader shader,
                boolean showTargetFps,
                boolean showActualFps,
                boolean showShaderTimeUs,
                boolean showAllocations,
                boolean showElapsedMs,
                boolean showFrameNumber
        ) {
            this.useHat = useHat;
            this.width = width;
            this.height = height;
            this.targetFps = targetFps;
            this.shaderName = shaderName;
            this.shader = shader;
            this.showTargetFps = showTargetFps;
            this.showActualFps = showActualFps;
            this.showShaderTimeUs = showShaderTimeUs;
            this.showAllocations = showAllocations;
            this.showElapsedMs = showElapsedMs;

            this.showFrameNumber = showFrameNumber;
            this.menu = new Menu(new JMenuBar())
                    .exit();
            if (showAllocations) {
                this.menu
                        .label("Vectors + Mats")
                        .sevenSegment(10, 15, $ -> allocations7Seg = $).space(20);
            }

            if (showShaderTimeUs) {
                this.menu
                        .label("Shader Time (us)")
                        .sevenSegment(6, 15, $ -> shaderTimeUs7Seg = $).space(20);
            }
            if (showFrameNumber) {
                this.menu
                        .label("Frame ").sevenSegment(6, 15, $ -> frameCount7Seg = $).space(20);
            }
            if (showElapsedMs) {
                this.menu
                        .label(showElapsedMs, "Elapsed (ms)")
                        .sevenSegment(6, 15, $ -> elapsedMs7Seg = $).space(20);
            }
            if (showTargetFps) {
                this.menu.label("Target Frames (per sec)").sevenSegment(4, 15, $ -> {
                        targetFps7Seg = $;
                        targetFps7Seg.set(targetFps);
                        }).space(20);

            }
            if (showActualFps) {
                this.menu.label("Actual Frames (per sec)").sevenSegment(4, 15, $ -> actualFps7Seg = $).space(20);
            }
            this.menu
                    .toggle("Stop", "Go", true, $ -> running = $, _ -> {
                    })
                    .space(40);
        }

        FrameControls allocations(int v) {
            if (showAllocations) {
                allocations7Seg.set(v);
            }
            return this;
        }

        FrameControls shaderTimeUs(int v) {
            if (showShaderTimeUs) {
                shaderTimeUs7Seg.set(v);
            }
            return this;
        }


        FrameControls actualFps(int v) {
            if (showActualFps) {
                actualFps7Seg.set(v);
            }
            return this;
        }

        FrameControls frameNumber(int v) {
            if (showFrameNumber) {
                frameCount7Seg.set(v);
            }
            return this;
        }

        FrameControls elapsedMs(int v) {
            if (showElapsedMs) {
                elapsedMs7Seg.set(v);
            }
            return this;
        }

        public FrameControls(
                boolean useHat,
                int width,
                int height,
                int targetFps,
                String shaderName,
                Shader shader
        ) {
            this(useHat,width,height,targetFps,shaderName,shader,true, true, false, false, false,
                    false);
        }
    }
}
