
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

import hat.Accelerator.Compute;
import hat.buffer.Uniforms;
import hat.types.vec2;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.util.carriers.ArenaAndLookupCarrier;

import javax.swing.JComponent;
import javax.swing.SwingUtilities;
import java.awt.Graphics;
import java.awt.Point;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionListener;
import java.awt.image.BufferedImage;
import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.util.concurrent.CyclicBarrier;
import java.util.function.Consumer;
import java.util.stream.IntStream;


public class ShaderViewer implements Runnable {
    final Config frameControls;
    final FloatImage floatImage;
    final BufferedImageViewer bufferedImageViewer;
    final Uniforms uniforms;
    volatile boolean running;
    final CyclicBarrier cyclicBarrier = new CyclicBarrier(1);
    private final Object doorBell = new Object();

    public ShaderViewer(Config frameControls) {
        this.frameControls = frameControls;
        // we need an arena even if we don't have an accelerator ;)
        ArenaAndLookupCarrier arenaAndLookupCarrier = frameControls.accelerator() == null
                ? ArenaAndLookupCarrier.of(MethodHandles.lookup(), Arena.global())
                : frameControls.accelerator();

        this.floatImage = FloatImage.of(arenaAndLookupCarrier, frameControls.width(), frameControls.height());
        this.uniforms = Uniforms.create(arenaAndLookupCarrier);
        this.bufferedImageViewer = new BufferedImageViewer(floatImage, point -> {
            uniforms.iMouse().x(point.x);
            uniforms.iMouse().y(point.y);
        });
    }

    public void startShader() {
        running = true;
        new Thread(this).start();
    }

    @Override
    public void run() {
        //double delta = 0;
        long startTimeNs = System.nanoTime();
        uniforms.iFrame(uniforms.iFrame() + 1);
        int w = floatImage.width();
        int h = floatImage.height();
        uniforms.iResolution().x(w);
        uniforms.iResolution().y(h);
        while (running) {
            long diffNs = System.nanoTime() - startTimeNs;

            long diffMs = diffNs / 1_000_000;
            float fdiffMs = (float) diffMs;
            uniforms.iTime(fdiffMs / 1_000f);
            long startNs = System.nanoTime();
            synchronized (floatImage) {
                if (frameControls.accelerator() == null) {
                    IntStream.range(0, floatImage.widthXHeight()).parallel().forEach(i -> {
                        vec2 fragCoord = vec2.vec2((float) i % floatImage.width(), (float) (floatImage.height() - (i / floatImage.width())));
                        vec4 inFragColor = vec4.vec4(0);
                        vec4 outFragColor = frameControls.shader().mainImage(uniforms, inFragColor, fragCoord);
                        floatImage.set(i, outFragColor);
                    });
                } else {
                    var funiforms = uniforms;
                    var fwidth = w;
                    var fheight = h;
                    var f32Array = floatImage.f32Array();
                    frameControls.accelerator().compute((@Reflect Compute) cc -> HATShader.compute(cc, funiforms, f32Array, fwidth, fheight));
                }
                floatImage.sync();
            }
            long endNs = System.nanoTime();
            frameControls.shaderTimeUs((int) (endNs - startNs) / 1000)
                    .actualFps((int) (uniforms.iFrame() * 1000 / diffMs))
                    .frameNumber((int) uniforms.iFrame());


            SwingUtilities.invokeLater(bufferedImageViewer::repaint);

            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
            }
            uniforms.iFrame(uniforms.iFrame() + 1);
        }
    }


    public static class BufferedImageViewer extends JComponent {
        private final FloatImage floatImage;

        public BufferedImageViewer(FloatImage floatImage, Consumer<Point> mouseLocationConsumer) {

            this.floatImage =floatImage;
            this.addMouseMotionListener(new MouseMotionListener() {
                @Override
                public void mouseDragged(MouseEvent e) {
                    mouseLocationConsumer.accept(e.getPoint());
                }

                @Override
                public void mouseMoved(MouseEvent e) {
                    mouseLocationConsumer.accept(e.getPoint());
                }
            });
        }

        @Override
        public void paint(Graphics graphics) {
            synchronized (floatImage) {
                graphics.drawImage(floatImage.bufferedImage(), 0, 0, this.getWidth(), this.getHeight(), null);
            }
        }

    }
}
