
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
import shade.ui.BufferedImageViewer;

import javax.swing.SwingUtilities;
import java.lang.foreign.Arena;
import java.lang.invoke.MethodHandles;
import java.util.stream.IntStream;

public class ShaderViewer implements Runnable{
    final Config frameControls;
    final FloatImage floatImage;
    final BufferedImageViewer bufferedImageViewer;
    final Uniforms uniforms;
    volatile boolean running;

    public ShaderViewer(Config frameControls) {
        this.frameControls = frameControls;
        // we need an arena even if we don't have an accelerator ;)
        ArenaAndLookupCarrier arenaAndLookupCarrier = frameControls.accelerator() == null
                ? ArenaAndLookupCarrier.of(MethodHandles.lookup(), Arena.global())
                : frameControls.accelerator();

        this.floatImage = FloatImage.of(arenaAndLookupCarrier, frameControls.width(), frameControls.height());
        this.uniforms = Uniforms.create(arenaAndLookupCarrier);
        this.bufferedImageViewer = new BufferedImageViewer(floatImage.bufferedImage(), point -> {
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

            long startTimeNs = System.nanoTime();

            double nsPerTick = 1000000000.0 / frameControls.targetFps(); // 2 Fixed Updates per second
            double delta = 0;
            long lastTimeNs = System.nanoTime();
            while (running) {
                long now = System.nanoTime();
                delta += (now - lastTimeNs) / nsPerTick;
                lastTimeNs = now;
                uniforms.iResolution().x(floatImage.width());
                uniforms.iResolution().y(floatImage.height());

                while (delta >= 1) {
                    long diffNs = lastTimeNs - startTimeNs;

                    long diffMs = diffNs / 1000000;
                    float fdiffMs = (float) diffMs;
                    uniforms.iTime(fdiffMs / 1000);
                    long startNs = System.nanoTime();

                    if (frameControls.showAllocations()) {
                      /*  ivec2.collect.set(true);
                        vec2.collect.set(true);
                        vec3.collect.set(true);
                        vec4.collect.set(true);
                        mat2.collect.set(true);
                        mat3.collect.set(true);
                        ivec2.count.set(0);
                        vec2.count.set(0);
                        vec3.count.set(0);
                        vec4.count.set(0);
                        mat2.count.set(0);
                        mat3.count.set(0); */
                    }
                    if (frameControls.running()) {
                        uniforms.iFrame(uniforms.iFrame() + 1);
                        synchronized (bufferedImageViewer) {
                            // We synchronize here and in the paint method.  To ensure that we don't copy memory segment mid compute.

                            if (frameControls.accelerator() == null) {
                                IntStream.range(0, floatImage.widthXHeight()).parallel().forEach(i -> {
                                    vec2 fragCoord = vec2.vec2((float) i % floatImage.width(), (float) (floatImage.height() - (i / floatImage.width())));
                                    vec4 inFragColor = vec4.vec4(0);
                                    vec4 outFragColor = frameControls.shader().mainImage(uniforms, inFragColor, fragCoord);
                                    floatImage.set(i, outFragColor);
                                });
                            } else {
                                var funiforms = uniforms;
                                var fwidth = frameControls.width();
                                var fheight = frameControls.height();
                                var f32Array = floatImage.f32Array();
                                frameControls.accelerator().compute((@Reflect Compute) cc -> HATShader.compute(cc, funiforms, f32Array, fwidth, fheight));
                            }
                            floatImage.sync();
                        }
                    }
                    long endNs = System.nanoTime();
                    if (frameControls.showAllocations()) {
                      //  frameControls.allocations(
                        //        ivec2.count.get() + vec2.count.get() + vec3.count.get() + vec4.count.get() + mat2.count.get() + mat3.count.get()
                       // );
                    }
                    frameControls.shaderTimeUs((int) (endNs - startNs) / 1000)
                            .actualFps((int) (uniforms.iFrame() * 1000 / diffMs))
                            .frameNumber((int) uniforms.iFrame())
                            .elapsedMs((int) diffMs);
                    delta -= 1f;
                }

                SwingUtilities.invokeLater(bufferedImageViewer::repaint);

                try {
                    Thread.sleep(2);
                } catch (InterruptedException e) {
                }
            }
        }



}
