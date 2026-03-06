
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
import java.util.concurrent.CyclicBarrier;
import java.util.stream.IntStream;


public class ShaderViewer implements Runnable{
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
            //double delta = 0;
    long startTimeNs = System.nanoTime();
    uniforms.iFrame(uniforms.iFrame() + 1);
    int w = floatImage.width();
    int h = floatImage.height();
    uniforms.iResolution().x(w);
    uniforms.iResolution().y(h);
            while (running) {



             ///   while (delta >= 10) {
                    long diffNs = System.nanoTime() - startTimeNs;

                    long diffMs = diffNs / 1_000_000;
                    float fdiffMs = (float) diffMs;
                    uniforms.iTime(fdiffMs/1_000f);
                    long startNs = System.nanoTime();

                   // if (frameControls.running()) {

                      //  synchronized (bufferedImageViewer) {
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
                                var fwidth = w;
                                var fheight = h;
                                var f32Array = floatImage.f32Array();
                                frameControls.accelerator().compute((@Reflect Compute) cc -> HATShader.compute(cc, funiforms, f32Array, fwidth, fheight));
                            }
                            floatImage.sync();
                      //  }
                    //}
                    long endNs = System.nanoTime();
                    System.out.println("shader time us = "+((int) (endNs - startNs) / 1000));
                 /*   frameControls.shaderTimeUs((int) (endNs - startNs) / 1000)
                            .actualFps((int) (uniforms.iFrame() * 1000 / diffMs))
                            .frameNumber((int) uniforms.iFrame())
                            .elapsedMs((int) diffMs);*/
                //    delta -= 1f;
              //  }

                SwingUtilities.invokeLater(bufferedImageViewer::repaint);

                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                }
                uniforms.iFrame(uniforms.iFrame() + 1);
            }
        }
        /*
    long startTimeNs = System.nanoTime();

    public static FloatImage runShader(FloatImage floatImage,Config frameControls, Uniforms uniforms) {
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
        return floatImage;
    }

    void loop(){


            long nowNs = System.nanoTime();
            uniforms.iResolution().x(floatImage.width());
            uniforms.iResolution().y(floatImage.height());
            while (true) {
                long elapsedMs = (nowNs-startTimeNs) / 1000000;
                uniforms.iTime((float)elapsedMs / 1000f);
                long shaderStartNs = System.nanoTime();

              ///  bufferedImageViewer.repaint();
                long shaderEndNs = System.nanoTime();
                frameControls.shaderTimeUs((int) ((shaderEndNs - shaderStartNs) / 1000))
                        .frameNumber((int) uniforms.iFrame())
                        .actualFps((int) (uniforms.iFrame() * 1000 / (elapsedMs+1)))
                        .elapsedMs((int) elapsedMs);

                SwingUtilities.invokeLater(bufferedImageViewer::repaint);
                //try {
                  //  cyclicBarrier.await();
               // }catch (BrokenBarrierException | InterruptedException e){
                 //   System.out.println("bar ex 2");
               // }
            }
        }
    }

     void doit() {

        SwingWorker<Void, FloatImage> worker = new SwingWorker<Void, FloatImage>() {
            @Override
            protected Void doInBackground() {
                while (!isCancelled()) {
                    FloatImage d = ShaderViewer.runShader();
                            publish(d); // Sends data to the "process" method in batches
                }
                return null;
            }

            @Override
            protected void process(FloatImage chunks) {
                // This runs on the EDT.
                // If the loop is fast, 'chunks' contains multiple data points.
                // for (Data d : chunks) {
                //   canvas.addPoint(d);
                // }
              //  canvas.repaint();
            }
        };
        worker.execute();
    } */


    /*
    import javax.swing.*;
import java.awt.*;
import java.awt.image.*;

public class FastRenderPanel extends JPanel {
    private BufferedImage cpuBuffer;
    private VolatileImage vramBuffer;
    private int[] pixels;

    public FastRenderPanel(int w, int h) {
        // 1. Create the CPU-side buffer
        cpuBuffer = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        // Grab the internal array for direct manipulation
        pixels = ((DataBufferInt) cpuBuffer.getRaster().getDataBuffer()).getData();
    }

    public void startLoop() {
        new Thread(() -> {
            while (true) {
                updateSimulation(); // Your hard loop updating 'pixels'
                render();           // Custom render call

                // Cap at ~60fps to keep the OS happy
                try { Thread.sleep(16); } catch (InterruptedException e) {}
            }
        }).start();
    }

    private void updateSimulation() {
        for (int i = 0; i < pixels.length; i++) {
            // Example: Fill with random noise or logic
            pixels[i] = (int)(Math.random() * 0xFFFFFF);
        }
    }

    private void render() {
        // Validation: VolatileImages can be "lost" if the window is resized or the OS
        // reclaims VRAM. We must check its status every frame.
        if (vramBuffer == null || vramBuffer.validate(getGraphicsConfiguration()) == VolatileImage.IMAGE_INCONSISTENT) {
            vramBuffer = createVolatileImage(cpuBuffer.getWidth(), cpuBuffer.getHeight());
        }

        do {
            // Check if the VRAM content was lost
            if (vramBuffer.validate(getGraphicsConfiguration()) == VolatileImage.IMAGE_RESTORED) {
                // Restoration logic if needed
            }

            // Copy CPU pixels to VRAM
            Graphics2D gVram = vramBuffer.createGraphics();
            gVram.drawImage(cpuBuffer, 0, 0, null);
            gVram.dispose();

            // Finally, paint the VRAM buffer to the UI
            Graphics g = this.getGraphics();
            if (g != null) {
                g.drawImage(vramBuffer, 0, 0, null);
                g.dispose();
            }

        } while (vramBuffer.contentsLost());

        // Ensure smooth rendering on Linux/macOS
        Toolkit.getDefaultToolkit().sync();
    }
}
     */
}
