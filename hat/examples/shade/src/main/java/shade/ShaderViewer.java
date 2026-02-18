
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
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.ComputeContext.Kernel;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import hat.types.ivec2;
import hat.types.mat2;
import hat.types.mat3;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import javax.swing.JComponent;
import javax.swing.SwingUtilities;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.AffineTransform;
import java.awt.geom.Point2D;
import java.util.stream.IntStream;

public class ShaderViewer implements Runnable {
    private final Accelerator accelerator;
    final protected AffineTransform transform = new AffineTransform();
    final ShaderFrame.FrameControls frameControls;
    final FloatImage floatImage;
    final JComponent panel;
    final Uniforms uniforms;
    volatile boolean running;

    protected float zoom = .95f; // set the zoom factor 1.0 = fit to screen
    protected float xOffset = 0; // 0 is centered -1 is to the left;
    protected float yOffset = 0; // 0 is centered -1 is to the top;

    Point mousePressedPosition;
    Point2D imageRelativeMouseDownPosition = new Point2D.Float();
    Point2D imageRelativeMovePosition = new Point2D.Float();

    public ShaderViewer(Accelerator accelerator, ShaderFrame.FrameControls frameControls) {
        this.panel = new JComponent() {
            @Override
            public void paint(Graphics g) {
                Graphics2D g2d = (Graphics2D) g;
                Dimension displaySize = getSize();
                Dimension imageSize = new Dimension(frameControls.width(), frameControls.height());
                AffineTransform safeTransform = g2d.getTransform();
                transform.setToIdentity();
                double scale = zoom * Math.min(displaySize.width / (double) imageSize.width, displaySize.height / (double) imageSize.height);
                transform.translate((1 + xOffset) * (displaySize.width - imageSize.width * scale) / 2,
                        (1 + yOffset) * (displaySize.height - imageSize.height * scale) / 2);
                transform.scale(scale, scale);
                g2d.transform(transform);
                synchronized (floatImage) {
                    g.drawImage(floatImage.bufferedImage(), 0, 0, imageSize.width, imageSize.height, null);
                }
                g2d.setTransform(safeTransform);
            }
        };
        this.accelerator = accelerator;
        this.frameControls = frameControls;
        this.floatImage = FloatImage.of(accelerator, frameControls.width(), frameControls.height());
        this.uniforms = Uniforms.create(accelerator);
        this.panel.addMouseListener(new MouseAdapter() {
            @Override
            public void mouseReleased(MouseEvent e) {
             /*  if (SwingUtilities.isLeftMouseButton(e)) {
                    Timer t = new Timer(1000, new ActionListener() {
                        @Override
                        public void actionPerformed(ActionEvent e) {
                            //  selection = null;
                            //  bestMatchOffset = null;
                            repaint();
                        }
                    });
                    t.setRepeats(false);
                    t.start();
                    repaint();
                }*/
            }

            @Override
            public void mousePressed(MouseEvent e) {
                /*
                if (SwingUtilities.isLeftMouseButton(e)) {
                    try {
                        var ptDst = transform.inverseTransform(e.getPoint(), null);
                        //  selection = new Selection(ptDst);
                    } catch (NoninvertibleTransformException e1) {
                        e1.printStackTrace();
                    }
                } else if (SwingUtilities.isRightMouseButton(e)) {
                    mousePressedPosition = e.getPoint();
                    try {
                        imageRelativeMouseDownPosition = transform.inverseTransform(e.getPoint(), null);
                    } catch (NoninvertibleTransformException e1) {
                        e1.printStackTrace();
                    }
                } */
            }

        });
        this.panel.addMouseWheelListener(e -> {
            zoom = zoom * (1 + e.getWheelRotation() / 10f);
            //  repaint(); */
        });
        this.panel.addMouseMotionListener(new MouseMotionAdapter() {
            @Override
            public void mouseDragged(MouseEvent e) {
                /*
                if (SwingUtilities.isRightMouseButton(e)) {
                    Point rightButonPoint = e.getPoint();
                    Dimension offsetFromInitialMousePress = new Dimension(rightButonPoint.x - mousePressedPosition.x, rightButonPoint.y - mousePressedPosition.y);
                    try {
                        imageRelativeMovePosition = transform.inverseTransform(e.getPoint(), null);
                        Dimension displaySize = getSize();
                        Dimension imageSize = new Dimension(width, height);
                        float scale = zoom *
                                Math.min(displaySize.width / (float) imageSize.width,
                                        displaySize.height / (float) imageSize.height);
                        xOffset = 2 * (offsetFromInitialMousePress.width / (displaySize.width - scale * imageSize.width));
                        yOffset = 2 * (offsetFromInitialMousePress.height / (displaySize.height - scale * imageSize.height));
                        xOffset = Math.max(Math.min(xOffset, 1), -1);
                        yOffset = Math.max(Math.min(yOffset, 1), -1);
                        repaint();
                    } catch (NoninvertibleTransformException e1) {
                        e1.printStackTrace();
                    }
                } else if (SwingUtilities.isLeftMouseButton(e)) {
                    try {
                        var ptDst = transform.inverseTransform(e.getPoint(), null);
                        //   selection.add(ptDst);
                        repaint();
                    } catch (NoninvertibleTransformException e1) {
                        // TODO Auto-generated catch block
                        e1.printStackTrace();
                    }
                }*/
            }

            @Override
            public void mouseMoved(MouseEvent e) {
                uniforms.iMouse().x(e.getX());
                uniforms.iMouse().y(e.getY());
            }
        });
    }

    public void start() {
        running = true;
        new Thread(this).start();
    }


    @Reflect
    public static vec4 shader(@MappableIface.RO Uniforms uniforms, vec4 fragColor, ivec2 fragCoord) {
        return vec4.vec4(0f, 0f, 1f, 0f);
    }

    @Reflect
    public static void penumbra(@MappableIface.RO KernelContext kc, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image) {
        if (kc.gix < kc.gsx) {
            // The image is essentially 3x
            int width = (int)uniforms.iResolution().x();
            int height = (int)uniforms.iResolution().y();
            var fragCoord = ivec2.ivec2(kc.gix % width, kc.gix / width);
            long offset = ((long) kc.gsx * height * 3) + (kc.gix * 3L);
            float r = image.array(offset + 0);
            float g = image.array(offset + 1);
            float b = image.array(offset + 2);
            var fragColor = shader(uniforms, vec4.vec4(r, g, b, 0f), fragCoord);
            image.array(offset + 0, fragColor.x());
            image.array(offset + 1, fragColor.y());
            image.array(offset + 2, fragColor.z());
        }
    }


    @Reflect
    static public void compute(final ComputeContext computeContext, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image, int width, int height) {
        computeContext.dispatchKernel(
                NDRange.of1D(width * height),               //0..S32Array2D.size()
                (@Reflect Kernel) kc -> penumbra(kc, uniforms, image));
    }

    final public void runShader(FloatImage floatImage) {
        if (!frameControls.useHat()) {
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
            // the following failed!
            //   accelerator.compute((@Reflect Compute) cc ->compute(cc,funiforms,floatImage.f32Array(),fwidth,fheight));
            accelerator.compute((@Reflect Compute) cc -> compute(cc, funiforms, f32Array, fwidth, fheight));
        }
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
                    ivec2.collect.set(true);
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
                    mat3.count.set(0);
                }
                if (frameControls.running()) {
                    uniforms.iFrame(uniforms.iFrame() + 1);
                    synchronized (floatImage) {
                        // We synchronize here and in the paint method.  To ensure that we don't copy memory segment mid compute.
                        runShader(floatImage);
                        floatImage.sync();
                    }
                }
                long endNs = System.nanoTime();
                if (frameControls.showAllocations()) {
                    frameControls.allocations(
                        ivec2.count.get() + vec2.count.get() + vec3.count.get() + vec4.count.get() + mat2.count.get() + mat3.count.get()
                    );
                }
                frameControls.shaderTimeUs((int) (endNs - startNs) / 1000)
                        .actualFps((int) (uniforms.iFrame() * 1000 / diffMs))
                        .frameNumber((int) uniforms.iFrame())
                        .elapsedMs((int) diffMs);
                delta -= 1f;
            }

            // Schedule Render on EDT
            SwingUtilities.invokeLater(panel::repaint);

            // Cap the loop to save CPU
            try {
                Thread.sleep(2);
            } catch (InterruptedException e) {
            }
        }
    }


}
