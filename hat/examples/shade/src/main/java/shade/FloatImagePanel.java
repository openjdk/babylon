
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
import shade.types.Shader;
import shade.types.Uniforms;
import shade.types.vec2;
import shade.types.vec4;

import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.util.stream.IntStream;

public class FloatImagePanel extends JPanel implements Runnable {
    protected AffineTransform transform = new AffineTransform();
    protected float zoom = .95f; // set the zoom factor 1.0 = fit to screen
    protected float xOffset = 0; // 0 is centered -1 is to the left;
    protected float yOffset = 0; // 0 is centered -1 is to the top;

    Point mousePressedPosition;
    Point2D imageRelativeMouseDownPosition = new Point2D.Float();
    Point2D imageRelativeMovePosition = new Point2D.Float();
    int width;
    int height;
    private Controls controls;
    private FloatImage floatImage;
    final Shader shader;
    final Uniforms uniforms;
    volatile boolean running;


    //static long runShader(FloatImage floatImage, Uniforms uniforms, Shader shader) {

   // }


    public FloatImagePanel(Accelerator accelerator, Controls controls, int width, int height, Shader shader) {
        this.width = width;
        this.height = height;
        this.controls = controls;
        this.shader = shader;
        this.floatImage = FloatImage.of(accelerator, width, height);
        this.uniforms = Uniforms.create(accelerator);
        addMouseListener(new MouseAdapter() {
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
        addMouseWheelListener(e -> {
          /*  zoom = zoom * (1 + e.getWheelRotation() / 10f);
            repaint(); */
        });
        addMouseMotionListener(new MouseMotionAdapter() {
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

    private Thread interpolationThread;

    public void start() {
        running = true;
        interpolationThread = new Thread(this);
        interpolationThread.start();
    }
    private float renderInterpolation = 0;

    @Override
    public void run() {
        long startTimeNs = System.nanoTime();

        double nsPerTick = 1000000000.0 / 30.0; // 60 Fixed Updates per second
        double delta = 0;
        long lastTimeNs = System.nanoTime();

        while (running) {
            long now = System.nanoTime();
            delta += (now - lastTimeNs) / nsPerTick;
            lastTimeNs = now;
            uniforms.iResolution().x(floatImage.width());
            uniforms.iResolution().y(floatImage.height());
            // Fixed Update Loop
            while (delta >= 1) {
                long diff = lastTimeNs - startTimeNs;
                long diffMs = diff / 1000000;
                uniforms.iFrame(uniforms.iFrame() + 1);
                uniforms.iTime(diffMs);

                long startNs = System.nanoTime();
                IntStream.range(0, floatImage.widthXHeight()).parallel().forEach(i -> {
                    vec2 fragCoord = vec2.vec2(i % floatImage.width(), (float)( i / floatImage.width()));
                    vec4 inFragColor = vec4.vec4(0);
                    vec4 outFragColor = shader.mainImage(uniforms, inFragColor, fragCoord);
                    floatImage.set(i, outFragColor);
                });
                floatImage.sync();
                long endNs = System.nanoTime();
                controls.shaderUs((int)(endNs-startNs)/1000)
                        .fps((int) (uniforms.iFrame() * 1000 / diffMs))
                        .frame((int) uniforms.iFrame())
                        .elapsedMs((int) diffMs);
                delta--;
            }

            // Calculate Interpolation for rendering
            renderInterpolation = (float) delta;

            // Schedule Render on EDT
            SwingUtilities.invokeLater(this::repaint);

            // Cap the loop to save CPU
            try {
                Thread.sleep(1);
            } catch (InterruptedException e) {
            }
        }
    }

    @Override
    public void paint(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        Dimension displaySize = getSize();
        Dimension imageSize = new Dimension(width, height);
        AffineTransform safeTransform = g2d.getTransform();
        transform.setToIdentity();
        double scale = zoom * Math.min(displaySize.width / (double) imageSize.width, displaySize.height / (double) imageSize.height);
        transform.translate((1 + xOffset) * (displaySize.width - imageSize.width * scale) / 2,
                (1 + yOffset) * (displaySize.height - imageSize.height * scale) / 2);
        transform.scale(scale, scale);
        g2d.transform(transform);
        g.drawImage(floatImage.bufferedImage(), 0, 0, imageSize.width, imageSize.height, null);
        g2d.setTransform(safeTransform);
    }

}
