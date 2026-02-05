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
import hat.buffer.F32Array;
import hat.util.ui.Menu;
import hat.util.ui.SevenSegmentDisplay;
import shade.types.ivec2;
import shade.types.vec2;
import shade.types.vec4;
import static shade.types.vec4.*;
import static shade.types.vec2.*;

import javax.swing.JFrame;
import javax.swing.JMenuBar;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Rectangle;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.event.MouseMotionAdapter;
import java.awt.geom.AffineTransform;
import java.awt.geom.NoninvertibleTransformException;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferFloat;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.io.IOException;
import java.lang.invoke.MethodHandles;
import java.util.List;
import java.util.stream.IntStream;


public class Main extends JFrame {
    public interface Shader{
        vec4 mainImage(vec4 fragColor, vec2 fragCoord, int iFrame, int iTime, ivec2 iMouse);
    }

    public final ImagePanel imagePanel;

    public static class ImagePanel extends JPanel implements Runnable{

        public static BufferedImage createFloatImage(int width, int height) {
            // We need an RGB colorspace.
            ColorSpace colorSpace = ColorSpace.getInstance(ColorSpace.CS_sRGB);

            // Create the Color Model. 32 bits per component, no alpha, non-premultiplied
            ColorModel colorModel = new ComponentColorModel(colorSpace, false, false,
                    Transparency.OPAQUE, DataBuffer.TYPE_FLOAT);

            // Create the Sample Model (Pixel Interleaved) bands for RGB, scanline stride is width * 3
            SampleModel sampleModel = new PixelInterleavedSampleModel(DataBuffer.TYPE_FLOAT,
                    width, height, 3, width * 3, new int[]{0, 1, 2});

            // Create the DataBuffer (an actual heap allocated  float array)
            DataBufferFloat dataBufferFloat = new DataBufferFloat(width * height * 3);

            // Create the Raster
            WritableRaster raster = Raster.createWritableRaster(sampleModel, dataBufferFloat, null);

            // Finally an actual BufferedImage
            return new BufferedImage(colorModel, raster, false, null);
        }

        protected F32Array f32Array;
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
        private BufferedImage image;

        final Shader shader;

        static long runShader(BufferedImage image, F32Array f32Array,int frame,int time, int mouseX, int mouseY, Shader shader){
            long start = System.nanoTime();
            WritableRaster raster = image.getRaster();
            DataBufferFloat buffer = (DataBufferFloat) raster.getDataBuffer();
            float[] data = buffer.getData();
            int iFrame = frame;
            int iTime = time;
            var iMouse = shade.types.ivec2.ivec2(mouseX,mouseY);
            int width = image.getWidth();
            int height = image.getHeight();

            IntStream.range(0, width * height).parallel().forEach(i -> {
                int x = i%width;
                int y = i%height;
                vec2 fragCoord = vec2((float)x/width,(float)y/height);
                vec4 inFragColor = vec4(0);
                vec4 outFragColor = shader.mainImage(inFragColor,fragCoord,iFrame,iTime,iMouse);
                data[i * 3 + 0] = outFragColor.x();
                data[i * 3 + 1] = outFragColor.y();
                data[i * 3 + 2] = outFragColor.z();
            });
            f32Array.copyFrom(data);
            return System.nanoTime()-start;
        }
        volatile boolean running;

        public ImagePanel(Accelerator accelerator, Controls controls, int width, int height, Shader shader) {

            this.width = width;
            this.height = height;
            this.controls = controls;
            this.shader = shader;
            this.f32Array = F32Array.create(accelerator, width * height);
            this.image = createFloatImage(width,height);
           // long ns = this.runShader(this.image, this.f32Array,0,0,0,0,shader);
           // System.out.println(ns/1000000+"ms");
            addMouseListener(new MouseAdapter() {
                @Override
                public void mouseReleased(MouseEvent e) {
                    if (SwingUtilities.isLeftMouseButton(e)) {
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
                    }
                }

                @Override
                public void mousePressed(MouseEvent e) {
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
                    }
                }

            });
            addMouseWheelListener(e -> {
                zoom = zoom * (1 + e.getWheelRotation() / 10f);
                repaint();
            });
            addMouseMotionListener(new MouseMotionAdapter() {
                @Override
                public void mouseDragged(MouseEvent e) {
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
                    }
                }
            });
        }
        float renderInterpolation=0;

        int frame = 0;
        //long startTimeNs = 0;
        int time = 0;

        @Override
        public void run() {
           // if (startTimeNs==0){
            long     startTimeNs=System.nanoTime();
            //}
            double nsPerTick = 1000000000.0 / 30.0; // 60 Fixed Updates per second
            double delta = 0;
            long lastTimeNs = System.nanoTime();

            while (running) {
                long now = System.nanoTime();
                delta += (now - lastTimeNs) / nsPerTick;
                lastTimeNs = now;

                // Fixed Update Loop
                while (delta >= 1) {
                    long diff = lastTimeNs-startTimeNs;
                    long diffMs = diff/1000000;
                    int us = (int)(runShader(image,f32Array,frame++,(int)(diffMs),0,0,shader)/1000);
                    controls.shaderMicroSeconds.set(us);
                    long framesPerSecond = frame*1000/diffMs;
                  //  System.out.println("diffMs="+diffMs+" startUs="+(startTimeNs/1000)+" nowUs="+(now/1000)+ " frame"+frame);
                    controls.framesPerSecond.set((int)framesPerSecond);
                    controls.frame.set(frame);
                    controls.elapsedMs.set((int)diffMs);
                    delta--;
                }

                // Calculate Interpolation for rendering
                renderInterpolation = (float) delta;

                // Schedule Render on EDT
                SwingUtilities.invokeLater(this::repaint);

                // Cap the loop to save CPU
                try { Thread.sleep(10); } catch (InterruptedException e) {}
            }
        }

        @Override
        public void paint(Graphics g) {
            Graphics2D g2d = (Graphics2D) g;
            g2d.setBackground(Color.BLACK);
            g2d.fillRect(0, 0, getWidth(), getHeight());
            if (f32Array != null) {
                Dimension displaySize = getSize();
                Dimension imageSize = new Dimension(width, height);
                AffineTransform safeTransform = g2d.getTransform();
                transform.setToIdentity();
                double scale = zoom * Math.min(displaySize.width / (double) imageSize.width, displaySize.height / (double) imageSize.height);
                transform.translate((1 + xOffset) * (displaySize.width - imageSize.width * scale) / 2,
                        (1 + yOffset) * (displaySize.height - imageSize.height * scale) / 2);
                transform.scale(scale, scale);
                g2d.transform(transform);
                g.drawImage(image, 0, 0, imageSize.width, imageSize.height, null);
                g2d.setTransform(safeTransform);
            }
        }
        Thread gameThread;
        public void start() {
            running = true;
            gameThread = new Thread(this);
            gameThread.start();
        }
    }
    public static class Controls {
        Menu menu;
        SevenSegmentDisplay shaderMicroSeconds;
        SevenSegmentDisplay framesPerSecond;
        SevenSegmentDisplay frame;
        SevenSegmentDisplay elapsedMs;
        Controls() {
            menu = new Menu(new JMenuBar())
                    .exit()
                    .space(40)
                    .label("Shader Time (us)").sevenSegment(6, 15, $ -> shaderMicroSeconds = $).space(20)
                    .label("Frame ").sevenSegment(6, 15, $ -> frame = $).space(20)
                    .label("Elapsed (ms)").sevenSegment(6, 15, $ -> elapsedMs = $).space(20)
                    .label("Frames (per sec)").sevenSegment(4, 15, $ -> framesPerSecond = $).space(20)

                    //    .slider(0, 100, 0, n -> sevenSegmentDisplay.set(n))
                   // .combo(List.of("One", "Two", "Three"), "One", System.out::println)
                   // .hradio(List.of("JavaMt", "JavaSeq", "HAT"), $ -> System.out.println($.value()))
                    .space(40);
        }
    }

    public Main(Accelerator accelerator, int width, int height, Shader shader) {
        super("HAT Toy");
        Controls controls = new Controls();
        setJMenuBar(controls.menu.menuBar());
        this.imagePanel = new ImagePanel(accelerator, controls, width, height, shader);
        setBounds(new Rectangle(width + 100, height + 200));
        setContentPane(imagePanel);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        // pack();
        // validate();
        setVisible(true);
        this.imagePanel.start();
    }

     public static void main(String[] args) throws IOException {
        var acc =  new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var shader = new Shader() {
            @Override
            public vec4 mainImage(vec4 inFragColor, vec2 fragCoord, int iTime, int iFrame, ivec2 iMouse) {
              //  if (fragCoord.x()>.2f && fragCoord.y()<.8f){
                    return vec4(fragCoord.x()*iFrame,.5f,fragCoord.y()*iFrame, 0.f);
               // }else{
                 //   return vec4(1f,1f,1f,0f);
                //}
            }
        };
        new Main(acc, 1024, 1024, shader);
    }

}