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
package experiments;

import hat.Accelerator;
import hat.ComputeContext;
import hat.buffer.S32Array2D;
import hat.buffer.S32RGBAImage;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import javax.swing.Box;
import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.WritableRaster;
import java.lang.invoke.MethodHandles;
import java.util.Arrays;

public class DirectRasterFrame extends JPanel implements Runnable {
    private final int WIDTH = 1024, HEIGHT = 1024;
    private BufferedImage canvas;
    private WritableRaster raster;
    private DataBufferInt dataBuffer;
    private int[] pixels;
    private Accelerator acc;
   // private F32x3Array2D arr;
    private boolean usePixels;
    private S32RGBAImage image;
    private float[] starX, starY, starZ;
    private final int NUM_STARS = 500000;
    static int FAR = 700;
    static int MID = 300;
    static int NEAR = 100;

    public DirectRasterFrame(Accelerator acc, boolean usePixels) {
        this.acc = acc;
        this.usePixels = usePixels;
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        this.image = S32RGBAImage.create(acc,WIDTH,HEIGHT);
       // this.arr = F32x3Array2D.create(acc, WIDTH,HEIGHT);
       // this.arr.clear();
        // Initialize the direct-access buffer
        this.canvas = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        this.raster = canvas.getRaster();
        this.dataBuffer = ((DataBufferInt) raster.getDataBuffer());
        this.pixels = dataBuffer.getData();

        // Initialize stars with random positions
        this.starX = new float[NUM_STARS];
        this.starY = new float[NUM_STARS];
        this.starZ = new float[NUM_STARS];
        for (int i = 0; i < NUM_STARS; i++) {
            this.starX[i] = (float) (Math.random() * 2 - 1) * WIDTH;
            this.starY[i] = (float) (Math.random() * 2 - 1) * HEIGHT;
            this.starZ[i] = (float) (Math.random() * WIDTH);
        }
    }

    public void run() {
        while (true) {
            long startNs = System.nanoTime();
            if (usePixels) {
                Arrays.fill(pixels, 0);

                for (int i = 0; i < NUM_STARS; i++) {
                    starZ[i] -= 4.0f; // Move star closer to camera
                    if (starZ[i] <= 0) starZ[i] = WIDTH; // Reset star depth

                    // Perspective projection: Divide by Z
                    int px = (int) (starX[i] / (starZ[i] / WIDTH)) + WIDTH / 2;
                    int py = (int) (starY[i] / (starZ[i] / WIDTH)) + HEIGHT / 2;

                    // 3. Bounds check and direct write
                    if (px > 1 && px < WIDTH - 2 && py > 1 && py < HEIGHT - 2) {
                        // Calculate brightness based on depth (Z)
                        int brightness = (int) (255 - (starZ[i] / WIDTH * 255));
                        int color = (brightness << 16) | (brightness << 8) | brightness;
                        pixels[py * WIDTH + px] = color;
                        if (starZ[i] < FAR) {
                            //System.out.println(starZ[i]);
                            pixels[py * WIDTH + px + 1] = color;
                            pixels[py * WIDTH + px - 1] = color;
                            pixels[(py + 1) * WIDTH + px] = color;
                            pixels[(py - 1) * WIDTH + px] = color;
                            if (starZ[i] < MID) {
                                pixels[(py - 1) * WIDTH + px + 1] = color;
                                pixels[(py - 1) * WIDTH + px - 1] = color;
                                pixels[(py + 1) * WIDTH + px + 1] = color;
                                pixels[(py + 1) * WIDTH + px - 1] = color;
                                if (starZ[i] < NEAR) {
                                    pixels[(py - 2) * WIDTH + px + 2] = color;
                                    pixels[(py - 2) * WIDTH + px - 2] = color;
                                    pixels[(py + 2) * WIDTH + px + 2] = color;
                                    pixels[(py + 2) * WIDTH + px - 2] = color;
                                }
                            }
                        }
                    }
                }
            } else{
                MappableIface.getMemorySegment(image).fill((byte)0x10);

                for (int i = 0; i < NUM_STARS; i++) {
                    starZ[i] -= 4.0f; // Move star closer to camera
                    if (starZ[i] <= 0) starZ[i] = WIDTH; // Reset star depth
                    // Perspective projection: Divide by Z
                    int px = (int) (starX[i] / (starZ[i] / WIDTH)) + WIDTH / 2;
                    int py = (int) (starY[i] / (starZ[i] / WIDTH)) + HEIGHT / 2;

                    // 3. Bounds check and direct write
                    if (px > 1 && px < WIDTH - 2 && py > 1 && py < HEIGHT - 2) {
                        // Calculate brightness based on depth (Z)
                        int brightness = (int) (255 - (starZ[i] / WIDTH * 255));
                        int color = (brightness << 16) | (brightness << 8) | brightness;
                        int pos = ((py * WIDTH) + px);
                        image.data(pos,color);
                        if (starZ[i] < FAR) {
                            image.data(pos+1,color);
                            image.data(pos-1,color);
                            image.data(pos+WIDTH,color);
                            image.data(pos-WIDTH,color);
                            if (starZ[i] < MID) {
                                image.data(pos+WIDTH+1,color);
                                image.data(pos+WIDTH-1,color);
                                image.data(pos-WIDTH+1,color);
                                image.data(pos-WIDTH-1,color);
                                if (starZ[i] < NEAR) {
                                    image.data(pos+WIDTH*2+2,color);
                                    image.data(pos+WIDTH*2-2,color);
                                    image.data(pos-WIDTH*2+2,color);
                                    image.data(pos-WIDTH*2-2,color);
                                }
                            }
                        }
                    }
                }
                    image.syncToRasterDataBuffer(dataBuffer);

            }

            long endNs = System.nanoTime();
            System.out.println((endNs-startNs)/1000000+"ms");

            // 4. Request UI to draw the modified image
            repaint();

            try { Thread.sleep(8); } catch (Exception e) {}
        }
    }

    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2d = (Graphics2D) g;
        g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2d.drawImage(canvas, 0, 0, null);
    }

    public static void main(String[] args) {
        JFrame f = new JFrame("Direct Pixel Starfield");
        DirectRasterFrame fs = new DirectRasterFrame(new Accelerator(MethodHandles.lookup()), true);
        f.add(fs);
        f.pack();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setVisible(true);
        new Thread(fs).start();
    }
}
