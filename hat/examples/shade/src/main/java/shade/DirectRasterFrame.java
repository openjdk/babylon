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

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Arrays;

public class DirectRasterFrame extends JPanel implements Runnable {
    private final int WIDTH = 1024, HEIGHT = 1024;
    private BufferedImage canvas;
    private int[] pixels;
    private float[] starX, starY, starZ;
    private final int NUM_STARS = 500000;

    public DirectRasterFrame() {
        setPreferredSize(new Dimension(WIDTH, HEIGHT));
        // Initialize the direct-access buffer
        canvas = new BufferedImage(WIDTH, HEIGHT, BufferedImage.TYPE_INT_RGB);
        pixels = ((DataBufferInt) canvas.getRaster().getDataBuffer()).getData();

        // Initialize stars with random positions
        starX = new float[NUM_STARS];
        starY = new float[NUM_STARS];
        starZ = new float[NUM_STARS];
        for (int i = 0; i < NUM_STARS; i++) {
            starX[i] = (float) (Math.random() * 2 - 1) * WIDTH;
            starY[i] = (float) (Math.random() * 2 - 1) * HEIGHT;
            starZ[i] = (float) (Math.random() * WIDTH);
        }
    }

    public void run() {
        while (true) {
            Arrays.fill(pixels, 0);
         //   long startNs = System.nanoTime();
            for (int i = 0; i < NUM_STARS; i++) {
                starZ[i] -= 4.0f; // Move star closer to camera
                if (starZ[i] <= 0) starZ[i] = WIDTH; // Reset star depth

                // Perspective projection: Divide by Z
                int px = (int) (starX[i] / (starZ[i] / WIDTH)) + WIDTH / 2;
                int py = (int) (starY[i] / (starZ[i] / WIDTH)) + HEIGHT / 2;

                // 3. Bounds check and direct write
                if (px > 1 && px < WIDTH-2 && py > 1 && py < HEIGHT-2) {
                    // Calculate brightness based on depth (Z)
                    int brightness = (int) (255 - (starZ[i] / WIDTH * 255));
                    int color = (brightness << 16) | (brightness << 8) | brightness;
                    pixels[py * WIDTH + px] = color;
                    pixels[(py) * WIDTH + px] = color;
                    if (starZ[i]<700) {
                        //System.out.println(starZ[i]);
                        pixels[py * WIDTH + px + 1] = color;
                        pixels[py * WIDTH + px - 1] = color;
                        pixels[(py + 1) * WIDTH + px] = color;
                        pixels[(py - 1) * WIDTH + px] = color;
                        if (starZ[i]<300){
                            pixels[(py-1) * WIDTH + px + 1] = color;
                            pixels[(py-1) * WIDTH + px - 1] = color;
                            pixels[(py + 1) * WIDTH + px+1] = color;
                            pixels[(py + 1) * WIDTH + px-1] = color;
                            if (starZ[i]<10){
                                pixels[(py-2) * WIDTH + px + 2] = color;
                                pixels[(py-2) * WIDTH + px - 2] = color;
                                pixels[(py + 2) * WIDTH + px+2] = color;
                                pixels[(py + 2) * WIDTH + px-2] = color;
                            }
                        }
                    }
                }
            }
          //  long endNs = System.nanoTime();
         //   System.out.println((endNs-startNs)/1000000+"ms");

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
        DirectRasterFrame fs = new DirectRasterFrame();
        f.add(fs);
        f.pack();
        f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        f.setVisible(true);
        new Thread(fs).start();
    }
}
