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

import javax.swing.JFrame;
import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferFloat;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;
import java.util.stream.IntStream;

public class FloatFrame {

    public static BufferedImage createFloatImage(int width, int height) {
        // 1. Define the Color Space (sRGB is standard)
        ColorSpace cs = ColorSpace.getInstance(ColorSpace.CS_sRGB);

        // 2. Create the Color Model
        // We use 32 bits per component, no alpha, non-premultiplied
        ColorModel cm = new ComponentColorModel(cs, false, false,
                Transparency.OPAQUE, DataBuffer.TYPE_FLOAT);

        // 3. Create the Sample Model (Pixel Interleaved)
        // 3 bands for RGB, scanline stride is width * 3
        SampleModel sm = new PixelInterleavedSampleModel(DataBuffer.TYPE_FLOAT,
                width, height, 3, width * 3, new int[]{0, 1, 2});

        // 4. Create the DataBuffer (the actual float array)
        DataBufferFloat db = new DataBufferFloat(width * height * 3);

        // 5. Create the Raster and the BufferedImage
        WritableRaster raster = Raster.createWritableRaster(sm, db, null);
        return new BufferedImage(cm, raster, false, null);
    }


    public static long  fillWithData(BufferedImage img) {
        WritableRaster raster = img.getRaster();
        DataBufferFloat buffer = (DataBufferFloat) raster.getDataBuffer();
        float[] data = buffer.getData();
        int width = img.getWidth();
        int height = img.getHeight();
        long start = System.nanoTime();
        boolean useIntStream = false;

        if (useIntStream) {
            boolean useParallel = false;
            if (useParallel) {
                IntStream.range(0, width * height).parallel().forEach(i -> {
                    //    var y = i / width;
                    var x = i % width;
                    int offset = i * 3;
                    // Example: Create a horizontal gradient
                    float intensity = (float) x / width;
                    data[offset] = intensity; // Red
                    data[offset + 1] = 0.5f;      // Green
                    data[offset + 2] = 1.0f;
                });
            }else{
                IntStream.range(0, width * height).forEach(i -> {
                    //    var y = i / width;
                    var x = i % width;
                    int offset = i * 3;
                    // Example: Create a horizontal gradient
                    float intensity = (float) x / width;
                    data[offset] = intensity; // Red
                    data[offset + 1] = 0.5f;      // Green
                    data[offset + 2] = 1.0f;
                });
            }
        } else {
            // Old school
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    // Calculate the array index for the current pixel
                    int offset = (y * width + x) * 3;
                    // Example: Create a horizontal gradient
                    float intensity = (float) x / width;
                    data[offset] = intensity; // Red
                    data[offset + 1] = 0.5f;      // Green
                    data[offset + 2] = 1.0f;      // Blue
                }
            }
        }

        long end = System.nanoTime();
        return end-start;
    }

    static class Frame extends JFrame{
        final Canvas canvas;
        Frame(String name, BufferedImage image){
            super(name);
            addWindowListener(new WindowAdapter() {
                public void windowClosing(WindowEvent we) {
                    System.exit(0);
                }
            });
            setLayout(new BorderLayout());
            canvas = new Canvas() {
                public void paint(Graphics g) {
                    Graphics2D g2 = (Graphics2D)g;

                     g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                    g2.drawImage(image, 0, 0, null);
                   // System.out.print(".");
                }
            };
            canvas.setSize(image.getWidth(), image.getHeight());
            add(canvas, BorderLayout.CENTER);
            pack();
            setVisible(true);
        }
    }


    static void main(String[] args) {
        final int width = 1024;
        final int height = 1024;
        var image = createFloatImage(width, height);
        System.out.println(fillWithData(image)/ 1000000 + " ms");;
        var frame = new Frame("FloatFrame",image);
    }
}
