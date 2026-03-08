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

import hat.types.vec2;
import static hat.types.vec2.vec2;
import hat.types.vec4;
import shade.shaders.JuliaShader;
import shade.shaders.TruchetShader;

import javax.swing.*;
import java.awt.*;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferFloat;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.VolatileImage;
import java.awt.image.WritableRaster;
import java.util.stream.IntStream;


public class JavaShader extends JPanel {
    private VolatileImage volatileImage;
    private BufferedImage buffer;
    private float[] f32x3Arr;
    private float ftime;
    JavaShader() {
        Timer timer = new Timer(5, e -> {
            ftime += 0.05f; // Increment "time" each frame
            repaint();
        });
        timer.start();
    }
    public void renderShader(int width, int height) {
        if (buffer == null || buffer.getWidth() != width) {
            ColorSpace colorSpace = ColorSpace.getInstance(ColorSpace.CS_sRGB);

            // Create the Color Model. 32 bits per component, no alpha, non-premultiplied
            ColorModel colorModel = new ComponentColorModel(colorSpace, false, false,
                    Transparency.OPAQUE, DataBuffer.TYPE_FLOAT);

            int channels = width * height * 3;
            // Create the Sample Model (Pixel Interleaved) bands for RGB, scanline stride is width * 3
            SampleModel sampleModel = new PixelInterleavedSampleModel(DataBuffer.TYPE_FLOAT,
                    width, height, 3, width * 3, new int[]{0, 1, 2});

            // Create the DataBuffer (an actual heap allocated  float array)
            DataBufferFloat dataBufferFloat = new DataBufferFloat(width * height * 3);

            // Create the Raster
            WritableRaster raster = Raster.createWritableRaster(sampleModel, dataBufferFloat, null);

            buffer = new BufferedImage(colorModel, raster, false, null);
            f32x3Arr = dataBufferFloat.getData();
        }

        vec2 fres = vec2(width,height);
        long startNs = System.nanoTime();

        IntStream.range(0, width*height).parallel().forEach(idx -> {
            int invertedHeight = idx/width;
            vec2 fragCoord = vec2.vec2((float) idx % width, (float) (height - invertedHeight));
           // vec4 col = TruchetShader.createPixel(fres,ftime,fragCoord);
            vec4 col = JuliaShader.createPixel(fres,ftime,fragCoord);
            f32x3Arr[idx*3] = col.x();
            f32x3Arr[idx*3+1] = col.y();
            f32x3Arr[idx*3+2] = col.z() ;
        });
        long endNs = System.nanoTime();
        System.out.println((endNs-startNs)/1000000);

    }

    @Override
    protected void paintComponent(Graphics g) {
        super.paintComponent(g);
        int w = getWidth();
        int h = getHeight();
        do {
            if (volatileImage == null || volatileImage.validate(getGraphicsConfiguration()) == VolatileImage.IMAGE_INCOMPATIBLE) {
                volatileImage = createVolatileImage(w, h);
            }
            renderShader(w, h);
            boolean why= true;
            if (why) {
                // Example I cloned suggested I do this...?
                var volatileGraphics2D = (Graphics2D) volatileImage.createGraphics();
                volatileGraphics2D.drawImage(buffer, 0, 0, null);
                volatileGraphics2D.dispose();
                g.drawImage(volatileImage, 0, 0, this);
            }else {
                g.drawImage(buffer, 0, 0, this);
            }
        } while (volatileImage.contentsLost());
    }

    public static void main(String[] args) {
        JFrame frame = new JFrame("Java Pixel Shader");
        frame.add(new JavaShader());
        frame.setSize(1024, 1024);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);
    }
}
