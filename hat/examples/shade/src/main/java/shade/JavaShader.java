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
import hat.types.vec4;
import shade.shaders.GroovyShader;
import shade.shaders.IntroShader;
import shade.shaders.JuliaShader;
import shade.shaders.MobiusShader;
import shade.shaders.MouseSensitiveShader;
import shade.shaders.PaintShader;
import shade.shaders.SeaScapeShader;
import shade.shaders.SpiralShader;
import shade.shaders.Truchet2Shader;
import shade.shaders.TruchetShader;
import shade.shaders.TutorialShader;
import shade.shaders.WavesShader;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.Transparency;
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

import static hat.types.vec2.vec2;


public class JavaShader extends JPanel {
    private VolatileImage volatileImage;
    private BufferedImage buffer;
    private float[] f32x3Arr;
    private float ftime;
    private int width;
    private int height;

    JavaShader(int width, int height) {
        this.width = width;
        this.height = height;
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

        buffer = new BufferedImage(colorModel, raster, false, null);
        f32x3Arr = dataBufferFloat.getData();
        volatileImage = createVolatileImage(width, height);
        startLoop();
    }

    public void startLoop() {
        new Thread(() -> {
            long startTimeNs = System.nanoTime();
            while (true) {
                updateSimulation((System.nanoTime() - startTimeNs) / 1000000000f);
                do {
                    // Check if the volatileImage content was lost (resize or invalid)
                    if (volatileImage == null || volatileImage.validate(getGraphicsConfiguration()) == VolatileImage.IMAGE_INCOMPATIBLE) {
                        volatileImage = createVolatileImage(width, height);
                    }

                    Graphics2D volatileGraphics2D = volatileImage.createGraphics();
                    volatileGraphics2D.drawImage(buffer, 0, 0, null);
                    volatileGraphics2D.dispose();


                    Graphics g = this.getGraphics();
                    //  if (g != null) {
                    g.drawImage(volatileImage, 0, 0, null);
                    g.dispose();
                    // }

                } while (volatileImage.contentsLost());
                Toolkit.getDefaultToolkit().sync();// Ensure smooth rendering on Linux/macOS
            }
        }).start();
    }


    public void updateSimulation(float elapsed) {
        ftime = elapsed;
        vec2 fres = vec2(width, height);
        JComponent c = this;
        var p = c.getMousePosition();
        if (p == null){
            p = new Point(0,0);
        }
     //   System.out.println("mouse "+p.x+","+p.y);
        vec2 fmouse = vec2(p.x,p.y);
        long startNs = System.nanoTime();
        IntStream.range(0, width * height).parallel().forEach(idx -> {
            int invertedHeight = idx / width;
            vec2 fragCoord = vec2.vec2((float) idx % width, (float) (height - invertedHeight));
                vec4 col =
                        //WavesShader.createPixel(fres,ftime,fmouse,fragCoord);
                        // SeaScapeShader.createPixel(fres,ftime,fmouse,fragCoord);
                        // Truchet2Shader.createPixel(fres,ftime,fmouse,fragCoord);
           // TruchetShader.createPixel(fres,ftime,fmouse,fragCoord);
                        GroovyShader.createPixel(fres,ftime,fmouse,fragCoord);
           // IntroShader.createPixel(fres,ftime,fmouse,fragCoord);
          // MouseSensitiveShader.createPixel(fres,ftime,fmouse,fragCoord);
            //   PaintShader.createPixel(fres,ftime,fmouse,fragCoord);
           // MobiusShader.createPixel(fres,ftime,fmouse,fragCoord);
           //  JuliaShader.createPixel(fres,ftime,fmouse,fragCoord);
                   //     SpiralShader.createPixel(fres, ftime, fmouse,fragCoord);
            // TutorialShader.createPixel(fres, ftime, fmouse,fragCoord);
            f32x3Arr[idx * 3] = col.x();
            f32x3Arr[idx * 3 + 1] = col.y();
            f32x3Arr[idx * 3 + 2] = col.z();
        });
        long endNs = System.nanoTime();
        System.out.println((endNs - startNs) / 1000000);
    }


    public static void main(String[] args) {
        JFrame frame = new JFrame("Java Pixel Shader");
        int width =1024;
        int height=1024;
        frame.setSize(width, height);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(new JavaShader(width, height));
        frame.setVisible(true);
    }
}
