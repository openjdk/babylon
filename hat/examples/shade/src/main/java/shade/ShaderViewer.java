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
import hat.buffer.F32Array;
import hat.buffer.Uniforms;
import hat.types.vec2;
import hat.types.vec4;

import javax.swing.JComponent;
import javax.swing.JFrame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Toolkit;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.event.ComponentAdapter;
import java.awt.event.ComponentEvent;
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
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.stream.IntStream;


public  class ShaderViewer {


    public interface Shader{
        void update(Uniforms uniforms, F32Array f32Array);
    }
    public JComponent view;
    private VolatileImage volatileImage;
    private BufferedImage buffer;
    private float[] f32x3Arr;
    private volatile boolean resized = true;
    private Accelerator acc;
    private Uniforms uniforms;
    private F32Array f32Array;
    private int width;
    private int height;
    private boolean useHat;
    private Class shaderClass;
    private Method mainImageMethod;


    ShaderViewer(Accelerator acc, Class shaderClass, int width, int height, boolean useHat) {
        this.acc = acc;
        this.shaderClass = shaderClass;
        try {
            this.mainImageMethod = shaderClass.getDeclaredMethod("mainImage", Uniforms.class, vec4.class, vec2.class);
        }catch (NoSuchMethodException e){
            throw new RuntimeException(e);
        }
        this.uniforms = Uniforms.create(acc);
        this.f32Array = F32Array.create(acc, width * height * 3);
        this.width = width;
        this.height = height;
        this.useHat = useHat;
        this.view = new JComponent() {
        };
        this.view.setSize(width, height);
        this.view.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                resized = true;
            }
        });
    }

    public void startLoop(Shader shader) {
        new Thread(() -> {
            uniforms.iFrame(0);
            long startTimeNs = System.nanoTime();
            while (true) {
                do {
                    // Check if the volatileImage content was lost (resize or invalid)
                    if (f32x3Arr == null || resized || volatileImage == null || volatileImage.validate(view.getGraphicsConfiguration()) == VolatileImage.IMAGE_INCOMPATIBLE) {
                        resized = false;
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
                        volatileImage = view.createVolatileImage(width, height);
                    }

                    long startNs = System.nanoTime();
                    var mouse = view.getMousePosition() instanceof Point point ? point : new Point(0, 0);

                    uniforms.iTime( (System.nanoTime() - startTimeNs) / 1000000000f);
                    uniforms.iMouse().x(mouse.x);
                    uniforms.iMouse().y(mouse.y);
                    uniforms.iResolution().x(width);
                    uniforms.iResolution().y(height);
                    if (useHat) {
                        shader.update(uniforms, f32Array);
                    }else {
                        IntStream.range(0, width * height).parallel().forEach(idx -> {
                            vec2 fragCoord = vec2.vec2((float) (idx % width), (float)  (idx / width));
                            vec4 fragColor = null;
                            try {
                                fragColor = (vec4) mainImageMethod.invoke(null,uniforms, vec4.vec4(1f), fragCoord);
                            } catch (IllegalAccessException e) {
                                throw new RuntimeException(e);
                            } catch (InvocationTargetException e) {
                                throw new RuntimeException(e);
                            }
                            f32Array.array(idx * 3, fragColor.x());
                            f32Array.array(idx * 3 + 1, fragColor.y());
                            f32Array.array(idx * 3 + 2, fragColor.z());
                        });
                    }
                    f32Array.copyTo(f32x3Arr);
                    uniforms.iFrame(uniforms.iFrame()+1);
                    long endNs = System.nanoTime();
                //    System.out.println((endNs - startNs) / 1000000);

                    Graphics2D volatileGraphics2D = volatileImage.createGraphics();
                    volatileGraphics2D.drawImage(buffer, 0, 0, null);
                    volatileGraphics2D.dispose();


                    Graphics g = view.getGraphics();
                    g.drawImage(volatileImage, 0, 0, null);
                    g.dispose();

                    Toolkit.getDefaultToolkit().sync();// Ensure smooth rendering on Linux/macOS
                } while (volatileImage == null || volatileImage.contentsLost());

            }
        }).start();
    }

    public static ShaderViewer of(Accelerator acc, Class<?> shaderClass, int width, int height, boolean useHat){
        var shader =   new ShaderViewer(acc,shaderClass, width,height, useHat);
        JFrame frame = new JFrame(shaderClass.getSimpleName());
        frame.setSize(shader.view.getWidth(),shader.view.getHeight());
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(shader.view);
        frame.setVisible(true);
        return shader;
    }

}
