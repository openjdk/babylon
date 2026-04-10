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
import hat.util.ui.SevenSegmentDisplay;

import javax.imageio.ImageIO;
import javax.swing.Box;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JMenuBar;
import javax.swing.JTextField;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Image;
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
import java.io.File;
import java.io.InputStream;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.IntConsumer;
import java.util.stream.IntStream;


public  class ShaderViewer {
    public interface Shader{
        void update(Uniforms uniforms, F32Array f32Array);
    }
    private final static ColorSpace colorSpace = ColorSpace.getInstance(ColorSpace.CS_sRGB);
    // Create the Color Model. 32 bits per component, no alpha, non-premultiplied
    private  final static ColorModel colorModel = new ComponentColorModel(colorSpace, false, false,
            Transparency.OPAQUE, DataBuffer.TYPE_FLOAT);
    public JComponent view;
    private VolatileImage volatileImage;
    private BufferedImage buffer;
    private float[] f32x3Arr;
    private volatile boolean resized = false;


    public record Texture(F32Array f32Array, int width, int height){}

    private final Config config;
    public record Config(
            Accelerator acc,
            Class<?> shaderClass,
            Method mainImageMethod,
            Uniforms uniforms,
            F32Array f32Array,
            int width,
            int height,
            Texture[] textures,
            JMenuBar menuBar,
            SevenSegmentDisplay fps,
            SevenSegmentDisplay shaderTimeMs,
            JComboBox<String> runWith
       ){
        public static Config of(Accelerator acc,Class<?> shaderClass, int width, int height, InputStream... textureInputStreams){
            try {
                var mainImageMethod =
                        textureInputStreams.length==0
                                ? shaderClass.getDeclaredMethod("mainImage", Uniforms.class, vec4.class, vec2.class)
                                : shaderClass.getDeclaredMethod("mainImage", Uniforms.class, vec4.class, vec2.class, F32Array.class, int.class, int.class);

                var uniforms = Uniforms.create(acc);
                var f32Array = F32Array.create(acc, width * height * 3);
                var menuBar = new JMenuBar();
                ((JButton) menuBar.add(new JButton("Exit"))).addActionListener(_ -> System.exit(0));
                menuBar.add(new JLabel("FPS:"));
                var fps = (SevenSegmentDisplay) menuBar.add(new SevenSegmentDisplay(4, 20, menuBar.getForeground(), menuBar.getBackground()));
                menuBar.add(new JLabel("Shader Time ms:"));
                var shaderTimeMs = (SevenSegmentDisplay) menuBar.add(new SevenSegmentDisplay(6, 20, menuBar.getForeground(), menuBar.getBackground()));
                var runWith = (JComboBox<String>) menuBar.add(new JComboBox(new String[]{"HAT", "Java MT", "Seq"}));
                Texture[] textures = new Texture[textureInputStreams.length];
                for (int i = 0; i < textureInputStreams.length; i++) {


                    //Files.readAllBytes(paths[i]);
                    Image textureImage = ImageIO.read(textureInputStreams[i]);
                    int tw = textureImage.getWidth(null);
                    int th = textureImage.getHeight(null);
                    SampleModel sampleModel = new PixelInterleavedSampleModel(DataBuffer.TYPE_FLOAT,
                            tw, th, 3, tw * 3, new int[]{0, 1, 2});
                    // Create the DataBuffer (an actual heap allocated  float array)
                    DataBufferFloat dataBufferFloat = new DataBufferFloat(tw * th * 3);
                    // Create the Raster
                    WritableRaster raster = Raster.createWritableRaster(sampleModel, dataBufferFloat, null);
                    BufferedImage buffer = new BufferedImage(colorModel, raster, false, null);
                    var graphics = buffer.createGraphics();
                    graphics.drawImage(textureImage,0,0, null);
                    var floats = dataBufferFloat.getData();
                    textures[i] = new Texture(F32Array.create(acc, tw* th*3), tw, th);
                    textures[i].f32Array.copyFrom(floats);
                }

                return new Config(acc,shaderClass,mainImageMethod,uniforms,f32Array,width,height,textures,menuBar,fps, shaderTimeMs,runWith);
                }catch (Throwable t){
                throw  new RuntimeException(t);

                }
        }

    }
    ShaderViewer(Config config){
        this.config = config;
        this.view = new JComponent() {};
        this.view.setSize(config.width, config.height);
        this.view.addComponentListener(new ComponentAdapter() {
            @Override
            public void componentResized(ComponentEvent e) {
                resized = true;
            }
        });
    }

    public void startLoop(Shader shader) {

        // Create the Sample Model (Pixel Interleaved) bands for RGB, scanline stride is width * 3
        SampleModel sampleModel = new PixelInterleavedSampleModel(DataBuffer.TYPE_FLOAT,
                config.width, config.height, 3, config.width * 3, new int[]{0, 1, 2});
        // Create the DataBuffer (an actual heap allocated  float array)
        DataBufferFloat dataBufferFloat = new DataBufferFloat(config.width * config.height * 3);
        // Create the Raster
        WritableRaster raster = Raster.createWritableRaster(sampleModel, dataBufferFloat, null);
        buffer = new BufferedImage(colorModel, raster, false, null);
        f32x3Arr = dataBufferFloat.getData();
        volatileImage = view.createVolatileImage(config.width, config.height);
        new Thread(() -> {
            config.uniforms.iFrame(0);
            long startTimeNs = System.nanoTime();
            while (true) {
                do {
                    // Check if the volatileImage content was lost (resize or invalid)
                    //  if ( resized ){
                    //    throw new RuntimeException("Dont resize");
                    // }

                    long startNs = System.nanoTime();
                    var mouse = view.getMousePosition() instanceof Point point ? point : new Point(0, 0);
                    config.uniforms.iTime((System.nanoTime() - startTimeNs) / 1000000000f);
                    config.uniforms.iMouse().x(mouse.x);
                    config.uniforms.iMouse().y(mouse.y);
                    config.uniforms.iResolution().x(config.width);
                    config.uniforms.iResolution().y(config.height);
                    IntConsumer intConsumer = idx -> {
                        vec2 fragCoord = vec2.vec2((float) (idx % config.width), (float) (config.height - (idx / config.width)));
                        try {
                            vec4 fragColor = (vec4) config.mainImageMethod.invoke(null, config.uniforms, vec4.vec4(1f), fragCoord);
                            config.f32Array.array(idx * 3, fragColor.x());
                            config.f32Array.array(idx * 3 + 1, fragColor.y());
                            config.f32Array.array(idx * 3 + 2, fragColor.z());
                        } catch (IllegalAccessException | InvocationTargetException e) {
                            throw new RuntimeException(e);
                        }
                    };
                    switch (config.runWith.getSelectedIndex()) {
                        case 2 -> IntStream.range(0, config.width * config.height).forEach(intConsumer);
                        case 1 -> IntStream.range(0, config.width * config.height).parallel().forEach(intConsumer);
                        case 0 -> shader.update(config.uniforms, config.f32Array);
                    }
                    config.f32Array.copyTo(f32x3Arr);
                    config.uniforms.iFrame(config.uniforms.iFrame() + 1);
                    long shaderMs = (System.nanoTime() - startNs) / 1000000;
                    config.fps.set((int) (1000 / shaderMs));
                    config.shaderTimeMs.set((int) shaderMs);
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



    public static ShaderViewer of(Config config){
        JFrame frame = new JFrame(config.shaderClass.getSimpleName());
        frame.setJMenuBar(config.menuBar);
        var shaderViewer = new ShaderViewer(config);
        frame.setSize(shaderViewer.view.getWidth(),shaderViewer.view.getHeight());
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.add(shaderViewer.view);
        frame.setVisible(true);
        return shaderViewer;
    }

    public static ShaderViewer of(Accelerator acc, Class<?> shaderClass, int width, int height){
        return of(Config.of(acc,shaderClass,width,height));
    }
}
