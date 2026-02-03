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
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import hat.buffer.S32RGBAImage;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import javax.swing.JFrame;
import javax.swing.JPanel;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.awt.image.WritableRaster;
import java.lang.invoke.MethodHandles;
import java.util.stream.IntStream;

import static shade.vec4.vec4;

public class NBodySansGL extends JFrame implements Runnable {
    public enum Mode {HAT,JavaMt, JavaSeq}
    public static class DirectRasterPanel extends JPanel {
        private BufferedImage bufferedImage;
        private WritableRaster writableRaster;
        private DataBufferInt dataBuffer;
        private S32RGBAImage image;

        public DirectRasterPanel(Accelerator acc, int width, int height) {
            setPreferredSize(new Dimension(width, height));
            this.image = S32RGBAImage.create(acc, width, height);
            this.bufferedImage = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            this.writableRaster = bufferedImage.getRaster();
            this.dataBuffer = ((DataBufferInt) writableRaster.getDataBuffer());
        }

        @Override
        protected void paintComponent(Graphics g) {
            Graphics2D g2d = (Graphics2D) g;
            g2d.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
            g2d.drawImage(bufferedImage, 0, 0, null);
        }
    }

    final Accelerator acc;
    final Mode mode;

    final float delT = .9f;
    final int width;
    final int height;
    final float espSqr = 0.9f;

    final float mass = .9f;
    final int bodies;

    final F32Array xyzPosFloatArr;
    final F32Array xyzVelFloatArr;
    DirectRasterPanel panel;

    public NBodySansGL(Accelerator acc, Mode mode,  int bodyCount, int width, int height) {
        super("NBody Sans OpenGL");
        this.mode = mode;
        this.acc = acc;
        this.bodies = bodyCount;
        this.width = width;
        this.height = height;
        this.xyzPosFloatArr = F32Array.create(acc, bodies * 4);
        this.xyzVelFloatArr = F32Array.create(acc, bodies * 4);
        panel = new DirectRasterPanel(acc, width, height);
        final float maxDist = width / 2;

        System.out.println(bodies + " particles");

        for (int body = 0; body < bodies; body++) {
            final float theta = (float) (Math.random() * Math.PI * 2);
            final float phi = (float) (Math.random() * Math.PI * 2);
            final float radius = (float) (Math.random() * maxDist);

            var radial = vec4(
                    (float) (radius * Math.cos(theta) * Math.sin(phi) + width / 2),
                    (float) (radius * Math.sin(theta) * Math.sin(phi) + height / 2),
                    (float) (radius * Math.cos(phi) + Math.min(width, height) / 2),
                    0f);
            xyzPosFloatArr.array(body * 4 + 0, radial.x());
            xyzPosFloatArr.array(body * 4 + 1, radial.y());
            xyzPosFloatArr.array(body * 4 + 2, radial.z());
        }

        add(panel);
        pack();
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setVisible(true);
        new Thread(this).start();
    }

    @Reflect
    public static void run(int bodyIdx, int bodies, @MappableIface.RW F32Array xyzPos, @MappableIface.RW F32Array xyzVel, @MappableIface.RW S32RGBAImage image, int imageWidth, float mass, float delT, float espSqr) {
        final int FAR = 500;
        final int MID = 300;
        final int NEAR = 100;
        float accx = 0.f;
        float accy = 0.f;
        float accz = 0.f;

        final float myPosx = xyzPos.array(bodyIdx * 4 + 0);
        final float myPosy = xyzPos.array(bodyIdx * 4 + 1);
        final float myPosz = xyzPos.array(bodyIdx * 4 + 2);
        final float myVelx = xyzVel.array(bodyIdx * 4 + 0);
        final float myVely = xyzVel.array(bodyIdx * 4 + 1);
        final float myVelz = xyzVel.array(bodyIdx * 4 + 2);

        for (int body = 0; body < bodies; body++) {
            final float dx = xyzPos.array(body * 4 + 0) - myPosx;
            final float dy = xyzPos.array(body * 4 + 1) - myPosy;
            final float dz = xyzPos.array(body * 4 + 2) - myPosz;
            final float invDist = 1 / (float) Math.sqrt((dx * dx) + (dy * dy) + (dz * dz) + espSqr);
            final float s = mass * invDist * invDist * invDist;
            accx = accx + (s * dx);
            accy = accy + (s * dy);
            accz = accz + (s * dz);
        }
        accx = accx * delT;
        accy = accy * delT;
        accz = accz * delT;

        float fx = myPosx + (myVelx + accx * .5f) * delT;
        float fy = myPosy + (myVely + accx * .5f) * delT;
        float fz = myPosz + (myVelz + accx * .5f) * delT;
        xyzPos.array(bodyIdx * 4 + 0, fx);
        xyzPos.array(bodyIdx * 4 + 1, fy);
        xyzPos.array(bodyIdx * 4 + 2, fz);

        xyzVel.array(bodyIdx * 4 + 0, myVelx + accx);
        xyzVel.array(bodyIdx * 4 + 1, myVely + accy);
        xyzVel.array(bodyIdx * 4 + 2, myVelz + accz);

        int x = (int) fx;
        int y = (int) fy;
        int z = (int) fz;
        if (x > 1 && x < imageWidth - 2 && y > 1 && y < imageWidth - 2) {
            // Calculate brightness based on depth (Z)
            int brightness = (255 - (z / imageWidth * 255));
            int color = (brightness << 16) | (brightness << 8) | brightness;
            int pos = ((y * imageWidth) + x);
            image.data(pos, color);
            if (z < FAR) {
                image.data(pos + 1, color);
                image.data(pos - 1, color);
                image.data(pos + imageWidth, color);
                image.data(pos - imageWidth, color);
                if (z < MID) {
                    image.data(pos + imageWidth + 1, color);
                    image.data(pos + imageWidth - 1, color);
                    image.data(pos - imageWidth + 1, color);
                    image.data(pos - imageWidth - 1, color);
                    if (z < NEAR) {
                        image.data(pos + imageWidth * 2 + 2, color);
                        image.data(pos + imageWidth * 2 - 2, color);
                        image.data(pos - imageWidth * 2 + 2, color);
                        image.data(pos - imageWidth * 2 - 2, color);
                    }
                }
            }
        }
    }

    @Reflect

    static public void nbodyKernel(
            @MappableIface.RO KernelContext kc,
            int bodies,
            @MappableIface.RW F32Array xyzPos,
            @MappableIface.RW F32Array xyzVel,
            @MappableIface.RW S32RGBAImage image,
            int imageWidth,
            float mass,
            float delT,
            float espSqr
    ) {
        run(kc.gix, bodies, xyzPos, xyzVel, image, imageWidth,  mass, delT, espSqr);
    }
    @Reflect

    static public void clearImage(
            @MappableIface.RO KernelContext kc,
            @MappableIface.RW S32RGBAImage image
    ) {
        image.data(kc.gix, 0);
    }
    @Reflect
    public static void nbodyCompute(
            @MappableIface.RO ComputeContext cc,
            int bodies,
            @MappableIface.RW F32Array xyzPos,
            @MappableIface.RW F32Array xyzVel,
            @MappableIface.RW S32RGBAImage image,
            int imageWidth,
            float mass,
            float delT,
            float espSqr
    ) {
        float cmass = mass;
        float cdelT = delT;
        float cespSqr = espSqr;
        int cbodies = bodies;
        int cimageWidth = imageWidth;

        cc.dispatchKernel(NDRange.of1D(imageWidth* image.height()), kc->clearImage(kc, image));

        cc.dispatchKernel(NDRange.of1D(bodies), kc -> nbodyKernel(kc, cbodies, xyzPos, xyzVel, image, cimageWidth,  cmass, cdelT, cespSqr));
    }




    public void run() {
        while (true) {
            long startNs = System.nanoTime();
            float cmass = mass;
            float cdelT = delT;
            float cespSqr = espSqr;
            int cbodies = bodies;
            int cimageWidth = width;
            F32Array cxyzPosFloatArr = xyzPosFloatArr;
            F32Array cxyzVelFloatArr = xyzVelFloatArr;
            S32RGBAImage cimage = panel.image;

            switch (mode){
                case HAT -> {
                    acc.compute((@Reflect Compute)
                            cc -> nbodyCompute(cc, cbodies, cxyzPosFloatArr, cxyzVelFloatArr, cimage, cimageWidth, cmass, cdelT, cespSqr));
                }
                case JavaMt -> {
                    MappableIface.getMemorySegment(panel.image).fill((byte) 0x00); // Dont do this if using HAT! ;)
                    IntStream.range(0, bodies).parallel().forEach(
                            i -> run(i, cbodies, cxyzPosFloatArr, cxyzVelFloatArr, cimage, cimageWidth, cmass, cdelT, cespSqr));
                }
                case JavaSeq -> {
                    MappableIface.getMemorySegment(panel.image).fill((byte) 0x00); // Dont do this if using HAT! ;)
                    IntStream.range(0, bodies).forEach(
                            i -> run(i, cbodies, cxyzPosFloatArr, cxyzVelFloatArr, cimage, cimageWidth, cmass, cdelT, cespSqr));
                }
            }
            panel.image.syncToRasterDataBuffer(panel.dataBuffer);

            long endNs = System.nanoTime();
            System.out.println((endNs - startNs) / 1000000 + "ms");
            repaint();
            try {
                Thread.sleep(1);
            } catch (Exception e) {
            }
        }
    }

    static void main(String[] args) {
        new NBodySansGL(new Accelerator(MethodHandles.lookup()),
                args.length>0?
                        switch (args[0]){
                            case "HAT"->Mode.HAT;
                            case "JavaMT"->Mode.JavaMt;
                            case "JavaSeq"->Mode.JavaSeq;
                            default -> throw new IllegalStateException("No such mode as "+args[0]);
                        }:Mode.HAT
                , 4096*4 , 1024, 1024);

    }

}
