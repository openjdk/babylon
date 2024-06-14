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
package violajones;


import hat.Accelerator;
import hat.buffer.BufferAllocator;
import hat.buffer.F32Array2D;
import hat.buffer.S32Array;
import violajones.buffers.GreyU16Image;
import violajones.buffers.RgbS08x3Image;
import violajones.ifaces.Cascade;
import violajones.ifaces.ResultTable;
import violajones.ifaces.ScaleTable;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.WindowConstants;
import java.awt.BasicStroke;
import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Container;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;

public class HaarViewer extends JFrame {
    final BufferedImage image;
    final RgbS08x3Image rgbS08x3Image;


    public static class IntegralWindow {
        final double integralScale = .25;
        BufferedImage integral;
        BufferedImage integralSq;
        GreyU16Image integralImageU16;
        GreyU16Image integralSqImageU16;
        JComponent integralImageView;
        JComponent integralSqImageView;
        final F32Array2D integralImageF32;
        final F32Array2D integralSqImageF32;

        public IntegralWindow(Container container, BufferAllocator bufferAllocator, F32Array2D integralImageF32, F32Array2D integralSqImageF32) {
            this.integralImageF32 = integralImageF32;
            this.integralSqImageF32 = integralSqImageF32;

            if (integralImageF32 != null && integralSqImageF32 != null) {
                int width = this.integralImageF32.width();
                int height = this.integralImageF32.height();
                this.integral = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
                this.integralSq = new BufferedImage(width, height, BufferedImage.TYPE_USHORT_GRAY);
                this.integralImageU16 = GreyU16Image.create(bufferAllocator, integral);
                this.integralSqImageU16 = GreyU16Image.create(bufferAllocator, integralSq);
                this.integralImageView = new JComponent() {
                    @Override
                    public void paint(Graphics g) {
                        Graphics2D g2 = (Graphics2D) g;
                        g2.scale(integralScale, integralScale);
                        g2.drawImage(integral, 0, 0, null);
                    }

                    @Override
                    public Dimension getPreferredSize() {
                        return new Dimension((int) (width * integralScale),
                                (int) (height * integralScale));
                    }
                };
                this.integralSqImageView = new JComponent() {
                    @Override
                    public void paint(Graphics g) {
                        Graphics2D g2 = (Graphics2D) g;
                        g2.scale(integralScale, integralScale);
                        g2.drawImage(integralSq, 0, 0, null);
                    }

                    @Override
                    public Dimension getPreferredSize() {
                        return new Dimension((int) (width * integralScale),
                                (int) (height * integralScale));
                    }
                };
                JPanel integralPanel = new JPanel();
                integralPanel.add(integralImageView);
                integralPanel.add(integralSqImageView);
                container.add(integralPanel, BorderLayout.SOUTH);
            }

        }

        public void show() {
            // There is no Java image of floats.
            // So we normalize a short grey array
            // This is slow, but we don't really care.  We could of course  use a kernel to do this.

            /*
             .kernel("floatToShortKernel").ptr("cascadeContext", Cascade.layout)
                .ptr("fromIntegral", JAVA_FLOAT)
                .ptr("toIntegral", JAVA_SHORT)
                .ptr("fromIntegralSq", JAVA_FLOAT)
                .ptr("toIntegralSq", JAVA_SHORT).body("""

                     toIntegral[ndrange.id.x] = (s16_t)(fromIntegral[ndrange.id.x]*(65536/fromIntegral[ndrange.id.maxX-1]));
                     toIntegralSq[ndrange.id.x] = (s16_t)(fromIntegralSq[ndrange.id.x]*(65536/fromIntegralSq[ndrange.id.maxX-1]));

                """)
             */
            long lastIdx = (long) this.integralImageF32.width() * this.integralImageF32.height() - 1;
            float maxAsFloat = this.integralImageF32.array(lastIdx);
            float maxValue = 65536 / maxAsFloat;
            float maxSqAsFloat = this.integralSqImageF32.array(lastIdx);
            float maxSqValue = 65536 / maxSqAsFloat;
            for (long i = 0; i < lastIdx; i++) {
                integralImageU16.data(i, (short) (integralImageF32.array(i) * maxValue));
                integralSqImageU16.data(i, (short) (integralSqImageF32.array(i) * maxSqValue));
            }
            this.integralImageU16.syncToRaster(integral);
            this.integralSqImageU16.syncToRaster(integralSq);
            this.integralImageView.repaint();
            this.integralSqImageView.repaint();
        }
    }

    final IntegralWindow integralWindow;
    final Cascade cascade;

    final JComponent imageView;

    ResultTable resultTable;
    S32Array resultIds;
    ScaleTable scaleTable;

    public void showResults(ResultTable resultTable, ScaleTable scaleTable, S32Array resultIds) {
        this.resultTable = resultTable;
        this.scaleTable = scaleTable;
        this.resultIds = resultIds;
        this.imageView.repaint();
    }

    public void showIntegrals() {
        if (integralWindow != null) {
            integralWindow.show();
        }
    }

    final double imageScale = .5;


    public HaarViewer(BufferAllocator bufferAllocator,
                      BufferedImage image,
                      RgbS08x3Image rgbS08x3Image,
                      Cascade cascade,
                      F32Array2D integralImageF32,
                      F32Array2D integralSqImageF32
    ) {
        super("HaarViz");
        this.image = image;
        this.rgbS08x3Image = rgbS08x3Image;
        this.cascade = cascade;

        this.setLayout(new BorderLayout());
        this.imageView = new JComponent() {
            @Override
            public void paint(Graphics g) {
                Graphics2D g2 = (Graphics2D) g;
                g2.scale(imageScale, imageScale);
                g2.drawImage(HaarViewer.this.image, 0, 0, null);

                if (resultTable != null && resultTable.atomicResultTableCount() > 0) {
                    g2.setStroke(new BasicStroke(2f));
                    g2.setColor(Color.red);
                    for (int i = 0; i < resultTable.atomicResultTableCount(); i++) {
                        if (i < resultTable.length()) {
                            ResultTable.Result result = resultTable.result(i);
                            g2.drawString(Integer.toString(i), result.x() - 10, result.y() - 5);
                            g2.draw(new Rectangle((int) result.x(), (int) result.y(),
                                    (int) result.width(), (int) result.height()));
                        } else {
                            System.out.println("more than " + resultTable.length() + " found");
                            break;
                        }
                    }

                }
                if (resultIds != null && scaleTable != null) {
                    g2.setStroke(new BasicStroke(2f));
                    g2.setColor(Color.red);
                    for (int i = 0; i < resultIds.length(); i++) {
                        // s32Array.copyTo();
                        int v = resultIds.array(i);
                        if (v != 0) {
                            int scalc = 0;
                            ScaleTable.Scale scale = scaleTable.scale(scalc++);
                            while (i >= scale.accumGridSizeMax()) {
                                scale = scaleTable.scale(scalc++);
                            }
                            int scaleGid = i - scale.accumGridSizeMin();

                            int x = (int) ((scaleGid % scale.gridWidth()) * scale.scaledXInc());
                            int y = (int) ((scaleGid / scale.gridWidth()) * scale.scaledYInc());
                            int w = scale.scaledFeatureWidth();
                            int h = scale.scaledFeatureHeight();
                            g2.drawString(Integer.toString(i), x - 10, y - 5);
                            g2.draw(new Rectangle(x, y, w, h));
                            // We have to map v's idx to a scaled x,y,w,h

                        }

                    }

                }
            }

            @Override
            public Dimension getPreferredSize() {
                return new Dimension((int) (HaarViewer.this.image.getWidth() * imageScale),
                        (int) (HaarViewer.this.image.getHeight() * imageScale));
            }
        };
        JPanel gridPanel = new JPanel();
        JPanel imagePanel = new JPanel();
        gridPanel.add(imageView);
        add(gridPanel, BorderLayout.CENTER);
        add(imagePanel, BorderLayout.EAST);
        this.integralWindow = new IntegralWindow(this, bufferAllocator, integralImageF32, integralSqImageF32);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        pack();
        setVisible(true);
    }

}
