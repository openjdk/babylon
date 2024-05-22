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
package mandel;


import hat.buffer.S32Array2D;

import javax.swing.JComponent;
import javax.swing.JFrame;
import javax.swing.WindowConstants;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.MemorySegment;

import static java.lang.foreign.ValueLayout.JAVA_INT;

public class MandelViewer extends JFrame {

    public static class PointF32{
        public final float x;
        public final float y;

        public PointF32(float x, float y) {
            this.x =x;
            this.y = y;
        }
    }

    public  final ImageViewer imageViewer;



    public static class ImageViewer extends JComponent{

        public PointF32 getZoomPoint(float scale) {
            waitForDoorbell();
            return new PointF32(
                    ((float) (to.x - (image.getWidth() / 2)) / image.getWidth()) * scale,
                    ((float) (to.y - (image.getHeight() / 2)) / image.getHeight()) * scale);
        }
        final  BufferedImage image;
        private final Object doorBell = new Object();
        public Point to = null;
        ImageViewer(BufferedImage image){
            super();
            this.image = image;
            addMouseListener(new MouseAdapter() {
                @Override
                public void mouseClicked(MouseEvent e) {
                    to = e.getPoint();
                    synchronized (doorBell) {
                        doorBell.notify();
                    }
                }
            });
            to = new Point(image.getWidth() / 2, image.getHeight() / 2);
        }
        @Override
        public Dimension getPreferredSize() {
            return new Dimension(image.getWidth(),image.getHeight());
        }

        @Override
        public void paintComponent(Graphics g1d) {
            super.paintComponent(g1d);
            Graphics2D g = (Graphics2D) g1d;
            g.drawImage(image, 0, 0, image.getWidth(),image.getHeight(), this);
        }

        public void waitForDoorbell() {
            to = null;
            while (to == null) {
                synchronized (doorBell) {
                    try {
                        doorBell.wait();
                    } catch (final InterruptedException ie) {
                        ie.getStackTrace();
                    }
                }
            }
        }
        public void syncWithRGB(S32Array2D s32Array2D) {
            long offset = s32Array2D.layout().byteOffset(MemoryLayout.PathElement.groupElement("array"));
            MemorySegment.copy(s32Array2D.memorySegment(), JAVA_INT, offset, ((DataBufferInt) image.getRaster().getDataBuffer()).getData(), 0, s32Array2D.size());
            this.repaint();
        }


    }
    public MandelViewer(String title, S32Array2D s32Array2D) {
        super(title);

        this.imageViewer = new ImageViewer(new BufferedImage(s32Array2D.width(), s32Array2D.height(), BufferedImage.TYPE_INT_RGB));
        this.getContentPane().add(this.imageViewer);
        this.pack();
        this.setLocationRelativeTo(null);
        this.setVisible(true);
        this.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
    }
}
