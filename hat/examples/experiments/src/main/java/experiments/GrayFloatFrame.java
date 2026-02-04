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

import java.awt.BorderLayout;
import java.awt.Canvas;
import java.awt.Color;
import java.awt.Frame;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.ComponentSampleModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferFloat;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;

public class GrayFloatFrame {


     static void main(String[] args) {
        final int width = 1024;
        final int height = 1024;

        SampleModel floatSampleModel = new ComponentSampleModel(DataBuffer.TYPE_FLOAT, width, height, 1, width, new int[]{0});
        DataBuffer floatDataBuffer = new DataBufferFloat(width * height);
        WritableRaster floatWritableRaster = Raster.createWritableRaster(floatSampleModel, floatDataBuffer, null);
        ColorSpace grayColorSpace = ColorSpace.getInstance(ColorSpace.CS_GRAY);
        ColorModel grayColorModel = new ComponentColorModel(grayColorSpace, false, true, Transparency.OPAQUE, DataBuffer.TYPE_FLOAT);
        final BufferedImage image = new BufferedImage(grayColorModel, floatWritableRaster, true, null);

        Graphics2D g2 = image.createGraphics();
                g2.setColor(Color.BLACK);
        g2.fillRect(0, 0, width, height);
        g2.setColor(Color.RED);
        g2.drawLine(0, 0, width, height);
        g2.drawLine(width, 0, 0, height);
        g2.drawOval(width / 4, height / 4, width / 2, height / 2);
        g2.dispose();

        Frame f = new Frame("GrayFloatFrame");
        f.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent we) {
                System.exit(0);
            }
        });
        f.setLayout(new BorderLayout());
        Canvas c = new Canvas() {
            public void paint(Graphics g) {
                g.drawImage(image, 0, 0, null);
            }
        };
        c.setSize(width, height);
        f.add(c, BorderLayout.CENTER);
        f.pack();
        f.show();
    }
}
