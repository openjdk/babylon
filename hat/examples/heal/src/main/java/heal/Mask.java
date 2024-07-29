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
package heal;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.Arrays;

class Mask {
    public final int[] data;
    public final int width;
    public final int height;

    public Mask(Selection selection) {
        width = selection.width()+2;
        height = selection.height()+2;
        Polygon polygon = new Polygon();
        for (int i = 0; i < selection.xyList.length(); i++) { // Not parallel!!! Polygon.addPoint()
            XYList.XY xy = selection.xyList.xy(i);
            polygon.addPoint(xy.x() - selection.x1() + 1, xy.y() - selection.y1() + 1);
        }
        BufferedImage maskImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
        data = ((DataBufferInt) (maskImg.getRaster().getDataBuffer())).getData();
        Arrays.fill(data, 0);
        Graphics2D g = maskImg.createGraphics();
        g.setColor(Color.WHITE);
        g.fillPolygon(polygon);
    }
}
