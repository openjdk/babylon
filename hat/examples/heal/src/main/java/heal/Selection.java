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
/*
 * Based on code from HealingBrush renderscript example
 *
 * https://github.com/yongjhih/HealingBrush/tree/master
 *
 * Copyright (C) 2015 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package heal;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Point;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Selection {
    public static class Mask {
        public final Selection selection;
        public final int[] maskRGBData;
        public final int width;
        public final int height;
        public final Polygon polygon;
        private Mask(Selection selection) {
            this.selection = selection;
            width = selection.width()+2;
            height = selection.height()+2;

            this.polygon = new Polygon();
            selection.pointList.forEach(p->
                    polygon.addPoint(p.x- selection.x1() + 1,p.y- selection.y1() + 1)
            );

            BufferedImage maskImg = new BufferedImage(width, height, BufferedImage.TYPE_INT_RGB);
            maskRGBData = ((DataBufferInt) (maskImg.getRaster().getDataBuffer())).getData();
            Arrays.fill(maskRGBData, 0);
            Graphics2D g = maskImg.createGraphics();
            g.setColor(Color.WHITE);
            g.fillPolygon(polygon);
        }
    }

    private Rectangle bounds = new Rectangle(Integer.MAX_VALUE,Integer.MAX_VALUE,Integer.MIN_VALUE,Integer.MIN_VALUE);
    final List<Point> pointList = new ArrayList<>();
    final Point first;
    Point prevPoint = null;

    Selection(Point2D point){
        this.first = new Point((int)point.getX(),(int)point.getY());
        this.prevPoint=first;
        this.bounds.add(first);
    }
    public void add(Point2D point){
        var newPoint = new Point((int)point.getX(),(int)point.getY());
        add(prevPoint, newPoint);
        bounds.add(newPoint);
        prevPoint = newPoint;
    }
    public Selection close(){
        add(first);
        return this;
    }

    public Mask getMask(){
        return new Mask(this);
    }

    public int x1(){
        return bounds.x;
    }
    public int y1(){
        return bounds.y;
    }
    public int width(){
        return bounds.width;
    }
    public int height(){
        return bounds.height;
    }
    public int x2(){
        return x1()+width();
    }
    public int y2(){
        return y1()+height();
    }
    private void add(Point2D from, Point2D to) {
        int x = (int)from.getX();
        int y = (int)from.getY();
        int w = (int)(to.getX() - from.getX());
        int h = (int)(to.getY() - from.getY());
        int dx1 = Integer.compare(w, 0);
        int dy1 = Integer.compare(h, 0);
        int dx2 = dx1;
        int dy2 = 0;
        int longest = Math.abs(w);
        int shortest = Math.abs(h);
        if (longest <= shortest) {
            longest = Math.abs(h);
            shortest = Math.abs(w);
            dy2 = Integer.compare(h, 0);
            dx2 = 0;
        }
        int numerator = longest >> 1;
        for (int i = 0; i <= longest; i++) {
            Point point  = new Point(x, y);
            pointList.add(point);
            bounds.add(point);
            numerator += shortest;
            if (numerator >= longest) {
                numerator -= longest;
                x += dx1;
                y += dy1;
            } else {
                x += dx2;
                y += dy2;
            }
        }
    }
}
