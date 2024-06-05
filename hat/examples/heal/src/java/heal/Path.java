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

import java.awt.Polygon;
import java.util.Arrays;

public class Path extends XYList {
    int x1=Integer.MAX_VALUE;
    int y1=Integer.MAX_VALUE;
    int x2=Integer.MIN_VALUE;
    int y2=Integer.MIN_VALUE;

    void add(int x,int y) {
        super.add(x, y);
        x1 = Math.min(x, x1);
        y1 = Math.min(y, y1);
        x2 = Math.max(x, x2);
        y2 = Math.max(y, y2);
    }
    Path(){
    }
    Path (int x, int y){
        super(x,y);
    }
    public Polygon getPolygon() {
        Polygon p = new Polygon();
        for (int i=0;i<length();i++){
            XY xy = (XYList.XY)xy(i);
            p.addPoint(xy.x(), xy.y());
        }
        return p;
    }

    public void extendTo(int x, int y){
        add(xy[length()*XYList.STRIDE-(2-XYList.Xidx)],
                xy[length()*XYList.STRIDE-(2-XYList.Yidx)], x, y);
    }

    public Path close(){
        extendTo(xy[0], xy[1]);
        xy = Arrays.copyOf(xy, length() * XYList.STRIDE);
        return this;
    }

    public void add(int x1, int y1, int x2, int y2) {
        int x = x1;
        int y = y1;
        int w = x2 - x;
        int h = y2 - y;
        int dx1 = Integer.compare(w, 0);
        int dy1 =Integer.compare(h, 0);
        int dx2 = dx1;
        int dy2 = 0;
        int longest = Math.abs(w);
        int shortest = Math.abs(h);
        if (!(longest > shortest)) {
            longest = Math.abs(h);
            shortest = Math.abs(w);
            dy2 = Integer.compare(h, 0);
            dx2 = 0;
        }
        int numerator = longest >> 1;
        for (int i = 0; i <= longest; i++) {
            add(x, y);
            numerator += shortest;
            if (!(numerator < longest)) {
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
