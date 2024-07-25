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

import java.awt.Rectangle;

public class Path  {

    final XYList xyList;
    private Rectangle bounds = new Rectangle(Integer.MAX_VALUE,Integer.MAX_VALUE,Integer.MIN_VALUE,Integer.MIN_VALUE);
    Path(XYList xyList){
        this.xyList =xyList;
    }

    public void add(int x, int y){
        if (xyList.length()>0) {
            XYList.XY lastxy = xyList.xy(xyList.length() - 1);
            add(lastxy.x(), lastxy.y(), x, y);
        }else{
            ( (XYListImpl)xyList).add(x, y);
            bounds.add(x,y);
        }
    }
    public Path close(){
        var first = xyList.xy(0);
        add(first.x(), first.y());
        return this;
    }

    int x1(){
        return bounds.x;
    }
    int y1(){
        return bounds.y;
    }
    int width(){
        return bounds.width;
    }
    int height(){
        return bounds.height;
    }
    int x2(){
        return x1()+width();
    }
    int y2(){
        return y1()+height();
    }

    private void add(int x1, int y1, int x2, int y2) {
        int x = x1;
        int y = y1;
        int w = x2 - x;
        int h = y2 - y;
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

            ( (XYListImpl)xyList).add(x, y);
            bounds.add(x,y);
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
