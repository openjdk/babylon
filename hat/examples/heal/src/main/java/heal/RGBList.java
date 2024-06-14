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

import java.util.Arrays;
import java.util.Iterator;

class RGBList implements S32RGBTable{
    final static int INIT=32;
    final static int STRIDE= 3;
    final static int Ridx= 0;
    final static int Gidx= 1;
    final static int Bidx= 2;
    int length;
    int[] rgb = new int[INIT*STRIDE];
    private RGB cursor = new RGB(this);
    @Override
    public RGB rgb(long idx) {
        cursor.set((int)idx);
        return cursor;
    }

    @Override
    public int length() {
        return length;
    }

    static public class RGB implements S32RGBTable.RGB{
        RGBList rgbList;
        int idx=-1;
        RGB(RGBList rgbList){
            this.rgbList = rgbList;
        }

        public RGB set(int idx) {
            this.idx = idx;

            return this;
        }

        @Override
        public int r() {
            return rgbList.rgb[idx*STRIDE+Ridx];
        }

        @Override
        public int g() {
            return rgbList.rgb[idx*STRIDE+Gidx];
        }

        @Override
        public int b() {

                return rgbList.rgb[idx*STRIDE+Bidx];


        }

        @Override
        public void r(int r) {
            rgbList.rgb[idx*STRIDE+Ridx]=r;
        }

        @Override
        public void g(int g) {
            rgbList.rgb[idx*STRIDE+Gidx]=g;
        }

        @Override
        public void b(int b) {
            rgbList.rgb[idx*STRIDE+Bidx]=b;
        }
    }


    void set(int idx, int r,int g, int b){
        rgb[idx*STRIDE+Ridx]=r;
        rgb[idx*STRIDE+Gidx]=g;
        rgb[idx*STRIDE+Bidx]=b;
    }

    void add(int r,int g, int b){
        if (length*STRIDE== rgb.length){
            rgb = Arrays.copyOf(rgb, rgb.length*STRIDE);
        }
        set(length, r, g, b);
        length++;
    }

    public void setRGB(int idx,int v) {
        set(idx, ((v >> 16) & 0xFF), ((v >> 8) & 0xFF), ((v >> 0) & 0xFF));
    }
    public void addRGB(int v) {
        add( ((v >> 16) & 0xFF), ((v >> 8) & 0xFF), ((v >> 0) & 0xFF));
    }


    RGBList(){
    }

    RGBList(RGBList list){
        length = list.length;
        rgb = Arrays.copyOf(list.rgb, list.rgb.length);
    }

}
