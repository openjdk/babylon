
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

public class XYListImpl implements XYList {
    final static int INIT= 32;
    final static int STRIDE= 2;
    final static int Xidx = 0;
    final static int Yidx = 1;
    private int length;
    protected int[] xy = new int[INIT*STRIDE];

    @Override
    public XYList.XY xy(long idx) {
        return new XY(this, (int) idx);
    }

    @Override
    public int length() {
        return length;
    }

    public static class XY implements XYList.XY{
        final XYListImpl xyList;
        final private int idx;
        public XY(XYListImpl table, int idx) {
            this.xyList=table;this.idx = idx;
        }

        @Override
        public int x() {
            return xyList.xy[idx*STRIDE+ Xidx];
        }

        @Override
        public int y() {
            return xyList.xy[idx*STRIDE+ Yidx];
        }


        @Override
        public void y(int y) {
            xyList.xy[idx*STRIDE+ Yidx] =y;
        }

        @Override
        public void x(int x) {
            xyList.xy[idx*STRIDE+ Yidx]=x;
        }
    }


    void add(int x,int y){
        if (length*STRIDE>=xy.length){
            xy = Arrays.copyOf(xy, xy.length*STRIDE);
        }
        xy[length*STRIDE+Xidx]=x;
        xy[length*STRIDE+Yidx]=y;
        length++;
    }

    XYListImpl(){
    }
}
