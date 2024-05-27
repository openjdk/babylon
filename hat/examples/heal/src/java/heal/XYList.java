
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

class XYList implements Iterable<XYList.XY>{
    final static int INIT= 32;
    final static int STRIDE= 2;
    final static int X= 0;
    final static int Y= 1;

    public class XY implements Iterator<XY>{
        int idx=-1;
        int x;
        int y;

        @Override
        public boolean hasNext() {
            return idx+1<size;
        }

        public XY set(int idx) {
            this.idx = idx;
            x = xy[idx*STRIDE+X];
            y = xy[idx*STRIDE+Y];
            return this;
        }

        @Override
        public XY next() {
            idx++;
            set(idx);
            return this;
        }

        @Override
        public void remove() {
           throw new IllegalStateException("remove not supported");
        }
    }

    int size;
    int xy[] = new int[INIT*STRIDE];
    void add(int x,int y){
        if (size*STRIDE==xy.length){
            xy = Arrays.copyOf(xy, xy.length*STRIDE);
        }
        xy[size*STRIDE+X]=x;
        xy[size*STRIDE+Y]=y;
        size++;
    }

    XYList(){
    }

    XYList(int x, int y){
        add(x,y);
    }


    @Override
    public Iterator<XY> iterator() {
        return new XY();
    }
}
