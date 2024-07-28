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

import java.awt.Rectangle;

public class BoxImpl implements Box{
    final Rectangle rectangle;
    public BoxImpl(int x1, int y1, int x2, int y2) {
         rectangle =new Rectangle(x1,y1,x2-x1,y2-y1);
    }


    @Override
    public int x1() {
        return rectangle.x;
    }

    @Override
    public void x1(int x1) {
        rectangle.x=x1;
    }

    @Override
    public int y1() {
        return rectangle.y;
    }

    @Override
    public void y1(int y1) {
        rectangle.y = y1;
    }


    @Override
    public int x2() {
        return rectangle.width+rectangle.x;
    }

    @Override
    public int y2() {
        return rectangle.height+rectangle.y;
    }

    @Override
    public void y2(int y2) {
        rectangle.height = y2-rectangle.y;
    }

    @Override
    public void x2(int x2) {
        rectangle.width = x2-rectangle.x;
    }
}
