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

import hat.Accelerator;
import hat.buffer.Buffer;
import hat.buffer.BufferAllocator;
import hat.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;

public interface Box extends Buffer {
    int x1();

    int y1();

    void y1(int y1);

    void x1(int x1);

    int x2();

    int y2();

    void y2(int y2);

    void x2(int x2);


    void width(int width);
    void height(int height);
    int width();
    int height();
    int area();
    void area(int area);
    Schema<Box> schema = Schema.of(Box.class, s -> s.fields("x1", "y1", "x2", "y2", "width", "height", "area"));

    static Box create(Accelerator accelerator, int x1, int y1, int x2, int y2) {
        Box box = schema.allocate(accelerator);
        box.x1(x1);
        box.y1(y1);
        box.x2(x2);
        box.y2(y2);
        box.width(x2-x1);
        box.height(y2-y1);
        box.area(box.width()* box.height());
        return box;
    }
}
