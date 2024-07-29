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
import hat.ifacemapper.Schema;

public interface XYRGBList extends Buffer {
    interface XYRGB extends Buffer.Struct{
        int x();
        int y();
        void y(int y);
        void x(int x);
        int r();
        int g();
        int b();
        void r(int r);
        void g(int g);
        void b(int b);
    }
    int length();
    XYRGB xyrgb(long idx);
    Schema<XYRGBList> schema= Schema.of(XYRGBList.class, s->s
            .arrayLen("length")
            .array("xyrgb", xy->xy
                    .fields("x","y","r","g","b")
            )
    );
    static XYRGBList create(Accelerator accelerator, Selection selection) {
        return  schema.allocate(accelerator,selection.pointList.size());
    }
}
