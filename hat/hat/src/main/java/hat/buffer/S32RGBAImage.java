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
package hat.buffer;

import hat.ifacemapper.Schema;

import java.lang.invoke.MethodHandles;

public interface S32RGBAImage extends ImageIfaceBuffer<S32RGBAImage> {
    int data(long idx);

    void data(long idx, int v);

    int width();
    void width(int width);
    int height();
    void height(int height);
    Schema<S32RGBAImage> schema = Schema.of(S32RGBAImage.class, s -> s
            .arrayLen("width", "height").stride(1).array("data")
    );


    static S32RGBAImage create(MethodHandles.Lookup lookup, BufferAllocator bufferAllocator, int width, int height){
        var instance = schema.allocate(lookup, bufferAllocator,width,height);
        instance.width(width);
        instance.height(height);
        return instance;
    }
}
