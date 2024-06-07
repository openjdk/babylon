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
package mandel.buffers;

import hat.Accelerator;
import hat.buffer.ImageBuffer;

import java.awt.image.BufferedImage;

import static java.lang.foreign.ValueLayout.JAVA_SHORT;

public interface RgbaS32Image extends ImageBuffer {
    private static RgbaS32Image create(Accelerator accelerator, int width, int height) {
        return ImageBuffer.create(accelerator, RgbaS32Image.class, width, height, BufferedImage.TYPE_INT_ARGB, 1, JAVA_SHORT);
    }

    static RgbaS32Image create(Accelerator accelerator, BufferedImage bufferedImage) {
        return create(accelerator, bufferedImage.getWidth(), bufferedImage.getHeight()).syncFromRaster(bufferedImage);

    }

    short data(long idx);

    void data(long idx, short v);

    default short get(int x, int y) {
        return data((long) y * width() + x);
    }

    default void set(int x, int y, short v) {
        data((long) y * width() + x, v);
    }
}
