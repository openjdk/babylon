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
package violajones.buffers;

import hat.Accelerator;
import hat.buffer.ImageBuffer;

import java.awt.image.BufferedImage;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.StructLayout;

import static java.lang.foreign.ValueLayout.JAVA_BYTE;

public interface RgbS08x3Image extends ImageBuffer {
    StructLayout layout =  ImageBuffer.createLayout(RgbS08x3Image.class,JAVA_BYTE);

    private static RgbS08x3Image create(Accelerator accelerator, int width, int height) {
        return ImageBuffer.create(accelerator, RgbS08x3Image.class,layout, width, height, BufferedImage.TYPE_INT_RGB, 3);
    }

    static RgbS08x3Image create(Accelerator accelerator, BufferedImage bufferedImage) {
        return create(accelerator, bufferedImage.getWidth(), bufferedImage.getHeight()).syncFromRaster(bufferedImage);

    }

    byte data(long idx);

    void data(long idx, byte v);

    default byte get(int x, int y, int deltaMod3) {
        return data(((long) y * width() * 3 + x * 3) + deltaMod3);
    }

    default void set(int x, int y, int deltaMod3, byte v) {
        data(((long) y * width() * 3 + x * 3) + deltaMod3, v);
    }

}
