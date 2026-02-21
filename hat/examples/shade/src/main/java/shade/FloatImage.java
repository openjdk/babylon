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
package shade;

import hat.buffer.F32Array;
import hat.types.vec4;
import optkl.util.carriers.ArenaAndLookupCarrier;

import java.awt.Transparency;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ColorModel;
import java.awt.image.ComponentColorModel;
import java.awt.image.DataBuffer;
import java.awt.image.DataBufferFloat;
import java.awt.image.PixelInterleavedSampleModel;
import java.awt.image.Raster;
import java.awt.image.SampleModel;
import java.awt.image.WritableRaster;

public record FloatImage(
        int width,
        int height,
        int widthXHeight,
        int channels,
        ColorSpace colorSpace,
        ColorModel colorModel,
        SampleModel sampleModel,
        DataBufferFloat dataBufferFloat,
        float[] data,
        WritableRaster raster,
        BufferedImage bufferedImage,
        F32Array f32Array
) {
    public static FloatImage of(ArenaAndLookupCarrier arenaAndLookupCarrier, int width, int height) {
        // We need an RGB colorspace.
        ColorSpace colorSpace = ColorSpace.getInstance(ColorSpace.CS_sRGB);

        // Create the Color Model. 32 bits per component, no alpha, non-premultiplied
        ColorModel colorModel = new ComponentColorModel(colorSpace, false, false,
                Transparency.OPAQUE, DataBuffer.TYPE_FLOAT);

        int channels = width * height * 3;
        // Create the Sample Model (Pixel Interleaved) bands for RGB, scanline stride is width * 3
        SampleModel sampleModel = new PixelInterleavedSampleModel(DataBuffer.TYPE_FLOAT,
                width, height, 3, width * 3, new int[]{0, 1, 2});

        // Create the DataBuffer (an actual heap allocated  float array)
        DataBufferFloat dataBufferFloat = new DataBufferFloat(width * height * 3);

        // Get the float pixels

        float[] data = dataBufferFloat.getData();

        // Create the Raster
        WritableRaster raster = Raster.createWritableRaster(sampleModel, dataBufferFloat, null);

        BufferedImage bufferedImage = new BufferedImage(colorModel, raster, false, null);

        F32Array f32Array = F32Array.create(arenaAndLookupCarrier, channels);

        return new FloatImage(width, height, width * height, channels,
                colorSpace, colorModel, sampleModel, dataBufferFloat, data, raster, bufferedImage, f32Array);
    }

    public void set(int i, vec4 outFragColor) {
        data[i * 3 + 0] = outFragColor.x();
        data[i * 3 + 1] = outFragColor.y();
        data[i * 3 + 2] = outFragColor.z();
    }

    public void sync() {
        f32Array.copyFrom(data);
    }
}
