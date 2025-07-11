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
package wrap.opengl;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.IOException;
import java.io.InputStream;
import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;


public class GLTexture {
    public final Arena arena;
    public final MemorySegment data;
    public final int width;
    public final int height;
    public int idx;

    public GLTexture(Arena arena, InputStream textureStream) {
        this.arena = arena;
        BufferedImage img = null;
        try {
            img = ImageIO.read(textureStream);
            this.width = img.getWidth();
            this.height = img.getHeight();
            BufferedImage image = new BufferedImage(width, height, BufferedImage.TYPE_4BYTE_ABGR_PRE);
            image.getGraphics().drawImage(img, 0, 0, null);
            var raster = image.getRaster();
            var dataBuffer = raster.getDataBuffer();
            data = arena.allocateFrom(ValueLayout.JAVA_BYTE, ((DataBufferByte) dataBuffer).getData());
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
