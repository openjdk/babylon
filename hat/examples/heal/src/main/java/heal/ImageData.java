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

import hat.buffer.S32Array2D;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferInt;
import java.io.IOException;
import java.io.InputStream;

public class ImageData implements S32Array2D {
    final BufferedImage bufferedImage;
     int widthField;
     int heightField;
    public final int[] arrayOfData;

    @Override
    public int width() {
        return widthField;
    }

    @Override
    public int height() {
        return heightField;
    }
   // public int length() {
   //     return width()*height();
    //}


    @Override
    public int array(long idx){
        return arrayOfData[(int)idx];
    }
    @Override
    public void array(long idx, int v){
        arrayOfData[(int) idx]=v;
    }

    private  ImageData(BufferedImage bufferedImage) {
        this.bufferedImage = bufferedImage;
        this.widthField=bufferedImage.getWidth();
        this.heightField=bufferedImage.getHeight();
        this.arrayOfData = ((DataBufferInt) (bufferedImage.getRaster().getDataBuffer())).getData();
    }
    static BufferedImage to(BufferedImage originalImage, int type){
        BufferedImage image=null;
        if (originalImage.getType() == type){
            image = originalImage;
        }else {
            image = new BufferedImage(originalImage.getWidth(), originalImage.getHeight(), type);
            image.getGraphics().drawImage(originalImage, 0, 0, null);
        }
        return image;
    }
    static ImageData of(InputStream inputStream){
        try {
           return new ImageData(ImageData.to(ImageIO.read(inputStream),BufferedImage.TYPE_INT_RGB));
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }
}
