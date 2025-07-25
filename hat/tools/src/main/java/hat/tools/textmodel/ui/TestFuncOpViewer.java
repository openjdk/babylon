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
package hat.tools.textmodel.ui;

import hat.ComputeContext;
import hat.KernelContext;
import hat.buffer.S32Array;
import hat.buffer.S32Array2D;
import hat.ifacemapper.MappableIface;
import hat.tools.textmodel.ui.FuncOpViewer;
import jdk.incubator.code.CodeReflection;
import jdk.incubator.code.Op;
import jdk.incubator.code.dialect.core.CoreOp;

import java.io.IOException;
import java.lang.reflect.Method;

public class TestFuncOpViewer {
    static class Compute {
        @CodeReflection
        public static void mandel(@MappableIface.RO KernelContext kc, @MappableIface.RW S32Array2D s32Array2D, @MappableIface.RO S32Array pallette, float offsetx, float offsety, float scale) {
            if (kc.x < kc.maxX) {
                float width = s32Array2D.width();
                float height = s32Array2D.height();
                float x = ((kc.x % s32Array2D.width()) * scale - (scale / 2f * width)) / width + offsetx;
                float y = ((kc.x / s32Array2D.width()) * scale - (scale / 2f * height)) / height + offsety;
                float zx = x;
                float zy = y;
                float new_zx;
                int colorIdx = 0;
                while ((colorIdx < pallette.length()) && (((zx * zx) + (zy * zy)) < 4f)) {
                    new_zx = ((zx * zx) - (zy * zy)) + x;
                    zy = (2f * zx * zy) + y;
                    zx = new_zx;
                    colorIdx++;
                }
                int color = colorIdx < pallette.length() ? pallette.array(colorIdx) : 0;
                s32Array2D.array(kc.x, color);
            }
        }


        @CodeReflection
        static public void compute(final ComputeContext computeContext, S32Array pallete, S32Array2D s32Array2D, float x, float y, float scale) {

            computeContext.dispatchKernel(
                    s32Array2D.width() * s32Array2D.height(), //0..S32Array2D.size()
                    kc -> mandel(kc, s32Array2D, pallete, x, y, scale));
        }

    }

    public static void main(String[] args) throws NoSuchMethodException, IOException {
        String methodName = "mandel";
        Method method = Compute.class.getDeclaredMethod(methodName,
                KernelContext.class, S32Array2D.class, S32Array.class, float.class, float.class, float.class);
        CoreOp.FuncOp javaFunc = Op.ofMethod(method).get();
        FuncOpViewer.launch(javaFunc);
    }
}