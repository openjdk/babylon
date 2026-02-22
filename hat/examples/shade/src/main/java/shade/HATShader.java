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

import hat.ComputeContext;
import hat.ComputeContext.Kernel;
import hat.KernelContext;
import hat.NDRange;
import hat.buffer.F32Array;
import hat.types.vec2;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import static hat.types.vec2.vec2;
import static hat.types.vec4.vec4;

public class HATShader {
    @Reflect
    public static vec4 mainImage(@MappableIface.RO Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        return vec4(0f, 0f, 1f, 0f);
    }

    @Reflect
    public static void penumbra(@MappableIface.RO KernelContext kc, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image) {
        if (kc.gix < kc.gsx) {
            // The image is essentially a vec3 array
            int width = (int) uniforms.iResolution().x();
            int height = (int) uniforms.iResolution().y();
            var fragCoord = vec2(kc.gix % width, kc.gix / width);
            long offset = ((long) kc.gsx * height * 3) + (kc.gix * 3L);
            var fragColor = mainImage(uniforms,
                    vec4(image.array(offset + 0), image.array(offset + 1), image.array(offset + 2), 0f),
                    fragCoord);
            image.array(offset + 0, fragColor.x());
            image.array(offset + 1, fragColor.y());
            image.array(offset + 2, fragColor.z());
        }
    }


    @Reflect
    static public void compute(final ComputeContext computeContext, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image, int width, int height) {
        computeContext.dispatchKernel(
                NDRange.of1D(width * height),               //0..S32Array2D.size()
                (@Reflect Kernel) kc -> penumbra(kc, uniforms, image));
    }
}
