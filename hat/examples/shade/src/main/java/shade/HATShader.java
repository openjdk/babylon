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
import hat.buffer.Uniforms;
import hat.types.F32;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import static hat.types.vec2.vec2;
import static hat.types.vec4.vec4;

public class HATShader {
    @Reflect
    public static vec4 mainImage(@MappableIface.RO Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
      //  vec3 fres =  uniforms.iResolution();
        float fTime = uniforms.iTime();
        var v = vec4(1f);
       // v = vec4.add(v,v);
        return vec4(1f, F32.abs(F32.cos(fTime)),F32.sin(fTime),0f);
    }

    @Reflect
    public static void penumbra(@MappableIface.RO KernelContext kc, @MappableIface.RO Uniforms uniforms, @MappableIface.RW F32Array image) {
        if (kc.gix < kc.gsx) {
            vec3 fres =  uniforms.iResolution();
            int width = (int) fres.x();
            int height = (int) fres.y();
            int x= kc.gix % width;
            int y= kc.gix / width;
            int offsetx = kc.gix*3;
            int offsety =offsetx+1;
            int offsetz=offsety+1;
            var fragCoord = vec2(x,y);
            var fragColor = vec4(image.array(offsetx), image.array(offsety), image.array(offsetz),0f);
            fragColor = mainImage(uniforms, fragColor, fragCoord);
            image.array(offsetx, fragColor.x());
            image.array(offsety, fragColor.y());
            image.array(offsetz, fragColor.z());
        }
    }


    @Reflect
    static public void compute(final ComputeContext computeContext, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image, int width, int height) {

        computeContext.dispatchKernel(
                NDRange.of1D(width * height),               //0..S32Array2D.size()
                (@Reflect Kernel) kc -> penumbra(kc, uniforms, image));
    }
}
