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
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;

import hat.types.F32;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import static hat.types.F32.*;
import static hat.types.vec2.*;
import static hat.types.vec3.*;
import static hat.types.vec4.*;

public class HATShader {
    @Reflect public static vec3 palette(float t) {
        vec3 a = vec3(0.5f, 0.5f, 0.5f);
        vec3 b = vec3(0.5f, 0.5f, 0.5f);
        vec3 c = vec3(1.0f, 1.0f, 1.0f);
        vec3 d = vec3(0.263f, 0.416f, 0.557f);
        return add(a, mul(b, cos(mul(add(mul(c, vec3(t)), d), vec3(6.28318f)))));
    }

    @Reflect
    static public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        vec2 fResolution = vec2(uniforms.iResolution().x(),uniforms.iResolution().y());
        float fTime = uniforms.iTime();
        vec2 uv = div(sub(mul(fragCoord, 2f), fResolution), fResolution.y());
        vec2 uv0 = uv;
        vec3 color = vec3(0f);
        for (float i = 0f; i < 4f; i++) {
            var uv1_5 = mul(uv, 1.5f);
            var f = fract(uv1_5);
            uv = sub(f, vec2(0.5f));
           vec3 col = palette(length(uv0) + i * .4f + fTime * .4f);
            float d = length(uv) * exp(-length(uv0));
            d = sin(d * 8f + fTime) / 8f;
            d = abs(d);
            d = pow(0.01f / d, 1.2f);
            color = add(color, mul(col, d));
        }

        fragColor = vec4(color, 1.0f);
      //  fragColor=vec4(1f,0f,0f,1f);
        return normalize(fragColor);
    }


   /* @Reflect
    public static vec4 mainImageOld(@MappableIface.RO Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        var  fTime = uniforms.iTime();
        var vec2 = uniforms.iResolution();
        var v = vec4(1f);
        // v = vec4.add(v,v);
        return vec4(vec2.x()/10f, abs(F32.cos(fTime)),sin(fTime),0f);

    } */

    @Reflect
    public static void penumbra(KernelContext kc, Uniforms uniforms, F32Array image) {
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
