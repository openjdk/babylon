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
package shade.shaders;

import hat.Accelerator;
import hat.ComputeContext;
import hat.Accelerator.Compute;
import hat.ComputeContext.Kernel;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;

import  static hat.types.F32.*;

import hat.buffer.F32Array;
import hat.types.vec2;
import static hat.types.vec2.*;
import hat.types.vec3;

import static hat.types.vec3.*;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;
import hat.buffer.Uniforms;
import shade.ShaderViewer;
import java.lang.invoke.MethodHandles;

/*
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{
    vec2 uv = (2.0 * fragCoord - iResolution.xy) / min(iResolution.x, iResolution.y);

    for (float i = 2.0; i < 13.0; i++)
    {
        uv.x += 0.4 / i * cos(i * 2.0 * uv.y + iTime) * cos(i * 1.5 * uv.y + iTime);
        uv.y += 0.4 / i * cos(i * 2.0 * uv.x + iTime);
    }

    vec3 col = cos(iTime / 4.0 - uv.xyx);
    col = step(0.0, col);
    col.b = col.g;

    // alpha for cineshader
    float alpha = 0.0;
    if (col.g > 0.0 || col.r > 0.0) alpha = 0.6;

    fragColor = vec4(col, alpha);
}

** SHADERDATA
{
   "title": "Painting with maths",
   "description": "Simple shader that gives an effect of painting. https://gubebra.itch.io/",
   "model": "person"
}

 */

//https://www.shadertoy.com/view/W33XW2
public class PaintShader  {
    @Reflect
    public static vec4 createPixel(vec2 fres, float ftime, vec2 fmouse, vec2 fragCoord){
        vec2 uv = div(sub(mul(2.0f,fragCoord),fres), min(fres.x(), fres.y()));
        for (float i = 2f; i < 13f; i++) {
            var cosyTime = cos(i * 2.0f * uv.y() + ftime);
            var cosxTime = cos(i * 2.0f * uv.x() + ftime);
            var dx = 0.4f / i * cosyTime *cos(i * 1.5f * uv.y() + ftime);
            var dy = 0.4f / i * cosxTime;
            uv = vec2.add(uv, vec2(dx,dy));
        }
        vec3 col = cos(div(ftime, vec3.sub(4.0f, vec3(uv.x(),uv.y(),uv.x()))));
        col = step(vec3.vec3(0.0f), col);
        col = vec3(col.x(),col.y(),col.y());

        // alpha for cineshader
        float alpha = 0.0f;
        if (col.y() > 0.0 || col.x() > 0.0f){
            alpha = 0.6f;
        }
        return(vec4.vec4(col, alpha));
    }
    @Reflect public static vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        return createPixel(vec2.vec2(uniforms.iResolution().x(),uniforms.iResolution().y()),uniforms.iTime(),vec2.vec2(uniforms.iMouse().x(),uniforms.iMouse().y()),fragCoord);

    }
    @Reflect
    public static void penumbra(@MappableIface.RO KernelContext kc, @MappableIface.RO Uniforms uniforms, @MappableIface.RW F32Array f32Array) {
        int width = (int) uniforms.iResolution().x();
        var fragColor = mainImage(uniforms, vec4.vec4(0f), vec2.vec2((float)(kc.gix % width), (float)(kc.gix / width)));
        f32Array.array(kc.gix * 3, fragColor.x());
        f32Array.array(kc.gix * 3+1, fragColor.y());
        f32Array.array(kc.gix * 3+2, fragColor.z());
    }

    @Reflect
    static public void compute(final ComputeContext computeContext, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image, int width, int height) {
        computeContext.dispatchKernel(NDRange.of1D(width * height), (@Reflect Kernel) kc -> penumbra(kc, uniforms, image));
    }

    private static void update(  Accelerator acc, Uniforms uniforms, F32Array f32Array, int width, int height) {
        acc.compute((@Reflect Compute) cc -> compute(cc, uniforms, f32Array, width, height));
    }

    static void main(String[] args) {
        var acc = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var shader = ShaderViewer.of(acc, PaintShader.class,1024, 1024, true);
        shader.startLoop((uniforms, f32Array) -> update( acc, uniforms, f32Array, shader.view.getWidth(), shader.view.getWidth()));
    }

}
