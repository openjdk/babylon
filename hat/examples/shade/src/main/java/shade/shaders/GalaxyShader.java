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
import hat.Accelerator.Compute;
import hat.ComputeContext;
import hat.ComputeContext.Kernel;
import hat.KernelContext;
import hat.NDRange;
import hat.backend.Backend;
import hat.buffer.F32Array;
import hat.buffer.Uniforms;

import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;
import shade.ShaderViewer;

import java.lang.invoke.MethodHandles;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import hat.types.mat2;
import static hat.types.F32.*;
import static hat.types.mat2.*;
import static hat.types.vec2.*;
import static hat.types.vec3.*;
import static hat.types.vec4.*;
/*
// https://www.shadertoy.com/view/MdXSzS
// The Big Bang - just a small explosion somewhere in a massive Galaxy of Universes.
// Outside of this there's a massive galaxy of 'Galaxy of Universes'... etc etc. :D

// To fake a perspective it takes advantage of the screen being wider than it is tall.

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
   vec2 uv = (fragCoord.xy / iResolution.xy) - .5;
   float t = iTime * .1 + ((.25 + .05 * sin(iTime * .1))/(length(uv.xy) + .07)) * 2.2;
   float si = sin(t);
   float co = cos(t);
   mat2 ma = mat2(co, si, -si, co);

   float v1, v2, v3;
   v1 = v2 = v3 = 0.0;

   float s = 0.0;
   for (int i = 0; i < 90; i++)
   {
      vec3 p = s * vec3(uv, 0.0);
      p.xy *= ma;
      p += vec3(.22, .3, s - 1.5 - sin(iTime * .13) * .1);
      for (int i = 0; i < 8; i++){
         p = abs(p) / dot(p,p) - 0.659;
      }
      v1 += dot(p,p) * .0015 * (1.8 + sin(length(uv.xy * 13.0) + .5  - iTime * .2));
      v2 += dot(p,p) * .0013 * (1.5 + sin(length(uv.xy * 14.5) + 1.2 - iTime * .3));
      v3 += length(p.xy*10.) * .0003;
      s  += .035;
   }

   float len = length(uv);
   v1 *= smoothstep(.7, .0, len);
   v2 *= smoothstep(.5, .0, len);
   v3 *= smoothstep(.9, .0, len);

   vec3 col = vec3( v3 * (1.5 + sin(iTime * .2) * .4),(v1 + v3) * .3, v2)
                + smoothstep(0.2, .0, len) * .85 + smoothstep(.0, .6, v3) * .3;

   fragColor=vec4(min(pow(abs(col), vec3(1.2)), 1.0), 1.0);
}

 */
//https://www.shadertoy.com/view/MdXSzS
public class GalaxyShader {
    @Reflect
    public static vec4 createPixel(vec2 fres, float ftime, vec2 fmouse, vec2 fragCoord){
        vec2 uv = sub(div(fragCoord, fres), .5f);
        float t = ftime * .1f + ((.25f + .05f * sin(ftime * .1f))/(length(uv) + .07f)) * 2.2f;
        float si = sin(t);
        float co = cos(t);
        mat2 ma = mat2(co, si, -si, co);

        float v1=0f;
        float v2=0f;
        float v3=0f;

        float s = 0.0f;
        for (int i = 0; i < 90; i++) {
            vec3 p = mul(s, vec3(uv.x(),uv.y(), 0.0f));
            vec2 p_xy = mul(vec2(p.x(),p.y()),ma);p = vec3(p_xy.x(),p_xy.y(),p.z());  //p.xy *= ma;
            p = add(p,vec3(.22f, .3f, s - 1.5f - sin(ftime * .13f) * .1f));
            for (int i2 = 0; i2 < 8; i2++)   {
                p = sub(div(abs(p), dot(p,p)),0.659f);
            }

            v1 += dot(p,p) * .0015f * (1.8f + sin(length(mul(uv, 13.0f)) + .5f  - ftime * .2f));
            v2 += dot(p,p) * .0013f * (1.5f + sin(length(mul(uv, 14.5f)) + 1.2f - ftime * .3f));
            v3 += length(mul(vec2(p.x(), p.y()),10f)) * .0003f;
            s  += .035f;
        }

        float len = length(uv);
        v1 *= smoothstep(.7f, .0f, len);
        v2 *= smoothstep(.5f, .0f, len);
        v3 *= smoothstep(.9f, .0f, len);

        vec3 col = add(vec3( v3 * (1.5f + sin(ftime * .2f) * .4f), (v1 + v3) * .3f, v2),
                 smoothstep(0.2f, .0f, len) * .85f + smoothstep(.0f, .6f, v3) * .3f);

        return normalize(vec4(min(pow(abs(col), vec3(1.2f)), 1.0f), 1.0f));
    }
/*
    @Reflect
    public static vec4 createPixel1(vec2 fres, float ftime, vec2 fmouse, vec2 fragCoord){
        //ut vec4 O, vec2 I
        vec4 O=vec4(0f);
        vec3 p= vec3(0f);
        vec3 // Ray position
                r = normalize(sub(vec3(add(fragCoord,fragCoord),0f) , vec3(fres.x(), fres.y(), fres.y()))); // Ray direction
                vec3 a = normalize(tan(add(ftime*.2f,vec3(0f,1f,2f)))); // Rotation axis
        float i=0; // Iterator
                float t=0f; // Distance
                float v=0f; // Density
                float n=0f; // Noise iterator
        // Raymarching loop

        for (O=add(O,i); i++<60.f;t+=v*.06f){
            p=mul(t,r);
            // Move camera back
            p = add(p, vec3(0f,0f,5f));

            // Rotate around rotation axis
            p = sub(mul(mul(a,dot(a,p)),2f),p);
            // Turbulence
            for (n=-.3f; n++<9f; p=add(p,mul(2f,div(sin(add(mul(p,n),ftime*n),n)))));
            // Density based on distance to sphere
            v = abs(length(p)-2.f)+.01f;
            // Color accumulation based on iteration and density
            O=add(O,div(exp(sin(add((i*.2f),vec4(0f,2f,4f,0f)))),v));
        }
        // Tone mapping
        O = F32.tanh(O/2e2f);
    return O;
    }
*/
    @Reflect public static vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        return createPixel(vec2.vec2(uniforms.iResolution().x(),uniforms.iResolution().y()),uniforms.iTime(),vec2.vec2(uniforms.iMouse().x(),uniforms.iMouse().y()),fragCoord);
    }


    @Reflect
    public static void penumbra(@MappableIface.RO KernelContext kc, @MappableIface.RO Uniforms uniforms, @MappableIface.RW F32Array f32Array) {
        int width = (int) uniforms.iResolution().x();
        int height = (int) uniforms.iResolution().y();
        var fragColor = mainImage(uniforms, vec4.vec4(0f), vec2.vec2((float)(kc.gix % width), (float)(height-(kc.gix / width))));
        f32Array.array(kc.gix * 3, fragColor.x());
        f32Array.array(kc.gix * 3+1, fragColor.y());
        f32Array.array(kc.gix * 3+2, fragColor.z());
    }

    @Reflect
    static public void compute(final ComputeContext computeContext, @MappableIface.RO Uniforms uniforms, @MappableIface.RO F32Array image, int width, int height) {
        computeContext.dispatchKernel(NDRange.of1D(width * height), (@Reflect Kernel) kc -> penumbra(kc, uniforms, image));
    }

    public static void update(  Accelerator acc, Uniforms uniforms, F32Array f32Array, int width, int height) {
        acc.compute((@Reflect Compute) cc -> compute(cc, uniforms, f32Array, width, height));
    }

    static void main(String[] args) {
        var acc = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var shader = ShaderViewer.of(acc, GalaxyShader.class,1200, 800);
        shader.startLoop((uniforms, f32Array) -> update( acc, uniforms, f32Array, shader.view.getWidth(), shader.view.getHeight()));
    }
}
