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
import hat.types.F32;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;
import shade.ShaderViewer;

import java.lang.invoke.MethodHandles;

import static hat.types.F32.clamp;
import static hat.types.F32.fract;
import static hat.types.F32.mix;
import static hat.types.F32.sin;
import static hat.types.vec2.abs;
import static hat.types.vec2.add;
import static hat.types.vec2.floor;
import static hat.types.vec2.length;
import static hat.types.vec2.mul;
import static hat.types.vec4.normalize;


public class TextureShader {
/*
#define STEP 9.0

float random (in vec2 _uv) {
    return fract(sin(dot(_uv.xy, vec2(12.9898,78.233))) * 43758.5453123);
}

// 2D Noise based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
float noise (in vec2 _uv) {
    vec2 i = floor(_uv);
    vec2 f = fract(_uv);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(i + vec2(1.0, 0.0));
    float c = random(i + vec2(0.0, 1.0));
    float d = random(i + vec2(1.0, 1.0));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = f*f*(3.0-2.0*f);
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return mix(a, b, u.x) +
            (c - a)* u.y * (1.0 - u.x) +
            (d - b) * u.x * u.y;
}

// 2D SDF from iquilez
float sdBox(in vec2 p, in vec2 b)
{
    vec2 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, d.y), 0.0);
}

// Get character
float getChar(in vec2 mc, in vec2 uvid, in vec2 uvst, in vec2 uv)
{
    // Mouse interaction
    float md = 1.0 - distance(uvid, mc*1.5)*4.0;

    // Noise
    vec2 n = vec2(
            noise((uvid*9.0+iTime*(0.04)) * 4.0)-0.5,
            noise((uvid*10.0+iTime*(-0.05)) * 6.0)-0.5
        );
    uvst += n*0.33*max(md*2.0, 1.0);

    // Numbers
    float charSize = clamp(md, 0.2, 0.6);
    vec2 charOffset = vec2(floor(random(uvid+3.1) * 9.99)/16.0, 12.0/16.0);
    vec2 dx = (dFdx(uv)*STEP)/charSize*0.025;
    vec2 dy = (dFdy(uv)*STEP)/charSize*0.025;
    vec2 s = (uvst-0.5)/charSize*0.025 + 1.0/32.0 + charOffset;
    float char = textureGrad(iChannel0, s, dx, dy).r;
    char *= step(sdBox(uvst-0.5, vec2(1.0)*charSize), 0.0);

    return char;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord)
{

    // Mouse coords
    vec2 mc = (iMouse.xy - 0.5*iResolution.xy)/iResolution.y;
    if (length(iMouse.xy) < 20.0) {
        mc = vec2(sin(iTime*0.5)*0.35, cos(iTime*0.36)*0.35);
    }

    // UVs
    vec2 uv = (fragCoord - 0.5*iResolution.xy)/iResolution.y;
    uv += mc*0.5;
    vec2 uvid = floor(uv * STEP) / STEP;
    vec2 uvst = fract(uv * STEP);

    // Colors
    vec3 col = vec3(0.06, 0.09, 0.12);
    vec3 charCol = vec3(0.8, 0.9, 0.96);

    // Character with overdraw
    float char = 0.0;
    for (float i=-1.0; i<2.0; i++) {
        for (float j=-1.0; j<2.0; j++) {
            char += getChar(
                mc,
                (floor(uv*STEP) + vec2(i,j))/STEP,
                uvst-vec2(i,j),
                uv
            );
        }
    }
    char = clamp(char, 0.0, 1.0);

    col = mix(col, charCol, char);

    fragColor = vec4(col, 1.0);
}

 */


static float  STEP =9.0f;
@Reflect  static public float random (vec2 _uv) {
    return fract(sin(vec2.dot(vec2.vec2(_uv.x(), _uv.y()), vec2.vec2(12.9898f,78.233f))) * 43758.5453123f);
}

// 2D Noise based on Morgan McGuire @morgan3d
// https://www.shadertoy.com/view/4dS3Wd
@Reflect public static float noise (vec2 _uv) {
    vec2 i = floor(_uv);
    vec2 f = vec2.fract(_uv);

    // Four corners in 2D of a tile
    float a = random(i);
    float b = random(add(i,vec2.vec2(1.0f, 0.0f)));
    float c = random(add(i,vec2.vec2(0.0f, 1.0f)));
    float d = random(add(i, vec2.vec2(1.0f, 1.0f)));

    // Smooth Interpolation

    // Cubic Hermine Curve.  Same as SmoothStep()
    vec2 u = mul(f, mul(f,(vec2.sub(3.0f, mul(2.0f,f)))));
    // u = smoothstep(0.,1.,f);

    // Mix 4 coorners percentages
    return mix(a, b, u.x()) +
            (c - a)* u.y() * (1.0f - u.x()) +
            (d - b) * u.x() * u.y();
}

// 2D SDF from iquilez
@Reflect public static float sdBox(vec2 p, vec2 b) {
    vec2 d = vec2.sub(abs(p), b);
    return length(vec2.max(d, 0.0f)) + F32.min(F32.max(d.x(), d.y()), 0.0f);
}

// Get character
@Reflect  public static float getChar(float ftime, F32Array iChannel0, vec2 mc, vec2 uvid, vec2 uvst, vec2 uv) {
    // Mouse interaction
    float md = 1.0f - vec2.distance(uvid, mul(mc,1.5f))*4.0f;

    // Noise
    vec2 n = vec2.vec2(
            noise(mul(add(mul(uvid,9.0f),ftime*0.04f),4.0f))-0.5f,
            noise(mul(vec2.sub(mul(uvid,10.0f),ftime*0.05f),  6.0f))-0.5f
        );
    uvst = add(uvst,mul(n,0.33f*F32.max(md*2.0f, 1.0f)));

    // Numbers
    float charSize = clamp(md, 0.2f, 0.6f);

/*    vec2 charOffset = vec2.vec2(floor(random(add(uvid,3.1f)) * 9.99f)/16.0f, 12.0f/16.0f);
    vec2 dx = (dFdx(uv)*STEP)/charSize*0.025f;
    vec2 dy = (dFdy(uv)*STEP)/charSize*0.025f;
    vec2 s = (uvst-0.5)/charSize*0.025f + 1.0f/32.0f + charOffset;
    float ch = textureGrad(iChannel0, s, dx, dy).r;
    ch = mul(ch,vec2.step(sdBox(vec2.sub(uvst,0.5f), mul(vec2.vec2(1.0f),charSize)), vec2.vec2(0.0f)));

    return ch;
    */
    return 0f;

}
/*
void mainImage(out vec4 fragColor, in vec2 fragCoord)
{

    // Mouse coords
    vec2 mc = (iMouse.xy - 0.5*iResolution.xy)/iResolution.y;
    if (length(iMouse.xy) < 20.0) {
        mc = vec2(sin(iTime*0.5)*0.35, cos(iTime*0.36)*0.35);
    }

    // UVs
    vec2 uv = (fragCoord - 0.5*iResolution.xy)/iResolution.y;
    uv += mc*0.5;
    vec2 uvid = floor(uv * STEP) / STEP;
    vec2 uvst = fract(uv * STEP);

    // Colors
    vec3 col = vec3(0.06, 0.09, 0.12);
    vec3 charCol = vec3(0.8, 0.9, 0.96);

    // Character with overdraw
    float char = 0.0;
    for (float i=-1.0; i<2.0; i++) {
        for (float j=-1.0; j<2.0; j++) {
            char += getChar(
                mc,
                (floor(uv*STEP) + vec2(i,j))/STEP,
                uvst-vec2(i,j),
                uv
            );
        }
    }
    char = clamp(char, 0.0, 1.0);

    col = mix(col, charCol, char);

    fragColor = vec4(col, 1.0);
}

 */


    @Reflect
    public static vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord, F32Array tex,int tw, int th) {
        vec3 fres = uniforms.iResolution();
        int w=(int)fres.x();
        int h= (int)fres.y();
        long idx = (long)((((h-fragCoord.y())*tw)+fragCoord.x())*3);
        var r = tex.array(idx+0);
        var g = tex.array(idx+1);
        var b = tex.array(idx+2);
        return vec4.vec4(r,g,b,1f);
    }
    @Reflect
    public static void penumbra(@MappableIface.RO KernelContext kc, @MappableIface.RO Uniforms uniforms,
                                @MappableIface.RW F32Array f32Array,

                                @MappableIface.RW F32Array tex,int tw, int th
    ) {
        int width = (int) uniforms.iResolution().x();
        int height = (int) uniforms.iResolution().y();
        var fragColor = mainImage(uniforms, vec4.vec4(0f),
                vec2.vec2((float)(kc.gix % width),
                        (float)(height-(kc.gix / width))),tex,tw,th
        );
        f32Array.array(kc.gix * 3, fragColor.x());
        f32Array.array(kc.gix * 3+1, fragColor.y());
        f32Array.array(kc.gix * 3+2, fragColor.z());
    }

    @Reflect
    static public void compute(final ComputeContext computeContext, @MappableIface.RO Uniforms uniforms,
                               @MappableIface.RO F32Array image, int width, int height
           ,F32Array t1, int t1w,int t1h) {
        computeContext.dispatchKernel(NDRange.of1D(width * height), (@Reflect Kernel) kc -> penumbra(kc, uniforms, image,t1,t1w,t1h));
    }

    private static void update(Accelerator acc, Uniforms uniforms, F32Array f32Array, int width, int height, ShaderViewer.Texture texture) {
        var tex = texture.f32Array();
        var tw = texture.width();
        var th = texture.height();
        acc.compute((@Reflect Compute) cc -> compute(cc, uniforms, f32Array, width, height,
                tex, tw,th));
    }

    static void main(String[] args) {
        var config = ShaderViewer.Config.of(
                new Accelerator(MethodHandles.lookup(), Backend.FIRST),
                TextureShader.class,1024, 1024,
                        TextureShader.class.getResourceAsStream("/images/numbers.png")
        );
        ShaderViewer.of(config).startLoop((uniforms, f32Array) -> update(
                config.acc(), uniforms, f32Array, config.width(), config.height(), config.textures()[0] ));
    }
}
