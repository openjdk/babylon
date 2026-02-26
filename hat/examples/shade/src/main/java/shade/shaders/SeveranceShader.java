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
import hat.backend.Backend;
import hat.types.F32;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import shade.Config;
import shade.Shader;
import shade.ShaderApp;
import shade.Uniforms;

import java.io.IOException;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.abs;
import static hat.types.F32.fract;
import static hat.types.F32.sin;
import static hat.types.F32.smoothstep;
import static hat.types.vec2.div;
import static hat.types.vec2.length;
import static hat.types.vec2.mul;
import static hat.types.vec2.sub;
import static hat.types.vec2.vec2;
import static hat.types.vec3.mix;
import static hat.types.vec3.vec3;
import static hat.types.vec4.normalize;
import static hat.types.vec4.vec4;

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

//https://www.shadertoy.com/view/W33XW2
public class SeveranceShader implements Shader {

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        return normalize(fragColor);
    }

    static Config controls = Config.of(
            Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST) : null,
            Integer.parseInt(System.getProperty("width", System.getProperty("size", "1024"))),
            Integer.parseInt(System.getProperty("height", System.getProperty("size", "1024"))),
            Integer.parseInt(System.getProperty("targetFps", "15")),
            new SeveranceShader()
    );

    static void main(String[] args) throws IOException {
        new ShaderApp(controls);
    }

}
