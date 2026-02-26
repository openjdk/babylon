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
import static hat.types.F32.exp;
import static hat.types.F32.pow;
import static hat.types.F32.sin;
import static hat.types.vec2.div;
import static hat.types.vec2.fract;
import static hat.types.vec2.length;
import static hat.types.vec2.mul;
import static hat.types.vec2.sub;
import static hat.types.vec2.vec2;
import static hat.types.vec3.add;
import static hat.types.vec3.cos;
import static hat.types.vec3.mul;
import static hat.types.vec3.vec3;
import static hat.types.vec4.normalize;
import static hat.types.vec4.vec4;

/* This animation is the material of my first youtube tutorial about creative
   coding, which is a video in which I try to introduce programmers to GLSL
   and to the wonderful world of shaders, while also trying to share my recent
   passion for this community.
                                       Video URL: https://youtu.be/f4s1h2YETNY


//https://iquilezles.org/articles/palettes/
vec3 palette( float t ) {
    vec3 a = vec3(0.5, 0.5, 0.5);
    vec3 b = vec3(0.5, 0.5, 0.5);
    vec3 c = vec3(1.0, 1.0, 1.0);
    vec3 d = vec3(0.263,0.416,0.557);

    return a + b*cos( 6.28318*(c*t+d) );
}

        //https://www.shadertoy.com/view/mtyGWy
        void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
            vec2 uv = (fragCoord * 2.0 - iResolution.xy) / iResolution.y;
            vec2 uv0 = uv;
            vec3 finalColor = vec3(0.0);

            for (float i = 0.0; i < 4.0; i++) {
                uv = fract(uv * 1.5) - 0.5;

                float d = length(uv) * exp(-length(uv0));

                vec3 col = palette(length(uv0) + i*.4 + iTime*.4);

                d = sin(d*8. + iTime)/8.;
                d = abs(d);

                d = pow(0.01 / d, 1.2);

                finalColor += col * d;
            }

            fragColor = vec4(finalColor, 1.0);
        }
 */
//https://www.shadertoy.com/view/mtyGWy
public class TutorialShader implements Shader {


    vec3 palette(float t) {
        vec3 a = vec3(0.5f, 0.5f, 0.5f);
        vec3 b = vec3(0.5f, 0.5f, 0.5f);
        vec3 c = vec3(1.0f, 1.0f, 1.0f);
        vec3 d = vec3(0.263f, 0.416f, 0.557f);
        return add(a, mul(b, cos(mul(add(mul(c, vec3(t)), d), vec3(6.28318f)))));
    }

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        vec2 fResolution = vec2(uniforms.iResolution().x(),uniforms.iResolution().y());
        float fTime = uniforms.iTime();
        vec2 uv = div(sub(mul(fragCoord, 2f), fResolution), fResolution.y());
        vec2 uv0 = uv;
        vec3 color = vec3(0f);
        for (float i = 0f; i < 4f; i++) {
            uv = sub(fract(mul(uv, 1.5f)), vec2(0.5f));
            vec3 col = palette(length(uv0) + i * .4f + fTime * .4f);
            float d = length(uv) * exp(-length(uv0));
            d = sin(d * 8f + fTime) / 8f;
            d = abs(d);
            d = pow(0.01f / d, 1.2f);
            color = add(color, mul(col, d));
        }

        fragColor = vec4(color, 1.0f);
        return normalize(fragColor);
    }

    static Config controls = Config.of(
            Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST) : null,
            Integer.parseInt(System.getProperty("width", System.getProperty("size", "1024"))),
            Integer.parseInt(System.getProperty("height", System.getProperty("size", "1024"))),
            Integer.parseInt(System.getProperty("targetFps", "12")),
            new TutorialShader()
    );

    static void main(String[] args) throws IOException {
        new ShaderApp(controls);
    }
}
