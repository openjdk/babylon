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

import hat.types.F32;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import static hat.types.F32.*;
import static hat.types.vec2.*;
import static hat.types.vec3.*;
import static hat.types.vec4.*;
import shade.Shader;
import shade.Uniforms;

import static hat.types.F32.smoothstep;

/*
// RANDOMNESS
//
// I don't know why, but GLSL does not have random number generators.
// This does not pose a problem if you are writing your code in
// a programming language that has random functions. That way
// you can generate the random values using the language and send
// those values to the shader via uniforms.
//
// But if you are using a system that only allows you to write
// the shader code, such as ShaderToy, then you need to write your own
// pseuo-random generators.
//
// Here is a pattern that I saw again and again in many different
// shaders at ShaderToy.
// Let's draw N different disks at random locations using this pattern.

float hash(float seed)
{
    // Return a "random" number based on the "seed"
    return fract(sin(seed) * 43758.5453);
}

vec2 hashPosition(float x)
{
    // Return a "random" position based on the "seed"
    return vec2(hash(x), hash(x * 1.1));
}

float disk(vec2 r, vec2 center, float radius) {
    return 1.0 - smoothstep( radius-0.005, radius+0.005, length(r-center));
}

float coordinateGrid(vec2 r) {
    vec3 axesCol = vec3(0.0, 0.0, 1.0);
    vec3 gridCol = vec3(0.5);
    float ret = 0.0;

    // Draw grid lines
    const float tickWidth = 0.1;
    for(float i=-2.0; i<2.0; i+=tickWidth) {
        // "i" is the line coordinate.
        ret += 1.-smoothstep(0.0, 0.005, abs(r.x-i));
        ret += 1.-smoothstep(0.0, 0.01, abs(r.y-i));
    }
    // Draw the axes
    ret += 1.-smoothstep(0.001, 0.005, abs(r.x));
    ret += 1.-smoothstep(0.001, 0.005, abs(r.y));
    return ret;
}

float plot(vec2 r, float y, float thickness) {
    return ( abs(y - r.y) < thickness ) ? 1.0 : 0.0;
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    vec2 p = vec2(fragCoord.xy / iResolution.xy);
    vec2 r =  2.0*vec2(fragCoord.xy - 0.5*iResolution.xy)/iResolution.y;
    float xMax = iResolution.x/iResolution.y;

    vec3 bgCol = vec3(0.3);
    vec3 col1 = vec3(0.216, 0.471, 0.698); // blue
    vec3 col2 = vec3(1.00, 0.329, 0.298); // yellow
    vec3 col3 = vec3(0.867, 0.910, 0.247); // red

    vec3 ret = bgCol;

    vec3 white = vec3(1.);
    vec3 gray = vec3(.3);
    if(r.y > 0.7) {

        // translated and rotated coordinate system
        vec2 q = (r-vec2(0.,0.9))*vec2(1.,20.);
        ret = mix(white, gray, coordinateGrid(q));

        // just the regular sin function
        float y = sin(5.*q.x) * 2.0 - 1.0;

        ret = mix(ret, col1, plot(q, y, 0.1));
    }
    else if(r.y > 0.4) {
        vec2 q = (r-vec2(0.,0.6))*vec2(1.,20.);
        ret = mix(white, col1, coordinateGrid(q));

        // take the decimal part of the sin function
        float y = fract(sin(5.*q.x)) * 2.0 - 1.0;

        ret = mix(ret, col2, plot(q, y, 0.1));
    }
    else if(r.y > 0.1) {
        vec3 white = vec3(1.);
        vec2 q = (r-vec2(0.,0.25))*vec2(1.,20.);

        ret = mix(white, gray, coordinateGrid(q));

        // scale up the outcome of the sine function
        // increase the scale and see the transition from
        // periodic pattern to chaotic pattern
        float scale = 10.0;
        float y = fract(sin(5.*q.x) * scale) * 2.0 - 1.0;

        ret = mix(ret, col1, plot(q, y, 0.2));
    }
    else if(r.y > -0.2) {
        vec3 white = vec3(1.);
        vec2 q = (r-vec2(0., -0.0))*vec2(1.,10.);
        ret = mix(white, col1, coordinateGrid(q));

        float seed = q.x;
        // Scale up with a big real number
        float y = fract(sin(seed) * 43758.5453) * 2.0 - 1.0;
        // this can be used as a pseudo-random value
        // These type of function, functions in which two inputs
        // that are close to each other (such as close q.x positions)
        // return highly different output values, are called "hash"
        // function.

        ret = mix(ret, col2, plot(q, y, 0.1));
    }
    else {
        vec2 q = (r-vec2(0., -0.6));

        // use the loop index as the seed
        // and vary different quantities of disks, such as
        // location and radius
        for(float i=0.0; i<6.0; i++) {
            // change the seed and get different distributions
            float seed = i + 0.0;
            vec2 pos = (vec2(hash(seed), hash(seed + 0.5))-0.5)*3.;;
            float radius = hash(seed + 3.5);
            pos *= vec2(1.0,0.3);
            ret = mix(ret, col1, disk(q, pos, 0.2*radius));
        }
    }

    vec3 pixel = ret;
    fragColor = vec4(pixel, 1.0);
}


 */

//https://www.shadertoy.com/view/Md23DV
public class RandShader implements Shader {

    static float hash(float seed) {
        // Return a "random" number based on the "seed"
        return F32.fract(sin(seed) * 43758.5453f);
    }

    static vec2 hashPosition(float x) {
        // Return a "random" position based on the "seed"
        return vec2(hash(x), hash(x * 1.1f));
    }

    static float disk(vec2 r, vec2 center, float radius) {
        return 1.0f - smoothstep(radius - 0.005f, radius + 0.005f, length(sub(vec2(r),center)));
    }

    static float coordinateGrid(vec2 r) {
        vec3 axesCol = vec3(0.0f, 0.0f, 1.0f);
        vec3 gridCol = vec3(0.5f);
        float ret = 0.0f;

        // Draw grid lines
        float tickWidth = 0.1f;
        for (float i = -2.0f; i < 2.0f; i += tickWidth) {
            // "i" is the line coordinate.
            ret += 1f - smoothstep(0.0f, 0.005f, abs(r.x() - i));
            ret += 1f - smoothstep(0.0f, 0.01f, abs(r.y() - i));
        }
        // Draw the axes
        ret += 1f - smoothstep(0.001f, 0.005f, abs(r.x()));
        ret += 1f - smoothstep(0.001f, 0.005f, abs(r.y()));
        return ret;
    }

    static float plot(vec2 r, float y, float thickness) {
        return (abs(y - r.y()) < thickness) ? 1.0f : 0.0f;
    }

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        vec2 fres = vec2(uniforms.iResolution());
        vec2 p = div(fragCoord,fres);
        // vec2 r =  2.0*vec2(fragCoord.xy - 0.5*iResolution.xy)/iResolution.y;
        vec2 r = div(mul(2f,sub(fragCoord,mul(5f, fres))),fres.y());

        float xMax = fres.x() / fres.y();

        vec3 bgCol = vec3(0.3f);
        vec3 col1 = vec3(0.216f, 0.471f, 0.698f); // blue
        vec3 col2 = vec3(1.00f, 0.329f, 0.298f); // yellow
        vec3 col3 = vec3(0.867f, 0.910f, 0.247f); // red

        vec3 ret = bgCol;

        vec3 white = vec3(1f);
        vec3 gray = vec3(.3f);
        if (r.y() > 0.7f) {
           // vec2 q = (r-vec2(0.,0.9))*vec2(1.,20.);
            vec2 q = mul(sub(r, vec2(0f, 0.9f)),vec2(1f, 20f));
            // translated and rotated coordinate system
            //vec2 q = r.sub(vec2(0f, 0.9f)).mul(vec2(1f, 20f));
            ret = mix(white, gray, coordinateGrid(q));

            // just the regular sin function
            float y = sin(5f * q.x()) * 2.0f - 1.0f;

            ret = mix(ret, col1, plot(q, y, 0.1f));
        } else if (r.y() > 0.4f) {
            //vec2 q = (r-vec2(0.,0.6))*vec2(1.,20.);
            vec2 q = mul(sub(r,vec2(0f, 0.6f) ),vec2(1f, 20f));
          //  vec2 q = r.sub(vec2(0f, 0.6f)).mul(vec2(1f, 20f));
            ret = mix(white, col1, coordinateGrid(q));

            // take the decimal part of the sin function
            float y = fract(sin(5f * q.x())) * 2.0f - 1.0f;

            ret = mix(ret, col2, plot(q, y, 0.1f));
        } else if (r.y() > 0.1f) {
            // vec3 white = vec3(1f);
            //vec2 q = (r-vec2(0.,0.25))*vec2(1.,20.);
            vec2 q= mul(sub(r,vec2(0f, 0.25f)), vec2(1f,20f));
           // vec2 q = r.sub(vec2(0f, 0.25f)).mul(vec2(1f, 20f));
            ret = mix(white, gray, coordinateGrid(q));

            // scale up the outcome of the sine function
            // increase the scale and see the transition from
            // periodic pattern to chaotic pattern
            float scale = 10.0f;
            float y = fract(sin(5f * q.x()) * scale) * 2.0f - 1.0f;

            ret = mix(ret, col1, plot(q, y, 0.2f));
        } else if (r.y() > -0.2f) {
            //vec3 white = vec3(1.);
//            vec2 q = (r-vec2(0., -0.0))*vec2(1.,10.);
            vec2 q = mul(r, vec2(1f, 10f));
           // vec2 q = r.sub(vec2(0f, -0.0f)).mul(vec2(1f, 10f));
            ret = mix(white, col1, coordinateGrid(q));

            float seed = q.x();
            // Scale up with a big real number
            float y = fract(sin(seed) * 43758.5453f) * 2.0f - 1.0f;
            // this can be used as a pseudo-random value
            // These type of function, functions in which two inputs
            // that are close to each other (such as close q.x positions)
            // return highly different output values, are called "hash"
            // function.

            ret = mix(ret, col2, plot(q, y, 0.1f));
        } else {
            //   vec2 q = (r-vec2(0., -0.6));
            vec2 q = sub(r,vec2(0f, -0.6f));

            // use the loop index as the seed
            // and vary different quantities of disks, such as
            // location and radius
            for (float i = 0.0f; i < 6.0f; i++) {
                // change the seed and get different distributions
                float seed = i + 0.0f;
                vec2 pos = mul(sub(vec2(hash(seed), hash(seed + 0.5f)),-0.5f),3f);
                float radius = hash(seed + 3.5f);
                pos = mul(pos, vec2(1.0f, 0.3f));
                ret = mix(ret, col1, disk(q, pos, 0.2f * radius));
            }
        }

        vec3 pixel = ret;
        fragColor = vec4(pixel, 1.0f);
        return fragColor;
    };



}
