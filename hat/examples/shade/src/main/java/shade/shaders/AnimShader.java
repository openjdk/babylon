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
import hat.buffer.F32Array;
import hat.types.mat2;
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;
import hat.buffer.Uniforms;
import shade.ShaderViewer;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.PI;
import static hat.types.F32.cos;
import static hat.types.F32.floor;
import static hat.types.F32.fract;
import static hat.types.F32.mod;
import static hat.types.F32.sin;
import static hat.types.F32.smoothstep;
import static hat.types.mat2.mat2;
import static hat.types.vec2.add;
import static hat.types.vec2.div;
import static hat.types.vec2.length;
import static hat.types.vec2.mul;
import static hat.types.vec2.sub;
import static hat.types.vec2.vec2;
import static hat.types.vec3.mix;
import static hat.types.vec3.vec3;
import static hat.types.vec4.vec4;


public class AnimShader  {
    @Reflect public static float rect(vec2 r, vec2 bottomLeft, vec2 topRight) {
        float ret;
        float d = 0.005f;
        ret = smoothstep(bottomLeft.x() - d, bottomLeft.x() + d, r.x());
        ret *= smoothstep(bottomLeft.y() - d, bottomLeft.y() + d, r.y());
        ret *= 1.0f - smoothstep(topRight.y() - d, topRight.y() + d, r.y());
        ret *= 1.0f - smoothstep(topRight.x() - d, topRight.x() + d, r.x());
        return ret;
    }

    @Reflect public static   float disk(vec2 r, vec2 center, float radius) {
        return 1.0f - smoothstep(radius - 0.005f, radius + 0.005f, length(sub(r, center)));
    }

    @Reflect public static vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        vec2 fres = vec2(uniforms.iResolution().x(),uniforms.iResolution().y());
        float ftime = uniforms.iTime();
        vec2 p = div(fragCoord, fres);
        vec2 r = mul(div(sub(fragCoord, mul(fres, .5f)), fres.y()), 2f);
        float xMax = fres.x() / fres.y();

        vec3 col1 = vec3(0.216f, 0.471f, 0.698f); // blue
        vec3 col2 = vec3(1.00f, 0.329f, 0.298f); // yellow
        vec3 col3 = vec3(0.867f, 0.910f, 0.247f); // red

        vec3 ret = vec3(0f, 0f, 0f);

        if (p.x() < 1f / 5f) { // Part I
            vec2 q = add(r, vec2(xMax * 4f / 5f, 0f));
            ret = vec3(0.2f);
            // y coordinate depends on time
            float y = uniforms.iTime();
            // mod constraints y to be between 0.0 and 2.0,
            // and y jumps from 2.0 to 0.0
            // substracting -1.0 makes why jump from 1.0 to -1.0
            y = mod(y, 2f) - 1f;
            ret = mix(ret, col1, disk(q, vec2(0f, y), 0f));
        } else if (p.x() < 2f / 5f) { // Part II
            vec2 q = add(r, vec2(xMax * 2f / 5f, 0f));
            ret = vec3(0.3f);
            // oscillation
            float amplitude = 0.8f;
            // y coordinate oscillates with a period of 0.5 seconds
            float y = 0.8f * sin(0.5f * uniforms.iTime() * PI * 2f);
            // radius oscillates too
            float radius = 0.15f + 0.05f * sin(uniforms.iTime() * 8.0f);
            ret = mix(ret, col1, disk(q, vec2(0f, y), radius));
        } else if (p.x() < 3. / 5.) { // Part III
            vec2 q = add(r, vec2(xMax * 0f / 5f, 0f));
            ret = vec3(0.4f);
            // booth coordinates oscillates
            float x = 0.2f * cos(uniforms.iTime() * 5.0f);
            // but they have a phase difference of PI/2
            float y = 0.3f * cos(uniforms.iTime() * 5.0f + PI / 2f);
            float radius = 0.2f + 0.1f * sin(ftime * 2.0f);
            // make the color mixture time dependent
            vec3 color = mix(col1, col2, sin(ftime) * 0.5f + 0.5f);
            ret = mix(ret, color, rect(q, vec2(x - 0.1f, y - 0.1f), vec2(x + 0.1f, y + 0.1f)));
            // try different phases, different amplitudes and different frequencies
            // for x and y coordinates
        } else if (p.x() < 4f / 5f) { // Part IV
            vec2 q = add(r, vec2(-xMax * 2f / 5f, 0f));
            ret = vec3(0.3f);
            for (float i = -1.0f; i < 1.0f; i += 0.2f) {
                float x = 0.2f * cos(ftime * 5.0f + i * PI);
                // y coordinate is the loop value
                float y = i;
                vec2 s = sub(q, vec2(x, y));
                // each box has a different phase
                float angle = ftime * 3f + i;
                mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
                s = mul(s, rot);
                ret = mix(ret, col1, rect(s, vec2(-0.06f, -0.06f), vec2(0.06f, 0.06f)));
            }
        } else if (p.x() < 1) { // Part V
            vec2 q = add(r, vec2(-xMax * 4f / 5f, 0f));
            ret = vec3(0.2f);
            // let stop and move again periodically
            float speed = 2.0f;
            float t = ftime * speed;
            float stopEveryAngle = PI / 2.0f;
            float stopRatio = 0.5f;
            float t1 = (floor(t) + smoothstep(0f, 1f - stopRatio, fract(t))) * stopEveryAngle;

            float x = -0.2f * cos(t1);
            float y = 0.3f * sin(t1);
            float dx = 0.1f + 0.03f * sin(t * 10.0f);
            float dy = 0.1f + 0.03f * sin(t * 10.0f + PI);
            ret = mix(ret, col1, rect(q, vec2(x - dx, y - dy), vec2(x + dx, y + dy)));
        }

        vec3 pixel = ret;
        fragColor = vec4(pixel, 1.0f);
        return fragColor;
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
        var shader = ShaderViewer.of(acc, AnimShader.class,1024, 1024);
        shader.startLoop((uniforms, f32Array) -> update( acc, uniforms, f32Array, shader.view.getWidth(), shader.view.getWidth()));
    }
}
