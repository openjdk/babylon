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
import hat.types.vec2;
import hat.types.vec3;
import hat.types.vec4;
import jdk.incubator.code.Reflect;
import optkl.ifacemapper.MappableIface;
import hat.buffer.Uniforms;
import shade.ShaderViewer;
import java.lang.invoke.MethodHandles;

import static hat.types.F32.abs;
import static hat.types.F32.cos;
import static hat.types.F32.max;
import static hat.types.F32.min;
import static hat.types.F32.sin;
import static hat.types.F32.smoothstep;
import static hat.types.vec2.add;
import static hat.types.vec2.div;
import static hat.types.vec2.dot;
import static hat.types.vec2.mul;
import static hat.types.vec2.round;
import static hat.types.vec2.sub;
import static hat.types.vec2.vec2;
import static hat.types.vec3.mix;
import static hat.types.vec3.vec3;
import static hat.types.vec4.normalize;
import static hat.types.vec4.vec4;

//https://www.shadertoy.com/view/4tXyWs
public class MobiusShader{


    /*
     vec2 ortho(vec2 v)
            {
                return vec2(v.y, -v.x);
            }
     */
    @Reflect
    public static  vec2 ortho(vec2 v) {
        return vec2(v.y(), -v.x());
    }

    /*
    void stroke(float dist, vec3 color, inout vec3 fragColor, float thickness, float aa)
            {
                float alpha = smoothstep(0.5 * (thickness + aa), 0.5 * (thickness - aa), abs(dist));
                fragColor = mix(fragColor, color, alpha);
            }
     */
    @Reflect public static  vec3 stroke(float dist, vec3 color, vec3 fragColor, float thickness, float aa) {
        float alpha = smoothstep(0.5f * (thickness + aa), 0.5f * (thickness - aa), abs(dist));
        return mix(fragColor, color, alpha);
    }
    /*

            void fill(float dist, vec3 color, inout vec3 fragColor, float aa)
            {
                float alpha = smoothstep(0.5*aa, -0.5*aa, dist);
                fragColor = mix(fragColor, color, alpha);
            }

     */

    @Reflect public static vec3 fill(float dist, vec3 color, vec3 fragColor, float aa) {
        float alpha = smoothstep(0.5f * aa, -0.5f * aa, dist);
        return mix(fragColor, color, alpha);
    }

    /*
    void renderGrid(vec2 pos, out vec3 fragColor)
            {
                vec3 background = vec3(1.0);
                vec3 axes = vec3(0.4);
                vec3 lines = vec3(0.7);
                vec3 sublines = vec3(0.95);
                float subdiv = 10.0;

                float thickness = 0.003;
                float aa = length(fwidth(pos));

                fragColor = background;

                vec2 toSubGrid = pos - round(pos*subdiv)/subdiv;
                stroke(min(abs(toSubGrid.x), abs(toSubGrid.y)), sublines, fragColor, thickness, aa);

                vec2 toGrid = pos - round(pos);
                stroke(min(abs(toGrid.x), abs(toGrid.y)), lines, fragColor, thickness, aa);

                stroke(min(abs(pos.x), abs(pos.y)), axes, fragColor, thickness, aa);
            }
     */
    @Reflect public static void renderGrid(vec2 pos, vec3 fragColor) {
        vec3 background = vec3(1.0f);
        vec3 axes = vec3(0.4f);
        vec3 lines = vec3(0.7f);
        vec3 sublines = vec3(0.95f);
        float subdiv = 10.0f;

        float thickness = 0.003f;
        float fwidthPos = 0.01f;
        float aa = fwidthPos;//?length(fwidthPos);

        fragColor = background;

        vec2 toSubGrid = sub(pos, div(vec2.round(mul(pos, subdiv)), subdiv));
        stroke(min(abs(toSubGrid.x()), abs(toSubGrid.y())), sublines, fragColor, thickness, aa);

        vec2 toGrid = sub(pos, round(pos));
        stroke(min(abs(toGrid.x()), abs(toGrid.y())), lines, fragColor, thickness, aa);

        stroke(min(abs(pos.x()), abs(pos.y())), axes, fragColor, thickness, aa);
    }

    /*
    float sdistLine(vec2 a, vec2 b, vec2 pos)
                {
                    return dot(pos - a, normalize(ortho(b - a)));
                }
    */
    @Reflect public static float sdistLine(vec2 a, vec2 b, vec2 pos) {
        return dot(sub(pos, a), vec2.normalize(ortho(sub(b, a))));
    }
    /*
            float sdistTri(vec2 a, vec2 b, vec2 c, vec2 pos)
            {
                return max( sdistLine(a, b, pos),
                        max(sdistLine(b, c, pos),
                            sdistLine(c, a, pos)));
            }
 */

    @Reflect public static  float sdistTri(vec2 a, vec2 b, vec2 c, vec2 pos) {
        return max(sdistLine(a, b, pos),
                max(sdistLine(b, c, pos),
                        sdistLine(c, a, pos)));
    }

    /*
    float sdistQuadConvex(vec2 a, vec2 b, vec2 c, vec2 d, vec2 pos)
            {
                return max(  sdistLine(a, b, pos),
                        max( sdistLine(b, c, pos),
                         max(sdistLine(c, d, pos),
                             sdistLine(d, a, pos))));
            }
     */
    @Reflect public static  float sdistQuadConvex(vec2 a, vec2 b, vec2 c, vec2 d, vec2 pos) {
        return max(sdistLine(a, b, pos),
                max(sdistLine(b, c, pos),
                        max(sdistLine(c, d, pos),
                                sdistLine(d, a, pos))));
    }

    /*
    void renderUnitSquare(vec2 pos, inout vec3 fragColor)
            {
            #if 0
                // Put a texture in there
                if (pos.x >= 0.0 && pos.y >= 0.0 && pos.x <= 1.0 && pos.y <= 1.0)
                {
                    fragColor.rgb = texture(iChannel0, pos).rgb;
                }
            #endif

                float dist = sdistQuadConvex(vec2(0, 0),
                                             vec2(1, 0),
                                             vec2(1, 1),
                                             vec2(0, 1), pos);
                stroke(dist, vec3(0, 0, 1), fragColor, 0.007, length(fwidth(pos)));
            }
     */
    @Reflect public static  vec3 renderUnitSquare(vec2 pos, vec3 fragColor) {

        float dist = sdistQuadConvex(vec2(0, 0),
                vec2(1, 0),
                vec2(1, 1),
                vec2(0, 1), pos);
        float fwidthPos = .0f;
        return stroke(dist, vec3(0, 0, 1), fragColor, 0.007f, fwidthPos/*length(fwidth(pos)*/);
    }

    /*
    void renderAxes(vec2 origin, vec2 pos, inout vec3 fragColor)
            {
                float len = 0.1;
                float thickness = 0.0075;
                float aa = length(fwidth(pos));

                float xshaft = sdistQuadConvex(origin + vec2(0.5*thickness),
                                               origin - vec2(0.5*thickness),
                                               origin + vec2(len, -0.5*thickness),
                                               origin + vec2(len, 0.5*thickness), pos);
                float xhead = sdistTri(origin + vec2(len, -2.0*thickness),
                                       origin + vec2(len + 6.0*thickness, 0),
                                       origin + vec2(len, 2.0*thickness), pos);

                fill(min(xshaft, xhead), vec3(1, 0, 0), fragColor, aa);

                float yshaft = sdistQuadConvex(origin - vec2(0.5*thickness),
                                               origin + vec2(0.5*thickness),
                                               origin + vec2(0.5*thickness, len),
                                               origin + vec2(-0.5*thickness, len), pos);
                float yhead = sdistTri(origin + vec2(2.0*thickness, len),
                                       origin + vec2(0, len + 6.0*thickness),
                                       origin + vec2(-2.0*thickness, len), pos);

                fill(min(yshaft, yhead), vec3(0, 0.75, 0), fragColor, aa);

            }
     */
    @Reflect public static   vec3 renderAxes(vec2 origin, vec2 pos, vec3 fragColor) {
        float len = 0.1f;
        float thickness = 0.0075f;
        float fwidthPos = 0.01f;
        float aa = fwidthPos;//length(fwidth(pos));

        float xshaft = sdistQuadConvex(add(origin, vec2(0.5f * thickness)),
                sub(origin, vec2(0.5f * thickness)),
                add(origin, vec2(len, -0.5f * thickness)),
                add(origin, vec2(len, 0.5f * thickness)), pos);

        float xhead = sdistTri(add(origin, vec2(len, -2.0f * thickness)),
                add(origin, vec2(len + 6.0f * thickness, 0f)),
                add(origin, vec2(len, 2.0f * thickness)), pos);

        fragColor = fill(min(xshaft, xhead), vec3(1f, 0f, 0f), fragColor, aa);

        float yshaft = sdistQuadConvex(add(origin, vec2(0.5f * thickness)),
                add(origin, vec2(0.5f * thickness)),
                add(origin, vec2(0.5f * thickness, len)),
                add(origin, vec2(-0.5f * thickness, len)), pos);

        float yhead = sdistTri(add(origin, vec2(2.0f * thickness, len)),
                add(origin, vec2(0, len + 6.0f * thickness)),
                add(origin, vec2(-2.0f * thickness, len)), pos);

        fragColor = fill(min(yshaft, yhead), vec3(0f, 0.75f, 0f), fragColor, aa);

        return fragColor;
    }

    /*
    vec2 cmul(vec2 a, vec2 b)
            {
                return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
            }
*/
    @Reflect public static  vec2 cmul(vec2 a, vec2 b) {
        return vec2(a.x() * b.x() - a.y() * b.y(), a.x() * b.y() + a.y() * b.x());
    }
    /*
            vec2 cdiv(vec2 a, vec2 b)
            {
                return cmul(a, vec2(b.x, -b.y)) / dot(b, b);
            }
     */

    @Reflect public static vec2 cdiv(vec2 a, vec2 b) {
        return div(cmul(a, vec2(b.x(), -b.y())), dot(b, b));
    }
    @Reflect public static vec4 createPixel(vec2 fres, float ftime, vec2 fmouse,vec2 fragCoord){
        vec4 fragColor = vec4(1f, 1f, 1f, 1f);
        float aspect =fres.x() / fres.y();
        vec2 pos = sub(mul(div(fragCoord,fres.y()), 1.5f), vec2((1.5f * aspect - 1.0f) / 2.0f, 0.25f));

        // apply a Möbius transformation to the plane
        vec2 a = vec2(1f, sin(0.4f * ftime));
        vec2 b = vec2(0f);
        vec2 c = vec2(0.5f * cos(0.6f * ftime), 0.5f * sin(0.5f * ftime));
        vec2 d = vec2(1f, cos(0.3f * ftime));
        pos = sub(pos, vec2(0.5f));
        pos = cdiv(add(cmul(a, pos), b), add(cmul(c, pos), d));
        pos = add(pos, vec2(0.5f));

        // render the grid and stuff
        fragColor = vec4(fragColor.x(), fragColor.y(), fragColor.z(), 1.0f);

        renderGrid(pos, vec3(fragColor.x(),fragColor.y(),fragColor.z()));
        fragColor = vec4(renderUnitSquare(pos, vec3(fragColor.x(),fragColor.y(),fragColor.z())), 1f);
        fragColor = vec4(renderAxes(vec2(0f), pos, vec3(fragColor.x(),fragColor.y(),fragColor.z())), 1f);
        return normalize(fragColor);
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

    public static void update(  Accelerator acc, Uniforms uniforms, F32Array f32Array, int width, int height) {
        acc.compute((@Reflect Compute) cc -> compute(cc, uniforms, f32Array, width, height));
    }

    static void main(String[] args) {
        var acc = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        var shader = ShaderViewer.of(acc, MobiusShader.class,1024, 1024, false);
        shader.startLoop((uniforms, f32Array) -> update( acc, uniforms, f32Array, shader.view.getWidth(), shader.view.getWidth()));
    }
}
