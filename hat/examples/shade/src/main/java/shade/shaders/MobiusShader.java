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
import static hat.types.vec4.xyz;

//https://www.shadertoy.com/view/4tXyWs
public class MobiusShader implements Shader {
    String glsSource = """
            vec2 ortho(vec2 v)
            {
                return vec2(v.y, -v.x);
            }

            void stroke(float dist, vec3 color, inout vec3 fragColor, float thickness, float aa)
            {
                float alpha = smoothstep(0.5 * (thickness + aa), 0.5 * (thickness - aa), abs(dist));
                fragColor = mix(fragColor, color, alpha);
            }

            void fill(float dist, vec3 color, inout vec3 fragColor, float aa)
            {
                float alpha = smoothstep(0.5*aa, -0.5*aa, dist);
                fragColor = mix(fragColor, color, alpha);
            }

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

            float sdistLine(vec2 a, vec2 b, vec2 pos)
            {
                return dot(pos - a, normalize(ortho(b - a)));
            }

            float sdistTri(vec2 a, vec2 b, vec2 c, vec2 pos)
            {
                return max( sdistLine(a, b, pos),
                        max(sdistLine(b, c, pos),
                            sdistLine(c, a, pos)));
            }

            float sdistQuadConvex(vec2 a, vec2 b, vec2 c, vec2 d, vec2 pos)
            {
                return max(  sdistLine(a, b, pos),
                        max( sdistLine(b, c, pos),
                         max(sdistLine(c, d, pos),
                             sdistLine(d, a, pos))));
            }

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

            vec2 cmul(vec2 a, vec2 b)
            {
                return vec2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
            }

            vec2 cdiv(vec2 a, vec2 b)
            {
                return cmul(a, vec2(b.x, -b.y)) / dot(b, b);
            }

            void mainImage(out vec4 fragColor, in vec2 fragCoord)
            {
                float aspect = iResolution.x / iResolution.y;
                      vec2 pos = (fragCoord / iResolution.y) * 1.5 - vec2((1.5*aspect - 1.0)/2.0, 0.25);

                // apply a Möbius transformation to the plane
                vec2 a = vec2(1, sin(0.4*iTime));
                vec2 b = vec2(0);
                vec2 c = vec2(0.5*cos(0.6*iTime), 0.5*sin(0.5*iTime));
                vec2 d = vec2(1, cos(0.3*iTime));
                pos -= vec2(0.5);
                pos = cdiv(cmul(a, pos) + b, cmul(c, pos) + d);
                pos += vec2(0.5);

                // render the grid and stuff
                fragColor.a = 1.0;
                      renderGrid(pos, fragColor.rgb);
                renderUnitSquare(pos, fragColor.rgb);
                renderAxes(vec2(0), pos, fragColor.rgb);
            }

            """;

    /*
     vec2 ortho(vec2 v)
            {
                return vec2(v.y, -v.x);
            }
     */
    vec2 ortho(vec2 v) {
        return vec2(v.y(), -v.x());
    }

    /*
    void stroke(float dist, vec3 color, inout vec3 fragColor, float thickness, float aa)
            {
                float alpha = smoothstep(0.5 * (thickness + aa), 0.5 * (thickness - aa), abs(dist));
                fragColor = mix(fragColor, color, alpha);
            }
     */
    vec3 stroke(float dist, vec3 color, vec3 fragColor, float thickness, float aa) {
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

    vec3 fill(float dist, vec3 color, vec3 fragColor, float aa) {
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
    void renderGrid(vec2 pos, vec3 fragColor) {
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
    float sdistLine(vec2 a, vec2 b, vec2 pos) {
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

    float sdistTri(vec2 a, vec2 b, vec2 c, vec2 pos) {
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
    float sdistQuadConvex(vec2 a, vec2 b, vec2 c, vec2 d, vec2 pos) {
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
    vec3 renderUnitSquare(vec2 pos, vec3 fragColor) {

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
    vec3 renderAxes(vec2 origin, vec2 pos, vec3 fragColor) {
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
    vec2 cmul(vec2 a, vec2 b) {
        return vec2(a.x() * b.x() - a.y() * b.y(), a.x() * b.y() + a.y() * b.x());
    }
    /*
            vec2 cdiv(vec2 a, vec2 b)
            {
                return cmul(a, vec2(b.x, -b.y)) / dot(b, b);
            }
     */

    vec2 cdiv(vec2 a, vec2 b) {
        return div(cmul(a, vec2(b.x(), -b.y())), dot(b, b));
    }

    @Override
    public vec4 mainImage(Uniforms uniforms, vec4 fragColor, vec2 fragCoord) {
        fragColor = vec4(1f, 1f, 1f, 1f);
        float aspect = uniforms.iResolution().x() / uniforms.iResolution().y();
        vec2 pos = sub(mul(div(fragCoord, uniforms.iResolution().y()), 1.5f), vec2((1.5f * aspect - 1.0f) / 2.0f, 0.25f));

        // apply a Möbius transformation to the plane
        vec2 a = vec2(1f, sin(0.4f * uniforms.iTime()));
        vec2 b = vec2(0f);
        vec2 c = vec2(0.5f * cos(0.6f * uniforms.iTime()), 0.5f * sin(0.5f * uniforms.iTime()));
        vec2 d = vec2(1f, cos(0.3f * uniforms.iTime()));
        pos = sub(pos, vec2(0.5f));
        pos = cdiv(add(cmul(a, pos), b), add(cmul(c, pos), d));
        pos = add(pos, vec2(0.5f));

        // render the grid and stuff
        fragColor = vec4(fragColor.x(), fragColor.y(), fragColor.z(), 1.0f);
        renderGrid(pos, xyz(fragColor));
        fragColor = vec4(renderUnitSquare(pos, xyz(fragColor)), 1f);
        fragColor = vec4(renderAxes(vec2(0f), pos, xyz(fragColor)), 1f);
        return normalize(fragColor);
    }

    static Config controls = Config.of(
            Boolean.getBoolean("hat") ? new Accelerator(MethodHandles.lookup(), Backend.FIRST) : null,
            Integer.parseInt(System.getProperty("width", System.getProperty("size", "512"))),
            Integer.parseInt(System.getProperty("height", System.getProperty("size", "512"))),
            Integer.parseInt(System.getProperty("targetFps", "10")),
            new MobiusShader()
    );

    static void main(String[] args) throws IOException {
        new ShaderApp(controls);
    }
}
