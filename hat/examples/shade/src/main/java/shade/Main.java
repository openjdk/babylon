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

import hat.Accelerator;
import hat.backend.Backend;
import shade.types.F32;
import shade.types.Shader;
import shade.types.Uniforms;
import shade.types.mat2;
import shade.types.vec2;
import shade.types.vec3;
import shade.types.vec4;

import javax.swing.JFrame;
import java.awt.Rectangle;
import java.io.IOException;
import java.lang.invoke.MethodHandles;

import static shade.types.F32.pow;
import static shade.types.F32.smoothstep;
import static shade.types.mat2.mat2;
import static shade.types.vec2.vec2;
import static shade.types.vec3.vec3;
import static shade.types.vec4.vec4;

public class Main extends JFrame {
    public final FloatImagePanel imagePanel;

    public Main(Accelerator accelerator, int width, int height, Shader shader) {
        super("HAT Toy");
        Controls controls = new Controls();
        setJMenuBar(controls.menu.menuBar());
        this.imagePanel = new FloatImagePanel(accelerator, controls, width, height, shader);
        setBounds(new Rectangle(width + 100, height + 200));
        setContentPane(imagePanel);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setVisible(true);
        this.imagePanel.start();
    }

    static Shader s1 = (uniforms, inFragColor, fragCoord) -> {
        int w = uniforms.iResolution().x();
        int wDiv3 = uniforms.iResolution().x() / 3;
        int h = uniforms.iResolution().y();
        int hDiv3 = uniforms.iResolution().y() / 3;
        boolean midx = (fragCoord.x() > wDiv3 && fragCoord.x() < (w - wDiv3));
        boolean midy = (fragCoord.y() > hDiv3 && fragCoord.y() < (h - hDiv3));
        if (uniforms.iMouse().x() > wDiv3) {
            if (midx && midy) {
                return vec4(fragCoord.x(), .0f, fragCoord.y(), 0.f);
            } else {
                return vec4(0f, 0f, .5f, 0f);
            }
        } else {
            return vec4(1f, 1f, .5f, 0f);
        }
    };

    static Shader s25 = (uniforms, fragColor, fragCoord) -> {
        //            ivec2->vec2
        var fres = vec2(uniforms.iResolution());
        //            vec2(fragCoord.xy / iResolution.xy);
        var p = fragCoord.div(fres);
        //            r = 2.0*vec2(fragCoord.xy - 0.5*iResolution.xy)/iResolution.y
        var r = fragCoord.sub(fres.mul(.5f)).div(fres.y()).mul(16f);

        float t = ((float) uniforms.iFrame()) / 15f;

        float v1 = F32.sin(r.x() + t);
        float v2 = F32.sin(r.y() + t);
        float v3 = F32.sin((r.x() + r.y()) + t);
        float v4 = F32.sin(r.length() + (1.7f * t));
        float v = v1 + v2 + v3 + v4;

        var ret = vec4(1f, 1f, 1f, 1f);

        if (p.x() < 1f / 10f) { // Part I
            ret = vec4(v1);
        } else if (p.x() < 2f / 10f) { // Part II
            // horizontal waves
            ret = vec4(v2);
        } else if (p.x() < 3f / 10f) { // Part III
            // diagonal waves
            ret = vec4(v3);
        } else if (p.x() < 4f / 10f) { // Part IV
            // circular waves
            ret = vec4(v4);
        } else if (p.x() < 5f / 10f) { // Part V
            // the sum of all waves
            ret = vec4(v);
        } else if (p.x() < 6f / 10f) { // Part VI
            // Add periodicity to the gradients
            ret = vec4(F32.sin(2f * v));
        } else { // Part VII
            // mix colors
            ret = vec4(F32.sin(v), F32.sin(v + 0.5f * F32.PI), F32.sin(v + F32.PI), 1f);
        }
        return ret.add(.5f).mul(.5f).clamp(0f, 1f);
    };


    static float square(vec2 r, vec2 bottomLeft, float side) {
        vec2 p = r.sub(bottomLeft);
        return (p.x() > 0.0f && p.x() < side && p.y() > 0f && p.y() < side) ? 1f : 0f;
    }

    static float character(vec2 r, vec2 bottomLeft, float charCode, float squareSide) {
        vec2 p = r.sub(bottomLeft);
        float ret = 0f;
        float num, quotient, remainder, divider;
        float x, y;
        num = charCode;
        for (int i = 0; i < 20; i++) {
            float boxNo = 19f - i;
            divider = pow(2f, boxNo);
            quotient = F32.floor(num / divider);
            remainder = num - quotient * divider;
            num = remainder;

            y = F32.floor(boxNo / 4f);
            x = boxNo - y * 4f;
            if (quotient == 1f) {
                ret += square(p, vec2(x, y).mul(squareSide), squareSide);
            }
        }
        return ret;
    }

    static mat2 rot(float th) {
        return mat2(F32.cos(th), -F32.sin(th), F32.sin(th), F32.cos(th));
    }


    static Shader intro = (Uniforms uniforms, vec4 fragColor, vec2 fragCoord) -> {
        float G = 990623f; // compressed characters :-)
        float L = 69919f;
        float S = 991119f;

        float t = uniforms.iFrame();
        var fres = vec2(uniforms.iResolution());

        var r = vec2(fragCoord.sub(0.5f).mul(fres)).div(fres.y());
        float c = 0.05f;
        vec2 x = r.add(F32.cos(0.3f * t), F32.sin(0.3f * t));
        var pL = x.mod(2.0f * c).sub(c).div(c);
        // var pL = vec2(F32.cos(0.3f*t),F32.sin(0.3f*t)), 2.0f*c).add(r).mod()-c/c;
        float circ = 1.0f - smoothstep(0.75f, 0.8f, pL.length());
        var rG = r.mul(rot(2f * F32.PI * smoothstep(0f, 1f, F32.mod(1.5f * t, 4.0f))));
        var rStripes = r.mul(rot(0.2f));

        float xMax = 0.5f * fres.x() / fres.y();
        float letterWidth = 2.0f * xMax * 0.9f / 4.0f;
        float side = letterWidth / 4f;
        float space = 2.0f * xMax * 0.1f / 5f;

        // to get rid off the y=0 horizontal blue line.
        float maskGS = character(r, vec2(-xMax + space, -2.5f * side).add(letterWidth + space, 0f).mul(0.0f), G, side);
        float maskG = character(rG, vec2(-xMax + space, -2.5f * side).add(letterWidth + space, 0f).mul(0.0f), G, side);
        float maskL1 = character(r, vec2(-xMax + space, -2.5f * side).add(letterWidth + space, 0f).mul(1.0f), L, side);
        float maskSS = character(r, vec2(-xMax + space, -2.5f * side).add(letterWidth + space, 0f).mul(2.0f), S, side);
        float maskS = character(r,
                vec2(-xMax + space, -2.5f * side)
                        .add(letterWidth + space, 0f)
                        .mul(2.0f).add(0.01f * F32.sin(2.1f * t), 0.012f * F32.cos(t)), S, side);
        float maskL2 = character(r, vec2(-xMax + space, -2.5f * side).add(letterWidth + space, 0f).mul(3.0f), L, side);
        float maskStripes = F32.step(0.25f, F32.mod(rStripes.x() - 0.5f * t, 0.5f));

        float i255 = 0.00392156862f;
        vec3 blue = vec3(43f, 172f, 181f).mul(i255);
        vec3 pink = vec3(232f, 77f, 91f).mul(i255);
        vec3 dark = vec3(59f, 59f, 59f).mul(i255);
        vec3 light = vec3(245f, 236f, 217f).mul(i255);
        vec3 green = vec3(180f, 204f, 18f).mul(i255);

        vec3 pixel = blue;
        pixel = vec3.mix(pixel, light, maskGS);
        pixel = vec3.mix(pixel, light, maskSS);
        pixel = pixel.sub(0.1f * maskStripes);
        pixel = vec3.mix(pixel, green, maskG);
        pixel = vec3.mix(pixel, pink, maskL1 * circ);
        pixel = vec3.mix(pixel, green, maskS);
        pixel = vec3.mix(pixel, pink, maskL2 * (1f - circ));
// need texture for this!
        // float dirt = pow(texture(iChannel0, 4.0f*r).x, 4.0f);
        // pixel = pixel.sub (0.2f*dirt - 0.1f).mul(maskG+maskS); // dirt
        pixel = pixel.sub(smoothstep(0.45f, 2.5f, r.length()));
        fragColor = vec4(pixel, 1.0f);
        return fragColor;
    };

    static float hash(float seed) {
        // Return a "random" number based on the "seed"
        return F32.fract(F32.sin(seed) * 43758.5453f);
    }

    static vec2 hashPosition(float x) {
        // Return a "random" position based on the "seed"
        return vec2(hash(x), hash(x * 1.1f));
    }

    static float disk(vec2 r, vec2 center, float radius) {
        return 1.0f - smoothstep(radius - 0.005f, radius + 0.005f, vec2(r).sub(center).length());
    }

    static float coordinateGrid(vec2 r) {
        vec3 axesCol = vec3(0.0f, 0.0f, 1.0f);
        vec3 gridCol = vec3(0.5f);
        float ret = 0.0f;

        // Draw grid lines
        float tickWidth = 0.1f;
        for (float i = -2.0f; i < 2.0f; i += tickWidth) {
            // "i" is the line coordinate.
            ret += 1f - smoothstep(0.0f, 0.005f, F32.abs(r.x() - i));
            ret += 1f - smoothstep(0.0f, 0.01f, F32.abs(r.y() - i));
        }
        // Draw the axes
        ret += 1f - smoothstep(0.001f, 0.005f, F32.abs(r.x()));
        ret += 1f - smoothstep(0.001f, 0.005f, F32.abs(r.y()));
        return ret;
    }

    static float plot(vec2 r, float y, float thickness) {
        return (F32.abs(y - r.y()) < thickness) ? 1.0f : 0.0f;
    }

    static Shader randy = (uniforms, fragColor, fragCoord) -> {
        vec2 fres = vec2(uniforms.iResolution());
        vec2 p = fragCoord.div(fres);
        vec2 r = vec2(fragCoord.sub(fres.mul(.5f))).div(fres.y()).mul(2f);
        ;
        float xMax = fres.x() / fres.y();

        vec3 bgCol = vec3(0.3f);
        vec3 col1 = vec3(0.216f, 0.471f, 0.698f); // blue
        vec3 col2 = vec3(1.00f, 0.329f, 0.298f); // yellow
        vec3 col3 = vec3(0.867f, 0.910f, 0.247f); // red

        vec3 ret = bgCol;

        vec3 white = vec3(1f);
        vec3 gray = vec3(.3f);
        if (r.y() > 0.7f) {

            // translated and rotated coordinate system
            vec2 q = r.sub(vec2(0f, 0.9f)).mul(vec2(1f, 20f));
            ret = vec3.mix(white, gray, coordinateGrid(q));

            // just the regular sin function
            float y = F32.sin(5f * q.x()) * 2.0f - 1.0f;

            ret = vec3.mix(ret, col1, plot(q, y, 0.1f));
        } else if (r.y() > 0.4f) {
            vec2 q = r.sub(vec2(0f, 0.6f)).mul(vec2(1f, 20f));
            ret = vec3.mix(white, col1, coordinateGrid(q));

            // take the decimal part of the sin function
            float y = F32.fract(F32.sin(5f * q.x())) * 2.0f - 1.0f;

            ret = vec3.mix(ret, col2, plot(q, y, 0.1f));
        } else if (r.y() > 0.1f) {
            // vec3 white = vec3(1f);
            vec2 q = r.sub(vec2(0f, 0.25f)).mul(vec2(1f, 20f));
            ret = vec3.mix(white, gray, coordinateGrid(q));

            // scale up the outcome of the sine function
            // increase the scale and see the transition from
            // periodic pattern to chaotic pattern
            float scale = 10.0f;
            float y = F32.fract(F32.sin(5f * q.x()) * scale) * 2.0f - 1.0f;

            ret = vec3.mix(ret, col1, plot(q, y, 0.2f));
        } else if (r.y() > -0.2f) {
            //vec3 white = vec3(1.);
            vec2 q = r.sub(vec2(0f, -0.0f)).mul(vec2(1f, 10f));
            ret = vec3.mix(white, col1, coordinateGrid(q));

            float seed = q.x();
            // Scale up with a big real number
            float y = F32.fract(F32.sin(seed) * 43758.5453f) * 2.0f - 1.0f;
            // this can be used as a pseudo-random value
            // These type of function, functions in which two inputs
            // that are close to each other (such as close q.x positions)
            // return highly different output values, are called "hash"
            // function.

            ret = vec3.mix(ret, col2, plot(q, y, 0.1f));
        } else {
            vec2 q = r.sub(vec2(0f, -0.6f));

            // use the loop index as the seed
            // and vary different quantities of disks, such as
            // location and radius
            for (float i = 0.0f; i < 6.0f; i++) {
                // change the seed and get different distributions
                float seed = i + 0.0f;
                vec2 pos = vec2(hash(seed), hash(seed + 0.5f)).sub(-0.5f).mul(3f);
                float radius = hash(seed + 3.5f);
                pos = pos.mul(vec2(1.0f, 0.3f));
                ret = vec3.mix(ret, col1, disk(q, pos, 0.2f * radius));
            }
        }

        vec3 pixel = ret;
        fragColor = vec4(pixel, 1.0f);
        return fragColor;
    };

    static float disk2(vec2 r, vec2 center, float radius) {
        return 1.0f - smoothstep(radius - 0.005f, radius + 0.005f, vec2(r).sub(center).length());
    }

    static float rect(vec2 r, vec2 bottomLeft, vec2 topRight) {
        float ret;
        float d = 0.005f;
        ret = smoothstep(bottomLeft.x() - d, bottomLeft.x() + d, r.x());
        ret *= smoothstep(bottomLeft.y() - d, bottomLeft.y() + d, r.y());
        ret *= 1.0f - smoothstep(topRight.y() - d, topRight.y() + d, r.y());
        ret *= 1.0f - smoothstep(topRight.x() - d, topRight.x() + d, r.x());
        return ret;
    }

    static Shader anim = (uniforms, fragColor, fragCoord) -> {
        vec2 fres = vec2(uniforms.iResolution());
        float ftime = uniforms.iTime();
        vec2 p = fragCoord.div(fres);
        vec2 r = vec2(fragCoord.sub(fres.mul(.5f))).div(fres.y()).mul(2f);
        float xMax = fres.x() / fres.y();

        vec3 col1 = vec3(0.216f, 0.471f, 0.698f); // blue
        vec3 col2 = vec3(1.00f, 0.329f, 0.298f); // yellow
        vec3 col3 = vec3(0.867f, 0.910f, 0.247f); // red

        vec3 ret = vec3(0f, 0f, 0f);

        if (p.x() < 1f / 5f) { // Part I
            vec2 q = r.add(vec2(xMax * 4f / 5f, 0f));
            ret = vec3(0.2f);
            // y coordinate depends on time
            float y = uniforms.iTime();
            // mod constraints y to be between 0.0 and 2.0,
            // and y jumps from 2.0 to 0.0
            // substracting -1.0 makes why jump from 1.0 to -1.0
            y = F32.mod(y, 2f) - 1f;
            ret = vec3.mix(ret, col1, disk(q, vec2(0f, y), 0f));
        } else if (p.x() < 2f / 5f) { // Part II
            vec2 q = r.add(vec2(xMax * 2f / 5f, 0f));
            ret = vec3(0.3f);
            // oscillation
            float amplitude = 0.8f;
            // y coordinate oscillates with a period of 0.5 seconds
            float y = 0.8f * F32.sin(0.5f * uniforms.iTime() * F32.PI * 2f);
            // radius oscillates too
            float radius = 0.15f + 0.05f * F32.sin(uniforms.iTime() * 8.0f);
            ret = vec3.mix(ret, col1, disk(q, vec2(0f, y), radius));
        } else if (p.x() < 3. / 5.) { // Part III
            vec2 q = r.add(vec2(xMax * 0f / 5f, 0f));
            ret = vec3(0.4f);
            // booth coordinates oscillates
            float x = 0.2f * F32.cos(uniforms.iTime() * 5.0f);
            // but they have a phase difference of PI/2
            float y = 0.3f * F32.cos(uniforms.iTime() * 5.0f + F32.PI / 2f);
            float radius = 0.2f + 0.1f * F32.sin(ftime * 2.0f);
            // make the color mixture time dependent
            vec3 color = vec3.mix(col1, col2, F32.sin(ftime) * 0.5f + 0.5f);
            ret = vec3.mix(ret, color, rect(q, vec2(x - 0.1f, y - 0.1f), vec2(x + 0.1f, y + 0.1f)));
            // try different phases, different amplitudes and different frequencies
            // for x and y coordinates
        } else if (p.x() < 4f / 5f) { // Part IV
            vec2 q = r.add(vec2(-xMax * 2f / 5f, 0f));
            ret = vec3(0.3f);
            for (float i = -1.0f; i < 1.0f; i += 0.2f) {
                float x = 0.2f * F32.cos(ftime * 5.0f + i * F32.PI);
                // y coordinate is the loop value
                float y = i;
                vec2 s = q.sub(vec2(x, y));
                // each box has a different phase
                float angle = ftime * 3f + i;
                mat2 rot = mat2(F32.cos(angle), -F32.sin(angle), F32.sin(angle), F32.cos(angle));
                s = s.mul(rot);
                ret = vec3.mix(ret, col1, rect(s, vec2(-0.06f, -0.06f), vec2(0.06f, 0.06f)));
            }
        } else if (p.x() < 1) { // Part V
            vec2 q = r.add(vec2(-xMax * 4f / 5f, 0f));
            ret = vec3(0.2f);
            // let stop and move again periodically
            float speed = 2.0f;
            float t = ftime * speed;
            float stopEveryAngle = F32.PI / 2.0f;
            float stopRatio = 0.5f;
            float t1 = (F32.floor(t) + smoothstep(0f, 1f - stopRatio, F32.fract(t))) * stopEveryAngle;

            float x = -0.2f * F32.cos(t1);
            float y = 0.3f * F32.sin(t1);
            float dx = 0.1f + 0.03f * F32.sin(t * 10.0f);
            float dy = 0.1f + 0.03f * F32.sin(t * 10.0f + F32.PI);
            ret = vec3.mix(ret, col1, rect(q, vec2(x - dx, y - dy), vec2(x + dx, y + dy)));
        }

        vec3 pixel = ret;
        fragColor = vec4(pixel, 1.0f);
        return fragColor;
    };


    static void main(String[] args) throws IOException {
        var acc = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        new Main(acc, 1024, 1024, s25);
    }
}