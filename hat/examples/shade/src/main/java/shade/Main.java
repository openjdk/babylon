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
import shade.types.mat3;
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
import static shade.types.mat3.mat3;
import static shade.types.vec2.length;
import static shade.types.vec2.vec2;
import static shade.types.vec3.clamp;
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


    //*
    // afl_ext 2017-2024
// MIT License

// Use your mouse to move the camera around! Press the Left Mouse Button on the image to look around!

    static float DRAG_MULT = 0.38f; // changes how much waves pull on the water
    static float WATER_DEPTH = 1f; // how deep is the water
    static float CAMERA_HEIGHT = 1.5f; // how high the camera should be
    static int ITERATIONS_RAYMARCH = 12; // waves iterations of raymarching
    static int ITERATIONS_NORMAL = 36; // waves iterations when calculating normals

    static vec2 NormalizedMouse(vec2 fMouse, vec2 fResolution) {
        return fMouse.div(fResolution);
    } // normalize mouse coords

    // Calculates wave value and its derivative,
// for the wave direction, position in space, wave frequency and time
    static vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
        float x = direction.dot(position) * frequency + timeshift;
        float wave = F32.exp(F32.sin(x) - 1.0f);
        float dx = wave * F32.cos(x);
        return vec2(wave, -dx);
    }

    // Calculates waves by summing octaves of various waves with various parameters
    static float getwaves(vec2 position, int iterations, float fTime) {
        float wavePhaseShift = position.length() * 0.1f; // this is to avoid every octave having exactly the same phase everywhere
        float iter = 0.0f; // this will help generating well distributed wave directions
        float frequency = 1.0f; // frequency of the wave, this will change every iteration
        float timeMultiplier = 2.0f; // time multiplier for the wave, this will change every iteration
        float weight = 1.0f;// weight in final sum for the wave, this will change every iteration
        float sumOfValues = 0.0f; // will store final sum of values
        float sumOfWeights = 0.0f; // will store final sum of weights
        for (int i = 0; i < iterations; i++) {
            // generate some wave direction that looks kind of random
            vec2 p = vec2(F32.sin(iter), F32.cos(iter));

            // calculate wave data
            vec2 res = wavedx(position, p, frequency, fTime * timeMultiplier + wavePhaseShift);

            // shift position around according to wave drag and derivative of the wave
            position = position.add(position.mul(res.y() * weight * DRAG_MULT));

            // add the results to sums
            sumOfValues += res.x() * weight;
            sumOfWeights += weight;

            // modify next octave ;
            weight = F32.mix(weight, 0.0f, 0.2f);
            frequency *= 1.18f;
            timeMultiplier *= 1.07f;

            // add some kind of random value to make next wave look random too
            iter += 1232.399963f;
        }
        // calculate and return
        return sumOfValues / sumOfWeights;
    }

    // Raymarches the ray from top water layer boundary to low water layer boundary
    static float raymarchwater(vec3 camera, vec3 start, vec3 end, float depth, float fTime) {
        vec3 pos = start;
        vec3 dir = end.sub(start).normalize();
        for (int i = 0; i < 64; i++) {
            // the height is from 0 to -depth
            float height = getwaves(vec2(pos.x(), pos.y()), ITERATIONS_RAYMARCH, fTime) * depth - depth;
            // if the waves height almost nearly matches the ray height, assume its a hit and return the hit distance
            if (height + 0.01f > pos.y()) {
                return pos.distance(camera);
            }
            // iterate forwards according to the height mismatch
            pos = pos.add(dir.mul(pos.y() - height));
        }
        // if hit was not registered, just assume hit the top layer,
        // this makes the raymarching faster and looks better at higher distances
        return start.distance(camera);
    }

    // Calculate normal at point by calculating the height at the pos and 2 additional points very close to pos
    static vec3 normal(vec2 pos, float e, float depth, float fTime) {
        vec2 ex = vec2(e, 0);
        float H = getwaves(pos, ITERATIONS_NORMAL, fTime) * depth;
        vec3 a = vec3(pos.x(), H, pos.y());
        return vec3.normalize(
                vec3.cross(
                        a.sub(vec3(pos.x() - e, getwaves(pos.sub(ex), ITERATIONS_NORMAL, fTime) * depth, pos.y())),
                        a.sub(vec3(pos.x(), getwaves(pos.add(ex), ITERATIONS_NORMAL, fTime) * depth, pos.y() + e))
                )
        );
    }

    // Helper function generating a rotation matrix around the axis by the angle
    static mat3 createRotationMatrixAxisAngle(vec3 axis, float angle) {
        float s = F32.sin(angle);
        float c = F32.cos(angle);
        float oc = 1.0f - c;
        return mat3(
                oc * axis.x() * axis.x() + c, oc * axis.x() * axis.y() - axis.z() * s, oc * axis.z() * axis.x() + axis.y() * s,
                oc * axis.x() * axis.y() + axis.z() * s, oc * axis.y() * axis.y() + c, oc * axis.y() * axis.z() - axis.x() * s,
                oc * axis.z() * axis.x() - axis.y() * s, oc * axis.y() * axis.z() + axis.x() * s, oc * axis.z() * axis.z() + c
        );
    }

    // Helper function that generates camera ray based on UV and mouse
    static vec3 getRay(vec2 fragCoord, vec2 fres, vec2 fMouse) {
        vec2 uv = fragCoord.div(fres.mul(2f).sub(1f)).mul(vec2(fres.x() / fres.y(), 1.0f));
        // for fisheye, uncomment following line and comment the next one
        //vec3 proj = normalize(vec3(uv.x, uv.y, 1.0) + vec3(uv.x, uv.y, -1.0) * pow(length(uv), 2.0) * 0.05);
        vec3 proj = vec3(uv.x(), uv.y(), 1.5f).normalize();
        if (fres.x() < 600.0) {
            return proj;
        }

        var m1 =createRotationMatrixAxisAngle(vec3(0.0f, -1.0f, 0.0f), 3.0f * ((NormalizedMouse(fMouse, fres).x() + 0.5f) * 2.0f - 1.0f));
        var m2 = createRotationMatrixAxisAngle(vec3(1.0f, 0.0f, 0.0f), 0.5f + 1.5f * (((NormalizedMouse(fMouse, fres).y()) == 0.0f ? 0.27f : NormalizedMouse(fMouse, fres).y() * 1.0f) * 2.0f - 1.0f));
        return proj.mul(m1).mul(m2);
    }

    // Ray-Plane intersection checker
    static float intersectPlane(vec3 origin, vec3 direction, vec3 point, vec3 normal) {
        return F32.clamp(vec3.dot(point.sub(origin), normal) / vec3.dot(direction, normal), -1.0f, 9991999.0f);
    }

    // Some very barebones but fast atmosphere approximation
    static vec3 extra_cheap_atmosphere(vec3 raydir, vec3 sundir, float fTime) {
        //sundir.y = max(sundir.y, -0.07);
        float special_trick = 1.0f / (raydir.y() * 1.0f + 0.1f);
        float special_trick2 = 1.0f / (sundir.y() * 11.0f + 1.0f);
        float raysundt = F32.pow(F32.abs(vec3.dot(sundir, raydir)), 2.0f);
        float sundt = F32.pow(F32.max(0.0f, vec3.dot(sundir, raydir)), 8.0f);
        float mymie = sundt * special_trick * 0.2f;
        vec3 suncolor=vec3(0f,0f,.7f);
       // vec3 suncolor = vec3.mix(
         //       vec3(1.0f), vec3.max(vec3(0.0f), vec3(1.0f).sub(vec3(5.5f, 13.0f, 22.4f).div(22.4f)
           //     ), special_trick2));
        vec3 bluesky = vec3(5.5f, 13.0f, 22.4f).div(22.4f).mul(suncolor);
        vec3 bluesky2 = vec3.max(vec3(0.0f), bluesky.sub(vec3(5.5f, 13.0f, 22.4f).mul(0.002f * (special_trick + -6.0f * sundir.y() * sundir.y()))));
        bluesky2 = bluesky2.mul(special_trick * (0.24f + raysundt * 0.24f));
        return bluesky2.mul(1.0f + F32.pow(1.0f - raydir.y(), 3.0f));
    }

    // Calculate where the sun should be, it will be moving around the sky
    static vec3 getSunDirection(float fTime) {
        return vec3(-0.0773502691896258f, 0.5f + F32.sin(fTime * 0.2f + 2.6f) * 0.45f, 0.5773502691896258f).normalize();
    }

    // Get atmosphere color for given direction
    static vec3 getAtmosphere(vec3 dir, float fTime) {
        return extra_cheap_atmosphere(dir, getSunDirection(fTime), fTime).mul(0.5f);
    }

    // Get sun color for given direction
    static float getSun(vec3 dir, float fTime) {
        return F32.pow(F32.max(0.0f, vec3.dot(dir, getSunDirection(fTime))), 720.0f) * 210.0f;
    }
//https://www.shadertoy.com/view/MdXyzX
    static Shader water = (uniforms, fragColor, fragCoord) -> {
        var fResolution = vec2(uniforms.iResolution());
        var fMouse = vec2(uniforms.iMouse());
        float fTime = uniforms.iTime();
        // get the ray
        vec3 ray = getRay(fragCoord, fResolution, fMouse);

        if (ray.y() >= 0.0) {
            // if ray.y is positive, render the sky
            vec3 C = getAtmosphere(ray, fTime).add(getSun(ray, fTime));
            fragColor = vec4(aces_tonemap(C.mul(2.0f)), 1.0f);
            return fragColor;
        } else {

            // now ray.y must be negative, water must be hit
            // define water planes
            vec3 waterPlaneHigh = vec3(0.0f, 0.0f, 0.0f);
            vec3 waterPlaneLow = vec3(0.0f, -WATER_DEPTH, 0.0f);

            // define ray origin, moving around
            vec3 origin = vec3(fTime * 0.2f, CAMERA_HEIGHT, 1f);

            // calculate intersections and reconstruct positions
            float highPlaneHit = intersectPlane(origin, ray, waterPlaneHigh, vec3(0.0f, 1.0f, 0.0f));
            float lowPlaneHit = intersectPlane(origin, ray, waterPlaneLow, vec3(0.0f, 1.0f, 0.0f));
            vec3 highHitPos = origin.add(ray.mul(highPlaneHit));
            vec3 lowHitPos = origin.add(ray.mul(lowPlaneHit));

            // raymatch water and reconstruct the hit pos
            float dist = raymarchwater(origin, highHitPos, lowHitPos, WATER_DEPTH, fTime);
            vec3 waterHitPos = origin.add(ray.mul(dist));

            // calculate normal at the hit position
            vec3 N = normal(vec2(waterHitPos.x(), waterHitPos.y()), 0.01f, WATER_DEPTH, fTime);

            // smooth the normal with distance to avoid disturbing high frequency noise
            N = vec3.mix(N, vec3(0.0f, 1.0f, 0.0f), 0.8f * F32.min(1.0f, F32.sqrt(dist * 0.01f) * 1.1f));

            // calculate fresnel coefficient
            float fresnel = (0.04f + (1.0f - 0.04f) * (pow(1.0f - F32.max(0.0f, vec3.dot(vec3(0f).sub(N), ray)), 5.0f)));

            // reflect the ray and make sure it bounces up
            vec3 R = vec3.normalize(vec3.reflect(ray, N));
            R = vec3(R.x(), F32.abs(R.y()), R.z());

            // calculate the reflection and approximate subsurface scattering
            vec3 reflection = getAtmosphere(R, fTime).add(getSun(R, fTime));
            vec3 scattering = vec3(0.0293f, 0.0698f, 0.1717f).mul(0.1f).mul(0.2f + (waterHitPos.y() + WATER_DEPTH) / WATER_DEPTH);

            // return the combined result
            vec3 C = reflection.mul(fresnel).add(scattering);
            fragColor = vec4(clamp(aces_tonemap(C.mul(2.0f)),0f,1f), 1.0f);
            return fragColor;
        }
    };

    static vec3 aces_tonemap(vec3 color){
        mat3 m1 = mat3(
                0.59719f, 0.07600f, 0.02840f,
                0.35458f, 0.90834f, 0.13383f,
                0.04823f, 0.01566f, 0.83777f
        );
        mat3 m2 = mat3(
                1.60475f, -0.10208f, -0.00327f,
                -0.53108f,  1.10813f, -0.07276f,
                -0.07367f, -0.00605f,  1.07602f
        );
        vec3 v = color.mul(m1);
        vec3 a = v.mul(v.add( + 0.0245786f)).sub(0.000090537f);
        vec3 b = v.mul((v.mul(0.983729f).add(0.4329510f)).add(0.238081f));
        var aOverBMulM2 = a.div(b).mul(m2);
        return vec3.clamp(vec3.pow(aOverBMulM2, 1.0f / 2.2f),0f,1f);
    }
     static Shader aces= (uniform,  fragColor,fragCoord )-> {
        // https://www.shadertoy.com/view/XsGfWV
        vec2 position = fragCoord.div(vec2(uniform.iResolution())).mul(2f).sub(1f); //fragCoord/iResolution.xy)* 2.0 - 1.0;
        position = vec2(position.x()+ uniform.iTime() * 0.2f, position.y()); // position.x += iTime * 0.2;

         //vec3 color = pow(
         //     sin(
         //        position.x * 4.0 + vec3(0.0, 1.0, 2.0) * 3.1415 * 2.0 / 3.0
         //     ) * 0.5 + 0.5,
         //     vec3(2.0)
         //     ) * (exp(
         //              abs(position.y) * 4.0
         //              ) - 1.0);
         vec3 v012 = vec3(0f, 1f, 2f);
         vec3 v0123x2Pi = v012.mul(F32.PI).mul(2f);
         vec3 v0123x2PiDiv3 = v0123x2Pi.div(3f);
         vec3  sinCoef = v0123x2PiDiv3.add(position.x()*4);
         vec3 color = vec3.pow(
                          vec3.sin(sinCoef).mul(0.5f).add(0.5f),
                         2f
                 )
                 .mul(
                         F32.exp(
                                 F32.abs(position.y()) * 4f
                         ) - 1f
                 );
         if(position.y() < 0f){
             color = aces_tonemap(color);
         }
        fragColor = vec4(clamp(color,0f,1f),1.0f);
        return fragColor;
    };

/*
// variant of https://shadertoy.com/view/3llcDl
// inspired by https://www.facebook.com/eric.wenger.547/videos/2727028317526304/

// variant of https://shadertoy.com/view/3llcDl
// inspired by https://www.facebook.com/eric.wenger.547/videos/2727028317526304/

void mainImage(out vec4 fragColor,  vec2 fragCoord ){

    vec2 fResolution = iResolution.xy;

    vec2  U = ((2.*fragCoord - fResolution)) / fResolution.y; // normalized coordinates
    vec2  z = U - vec2(-1,0);
    U.x = U.x-.5;                      // Moebius transform

    U = U * mat2(z,-z.y,z.x) / dot(U,U);

    U = U+.5;
                  // offset   spiral, zoom   phase            // spiraling
    U =   log(length(U))*vec2(.5, -.5) + iTime/8. + atan(U.y, U.x)/6.2832 * vec2(6, 1);
    // n
    U = U * 3./vec2(2,1);
    z = //vec2(1);
    fwidth(U);
    U = fract(U)*5.;

    vec2 I = floor(U);
    U = fract(U);              // subdiv big square in 5x5
    I.x = mod( I.x - 2.*I.y , 5.);                            // rearrange
    U = vec2(U.x+ float(I.x==1.||I.x==3.),U.y+float(I.x<2.));     // recombine big tiles

    float id = -1.;

    if (I.x != 4.){
        U =U/2.;                                     // but small times
        id = mod(floor(I.x/2.)+I.y,5.);
    }
    U = abs(fract(U)*2.-1.); float v = max(U.x,U.y);          // dist to border
    fragColor =   smoothstep(.7,-.7, (v-.95)/( abs(z.x-z.y)>1.?.1:z.y*8.))  // draw AA tiles
        * (id<0.?vec4(1): .6 + .6 * cos( id  + vec4(0,23,21,0)  ) );// color
}
 */
    static Shader spiral = (uniforms, fragColor, fragCoord)->{
        // variant of https://shadertoy.com/view/3llcDl
// inspired by https://www.facebook.com/eric.wenger.547/videos/2727028317526304/
            float fTime = uniforms.iTime();
            float fFrame = uniforms.iFrame();
            var fResolution  =vec2(uniforms.iResolution());

            var U = fragCoord.mul(2f).sub(fResolution).div(fResolution.y());
            // normalized coordinates
            var z = U.sub(-1f,0f);

            U = U.sub(.5f,0f);
            U = U.mul(mat2(z.x(), z.y(), -z.y(),z.x())).div(vec2.dot(U,U));
            // offset   spiral, zoom   phase            // spiraling
            U = U.add(.5f,0f);

            U = vec2(//U =   log(length(U))*vec2(.5, -.5) + iTime/8. + atan(U.y, U.x)/6.2832 * vec2(6, 1);
                    F32.log(U.length())).mul(.5f, -.5f)
                    .add(fTime/8)
                    .add(vec2.atan(U.y(),U.x()).div(6.2832f).mul(6f,1f));

            U = U.mul(vec2(3f).div(vec2(2f,1f)));
            z = vec2(.1f);//fwidth(U); // this resamples the image.  Not sure how we do this!
            U = U.fract().mul(5f);
            vec2 I = U.floor();
            U = U.fract();             // subdiv big square in 5x5
            I = vec2(F32.mod( I.x() - 2.f*I.y() , 5f),I.y());                            // rearrange
            U = U.add((I.x()==1f||I.x()==3f)?1f:0f, I.x()<2.0?1f:0f);     // recombine big tiles
            float id = -1f;
            if (I.x()!=4f) {
                U = U.div(2f);                                     // but small times
                id = F32.mod(F32.floor(I.x() / 2f) + I.y(), 5f);
            }
            U = vec2.abs(U.fract().mul(2f).sub(1f));
            float v = F32.max(U.x(),U.y());          // dist to border

            return
                    vec4.smoothstep(
                            vec4(.7f),
                            vec4(-.7f),
                            vec4(v-.95f).div(F32.abs(z.x()-z.y())>1f
                                    ?.1f
                                    :z.y()*8f
                            )
                            .mul(id<0f
                               ? vec4(1f)
                               : vec4(.6f).add(.6f).mul(
                                       vec4.cos( vec4(id).add(vec4(0f,23f,21f,0f)))
                                    )
                            )
            );// color

    };

    static void main(String[] args) throws IOException {
        var acc = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        enum SHADER{
            Blue((uniform, fragColor, fragCoord)->{
                return vec4(0f,0f,1f,0f);
            }),
            Gradient((uniforms, fragColor, fragCoord) -> {
                var fResolution = vec2(uniforms.iResolution());
                float fFrame = uniforms.iFrame();
                var uv = fragCoord.div(fResolution);
                return vec4(uv.x(),uv.y(),F32.max(fFrame/100f,1f),0f);
            }),

            S1(s1),S25(s25),Randy(randy),Spiral(spiral),Anim(anim),Water(water),Intro(intro);

            Shader shader;
            SHADER(Shader shader){
                this.shader=shader;
            }
        }
        new Main(acc, 1024, 1024, SHADER.Spiral.shader);
    }
}