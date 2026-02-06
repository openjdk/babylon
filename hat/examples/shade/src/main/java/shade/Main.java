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
import jdk.incubator.code.dialect.java.JavaOp;
import shade.types.F32;
import shade.types.Shader;
import shade.types.Uniforms;

import javax.swing.JFrame;
import java.awt.Rectangle;
import java.io.IOException;
import java.lang.invoke.MethodHandles;

import  shade.types.vec2;
import shade.types.vec4;
import shade.types.mat2;

import shade.types.vec3;

import static shade.types.F32.pow;
import static shade.types.vec2.vec2;

import static shade.types.vec3.vec3;
import static shade.types.vec4.vec4;

import static shade.types.mat2.mat2;
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

    static Shader s1 =  (uniforms, inFragColor, fragCoord) -> {
            int w = uniforms.iResolution().x();
            int wDiv3 = uniforms.iResolution().x()/3;
            int h = uniforms.iResolution().y();
            int hDiv3 = uniforms.iResolution().y()/3;
            boolean midx = (fragCoord.x() > wDiv3 && fragCoord.x() <(w-wDiv3) );
            boolean midy = (fragCoord.y() > hDiv3 && fragCoord.y() <(h-hDiv3));
            if (uniforms.iMouse().x()>wDiv3) {
                if (midx && midy) {
                    return vec4(fragCoord.x(), .0f, fragCoord.y(), 0.f);
                } else {
                    return vec4(0f, 0f, .5f, 0f);
                }
            }else {
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

        float t = ((float)uniforms.iFrame())/15f;

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
        return ret.add(.5f).mul(.5f).clamp(0f,1f);
    };


    static float square(vec2 r, vec2 bottomLeft, float side) {
        vec2 p = r.sub(bottomLeft);
        return ( p.x() > 0.0f && p.x() < side && p.y()>0f && p.y() < side ) ? 1f : 0f;
    }

    static float character(vec2 r, vec2 bottomLeft, float charCode, float squareSide) {
        vec2 p = r.sub(bottomLeft);
        float ret = 0f;
        float num, quotient, remainder, divider;
        float x, y;
        num = charCode;
        for(int i=0; i<20; i++) {
            float boxNo = 19f-i;
            divider = pow(2f, boxNo);
            quotient = F32.floor(num / divider);
            remainder = num - quotient*divider;
            num = remainder;

            y = F32.floor(boxNo/4f);
            x = boxNo - y*4f;
            if(quotient == 1f) {
                ret += square( p, vec2(x, y).mul(squareSide), squareSide );
            }
        }
        return ret;
    }

    static mat2 rot(float th) { return mat2(F32.cos(th), -F32.sin(th), F32.sin(th), F32.cos(th)); }


    static Shader intro=(Uniforms uniforms, vec4 fragColor, vec2 fragCoord )-> {
        float G = 990623f; // compressed characters :-)
        float L = 69919f;
        float S = 991119f;

        float t = uniforms.iFrame();
        var fres = vec2(uniforms.iResolution());

        var  r = vec2(fragCoord.sub( 0.5f).mul(fres)).div(fres.y());
        float c = 0.05f;
        vec2 x = r.add(F32.cos(0.3f*t),F32.sin(0.3f*t));
        var pL = x.mod(2.0f*c).sub(c).div(c);
       // var pL = vec2(F32.cos(0.3f*t),F32.sin(0.3f*t)), 2.0f*c).add(r).mod()-c/c;
        float circ = 1.0f-F32.smoothstep(0.75f, 0.8f, pL.length());
        var rG = r.mul(rot(2f*F32.PI*F32.smoothstep(0f,1f,F32.mod(1.5f*t,4.0f))));
        var rStripes = r.mul(rot(0.2f));

        float xMax = 0.5f*fres.x()/fres.y();
        float letterWidth = 2.0f*xMax*0.9f/4.0f;
        float side = letterWidth/4f;
        float space = 2.0f*xMax*0.1f/5f;

         // to get rid off the y=0 horizontal blue line.
        float maskGS = character(r, vec2(-xMax+space, -2.5f*side).add(letterWidth+space, 0f).mul(0.0f), G, side);
        float maskG = character(rG, vec2(-xMax+space, -2.5f*side).add(letterWidth+space, 0f).mul(0.0f), G, side);
        float maskL1 = character(r, vec2(-xMax+space, -2.5f*side).add(letterWidth+space, 0f).mul(1.0f), L, side);
        float maskSS = character(r, vec2(-xMax+space, -2.5f*side).add(letterWidth+space, 0f).mul(2.0f), S, side);
        float maskS = character(r,
                vec2(-xMax+space, -2.5f*side)
                        .add(letterWidth+space, 0f)
                        .mul(2.0f).add(0.01f*F32.sin(2.1f*t),0.012f*F32.cos(t)), S, side);
        float maskL2 = character(r, vec2(-xMax+space, -2.5f*side).add(letterWidth+space, 0f).mul(3.0f), L, side);
        float maskStripes = F32.step(0.25f, F32.mod(rStripes.x() - 0.5f*t, 0.5f));

        float i255 = 0.00392156862f;
        vec3 blue = vec3(43f, 172f, 181f).mul(i255);
        vec3 pink = vec3(232f, 77f, 91f).mul(i255);
        vec3 dark = vec3(59f, 59f, 59f).mul(i255);
        vec3 light = vec3(245f, 236f, 217f).mul(i255);
        vec3 green = vec3(180f, 204f, 18f).mul(i255);

        vec3 pixel = blue;
        pixel = vec3.mix(pixel, light, maskGS);
        pixel = vec3.mix(pixel, light, maskSS);
        pixel = pixel.sub(0.1f*maskStripes);
        pixel = vec3.mix(pixel, green, maskG);
        pixel = vec3.mix(pixel, pink, maskL1*circ);
        pixel = vec3.mix(pixel, green, maskS);
        pixel = vec3.mix(pixel, pink, maskL2*(1f-circ));
// need texture for this!
       // float dirt = pow(texture(iChannel0, 4.0f*r).x, 4.0f);
       // pixel = pixel.sub (0.2f*dirt - 0.1f).mul(maskG+maskS); // dirt
        pixel = pixel.sub(F32.smoothstep(0.45f, 2.5f, r.length()));
        fragColor = vec4(pixel, 1.0f);
        return fragColor;
    };

    static void main(String[] args) throws IOException {
        var acc = new Accelerator(MethodHandles.lookup(), Backend.FIRST);
        new Main(acc, 1024, 1024, s25);
    }
}